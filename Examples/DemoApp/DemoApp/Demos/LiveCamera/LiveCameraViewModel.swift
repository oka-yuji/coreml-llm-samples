import CoreMLBackend
import Foundation
import LLMCore
import Observation

enum LiveCycleError: Error, CustomStringConvertible {
    case emptyResponse

    var description: String {
        switch self {
        case .emptyResponse: return "the model returned no text for this frame"
        }
    }
}

enum LiveEngineProvision {

    enum Outcome {
        case ready(ChatViewModel.EngineHandle)
        case unavailable(String)
    }

    static let noBundleMessage =
        "No vision-capable bundle. Load Gemma 4 E2B (pal6) and add vision assets."

    static func hasVisionSidecar(_ bundle: URL) -> Bool {
        let fileManager = FileManager.default
        return ["vision_fp16.mlmodelc", "vision_fp16.mlpackage"].contains {
            fileManager.fileExists(atPath: bundle.appending(path: $0).path(percentEncoded: false))
        }
    }

    static func visionCapableBundles() -> [URL] {
        let fileManager = FileManager.default
        var seen = Set<String>()
        var found: [URL] = []

        func consider(_ url: URL) {
            let path = url.path(percentEncoded: false)
            guard !seen.contains(path) else { return }
            seen.insert(path)
            let manifest = url.appending(path: "manifest.json")
            guard fileManager.fileExists(atPath: manifest.path(percentEncoded: false)) else { return }
            guard hasVisionSidecar(url) else { return }
            found.append(url)
        }

        for model in LLMModels.supported() {
            if let url = ModelStorage.locateBundle(folderName: model.bundleFolderName) {
                consider(url)
            }
        }
        for root in [ModelStorage.modelsRoot(), ModelStorage.documentsDirectory()] {
            let children = (try? fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil)) ?? []
            for child in children { consider(child) }
        }
        return found
    }

    static func displayName(_ url: URL) -> String {
        (try? ModelBundle(contentsOf: url).manifest.name) ?? url.lastPathComponent
    }

    @MainActor
    static func ensureVisionEngine(
        chat: ChatViewModel, status: @MainActor (String) -> Void
    ) async -> Outcome {
        if chat.supportsImageAttachment, let handle = chat.engineHandle { return .ready(handle) }
        let candidates = visionCapableBundles()
        guard !candidates.isEmpty else { return .unavailable(noBundleMessage) }
        var lastFailure: String?
        for url in candidates {
            status("Loading \(displayName(url))…")
            await chat.loadModel(path: url.path(percentEncoded: false))
            if chat.supportsImageAttachment, let handle = chat.engineHandle { return .ready(handle) }
            if case .failed(let reason) = chat.phase { lastFailure = reason }
        }
        if let lastFailure {
            return .unavailable("Loading a vision-capable bundle failed: \(lastFailure)")
        }
        return .unavailable(noBundleMessage)
    }
}

struct LiveCycleReport: Sendable {
    var index: Int
    var frameLabel: String
    var caption: String
    var captureSeconds: Double
    var encodeSeconds: Double
    var feedSeconds: Double
    var generateSeconds: Double
    var cycleSeconds: Double
    var promptTokens: Int
    var imageRows: Int
    var generatedTokens: Int
    var tokensPerSecond: Double
    var thermal: String
    var footprintBytes: Int?

    var breakdownLine: String {
        String(
            format: "capture %.2fs | encode %.2fs | feed %.2fs | gen %.2fs | cycle %.2fs",
            captureSeconds, encodeSeconds, feedSeconds, generateSeconds, cycleSeconds)
    }

    var detailLine: String {
        var parts = [breakdownLine]
        parts.append("prompt \(promptTokens)")
        if imageRows > 0 { parts.append("image \(imageRows) rows") }
        parts.append("\(generatedTokens) tok")
        parts.append(String(format: "%.1f tok/s", tokensPerSecond))
        parts.append("thermal \(thermal)")
        if let footprintBytes {
            parts.append(String(format: "mem %.0fMB", Double(footprintBytes) / 1_048_576))
        }
        return parts.joined(separator: "  |  ")
    }
}

enum LiveCyclePhase: String {
    case idle
    case encode
    case feed
    case generate

    init(_ phase: VLMPhase) {
        switch phase {
        case .encode: self = .encode
        case .feed: self = .feed
        case .generate: self = .generate
        }
    }

    var label: String {
        switch self {
        case .idle: return "idle"
        case .encode: return "encode"
        case .feed: return "feed"
        case .generate: return "generate"
        }
    }

    var glyph: String {
        switch self {
        case .idle: return "pause.circle"
        case .encode: return "camera.metering.matrix"
        case .feed: return "arrow.right.to.line"
        case .generate: return "text.bubble"
        }
    }
}

@MainActor
@Observable
final class LiveCameraViewModel {

    static let question = "In one short sentence, what is visible in this image?"
    static let maxNewTokens = 24
    static let maxConsecutiveFailures = 3

    var caption: String = ""
    var statusLine: String = ""
    var breakdown: String = ""
    var cycleCount: Int = 0
    var lastCycleSeconds: Double = 0
    var isRunning: Bool = false
    var isPausedByThermal: Bool = false
    var thermalState: String = LiveCameraViewModel.thermalName()
    var errorText: String = ""
    var cyclePhase: LiveCyclePhase = .idle

    @ObservationIgnored var onCycle: (@MainActor (LiveCycleReport) -> Void)?
    @ObservationIgnored private(set) var loop: Task<Void, Never>?
    @ObservationIgnored private var stopRequested = false
    @ObservationIgnored private var thermalWatcher: Task<Void, Never>?
    @ObservationIgnored private var runID = 0

    var canStart: Bool { !isRunning }

    func startThermalWatch() {
        guard thermalWatcher == nil else { return }
        thermalWatcher = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.thermalState = Self.thermalName()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    func stopThermalWatch() {
        thermalWatcher?.cancel()
        thermalWatcher = nil
    }

    static func thermalName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static func isTooHot() -> Bool {
        switch ProcessInfo.processInfo.thermalState {
        case .serious, .critical: return true
        default: return false
        }
    }

    func start(
        engine: CoreMLEngine,
        source: any LiveFrameSource,
        speculative: Bool,
        cycleLimit: Int? = nil,
        modelID: String? = nil,
        hfRevision: String? = nil,
        computeUnits: String? = nil,
        bundleFolder: String? = nil
    ) {
        guard !isRunning else { return }
        stopRequested = false
        isRunning = true
        errorText = ""
        cycleCount = 0
        cyclePhase = .idle
        runID += 1
        let myRunID = runID
        statusLine = "Starting…"
        let question = Self.question
        let maxNew = Self.maxNewTokens

        loop = Task { [weak self] in
            guard let self else { return }
            var index = 0
            var failures = 0
            let clock = ContinuousClock()
            while !Task.isCancelled, !self.stopRequested {
                if let cycleLimit, index >= cycleLimit { break }

                self.thermalState = Self.thermalName()
                if Self.isTooHot() {
                    if !self.isPausedByThermal {
                        self.isPausedByThermal = true
                        self.statusLine =
                            "Paused: thermal state is \(self.thermalState). Waiting for the device to cool down."
                    }
                    try? await Task.sleep(for: .seconds(2))
                    continue
                }
                if self.isPausedByThermal {
                    self.isPausedByThermal = false
                    self.statusLine = "Resumed at thermal \(self.thermalState)."
                }

                let cycleStart = clock.now
                do {
                    let captureStart = clock.now
                    let frame = try await Task.detached(priority: .userInitiated) {
                        try source.nextFrame()
                    }.value
                    let captureSeconds = (clock.now - captureStart) / .seconds(1)
                    let label = frame.label

                    self.cyclePhase = .encode
                    let info = try await engine.generateWithImage(
                        frame: frame.image, question: question, maxNew: maxNew,
                        speculative: speculative,
                        onPhase: { phase in
                            Task { @MainActor [weak self] in
                                guard let self, self.runID == myRunID, self.isRunning else { return }
                                self.cyclePhase = LiveCyclePhase(phase)
                            }
                        })
                    guard !Task.isCancelled else { break }
                    self.cyclePhase = .idle

                    let cycleSeconds = (clock.now - cycleStart) / .seconds(1)
                    let text = info.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !text.isEmpty else { throw LiveCycleError.emptyResponse }
                    index += 1
                    failures = 0
                    self.errorText = ""
                    let report = LiveCycleReport(
                        index: index,
                        frameLabel: label,
                        caption: text,
                        captureSeconds: captureSeconds,
                        encodeSeconds: info.visionEncodeSeconds,
                        feedSeconds: info.prefillSeconds,
                        generateSeconds: info.decodeSeconds,
                        cycleSeconds: cycleSeconds,
                        promptTokens: info.promptTokens,
                        imageRows: info.imageRows,
                        generatedTokens: info.generatedTokens,
                        tokensPerSecond: info.decodeTokensPerSecond,
                        thermal: Self.thermalName(),
                        footprintBytes: info.peakMemoryBytes)

                    self.caption = text
                    self.cycleCount = index
                    self.lastCycleSeconds = cycleSeconds
                    self.breakdown = report.breakdownLine
                    self.statusLine = report.detailLine
                    self.thermalState = report.thermal
                    MetricsLog.liveCycle(
                        report: report, modelID: modelID, hfRevision: hfRevision,
                        computeUnits: computeUnits, bundleFolder: bundleFolder)
                    self.onCycle?(report)
                } catch is CancellationError {
                    break
                } catch {
                    failures += 1
                    let reason = String(describing: error)
                    self.errorText = reason
                    MetricsLog.error(
                        phase: "live-cycle", reason: reason, modelID: modelID,
                        hfRevision: hfRevision, computeUnits: computeUnits,
                        bundleFolder: bundleFolder, promptTokens: nil, contextLength: nil)
                    if failures >= Self.maxConsecutiveFailures {
                        self.statusLine =
                            "Stopped after \(failures) failed cycles in a row: \(reason)"
                        break
                    }
                    self.statusLine =
                        "Cycle failed (\(failures)/\(Self.maxConsecutiveFailures)): \(reason)"
                    try? await Task.sleep(for: .seconds(1))
                }
            }
            guard self.runID == myRunID else { return }
            self.isRunning = false
            self.isPausedByThermal = false
            self.cyclePhase = .idle
            if self.errorText.isEmpty {
                self.statusLine = self.cycleCount > 0
                    ? "Stopped after \(self.cycleCount) cycles. \(self.breakdown)"
                    : "Stopped."
            }
        }
    }

    func stop() {
        guard isRunning else { return }
        stopRequested = true
        loop?.cancel()
        isRunning = false
        isPausedByThermal = false
        cyclePhase = .idle
        statusLine = cycleCount > 0
            ? "Stopped after \(cycleCount) cycles. \(breakdown)"
            : "Stopped."
    }

    func cancel() {
        stopRequested = true
        loop?.cancel()
        loop = nil
        isRunning = false
        isPausedByThermal = false
        cyclePhase = .idle
        stopThermalWatch()
    }
}
