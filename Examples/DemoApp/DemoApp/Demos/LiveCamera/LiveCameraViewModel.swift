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

enum LiveCaptionLanguage: String, CaseIterable, Identifiable, Sendable {
    case english = "en"
    case japanese = "ja"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .english: return "EN"
        case .japanese: return "日本語"
        }
    }

    var question: String {
        switch self {
        case .english:
            return "In one short sentence, what is visible in this image?"
        case .japanese:
            return "この画像に写っているものを、短い一文で説明してください。"
        }
    }

    var maxNewTokens: Int {
        switch self {
        case .english: return 32
        case .japanese: return 40
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

    static let bundleDefaultsKey = "liveCameraVisionBundle"

    static func hasVisionSidecar(_ bundle: URL) -> Bool {
        let fileManager = FileManager.default
        return ["vision_fp16.mlmodelc", "vision_fp16.mlpackage"].contains {
            fileManager.fileExists(atPath: bundle.appending(path: $0).path(percentEncoded: false))
        }
    }

    static func runsOnThisPlatform(_ url: URL) -> Bool {
        guard let model = LLMModels.all.first(where: {
            $0.bundleFolderName == url.lastPathComponent
        }) else { return true }
        return model.supportedPlatforms.contains(LLMModels.currentPlatform)
    }

    static func identityKey(_ url: URL) -> String {
        var path = url.resolvingSymlinksInPath().standardizedFileURL.path(percentEncoded: false)
        while path.count > 1, path.hasSuffix("/") { path.removeLast() }
        return path
    }

    static func visionCapableBundles() -> [URL] {
        let fileManager = FileManager.default
        var seen = Set<String>()
        var found: [URL] = []

        func consider(_ url: URL) {
            let path = identityKey(url)
            guard !seen.contains(path) else { return }
            seen.insert(path)
            let manifest = url.appending(path: "manifest.json")
            guard fileManager.fileExists(atPath: manifest.path(percentEncoded: false)) else { return }
            guard hasVisionSidecar(url) else { return }
            guard runsOnThisPlatform(url) else { return }
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

    static func shortName(_ url: URL) -> String { url.lastPathComponent }

    static func rememberBundle(_ url: URL) {
        UserDefaults.standard.set(url.lastPathComponent, forKey: bundleDefaultsKey)
    }

    static func initialSelection(among candidates: [URL], loadedPath: String) -> URL? {
        let stored = UserDefaults.standard.string(forKey: bundleDefaultsKey)
        if let stored, let match = candidates.first(where: { $0.lastPathComponent == stored }) {
            return match
        }
        let loadedKey = identityKey(URL(fileURLWithPath: loadedPath, isDirectory: true))
        if let loaded = candidates.first(where: { identityKey($0) == loadedKey }) {
            return loaded
        }
        return candidates.first
    }

    @MainActor
    static func ensureVisionEngine(
        chat: ChatViewModel, preferred: URL? = nil, status: @MainActor (String) -> Void
    ) async -> Outcome {
        if let preferred {
            let loaded = identityKey(URL(fileURLWithPath: chat.loadedPath, isDirectory: true))
            if loaded == identityKey(preferred),
               chat.supportsImageAttachment, let handle = chat.engineHandle { return .ready(handle) }
            status("Loading \(displayName(preferred))…")
            await chat.loadModel(path: preferred.path(percentEncoded: false))
            if chat.supportsImageAttachment, let handle = chat.engineHandle { return .ready(handle) }
            if case .failed(let reason) = chat.phase {
                return .unavailable("Loading \(shortName(preferred)) failed: \(reason)")
            }
            return .unavailable("\(shortName(preferred)) has no usable vision assets.")
        }
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
    var language: LiveCaptionLanguage
    var caption: String
    var captureSeconds: Double
    var encodeSeconds: Double
    var encodeWaitSeconds: Double
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
            format: "capture %.2fs | encode %.2fs | wait %.2fs | feed %.2fs | gen %.2fs | cycle %.2fs",
            captureSeconds, encodeSeconds, encodeWaitSeconds, feedSeconds, generateSeconds,
            cycleSeconds)
    }

    var detailLine: String {
        var parts = [breakdownLine]
        parts.append("lang \(language.rawValue)")
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

struct LiveReadyFrame: Sendable {
    let label: String
    let encoded: LiveEncodedFrame
    let captureSeconds: Double
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

    static let maxConsecutiveFailures = 3
    static let languageDefaultsKey = "liveCameraCaptionLanguage"

    var language: LiveCaptionLanguage
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
    @ObservationIgnored var onPartialCaption: (@MainActor (String, Int) -> Void)?
    @ObservationIgnored private(set) var loop: Task<Void, Never>?
    @ObservationIgnored private var stopRequested = false
    @ObservationIgnored private var thermalWatcher: Task<Void, Never>?
    @ObservationIgnored private var runID = 0
    @ObservationIgnored private var cycleSeq = 0
    @ObservationIgnored private var streamedTokens = 0
    @ObservationIgnored private var prefetched: Task<LiveReadyFrame, Error>?
    @ObservationIgnored private(set) var discardedPrefetches = 0

    init() {
        let stored = UserDefaults.standard.string(forKey: Self.languageDefaultsKey)
        language = stored.flatMap(LiveCaptionLanguage.init(rawValue:)) ?? .english
    }

    func selectLanguage(_ next: LiveCaptionLanguage) {
        guard next != language else { return }
        language = next
        UserDefaults.standard.set(next.rawValue, forKey: Self.languageDefaultsKey)
    }

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
        streaming: Bool = true,
        prefetch: Bool = false,
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

        loop = Task { [weak self] in
            guard let self else { return }
            var index = 0
            var failures = 0
            let clock = ContinuousClock()
            var encodeHandle: LiveVisionEncodeHandle?
            defer { self.discardPrefetch() }
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
                let language = self.language
                let question = language.question
                let maxNew = language.maxNewTokens
                self.cycleSeq += 1
                let mySeq = self.cycleSeq
                self.streamedTokens = 0
                var onPartial: (@Sendable (String, Int) -> Void)?
                if streaming {
                    onPartial = { [weak self] text, tokens in
                        Task { @MainActor in
                            guard let self, self.runID == myRunID, self.isRunning else { return }
                            guard self.cycleSeq == mySeq, tokens > self.streamedTokens else { return }
                            self.streamedTokens = tokens
                            self.caption = text
                            self.onPartialCaption?(text, tokens)
                        }
                    }
                }
                do {
                    let handle: LiveVisionEncodeHandle
                    if let encodeHandle {
                        handle = encodeHandle
                    } else {
                        handle = try await engine.liveVisionEncodeHandle()
                        encodeHandle = handle
                    }

                    self.cyclePhase = .encode
                    let waitStart = clock.now
                    let ready: LiveReadyFrame
                    if let pending = self.prefetched {
                        self.prefetched = nil
                        ready = try await pending.value
                    } else {
                        ready = try await Self.captureAndEncode(
                            source: source, handle: handle, clock: clock).value
                    }
                    let encodeWaitSeconds = (clock.now - waitStart) / .seconds(1)
                    guard !Task.isCancelled, !self.stopRequested else { break }

                    let isLastCycle = cycleLimit.map { index + 1 >= $0 } ?? false
                    let info = try await engine.generateWithEncodedFrame(
                        ready.encoded, question: question, maxNew: maxNew,
                        speculative: speculative,
                        onPhase: { phase in
                            Task { @MainActor [weak self] in
                                guard let self, self.runID == myRunID, self.isRunning else { return }
                                self.cyclePhase = LiveCyclePhase(phase)
                                guard prefetch, phase == .generate, self.cycleSeq == mySeq,
                                      !isLastCycle, self.prefetched == nil else { return }
                                self.prefetched = Self.captureAndEncode(
                                    source: source, handle: handle, clock: clock)
                            }
                        },
                        onPartial: onPartial)
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
                        frameLabel: ready.label,
                        language: language,
                        caption: text,
                        captureSeconds: ready.captureSeconds,
                        encodeSeconds: info.visionEncodeSeconds,
                        encodeWaitSeconds: encodeWaitSeconds,
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

    private func discardPrefetch() {
        guard let prefetched else { return }
        prefetched.cancel()
        self.prefetched = nil
        discardedPrefetches += 1
    }

    nonisolated private static func captureAndEncode(
        source: any LiveFrameSource, handle: LiveVisionEncodeHandle, clock: ContinuousClock
    ) -> Task<LiveReadyFrame, Error> {
        Task.detached(priority: .userInitiated) {
            try Task.checkCancellation()
            let t0 = clock.now
            let frame = try source.nextFrame()
            let captureSeconds = (clock.now - t0) / .seconds(1)
            try Task.checkCancellation()
            let encoded = try await handle.encode(frame.image)
            return LiveReadyFrame(
                label: frame.label, encoded: encoded, captureSeconds: captureSeconds)
        }
    }

    func stop() {
        guard isRunning else { return }
        stopRequested = true
        loop?.cancel()
        discardPrefetch()
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
        discardPrefetch()
        loop = nil
        isRunning = false
        isPausedByThermal = false
        cyclePhase = .idle
        stopThermalWatch()
    }
}
