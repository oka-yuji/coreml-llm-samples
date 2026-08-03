import CoreMLBackend
import Foundation
import LLMCore
import Observation

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

@MainActor
@Observable
final class LiveCameraViewModel {

    static let question = "In one short sentence, what is visible in this image?"
    static let maxNewTokens = 24

    var caption: String = ""
    var statusLine: String = ""
    var breakdown: String = ""
    var cycleCount: Int = 0
    var lastCycleSeconds: Double = 0
    var isRunning: Bool = false
    var isPausedByThermal: Bool = false
    var thermalState: String = LiveCameraViewModel.thermalName()
    var errorText: String = ""

    @ObservationIgnored var onCycle: (@MainActor (LiveCycleReport) -> Void)?
    @ObservationIgnored private(set) var loop: Task<Void, Never>?
    @ObservationIgnored private var stopRequested = false
    @ObservationIgnored private var thermalWatcher: Task<Void, Never>?

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

    static var stagingURL: URL {
        FileManager.default.temporaryDirectory.appending(path: "live-camera-frame.jpg")
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
        statusLine = "Starting…"
        let staging = Self.stagingURL
        let question = Self.question
        let maxNew = Self.maxNewTokens

        loop = Task { [weak self] in
            guard let self else { return }
            var index = 0
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
                    let label = try await Task.detached(priority: .userInitiated) {
                        try source.stageFrame(to: staging)
                    }.value
                    let captureSeconds = (clock.now - captureStart) / .seconds(1)

                    let info = try await engine.generateWithImage(
                        imageURL: staging, question: question, maxNew: maxNew,
                        speculative: speculative)
                    guard !Task.isCancelled else { break }

                    let cycleSeconds = (clock.now - cycleStart) / .seconds(1)
                    index += 1
                    let text = info.text.trimmingCharacters(in: .whitespacesAndNewlines)
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
                    self.errorText = String(describing: error)
                    self.statusLine = "Stopped after an error: \(error)"
                    break
                }
            }
            self.isRunning = false
            self.isPausedByThermal = false
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
        statusLine = "Stopping after the current frame…"
    }

    func cancel() {
        stopRequested = true
        loop?.cancel()
        loop = nil
        isRunning = false
        stopThermalWatch()
    }
}
