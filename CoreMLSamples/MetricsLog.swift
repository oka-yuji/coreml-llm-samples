import Foundation
import LLMCore

#if canImport(UIKit)
    import UIKit
#endif

struct MetricsRecord: Codable {
    var kind: String
    var timestamp: String
    var deviceModel: String
    var osVersion: String
    var appBuild: String

    var modelID: String?
    var hfRevision: String?
    var computeUnits: String?
    var bundleFolder: String?

    var promptTokens: Int?
    var reusedTokens: Int?
    var generatedTokens: Int?
    var finishReason: String?

    var errorPhase: String?
    var errorReason: String?
    var contextLength: Int?

    var modelLoadSeconds: Double?
    var ttftSeconds: Double?
    var prefillSeconds: Double?
    var captureSeconds: Double?
    var visionEncodeSeconds: Double?
    var visionEncodeWaitSeconds: Double?
    var cycleSeconds: Double?
    var cycleIndex: Int?
    var captionLanguage: String?
    var captionMode: String?
    var imageRows: Int?
    var decodeTokPerSec: Double?
    var perTokenMillis: [Double]?
    var perTokenMinMs: Double?
    var perTokenMedianMs: Double?
    var perTokenMeanMs: Double?
    var perTokenMaxMs: Double?
    var feedWidths: [Int]?

    var specEnabled: Bool?
    var specRounds: Int?
    var specDrafted: Int?
    var specAccepted: Int?
    var specAcceptanceRate: Double?
    var specFallbackRounds: Int?

    var kvOp: String?
    var kvBytes: Int?
    var kvDurationSeconds: Double?

    var footprintAfterPrefillMB: Double?
    var footprintAtEndMB: Double?
    var peakFootprintMB: Double?
    var availableMemoryMB: Double?
    var sysWiredMB: Double?

    var thermalStateStart: String?
    var thermalStateEnd: String?
    var batteryLevel: Double?
    var batteryState: String?

    var models: [String]?

    var citationsTotal: Int?
    var citationsSeen: Int?
    var citationMeanOverlap: Double?

    var agentSession: String?
    var agentTurn: Int?
    var agentStep: Int?
    var agentEffort: String?
    var agentToolCalls: [String]?
    var agentRerenderDiverged: Bool?
    var agentThinkingChars: Int?
    var agentAnswerChars: Int?

    var toolName: String?
    var toolArg: String?
    var toolSeconds: Double?
    var toolResultChars: Int?
    var toolError: Bool?
}

struct AgentMetricsContext {
    var session: String?
    var turn: Int?
    var step: Int?
    var effort: String?
    var toolCalls: [String]?
    var rerenderDiverged: Bool?
    var thinkingChars: Int?
    var answerChars: Int?
}

extension MetricsRecord {
    mutating func apply(_ agent: AgentMetricsContext?) {
        guard let agent else { return }
        agentSession = agent.session
        agentTurn = agent.turn
        agentStep = agent.step
        agentEffort = agent.effort
        agentToolCalls = agent.toolCalls
        agentRerenderDiverged = agent.rerenderDiverged
        agentThinkingChars = agent.thinkingChars
        agentAnswerChars = agent.answerChars
    }
}

enum MetricsLog {
    static var fileURL: URL {
        documentsDirectory().appending(path: "metrics.jsonl")
    }

    @MainActor
    static func session(models: [String]) {
        var record = base(kind: "session")
        record.models = models
        append(record)
    }

    @MainActor
    static func message(
        metrics: GenerationMetrics, modelID: String?, hfRevision: String?,
        computeUnits: String?, bundleFolder: String?, modelLoadSeconds: Double?,
        citationsTotal: Int? = nil, citationsSeen: Int? = nil, citationMeanOverlap: Double? = nil,
        agent: AgentMetricsContext? = nil
    ) {
        var record = base(kind: "message")
        record.apply(agent)
        record.citationsTotal = citationsTotal
        record.citationsSeen = citationsSeen
        record.citationMeanOverlap = citationMeanOverlap
        record.modelID = modelID
        record.hfRevision = hfRevision
        record.computeUnits = computeUnits
        record.bundleFolder = bundleFolder
        record.promptTokens = metrics.promptTokens
        record.reusedTokens = metrics.reusedTokens
        record.generatedTokens = metrics.generatedTokens
        record.finishReason = metrics.finishReason?.rawValue
        record.modelLoadSeconds = modelLoadSeconds
        record.ttftSeconds = metrics.timeToFirstToken / .seconds(1)
        record.prefillSeconds = metrics.prefillSeconds
        record.decodeTokPerSec = metrics.decodeTokensPerSecond
        if let deltas = metrics.perTokenMillis, !deltas.isEmpty {
            record.perTokenMillis = deltas
            let sorted = deltas.sorted()
            record.perTokenMinMs = sorted.first
            record.perTokenMaxMs = sorted.last
            record.perTokenMedianMs = sorted[sorted.count / 2]
            record.perTokenMeanMs = deltas.reduce(0, +) / Double(deltas.count)
        }
        record.feedWidths = metrics.feedWidths
        record.specEnabled = metrics.specEnabled
        record.specRounds = metrics.specRounds
        record.specDrafted = metrics.specDrafted
        record.specAccepted = metrics.specAccepted
        record.specAcceptanceRate = metrics.draftAcceptanceRate
        record.specFallbackRounds = metrics.specFallbackRounds
        record.footprintAfterPrefillMB = megabytes(metrics.footprintAfterPrefillBytes)
        record.footprintAtEndMB = megabytes(metrics.footprintAtEndBytes)
        record.peakFootprintMB = megabytes(metrics.peakMemoryBytes)
        record.availableMemoryMB = megabytes(metrics.availableMemoryBytes)
        record.thermalStateStart = metrics.thermalStateStart
        record.thermalStateEnd = metrics.thermalStateEnd
        append(record)
    }

    @MainActor
    static func agentTool(
        context: AgentMetricsContext, name: String, arg: String?, seconds: Double,
        resultChars: Int, isError: Bool, modelID: String?, bundleFolder: String?
    ) {
        var record = base(kind: "agent-tool")
        record.apply(context)
        record.modelID = modelID
        record.bundleFolder = bundleFolder
        record.toolName = name
        record.toolArg = arg.map { String($0.prefix(200)) }
        record.toolSeconds = seconds
        record.toolResultChars = resultChars
        record.toolError = isError
        append(record)
    }

    @MainActor
    static func liveCycle(
        report: LiveCycleReport, modelID: String?, hfRevision: String?,
        computeUnits: String?, bundleFolder: String?
    ) {
        var record = base(kind: "live-cycle")
        record.modelID = modelID
        record.hfRevision = hfRevision
        record.computeUnits = computeUnits
        record.bundleFolder = bundleFolder
        record.cycleIndex = report.index
        record.captionLanguage = report.language.rawValue
        record.captionMode = report.mode.rawValue
        record.imageRows = report.imageRows
        record.promptTokens = report.promptTokens
        record.generatedTokens = report.generatedTokens
        record.captureSeconds = report.captureSeconds
        record.visionEncodeSeconds = report.encodeSeconds
        record.visionEncodeWaitSeconds = report.encodeWaitSeconds
        record.prefillSeconds = report.feedSeconds
        record.ttftSeconds = report.encodeWaitSeconds + report.feedSeconds
        record.cycleSeconds = report.cycleSeconds
        record.decodeTokPerSec = report.tokensPerSecond
        record.peakFootprintMB = megabytes(report.footprintBytes)
        record.availableMemoryMB = megabytes(availableMemoryBytes())
        record.thermalStateEnd = report.thermal
        append(record)
    }

    @MainActor
    static func kv(
        info: KVCheckpointInfo, op: String, modelID: String?, hfRevision: String?,
        computeUnits: String?, bundleFolder: String?
    ) {
        var record = base(kind: "kv")
        record.modelID = modelID
        record.hfRevision = hfRevision
        record.computeUnits = computeUnits
        record.bundleFolder = bundleFolder
        record.kvOp = op
        record.kvBytes = info.fileBytes
        record.kvDurationSeconds = op == "save" ? info.exportSeconds : info.importSeconds
        record.promptTokens = info.tokenCount
        record.generatedTokens = info.continuation?.count
        record.prefillSeconds = info.prefillSeconds
        record.specRounds = info.pldRounds
        record.specDrafted = info.pldDraftedTokens
        record.specAccepted = info.pldAcceptedTokens
        record.specFallbackRounds = info.pldFallbackRounds
        record.feedWidths = info.residentPrefillWidths
        record.peakFootprintMB = megabytes(info.peakMemoryBytes)
        record.thermalStateEnd = thermalName()
        append(record)
    }

    @MainActor
    static func error(
        phase: String, reason: String, modelID: String?, hfRevision: String?,
        computeUnits: String?, bundleFolder: String?, promptTokens: Int?, contextLength: Int?
    ) {
        var record = base(kind: "error")
        record.modelID = modelID
        record.hfRevision = hfRevision
        record.computeUnits = computeUnits
        record.bundleFolder = bundleFolder
        record.errorPhase = phase
        record.errorReason = reason
        record.promptTokens = promptTokens
        record.contextLength = contextLength
        record.finishReason = FinishReason.error.rawValue
        record.availableMemoryMB = megabytes(availableMemoryBytes())
        record.thermalStateEnd = thermalName()
        append(record)
    }

    @MainActor
    static func cancelledMessage(
        modelID: String?, hfRevision: String?, computeUnits: String?, bundleFolder: String?
    ) {
        var record = base(kind: "message")
        record.modelID = modelID
        record.hfRevision = hfRevision
        record.computeUnits = computeUnits
        record.bundleFolder = bundleFolder
        record.finishReason = FinishReason.cancelled.rawValue
        append(record)
    }

    @MainActor
    private static func base(kind: String) -> MetricsRecord {
        let (model, os) = deviceAndOS()
        let (level, state) = battery()
        var record = MetricsRecord(
            kind: kind,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            deviceModel: model, osVersion: os, appBuild: appBuild(),
            thermalStateStart: thermalName(), batteryLevel: level, batteryState: state)
        record.sysWiredMB = megabytes(sysWiredBytes())
        return record
    }

    private static let queue = DispatchQueue(label: "coreml-llm-samples.metrics-log")

    private static func append(_ record: MetricsRecord) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard var encoded = try? encoder.encode(record) else { return }
        encoded.append(0x0A)
        let data = encoded
        let url = fileURL
        queue.async { writeLine(data, to: url) }
    }

    static func flush() async {
        await withCheckedContinuation { continuation in
            queue.async { continuation.resume() }
        }
    }

    private static func writeLine(_ data: Data, to url: URL) {
        let descriptor = open(url.path(percentEncoded: false), O_WRONLY | O_APPEND | O_CREAT, 0o644)
        guard descriptor >= 0 else { return }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        try? handle.write(contentsOf: data)
        close(descriptor)
        excludeFromBackup(url)
    }

    private static func documentsDirectory() -> URL {
        (try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? URL(fileURLWithPath: NSHomeDirectory()).appending(path: "Documents")
    }

    private static func megabytes(_ bytes: Int?) -> Double? {
        guard let bytes else { return nil }
        return Double(bytes) / 1_048_576
    }

    /// System-wide wired memory (host_statistics64 wire_count). On iOS this is where ANE
    /// weight residency shows up, so it sits next to the per-stage phys_footprint numbers.
    static func sysWiredBytes() -> Int? {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) { pointer -> kern_return_t in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return Int(stats.wire_count) * sysconf(_SC_PAGESIZE)
    }

    private static func availableMemoryBytes() -> Int? {
        #if os(iOS)
            return Int(os_proc_available_memory())
        #else
            return nil
        #endif
    }

    private static func appBuild() -> String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "\(short) (\(build))"
    }

    private static func deviceAndOS() -> (String, String) {
        #if os(macOS)
            let key = "hw.model"
        #else
            let key = "hw.machine"
        #endif
        var size = 0
        sysctlbyname(key, nil, &size, nil, 0)
        var value = [UInt8](repeating: 0, count: max(size, 1))
        sysctlbyname(key, &value, &size, nil, 0)
        let machine = String(decoding: value.prefix(while: { $0 != 0 }), as: UTF8.self)
        return (machine, ProcessInfo.processInfo.operatingSystemVersionString)
    }

    private static func thermalName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    @MainActor
    private static func battery() -> (Double?, String?) {
        #if canImport(UIKit) && !os(macOS)
            UIDevice.current.isBatteryMonitoringEnabled = true
            let level = UIDevice.current.batteryLevel
            let state: String
            switch UIDevice.current.batteryState {
            case .charging: state = "charging"
            case .full: state = "full"
            case .unplugged: state = "unplugged"
            default: state = "unknown"
            }
            return (level >= 0 ? Double(level) : nil, state)
        #else
            return (nil, nil)
        #endif
    }

    private static func excludeFromBackup(_ url: URL) {
        #if !os(macOS)
            var target = url
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try? target.setResourceValues(values)
        #endif
    }
}

/// Where this app's Core ML artifacts live, and how big they are.
///
/// `MLModel(contentsOf:)` specializes a model for the ANE, and a hit on that specialization makes
/// a chunk load in ~0.2 s where a miss costs ~10 s. On macOS the artifacts land in
/// `~/Library/Caches/com.apple.e5rt.e5bundlecache`; on iOS that directory is **not** in the app's
/// container, so the app can neither measure nor clear it. The scan below reports the whole
/// container `Library/` tree so the growth after a load is visible wherever it lands.
///
/// `CORELLM_CLEAR_E5RT_CACHE=1` deletes any `*e5*bundlecache*` directory it finds inside this
/// app's own container, and says so when there is none.
enum E5RTCache {
    /// Read once, at startup, before anything loads a model. Reading it performs the clear.
    static let startupLines: [String] = report(phase: "startup", clear: true)

    /// Same scan, after a load, so the caller can see which directory grew.
    static func afterLoadLines() -> [String] { report(phase: "afterload", clear: false) }

    private static var libraryURL: URL? {
        FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first
    }

    private static func report(phase: String, clear: Bool) -> [String] {
        guard let library = libraryURL else { return ["[e5rt] \(phase) no library directory"] }
        var lines = ["[e5rt] \(phase) library=\(library.path(percentEncoded: false))"]
        for entry in children(of: library) {
            let (files, bytes) = measure(entry)
            lines.append(
                String(
                    format: "[e5rt] %@ %@ files=%d size=%.1f MiB",
                    phase, entry.lastPathComponent, files, Double(bytes) / 1_048_576))
            // One level deeper for the two that matter, so a growing cache is attributable.
            guard ["Caches", "Application Support"].contains(entry.lastPathComponent) else { continue }
            for child in children(of: entry) {
                let (f, b) = measure(child)
                lines.append(
                    String(
                        format: "[e5rt] %@   %@/%@ files=%d size=%.1f MiB",
                        phase, entry.lastPathComponent, child.lastPathComponent,
                        f, Double(b) / 1_048_576))
            }
        }
        let found = bundleCaches(under: library)
        lines.append(
            "[e5rt] \(phase) e5bundlecache in container: "
                + (found.isEmpty ? "none" : found.map { $0.path(percentEncoded: false) }.joined(separator: " ")))
        guard clear, ProcessInfo.processInfo.environment["CORELLM_CLEAR_E5RT_CACHE"] == "1" else {
            return lines
        }
        guard !found.isEmpty else {
            lines.append(
                "[e5rt] clear requested but nothing to clear — "
                    + "on iOS the specialization cache is outside the app container")
            return lines
        }
        for url in found {
            do {
                try FileManager.default.removeItem(at: url)
                lines.append("[e5rt] cleared \(url.lastPathComponent)")
            } catch {
                lines.append("[e5rt] clear failed \(url.lastPathComponent): \(error)")
            }
        }
        return lines
    }

    private static func children(of url: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil, options: [])) ?? []).sorted {
                $0.lastPathComponent < $1.lastPathComponent
            }
    }

    /// Any directory in the container whose name looks like the e5rt bundle cache.
    private static func bundleCaches(under root: URL) -> [URL] {
        guard
            let walker = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: [.isDirectoryKey],
                options: [], errorHandler: { _, _ in true })
        else { return [] }
        var found: [URL] = []
        for case let child as URL in walker
        where child.lastPathComponent.lowercased().contains("bundlecache") {
            found.append(child)
            walker.skipDescendants()
        }
        return found
    }

    /// `(file count, bytes)` for a directory tree; `(0, 0)` when it is not there.
    private static func measure(_ url: URL) -> (Int, Int) {
        guard
            let walker = FileManager.default.enumerator(
                at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
                options: [], errorHandler: { _, _ in true })
        else { return (0, 0) }
        var files = 0
        var bytes = 0
        for case let child as URL in walker {
            let values = try? child.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            guard values?.isRegularFile == true else { continue }
            files += 1
            bytes += values?.fileSize ?? 0
        }
        return (files, bytes)
    }
}
