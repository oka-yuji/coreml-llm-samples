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

    var modelLoadSeconds: Double?
    var ttftSeconds: Double?
    var prefillSeconds: Double?
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

    var thermalStateStart: String?
    var thermalStateEnd: String?
    var batteryLevel: Double?
    var batteryState: String?

    var models: [String]?
}

enum MetricsLog {
    static var fileURL: URL {
        documentsDirectory().appending(path: "metrics.jsonl")
    }

    static func session(models: [String]) {
        var record = base(kind: "session")
        record.models = models
        append(record)
    }

    static func message(
        metrics: GenerationMetrics, modelID: String?, hfRevision: String?,
        computeUnits: String?, bundleFolder: String?, modelLoadSeconds: Double?
    ) {
        var record = base(kind: "message")
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

    private static func base(kind: String) -> MetricsRecord {
        let (model, os) = deviceAndOS()
        let (level, state) = battery()
        return MetricsRecord(
            kind: kind,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            deviceModel: model, osVersion: os, appBuild: appBuild(),
            thermalStateStart: thermalName(), batteryLevel: level, batteryState: state)
    }

    private static func append(_ record: MetricsRecord) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard var data = try? encoder.encode(record) else { return }
        data.append(0x0A)
        let url = fileURL
        let fileManager = FileManager.default
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: url, options: .atomic)
        }
        excludeFromBackup(url)
        _ = fileManager
    }

    private static func documentsDirectory() -> URL {
        (try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? URL(fileURLWithPath: NSHomeDirectory()).appending(path: "Documents")
    }

    private static func megabytes(_ bytes: Int?) -> Double? {
        guard let bytes else { return nil }
        return Double(bytes) / 1_048_576
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
        var value = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname(key, &value, &size, nil, 0)
        let machine = String(cString: value)
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
