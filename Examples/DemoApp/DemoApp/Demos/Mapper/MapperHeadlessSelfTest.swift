#if os(iOS)
import CoreMLBackend
import Observation
import SwiftUI

/// Headless run of the whole Mapper measurement plan, so the numbers come off the
/// device without anyone tapping through the UI.
///
///     xcrun devicectl device process launch --console \
///       --device <udid> com.coreml-llm-samples.DemoApp --mapper-selftest
///
/// Every line also lands in `mapper-selftest-output.txt` in the app's Documents.
enum MapperHeadlessSelfTest {
    static var isRequested: Bool { CommandLine.arguments.contains("--mapper-selftest") }
    static let fileName = "mapper-selftest-output.txt"
}

@MainActor
@Observable
final class MapperHeadlessLog {
    private(set) var lines: [String] = []
    private(set) var finished = false
    @ObservationIgnored private let url: URL

    init(fileName: String = MapperHeadlessSelfTest.fileName) {
        url = ModelStorage.documentsDirectory().appending(path: fileName)
        try? Data().write(to: url, options: .atomic)
    }

    func record(_ line: String) {
        print(line)
        lines.append(line)
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: Data((line + "\n").utf8))
    }

    func markFinished() { finished = true }
}

struct MapperHeadlessSelfTestView: View {
    @State private var log = MapperHeadlessLog()

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if log.finished {
                    Image(systemName: "checkmark.circle").foregroundStyle(.green)
                } else {
                    ProgressView().controlSize(.small)
                }
                Text(log.finished ? "Mapper self-test finished" : "Mapper self-test running")
                    .font(.headline)
            }
            ScrollView {
                Text(log.lines.joined(separator: "\n"))
                    .font(.system(size: 10, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
        }
        .padding()
        .task { await run() }
    }

    private func megabytes(_ bytes: Int?) -> String {
        bytes.map { String(format: "%.1f MB", Double($0) / 1_048_576) } ?? "—"
    }

    private func run() async {
        let device = await UIDevice.current.model
        let os = await UIDevice.current.systemVersion
        log.record("=== Mapper self-test — \(device) iOS \(os) — \(Date().ISO8601Format()) ===")

        let before = mapperMemoryFootprint()
        log.record("memory before load: \(megabytes(before))")

        let firstLoad: Double
        let mapper: GLiNERMapper
        do {
            let started = Date()
            let classifier = try GLiNERClassifier(
                modelURL: try Self.modelURL(), tokenizerFolder: try Self.tokenizerURL())
            firstLoad = Date().timeIntervalSince(started)
            mapper = GLiNERMapper(classifier: classifier)
        } catch {
            log.record("LOAD FAILED: \(error)")
            log.markFinished()
            return
        }
        let afterLoad = mapperMemoryFootprint()
        log.record(String(format: "model load (1st): %.3f s", firstLoad))
        log.record("memory after load: \(megabytes(afterLoad))")

        var peak = afterLoad ?? 0
        func notePeak() { peak = max(peak, mapperMemoryFootprint() ?? 0) }

        // --- parity + accuracy over the 30 fixture sentences ---
        guard let fixture = try? MapperFixture.load() else {
            log.record("FIXTURE MISSING")
            log.markFinished()
            return
        }
        let test = MapperSelfTest(mapper: mapper, fixture: fixture)

        let parity = test.runParity()
        notePeak()
        log.record("parity: calls=\(parity.calls) arrayMismatches=\(parity.arrayMismatches) "
                   + "scores=\(parity.scoresCompared) over0.02=\(parity.scoresOverBudget) "
                   + "argmaxDisagreements=\(parity.argmaxDisagreements) "
                   + String(format: "worst=%.6f mean=%.6f", parity.worstDelta, parity.meanDelta))
        log.record("  worst: \(parity.worstDetail)")

        let report = test.runAccuracy()
        notePeak()
        log.record("--- per-field accuracy (30 sentences, rules unchanged) ---")
        for score in report.fieldScores {
            log.record(String(format: "  %@: %d/%d (%.1f%%)",
                              score.name, score.matched, score.total, score.rate * 100))
        }
        log.record(String(format: "  ALL FIELDS: %d/%d (%.1f%%)",
                          report.allFieldsMatched, report.sentenceCount, report.allFieldsRate * 100))
        log.record(String(format: "  [reference only] expected in top-2 labels (mentioned list fields): "
                          + "%d/%d (%.1f%%)", report.top2Matched, report.top2Total, report.top2Rate * 100))

        log.record("--- mismatches (\(report.failures.count)) ---")
        for failure in report.failures { log.record("  \(failure)") }

        // --- experiment: ask "is this field mentioned?" as its own question ---
        // Not the demo default. Same thresholds, same sentences; the only change is
        // one extra binary call per list field.
        let gated = GLiNERMapper(classifier: mapper.classifier, useMentionGate: true)
        let gatedReport = MapperSelfTest(mapper: gated, fixture: fixture).runAccuracy()
        notePeak()
        log.record("--- per-field accuracy WITH mention gate (experiment, default OFF) ---")
        for (off, on) in zip(report.fieldScores, gatedReport.fieldScores) {
            log.record(String(format: "  %@: OFF %d/%d (%.1f%%) -> ON %d/%d (%.1f%%)",
                              on.name, off.matched, off.total, off.rate * 100,
                              on.matched, on.total, on.rate * 100))
        }
        log.record(String(format: "  ALL FIELDS: OFF %d/%d (%.1f%%) -> ON %d/%d (%.1f%%)",
                          report.allFieldsMatched, report.sentenceCount, report.allFieldsRate * 100,
                          gatedReport.allFieldsMatched, gatedReport.sentenceCount,
                          gatedReport.allFieldsRate * 100))

        // --- actual output per sentence, for the demo script ---
        log.record("--- full output per sentence ---")
        for testCase in fixture.cases {
            guard let outcome = try? mapper.map(text: testCase.text) else { continue }
            let rendered = outcome.fields.map { field in
                let top = field.topTwo.prefix(2)
                    .map { "\($0.label):\(String(format: "%.2f", $0.score))" }.joined(separator: ",")
                return "\(field.name)=\(field.value.isEmpty ? "未言及" : field.value)"
                    + "[\(field.status.rawValue)\(field.note.isEmpty ? "" : "/" + field.note)]{\(top)}"
            }.joined(separator: " | ")
            log.record("  「\(testCase.text)」 -> \(rendered)")
        }
        notePeak()

        // --- timing: 20 consecutive runs of one sentence ---
        let timingText = fixture.cases.first?.text ?? "吐出圧0.45MPa、温度72度"
        var samples: [Double] = []
        for _ in 0..<20 {
            let started = Date()
            _ = try? mapper.map(text: timingText)
            samples.append(Date().timeIntervalSince(started) * 1000)
            notePeak()
        }
        let sorted = samples.sorted()
        let median = sorted[sorted.count / 2]
        let p95 = sorted[min(sorted.count - 1, Int(ceil(0.95 * Double(sorted.count))) - 1)]
        let calls = (try? mapper.map(text: timingText))?.modelCalls ?? 0
        log.record(String(format: "per-sentence (20 runs, %d model calls): median=%.1f ms p95=%.1f ms "
                          + "min=%.1f max=%.1f  text=「%@」",
                          calls, median, p95, sorted.first ?? 0, sorted.last ?? 0, timingText))

        var gatedSamples: [Double] = []
        for _ in 0..<20 {
            let started = Date()
            _ = try? gated.map(text: timingText)
            gatedSamples.append(Date().timeIntervalSince(started) * 1000)
            notePeak()
        }
        let gatedSorted = gatedSamples.sorted()
        let gatedCalls = (try? gated.map(text: timingText))?.modelCalls ?? 0
        log.record(String(format: "per-sentence WITH gate (20 runs, %d model calls): median=%.1f ms "
                          + "p95=%.1f ms",
                          gatedCalls, gatedSorted[gatedSorted.count / 2],
                          gatedSorted[min(gatedSorted.count - 1, Int(ceil(0.95 * Double(gatedSorted.count))) - 1)]))
        log.record("memory peak: \(megabytes(peak))")

        // --- second load, same process (model already compiled) ---
        if let second = try? {
            let started = Date()
            _ = try GLiNERClassifier(
                modelURL: try Self.modelURL(), tokenizerFolder: try Self.tokenizerURL())
            return Date().timeIntervalSince(started)
        }() {
            log.record(String(format: "model load (2nd): %.3f s", second))
        }
        notePeak()
        log.record("memory peak (final): \(megabytes(peak))")

        // --- image input: Vision OCR on the bundled synthetic slip, then map it ---
        if let slip = Bundle.main.url(forResource: "inspection-slip", withExtension: "png"),
            let data = try? Data(contentsOf: slip) {
            let started = Date()
            if let ocr = try? MapperOCR.recognize(imageData: data) {
                log.record(String(format: "--- image input: Vision OCR (%.0f ms) ---",
                                  Date().timeIntervalSince(started) * 1000))
                for line in ocr.split(separator: "\n") { log.record("  ocr| \(line)") }
                if let outcome = try? mapper.map(text: ocr) {
                    let rendered = outcome.fields.map {
                        "\($0.name)=\($0.value.isEmpty ? "未言及" : $0.value)"
                        + "[\($0.status.rawValue)]\(String(format: "(%.2f)", $0.score))"
                    }.joined(separator: " | ")
                    log.record("  mapped| \(rendered)")
                } else {
                    log.record("  mapped| FAILED (likely over the 128-subword bucket)")
                }
            } else {
                log.record("--- image input: OCR failed ---")
            }
            notePeak()
        }

        // --- on-device speech availability ---
        let speech = MapperSpeech()
        speech.refreshAvailability()
        log.record("speech: \(speech.statusText)")

        log.record("=== done ===")
        log.markFinished()
    }

    private static func modelURL() throws -> URL {
        let name = "gliner2_multi_classification_fp16_L128_K8"
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") { return url }
        if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") { return url }
        throw MapperError.missingResource("\(name).mlmodelc")
    }

    private static func tokenizerURL() throws -> URL {
        if let url = Bundle.main.url(forResource: "tokenizer", withExtension: nil) { return url }
        if let url = Bundle.main.url(forResource: "tokenizer", withExtension: "json") {
            return url.deletingLastPathComponent()
        }
        throw MapperError.missingResource("tokenizer/")
    }
}
#endif
