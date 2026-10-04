#if os(iOS)
import CoreMLBackend
import Observation
import SwiftUI

/// Runs the same 30 sentences through the Gemma 4 Core ML engine the app already
/// ships, so the generative route can be compared with the GLiNER2.5 classifier
/// under identical conditions: same six fields, same expectations, same scoring.
///
///     xcrun devicectl device process launch --console --device <udid> \
///       com.coreml-llm-samples.DemoApp --gemma-mapper-selftest --model <bundle folder>
///
/// The Gemma engine itself is not modified — this only drives `ChatViewModel`.
enum GemmaMapperSelfTest {
    static var isRequested: Bool { CommandLine.arguments.contains("--gemma-mapper-selftest") }
    static let fileName = "gemma-mapper-selftest-output.txt"

    /// `--model <folder>`; defaults to the catalog's E2B bundle.
    static var modelPath: String {
        let arguments = CommandLine.arguments
        if let index = arguments.firstIndex(of: "--model"), index + 1 < arguments.count {
            return arguments[index + 1]
        }
        return "gemma-4-e2b-speculative-pal6"
    }

    /// `--max-tokens <n>`: the sheet is ~120 tokens of JSON; the cap stops runaway prose.
    static var maxTokens: Int {
        let arguments = CommandLine.arguments
        if let index = arguments.firstIndex(of: "--max-tokens"), index + 1 < arguments.count,
            let value = Int(arguments[index + 1]) {
            return value
        }
        return 256
    }
}

struct GemmaMapperSelfTestView: View {
    @State private var log = MapperHeadlessLog(fileName: GemmaMapperSelfTest.fileName)

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if log.finished {
                    Image(systemName: "checkmark.circle").foregroundStyle(.green)
                } else {
                    ProgressView().controlSize(.small)
                }
                Text(log.finished ? "Gemma mapper self-test finished" : "Gemma mapper self-test running")
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

    private static func directorySize(_ url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
        return total
    }

    private func run() async {
        let device = UIDevice.current.model
        let os = UIDevice.current.systemVersion
        log.record("=== Gemma mapper self-test — \(device) iOS \(os) — \(Date().ISO8601Format()) ===")

        let path = SelfTest.resolve(GemmaMapperSelfTest.modelPath)
        log.record("model: \(path)")
        let bundleBytes = Self.directorySize(URL(filePath: path))
        log.record(String(format: "model size on disk: %.2f GB", Double(bundleBytes) / 1_073_741_824))
        guard FileManager.default.fileExists(atPath: path) else {
            log.record("MODEL NOT FOUND — download it in the Models tab first")
            log.markFinished()
            return
        }

        guard let fixture = try? MapperFixture.load() else {
            log.record("FIXTURE MISSING")
            log.markFinished()
            return
        }

        log.record("--- prompt (fixed, one prompt for all 30 sentences) ---")
        for line in MapperSheetJSON.prompt(for: "<文>").split(separator: "\n", omittingEmptySubsequences: false) {
            log.record("  | \(line)")
        }

        let before = mapperMemoryFootprint()
        log.record("memory before load: \(megabytes(before))")

        let viewModel = ChatViewModel()
        viewModel.maxTokens = GemmaMapperSelfTest.maxTokens
        let loadStart = Date()
        await viewModel.loadModel(path: path)
        let loadSeconds = Date().timeIntervalSince(loadStart)
        guard viewModel.isModelLoaded else {
            log.record("LOAD FAILED: \(viewModel.phaseDescription)")
            log.markFinished()
            return
        }
        let afterLoad = mapperMemoryFootprint()
        log.record(String(format: "model load: %.3f s", loadSeconds))
        log.record("memory after load: \(megabytes(afterLoad))")
        var peak = afterLoad ?? 0
        func notePeak() { peak = max(peak, mapperMemoryFootprint() ?? 0) }

        var matched: [String: Int] = [:]
        var allFieldsMatched = 0
        var parseFailures = 0
        var totals: [Double] = []
        var prefills: [Double] = []
        var failures: [String] = []

        for testCase in fixture.cases {
            // Each sentence is independent: clear the KV so no sentence sees another.
            viewModel.reset()
            await viewModel.engineHandle?.engine.resetConversation()

            viewModel.input = MapperSheetJSON.prompt(for: testCase.text)
            let started = Date()
            guard viewModel.send() else {
                log.record("  send rejected for 「\(testCase.text)」 (\(viewModel.phaseDescription))")
                parseFailures += 1
                continue
            }
            await viewModel.generation?.value
            let total = Date().timeIntervalSince(started) * 1000
            notePeak()

            let reply = viewModel.messages.last { $0.role == .assistant }?.text ?? ""
            // TTFT from the status line the app already prints = prefill (+ first token).
            let ttft = (SelfTest.firstNumber(in: viewModel.statusLine, after: "TTFT ") ?? 0) * 1000
            totals.append(total)
            prefills.append(ttft)

            guard let fields = MapperSheetJSON.parse(reply) else {
                parseFailures += 1
                failures.append("\(testCase.text): JSON parse failed — "
                    + reply.replacingOccurrences(of: "\n", with: " ").prefix(160).description)
                log.record("  「\(testCase.text)」 -> PARSE FAILED")
                continue
            }

            var allMatch = true
            for field in fields {
                let expected = testCase.expected[field.key]
                if MapperScoring.matches(field, expected: expected) {
                    matched[field.key, default: 0] += 1
                } else {
                    allMatch = false
                    failures.append("\(testCase.text) / \(field.name): got "
                        + "\(field.value.isEmpty ? "未言及" : field.value) expected \(expected ?? "未言及")")
                }
            }
            if allMatch { allFieldsMatched += 1 }

            let rendered = fields.map {
                "\($0.name)=\($0.value.isEmpty ? "未言及" : $0.value)"
            }.joined(separator: " | ")
            log.record(String(format: "  「%@」 -> %@  [%.0f ms, prefill %.0f ms]",
                              testCase.text, rendered, total, ttft))
            log.record("    raw| " + reply.replacingOccurrences(of: "\n", with: " "))
        }

        log.record("--- per-field accuracy (same 30 sentences, same scoring as GLiNER) ---")
        for field in MapperForm.fields {
            let count = matched[field.key] ?? 0
            log.record(String(format: "  %@: %d/%d (%.1f%%)", field.name, count,
                              fixture.cases.count, Double(count) / Double(fixture.cases.count) * 100))
        }
        log.record(String(format: "  ALL FIELDS: %d/%d (%.1f%%)", allFieldsMatched,
                          fixture.cases.count,
                          Double(allFieldsMatched) / Double(fixture.cases.count) * 100))
        log.record("  JSON parse failures: \(parseFailures)/\(fixture.cases.count)")

        func stat(_ values: [Double], _ label: String) {
            guard !values.isEmpty else { return }
            let sorted = values.sorted()
            log.record(String(format: "  %@: median=%.0f ms p95=%.0f ms min=%.0f max=%.0f (n=%d)",
                              label, sorted[sorted.count / 2],
                              sorted[min(sorted.count - 1, Int(ceil(0.95 * Double(sorted.count))) - 1)],
                              sorted.first ?? 0, sorted.last ?? 0, sorted.count))
        }
        log.record("--- time per sentence (prompt in -> JSON complete) ---")
        stat(totals, "total")
        stat(prefills, "prefill (TTFT)")
        stat(zip(totals, prefills).map { $0 - $1 }, "decode")
        log.record("memory peak: \(megabytes(peak))")

        log.record("--- mismatches (\(failures.count)) ---")
        for failure in failures { log.record("  \(failure)") }

        log.record("=== done ===")
        log.markFinished()
    }
}
#endif
