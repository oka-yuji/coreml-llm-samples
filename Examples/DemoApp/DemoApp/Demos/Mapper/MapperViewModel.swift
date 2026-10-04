import CoreML
import CoreMLBackend
import Foundation
import Observation

@MainActor
@Observable
final class MapperViewModel {
    var inputText = "2番ポンプ、吐出圧0.45MPa、温度72度"
    var fields: [MapperFieldResult] = []
    var status = "モデル未読み込み"
    var isBusy = false
    var lastRunSeconds: Double?
    var lastModelCalls = 0
    var metrics = MapperMetrics()
    var selfTestReport: MapperSelfTestReport?
    var speech = MapperSpeech()
    var errorText: String?

    private var mapper: GLiNERMapper?

    /// The model ships inside the app so the demo needs no network at all.
    /// Xcode compiles the `.mlpackage` to `.mlmodelc` at build time.
    private nonisolated static func modelURL() throws -> URL {
        let name = "gliner2_multi_classification_fp16_L128_K8"
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") { return url }
        if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") { return url }
        throw MapperError.missingResource("\(name).mlmodelc")
    }

    private nonisolated static func tokenizerURL() throws -> URL {
        if let url = Bundle.main.url(forResource: "tokenizer", withExtension: nil) { return url }
        // Folder references land flat in some configurations; fall back to the file.
        if let url = Bundle.main.url(forResource: "tokenizer", withExtension: "json") {
            return url.deletingLastPathComponent()
        }
        throw MapperError.missingResource("tokenizer/")
    }

    func loadModel() async {
        guard mapper == nil, !isBusy else { return }
        isBusy = true
        status = "モデル読み込み中…"
        metrics.footprintBeforeLoad = mapperMemoryFootprint()
        do {
            let (built, seconds) = try await Task.detached(priority: .userInitiated) {
                let started = Date()
                let classifier = try GLiNERClassifier(
                    modelURL: try Self.modelURL(), tokenizerFolder: try Self.tokenizerURL())
                return (GLiNERMapper(classifier: classifier), Date().timeIntervalSince(started))
            }.value
            mapper = built
            metrics.modelLoadSeconds = seconds
            metrics.footprintAfterLoad = mapperMemoryFootprint()
            metrics.footprintPeak = metrics.footprintAfterLoad
            status = String(format: "読み込み完了 %.2f 秒", seconds)
        } catch {
            errorText = "\(error)"
            status = "読み込み失敗"
        }
        isBusy = false
    }

    func map() async {
        guard !isBusy else { return }
        if mapper == nil { await loadModel() }
        guard let mapper else { return }
        let text = inputText
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        isBusy = true
        status = "マッピング中…"
        do {
            let outcome = try await Task.detached(priority: .userInitiated) {
                try mapper.map(text: text)
            }.value
            fields = outcome.fields
            lastRunSeconds = outcome.seconds
            lastModelCalls = outcome.modelCalls
            noteFootprint()
            status = String(format: "完了 %.0f ms(モデル呼び出し %d 回)",
                            outcome.seconds * 1000, outcome.modelCalls)
        } catch {
            errorText = "\(error)"
            status = "マッピング失敗"
        }
        isBusy = false
    }

    /// 20 runs of one sentence: median and p95, per the measurement plan.
    func measure(repetitions: Int = 20) async {
        guard !isBusy else { return }
        if mapper == nil { await loadModel() }
        guard let mapper else { return }
        let text = inputText
        isBusy = true
        status = "計測中…"
        let samples = await Task.detached(priority: .userInitiated) { () -> [Double] in
            var values: [Double] = []
            for _ in 0..<repetitions {
                let started = Date()
                _ = try? mapper.map(text: text)
                values.append(Date().timeIntervalSince(started) * 1000)
            }
            return values
        }.value
        let sorted = samples.sorted()
        if !sorted.isEmpty {
            metrics.perSentenceMedianMs = sorted[sorted.count / 2]
            // Nearest-rank p95.
            metrics.perSentenceP95Ms = sorted[min(sorted.count - 1, Int(ceil(0.95 * Double(sorted.count))) - 1)]
            metrics.repetitions = sorted.count
        }
        noteFootprint()
        status = String(format: "計測完了 中央値 %.0f ms / p95 %.0f ms(%d 回)",
                        metrics.perSentenceMedianMs ?? 0, metrics.perSentenceP95Ms ?? 0, sorted.count)
        isBusy = false
    }

    /// Reload with a fresh MLModel to time a warm (already compiled) load.
    func measureSecondLoad() async {
        guard !isBusy else { return }
        isBusy = true
        status = "2 回目の読み込みを計測中…"
        let seconds = await Task.detached(priority: .userInitiated) { () -> Double? in
            let started = Date()
            guard (try? GLiNERClassifier(
                modelURL: try Self.modelURL(), tokenizerFolder: try Self.tokenizerURL())) != nil
            else { return nil }
            return Date().timeIntervalSince(started)
        }.value
        metrics.secondLoadSeconds = seconds
        noteFootprint()
        status = seconds.map { String(format: "2 回目の読み込み %.2f 秒", $0) } ?? "2 回目の読み込み失敗"
        isBusy = false
    }

    func runSelfTest() async {
        guard !isBusy else { return }
        if mapper == nil { await loadModel() }
        guard let mapper else { return }
        isBusy = true
        status = "自己テスト実行中(30 文)…"
        do {
            let fixture = try MapperFixture.load()
            let test = MapperSelfTest(mapper: mapper, fixture: fixture)
            var report = await Task.detached(priority: .userInitiated) { test.runAccuracy() }.value
            let parity = await Task.detached(priority: .userInitiated) { test.runParity() }.value
            report.worstScoreDelta = parity.worstDelta
            report.arrayMismatches = parity.arrayMismatches
            report.parityCalls = parity.calls
            report.scoresOverBudget = parity.scoresOverBudget
            report.argmaxDisagreements = parity.argmaxDisagreements
            selfTestReport = report
            noteFootprint()
            status = String(
                format: "自己テスト完了 全項目一致 %d/%d、Python との最大差 %.4f",
                report.allFieldsMatched, report.sentenceCount, parity.worstDelta)
        } catch {
            errorText = "\(error)"
            status = "自己テスト失敗"
        }
        isBusy = false
    }

    private func noteFootprint() {
        guard let now = mapperMemoryFootprint() else { return }
        metrics.footprintPeak = max(metrics.footprintPeak ?? 0, now)
    }

    func startDictation() {
        speech.start { [weak self] text in
            guard let self, !text.isEmpty else { return }
            // The transcript is editable before mapping, so a speech slip never
            // looks like a model mistake.
            self.inputText = text
        }
    }
}
