#if os(iOS)
import AVFoundation
import CoreMLBackend
import Speech
import SwiftUI

/// Drives the voice dialog with nobody talking.
///
/// It cannot test recognition — that needs a human — but it does exercise
/// everything around it: the ja-JP voice, the audio session switch, the silence cap,
/// the "聞き取れませんでした" retry and the skip path, and it proves the state machine
/// terminates instead of hanging. Run it before handing the phone to a person.
///
///     xcrun devicectl device process launch --console --device <udid> \
///       com.coreml-llm-samples.DemoApp --mapper-dialog-selftest
enum MapperDialogSelfTest {
    static var isRequested: Bool { CommandLine.arguments.contains("--mapper-dialog-selftest") }
    static let fileName = "mapper-dialog-selftest-output.txt"
}

struct MapperDialogSelfTestView: View {
    @State private var log = MapperHeadlessLog(fileName: MapperDialogSelfTest.fileName)
    @State private var model = MapperViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if log.finished {
                    Image(systemName: "checkmark.circle").foregroundStyle(.green)
                } else {
                    ProgressView().controlSize(.small)
                }
                Text(log.finished ? "Dialog self-test finished" : "Dialog self-test running")
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

    private func run() async {
        log.record("=== Mapper dialog self-test — \(UIDevice.current.model) "
                   + "iOS \(UIDevice.current.systemVersion) — \(Date().ISO8601Format()) ===")
        log.record("NOTE: nobody is speaking, so every answer times out. This checks the "
                   + "voice, the audio session, the retry/skip path and that it terminates.")

        // --- permissions ---
        let speechAuth = SFSpeechRecognizer.authorizationStatus()
        log.record("speech authorization: \(speechAuth.rawValue) "
                   + "(\(speechAuth == .authorized ? "authorized" : "NOT authorized"))")
        log.record("microphone: \(AVAudioApplication.shared.recordPermission == .granted ? "granted" : "NOT granted")")
        let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
        log.record("ja-JP recognizer: \(recognizer != nil ? "present" : "missing"), "
                   + "onDevice=\(recognizer?.supportsOnDeviceRecognition ?? false)")

        // --- voices ---
        let japanese = AVSpeechSynthesisVoice.speechVoices().filter { $0.language == "ja-JP" }
        log.record("ja-JP voices: \(japanese.count)")
        for voice in japanese {
            let quality = switch voice.quality {
            case .premium: "premium"
            case .enhanced: "enhanced"
            default: "default"
            }
            log.record("  - \(voice.name) [\(quality)] \(voice.identifier)")
        }
        log.record("chosen: \(MapperTTS().voiceDescription)")

        // --- questions, so the wording can be reviewed without listening ---
        log.record("--- questions ---")
        for field in MapperForm.fields {
            log.record("  \(field.name): 「\(MapperDialogController.question(for: field))」")
        }

        // --- keyword and unit helpers ---
        // Expected answers are asserted, not just printed: a wrong "yes" here means a
        // wrong value is written into the sheet without anyone noticing.
        log.record("--- yes/no keywords ---")
        let yesNoCases: [(String, Bool?)] = [
            ("はい", true), ("うん", true), ("ええ", true), ("合ってる", true),
            ("合ってます", true), ("オッケー", true), ("OK", true), ("それで", true),
            ("そう", true), ("いいです", true),
            ("問題ないです", true), ("問題ありません", true), ("大丈夫", true),
            ("いいえ", false), ("いや、違う", false), ("違います", false),
            ("ちがいます", false), ("そうではない", false), ("そうじゃない", false),
            ("やり直し", false), ("もう一度", false), ("だめ", false),
            ("", nil), ("えーっと", nil), ("わかりません", nil),
        ]
        var yesNoFailures = 0
        for (sample, expected) in yesNoCases {
            let got = MapperDialogController.yesNo(sample)
            let word = { (value: Bool?) in value.map { $0 ? "肯定" : "否定" } ?? "判定不能" }
            let ok = got == expected
            if !ok { yesNoFailures += 1 }
            log.record("  \(ok ? "ok " : "NG ")「\(sample)」 -> \(word(got)) (期待 \(word(expected)))")
        }
        log.record("  yes/no: \(yesNoCases.count - yesNoFailures)/\(yesNoCases.count) 一致")
        log.record("--- spoken unit normalisation ---")
        for sample in ["0.45メガパスカル", "72度", "摂氏72度", "0.5キロパスカル", "れいてんよんご"] {
            let normalised = MapperDialogController.normaliseSpokenUnits(sample)
            let candidate = MapperForm.numberCandidates(in: normalised).first
            log.record("  「\(sample)」 -> 「\(normalised)」 -> "
                       + (candidate.map { "\($0.value)\($0.unit)" } ?? "数値なし"))
        }

        // --- exact option matches: these must not reach the classifier ---
        log.record("--- option matches ---")
        let oilLeak = MapperForm.fields.first { $0.key == "oil_leak" }!
        let sound = MapperForm.fields.first { $0.key == "abnormal_sound" }!
        let nextAction = MapperForm.fields.first { $0.key == "next_action" }!
        let matchCases: [(MapperForm.Field, String, String?)] = [
            (oilLeak, "なし", "なし"),
            (oilLeak, "なし。", "なし"),
            (oilLeak, "漏れていない", "なし"),
            (oilLeak, "あり", "あり"),
            (sound, "未点検", "未点検"),
            (sound, "異常なし", "正常"),
            (sound, "問題ありません", "正常"),
            (nextAction, "グリスアップ", "給油"),
            (nextAction, "掃除", "清掃"),
            // Not a plain option name: the classifier should still decide these.
            (sound, "少し音がする", nil),
        ]
        var matchFailures = 0
        for (field, heard, expected) in matchCases {
            let got = MapperDialogController.exactOption(for: field, heard: heard)
            let ok = got == expected
            if !ok { matchFailures += 1 }
            log.record("  \(ok ? "ok " : "NG ")\(field.name)「\(heard)」 -> "
                       + "\(got ?? "モデルに回す") (期待 \(expected ?? "モデルに回す"))")
        }
        log.record("  matches: \(matchCases.count - matchFailures)/\(matchCases.count) 一致")

        // --- an empty final must not throw away what was already heard ---
        log.record("--- final resolution ---")
        let finalCases: [(String?, String, String)] = [
            // The device case: partial had the answer, final came back empty.
            (nil, "未点検", "未点検"),
            ("", "未点検", "未点検"),
            ("   ", "0.25メガパスカル", "0.25メガパスカル"),
            // A real final wins over the partial it supersedes.
            ("20度", "20", "20度"),
            // Nothing heard at all stays nothing.
            ("", "", ""),
            (nil, "", ""),
        ]
        var finalFailures = 0
        for (text, partial, expected) in finalCases {
            let got = MapperSpeech.resolveFinal(text: text, lastPartial: partial)
            let ok = got == expected
            if !ok { finalFailures += 1 }
            log.record("  \(ok ? "ok " : "NG ")final=\(text.map { "「\($0)」" } ?? "nil") "
                       + "partial「\(partial)」 -> 「\(got)」(期待「\(expected)」)")
        }
        log.record("  final: \(finalCases.count - finalFailures)/\(finalCases.count) 一致")

        // --- which answers skip the yes/no round trip ---
        log.record("--- confirmation skipping ---")
        func numberField(_ key: String) -> MapperForm.Field {
            MapperForm.fields.first { $0.key == key }!
        }
        func sample(_ value: String, _ score: Float, _ note: String) -> MapperFieldResult {
            MapperFieldResult(key: "k", name: "n", value: value, unit: "", score: score,
                              status: .autoFilled, topTwo: [], note: note)
        }
        let skipCases: [(MapperForm.Field, MapperFieldResult, String?)] = [
            (sound, sample("未点検", 1, "一致"), "一致"),
            (sound, sample("正常", 0.92, ""), "自動(0.92)"),
            (sound, sample("正常", 0.85, ""), "自動(0.85)"),
            // Below the threshold: still worth asking.
            (sound, sample("正常", 0.84, ""), nil),
            // Numbers are always confirmed, however confident the reading looks.
            (numberField("discharge_pressure"), sample("0.25", 1, ""), nil),
            (numberField("temperature"), sample("20", 1, "一致"), nil),
        ]
        var skipFailures = 0
        for (field, result, expected) in skipCases {
            let got = MapperDialogController.autoAcceptReason(for: field, result: result)
            let ok = got == expected
            if !ok { skipFailures += 1 }
            log.record("  \(ok ? "ok " : "NG ")\(field.name) \(result.value)"
                       + String(format: "(%.2f)", result.score)
                       + " -> \(got ?? "確認する") (期待 \(expected ?? "確認する"))")
        }
        log.record("  skipping: \(skipCases.count - skipFailures)/\(skipCases.count) 一致")

        // --- a retry has to name the field, or the next answer lands on it ---
        log.record("--- retry wording ---")
        var wordingFailures = 0
        for field in MapperForm.fields {
            let question = MapperDialogController.question(for: field)
            let retry = "では、もう一度。" + question
            let ok = retry.contains(field.name)
            if !ok { wordingFailures += 1 }
            log.record("  \(ok ? "ok " : "NG ")\(field.name): 「\(retry)」")
        }
        log.record("  retry wording: \(MapperForm.fields.count - wordingFailures)/"
                   + "\(MapperForm.fields.count) 項目名を含む")

        // --- the real run, with silence for every answer ---
        log.record("--- running the dialog with no speaker (expect 6 skips) ---")
        log.record("step: loading model")
        await model.loadModel()
        log.record("step: model loaded (\(model.status))")
        guard let dialog = model.dialog else {
            log.record("dialog unavailable (model load failed)")
            log.markFinished()
            return
        }
        log.record("step: dialog created")

        // --- number rules: corrections win, a foreign unit is flagged ---
        log.record("--- number answers ---")
        let pressure = MapperForm.fields.first { $0.key == "discharge_pressure" }!
        let temperature = MapperForm.fields.first { $0.key == "temperature" }!
        let numberCases: [(MapperForm.Field, String, String, MapperFieldResult.Status)] = [
            (pressure, "0.45メガパスカル", "0.45MPa", .autoFilled),
            // A spoken correction: the last number is the answer, not the first.
            (pressure, "0.45、違った、0.54メガパスカル", "0.54MPa", .autoFilled),
            (pressure, "0.3じゃなくて0.7メガパスカル", "0.7MPa", .autoFilled),
            // Wrong unit for this field — flagged rather than filled in silently.
            // 度 answered to 吐出圧 is the wrong quantity entirely.
            (pressure, "72度", "72度", .review),
            // 度 answered to 温度 is just how people say ℃.
            (temperature, "72度", "72度", .autoFilled),
        ]
        var numberFailures = 0
        for (field, heard, expectedValue, expectedStatus) in numberCases {
            let result = model.dialog?.decide(field: field, heard: heard)
            let got = result.map { "\($0.value)\($0.unit)" } ?? "なし"
            let status = result?.status
            let ok = got == expectedValue && status == expectedStatus
            if !ok { numberFailures += 1 }
            log.record("  \(ok ? "ok " : "NG ")\(field.name)「\(heard)」 -> \(got)"
                       + "[\(status?.rawValue ?? "—")] (期待 \(expectedValue)[\(expectedStatus.rawValue)])"
                       + (result?.note.isEmpty == false ? " note=\(result!.note)" : ""))
        }
        log.record("  numbers: \(numberCases.count - numberFailures)/\(numberCases.count) 一致")

        log.record("step: requesting permission")
        let granted = await model.speech.requestPermissionAndWait()
        log.record("step: permission granted=\(granted)")
        log.record("step: speaking one test utterance")
        await MapperTTS().speak("テストです")
        log.record("step: test utterance done")
        let started = Date()
        dialog.onEvent = { log.record("  dialog| \($0)") }

        // Scripted answers, standing in for recognised speech. They walk every branch
        // of the state machine: a clean answer, a retry after nothing was heard, a
        // rejected value that gets re-asked, and a field skipped after two silences.
        var script = [
            "0.45メガパスカル", "はい",          // 吐出圧: accepted first time
            "", "72度", "はい",                   // 温度: nothing heard, then accepted
            "なし", "いいえ", "あり", "はい",      // 油漏れ: rejected, re-asked, accepted
            "正常", "はい",                       // 異音: accepted
            "", "",                               // 振動: silent twice -> skipped
            "給油", "はい",                       // 次回作業: accepted
        ]
        log.record("step: starting dialog with \(script.count) scripted answers")
        let runner = Task { await dialog.start() }

        // Feed an answer whenever the dialog is actually waiting for one.
        while !runner.isCancelled, dialog.stage != .finished {
            try? await Task.sleep(for: .milliseconds(80))
            if !script.isEmpty, dialog.provideAnswer(script[0]) {
                log.record("    fed| 「\(script.removeFirst())」")
            }
            if Date().timeIntervalSince(started) > 120 {
                log.record("TIMED OUT waiting for the dialog")
                dialog.cancel()
                break
            }
        }
        await runner.value
        log.record("step: unused scripted answers: \(script.count)")
        log.record(String(format: "dialog finished in %.1f s, stage=%@",
                          Date().timeIntervalSince(started), dialog.stage.rawValue))
        log.record("entries: \(dialog.entries.count)")
        for entry in dialog.entries {
            log.record(String(format: "  %@ | Q「%@」 heard「%@」 -> %@ confirm=%@ %.1fs",
                              entry.field, entry.question, entry.heard, entry.decided,
                              entry.confirmation, entry.seconds))
        }
        log.record("results: " + dialog.results.map {
            "\($0.name)=\($0.value.isEmpty ? "未入力" : $0.value)"
        }.joined(separator: " | "))
        log.record(dialog.roundTripSummary)

        log.record("=== done ===")
        log.markFinished()
    }
}
#endif
