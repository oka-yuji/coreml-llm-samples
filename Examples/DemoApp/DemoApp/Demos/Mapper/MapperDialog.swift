#if os(iOS)
import AVFoundation
import CoreMLBackend
import Foundation
import Observation
import Speech

/// Japanese text-to-speech for the dialog. `speak` returns when the utterance has
/// actually finished playing, which is what lets the microphone open only after the
/// phone has stopped talking — otherwise it hears itself.
@MainActor
final class MapperTTS: NSObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    /// Keyed by the utterance it belongs to. On a real device the delegate callback
    /// for one utterance can land after the next `speak` has already installed its
    /// continuation; resuming the wrong one traps the process, so every callback has
    /// to prove it owns the pending continuation.
    private var pending: (id: ObjectIdentifier, continuation: CheckedContinuation<Void, Never>)?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    /// Identifier of the voice the person picked; empty means choose automatically.
    static let preferenceKey = "mapper.voiceIdentifier"

    /// Every ja-JP voice installed, best quality first and then by name.
    static var japaneseVoices: [AVSpeechSynthesisVoice] {
        AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language == "ja-JP" }
            .sorted {
                qualityRank($0.quality) == qualityRank($1.quality)
                    ? $0.name < $1.name
                    : qualityRank($0.quality) > qualityRank($1.quality)
            }
    }

    static func qualityRank(_ quality: AVSpeechSynthesisVoiceQuality) -> Int {
        switch quality {
        case .premium: 3
        case .enhanced: 2
        default: 1
        }
    }

    static func qualityName(_ quality: AVSpeechSynthesisVoiceQuality) -> String {
        switch quality {
        case .premium: "premium"
        case .enhanced: "enhanced"
        default: "default"
        }
    }

    static func label(for voice: AVSpeechSynthesisVoice) -> String {
        "\(voice.name)(\(qualityName(voice.quality)))"
    }

    /// The saved voice if it is still installed, otherwise the best one available:
    /// premium, then enhanced, then the compact default.
    static var voice: AVSpeechSynthesisVoice? {
        let saved = UserDefaults.standard.string(forKey: preferenceKey) ?? ""
        if !saved.isEmpty, let match = japaneseVoices.first(where: { $0.identifier == saved }) {
            return match
        }
        return japaneseVoices.first ?? AVSpeechSynthesisVoice(language: "ja-JP")
    }

    static var voiceLabel: String {
        guard let voice else { return "ja-JP の音声なし" }
        return label(for: voice)
    }

    var voiceDescription: String { Self.voiceLabel }

    /// Did the last utterance report back on its own, or did it have to be timed out?
    private(set) var lastSpeakTimedOut = false
    private(set) var isSpeaking = false

    /// Speaks and waits for `didFinish` — but never forever.
    ///
    /// When the session is configured for `.playAndRecord` the synthesizer can stay
    /// silent and never call its delegate, and an unbounded wait here freezes the
    /// whole dialog: no speech, no microphone, no crash. That is the exact symptom
    /// this timeout exists to prevent.
    /// How long to allow for one utterance: long sentences legitimately take longer,
    /// and timing the closing summary out would have been read as "speech is broken"
    /// and dropped the run into text mode.
    static func watchdogSeconds(for text: String) -> Double {
        max(8, 3 + 0.35 * Double(text.count))
    }

    func speak(_ text: String) async {
        let timeout = Duration.seconds(Self.watchdogSeconds(for: text))
        isSpeaking = true
        defer { isSpeaking = false }
        lastSpeakTimedOut = false
        let watchdog = Task { [weak self] in
            try? await Task.sleep(for: timeout)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.pending != nil else { return }
                self.lastSpeakTimedOut = true
                self.synthesizer.stopSpeaking(at: .immediate)
                self.pending?.continuation.resume()
                self.pending = nil
            }
        }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let utterance = AVSpeechUtterance(string: text)
            // Re-read each time: the picker can change it between questions.
            utterance.voice = Self.voice
            // A previous utterance that never reported back would strand its caller;
            // release it before taking over.
            pending?.continuation.resume()
            pending = (ObjectIdentifier(utterance), continuation)
            synthesizer.speak(utterance)
        }
        watchdog.cancel()
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
        pending?.continuation.resume()
        pending = nil
    }

    private func finish(_ id: ObjectIdentifier) {
        guard let pending, pending.id == id else { return }
        self.pending = nil
        pending.continuation.resume()
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance
    ) {
        // ObjectIdentifier is Sendable; the utterance itself is not.
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in self.finish(id) }
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance
    ) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in self.finish(id) }
    }
}

/// One field at a time, by voice: ask → listen → map → confirm → next.
///
/// Asking per field removes the hardest problem the batch mode has. In a free
/// sentence the model must decide *whether* a field was mentioned at all, and that
/// is where it fails (it invents values, and reads negation as absence). Here the
/// question already names the field, so "was it mentioned?" never has to be asked:
/// the list classifier runs on the candidates alone, with no 言及なし label.
@MainActor
@Observable
final class MapperDialogController {
    enum Stage: String {
        case idle = "待機"
        case asking = "質問中"
        case listening = "聞いています"
        case mapping = "判定中"
        case confirming = "確認中"
        case finished = "完了"
    }

    struct Entry: Identifiable {
        let id = UUID()
        var field: String
        var question: String
        var heard: String
        var decided: String
        var confirmation: String
        var seconds: Double
    }

    var stage: Stage = .idle {
        didSet { refreshDiagnostics() }
    }
    var currentField = ""
    var entries: [Entry] = []
    var results: [MapperFieldResult] = []
    var partial = ""
    var voiceNote = ""
    /// Non-empty while a yes/no answer is expected, so the view can show those buttons.
    var confirmPrompt = ""
    /// One line for the panel: permissions, on-device support, session, stage.
    var diagnostics = ""
    private var sessionState = "未設定"
    private var ttsFailures = 0

    /// Diagnostics plus the whole event trail, for the copy button. Only eight lines
    /// are on screen, and the useful one is usually further back.
    var fullLog: String {
        ([diagnostics, voiceNote] + events).joined(separator: "\n")
    }

    func refreshDiagnostics() {
        diagnostics = "権限: \(speech.permissionSummary) | 端末内認識: "
            + (speech.supportsOnDevice ? "可" : "不可")
            + " | 経路: \(sessionState) | 声: \(MapperTTS.voiceLabel)"
            + " | 状態: \(stage.rawValue)"
    }
    /// Text mode is the default: on-screen questions answered by tapping.
    ///
    /// The voice path is opt-in because `SFSpeechRecognizer`'s recognition task kills
    /// the process on this device shortly after it starts — see the results page. The
    /// state machine, the questions, the per-field classification and the confirmation
    /// are identical either way; only how the answer arrives differs.
    var textOnlyMode = true

    /// Trace hook: the headless self-test prints these so a hang or a crash can be
    /// placed exactly.
    var onEvent: ((String) -> Void)?
    /// Kept for the panel as well, so a silent run can be read on the phone.
    var events: [String] = []
    private func note(_ message: String) {
        events.append(message)
        if events.count > 1000 { events.removeFirst(events.count - 1000) }
        onEvent?(message)
    }

    private var answerContinuation: CheckedContinuation<String, Never>?
    private let tts = MapperTTS()
    private let speech: MapperSpeech
    private let classifier: GLiNERClassifier
    private var cancelled = false

    init(speech: MapperSpeech, classifier: GLiNERClassifier) {
        self.speech = speech
        self.classifier = classifier
        speech.onEvent = { [weak self] message in self?.note(message) }
    }

    var roundTripSummary: String {
        let times = entries.map(\.seconds)
        guard !times.isEmpty else { return "—" }
        let sorted = times.sorted()
        return String(format: "1 項目あたり 中央値 %.1f 秒 / 最長 %.1f 秒(%d 項目)",
                      sorted[sorted.count / 2], sorted.last ?? 0, sorted.count)
    }

    // MARK: - Questions

    /// How a unit is pronounced. Display keeps the symbol; only speech is spelled out.
    static let spokenUnits = [
        "MPa": "メガパスカル", "kPa": "キロパスカル", "℃": "度",
        "%": "パーセント", "mm": "ミリメートル",
    ]

    static func spoken(_ unit: String) -> String { spokenUnits[unit] ?? unit }

    /// "72度" and "72℃" are the same temperature. Treating them as different units
    /// would flag the ordinary way of saying it as a mismatch.
    static func sameUnit(_ a: String, _ b: String) -> Bool {
        if a == b { return true }
        let celsius: Set<String> = ["℃", "度"]
        return celsius.contains(a) && celsius.contains(b)
    }

    static func question(for field: MapperForm.Field) -> String {
        switch field.kind {
        case .number(let unit): "\(field.name)は、何\(spoken(unit))ですか？"
        case .list(let options): "\(field.name)は、\(options.joined(separator: "、"))？"
        }
    }

    /// Supplied by the view when the person taps an answer in text mode.
    /// Returns false when nothing was waiting, which lets a scripted driver poll.
    @discardableResult
    func provideAnswer(_ text: String) -> Bool {
        guard let continuation = answerContinuation else { return false }
        answerContinuation = nil
        continuation.resume(returning: text)
        return true
    }

    /// Options to offer as buttons for the field being asked.
    var answerChoices: [String] {
        guard let field = MapperForm.fields.first(where: { $0.name == currentField }) else { return [] }
        if case .list(let options) = field.kind { return options }
        return []
    }

    // MARK: - Run

    /// True from the moment `start` is entered until it returns. `stage` is not
    /// enough: it stays `.idle` through the permission prompt and the route setup, so
    /// a second tap in that window would start a second dialog on the same microphone.
    private(set) var isRunning = false

    func start() async {
        guard !isRunning, stage == .idle || stage == .finished else { return }
        isRunning = true
        defer { isRunning = false }
        cancelled = false
        entries = []
        results = []
        // Ask once, up front: a prompt appearing mid-question would skip the answer.
        note("mode: \(textOnlyMode ? "text" : "voice")")
        ttsFailures = 0
        events = []
        sessionState = textOnlyMode ? "不要(タップ回答)" : "未設定"
        var granted = false
        if !textOnlyMode {
            // Ask for both before the first question; a prompt appearing mid-question
            // would eat the answer.
            granted = await speech.requestPermissionAndWait()
            note("permission: \(speech.permissionSummary) -> usable=\(granted)")
            if !granted {
                textOnlyMode = true
                speech.usesPlayAndRecord = false
                await tts.speak("設定でマイクと音声認識を許可してください。タップでお答えください。")
                voiceNote = "マイクか音声認識が許可されていません。設定アプリで許可してください"
            }
        }
        refreshDiagnostics()
        voiceNote = textOnlyMode
            ? "画面の質問にタップで答えます(音声入力は実験扱い)"
            : "読み上げ: \(tts.voiceDescription)"
        speech.usesPlayAndRecord = !textOnlyMode
        speech.silenceTimeout = 1.2
        speech.maxDuration = 6

        // The category is configured once, here, before a single word is spoken.
        // Changing it between speaking and listening is what made AVAudioEngine hand
        // back a stale input format and trap inside installTap.
        if !textOnlyMode {
            // Choose the route and let it settle before the first word is spoken:
            // speaking during an HFP switch clips the start, and recording during one
            // captures silence.
            note("session: choosing route")
            let route = await speech.configureVoiceSession()
            if let failure = speech.lastSessionError {
                sessionState = "失敗(\(failure))"
                textOnlyMode = true
                speech.usesPlayAndRecord = false
                voiceNote = "音声セッションを設定できませんでした: \(failure)"
            } else {
                sessionState = route
            }
        }

        for field in MapperForm.fields {
            if cancelled { break }
            note("field: \(field.name)")
            await ask(field)
        }
        note("all fields done")

        speech.silenceTimeout = nil
        speech.usesPlayAndRecord = false
        stage = .finished
        currentField = ""
        if !cancelled { await readBackSheet() }
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
    }

    func cancel() {
        cancelled = true
        answerContinuation?.resume(returning: "")
        answerContinuation = nil
        tts.stop()
        speech.stop()
        stage = .finished
        currentField = ""
    }

    private func ask(_ field: MapperForm.Field) async {
        currentField = field.name
        let question = Self.question(for: field)
        let started = Date()

        // Two attempts to get a usable answer, then the field is left empty.
        var rejected = false
        for attempt in 0..<2 {
            if cancelled { return }
            stage = .asking
            note("  say (attempt \(attempt))")
            // Always restate the question. A bare "もう一度お願いします。" leaves the
            // person with nothing to answer: on the device log one of them replied
            // 「いいえ」 again, and another answered the *next* field's question.
            let reason: String
            if attempt == 0 { reason = "" } else if rejected {
                reason = "では、もう一度。"
            } else if speech.heardNothing {
                reason = "もう一度お願いします。"
            } else {
                reason = "聞き取れませんでした。もう一度。"
            }
            await say(reason + question)
            note("  say done")
            if cancelled { return }

            note("  listen start")
            let heard = await listen(hints: Self.hints(for: field))
            note("  listen done: 「\(heard)」")
            if cancelled { return }
            guard !heard.isEmpty else { continue }

            stage = .mapping
            guard let result = decide(field: field, heard: heard) else {
                note("    decided: なし(判定できず)")
                continue
            }
            note("    decided: \(result.value)\(result.unit) "
                 + String(format: "(%.2f) by ", result.score)
                 + (result.note == "一致" ? "一致" : "モデル"))

            stage = .confirming
            let confirmed = await confirm(result)
            entries.append(Entry(
                field: field.name, question: question, heard: heard,
                decided: result.value + result.unit,
                confirmation: confirmed ? "はい" : "いいえ",
                seconds: Date().timeIntervalSince(started)))
            if confirmed {
                results.append(result)
                return
            }
            // Not confirmed: ask the same field again, saying so.
            rejected = true
        }

        // Nothing usable after two tries.
        results.append(MapperFieldResult(
            key: field.key, name: field.name, value: "", unit: "", score: 0,
            status: .absent, topTwo: [], note: "未入力"))
        entries.append(Entry(
            field: field.name, question: question, heard: "", decided: "未入力",
            confirmation: "—", seconds: Date().timeIntervalSince(started)))
    }

    /// Decide the value for a field whose name the question already gave.
    ///
    /// No 言及なし label: the answer is known to be about this field, so the only
    /// question left is which candidate (or which number) it carries.
    /// Not private so the self-test can assert the number rules without speaking.
    func decide(field: MapperForm.Field, heard: String) -> MapperFieldResult? {
        switch field.kind {
        case .list(let options):
            // Answered with the option itself: no judgement call to make.
            if let exact = Self.exactOption(for: field, heard: heard) {
                return MapperFieldResult(
                    key: field.key, name: field.name, value: exact, unit: "",
                    score: 1, status: .autoFilled, topTwo: [(label: exact, score: 1)],
                    note: "一致")
            }
            guard let scores = try? classifier.scores(
                text: heard, task: field.name, labels: options) else { return nil }
            let ranked = GLiNERMapper.rank(labels: options, scores: scores)
            guard let best = ranked.first else { return nil }
            return MapperFieldResult(
                key: field.key, name: field.name, value: best.label, unit: "",
                score: best.score,
                status: best.score >= MapperForm.autoFillThreshold ? .autoFilled : .review,
                topTwo: Array(ranked.prefix(2)), note: "")
        case .number(let unit):
            let normalised = Self.normaliseSpokenUnits(heard)
            let candidates = MapperForm.numberCandidates(in: normalised)
            guard !candidates.isEmpty else { return nil }
            // Spoken corrections put the real answer last: "0.45、違った、0.54".
            let corrected = Self.correctionWords.contains { normalised.contains($0) }
            guard let candidate = corrected ? candidates.last : candidates.first else { return nil }
            // A unit that is not this field's own usually means the number belongs
            // somewhere else ("72度" answered to 吐出圧), so do not auto-fill it.
            let mismatched = !candidate.unit.isEmpty && !Self.sameUnit(candidate.unit, unit)
            return MapperFieldResult(
                key: field.key, name: field.name, value: candidate.value,
                unit: candidate.unit.isEmpty ? unit : candidate.unit,
                score: 1, status: mismatched ? .review : .autoFilled, topTwo: [],
                note: mismatched ? "単位が違うかもしれません" : (corrected ? "訂正を採用" : ""))
        }
    }

    static let correctionWords = ["違った", "じゃなくて", "訂正", "間違えた", "ではなく"]

    /// Spoken forms that plainly mean one option.
    ///
    /// Kept here rather than in `MapperForm` so the batch mode's labels are untouched.
    /// The point is not to be clever: when someone answers with the option itself, or
    /// an everyday equivalent, the classifier should not get a vote — on the device
    /// log it took "なし" and "未点検" and landed somewhere else, and the person then
    /// had to refuse the confirmation.
    static let optionAliases: [String: [String]] = [
        "正常": ["異常なし", "問題なし", "問題ない", "問題ありません", "大丈夫", "異常ありません"],
        "異常": ["異常あり", "おかしい", "変な音", "異常です"],
        "未点検": ["点検していない", "まだ", "見ていない", "未確認"],
        "あり": ["ある", "漏れている", "漏れてる"],
        "給油": ["グリスアップ", "オイル補給"],
        "交換": ["取り替え"],
        "清掃": ["掃除"],
    ]

    /// "なし" means different things to different fields, so it is resolved per field.
    static func aliases(for option: String, field: MapperForm.Field) -> [String] {
        if option == "なし" {
            switch field.key {
            case "oil_leak": return ["ない", "漏れていない", "ありません"]
            case "next_action": return ["ない", "不要", "特になし", "ありません"]
            default: return []
            }
        }
        return optionAliases[option] ?? []
    }

    /// The option the answer names outright, or nil to let the classifier decide.
    static func exactOption(for field: MapperForm.Field, heard: String) -> String? {
        guard case .list(let options) = field.kind else { return nil }
        let strippable = CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters)
        let trimmed = heard.trimmingCharacters(in: strippable)
        guard !trimmed.isEmpty else { return nil }
        for option in options {
            if trimmed == option { return option }
            if aliases(for: option, field: field).contains(trimmed) { return option }
        }
        return nil
    }

    /// Words worth priming the recogniser with for this field.
    static func hints(for field: MapperForm.Field) -> [String] {
        switch field.kind {
        case .list(let options):
            return options + options.flatMap { aliases(for: $0, field: field) }
        case .number(let unit):
            return [spoken(unit), unit]
        }
    }

    /// Dictation writes units as words. Kana numerals ("れいてんよんご") stay out of
    /// scope, as in the batch path — only the unit is normalised, never the digits.
    static func normaliseSpokenUnits(_ text: String) -> String {
        var out = text
        for (spoken, symbol) in [
            ("メガパスカル", "MPa"), ("メガパスカール", "MPa"), ("キロパスカル", "kPa"),
            ("パーセント", "%"), ("ミリメートル", "mm"), ("ミリメーター", "mm"),
            ("摂氏", ""), ("度シー", "℃"), ("度C", "℃"),
            // Last: the longer spellings above must match before this one.
            ("メガスカ", "MPa"), ("メガ", "MPa"),
        ] {
            out = out.replacingOccurrences(of: spoken, with: symbol)
        }
        return out
    }

    private func confirm(_ result: MapperFieldResult) async -> Bool {
        // Spoken form spells the unit out; the panel keeps the symbol.
        let spokenValue = "\(result.value) \(Self.spoken(result.unit))"
        for attempt in 0..<2 {
            if cancelled { return false }
            // The second pass repeats the value rather than only asking for a yes or
            // no — after a silence the person no longer has it in mind.
            await say(attempt == 0
                ? "\(spokenValue) で合っていますか？"
                : "\(spokenValue) で合っていますか？ はい か いいえ でお答えください。")
            if cancelled { return false }
            stage = .confirming
            confirmPrompt = "\(result.value)\(result.unit) で合っていますか？"
            let answer = await listen(hints: ["はい", "いいえ"])
            confirmPrompt = ""
            if let yes = Self.yesNo(answer) { return yes }
            // No classifier fallback here on purpose. It was being handed empty and
            // ambiguous answers and turning them into a "yes", which silently accepts
            // a wrong value — the one outcome an inspection sheet must not produce.
            note("    confirm: 判定できず(「\(answer)」)")
        }
        // Two tries without a clear answer: treat as not confirmed.
        note("    confirm: 未確認のため「いいえ」として扱います")
        return false
    }

    /// Reads a spoken yes or no, or nil when it is neither.
    ///
    /// Order matters more than the word lists. "そうではない" contains "そう", so a
    /// plain positive-first scan read a refusal as agreement. Negatives are therefore
    /// checked before positives — except for the set phrases that contain a negative
    /// word and still mean yes ("問題ありません", "問題ない", "大丈夫"), which are
    /// matched first of all.
    static func yesNo(_ text: String) -> Bool? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let lowered = trimmed.lowercased()

        // (1) Positive phrases that contain a negative word.
        let positivePhrases = ["問題ありません", "問題ない", "大丈夫"]
        if positivePhrases.contains(where: lowered.contains) { return true }

        // (2) Refusals.
        let negatives = [
            "いいえ", "いや", "違います", "ちがいます", "違う", "ちがう",
            "ではない", "じゃない", "やり直し", "もう一度", "だめ", "ダメ",
        ]
        if negatives.contains(where: lowered.contains) { return false }

        // (3) Agreement.
        let positives = [
            "はい", "はーい", "うん", "ええ", "オッケー", "おっけー", "オーケー",
            "ok", "ｏｋ", "合ってます", "あってます", "合ってる", "あってる",
            "正しい", "それで", "いいです", "そう",
        ]
        if positives.contains(where: lowered.contains) { return true }

        return nil
    }

    private func readBackSheet() async {
        let blanks = results.filter { $0.status == .absent }.count
        await say(blanks > 0
            ? "記入が終わりました。未入力 \(blanks) 件です。"
            : "記入が終わりました。")
    }

    private func say(_ text: String) async {
        guard !textOnlyMode else { return }
        note("    tts start: 「\(text)」")
        await tts.speak(text)
        if tts.lastSpeakTimedOut {
            note("    tts TIMED OUT — no didFinish; speech is not playing")
            ttsFailures += 1
            // Two silent utterances mean the voice path is not working at all.
            if ttsFailures >= 2, !textOnlyMode {
                textOnlyMode = true
                speech.usesPlayAndRecord = false
                voiceNote = "読み上げが始まらないため、タップ回答に切り替えました"
                note("    falling back to text mode")
            }
        } else {
            note("    tts done")
        }
    }

    /// One answer: starts the recogniser and waits for the silence rule or the cap.
    ///
    /// The pause and the session bounce are not superstition: starting AVAudioEngine
    /// while speech playback is still giving up the route makes it raise an
    /// Objective-C exception, which Swift cannot catch and which takes the process
    /// with it. Letting the route settle first is what stopped it.
    private func listen(hints: [String] = []) async -> String {
        // Text mode never opens the microphone: the answer arrives from a tap. The
        // stage still has to move, because that is what puts the buttons on screen.
        if textOnlyMode {
            stage = .listening
            return await withCheckedContinuation { continuation in
                answerContinuation = continuation
            }
        }
        // Let speech playback give up the route before recording starts. The session
        // itself is left alone — re-setting the category here would drop the headset.
        try? await Task.sleep(for: .milliseconds(300))
        // Only now is the microphone about to open, so only now does the panel claim
        // to be listening.
        stage = .listening
        partial = ""
        return await withCheckedContinuation { (continuation: CheckedContinuation<String, Never>) in
            var resumed = false
            speech.start(hints: hints) { text in
                guard !resumed else { return }
                resumed = true
                continuation.resume(returning: text)
            }
            // `start` bails out silently when permission or on-device support is
            // missing; without this the dialog would hang forever.
            if !speech.isRecording, !resumed {
                resumed = true
                continuation.resume(returning: "")
            }
        }
    }
}
#endif
