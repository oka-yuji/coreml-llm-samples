import AVFoundation
import Foundation
import Observation
import Speech

/// Running RMS of the microphone, reported about twice a second.
///
/// It answers the question the recogniser cannot: when nothing is recognised, is
/// the microphone delivering silence, or audio the recogniser cannot parse? Around
/// -80 dB means nothing is arriving at all.
///
/// `@unchecked Sendable`: AVAudioEngine calls the tap from one render thread, and
/// these fields are touched nowhere else.
final class AudioLevelProbe: @unchecked Sendable {
    private var lastReport = Date.distantPast
    private var peak: Float = 0

    private var quietRun = 0

    /// Counts consecutive quiet reports; returns the running count.
    func noteQuiet(_ isQuiet: Bool) -> Int {
        quietRun = isQuiet ? quietRun + 1 : 0
        return quietRun
    }

    /// Returns a dBFS value when it is time to report, otherwise nil.
    func add(_ buffer: AVAudioPCMBuffer) -> Float? {
        guard let channel = buffer.floatChannelData?[0] else { return nil }
        var sum: Float = 0
        let count = Int(buffer.frameLength)
        for index in 0..<count { sum += channel[index] * channel[index] }
        let rms = count > 0 ? (sum / Float(count)).squareRoot() : 0
        peak = max(peak, rms)
        guard Date().timeIntervalSince(lastReport) >= 0.5 else { return nil }
        lastReport = Date()
        let value = peak
        peak = 0
        return value > 0 ? 20 * log10(value) : -120
    }
}

/// On-device Japanese dictation. Never falls back to the cloud: if the device has no
/// on-device model for ja-JP the button is disabled and the screen says so, because
/// the whole point of the demo is that nothing leaves the phone.
@MainActor
@Observable
final class MapperSpeech {
    var isRecording = false
    var isAvailable = false
    var statusText = "音声入力: 未確認"
    var partialText = ""

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
    private var audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var onFinal: ((String) -> Void)?
    private var silenceTimer: Timer?
    private var capTimer: Timer?

    /// Dialog mode: stop once the transcript has not changed for this long, so the
    /// person does not have to press anything. nil keeps the manual stop button.
    var silenceTimeout: TimeInterval?
    /// Hard ceiling on one answer, whatever happens.
    var maxDuration: TimeInterval = 6
    /// Dialog mode also plays speech, so the session has to allow playback.
    var usesPlayAndRecord = false

    /// e.g. "マイク=許可, 音声認識=未" — shown in the dialog panel.
    var permissionSummary: String {
        let speechWord: String = switch SFSpeechRecognizer.authorizationStatus() {
        case .authorized: "許可"
        case .notDetermined: "未"
        default: "拒否"
        }
        #if os(iOS)
        let micWord: String = switch AVAudioApplication.shared.recordPermission {
        case .granted: "許可"
        case .undetermined: "未"
        default: "拒否"
        }
        #else
        let micWord = "—"
        #endif
        return "マイク=\(micWord), 音声認識=\(speechWord)"
    }

    var supportsOnDevice: Bool { recognizer?.supportsOnDeviceRecognition ?? false }

    /// What the audio is actually flowing through — the detail that explains a
    /// 24 kHz input or a silent microphone.
    func routeDescription() -> String {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        let inputs = session.currentRoute.inputs
            .map { "\($0.portType.rawValue)/\($0.portName)" }.joined(separator: ",")
        let outputs = session.currentRoute.outputs
            .map { "\($0.portType.rawValue)/\($0.portName)" }.joined(separator: ",")
        return "in=[\(inputs.isEmpty ? "none" : inputs)] out=[\(outputs.isEmpty ? "none" : outputs)] "
            + "\(Int(session.sampleRate))Hz"
        #else
        return "n/a"
        #endif
    }

    #if os(iOS)
    static let bluetoothPorts: Set<AVAudioSession.Port> = [.bluetoothHFP, .bluetoothLE, .bluetoothA2DP]

    /// Chooses the route for the dialog and waits until it is actually in effect.
    ///
    /// With a headset paired we keep **both** ends on it — adding `.defaultToSpeaker`
    /// would send playback to the handset while recording stayed on the headset, which
    /// is the worst of both. With no headset we pin the built-in mic and speaker.
    ///
    /// The wait matters: switching into HFP takes a moment, and recording started
    /// mid-switch captures silence — which is what "No speech detected" looked like.
    @discardableResult
    func configureVoiceSession() async -> String {
        let session = AVAudioSession.sharedInstance()
        do {
            // A record-capable category first, so availableInputs lists the headset.
            try session.setCategory(
                .playAndRecord, mode: .spokenAudio, options: [.allowBluetoothHFP, .duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)

            let bluetooth = session.availableInputs?.first { Self.bluetoothPorts.contains($0.portType) }
            if let bluetooth {
                try session.setCategory(
                    .playAndRecord, mode: .spokenAudio, options: [.allowBluetoothHFP, .duckOthers])
                try? session.setPreferredInput(bluetooth)
                onEvent?("    route: Bluetooth を使います(\(bluetooth.portName))")
            } else {
                try session.setCategory(
                    .playAndRecord, mode: .spokenAudio, options: [.defaultToSpeaker, .duckOthers])
                if let builtIn = session.availableInputs?.first(where: { $0.portType == .builtInMic }) {
                    try? session.setPreferredInput(builtIn)
                }
                onEvent?("    route: 本体マイク／スピーカーを使います")
            }
            try session.setActive(true, options: .notifyOthersOnDeactivation)

            // Wait for the chosen port to actually become the current input.
            if let wanted = bluetooth?.portType {
                let deadline = Date().addingTimeInterval(1.5)
                while Date() < deadline,
                    session.currentRoute.inputs.first?.portType != wanted {
                    try? await Task.sleep(for: .milliseconds(100))
                }
            }
            lastSessionError = nil
        } catch {
            lastSessionError = error.localizedDescription
            onEvent?("    route: セッション設定に失敗 \(error.localizedDescription)")
        }
        logRoute("start")
        return routeDescription()
    }
    #endif

    private(set) var lastSessionError: String?
    /// Why the last listen produced nothing, e.g. "No speech detected".
    private(set) var lastRecognitionError: String?
    /// True when the recogniser heard nothing at all, as opposed to failing.
    var heardNothing: Bool { (lastRecognitionError ?? "").contains("No speech detected") }

    func logRoute(_ label: String) {
        let description = routeDescription()
        print("[speech] route(\(label)): \(description)")
        onEvent?("    route(\(label)): \(description)")
    }

    /// Trace hook so the dialog can show what the recogniser is doing.
    var onEvent: ((String) -> Void)?

    func refreshAvailability() {
        guard let recognizer else {
            statusText = "音声入力: ja-JP の認識器なし"
            isAvailable = false
            return
        }
        guard recognizer.supportsOnDeviceRecognition else {
            statusText = "音声入力: 端末内認識なし(クラウドへは送りません)"
            isAvailable = false
            return
        }
        switch SFSpeechRecognizer.authorizationStatus() {
        case .authorized:
            isAvailable = true
            statusText = "音声入力: 端末内(ja-JP)で利用可"
        case .notDetermined:
            // Ask on the first tap rather than on screen open.
            isAvailable = true
            statusText = "音声入力: 端末内(ja-JP)。最初のタップで許可を求めます"
        default:
            isAvailable = false
            statusText = "音声入力: 許可されていません(設定で許可してください)"
        }
    }

    func requestPermission() {
        SFSpeechRecognizer.requestAuthorization { [weak self] _ in
            Task { @MainActor in self?.refreshAvailability() }
        }
        #if os(iOS)
        AVAudioApplication.requestRecordPermission { _ in }
        #endif
    }

    /// Ask and wait. The dialog needs the grant before its first question, otherwise
    /// every answer is skipped while the prompt is still on screen.
    @discardableResult
    func requestPermissionAndWait() async -> Bool {
        if SFSpeechRecognizer.authorizationStatus() == .notDetermined {
            _ = await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
                SFSpeechRecognizer.requestAuthorization { status in
                    continuation.resume(returning: status == .authorized)
                }
            }
        }
        #if os(iOS)
        if AVAudioApplication.shared.recordPermission == .undetermined {
            _ = await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
                AVAudioApplication.requestRecordPermission { continuation.resume(returning: $0) }
            }
        }
        #endif
        refreshAvailability()
        #if os(iOS)
        // The microphone matters as much as the recogniser; returning true on the
        // speech grant alone let the dialog start with no way to hear anything.
        let micOK = AVAudioApplication.shared.recordPermission == .granted
        #else
        let micOK = true
        #endif
        return SFSpeechRecognizer.authorizationStatus() == .authorized && micOK
    }

    func start(onFinal: @escaping (String) -> Void) {
        guard !isRecording, let recognizer, recognizer.supportsOnDeviceRecognition else { return }
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            // First tap: ask. The next tap starts recording.
            requestPermission()
            return
        }
        print("[speech] guards ok, authorized; available=\(recognizer.isAvailable) "
              + "onDevice=\(recognizer.supportsOnDeviceRecognition)")
        self.onFinal = onFinal
        partialText = ""
        lastRecognitionError = nil

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        // The contract of this demo: on-device only.
        request.requiresOnDeviceRecognition = true
        self.request = request
        print("[speech] request created")

        do {
            #if os(iOS)
            let session = AVAudioSession.sharedInstance()
            // In dialog mode the category is set once, by the dialog, before any
            // speech plays. Re-setting it here is what made the input node hand back
            // a stale format and trap inside installTap.
            if !usesPlayAndRecord {
                try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            }
            // The dialog already chose and settled the route in configureVoiceSession();
            // re-setting the category here would undo the Bluetooth preference.
            if !usesPlayAndRecord {
                try session.setActive(true, options: .notifyOthersOnDeactivation)
            }
            logRoute("listen")
            print("[speech] session active")
            #endif

            // A fresh engine each turn, so no format is carried over from the last one.
            audioEngine = AVAudioEngine()
            let input = audioEngine.inputNode
            // `outputFormat` is the node's own tap format (Apple's SpeakToMe sample
            // uses it); `inputFormat` can disagree with the hardware right after
            // speech playback, and AVAudioEngine raises an Objective-C exception for
            // that — which Swift cannot catch, so it kills the process.
            print("[speech] engine created")
            let format = input.outputFormat(forBus: 0)
            print("[speech] format rate=\(format.sampleRate) ch=\(format.channelCount)")
            guard format.sampleRate > 0, format.channelCount > 0 else {
                statusText = "音声入力: 入力フォーマットが無効です(サンプルレート \(format.sampleRate))"
                onEvent?("    stt: invalid input format")
                self.request = nil
                return
            }
            // @Sendable makes this nonisolated. Without it the closure is created in a
            // @MainActor context and inferred main-actor-isolated, and the Swift 6
            // runtime traps the moment the audio render thread calls it.
            nonisolated(unsafe) let sink = request
            let probe = AudioLevelProbe()
            nonisolated(unsafe) let report = self.onEvent
            let route = routeDescription()
            input.installTap(onBus: 0, bufferSize: 1024, format: format) { @Sendable buffer, _ in
                sink.append(buffer)
                guard let decibels = probe.add(buffer) else { return }
                print(String(format: "[speech] level: %.0f dB", decibels))
                // Four quiet readings in a row is about two seconds of nothing.
                let quiet = probe.noteQuiet(decibels < -70)
                let line = quiet == 4
                    ? "    mic: 音が届いていません(\(route))"
                    : String(format: "    mic: %.0f dB", decibels)
                Task { @MainActor in report?(line) }
            }
            print("[speech] tap installed")
            audioEngine.prepare()
            try audioEngine.start()
            print("[speech] engine started")
        } catch {
            statusText = "音声入力: 開始失敗 \(error.localizedDescription)"
            onEvent?("    stt start failed: \(error.localizedDescription)")
            self.request = nil
            return
        }

        print("[speech] creating recognition task (onDevice=\(request.requiresOnDeviceRecognition) "
              + "available=\(recognizer.isAvailable) onDeviceSupported=\(recognizer.supportsOnDeviceRecognition))")
        // Speech calls this from its own queue. `@Sendable` marks it nonisolated;
        // without it Swift 6 infers main-actor isolation from the enclosing context
        // and the runtime traps on entry — before a single line of the body runs,
        // which is exactly what the device trace showed.
        task = recognizer.recognitionTask(with: request) { @Sendable [weak self] result, error in
            // Flatten to plain values here, on whatever thread Speech used, so nothing
            // non-Sendable crosses into the main actor.
            let text = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            let failure = error?.localizedDescription
            print("[speech] callback text=\(text ?? "nil") final=\(isFinal) error=\(failure ?? "none")")
            Task { @MainActor in
                guard let self else { return }
                if let text, text != self.partialText {
                    self.partialText = text
                    self.onEvent?("    stt partial: 「\(text)」")
                    self.armSilenceTimer()
                }
                if let failure {
                    self.lastRecognitionError = failure
                    self.onEvent?("    stt error: \(failure)")
                }
                // Only the finalised transcript goes into the text field.
                if isFinal {
                    self.finish(with: self.partialText)
                } else if failure != nil, self.isRecording {
                    self.finish(with: self.partialText)
                }
            }
        }
        print("[speech] task created")
        onEvent?("    stt: listening")
        isRecording = true
        statusText = "音声入力: 録音中(端末内)"
        // Deliberately not armed yet: the 1.2 s silence rule counts from the first
        // partial result. Before anything is recognised only the hard cap applies,
        // otherwise a slow start would be cut off after 1.2 s of thinking.
        capTimer?.invalidate()
        capTimer = Timer.scheduledTimer(withTimeInterval: maxDuration, repeats: false) { [weak self] _ in
            Task { @MainActor in self?.stop() }
        }
    }

    /// Restarted on every transcript change; firing means the person stopped talking.
    private func armSilenceTimer() {
        guard let silenceTimeout else { return }
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceTimeout, repeats: false) {
            [weak self] _ in Task { @MainActor in self?.stop() }
        }
    }

    func stop() {
        guard isRecording else { return }
        silenceTimer?.invalidate()
        capTimer?.invalidate()
        request?.endAudio()
        // The final callback delivers the text; stop the audio path now.
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        statusText = "音声入力: 認識中…"
    }

    private func finish(with text: String) {
        guard isRecording else { return }
        isRecording = false
        silenceTimer?.invalidate()
        capTimer?.invalidate()
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        task?.cancel()
        task = nil
        request = nil
        #if os(iOS)
        // In dialog mode the session stays active: speech playback follows straight
        // after, and tearing the session down between turns makes it stutter.
        if !usesPlayAndRecord { try? AVAudioSession.sharedInstance().setActive(false) }
        #endif
        // Hand the callback over exactly once: a late finish must not drive the
        // next question's turn.
        let deliver = onFinal
        onFinal = nil
        deliver?(text)
        statusText = text.isEmpty
            ? "音声入力: 認識できませんでした"
            : "音声入力: 認識完了(編集してからマッピングできます)"
    }
}
