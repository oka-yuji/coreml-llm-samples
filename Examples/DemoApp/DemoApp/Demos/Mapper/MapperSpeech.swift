import AVFoundation
import Foundation
import Observation
import Speech

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
        return SFSpeechRecognizer.authorizationStatus() == .authorized
    }

    func start(onFinal: @escaping (String) -> Void) {
        guard !isRecording, let recognizer, recognizer.supportsOnDeviceRecognition else { return }
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            // First tap: ask. The next tap starts recording.
            requestPermission()
            return
        }
        print("[speech] guards ok, authorized")
        self.onFinal = onFinal
        partialText = ""

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
            try session.setActive(true, options: .notifyOthersOnDeactivation)
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
                self.request = nil
                return
            }
            input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
                request.append(buffer)
            }
            print("[speech] tap installed")
            audioEngine.prepare()
            try audioEngine.start()
            print("[speech] engine started")
        } catch {
            statusText = "音声入力: 開始失敗 \(error.localizedDescription)"
            self.request = nil
            return
        }

        print("[speech] creating recognition task (onDevice=\(request.requiresOnDeviceRecognition))")
        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            // Read everything here, on the callback's own thread, and hop to the main
            // actor with plain values only. SFSpeechRecognitionResult is not Sendable,
            // and carrying it across isolation is what killed the process.
            let text = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            let failed = error != nil
            print("[speech] callback text=\(text ?? "nil") final=\(isFinal) error=\(failed)")
            Task { @MainActor in
                guard let self else { return }
                if let text, text != self.partialText {
                    self.partialText = text
                    self.armSilenceTimer()
                }
                // Only the finalised transcript goes into the text field.
                if isFinal {
                    self.finish(with: self.partialText)
                } else if failed, self.isRecording {
                    self.finish(with: self.partialText)
                }
            }
        }
        print("[speech] task created")
        isRecording = true
        statusText = "音声入力: 録音中(端末内)"
        armSilenceTimer()
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
