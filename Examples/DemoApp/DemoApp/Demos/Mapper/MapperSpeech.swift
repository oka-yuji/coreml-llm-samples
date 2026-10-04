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
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var onFinal: ((String) -> Void)?

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
        AVAudioApplication.requestRecordPermission { _ in }
    }

    func start(onFinal: @escaping (String) -> Void) {
        guard !isRecording, let recognizer, recognizer.supportsOnDeviceRecognition else { return }
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            // First tap: ask. The next tap starts recording.
            requestPermission()
            return
        }
        self.onFinal = onFinal
        partialText = ""

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        // The contract of this demo: on-device only.
        request.requiresOnDeviceRecognition = true
        self.request = request

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            try session.setActive(true, options: .notifyOthersOnDeactivation)

            let input = audioEngine.inputNode
            input.removeTap(onBus: 0)
            input.installTap(onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)) {
                buffer, _ in request.append(buffer)
            }
            audioEngine.prepare()
            try audioEngine.start()
        } catch {
            statusText = "音声入力: 開始失敗 \(error.localizedDescription)"
            return
        }

        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let result {
                    self.partialText = result.bestTranscription.formattedString
                    // Only the finalised transcript goes into the text field.
                    if result.isFinal { self.finish(with: self.partialText) }
                }
                if error != nil, self.isRecording { self.finish(with: self.partialText) }
            }
        }
        isRecording = true
        statusText = "音声入力: 録音中(端末内)"
    }

    func stop() {
        guard isRecording else { return }
        request?.endAudio()
        // The final callback delivers the text; stop the audio path now.
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        statusText = "音声入力: 認識中…"
    }

    private func finish(with text: String) {
        guard isRecording else { return }
        isRecording = false
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        task?.cancel()
        task = nil
        request = nil
        try? AVAudioSession.sharedInstance().setActive(false)
        onFinal?(text)
        statusText = text.isEmpty
            ? "音声入力: 認識できませんでした"
            : "音声入力: 認識完了(編集してからマッピングできます)"
    }
}
