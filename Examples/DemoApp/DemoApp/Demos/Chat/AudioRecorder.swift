import AVFoundation
import CoreMLBackend
import Foundation

final class AudioRecorder: @unchecked Sendable {

    enum RecorderError: Error, CustomStringConvertible {
        case cannotMakeTargetFormat
        case cannotMakeConverter
        var description: String {
            switch self {
            case .cannotMakeTargetFormat: return "cannot describe 16 kHz mono audio"
            case .cannotMakeConverter: return "the input device cannot be resampled to 16 kHz mono"
            }
        }
    }

    private final class SingleBufferFeed: @unchecked Sendable {
        private var buffer: AVAudioPCMBuffer?

        init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }

        func next(_ status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioPCMBuffer? {
            guard let pending = buffer else {
                status.pointee = .noDataNow
                return nil
            }
            buffer = nil
            status.pointee = .haveData
            return pending
        }
    }

    private let engine = AVAudioEngine()
    private let lock = NSLock()
    private let capacity: Int
    private var collected: [Float] = []
    private var converter: AVAudioConverter?
    private var target: AVAudioFormat?
    private var running = false

    init(maxSeconds: Double) {
        capacity = Int(maxSeconds * Double(AudioFileLoader.sampleRate))
    }

    var sampleCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return collected.count
    }

    var isFull: Bool { sampleCount >= capacity }

    static func requestPermission() async -> Bool {
        #if os(iOS)
        switch AVAudioApplication.shared.recordPermission {
        case .granted: return true
        case .undetermined: return await AVAudioApplication.requestRecordPermission()
        default: return false
        }
        #else
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
        #endif
    }

    static var permissionDeniedMessage: String {
        #if os(iOS)
        return "Microphone access is off. Turn it on in Settings > Privacy & Security > "
            + "Microphone, then try again."
        #else
        return "Microphone access is off. Turn it on in System Settings > "
            + "Privacy & Security > Microphone, then try again."
        #endif
    }

    func start() throws {
        guard !running else { return }
        guard let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(AudioFileLoader.sampleRate),
            channels: 1, interleaved: false) else {
            throw RecorderError.cannotMakeTargetFormat
        }
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .default)
        try session.setActive(true)
        #endif
        let input = engine.inputNode
        let source = input.outputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: source, to: target) else {
            Self.deactivateSession()
            throw RecorderError.cannotMakeConverter
        }
        self.target = target
        self.converter = converter
        lock.lock()
        collected.removeAll(keepingCapacity: true)
        collected.reserveCapacity(capacity)
        lock.unlock()

        input.installTap(onBus: 0, bufferSize: 4096, format: source) { [weak self] buffer, _ in
            self?.append(buffer)
        }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            self.converter = nil
            Self.deactivateSession()
            throw error
        }
        running = true
    }

    private static func deactivateSession() {
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        #endif
    }

    @discardableResult
    func stop() -> [Float] {
        if running {
            engine.stop()
            engine.inputNode.removeTap(onBus: 0)
            running = false
            Self.deactivateSession()
        }
        converter = nil
        target = nil
        lock.lock()
        let out = collected
        collected.removeAll(keepingCapacity: false)
        lock.unlock()
        return out
    }

    private func append(_ buffer: AVAudioPCMBuffer) {
        guard let converter, let target, buffer.frameLength > 0 else { return }
        let ratio = target.sampleRate / buffer.format.sampleRate
        let frames = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: frames) else { return }
        let feed = SingleBufferFeed(buffer)
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in feed.next(status) }
        guard error == nil, let channel = out.floatChannelData, out.frameLength > 0 else { return }
        let produced = Int(out.frameLength)
        lock.lock()
        let room = max(0, capacity - collected.count)
        if room > 0 {
            collected.append(contentsOf: UnsafeBufferPointer(
                start: channel[0], count: min(room, produced)))
        }
        lock.unlock()
    }
}
