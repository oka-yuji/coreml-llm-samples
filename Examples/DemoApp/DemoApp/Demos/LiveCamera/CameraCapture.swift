@preconcurrency import AVFoundation
import CoreGraphics
import CoreImage
import CoreMLBackend
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum CameraError: Error, CustomStringConvertible {
    case permissionDenied
    case noCamera
    case cannotConfigure(String)
    case noFrame
    case cannotEncodeFrame
    case cannotDecodeFile(String)

    var description: String {
        switch self {
        case .permissionDenied:
            return "Camera access is not granted. Enable it in Settings, under Privacy > Camera."
        case .noCamera:
            return "No camera is available on this device."
        case .cannotConfigure(let why):
            return "Cannot configure the capture session: \(why)"
        case .noFrame:
            return "No camera frame has arrived yet."
        case .cannotEncodeFrame:
            return "Cannot encode the camera frame as JPEG."
        case .cannotDecodeFile(let name):
            return "Cannot decode the image file \(name)."
        }
    }
}

struct LiveFrame: Sendable {
    let image: LiveFrameImage
    let label: String
}

protocol LiveFrameSource: AnyObject, Sendable {
    func nextFrame() throws -> LiveFrame
}

final class CameraCapture: NSObject, LiveFrameSource, @unchecked Sendable {
    let session = AVCaptureSession()

    private let output = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "live-camera.session")
    private let frameQueue = DispatchQueue(label: "live-camera.frames")
    private let lock = NSLock()
    private let ciContext = CIContext()
    private var latest: CVPixelBuffer?
    private var configured = false

    static func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        default:
            return false
        }
    }

    func configure() throws {
        guard !configured else { return }
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        if session.canSetSessionPreset(.hd1280x720) {
            session.sessionPreset = .hd1280x720
        }

        #if os(iOS)
        let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        #else
        let device = AVCaptureDevice.default(for: .video)
        #endif
        guard let device else { throw CameraError.noCamera }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw CameraError.cannotConfigure("the camera input was rejected")
        }
        session.addInput(input)

        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: frameQueue)
        guard session.canAddOutput(output) else {
            throw CameraError.cannotConfigure("the video output was rejected")
        }
        session.addOutput(output)

        if let connection = output.connection(with: .video),
           connection.isVideoRotationAngleSupported(portraitRotationAngle) {
            connection.videoRotationAngle = portraitRotationAngle
        }
        Self.limitFrameRate(of: device, to: Self.captureFramesPerSecond)
        configured = true
    }

    static let captureFramesPerSecond: Int32 = 30

    private static func limitFrameRate(of device: AVCaptureDevice, to fps: Int32) {
        let wanted = CMTime(value: 1, timescale: fps)
        let supported = device.activeFormat.videoSupportedFrameRateRanges.contains {
            CMTimeCompare(wanted, $0.minFrameDuration) >= 0
                && CMTimeCompare(wanted, $0.maxFrameDuration) <= 0
        }
        guard supported, (try? device.lockForConfiguration()) != nil else { return }
        device.activeVideoMinFrameDuration = wanted
        device.unlockForConfiguration()
    }

    private var portraitRotationAngle: CGFloat { 90 }

    func start() {
        sessionQueue.async { [session] in
            guard !session.isRunning else { return }
            session.startRunning()
        }
    }

    func stop() {
        sessionQueue.async { [session] in
            guard session.isRunning else { return }
            session.stopRunning()
        }
        lock.lock()
        latest = nil
        lock.unlock()
    }

    var hasFrame: Bool {
        lock.lock()
        defer { lock.unlock() }
        return latest != nil
    }

    func nextFrame() throws -> LiveFrame {
        let cgImage = try latestCGImage()
        return LiveFrame(
            image: LiveFrameImage(cgImage), label: "camera \(cgImage.width)x\(cgImage.height)")
    }

    func stageFrame(to url: URL) throws -> String {
        let cgImage = try latestCGImage()
        try Self.writeJPEG(cgImage, to: url)
        return "camera \(cgImage.width)x\(cgImage.height)"
    }

    private func latestCGImage() throws -> CGImage {
        lock.lock()
        let buffer = latest
        lock.unlock()
        guard let buffer else { throw CameraError.noFrame }
        let image = CIImage(cvPixelBuffer: buffer)
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            throw CameraError.cannotEncodeFrame
        }
        return cgImage
    }

    static func writeJPEG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw CameraError.cannotEncodeFrame
        }
        CGImageDestinationAddImage(
            destination, image,
            [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw CameraError.cannotEncodeFrame
        }
    }
}

extension CameraCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lock.lock()
        latest = buffer
        lock.unlock()
    }
}

final class FileSequenceFrameSource: LiveFrameSource, @unchecked Sendable {
    private let urls: [URL]
    private let lock = NSLock()
    private var index = 0

    init(urls: [URL]) { self.urls = urls }

    func nextFrame() throws -> LiveFrame {
        lock.lock()
        let current = index
        index += 1
        lock.unlock()
        guard !urls.isEmpty else { throw CameraError.noFrame }
        let source = urls[current % urls.count]
        guard let imageSource = CGImageSourceCreateWithURL(source as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            throw CameraError.cannotDecodeFile(source.lastPathComponent)
        }
        return LiveFrame(image: LiveFrameImage(cgImage), label: source.lastPathComponent)
    }
}
