import CoreGraphics
import Foundation

public struct LiveFrameImage: @unchecked Sendable {
    public let cgImage: CGImage

    public init(_ cgImage: CGImage) { self.cgImage = cgImage }

    public var width: Int { cgImage.width }
    public var height: Int { cgImage.height }
}

public enum VLMPhase: String, Sendable {
    case encode
    case feed
    case generate
}

public struct LiveVisionEncoderInfo: Sendable {
    public let imageRows: Int
    public let seconds: Double
    public let warmUpSeconds: Double
    public let computeUnits: String
}

public struct LiveEncodedFrame: Sendable {
    let soft: SoftTokenRows
    public let imageRows: Int
    public let encodeSeconds: Double
}

public struct LiveVisionEncodeHandle: Sendable {
    let encoder: VisionEncoder

    public func encode(_ frame: LiveFrameImage) async throws -> LiveEncodedFrame {
        let clock = ContinuousClock()
        let t0 = clock.now
        let soft = try await encoder.encode(image: frame.cgImage, releaseAfter: false)
        return LiveEncodedFrame(
            soft: soft, imageRows: soft.rows,
            encodeSeconds: (clock.now - t0) / .seconds(1))
    }
}

public struct LiveVisionPrefillInfo: Sendable {
    public let promptTokens: Int
    public let prefillWidths: [Int]
    public let seconds: Double
}

public struct LiveVisionPrewarm: Sendable {
    public let encoder: LiveVisionEncoderInfo
    public let prefill: LiveVisionPrefillInfo

    public var imageRows: Int { encoder.imageRows }
    public var promptTokens: Int { prefill.promptTokens }
    public var prefillWidths: [Int] { prefill.prefillWidths }

    public var summary: String {
        String(
            format: "vision encoder %.2fs on %@ (first predict %.2fs), prefill widths %@ %.2fs, "
                + "prompt %d tokens (%d image rows)",
            encoder.seconds, encoder.computeUnits, encoder.warmUpSeconds,
            "\(prefill.prefillWidths)", prefill.seconds,
            prefill.promptTokens, encoder.imageRows)
    }
}
