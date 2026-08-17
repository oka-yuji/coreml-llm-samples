import CoreGraphics
import CoreML
import Foundation
import ImageIO

enum VisionPreprocess {
    static let side = 768
    static let patch = 16
    static let grid = side / patch
    static let numPatches = grid * grid
    static let patchDim = patch * patch * 3

    enum PreprocessError: Error, CustomStringConvertible {
        case cannotDecodeImage(String)
        case cannotMakeContext
        case badPixelCount(expected: Int, got: Int)
        var description: String {
            switch self {
            case .cannotDecodeImage(let p): return "cannot decode image: \(p)"
            case .cannotMakeContext: return "cannot create the 768x768 RGBA context"
            case .badPixelCount(let e, let g): return "pixel count mismatch (expected \(e), actual \(g))"
            }
        }
    }

    static func patchify(pixelsHWC pixels: [Float]) throws -> MLMultiArray {
        let expected = side * side * 3
        guard pixels.count == expected else {
            throw PreprocessError.badPixelCount(expected: expected, got: pixels.count)
        }
        let out = try MLMultiArray(
            shape: [1, NSNumber(value: numPatches), NSNumber(value: patchDim)], dataType: .float16)
        out.withF16 { buf in
            let dst = buf.baseAddress!
            pixels.withUnsafeBufferPointer { src in
                for h in 0..<side {
                    let ph = h / patch, ih = h % patch
                    let rowBase = ph * grid
                    let colBaseH = ih * (patch * 3)
                    let srcRow = h * side * 3
                    for w in 0..<side {
                        let pw = w / patch, iw = w % patch
                        let row = rowBase + pw
                        let col = colBaseH + iw * 3
                        let d = dst + row * patchDim + col
                        let s = srcRow + w * 3
                        d[0] = Float16(src[s]); d[1] = Float16(src[s + 1]); d[2] = Float16(src[s + 2])
                    }
                }
            }
        }
        return out
    }

    static func patches(from image: CGImage) throws -> MLMultiArray {
        let pixels = try unitRGBPixelsHWC(from: image)
        return try patchify(pixelsHWC: pixels)
    }

    static func patches(fromImageAt url: URL) throws -> MLMultiArray {
        try patches(from: try loadCGImage(from: url))
    }

    static func blankPatches() throws -> MLMultiArray {
        let out = try MLMultiArray(
            shape: [1, NSNumber(value: numPatches), NSNumber(value: patchDim)], dataType: .float16)
        out.withF16 { $0.update(repeating: 0) }
        return out
    }

    static func loadCGImage(from url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw PreprocessError.cannotDecodeImage(url.path(percentEncoded: false))
        }
        return image
    }

    static func unitRGBPixelsHWC(from image: CGImage) throws -> [Float] {
        let n = side * side
        let bytesPerRow = side * 4
        let raw = UnsafeMutableRawPointer.allocate(byteCount: n * 4, alignment: 16)
        defer { raw.deallocate() }
        raw.initializeMemory(as: UInt8.self, repeating: 0, count: n * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(
            data: raw, width: side, height: side, bitsPerComponent: 8,
            bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo) else {
            throw PreprocessError.cannotMakeContext
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))

        var pixels = [Float](repeating: 0, count: n * 3)
        let src = raw.bindMemory(to: UInt8.self, capacity: n * 4)
        let inv: Float = 1.0 / 255.0
        for i in 0..<n {
            let s = i * 4
            let d = i * 3
            pixels[d] = Float(src[s]) * inv
            pixels[d + 1] = Float(src[s + 1]) * inv
            pixels[d + 2] = Float(src[s + 2]) * inv
        }
        return pixels
    }
}
