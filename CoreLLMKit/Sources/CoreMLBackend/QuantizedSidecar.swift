import Foundation
import LLMCore

final class QuantizedSidecar {
    enum Mode {
        case fp16
        case int8(scale: Data, runtimeScale: Float)
    }

    private let data: Data
    let rows: Int
    let cols: Int
    private let mode: Mode

    init(fp16 url: URL, rows: Int, cols: Int) throws {
        self.data = try Data(contentsOf: url, options: [.alwaysMapped])
        let expected = rows * cols * MemoryLayout<Float16>.size
        guard data.count == expected else {
            throw LLMEngineError.incompatibleBundle(
                reason: "\(url.lastPathComponent): fp16 size mismatch (actual \(data.count), expected \(expected))")
        }
        self.rows = rows
        self.cols = cols
        self.mode = .fp16
    }

    init(int8 url: URL, scale scaleURL: URL, rows: Int, cols: Int, runtimeScaleSqrt: Double) throws {
        self.data = try Data(contentsOf: url, options: [.alwaysMapped])
        let expected = rows * cols
        guard data.count == expected else {
            throw LLMEngineError.incompatibleBundle(
                reason: "\(url.lastPathComponent): int8 size mismatch (actual \(data.count), expected \(expected))")
        }
        let scale = try Data(contentsOf: scaleURL, options: [.alwaysMapped])
        guard scale.count == rows * MemoryLayout<Float32>.size else {
            throw LLMEngineError.incompatibleBundle(
                reason: "\(scaleURL.lastPathComponent): scale size mismatch (actual \(scale.count), expected \(rows * 4))")
        }
        self.rows = rows
        self.cols = cols
        self.mode = .int8(scale: scale, runtimeScale: Float(runtimeScaleSqrt.squareRoot()))
    }

    func read(row: Int, offset: Int = 0, count: Int? = nil, into destination: UnsafeMutablePointer<Float16>) {
        let n = count ?? cols
        precondition(row >= 0 && row < rows && offset + n <= cols)
        switch mode {
        case .fp16:
            let byteOffset = (row * cols + offset) * MemoryLayout<Float16>.size
            data.withUnsafeBytes { raw in
                _ = memcpy(destination, raw.baseAddress!.advanced(by: byteOffset), n * MemoryLayout<Float16>.size)
            }
        case .int8(let scale, let runtimeScale):
            let s = scale.withUnsafeBytes { $0.load(fromByteOffset: row * 4, as: Float32.self) }
            let factor = (s / 127.0) * runtimeScale
            let base = row * cols + offset
            data.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: Int8.self).baseAddress!
                for i in 0..<n {
                    destination[i] = Float16(Float(p[base + i]) * factor)
                }
            }
        }
    }
}
