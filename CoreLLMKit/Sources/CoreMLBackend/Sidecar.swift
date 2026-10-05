import Foundation
import LLMCore

final class Sidecar {
    private let data: Data
    let rows: Int
    let cols: Int

    init(url: URL, rows: Int, cols: Int) throws {
        self.data = try Data(contentsOf: url, options: [.alwaysMapped])
        let expected = rows * cols * MemoryLayout<Float16>.size
        guard data.count == expected else {
            throw LLMEngineError.incompatibleBundle(
                reason: "\(url.lastPathComponent): size mismatch (actual \(data.count) bytes, expected \(expected) bytes)"
            )
        }
        self.rows = rows
        self.cols = cols
    }

    func read(row: Int, offset: Int = 0, count: Int? = nil, into destination: UnsafeMutableRawPointer) {
        let n = count ?? cols
        precondition(row >= 0 && row < rows && offset + n <= cols)
        let byteOffset = (row * cols + offset) * MemoryLayout<Float16>.size
        data.withUnsafeBytes { raw in
            _ = memcpy(destination, raw.baseAddress!.advanced(by: byteOffset), n * MemoryLayout<Float16>.size)
        }
    }
}
