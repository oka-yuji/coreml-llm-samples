import CoreML
import Foundation

let memory = UnsafeMutableRawPointer.allocate(byteCount: 128, alignment: 16)
defer { memory.deallocate() }
memory.initializeMemory(as: UInt8.self, repeating: 0xA5, count: 128)
let array = try MLMultiArray(dataPointer: memory, shape: [32], dataType: .int8, strides: [1], deallocator: nil)
memset(array.dataPointer, 0, array.count * 4)
let oldCorrupted = (32..<128).filter { memory.load(fromByteOffset: $0, as: UInt8.self) != 0xA5 }.count
memory.initializeMemory(as: UInt8.self, repeating: 0xA5, count: 128)
array.withUnsafeMutableBytes { raw, _ in
    _ = raw.initializeMemory(as: UInt8.self, repeating: 0)
}
let newCorrupted = (32..<128).filter { memory.load(fromByteOffset: $0, as: UInt8.self) != 0xA5 }.count
let allZero = (0..<32).allSatisfy { memory.load(fromByteOffset: $0, as: UInt8.self) == 0 }
print("old extra bytes overwritten: \(oldCorrupted)")
print("safe fill extra bytes overwritten: \(newCorrupted), logical input all zero: \(allZero)")
precondition(oldCorrupted == 96 && newCorrupted == 0 && allZero)
