import CoreML
import Foundation

enum CompiledModelStore {

    static func compiledModelURL(bundleURL: URL, name: String) async throws -> URL {
        let fileManager = FileManager.default
        let compiledName = name.replacingOccurrences(of: ".mlpackage", with: ".mlmodelc")

        let sibling = bundleURL.appending(path: compiledName)
        if fileManager.fileExists(atPath: sibling.path(percentEncoded: false)) { return sibling }

        let packageURL = bundleURL.appending(path: name)

        #if os(macOS)

        return try await compileAndCache(package: packageURL, destination: sibling)
        #else

        let cached = try applicationSupportCompiledURL(bundleURL: bundleURL, compiledName: compiledName)
        if fileManager.fileExists(atPath: cached.path(percentEncoded: false)) { return cached }
        try fileManager.createDirectory(
            at: cached.deletingLastPathComponent(), withIntermediateDirectories: true)
        return try await compileAndCache(package: packageURL, destination: cached)
        #endif
    }

    private static func compileAndCache(package: URL, destination: URL) async throws -> URL {
        let temporary = try await MLModel.compileModel(at: package)
        do {
            try FileManager.default.moveItem(at: temporary, to: destination)
        } catch let error as CocoaError where error.code == .fileWriteFileExists {

            try? FileManager.default.removeItem(at: temporary)
        }
        return destination
    }

    #if !os(macOS)

    private static func applicationSupportCompiledURL(bundleURL: URL, compiledName: String) throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        return base
            .appending(path: "CoreLLMCompiledModels", directoryHint: .isDirectory)
            .appending(path: stableKey(for: bundleURL), directoryHint: .isDirectory)
            .appending(path: compiledName)
    }

    private static func stableKey(for url: URL) -> String {
        let path = url.standardizedFileURL.path(percentEncoded: false)
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in path.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return "\(url.lastPathComponent)-\(String(hash, radix: 16))"
    }
    #endif
}
