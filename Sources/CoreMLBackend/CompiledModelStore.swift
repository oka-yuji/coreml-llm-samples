import CoreML
import Foundation

/// `.mlpackage` を `.mlmodelc` にコンパイルしてキャッシュする場所を一元管理する。
///
/// 探索・キャッシュ方針(プラットフォーム別):
/// 1. バンドル内に `.mlmodelc`(兄弟)が既にあればそれを返す。**再コンパイルしない**。
///    Mac 出荷バンドルは compile 済み `.mlmodelc` を同梱するため、常にこの高速パスに乗る。
/// 2. 無ければ端末上で `MLModel.compileModel` する。キャッシュ先は OS で分ける:
///    - **macOS**: 従来どおり `.mlpackage` の隣(バンドル内)に置く。**数値・パスとも従来不変**。
///    - **iOS / visionOS**: Application Support にバンドル同一性でキーして置く。iOS の Documents
///      バンドルは `LSSupportsOpeningDocumentsInPlace` で **read-only の場所**を指すことがあり、
///      バンドル内に書けない/書きたくないため。次回起動は App Support の compile 済みを再利用する。
///
/// この分離により Mac 経路のコンパイル・ロード挙動はビットレベルで従来と一致する(iOS 分岐は加算のみ)。
enum CompiledModelStore {
    /// `bundleURL` 内の `name`(= `.mlpackage`)に対応する compile 済み `.mlmodelc` の URL を返す。
    /// 初回のみ端末上コンパイルし、以降はキャッシュを返す。
    static func compiledModelURL(bundleURL: URL, name: String) async throws -> URL {
        let fileManager = FileManager.default
        let compiledName = name.replacingOccurrences(of: ".mlpackage", with: ".mlmodelc")

        // 1) バンドル内に compile 済みがあれば再コンパイルしない(Mac 出荷物・iOS 2 回目以降のバンドル内配置)。
        let sibling = bundleURL.appending(path: compiledName)
        if fileManager.fileExists(atPath: sibling.path()) { return sibling }

        let packageURL = bundleURL.appending(path: name)

        #if os(macOS)
        // 既存 macOS 挙動: バンドル内に `.mlmodelc` をキャッシュ(場所も方法も従来と同一)。
        return try await compileAndCache(package: packageURL, destination: sibling)
        #else
        // iOS / visionOS: Application Support にキャッシュ。バンドルが read-only でも失敗しない。
        let cached = try applicationSupportCompiledURL(bundleURL: bundleURL, compiledName: compiledName)
        if fileManager.fileExists(atPath: cached.path()) { return cached }
        try fileManager.createDirectory(
            at: cached.deletingLastPathComponent(), withIntermediateDirectories: true)
        return try await compileAndCache(package: packageURL, destination: cached)
        #endif
    }

    /// `.mlpackage` をコンパイルし、一時出力を `destination` へ移動する。並行コンパイルの競合は無害に握りつぶす。
    private static func compileAndCache(package: URL, destination: URL) async throws -> URL {
        let temporary = try await MLModel.compileModel(at: package)
        do {
            try FileManager.default.moveItem(at: temporary, to: destination)
        } catch let error as CocoaError where error.code == .fileWriteFileExists {
            // 並行ロードとの競合で先を越された場合は、既存のキャッシュをそのまま使う。
            try? FileManager.default.removeItem(at: temporary)
        }
        return destination
    }

    #if !os(macOS)
    /// iOS / visionOS: `~/Library/Application Support/CoreLLMCompiledModels/<バンドルキー>/<name>.mlmodelc`。
    private static func applicationSupportCompiledURL(bundleURL: URL, compiledName: String) throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        return base
            .appending(path: "CoreLLMCompiledModels", directoryHint: .isDirectory)
            .appending(path: stableKey(for: bundleURL), directoryHint: .isDirectory)
            .appending(path: compiledName)
    }

    /// バンドルの安定キー。String.hashValue はプロセス毎に seed が変わり再起動で不一致になる(= 毎回再コンパイル)ため、
    /// 決定的な FNV-1a 64bit を使う。同一インストールのアプリコンテナ絶対パスは不変なので一意・安定。
    private static func stableKey(for url: URL) -> String {
        let path = url.standardizedFileURL.path()
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in path.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return "\(url.lastPathComponent)-\(String(hash, radix: 16))"
    }
    #endif
}
