import Foundation

enum ModelStorage {
    static let markerName = ".download-complete.json"

    struct CompletionMarker: Codable, Sendable {
        let fileCount: Int
        let totalBytes: Int64
        let completedAt: String
    }

    static func appBaseDirectory() -> URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: "Library/DemoApp")
    }

    static func modelsRoot() -> URL {
        let root = appBaseDirectory().appending(path: "models")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        excludeFromBackup(root)
        return root
    }

    static func bundleDirectory(for repoID: String) -> URL {
        modelsRoot().appending(path: repoID)
    }

    static func writeCompletionMarker(at bundleDirectory: URL, fileCount: Int, totalBytes: Int64) {
        let marker = CompletionMarker(
            fileCount: fileCount,
            totalBytes: totalBytes,
            completedAt: ISO8601DateFormatter().string(from: Date())
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(marker) else { return }
        try? data.write(to: bundleDirectory.appending(path: markerName))
    }

    static func readCompletionMarker(at bundleDirectory: URL) -> CompletionMarker? {
        let url = bundleDirectory.appending(path: markerName)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(CompletionMarker.self, from: data)
    }

    static func isComplete(bundleDirectory: URL) -> Bool {
        let fileManager = FileManager.default
        let hasMarker = fileManager.fileExists(atPath: bundleDirectory.appending(path: markerName).path(percentEncoded: false))
        let hasManifest = fileManager.fileExists(atPath: bundleDirectory.appending(path: "manifest.json").path(percentEncoded: false))
        return hasMarker && hasManifest
    }

    static func deleteBundle(for repoID: String) throws {
        let directory = bundleDirectory(for: repoID)
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: directory.path(percentEncoded: false)) {
            try fileManager.removeItem(at: directory)
        }
        let owner = directory.deletingLastPathComponent()
        let root = modelsRoot()
        if owner.path(percentEncoded: false) != root.path(percentEncoded: false),
           let contents = try? fileManager.contentsOfDirectory(atPath: owner.path(percentEncoded: false)),
           contents.isEmpty {
            try? fileManager.removeItem(at: owner)
        }
    }

    static func deleteAll() throws {
        let root = modelsRoot()
        let fileManager = FileManager.default
        let contents = (try? fileManager.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        for item in contents {
            try fileManager.removeItem(at: item)
        }
    }

    static func directorySize(at url: URL) -> UInt64 {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: []
        ) else {
            return 0
        }
        var total: UInt64 = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
                  values.isRegularFile == true,
                  let size = values.fileSize else {
                continue
            }
            total += UInt64(size)
        }
        return total
    }

    static func availableDiskSpace() -> UInt64 {
        let probe = FileManager.default.homeDirectoryForCurrentUser
        if let values = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let capacity = values.volumeAvailableCapacityForImportantUsage {
            return UInt64(max(0, capacity))
        }
        if let attributes = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()),
           let free = attributes[.systemFreeSize] as? UInt64 {
            return free
        }
        return 0
    }

    static func excludeFromBackup(_ url: URL) {
        var target = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? target.setResourceValues(values)
    }
}
