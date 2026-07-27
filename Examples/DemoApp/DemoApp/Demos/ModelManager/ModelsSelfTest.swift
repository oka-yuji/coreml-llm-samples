import Foundation

private final class ByteCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Int64 = 0
    func set(_ newValue: Int64) {
        lock.lock(); defer { lock.unlock() }
        if newValue > value { value = newValue }
    }
    func get() -> Int64 {
        lock.lock(); defer { lock.unlock() }
        return value
    }
}

enum ModelsSelfTest {
    static func run() -> Never {
        let args = CommandLine.arguments
        func value(_ flag: String) -> String? {
            if let i = args.firstIndex(of: flag), i + 1 < args.count { return args[i + 1] }
            return args.first { $0.hasPrefix(flag + "=") }.map { String($0.dropFirst(flag.count + 1)) }
        }
        let subcommand = value("--models-selftest") ?? args.first { arg in
            ["list", "download", "cancel", "delete", "deleteall", "verify"].contains(arg)
        } ?? "verify"
        let repoID = value("--repo") ?? LLMModels.downloadable(on: .macOS).first?.hfRepoID ?? ""
        let revision = value("--revision") ?? LLMModels.all.first(where: { $0.hfRepoID == repoID })?.hfRevision ?? "main"
        let seconds = value("--seconds").flatMap { Double($0) } ?? 5

        Task { @MainActor in
            let code = await execute(subcommand: subcommand, repoID: repoID, revision: revision, seconds: seconds)
            exit(code)
        }
        dispatchMain()
    }

    private static func out(_ s: String) { FileHandle.standardOutput.write(Data((s + "\n").utf8)) }
    private static func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

    private static func onDiskStats(_ directory: URL) -> (count: Int, bytes: Int64) {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: []
        ) else { return (0, 0) }
        var count = 0
        var bytes: Int64 = 0
        for case let url as URL in enumerator {
            guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  values.isRegularFile == true, let size = values.fileSize else { continue }
            if url.lastPathComponent == ModelStorage.markerName { continue }
            count += 1
            bytes += Int64(size)
        }
        return (count, bytes)
    }

    @MainActor
    private static func execute(subcommand: String, repoID: String, revision: String, seconds: Double) async -> Int32 {
        guard !repoID.isEmpty else {
            err("models-selftest: no repo id")
            return 2
        }
        out("[models-selftest] subcommand=\(subcommand) repo=\(repoID) rev=\(revision)")
        out("[models-selftest] token_detected=\(HFToken.isAvailable)")
        let folderName = LLMModels.all.first { $0.hfRepoID == repoID }?.bundleFolderName ?? repoID
        let destination = ModelStorage.bundleDirectory(for: folderName)
        out("[models-selftest] destination=\(destination.path(percentEncoded: false))")

        switch subcommand {
        case "list":
            return await listCommand(repoID: repoID, revision: revision)
        case "download":
            return await downloadCommand(repoID: repoID, revision: revision, destination: destination)
        case "cancel":
            return await cancelCommand(repoID: repoID, revision: revision, destination: destination, seconds: seconds)
        case "delete":
            try? ModelStorage.deleteBundle(for: repoID)
            let exists = FileManager.default.fileExists(atPath: destination.path(percentEncoded: false))
            out("EXISTS_AFTER_DELETE=\(exists)")
            return exists ? 1 : 0
        case "deleteall":
            try? ModelStorage.deleteAll()
            let remaining = (try? FileManager.default.contentsOfDirectory(atPath: ModelStorage.modelsRoot().path(percentEncoded: false)))?.count ?? 0
            out("MODELS_ROOT_ENTRIES=\(remaining)")
            return remaining == 0 ? 0 : 1
        case "verify":
            let complete = ModelStorage.isComplete(bundleDirectory: destination)
            let stats = onDiskStats(destination)
            out("IS_COMPLETE=\(complete)")
            out("ONDISK_FILES=\(stats.count)")
            out("ONDISK_BYTES=\(stats.bytes)")
            if let marker = ModelStorage.readCompletionMarker(at: destination) {
                out("MARKER_FILES=\(marker.fileCount)")
                out("MARKER_BYTES=\(marker.totalBytes)")
            }
            return complete ? 0 : 1
        default:
            err("models-selftest: unknown subcommand \(subcommand)")
            return 2
        }
    }

    @MainActor
    private static func listCommand(repoID: String, revision: String) async -> Int32 {
        do {
            let files = try await HubDownloader().fileList(repoID: repoID, revision: revision)
            let total = files.reduce(Int64(0)) { $0 + $1.size }
            out("TREE_FILES=\(files.count)")
            out("TREE_BYTES=\(total)")
            for file in files.sorted(by: { $0.path < $1.path }) {
                out(String(format: "%13d  %@", file.size, file.path))
            }
            return 0
        } catch {
            err("models-selftest list failed: \(error)")
            return 1
        }
    }

    @MainActor
    private static func downloadCommand(repoID: String, revision: String, destination: URL) async -> Int32 {
        let start = Date()
        do {
            let treeFiles = try await HubDownloader().fileList(repoID: repoID, revision: revision)
            let treeBytes = treeFiles.reduce(Int64(0)) { $0 + $1.size }
            out("TREE_FILES=\(treeFiles.count)")
            out("TREE_BYTES=\(treeBytes)")

            let worker = HubDownloader()
            let summary = try await worker.download(repoID: repoID, revision: revision, to: destination) { progress in
                if progress.downloadedBytes % 500_000_000 < 20_000_000 {
                    Self.out(String(format: "[progress] %.1f%%  %@  %@",
                                    progress.fraction * 100,
                                    ByteFormatting.formatBytes(progress.downloadedBytes),
                                    ByteFormatting.formatSpeed(progress.bytesPerSecond)))
                }
            }
            let elapsed = Date().timeIntervalSince(start)
            let stats = onDiskStats(destination)
            out("SUMMARY_FILES=\(summary.fileCount)")
            out("SUMMARY_BYTES=\(summary.totalBytes)")
            out("ONDISK_FILES=\(stats.count)")
            out("ONDISK_BYTES=\(stats.bytes)")
            out("MARKER_PRESENT=\(ModelStorage.readCompletionMarker(at: destination) != nil)")
            out("IS_COMPLETE=\(ModelStorage.isComplete(bundleDirectory: destination))")
            out(String(format: "ELAPSED_SECONDS=%.1f", elapsed))
            let match = summary.fileCount == treeFiles.count
                && summary.totalBytes == treeBytes
                && stats.count == treeFiles.count
                && stats.bytes == treeBytes
                && ModelStorage.isComplete(bundleDirectory: destination)
            out("MATCH=\(match)")
            return match ? 0 : 1
        } catch {
            err("models-selftest download failed: \(error)")
            return 1
        }
    }

    @MainActor
    private static func cancelCommand(repoID: String, revision: String, destination: URL, seconds: Double) async -> Int32 {
        let downloader = HubDownloader()
        let counter = ByteCounter()
        let worker = Task {
            do {
                _ = try await downloader.download(repoID: repoID, revision: revision, to: destination) { progress in
                    counter.set(progress.downloadedBytes)
                }
            } catch {
                Self.out("[cancel] worker ended: \(error.localizedDescription)")
            }
        }

        var elapsed = 0.0
        while elapsed < seconds {
            try? await Task.sleep(for: .milliseconds(500))
            elapsed += 0.5
        }
        let before = counter.get()
        out("BYTES_BEFORE_CANCEL=\(before)")
        downloader.cancel()
        worker.cancel()
        let cancelledAt = Date()

        try? await Task.sleep(for: .seconds(3))
        let after = counter.get()
        out("BYTES_AFTER_CANCEL=\(after)")
        out("BYTES_DELTA_AFTER_CANCEL=\(after - before)")
        out(String(format: "SETTLE_SECONDS=%.1f", Date().timeIntervalSince(cancelledAt)))
        let stats = onDiskStats(destination)
        out("COMPLETED_FILES_ON_DISK=\(stats.count)")
        out("ONDISK_BYTES=\(stats.bytes)")
        out("IS_COMPLETE=\(ModelStorage.isComplete(bundleDirectory: destination))")

        let stopped = (after - before) < 8_000_000
        let startedTransfer = before > 0
        out("IO_STOPPED=\(stopped)")
        out("STARTED_TRANSFER=\(startedTransfer)")
        return (stopped && startedTransfer) ? 0 : 1
    }
}
