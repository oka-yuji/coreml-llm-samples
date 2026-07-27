import Foundation
import Observation

@MainActor
@Observable
final class ModelRowState {
    let model: LLMModel
    var isDownloading = false
    var progress: Double = 0
    var downloadedBytes: Int64 = 0
    var totalBytes: Int64 = 0
    var bytesPerSecond: Double = 0
    var error: String?
    var isDownloaded = false
    var diskSize: UInt64 = 0

    @ObservationIgnored var downloader: HubDownloader?
    @ObservationIgnored var task: Task<Void, Never>?

    init(model: LLMModel) {
        self.model = model
    }

    var estimatedTimeRemaining: TimeInterval? {
        guard bytesPerSecond > 0, totalBytes > downloadedBytes else { return nil }
        return Double(totalBytes - downloadedBytes) / bytesPerSecond
    }

    var bundleDirectory: URL {
        ModelStorage.bundleDirectory(for: model.bundleFolderName)
    }
}

@MainActor
@Observable
final class ModelsViewModel {
    private(set) var rows: [ModelRowState]
    var modelsDirectorySize: UInt64 = 0
    var availableDiskSpace: UInt64 = 0

    var hfToken: String = HFToken.stored ?? "" {
        didSet { HFToken.save(hfToken) }
    }

    var tokenAvailable: Bool { HFToken.isAvailable }

    init() {
        rows = LLMModels.downloadable().map { ModelRowState(model: $0) }
        refresh()
    }

    func row(for id: String) -> ModelRowState? {
        rows.first { $0.model.id == id }
    }

    var isBusy: Bool {
        rows.contains { $0.isDownloading }
    }

    func refresh() {
        for row in rows {
            let directory = row.bundleDirectory
            row.isDownloaded = ModelStorage.isComplete(bundleDirectory: directory)
            row.diskSize = row.isDownloaded ? ModelStorage.directorySize(at: directory) : 0
        }
        modelsDirectorySize = ModelStorage.directorySize(at: ModelStorage.modelsRoot())
        availableDiskSpace = ModelStorage.availableDiskSpace()
    }

    func startDownload(_ id: String) {
        guard let row = row(for: id), !row.isDownloading,
              let repoID = row.model.hfRepoID, let revision = row.model.hfRevision else { return }
        row.error = nil
        row.isDownloading = true
        row.progress = 0
        row.downloadedBytes = 0
        row.bytesPerSecond = 0
        row.totalBytes = Int64(LLMModel.parseSizeHint(row.model.approxSizeText))

        let quickAvailable = ModelStorage.availableDiskSpace()
        if row.totalBytes > 0, quickAvailable > 0, quickAvailable < UInt64(row.totalBytes) {
            row.error = "Insufficient disk space. Required \(ByteFormatting.formatBytes(row.totalBytes)), available \(ByteFormatting.formatBytes(quickAvailable))"
            row.isDownloading = false
            return
        }

        let downloader = HubDownloader()
        row.downloader = downloader
        let destination = row.bundleDirectory

        row.task = Task { [weak self] in
            guard let self else { return }
            do {
                _ = try await downloader.download(repoID: repoID, revision: revision, to: destination) { progress in
                    Task { @MainActor in self.applyProgress(id, progress) }
                }
                self.finishDownload(id, error: nil)
            } catch {
                self.finishDownload(id, error: error)
            }
        }
    }

    private func applyProgress(_ id: String, _ progress: DownloadProgress) {
        guard let row = row(for: id) else { return }
        row.progress = progress.fraction
        row.downloadedBytes = progress.downloadedBytes
        row.totalBytes = progress.totalBytes
        row.bytesPerSecond = progress.bytesPerSecond
    }

    private func finishDownload(_ id: String, error: Error?) {
        guard let row = row(for: id) else { return }
        row.isDownloading = false
        row.downloader = nil
        if let error, !Self.isCancellation(error) {
            row.error = (error as? LocalizedError)?.errorDescription ?? String(describing: error)
        }
        refresh()
    }

    func cancelDownload(_ id: String) {
        guard let row = row(for: id) else { return }
        row.downloader?.cancel()
        row.task?.cancel()
        row.isDownloading = false
        refresh()
    }

    func delete(_ id: String) {
        guard let row = row(for: id) else { return }
        try? ModelStorage.deleteBundle(for: row.model.bundleFolderName)
        refresh()
    }

    func deleteAll() {
        guard !isBusy else { return }
        try? ModelStorage.deleteAll()
        refresh()
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if case HubDownloadError.cancelled = error { return true }
        return (error as NSError).code == NSURLErrorCancelled
    }
}
