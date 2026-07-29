import Foundation

struct HubFile: Sendable {
    let path: String
    let size: Int64
}

struct DownloadProgress: Sendable {
    let fraction: Double
    let downloadedBytes: Int64
    let totalBytes: Int64
    let bytesPerSecond: Double
}

struct DownloadSummary: Sendable {
    let fileCount: Int
    let totalBytes: Int64
}

enum HubDownloadError: LocalizedError {
    case invalidURL(String)
    case httpError(statusCode: Int, url: String)
    case invalidResponse(url: String)
    case cancelled
    case insufficientDiskSpace(required: Int64, available: Int64)

    var errorDescription: String? {
        switch self {
        case .invalidURL(let url):
            return "Invalid URL: \(url)"
        case .httpError(let code, let url):
            return "HTTP \(code) for \(url)"
        case .invalidResponse(let url):
            return "Invalid response from \(url)"
        case .cancelled:
            return "Download cancelled"
        case .insufficientDiskSpace(let required, let available):
            return "Insufficient disk space. Required \(ByteFormatting.formatBytes(required)), available \(ByteFormatting.formatBytes(available))"
        }
    }
}

final class HubDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let baseURL = "https://huggingface.co"
    private let lock = NSLock()
    private var entries: [Int: Entry] = [:]
    private var liveTasks: [Int: URLSessionDownloadTask] = [:]
    private var didCancel = false

    private struct Entry {
        let destinationFile: URL
        let tracker: ProgressTracker
        let progressHandler: (@Sendable (DownloadProgress) -> Void)?
        let continuation: CheckedContinuation<URLResponse, Error>
        var lastBytes: Int64 = 0
        var didResume = false
    }

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 3600
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    func fileList(repoID: String, revision: String) async throws -> [HubFile] {
        let urlString = "\(baseURL)/api/models/\(repoID)/tree/\(revision)?recursive=true"
        guard let url = URL(string: urlString) else {
            throw HubDownloadError.invalidURL(urlString)
        }
        let (data, response) = try await session.data(for: URLRequest(url: url))
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw HubDownloadError.httpError(statusCode: code, url: urlString)
        }
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw HubDownloadError.invalidResponse(url: urlString)
        }
        var result: [HubFile] = []
        for entry in raw {
            guard let type = entry["type"] as? String, type == "file",
                  let path = entry["path"] as? String else {
                continue
            }
            let size = (entry["size"] as? NSNumber)?.int64Value ?? 0
            result.append(HubFile(path: path, size: size))
        }
        return result
    }

    @discardableResult
    func download(
        repoID: String,
        revision: String,
        to destinationDir: URL,
        progress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> DownloadSummary {
        defer { session.finishTasksAndInvalidate() }

        let files = try await fileList(repoID: repoID, revision: revision)
        guard !files.isEmpty else {
            throw HubDownloadError.invalidResponse(url: repoID)
        }
        let totalBytes = files.reduce(Int64(0)) { $0 + $1.size }

        try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)
        ModelStorage.excludeFromBackup(ModelStorage.modelsRoot())

        var alreadyPresent: Int64 = 0
        for file in files {
            let destination = destinationDir.appending(path: file.path)
            if let existing = fileSize(at: destination), existing == file.size, file.size > 0 {
                alreadyPresent += file.size
            }
        }
        let needed = totalBytes - alreadyPresent
        let available = Int64(ModelStorage.availableDiskSpace())
        if available > 0, available < needed {
            throw HubDownloadError.insufficientDiskSpace(required: needed, available: available)
        }

        let tracker = ProgressTracker(totalBytes: totalBytes)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for file in files {
                group.addTask { [self] in
                    try await downloadFile(
                        file: file,
                        repoID: repoID,
                        revision: revision,
                        destinationDir: destinationDir,
                        tracker: tracker,
                        progress: progress
                    )
                }
            }
            for try await _ in group {}
        }

        ModelStorage.writeCompletionMarker(at: destinationDir, fileCount: files.count, totalBytes: totalBytes)
        return DownloadSummary(fileCount: files.count, totalBytes: totalBytes)
    }

    func cancel() {
        lock.lock()
        didCancel = true
        let tasks = Array(liveTasks.values)
        lock.unlock()
        for task in tasks {
            task.cancel()
        }
    }

    private func downloadFile(
        file: HubFile,
        repoID: String,
        revision: String,
        destinationDir: URL,
        tracker: ProgressTracker,
        progress: (@Sendable (DownloadProgress) -> Void)?
    ) async throws {
        if isCancelled() {
            throw HubDownloadError.cancelled
        }
        let destination = destinationDir.appending(path: file.path)
        if let existing = fileSize(at: destination), existing == file.size, file.size > 0 {
            let result = tracker.add(bytes: file.size)
            if result.shouldReport {
                progress?(DownloadProgress(
                    fraction: result.fraction,
                    downloadedBytes: result.completed,
                    totalBytes: tracker.totalBytes,
                    bytesPerSecond: result.bytesPerSecond ?? 0
                ))
            }
            return
        }
        let encodedPath = file.path
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        let urlString = "\(baseURL)/\(repoID)/resolve/\(revision)/\(encodedPath)"
        guard let url = URL(string: urlString) else {
            throw HubDownloadError.invalidURL(urlString)
        }
        let response = try await downloadOne(
            url: url,
            destinationFile: destination,
            tracker: tracker,
            progress: progress
        )
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            try? FileManager.default.removeItem(at: destination)
            throw HubDownloadError.httpError(statusCode: http.statusCode, url: urlString)
        }
    }

    private func downloadOne(
        url: URL,
        destinationFile: URL,
        tracker: ProgressTracker,
        progress: (@Sendable (DownloadProgress) -> Void)?
    ) async throws -> URLResponse {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<URLResponse, Error>) in
            lock.lock()
            if didCancel {
                lock.unlock()
                continuation.resume(throwing: HubDownloadError.cancelled)
                return
            }
            let task = session.downloadTask(with: URLRequest(url: url))
            let id = task.taskIdentifier
            entries[id] = Entry(
                destinationFile: destinationFile,
                tracker: tracker,
                progressHandler: progress,
                continuation: continuation
            )
            liveTasks[id] = task
            lock.unlock()
            task.resume()
        }
    }

    private func isCancelled() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return didCancel
    }

    private func fileSize(at url: URL) -> Int64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) else { return nil }
        return (attributes[.size] as? NSNumber)?.int64Value
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let id = downloadTask.taskIdentifier
        lock.lock()
        guard var entry = entries[id] else {
            lock.unlock()
            return
        }
        let delta = totalBytesWritten - entry.lastBytes
        entry.lastBytes = totalBytesWritten
        let tracker = entry.tracker
        let handler = entry.progressHandler
        entries[id] = entry
        lock.unlock()

        guard delta > 0 else { return }
        let result = tracker.add(bytes: delta)
        if result.shouldReport {
            handler?(DownloadProgress(
                fraction: result.fraction,
                downloadedBytes: result.completed,
                totalBytes: tracker.totalBytes,
                bytesPerSecond: result.bytesPerSecond ?? 0
            ))
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        let id = downloadTask.taskIdentifier
        lock.lock()
        guard let entry = entries[id], !entry.didResume else {
            lock.unlock()
            return
        }
        let destination = entry.destinationFile
        lock.unlock()

        let moveError: Error?
        do {
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if FileManager.default.fileExists(atPath: destination.path) {
                try? FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: location, to: destination)
            moveError = nil
        } catch {
            moveError = error
        }

        lock.lock()
        liveTasks.removeValue(forKey: id)
        guard var entry = entries[id], !entry.didResume else {
            lock.unlock()
            return
        }
        entry.didResume = true
        entries[id] = entry
        let continuation = entry.continuation
        lock.unlock()

        if let moveError {
            continuation.resume(throwing: moveError)
        } else if let response = downloadTask.response {
            continuation.resume(returning: response)
        } else {
            try? FileManager.default.removeItem(at: destination)
            continuation.resume(throwing: HubDownloadError.invalidResponse(url: downloadTask.originalRequest?.url?.absoluteString ?? ""))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        let id = task.taskIdentifier
        lock.lock()
        liveTasks.removeValue(forKey: id)
        guard var entry = entries[id], !entry.didResume else {
            entries.removeValue(forKey: id)
            lock.unlock()
            return
        }
        entry.didResume = true
        entries[id] = entry
        let continuation = entry.continuation
        lock.unlock()

        if let error {
            if (error as NSError).code == NSURLErrorCancelled {
                continuation.resume(throwing: HubDownloadError.cancelled)
            } else {
                continuation.resume(throwing: error)
            }
        } else {
            continuation.resume(throwing: HubDownloadError.invalidResponse(url: task.originalRequest?.url?.absoluteString ?? ""))
        }

        lock.lock()
        entries.removeValue(forKey: id)
        lock.unlock()
    }
}

final class ProgressTracker: @unchecked Sendable {
    let totalBytes: Int64
    private let startTime = Date()
    private let lock = NSLock()
    private var completedBytes: Int64 = 0
    private var lastReportedAt = Date.distantPast

    struct Result {
        let fraction: Double
        let completed: Int64
        let bytesPerSecond: Double?
        let shouldReport: Bool
    }

    init(totalBytes: Int64) {
        self.totalBytes = totalBytes
    }

    func add(bytes: Int64) -> Result {
        let now = Date()
        lock.lock()
        completedBytes += bytes
        let isComplete = totalBytes > 0 && completedBytes >= totalBytes
        let shouldReport = now.timeIntervalSince(lastReportedAt) >= 0.1 || isComplete
        if shouldReport { lastReportedAt = now }
        let completed = completedBytes
        lock.unlock()

        let fraction = totalBytes > 0 ? min(Double(completed) / Double(totalBytes), 1.0) : 0
        let elapsed = now.timeIntervalSince(startTime)
        let bytesPerSecond = elapsed > 0.5 ? Double(completed) / elapsed : nil
        return Result(fraction: fraction, completed: completed, bytesPerSecond: bytesPerSecond, shouldReport: shouldReport)
    }
}
