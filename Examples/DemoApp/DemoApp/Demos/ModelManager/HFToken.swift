import Foundation

enum HFToken {
    static func detect() -> String? {
        if let env = ProcessInfo.processInfo.environment["HF_TOKEN"] {
            let trimmed = env.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let candidates = [
            home.appending(path: ".cache/huggingface/token"),
            home.appending(path: ".huggingface/token")
        ]
        for url in candidates {
            guard let raw = try? String(contentsOf: url, encoding: .utf8) else { continue }
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return nil
    }

    static var isAvailable: Bool { detect() != nil }
}
