import Foundation

enum HFToken {
    private static let defaultsKey = "huggingface_token"

    static var stored: String? {
        let raw = UserDefaults.standard.string(forKey: defaultsKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (raw?.isEmpty == false) ? raw : nil
    }

    static func save(_ token: String) {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            UserDefaults.standard.removeObject(forKey: defaultsKey)
        } else {
            UserDefaults.standard.set(trimmed, forKey: defaultsKey)
        }
    }

    static func detect() -> String? {
        if let env = ProcessInfo.processInfo.environment["HF_TOKEN"] {
            let trimmed = env.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        if let stored { return stored }
        #if os(macOS)
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
        #endif
        return nil
    }

    static var isAvailable: Bool { detect() != nil }
}
