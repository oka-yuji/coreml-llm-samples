import Foundation

struct LLMModel: Identifiable, Sendable {
    enum Platform: String, Sendable, CaseIterable {
        case iOS
        case macOS
    }

    enum Source: Sendable {
        case huggingFace(repoID: String, revision: String, folderName: String)
        case localBundle(folderName: String)
    }

    let id: String
    let displayName: String
    let source: Source
    let supportedPlatforms: Set<Platform>
    let verifiedOn: String
    let approxSizeText: String

    var hfRepoID: String? {
        if case .huggingFace(let repoID, _, _) = source { return repoID }
        return nil
    }

    var hfRevision: String? {
        if case .huggingFace(_, let revision, _) = source { return revision }
        return nil
    }

    var bundleFolderName: String {
        switch source {
        case .huggingFace(_, _, let folderName): return folderName
        case .localBundle(let folderName): return folderName
        }
    }

    var isDownloadable: Bool { hfRepoID != nil }

    static func parseSizeHint(_ hint: String) -> UInt64 {
        let trimmed = hint
            .trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: "~", with: "")
        let pattern = #"([0-9.]+)\s*(TB|GB|MB|KB)"#
        guard let match = trimmed.range(of: pattern, options: [.regularExpression, .caseInsensitive]) else {
            return 0
        }
        let matched = String(trimmed[match])
        guard let numRange = matched.range(of: #"[0-9.]+"#, options: .regularExpression),
              let value = Double(matched[numRange]) else {
            return 0
        }
        let upper = matched.uppercased()
        if upper.contains("TB") { return UInt64(value * 1_000_000_000_000) }
        if upper.contains("GB") { return UInt64(value * 1_000_000_000) }
        if upper.contains("MB") { return UInt64(value * 1_000_000) }
        if upper.contains("KB") { return UInt64(value * 1_000) }
        return 0
    }
}

enum LLMModels {
    static let all: [LLMModel] = [
        LLMModel(
            id: "gemma-4-12b-it-coreml-128k",
            displayName: "Gemma 4 12B IT — 128K Context Ladder",
            source: .huggingFace(
                repoID: "okayuji/gemma-4-12b-it-coreml-128k",
                revision: "main",
                folderName: "okayuji/gemma-4-12b-it-coreml-128k"),
            supportedPlatforms: [.macOS],
            verifiedOn: "M4 Max, macOS 26",
            approxSizeText: "10.2 GB"
        ),
        LLMModel(
            id: "gemma-4-e2b-speculative-pal6",
            displayName: "Gemma 4 E2B Speculative (pal6)",
            source: .huggingFace(
                repoID: "okayuji/Gemma-4-E2B-it-coreml-speculative",
                revision: "6b95bf36211b0a21bf35af8721960e01409ae5b9",
                folderName: "gemma-4-e2b-speculative-pal6"),
            supportedPlatforms: [.iOS, .macOS],
            verifiedOn: "iPhone 15, iOS 26",
            approxSizeText: "4.9 GB"
        ),
        LLMModel(
            id: "gemma-4-e4b-speculative",
            displayName: "Gemma 4 E4B Speculative (Mac GPU)",
            source: .huggingFace(
                repoID: "okayuji/Gemma-4-E4B-it-coreml-speculative",
                revision: "4ce978f9c5ba47c3c0ffce366d9ef3e73039f96e",
                folderName: "gemma-4-e4b-speculative"),
            supportedPlatforms: [.macOS],
            verifiedOn: "M4 Max, macOS 26",
            approxSizeText: "6.5 GB"
        )
    ]

    static var currentPlatform: LLMModel.Platform {
        #if os(macOS)
        .macOS
        #else
        .iOS
        #endif
    }

    static func supported(on platform: LLMModel.Platform = currentPlatform) -> [LLMModel] {
        all.filter { $0.supportedPlatforms.contains(platform) }
    }

    static func downloadable(on platform: LLMModel.Platform = currentPlatform) -> [LLMModel] {
        supported(on: platform).filter(\.isDownloadable)
    }

    static func model(id: String) -> LLMModel? {
        all.first { $0.id == id }
    }
}
