import Foundation

struct CatalogModel: Identifiable, Sendable {
    let id: String
    let displayName: String
    let hfRepoID: String
    let revision: String
    let approxSizeText: String

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

enum ModelCatalog {
    static let all: [CatalogModel] = [
        CatalogModel(
            id: "gemma-4-12b-it-coreml-128k",
            displayName: "Gemma 4 12B IT — 128K Context Ladder",
            hfRepoID: "okayuji/gemma-4-12b-it-coreml-128k",
            revision: "main",
            approxSizeText: "10.2 GB"
        )
    ]

    static func model(id: String) -> CatalogModel? {
        all.first { $0.id == id }
    }
}
