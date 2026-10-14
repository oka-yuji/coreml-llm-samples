import Foundation
import LLMCore
import PDFKit
import SwiftUI

// Retrieval for the Chat demo's RAG mode. The app searches and reads — the model never calls a
// tool — so a 2 B model with a 2048-token window spends its whole prompt on passages and the
// question. Search and page reading are the Agent demo's (`AgentTools`); what is new here is
// collecting passages from several pages at once, ranking them against the question, and fitting
// the best of them into a token budget.

struct RAGSettings: Sendable {

    var web = false
    var local = false

    /// Shared with the Agent demo through `UserDefaults`, so a Brave key set there works here.
    var searchResults = 5
    var braveAPIKey = ""

    /// How many of the search results to open, and how much of each page to keep.
    var pagesToRead = 3
    var pageCharacterBudget = 1_500

    var documentsDirectory: URL = RAGRetriever.defaultDocumentsDirectory()

    var isOn: Bool { web || local }

    /// Shares the Agent demo's Brave key rather than asking for a second one.
    @MainActor
    static func fromDefaults() -> RAGSettings {
        var settings = RAGSettings()
        settings.braveAPIKey = UserDefaults.standard
            .string(forKey: AgentViewModel.braveKeyDefaultsKey) ?? ""
        return settings
    }

    var provider: String { braveAPIKey.isEmpty ? "DuckDuckGo" : "Brave" }
}

struct RAGRetrieval: Sendable {

    /// Packed and numbered: `sources[0]` is `資料[1]`.
    var sources: [RAGSource] = []

    /// One line per failure — no network, a page that would not load — shown in the UI and logged.
    var notes: [String] = []

    var searchResults = 0
    var pagesRead = 0
    var candidates = 0
    var sourceTokens = 0
    var seconds: Double = 0

    var citations: [String] { RAGContext.citations(sources) }

    var summary: String {
        "search \(searchResults) / pages \(pagesRead) / passages \(candidates) → "
            + "\(sources.count) sources, \(sourceTokens) tokens"
    }
}

enum RAGRetriever {

    /// `Documents/rag/` — on iOS this is the folder the Files app and `devicectl copy to` reach.
    /// Created on first use so there is somewhere to drop documents.
    static func defaultDocumentsDirectory() -> URL {
        let url = ModelStorage.documentsDirectory().appending(path: "rag", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Roughly what a SentencePiece tokenizer makes of Japanese and English, for when no model is
    /// loaded to ask. Only ever decides how many passages fit, never what is sent.
    static func estimateTokens(_ text: String) -> Int {
        let wide = text.unicodeScalars.filter { !$0.isASCII }.count
        return max(1, wide + (text.unicodeScalars.count - wide) / 4)
    }

    /// Everything between the question and the prompt. Never throws: a source that fails becomes a
    /// note and the rest of the answer goes ahead without it. `count` is async because the
    /// tokenizer lives inside the engine actor, so every candidate is measured once, up front.
    static func gather(
        question: String, settings: RAGSettings, count: @Sendable (String) async -> Int
    ) async -> RAGRetrieval {
        let start = ContinuousClock.now
        var out = RAGRetrieval()
        var candidates: [RAGSource] = []

        if settings.local {
            let local = RAGDocuments.load(directory: settings.documentsDirectory) { url in
                PDFDocument(url: url)?.string
            }
            if local.isEmpty {
                out.notes.append("端末内文書: \(settings.documentsDirectory.path(percentEncoded: false)) "
                    + "に読める文書がありません")
            }
            candidates += local
        }

        if settings.web {
            candidates += await webCandidates(question: question, settings: settings, into: &out)
        }

        out.candidates = candidates.count
        var table: [String: Int] = [:]
        for candidate in candidates {
            let measured = RAGContext.measuredBlock(candidate)
            if table[measured] == nil { table[measured] = await count(measured) }
        }
        let tokens: (String) -> Int = { table[$0] ?? estimateTokens($0) }
        out.sources = RAGContext.select(
            candidates, question: question, count: tokens, rank: bestRanking)
        out.sourceTokens = out.sources.map { tokens(RAGContext.measuredBlock($0)) }.reduce(0, +)
        out.seconds = (ContinuousClock.now - start) / .seconds(1)
        return out
    }

    /// The Agent demo's FTS5/BM25 ranking, falling back to term counting when SQLite has no FTS5
    /// or nothing matched.
    static func bestRanking(_ passages: [String], query: String) -> [Int] {
        let ranked = AgentTools.bm25Ranking(passages, terms: AgentTools.matchTerms(query))
        return ranked.isEmpty ? RAGContext.rank(passages, query: query) : ranked
    }

    // MARK: - Web

    private static func webCandidates(
        question: String, settings: RAGSettings, into out: inout RAGRetrieval
    ) async -> [RAGSource] {
        let results: [AgentTools.SearchResult]
        do {
            results = settings.braveAPIKey.isEmpty
                ? try await AgentTools.duckDuckGoResults(
                    query: question, limit: settings.searchResults)
                : try await AgentTools.braveResults(
                    query: question, limit: settings.searchResults, key: settings.braveAPIKey)
        } catch let failure as AgentToolMessage {
            out.notes.append("Web 検索: \(failure.text)")
            return []
        } catch {
            out.notes.append("Web 検索: ネットワークエラー — \(error.localizedDescription)")
            return []
        }
        out.searchResults = results.count
        guard !results.isEmpty else {
            out.notes.append("Web 検索: \"\(question)\" の結果が 0 件")
            return []
        }

        let wanted = results
            .prefix(settings.pagesToRead)
            .compactMap { result -> (AgentTools.SearchResult, URL)? in
                guard let url = URL(string: result.url), let scheme = url.scheme?.lowercased(),
                      scheme == "http" || scheme == "https",
                      !AgentTools.isLocalOrPrivate(url.host()) else { return nil }
                return (result, url)
            }

        let budget = settings.pageCharacterBudget
        let read = await withTaskGroup(of: (Int, [RAGSource], String?).self) { group in
            for (index, pair) in wanted.enumerated() {
                group.addTask {
                    await passages(from: pair.0, url: pair.1, question: question, budget: budget)
                        .map { (index, $0, nil) } ?? (index, [], "ページ取得に失敗: \(pair.1.absoluteString)")
                }
            }
            var rows: [(Int, [RAGSource], String?)] = []
            for await row in group { rows.append(row) }
            return rows.sorted { $0.0 < $1.0 }
        }
        out.pagesRead = read.filter { !$0.1.isEmpty }.count
        out.notes += read.compactMap(\.2)
        return read.flatMap(\.1)
    }

    /// One page: raw HTML for the date, its visible text cut down to the part that answers the
    /// question, then split into passages the budget can pick from.
    private static func passages(
        from result: AgentTools.SearchResult, url: URL, question: String, budget: Int
    ) async -> [RAGSource]? {
        var html = ""
        var text = ""
        do {
            html = try await AgentTools.download(url: url)
            text = AgentTools.plainText(fromHTML: html)
        } catch {
            AgentTools.log("[rag] \(url.absoluteString) direct read failed: \(error)")
        }
        if text.count < 400 {
            // A page that ships an empty shell and fills it with JavaScript. Render it instead.
            guard let rendered = try? await AgentPageReader.readableText(url: url) else { return nil }
            text = AgentTools.condense(rendered)
        }
        guard !text.isEmpty else { return nil }

        let date = RAGContext.publishedDate(html: html)
            ?? RAGContext.isoDate(in: result.snippet)
            ?? RAGContext.isoDate(in: String(text.prefix(600)))
            ?? ""
        let relevant = AgentTools.pageText(text, query: question, budget: budget)
        let title = result.title.isEmpty ? (url.host() ?? url.absoluteString) : result.title
        return RAGContext.chunks(relevant, size: 400, overlap: 80).map {
            RAGSource(title: title, url: url.absoluteString, date: date, text: $0)
        }
    }
}


// MARK: - UI

/// The RAG switches for the chat status bar: the mode, and which sources it may use.
struct RAGControls: View {
    @Environment(ChatViewModel.self) private var vm

    var body: some View {
        @Bindable var vm = vm
        return HStack(spacing: 6) {
            Toggle("RAG", isOn: $vm.ragEnabled)
                .toggleStyle(.switch)
                .controlSize(.mini)
                .disabled(vm.isGenerating)
            if vm.ragEnabled {
                Toggle("Web", isOn: $vm.ragWeb)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .disabled(vm.isGenerating)
                Toggle("Docs", isOn: $vm.ragLocal)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .disabled(vm.isGenerating)
            }
        }
        .font(.caption)
        .fixedSize()
    }
}

/// The numbered sources an answer was built from, and any retrieval that failed.
struct RAGSourceFooter: View {
    let sources: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            if !sources.isEmpty {
                Text("出典")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(sources, id: \.self) { line in
                    Text(line)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }
}
