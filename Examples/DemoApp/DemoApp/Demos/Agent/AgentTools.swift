import Foundation
import SQLite3
import WebKit

struct AgentToolOutput: Sendable {
    var text: String
    var seen: [String: String] = [:]

    var seconds: Double = 0
}

struct AgentToolSettings: Sendable {
    var pageCharacterBudget: Int = 4_000
    var searchResults: Int = 5
    var braveAPIKey: String = ""

    var notesDirectory: URL = AgentTools.defaultNotesDirectory()

    var allowFileOperations = false
}

struct AgentToolMessage: Error, Sendable {
    var text: String
    var fatal = false
}

enum AgentTools {

    struct SearchResult: Equatable, Sendable {
        var title: String
        var url: String
        var snippet: String
    }

    static let names = ["web_search", "fetch_page", "write_note", "move_note"]

    static let userAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        + "(KHTML, like Gecko) Version/18.0 Safari/605.1.15"

    static let maxCallsPerTurn = 3
    static let ignoredCallMessage = "ignored: max \(maxCallsPerTurn) tool calls per turn"

    static func runAll(_ calls: [AgentToolCall], settings: AgentToolSettings) async -> [AgentToolOutput] {
        let executed = calls.prefix(maxCallsPerTurn)
        var outputs: [AgentToolOutput]
        if executed.contains(where: { fileTools.contains($0.name) }) {
            outputs = []
            for call in executed { outputs.append(await timed(call, settings: settings)) }
        } else {
            outputs = await withTaskGroup(of: (Int, AgentToolOutput).self) { group in
                for (index, call) in executed.enumerated() {
                    group.addTask { (index, await timed(call, settings: settings)) }
                }
                var byIndex: [Int: AgentToolOutput] = [:]
                for await (index, output) in group { byIndex[index] = output }
                return executed.indices.map {
                    byIndex[$0] ?? AgentToolOutput(text: "error: tool did not run")
                }
            }
        }
        outputs += calls.dropFirst(maxCallsPerTurn).map { _ in AgentToolOutput(text: ignoredCallMessage) }
        return outputs
    }

    static let fileTools: Set<String> = ["write_note", "move_note"]

    private static func timed(_ call: AgentToolCall, settings: AgentToolSettings) async -> AgentToolOutput {
        let start = ContinuousClock.now
        var output = await run(call, settings: settings)
        output.seconds = (ContinuousClock.now - start) / .seconds(1)
        return output
    }

    static func run(_ call: AgentToolCall, settings: AgentToolSettings) async -> AgentToolOutput {
        switch call.name {
        case "web_search":
            let query = (call.value("query") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !query.isEmpty else {
                return AgentToolOutput(text: "error: web_search needs a query parameter")
            }
            let requested = call.value("max_results")
                .flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            let limit = min(10, max(1, requested ?? settings.searchResults))
            do {
                return try await webSearch(query: query, limit: limit, braveAPIKey: settings.braveAPIKey)
            } catch let failure as AgentToolMessage {
                return AgentToolOutput(text: failure.text)
            } catch {
                return AgentToolOutput(text: "error: network — \(error.localizedDescription)")
            }
        case "fetch_page":
            let raw = (call.value("url") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard let url = URL(string: raw), let scheme = url.scheme?.lowercased(),
                  scheme == "http" || scheme == "https", url.host != nil else {
                return AgentToolOutput(text: "error: fetch_page needs an absolute http or https URL")
            }
            guard !isLocalOrPrivate(url.host()) else {
                return AgentToolOutput(text: localAddressRefusal)
            }
            do {
                return try await fetchPage(
                    url: url, budget: settings.pageCharacterBudget, query: call.value("query") ?? "")
            } catch let failure as AgentToolMessage {
                return AgentToolOutput(text: failure.text)
            } catch {
                return AgentToolOutput(text: "error: fetch_page failed (\(error))")
            }
        case "write_note":
            let title = call.value("title") ?? ""
            let content = call.value("content") ?? ""
            guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return AgentToolOutput(text: "error: write_note needs a content parameter")
            }
            do {
                return AgentToolOutput(text: "saved to " + (try writeNote(
                    title: title, content: content, directory: settings.notesDirectory)))
            } catch let failure as AgentToolMessage {
                return AgentToolOutput(text: failure.text)
            } catch {
                return AgentToolOutput(text: "error: write_note failed (\(error))")
            }
        case "move_note":
            guard settings.allowFileOperations else {
                return AgentToolOutput(
                    text: "error: file operations are disabled in Settings (Allow file operations)")
            }
            do {
                return AgentToolOutput(text: "moved to " + (try moveNote(
                    title: call.value("title") ?? "", destination: call.value("destination") ?? "",
                    notesDirectory: settings.notesDirectory)))
            } catch let failure as AgentToolMessage {
                return AgentToolOutput(text: failure.text)
            } catch {
                return AgentToolOutput(text: "error: move_note failed (\(error))")
            }
        default:
            return AgentToolOutput(
                text: "error: unknown tool \"\(call.name)\". Available tools: \(names.joined(separator: ", "))")
        }
    }

    static func webSearch(
        query: String, limit: Int, braveAPIKey: String
    ) async throws -> AgentToolOutput {
        let results = braveAPIKey.isEmpty
            ? try await duckDuckGoResults(query: query, limit: limit)
            : try await braveResults(query: query, limit: limit, key: braveAPIKey)
        guard !results.isEmpty else { return AgentToolOutput(text: "no results for \"\(query)\"") }
        let text = results.enumerated().map { index, result in
            "\(index + 1). \(result.title)\n   \(result.url)\n   \(result.snippet)"
        }.joined(separator: "\n")
        var seen: [String: String] = [:]
        for result in results where !result.url.isEmpty {
            seen[result.url] = result.title + "\n" + result.snippet
        }
        return AgentToolOutput(text: text, seen: seen)
    }

    static func searchStatusError(_ status: Int, brave: Bool) -> String? {
        guard status != 200 else { return nil }
        if brave, status == 401 || status == 403 {
            return "error: search backend returned HTTP \(status) (Brave API key rejected). "
                + "Fix or clear the key in Settings; do not retry this call."
        }
        if brave, status == 402 {
            return "error: search backend returned HTTP 402 (Brave API needs a paid plan). "
                + "Clear the key in Settings to fall back to DuckDuckGo; do not retry this call."
        }
        return "error: search backend returned HTTP \(status) (rate limited or blocked). "
            + "Do not retry the same query immediately; try a different tool or rephrase once."
    }

    private static func checkStatus(_ response: URLResponse, brave: Bool) throws {
        guard let http = response as? HTTPURLResponse,
              let message = searchStatusError(http.statusCode, brave: brave) else { return }
        throw AgentToolMessage(text: message)
    }

    static func duckDuckGoResults(query: String, limit: Int) async throws -> [SearchResult] {
        var components = URLComponents(string: "https://html.duckduckgo.com/html/")
        components?.queryItems = [URLQueryItem(name: "q", value: query)]
        guard let url = components?.url else { return [] }
        let (data, response) = try await URLSession.shared.data(for: get(url))
        try checkStatus(response, brave: false)
        return Array(parseDuckDuckGo(String(decoding: data, as: UTF8.self)).prefix(limit))
    }

    private struct BravePayload: Decodable {
        struct Web: Decodable {
            struct Item: Decodable {
                var title: String?
                var url: String?
                var description: String?
            }
            var results: [Item]?
        }
        var web: Web?
    }

    static func braveResults(query: String, limit: Int, key: String) async throws -> [SearchResult] {
        var components = URLComponents(string: "https://api.search.brave.com/res/v1/web/search")
        components?.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "count", value: String(limit))
        ]
        guard let url = components?.url else { return [] }
        var request = get(url)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(key, forHTTPHeaderField: "X-Subscription-Token")
        let (data, response) = try await URLSession.shared.data(for: request)
        try checkStatus(response, brave: true)
        let payload = try JSONDecoder().decode(BravePayload.self, from: data)
        return (payload.web?.results ?? []).prefix(limit).map {
            SearchResult(
                title: decodeEntities($0.title ?? ""),
                url: $0.url ?? "",
                snippet: plainText(fromHTML: $0.description ?? ""))
        }
    }

    static let localAddressRefusal = "error: fetch_page refuses local or private addresses"

    static let maximumPageBytes = 5 * 1_048_576

    static let readableContentTypes = ["text/html", "text/plain", "application/xhtml+xml"]

    static func isLocalOrPrivate(_ host: String?) -> Bool {
        guard var bare = host?.lowercased(), !bare.isEmpty else { return true }
        if bare.hasPrefix("["), bare.hasSuffix("]") { bare = String(bare.dropFirst().dropLast()) }
        if bare.contains(":") {
            return bare == "::1" || bare == "::" || bare.hasPrefix("fe80:")
                || bare.hasPrefix("fc") || bare.hasPrefix("fd")
        }
        if bare == "localhost" || bare.hasSuffix(".localhost") || bare.hasSuffix(".local") { return true }
        let parts = bare.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return false }
        let octets = parts.compactMap { Int($0) }
        guard octets.count == 4, octets.allSatisfy({ (0...255).contains($0) }) else { return false }
        switch (octets[0], octets[1]) {
        case (0, _), (127, _), (10, _), (192, 168), (169, 254): return true
        case (172, 16...31): return true
        default: return false
        }
    }

    static func fetchPage(url: URL, budget: Int, query: String = "") async throws -> AgentToolOutput {
        var full: String
        do {
            full = condense(try await AgentPageReader.readableText(url: url))
        } catch let failure as AgentToolMessage where failure.fatal {
            throw failure
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            log("[fetch] webview failed: \(error)")
            full = plainText(fromHTML: try await download(url: url))
        }

        return AgentToolOutput(
            text: pageText(full, query: query, budget: budget), seen: [url.absoluteString: full])
    }

    static func download(url: URL) async throws -> String {
        let (stream, response) = try await URLSession.shared.bytes(for: get(url))
        if let http = response as? HTTPURLResponse, http.statusCode >= 400 {
            throw AgentToolMessage(
                text: "error: HTTP \(http.statusCode) for \(url.absoluteString)", fatal: true)
        }
        guard !isLocalOrPrivate(response.url?.host()) else {
            throw AgentToolMessage(text: localAddressRefusal, fatal: true)
        }
        let type = (response.mimeType ?? "").lowercased()
        guard readableContentTypes.contains(type) else {
            throw AgentToolMessage(
                text: "error: unsupported content type \(type.isEmpty ? "unknown" : type)", fatal: true)
        }
        var bytes: [UInt8] = []
        bytes.reserveCapacity(1 << 16)
        for try await byte in stream {
            bytes.append(byte)
            if bytes.count > maximumPageBytes {
                throw AgentToolMessage(text: "error: page too large", fatal: true)
            }
        }
        return decodeText(Data(bytes), encodingName: response.textEncodingName)
    }

    static func decodeText(_ data: Data, encodingName: String?) -> String {
        guard let name = encodingName, !name.isEmpty else { return String(decoding: data, as: UTF8.self) }
        let identifier = CFStringConvertIANACharSetNameToEncoding(name as CFString)
        guard identifier != kCFStringEncodingInvalidId else {
            return String(decoding: data, as: UTF8.self)
        }
        let encoding = String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(identifier))
        return String(data: data, encoding: encoding) ?? String(decoding: data, as: UTF8.self)
    }

    static func clip(_ text: String, budget: Int) -> String {
        let limit = max(500, budget)
        guard text.count > limit else { return text }
        return String(text.prefix(limit)) + "\n[truncated to \(limit) characters]"
    }

    static let chunkCharacters = 400

    static func pageText(_ full: String, query: String, budget: Int) -> String {
        let wanted = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !wanted.isEmpty else { return clip(full, budget: budget) }
        let pieces = chunks(full)
        guard let selected = selectPassages(pieces, query: wanted, budget: budget) else {
            log("[fetch_page] query \(wanted.debugDescription) matched nothing in \(pieces.count) "
                + "chunks — returning the first \(budget) characters instead")
            return clip(full, budget: budget)
        }
        log("[fetch_page] query \(wanted.debugDescription): \(selected.count) of \(full.count) chars "
            + "from \(pieces.count) chunks")
        return selected
    }

    static func chunks(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        func flush() {
            if !current.isEmpty { out.append(current) }
            current = ""
        }
        func add(_ piece: String) {
            if current.isEmpty {
                current = piece
            } else if current.count + 1 + piece.count <= chunkCharacters {
                current += "\n" + piece
            } else {
                flush()
                current = piece
            }
        }
        for paragraph in text.split(separator: "\n", omittingEmptySubsequences: true) {
            if paragraph.count <= chunkCharacters { add(String(paragraph)); continue }
            for sentence in sentences(String(paragraph)) {
                guard sentence.count > chunkCharacters else { add(sentence); continue }
                flush()
                var rest = Substring(sentence)
                while !rest.isEmpty {
                    out.append(String(rest.prefix(chunkCharacters)))
                    rest = rest.dropFirst(chunkCharacters)
                }
            }
        }
        flush()
        return out
    }

    static func sentences(_ paragraph: String) -> [String] {
        var out: [String] = []
        var current = ""
        for character in paragraph {
            current.append(character)
            if "\u{3002}\u{FF0E}.!?\u{FF01}\u{FF1F}".contains(character) {
                out.append(current)
                current = ""
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    static func selectPassages(_ pieces: [String], query: String, budget: Int) -> String? {
        guard pieces.count > 1 else { return nil }
        let terms = matchTerms(query)
        var ranked = bm25Ranking(pieces, terms: terms)
        if ranked.isEmpty { ranked = substringRanking(pieces, terms: terms) }
        guard !ranked.isEmpty else { return nil }
        var picked: Set<Int> = [0]
        var used = pieces[0].count
        for index in ranked where !picked.contains(index) {
            let cost = pieces[index].count + 1
            guard used + cost <= max(500, budget) else { continue }
            picked.insert(index)
            used += cost
        }
        guard picked.count > 1 else { return nil }
        var out: [String] = []
        var previous = -1
        for index in picked.sorted() {
            if previous >= 0, index != previous + 1 { out.append("…") }
            out.append(pieces[index])
            previous = index
        }
        if previous < pieces.count - 1 { out.append("…") }
        return out.joined(separator: "\n")
    }

    static func matchTerms(_ query: String) -> [String] {
        var terms: [String] = []
        for run in query.split(whereSeparator: { !($0.isLetter || $0.isNumber) }) {
            guard run.count >= 3 else { continue }
            if run.allSatisfy(\.isASCII) { terms.append(String(run)); continue }
            let characters = Array(run)
            for start in 0...(characters.count - 3) {
                terms.append(String(characters[start..<(start + 3)]))
            }
        }
        var seen = Set<String>()
        return Array(terms.filter { seen.insert($0).inserted }.prefix(32))
    }

    static func bm25Ranking(_ pieces: [String], terms: [String]) -> [Int] {
        guard !terms.isEmpty else { return [] }
        let match = terms.map { "\"\($0)\"" }.joined(separator: " OR ")
        var db: OpaquePointer?
        guard sqlite3_open(":memory:", &db) == SQLITE_OK else {
            sqlite3_close(db)
            return []
        }
        defer { sqlite3_close(db) }
        guard sqlite3_exec(
            db, "CREATE VIRTUAL TABLE c USING fts5(t, tokenize='trigram')", nil, nil, nil) == SQLITE_OK
        else {
            log("[fetch_page] FTS5 unavailable — ranking passages by substring instead")
            return []
        }
        var insert: OpaquePointer?
        guard sqlite3_prepare_v2(db, "INSERT INTO c(rowid, t) VALUES (?, ?)", -1, &insert, nil)
            == SQLITE_OK else { return [] }
        for (index, piece) in pieces.enumerated() {
            sqlite3_bind_int64(insert, 1, Int64(index))
            sqlite3_bind_text(insert, 2, piece, -1, sqliteTransient)
            sqlite3_step(insert)
            sqlite3_reset(insert)
        }
        sqlite3_finalize(insert)
        var select: OpaquePointer?
        guard sqlite3_prepare_v2(
            db, "SELECT rowid FROM c WHERE c MATCH ? ORDER BY bm25(c)", -1, &select, nil) == SQLITE_OK
        else { return [] }
        defer { sqlite3_finalize(select) }
        sqlite3_bind_text(select, 1, match, -1, sqliteTransient)
        var ranked: [Int] = []
        while sqlite3_step(select) == SQLITE_ROW { ranked.append(Int(sqlite3_column_int64(select, 0))) }
        return ranked
    }

    static func substringRanking(_ pieces: [String], terms: [String]) -> [Int] {
        guard !terms.isEmpty else { return [] }
        let needles = terms.map { $0.lowercased() }
        var scored: [(index: Int, hits: Int)] = []
        for (index, piece) in pieces.enumerated() {
            let haystack = piece.lowercased()
            var hits = 0
            for needle in needles where haystack.contains(needle) { hits += 1 }
            if hits > 0 { scored.append((index, hits)) }
        }
        scored.sort { $0.hits == $1.hits ? $0.index < $1.index : $0.hits > $1.hits }
        return scored.map(\.index)
    }

    private static let sqliteTransient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    static func log(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    static func defaultNotesDirectory() -> URL {
        ModelStorage.applicationSupportDirectory()
            .appending(path: "agent-notes", directoryHint: .isDirectory)
    }

    private static var homeDirectory: URL { URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true) }

    static func defaultFileRoots(notesDirectory: URL) -> [URL] {
        let home = homeDirectory
        let named = ["Desktop", "Documents", "Downloads"].map {
            home.appending(path: $0, directoryHint: .isDirectory)
        }
        let homePath = comparablePath(home)
        return (named + [notesDirectory])
            .map { $0.standardizedFileURL.resolvingSymlinksInPath() }
            .filter { root in
                let path = comparablePath(root)
                return path != homePath && !homePath.hasPrefix(path == "/" ? "/" : path + "/")
            }
    }

    static func moveNote(
        title: String, destination: String,
        notesDirectory: URL = AgentTools.defaultNotesDirectory(), allowedRoots: [URL]? = nil
    ) throws -> String {
        let source = try noteURL(title: title, in: notesDirectory)
        let folder = try destinationFolder(
            destination, allowedRoots: allowedRoots ?? defaultFileRoots(notesDirectory: notesDirectory))
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let target = folder.appending(path: source.lastPathComponent)
        let targetPath = target.path(percentEncoded: false)
        guard !FileManager.default.fileExists(atPath: targetPath) else {
            throw AgentToolMessage(
                text: "error: \(targetPath) already exists; choose another destination or title")
        }
        try FileManager.default.moveItem(at: source, to: target)
        return targetPath
    }

    private static func noteURL(title: String, in directory: URL) throws -> URL {
        let wanted = title.trimmingCharacters(in: .whitespacesAndNewlines)
        for name in [noteFileName(title), wanted.hasSuffix(".md") ? wanted : wanted + ".md"] {
            let url = directory.appending(path: name)
            guard comparablePath(url.deletingLastPathComponent()) == comparablePath(directory),
                  FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else { continue }
            return url
        }
        let existing = ((try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.contentModificationDateKey])) ?? [])
            .sorted { modificationDate($0) > modificationDate($1) }
            .prefix(10).map(\.lastPathComponent)
        throw AgentToolMessage(text: "error: no note titled \"\(wanted)\". Existing notes: "
            + (existing.isEmpty ? "none" : existing.joined(separator: ", ")))
    }

    private static func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
            ?? .distantPast
    }

    private static func comparablePath(_ url: URL) -> String {
        var path = url.standardizedFileURL.resolvingSymlinksInPath().path(percentEncoded: false)
        while path.count > 1, path.hasSuffix("/") { path.removeLast() }
        return path.lowercased()
    }

    private static func destinationFolder(_ destination: String, allowedRoots: [URL]) throws -> URL {
        let text = (destination.trimmingCharacters(in: .whitespacesAndNewlines) as NSString)
            .expandingTildeInPath
        let url = text.hasPrefix("/")
            ? URL(fileURLWithPath: text, isDirectory: true)
            : homeDirectory.appending(path: text, directoryHint: .isDirectory)
        let resolved = url.standardizedFileURL.resolvingSymlinksInPath()
        let path = comparablePath(resolved)
        let inside = allowedRoots.contains {
            let root = comparablePath($0)
            return path == root || path.hasPrefix(root + "/")
        }
        guard inside else {
            throw AgentToolMessage(
                text: "error: destination must be Desktop, Documents, Downloads or a folder inside them")
        }
        return resolved
    }

    static func writeNote(
        title: String, content: String, directory: URL = AgentTools.defaultNotesDirectory()
    ) throws -> String {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let name = noteFileName(title)
        var url = directory.appending(path: name)
        if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
            let base = String(name.dropLast(3))
            guard let free = (2...99).lazy
                .map({ directory.appending(path: "\(base) \($0).md") })
                .first(where: { !FileManager.default.fileExists(atPath: $0.path(percentEncoded: false)) })
            else {
                throw AgentToolMessage(text: "error: \(name) already exists; choose another title")
            }
            url = free
        }
        let heading = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasHeading = content.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("#")
        let body = (heading.isEmpty || hasHeading ? "" : "# \(heading)\n\n") + content + "\n"
        try Data(body.utf8).write(to: url, options: .atomic)
        return url.path(percentEncoded: false)
    }

    static func noteFileName(_ title: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " -_"))
        let mapped = String(title.unicodeScalars.map { allowed.contains($0) ? Character($0) : "-" })
        let cleaned = mapped.trimmingCharacters(in: .whitespaces)
        return (cleaned.isEmpty ? "note" : String(cleaned.prefix(80))) + ".md"
    }

    private static func get(_ url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        request.timeoutInterval = 20
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        return request
    }

    static func parseDuckDuckGo(_ html: String) -> [SearchResult] {
        let anchorPattern = /<a\b([^>]*)>(.*?)<\/a>/.dotMatchesNewlines()
        let hrefPattern = /href="([^"]*)"/
        var results: [SearchResult] = []
        for match in html.matches(of: anchorPattern) {
            let attributes = String(match.1)
            if attributes.contains("result__a") {
                let href = attributes.firstMatch(of: hrefPattern).map { String($0.1) } ?? ""
                results.append(SearchResult(
                    title: plainText(fromHTML: String(match.2)),
                    url: resolveRedirect(href), snippet: ""))
            } else if attributes.contains("result__snippet"), let last = results.indices.last,
                      results[last].snippet.isEmpty {
                results[last].snippet = plainText(fromHTML: String(match.2))
            }
        }
        return results
    }

    static func resolveRedirect(_ href: String) -> String {
        let absolute = href.hasPrefix("//") ? "https:" + href : href
        guard absolute.contains("uddg="), let components = URLComponents(string: decodeEntities(absolute)),
              let target = components.queryItems?.first(where: { $0.name == "uddg" })?.value else {
            return absolute
        }
        return target
    }

    static func plainText(fromHTML html: String) -> String {
        var text = html.replacing(/<!--.*?-->/.dotMatchesNewlines(), with: " ")
        text = text.replacing(/<script\b.*?<\/script>/.dotMatchesNewlines().ignoresCase(), with: " ")
        text = text.replacing(/<style\b.*?<\/style>/.dotMatchesNewlines().ignoresCase(), with: " ")
        text = text.replacing(/<noscript\b.*?<\/noscript>/.dotMatchesNewlines().ignoresCase(), with: " ")
        text = text.replacing(/<(br|\/p|\/div|\/h[1-6]|\/li|\/tr|\/table)\b[^>]*>/.ignoresCase(), with: "\n")
        text = text.replacing(/<[^>]*>/.dotMatchesNewlines(), with: " ")
        return condense(decodeEntities(text))
    }

    static func condense(_ text: String) -> String {
        text.split(separator: "\n")
            .map { $0.split(whereSeparator: \.isWhitespace).joined(separator: " ") }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    static let namedEntities = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": " ", "hellip": "…",
        "mdash": "—", "ndash": "–", "rsquo": "’", "lsquo": "‘", "ldquo": "“", "rdquo": "”"
    ]

    static func decodeEntities(_ text: String) -> String {
        guard text.contains("&") else { return text }
        return text.replacing(/&(#[xX]?[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);/) { match in
            let body = match.1
            guard body.hasPrefix("#") else { return namedEntities[String(body)] ?? String(match.0) }
            let hex = body.hasPrefix("#x") || body.hasPrefix("#X")
            let digits = body.dropFirst(hex ? 2 : 1)
            guard let code = UInt32(digits, radix: hex ? 16 : 10), let scalar = Unicode.Scalar(code) else {
                return String(match.0)
            }
            return String(Character(scalar))
        }
    }
}

@MainActor
final class AgentPageReader: NSObject, WKNavigationDelegate {

    static let timeout: Double = 20

    static let settle: Duration = .milliseconds(1500)

    private static let script = """
        (() => {
          const body = document.body ? document.body.innerText || '' : '';
          let best = '';
          for (const el of document.querySelectorAll('article, main, [role=main]')) {
            const text = el.innerText || '';
            if (text.length > best.length) best = text;
          }
          return best.length > body.length / 2 ? best : body;
        })()
        """

    static let timedOutMessage = AgentToolMessage(text: "error: page load timed out", fatal: true)

    private var navigation: CheckedContinuation<Void, Error>?
    private var evaluation: CheckedContinuation<String?, Error>?
    private var view: WKWebView?
    private var statusFailure: AgentToolMessage?

    static func readableText(url: URL) async throws -> String {
        try await AgentPageReader().text(from: url)
    }

    private func text(from url: URL) async throws -> String {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        let view = WKWebView(
            frame: CGRect(x: 0, y: 0, width: 1280, height: 900), configuration: configuration)
        self.view = view
        view.navigationDelegate = self
        view.customUserAgent = AgentTools.userAgent
        defer {
            view.stopLoading()
            view.navigationDelegate = nil
            self.view = nil
        }

        let watchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(Self.timeout))
            guard !Task.isCancelled else { return }
            self?.abort(with: Self.timedOutMessage)
        }
        defer { watchdog.cancel() }

        return try await withTaskCancellationHandler {
            do {
                try await withCheckedThrowingContinuation { (waiter: CheckedContinuation<Void, Error>) in
                    navigation = waiter
                    view.load(URLRequest(url: url, timeoutInterval: Self.timeout))
                }
            } catch {
                throw statusFailure ?? error
            }
            if let statusFailure { throw statusFailure }
            if let final = view.url, AgentTools.isLocalOrPrivate(final.host()) {
                throw AgentToolMessage(text: AgentTools.localAddressRefusal, fatal: true)
            }
            try await Task.sleep(for: Self.settle)
            let value = try await withCheckedThrowingContinuation {
                (waiter: CheckedContinuation<String?, Error>) in
                evaluation = waiter
                view.evaluateJavaScript(Self.script, completionHandler: { [weak self] value, error in
                    guard let self else { return }
                    if let error {
                        self.finishEvaluation(.failure(error))
                    } else {
                        self.finishEvaluation(.success(value as? String))
                    }
                })
            }
            guard let text = value,
                  !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AgentToolMessage(text: "fetch_page: the web view found no text")
            }
            return text
        } onCancel: {
            Task { @MainActor in self.abort(with: CancellationError()) }
        }
    }

    private func abort(with error: Error) {
        view?.stopLoading()
        finish(.failure(error))
        finishEvaluation(.failure(error))
    }

    private func finish(_ result: Result<Void, Error>) {
        guard let waiter = navigation else { return }
        navigation = nil
        waiter.resume(with: result)
    }

    private func finishEvaluation(_ result: Result<String?, Error>) {
        guard let waiter = evaluation else { return }
        evaluation = nil
        waiter.resume(with: result)
    }

    func webView(
        _ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping @MainActor (WKNavigationResponsePolicy) -> Void
    ) {
        if let http = navigationResponse.response as? HTTPURLResponse, http.statusCode >= 400 {
            statusFailure = AgentToolMessage(
                text: "error: HTTP \(http.statusCode) for "
                    + (navigationResponse.response.url?.absoluteString ?? "the page"),
                fatal: true)
            decisionHandler(.cancel)
            finish(.success(()))
            return
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        finish(.success(()))
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        finish(.failure(error))
    }

    func webView(
        _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error
    ) {
        finish(.failure(error))
    }
}
