import Foundation
import LLMCore
import Testing

// Everything the RAG demo does before the model sees anything: splitting, ranking, the token
// budget, the prompt and the dates. No network, no model — `swift test --filter RAG` runs it all.

/// Stands in for the tokenizer: one token per two characters is close enough for Japanese to make
/// the budget assertions mean something, and it is deterministic.
private func fakeCount(_ text: String) -> Int { max(1, text.count / 2) }

@Suite("RAG context")
struct RAGContextSuite {

    // MARK: splitting

    @Test("the splitter keeps pieces under the size and overlaps them")
    func chunkBoundaries() {
        let text = (1...40).map { "これは\($0)番目の文です。" }.joined()
        let pieces = RAGContext.chunks(text, size: 120, overlap: 30)
        #expect(pieces.count > 1)
        for piece in pieces { #expect(piece.count <= 120 + 30) }
        // Every piece after the first repeats the tail of the one before it.
        for (previous, next) in zip(pieces, pieces.dropFirst()) {
            let head = next.split(separator: "\n").first.map(String.init) ?? next
            #expect(previous.hasSuffix(head), "\(head.debugDescription) is not the tail of the previous piece")
        }
        // Nothing is lost: every sentence survives in some piece.
        for number in [1, 17, 40] {
            #expect(pieces.contains { $0.contains("これは\(number)番目の文です。") })
        }
    }

    @Test("a paragraph with no sentence breaks is cut at the size")
    func hardSplit() {
        let pieces = RAGContext.chunks(String(repeating: "あ", count: 250), size: 100, overlap: 20)
        #expect(pieces.count == 3)
        #expect(pieces[0].count == 100)
    }

    @Test("an empty document yields no passages")
    func emptySplit() {
        #expect(RAGContext.chunks("").isEmpty)
    }

    // MARK: budget

    @Test("packing never exceeds the token budget and keeps the best first")
    func packStaysUnderBudget() {
        let sources = (1...12).map {
            RAGSource(title: "出典\($0)", url: "https://example.com/\($0)", date: "2026-09-0\($0 % 9 + 1)",
                      text: String(repeating: "本文", count: 100))
        }
        let packed = RAGContext.pack(sources, budget: 900, count: fakeCount)
        #expect(!packed.isEmpty)
        #expect(packed.count < sources.count)
        #expect(packed.first == sources.first)
        let used = packed.map { fakeCount(RAGContext.measuredBlock($0)) + 2 }.reduce(0, +)
        #expect(used <= 900)
    }

    @Test("a single passage larger than the budget is dropped, not truncated into the prompt")
    func oversizedPassageDropped() {
        let huge = RAGSource(title: "長い", url: "https://example.com/a", text: String(repeating: "字", count: 5_000))
        let small = RAGSource(title: "短い", url: "https://example.com/b", text: "短い本文")
        let packed = RAGContext.pack([huge, small], budget: 900, count: fakeCount)
        #expect(packed == [small])
    }

    // MARK: prompt and citations

    @Test("the prompt numbers the sources and the citations match those numbers")
    func promptAndCitations() {
        let sources = [
            RAGSource(title: "点検手順書", url: "tenken.md", date: "2026-04-01", text: "月次点検は毎月1日に行う。"),
            RAGSource(title: "ニュース", url: "https://example.com/n", text: "本文")
        ]
        let prompt = RAGContext.prompt(question: "月次点検はいつ?", sources: sources, today: "2026-09-24")
        #expect(prompt.hasPrefix(RAGContext.instruction(today: "2026-09-24")))
        #expect(prompt.contains("資料[1] 点検手順書 / tenken.md / 2026-04-01"))
        #expect(prompt.contains("資料[2] ニュース / https://example.com/n / 日付不明"))
        #expect(prompt.hasSuffix("質問: 月次点検はいつ?"))

        let citations = RAGContext.citations(sources)
        #expect(citations == [
            "[1] 点検手順書 — tenken.md (2026-04-01)",
            "[2] ニュース — https://example.com/n (日付不明)"
        ])
    }

    @Test("the instruction is byte-identical between turns so the KV prefix is reused")
    func instructionIsStable() {
        let a = RAGContext.prompt(question: "質問A", sources: [], today: "2026-09-24")
        let b = RAGContext.prompt(question: "まったく別の質問B", sources: [], today: "2026-09-24")
        let shared = zip(a, b).prefix { $0 == $1 }.count
        #expect(shared >= RAGContext.instruction(today: "2026-09-24").count)
    }

    // MARK: local documents

    @Test("two Japanese documents are indexed and the right one comes back")
    func localDocumentsSearch() throws {
        let directory = URL(filePath: NSTemporaryDirectory())
            .appending(path: "rag-test-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try Data("""
            # 発電機 点検手順書 2026-04-01
            月次点検では冷却水の量と油圧を確認する。冷却水が下限線を下回っていたら補充する。
            """.utf8).write(to: directory.appending(path: "tenken.md"))
        try Data("""
            # 社内ネットワーク 利用規約
            無線LANのパスワードは半年ごとに変更する。来客用ネットワークは業務端末に使わない。
            """.utf8).write(to: directory.appending(path: "network.txt"))
        // Not an indexed extension — it must be ignored.
        try Data("noise".utf8).write(to: directory.appending(path: "notes.json"))

        let passages = RAGDocuments.load(directory: directory)
        #expect(passages.count == 2)
        #expect(Set(passages.map(\.url)) == ["tenken.md", "network.txt"])
        #expect(passages.first { $0.url == "tenken.md" }?.date == "2026-04-01")
        #expect(passages.first { $0.url == "network.txt" }?.date == "")

        let selected = RAGContext.select(passages, question: "冷却水はどうする?", count: fakeCount)
        #expect(selected.first?.url == "tenken.md")
        #expect(selected.first?.text.contains("冷却水") == true)
    }

    @Test("a question that matches nothing still returns passages, just unranked")
    func rankingFallsThrough() {
        let passages = [RAGSource(title: "a", url: "a", text: "まったく関係のない本文")]
        #expect(RAGContext.rank(passages.map(\.text), query: "xyzzy").isEmpty)
        #expect(RAGContext.select(passages, question: "xyzzy", count: fakeCount) == passages)
    }

    // MARK: dates

    @Test("the published date comes from the meta tag, the time tag, then the text")
    func dateExtraction() {
        let meta = """
            <html><head><meta charset="utf-8">
            <meta property="article:published_time" content="2026-09-22T08:15:00+09:00">
            <time datetime="2020-01-01">ignored</time></head><body>2019/05/05</body></html>
            """
        #expect(RAGContext.publishedDate(html: meta) == "2026-09-22")

        let time = "<article><time datetime='2026-09-23T10:00:00Z'>きのう</time><p>本文</p></article>"
        #expect(RAGContext.publishedDate(html: time) == "2026-09-23")

        // No meta, no time tag: the caller falls back to the snippet / visible text tier.
        let plain = "<div class='byline'>2026年9月7日 配信</div><p>本文</p>"
        #expect(RAGContext.publishedDate(html: plain) == nil)
        #expect(RAGContext.isoDate(in: "2026年9月7日 配信 本文") == "2026-09-07")

        #expect(RAGContext.publishedDate(html: "<p>日付のない本文</p>") == nil)
        // A meta tag whose name is not a publication date must not be taken.
        #expect(RAGContext.publishedDate(html: "<meta name='author' content='2026-01-01'>") == nil)
    }

    @Test("dates in search snippets are normalised")
    func snippetDates() {
        #expect(RAGContext.isoDate(in: "2026/9/5 — 記事の要約") == "2026-09-05")
        #expect(RAGContext.isoDate(in: "2026年12月31日") == "2026-12-31")
        #expect(RAGContext.isoDate(in: "2026-13-01") == nil)
        #expect(RAGContext.isoDate(in: "no date here") == nil)
    }
}
