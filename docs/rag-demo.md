# RAG demo — answering from the web and from documents on the device

The Chat demo can answer from material it fetched itself: pages it found on the web, and files you
put in `Documents/rag/`. It cites what it used, with the date the page carries, so an answer about
something that happened this week is checkable rather than remembered.

The model is Gemma 4 E2B (`gemma-4-e2b-round5-vd`), a 2 B model with a **2048-token** window. That
number is the whole design.

## How it works

The app retrieves; the model never calls a tool. A 2 B model is not reliable at emitting tool calls,
and at 2048 tokens there is no room for a call-and-return round trip anyway. So the pipeline runs
in the app and the model sees one finished prompt:

```
question
  ├─ web on?   web_search (top 5) → read the top 3 pages in parallel
  │              → each page: 1500 characters around the question → ~400-char passages
  └─ docs on?  Documents/rag/*.{txt,md,pdf} → ~500-char passages, 100-char overlap
                        ↓
        rank every passage against the question (SQLite FTS5 / BM25)
                        ↓
        fill 900 tokens from the top  →  prompt (~1000 tokens)  →  generate
```

The prompt is a fixed instruction block, then `資料[1] title / url / date` + passage for each source,
then the question. The instruction is byte-identical on every turn of a day, so the engine's KV
prefix match covers it: the second and later RAG turns reprefill only the passages and the question.
The self-test log shows this as `prompt 1026 (+reused 104)`.

RAG turns are not added to the chat history — the prompt already carries everything, and keeping the
instruction at position 0 is what makes the reuse work.

### Dates

A page's date comes from `<meta property="article:published_time">` (and the other usual meta
names), then `<time datetime>`, then a date in the search snippet, then a date in the first 600
characters of the visible text. Nothing found means the source is labelled `日付不明`. Raw HTML is
never scanned wholesale for a date — that picks up script and attribute noise.

Today's date goes into the instruction so the model dates things from the sources instead of from
what it half-remembers about when "now" is.

### Splitting

Documents split on paragraphs and sentences up to ~500 characters, and each piece repeats the last
100 characters of the one before it, so a fact that straddles a boundary still sits whole in one of
them. Web pages are cut to the part that answers the question first (`AgentTools.pageText`), then
split into ~400-character passages so three pages can each contribute their best bit.

### When something fails

No network, a search that returns nothing, a page that will not load — each becomes one line under
the answer, and the remaining sources are used anyway. With no sources at all the app answers
`資料が取得できませんでした。` and does not run the model.

## Using it

**In the app.** Turn on **RAG** in the chat status bar, then pick **Web**, **Docs**, or both. The
answer carries a `出典` list numbered the same as the `[1]` marks in the text. The status line shows
`sources N | source tok N | prompt(rag) N tok | retrieval N.Ns` next to the usual TTFT and tok/s.

**Documents.** Put `.txt`, `.md` or `.pdf` files in `Documents/rag/`. On macOS that is
`~/Documents/rag/`; on iOS it is the app's Documents folder, reachable from the Files app or with

```bash
xcrun devicectl device copy to --device <UDID> --domain-type appDataContainer \
  --domain-identifier com.coreml-llm-samples.DemoApp \
  --source ./手順書.md --destination Documents/rag/手順書.md
```

PDFs are read with PDFKit. The folder is re-read on every question, so an edited file is picked up
straight away; there is no stored index to rebuild.

**Search.** DuckDuckGo by default. Setting a Brave Search API key in the Agent demo's settings
switches this demo over too — it is the same key.

**From the command line.**

```bash
DemoApp --selftest --model <bundle> --max-tokens 128 \
  --rag "質問1|質問2|質問3" [--rag-web off] [--rag-local off]
```

Each turn logs the search and page counts, every source with its URL and date, the token counts and
the answer:

```
[rag] turn=0 search 5 / pages 3 / passages 17 → 4 sources, 844 tokens
[rag] turn=0 source [1] Claude Opus 5.5 のリリース日と変更点 | EvoLink — https://evolink.ai/... (2026-09-22)
[rag] turn=0 answer=Claude Opus 5.5 は 2026 年 9 月 22 日にリリースされました [1, 2, 3]。
[selftest] TTFT 137.94s | 11.6 tok/s | prompt 973 | ... | sources 4 | source tok 844 | prompt(rag) 964 tok | retrieval 1.7s
```

## Limits

- **2048 tokens.** Sources get 900, the whole prompt is kept near 1500, which leaves ~500 to answer
  in. That is roughly three to four passages. A question needing a wide survey will not fit.
- **The first RAG turn is expensive.** A ~1000-token prompt is the first thing that reaches the
  512-wide prefill function, and specializing it costs 138 s on an iPhone 15 from cold. Later turns
  reuse the instruction prefix and land at 7–8 s. On a Mac the same turns are 3–4 s.
- **A 2 B model stays close to the text.** It quotes and it cites, but it does not reconcile sources
  that disagree, and asked about something no source covers it will say so and then often add a
  neighbouring fact from the material anyway.
- **No JavaScript, usually.** Pages are read as raw HTML. One that ships an empty shell is
  re-read through a web view, which is slower.
- **The retrieval order is lexical.** FTS5 trigram matching finds Japanese and English without a
  word splitter, but it matches words, not meaning: a question phrased differently from the source
  ranks it lower.

## Where the code is

| | |
|---|---|
| `Sources/LLMCore/RAGContext.swift` | splitting, ranking, the token budget, the prompt, dates — no network, no files, no UI |
| `Examples/DemoApp/DemoApp/Demos/Chat/ChatRAG.swift` | retrieval, the local folder, the two switches and the source footer |
| `Examples/DemoApp/DemoApp/Demos/Chat/ChatViewModel.swift` | `consumeRAGTurn` — retrieve, then generate |
| `Tests/CoreMLBackendTests/RAGContextTests.swift` | `swift test --filter RAG` |

Search and page reading are the Agent demo's `AgentTools` (`docs/agent-demo.md` §2–3), reused as-is.
