# The Agent demo

`DemoApp`'s **Agent** screen runs a tool-calling loop against a Core ML bundle on the machine in front
of you. Nothing about the loop is remote: the model, the search fetch, the page reader, and the notes
all live on this device. The README has the short version; this is the contract.

The demo needs a bundle whose chat template is ChatML — in this catalog, the **Qwen3.8 27B Agent**
bundle (macOS, 16,384-token context). Getting it:
[okayuji/Qwen3.8-27B-coreml-agent](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent), in-app
from the **Models** screen or with
`hf download okayuji/Qwen3.8-27B-coreml-agent --local-dir ./models/qwen38-27b-agent`.

---

## 1. The loop

One step is: render the conversation into the ChatML prompt, generate, parse the reply.

- If the reply contains no `<tool_call>` block, that reply is the answer and the task ends.
- If it contains one, the app runs the tool, appends the result to the conversation as a
  `<tool_response>`, and generates again. Up to **3** calls in one reply run concurrently; a fourth is
  not run and comes back as `ignored: max 3 tool calls per turn`.
- Results are appended **in call order**, not completion order, because the chat template folds
  consecutive tool results into one user turn and the KV cache is matched byte for byte against what
  was rendered last step.
- The loop stops at **Max steps** with a "Step limit reached" message if the model never answers.

The system prompt carries today's date as `yyyy-MM-dd (EEE)` in the device's time zone, with an
English weekday name whatever the user's locale is, because the chat template has no notion of today
and "tomorrow's weather" would otherwise be searched against a date the model invented.

A follow-up question continues the same conversation, so the KV cache from the previous turn is
reused; **New Task** clears the conversation and resets the engine.

## 2. Tool contracts

Every tool returns one plain-text block that goes straight to the model. A failure is a single line
starting with `error: ` describing what went wrong — the system prompt tells the model to say why in
one sentence and retry once with a different query or tool, never to repeat the failing call verbatim.

| Tool | Arguments | Returns |
|---|---|---|
| `web_search` | `query` (required), `max_results` (1–10, default 5) | title, url and snippet per result |
| `fetch_page` | `url` (required, absolute http/https), `query` (optional) | the page's readable text, truncated to **Page budget** characters |
| `write_note` | `title`, `content` (both required) | `saved to <path>` |
| `move_note` | `title`, `destination` (both required) | `moved to <path>` |

- `fetch_page` renders the page in a `WKWebView` and takes its text, so pages that build their body in
  JavaScript still read. It refuses local and private addresses. Without `query` it returns the start
  of the page; with `query` it returns the passages that match it, which is usually what you want,
  because the budget is small next to a real article.
- `write_note` saves Markdown to the notes folder. The file name comes from the title with anything
  outside `[A-Za-z0-9 _-]` replaced by `-` and capped at 80 characters, plus `.md`; the title becomes
  an `# H1` unless the content already starts with a heading. A title whose file already exists is
  saved as `<name> 2.md`, `<name> 3.md` and so on rather than overwriting it or being refused, so a
  repeated title costs the agent no steps. The tool spec asks the model to keep a note under about
  2,000 characters.
- `move_note` exists only while **Allow file operations** is on: with the toggle off, the tool is not
  in the prompt at all, and a call to it is refused rather than run. It moves a note the agent itself
  wrote, and only into `~/Desktop`, `~/Documents`, `~/Downloads`, the notes folder, or a folder under
  one of those. It never overwrites an existing file and never deletes anything.

## 3. Settings

| Setting | Default | What it changes |
|---|---|---|
| Reasoning effort | `Low` | `Off` skips the thinking block entirely and is the fastest; `Low` asks for brief thinking; `Medium` adds no instruction, which is the chat template's own default; `Xhigh` asks the model to check its assumptions. |
| Max steps | 8 | The step ceiling, a slider from 1 to 16. A search-then-read task usually finishes in 2–4 steps; a measured weather-then-note task used 7 of 8 — two searches, three page reads, the note, and the answer. |
| Page budget | 10,000 | How many characters `fetch_page` returns per page, a slider from 500 to 20,000 in steps of 500. |
| Brave Search API key | empty | Empty means DuckDuckGo. Stored in `UserDefaults` in the clear. |
| Allow file operations | off | Adds `move_note`, as above. |
| Notes folder (macOS) | app folder | Where `write_note` saves and `move_note` looks. |

**Page budget is the setting that costs time.** Whatever a page returns is prompt on the next step and
has to be prefilled. In one measured session a 9,928-character page became 2,231 new prompt tokens and
27.7 s of prefill. Raising the budget buys the model more of the page and pays for it on every
subsequent step of the task; lowering it makes each step cheaper and the passage selection matter more.

Changing any setting mid-task rewrites the system prompt, so the next generation cannot reuse the KV
cache and prefills the whole conversation.

## 4. Headless

```bash
DemoApp --agent-e2e --model <bundle> --task "…" \
  [--effort off|low|medium|xhigh] [--max-steps N] [--page-budget N] \
  [--max-tokens N] [--deadline SECONDS] [--file-ops] [--notes-dir <path>]
```

It prints each entry as `--- <kind> <title>` followed by its text, `[e2e] …` progress lines, and one
verdict line: `AGENT E2E COMPLETED  diverges=N  capSteps=N`, or `AGENT E2E INCOMPLETE` with exit 1.
The line before the verdict, `[e2e] outcome: <word>`, says which ending it was: `answered` (the only
one that counts as complete), `step-limit`, `overflow`, `cap` (the answer hit the token limit),
`stopped` (the `--deadline` fired or the run was stopped), or `failed`.
`diverges` counts steps whose re-rendered prompt did not match the previous one byte for byte (each
costs a full re-prefill), and `capSteps` counts steps that stopped at the token cap, which loses
whatever the model was writing — an unclosed `<tool_call>` is dropped, so a run can "finish" without
having done the work. Read both before trusting the verdict.

```bash
DemoApp --agent-selftest [--offline]
```

Checks prompt rendering against golden strings, the tool-call parser, the note and move rules, the
transcript format, and the search and page parsers. It prints `PASS  <name>` / `FAIL  <name>` /
`SKIP  <name>` per case and ends with `AGENT SELFTEST PASS` or `AGENT SELFTEST FAIL (n)`. `--offline`
skips the cases that need the network. Neither entry point opens a window.

## 5. Logs

Two files, both plain text:

**`~/Documents/metrics.jsonl`** — one JSON object per line. Agent rows carry `agentSession`,
`agentTurn` and `agentStep`, so a task's rows group by session and order by step.

- `kind` `message`, one per generation: `promptTokens`, `reusedTokens` (how much of the prompt the KV
  cache covered), `generatedTokens`, `ttftSeconds`, `decodeTokPerSec`, `finishReason`,
  `peakFootprintMB`, `agentEffort`, `agentToolCalls`, `agentRerenderDiverged`, `agentThinkingChars`,
  `agentAnswerChars`, and, when the step answered, `citationsTotal` / `citationsSeen` /
  `citationMeanOverlap` — how many URLs the answer cited and how many of those the task had actually
  retrieved.
- `kind` `agent-tool`, one per tool call: `toolName`, `toolArg` (the url, query or title, first 200
  characters), `toolSeconds`, `toolResultChars`, `toolError`.

**`~/Library/Application Support/DemoApp/agent-transcripts/<session>.md`** — the whole task, rewritten
at the end of every turn. A header of model, session, effort, max steps, page budget, search provider,
date and app build; then `## You` and `## Assistant` for the conversation, `### Thinking (n chars)`,
`### Calling <tool>` and `### <tool> result (n chars)` for the machinery, each in a four-backtick
fence so a tool result containing ``` still round-trips. Nothing is truncated. The **Copy** button in
the header puts the same text on the clipboard.

Notes go to `~/Library/Application Support/DemoApp/agent-notes/` unless another folder is set.

## 6. Known limits

- **A hybrid bundle cannot rewind its KV cache.** Its recurrent state moves forward only, so any change
  to a prefix already fed — a settings change, a **Stop**, a re-render that comes out one byte
  different — means the next generation prefills the whole conversation instead of resuming. It is
  correct, just slow: a full re-prefill of a long conversation has been measured at 62 s.
- **The context is the ceiling on how much page text a task can hold.** At 16,384 tokens and roughly
  2,200 tokens per 10,000-character page, two or three large pages plus the conversation is the
  practical limit. Past that the app drops the oldest tool result and retries, as many times as it
  takes to fit — the two most recent tool turns are never dropped, because they are what the model is
  working on, and the question and every assistant turn are never touched at all. When there is
  nothing left to drop it stops with "Context window is full".
- **Search is a scraper unless you supply a Brave key.** The DuckDuckGo path parses an HTML page, so it
  breaks when that page changes, and it is rate-limited by whoever is on the other end.

What the loop was measured doing — twelve fixed tasks, 48 runs, the grader and the per-run tables — is
recorded in [results/2026-09-13-qwen38-agent-benchmark.md](results/2026-09-13-qwen38-agent-benchmark.md).
