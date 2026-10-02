# Core ML LLM Samples

Core ML–native conversions of open LLMs for **Apple Silicon**. Every model ships with a shared Swift
runtime you can clone and run, every benchmark is quoted with its measurement conditions and its
source, and every conversion is gated **bit-exact** against its reference implementation. That
verification discipline is the house style — the receipts are in each model card.

## Maintainer

**岡優志（おかゆうじ / okayuji）** — iOS / on-device AI / Core ML engineer.

- GitHub: [oka-yuji](https://github.com/oka-yuji)
- Hugging Face: [okayuji](https://huggingface.co/okayuji)
- Profile: [岡優志 / okayuji](https://www.tukuru-app.com/okayuji.html)
- Zenn: [oka_yuuji](https://zenn.dev/oka_yuuji)
- X: [oka_yuuji](https://x.com/oka_yuuji)
- Company: [株式会社Tukuru](https://www.tukuru-app.com/)

[日本語版 →](README.ja.md)

---

## Models

| Model | Size | Context | Speed | HF | Article | Demo | License |
|---|---|---|---|---|---|---|---|
| [Gemma 4 12B IT — 128K Context Ladder](samples/gemma-4-12b-128k.md) | 6.7 GB (int4) | 131,072 | ~11 tok/s (32K mode, M4 Max) | [okayuji/gemma-4-12b-it-coreml-128k](https://huggingface.co/okayuji/gemma-4-12b-it-coreml-128k) | [article](https://medium.com/@yu.j.0513/running-gemma-4-12b-with-a-128k-context-on-core-ml-23d6918dd370) | [demo](https://x.com/oka_yuuji/status/2080660675333161154/video/1) | Apache-2.0 |
| [Gemma 4 E2B IT — ANE Speculative (iOS · macOS)](docs/e2b-speculative-device.md) | ~4.9 GB (pal6+int8) | 2,048 | ~12 tok/s (iPhone 15) · ~16 tok/s (17 Pro) | [okayuji/Gemma-4-E2B-it-coreml-speculative](https://huggingface.co/okayuji/Gemma-4-E2B-it-coreml-speculative) | — | — | Apache-2.0 |
| [Gemma 4 E4B IT — Mac GPU Speculative](docs/e4b-speculative-mac.md) | 6.5 GB (int4+int8) | 2,048 | ~31 tok/s (M4 Max GPU) | [okayuji/Gemma-4-E4B-it-coreml-speculative](https://huggingface.co/okayuji/Gemma-4-E4B-it-coreml-speculative) | — | — | Apache-2.0 |
| [Qwen3.8 27B Agent](samples/qwen38-27b-agent.md) | 30 GB (int8) | 16,384 | 16.7 tok/s (M4 Max) | [okayuji/Qwen3.8-27B-coreml-agent](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent) | coming soon | — | Apache-2.0 |

> **Speed** is one representative figure; the full measurement conditions and every number live in
> each model's card.
>
> **Gemma 4 E2B** ships lossless prompt-lookup speculation verified **byte-identical on an iPhone 15
> (A16) Neural Engine**, plus cross-machine KV restore — details and the memory ledger are in its card.

### Modalities

Text is the baseline for every model. Image and audio support arrives as separate encoders inside the
same Hugging Face repo, so downloading a bundle brings whatever that bundle supports. **Download**
below is the full size with every encoder included.

| Model | Download | Text | Image in chat | Audio in chat | Live Camera |
|---|---|---|---|---|---|
| Gemma 4 12B IT — 128K | 10.2 GB | macOS | — | — | — |
| Gemma 4 E2B Speculative | 5.9 GB | iOS · macOS | iOS · macOS | iOS · macOS | iOS · macOS |
| Gemma 4 E4B Speculative | 6.8 GB | macOS | macOS | — | macOS |
| Qwen3.8 27B Agent | [30 GB](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent) | macOS | — | — | — |

E4B is macOS-only throughout: its language weights alone exceed an iPhone's memory budget. Its image
encoder is the same tower as E2B's — **658 of 658 tensors bit-identical** — differing only in the
projection width that feeds the language model, so each bundle carries its own copy and the app
refuses a mismatched one. Audio is E2B-only; there is no E4B audio encoder. The **Qwen3.8 27B Agent**
bundle is text-only and macOS-only — see **Agent** below. **Live Camera** is a demo screen on iOS; on
macOS the same loop runs headlessly (`--live-selftest`).

---

## Quick start

The featured model is **Gemma 4 12B IT — 128K Context Ladder** (see its
[model card](samples/gemma-4-12b-128k.md) for benchmarks, requirements, and limitations).

**Requirements:** an Apple Silicon Mac, macOS 26+, and the Swift 6.2 toolchain (Xcode 26).

```bash
# 1. Clone
git clone https://github.com/oka-yuji/coreml-llm-samples.git
cd coreml-llm-samples

# 2. Download the model bundle (~11 GB)
./scripts/download-model.sh
# → ./models/gemma-4-12b-it-coreml-128k

# 3. Chat
swift run -c release corellm-chat --model ./models/gemma-4-12b-it-coreml-128k --stats
```

### Or open the Xcode project

Prefer a GUI? Open `Examples/DemoApp/DemoApp.xcodeproj` in Xcode and press Run. `DemoApp` is a demo
list — a sidebar of demos with the selected one shown on the right. On macOS the demos are **Chat**,
**Agent**, and **Models**; **Live Camera** is shown on iOS only. Chat is selected on launch, and more
models and modalities each add a row and a screen here.

The **Models** screen downloads model bundles in-app from Hugging Face, with progress, cancel, and
delete, and loads a finished bundle straight into Chat. Every model it can download lives in a public
repo, so the download runs anonymously — no Hugging Face token or sign-in is needed.

The **Chat** demo is the conversation screen. Select a model in **Models** and press **Load in Chat**
to switch here and talk. Responses stream as they generate, and the status line reports the last
turn's tokens/second and time-to-first-token. The first reply of a run pays the one-time GPU
specialization cost described under **First run is slow** in the
[model card](samples/gemma-4-12b-128k.md); later replies are fast.

Chat also takes attachments when the loaded bundle carries the matching encoder: attach an image and
ask about it, or record a clip and have it transcribed. The composer only offers what the bundle
supports, so the buttons are absent rather than broken. An image costs 256 of the 2,048 context
tokens and a 30-second recording costs up to 750, which is why an attachment turn is best kept short.

The **Agent** demo hands the model three tools — `web_search`, `fetch_page`, and `write_note` — and
lets it work in steps: it generates, and when the reply contains a tool call the app runs the tool,
appends the result to the conversation, and generates again, until it answers without calling a tool.
[docs/agent-demo.md](docs/agent-demo.md) has the loop, the tool contracts, the headless driver, and
the known limits.

It needs a bundle whose chat template is ChatML; in this catalog that is the **Qwen3.8 27B Agent**
bundle, macOS only. Its peak resident footprint is about 56 GB — 55.7–56.9 GB across 48 headless runs
and 56.4 GB in a GUI session, both on an M4 Max with 128 GB — so a 64 GB Mac is not enough. Getting
the bundle: [okayuji/Qwen3.8-27B-coreml-agent](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent),
in-app from the **Models** screen or from the command line:

```bash
hf download okayuji/Qwen3.8-27B-coreml-agent --local-dir ./models/qwen38-27b-agent
```

Its card, [samples/qwen38-27b-agent.md](samples/qwen38-27b-agent.md), has the benchmarks and the
requirements; the measurement behind them — twelve tasks, 48 runs, the grader, the per-run tables — is
in [docs/results/2026-09-13-qwen38-agent-benchmark.md](docs/results/2026-09-13-qwen38-agent-benchmark.md).

The settings above the composer:

- **Reasoning effort** — `Off` skips thinking entirely, `Low` (the default) asks for brief thinking,
  `Medium` adds no instruction at all, `Xhigh` asks it to check its assumptions. Off is fastest.
- **Max steps** — how many generate-and-run rounds before the agent gives up. Default 8.
- **Page budget** — how many characters `fetch_page` returns per page. Default 10,000. A bigger budget
  is a longer prompt on the next step, so it is paid for in prefill time.
- **Brave Search API key** — optional. Empty means DuckDuckGo. It is kept in `UserDefaults` as plain
  text, so treat it as a local convenience, not a secret store.
- **Allow file operations** — off by default. On, it adds a fourth tool, `move_note`, which moves a
  note the agent itself wrote into Desktop, Documents, Downloads, or a folder under one of them. It
  never overwrites and never deletes, and with the toggle off the tool is absent from the prompt.
- **Notes folder** (macOS) — where `write_note` saves. Empty means the app's own folder. A title that
  already has a note is saved beside it as `<name> 2.md` rather than overwriting it.

Every step appends a row to `~/Documents/metrics.jsonl`, and the whole task — thinking, tool
arguments, tool results, answers — is written to
`~/Library/Application Support/DemoApp/agent-transcripts/<session>.md` at the end of each turn. Notes
go to `~/Library/Application Support/DemoApp/agent-notes/` unless you pick another folder. **Copy** in
the header puts the same transcript on the clipboard.

**Using a bundle you downloaded by hand.** The **Models** screen treats a bundle as downloaded when
its folder is at `~/Library/Application Support/DemoApp/models/<folder>/` (the folder name comes from
the catalog; the agent bundle's is `qwen38-27b-agent`), has `manifest.json` at the top level, and has
a file named `.download-complete.json`, which the in-app download writes when it finishes. Move or
symlink the folder from `hf download` there and create that file (its contents are not read, so an
empty file will do); the row then shows the size and **Load in Chat** instead of **Download**.

The **Live Camera** demo points the camera at the world and captions what it sees, cycle after cycle,
in English or Japanese. Each cycle is independent — the context is reset every time, so a caption
never drifts on the last one — and captions stream in as they generate. It needs a bundle with an
image encoder; when more than one is installed a **Model** menu appears above the status line.

`DemoApp` is a small SwiftUI app that links the same `LLMCore` and `CoreMLBackend` libraries as the
CLI, so it runs the identical engine. It builds for macOS 26 and iOS 26 from one source tree, with
two schemes: `DemoApp` and `DemoApp-iOS`. It is a local development sample with App Sandbox disabled
so it can open a bundle from any path, not an App Store build.

To build it from the command line instead of Xcode, pin the architecture:
`xcodebuild ARCHS=arm64 -project Examples/DemoApp/DemoApp.xcodeproj -scheme DemoApp -configuration Release build`
(the bundled package is Apple Silicon only). Running from Xcode needs no such flag.

---

## Repository layout

```
README.md / README.ja.md   this index — the model table + quick start
samples/                   one self-contained model card per model (start here to pick a model)
Sources/                   shared Swift runtime: CoreLLMKit (LLMCore + CoreMLBackend) + the corellm-chat CLI
Examples/DemoApp/          SwiftUI demo app (macOS + iOS) — demos: Chat, Agent, Models (Live Camera on iOS)
scripts/download-model.sh  fetch a model bundle from Hugging Face
docs/                      cross-model engine notes — architecture.md, verification.md, agent-demo.md
LICENSE                    MIT (covers the code)
```

The Swift runtime under `Sources/` is shared by every model here; adding a model means adding
its card and its Hugging Face bundle, not a new runtime.

---

## How these samples are organized

Each row in the table links to a **model card** under `samples/`. A card is self-contained: who the
model is for, what makes the conversion notable, a benchmark table with its conditions and sources,
requirements, limitations, troubleshooting, and the weights' license. The HF column points to the
Hugging Face repo that hosts the actual weights; the code that runs them lives in this repository.
Numbers on this index are single representative figures — the card is the source of truth for the
conditions behind them.

## License

- **Code:** MIT — see [LICENSE](LICENSE). The shared Swift runtime under `Sources/` is MIT for every
  model here.
- **Model weights:** distributed separately on Hugging Face, each under its own license (see the
  **License** column above and the weights section of the relevant model card). The weights are
  *not* covered by this repository's MIT license.
