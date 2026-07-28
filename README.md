# Core ML LLM Samples

Core ML–native conversions of open LLMs for **Apple Silicon**. Every model ships with a shared Swift
runtime you can clone and run, every benchmark is quoted with its measurement conditions and its
source, and every conversion is gated **bit-exact** against its reference implementation. That
verification discipline is the house style — the receipts are in each model card.

[日本語版 →](README.ja.md)

---

## Models

| Model | Size | Context | Speed | HF | Article | Demo | License |
|---|---|---|---|---|---|---|---|
| [Gemma 4 12B IT — 128K Context Ladder](samples/gemma-4-12b-128k.md) | 6.7 GB (int4) | 131,072 | ~11 tok/s (32K mode, M4 Max) | [okayuji/gemma-4-12b-it-coreml-128k](https://huggingface.co/okayuji/gemma-4-12b-it-coreml-128k) | [article](https://medium.com/@yu.j.0513/running-gemma-4-12b-with-a-128k-context-on-core-ml-23d6918dd370) | [demo](https://x.com/oka_yuuji/status/2080660675333161154/video/1) | Apache-2.0 |
| [Gemma 4 E2B IT — ANE Speculative (iOS · macOS)](docs/e2b-speculative-device.md) | ~4.9 GB (pal6+int8) | 2,048 | ~12 tok/s (iPhone 15) · ~16 tok/s (17 Pro) | [okayuji/Gemma-4-E2B-it-coreml-speculative](https://huggingface.co/okayuji/Gemma-4-E2B-it-coreml-speculative) | — | — | Apache-2.0 |
| [Gemma 4 E4B IT — Mac GPU Speculative](docs/e4b-speculative-mac.md) | 6.5 GB (int4+int8) | 2,048 | ~30 tok/s (M4 Max GPU) | [okayuji/Gemma-4-E4B-it-coreml-speculative](https://huggingface.co/okayuji/Gemma-4-E4B-it-coreml-speculative) | — | — | Apache-2.0 |

> **Speed** is one representative figure; the full measurement conditions and every number live in
> each model's card.
>
> **Gemma 4 E2B** ships lossless prompt-lookup speculation verified **byte-identical on an iPhone 15
> (A16) Neural Engine**, plus cross-machine KV restore — details and the memory ledger are in its card.

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
list — a sidebar of demos with the selected one shown on the right. It has two demos, **Chat** and
**Models**, with Chat selected on launch; more models and modalities each add a row and a screen here.

The **Models** screen downloads model bundles in-app from Hugging Face, with progress, cancel, and
delete, and loads a finished bundle straight into Chat. For a private or gated repo it auto-detects a
token from `HF_TOKEN` or the `hf` CLI; a public repo needs none.

The **Chat** demo is the conversation screen. Select a model in **Models** and press **Load in Chat**
to switch here and talk. Responses stream as they generate, and the status line reports the last
turn's tokens/second and time-to-first-token. The first reply of a run pays the one-time GPU
specialization cost described under **First run is slow** in the
[model card](samples/gemma-4-12b-128k.md); later replies are fast.

`DemoApp` is a small macOS 26 SwiftUI app that links the same `LLMCore` and `CoreMLBackend` libraries
as the CLI, so it runs the identical engine. It is a local development sample with App Sandbox
disabled so it can open a bundle from any path, not an App Store build.

To build it from the command line instead of Xcode, pin the architecture:
`xcodebuild ARCHS=arm64 -project Examples/DemoApp/DemoApp.xcodeproj -scheme DemoApp -configuration Release build`
(the bundled package is Apple Silicon only). Running from Xcode needs no such flag.

---

## Repository layout

```
README.md / README.ja.md   this index — the model table + quick start
samples/                   one self-contained model card per model (start here to pick a model)
Sources/                   shared Swift runtime: CoreLLMKit (LLMCore + CoreMLBackend) + the corellm-chat CLI
Examples/DemoApp/          macOS SwiftUI demo app — a sidebar of demos (Chat, Models); links the Sources/ runtime
scripts/download-model.sh  fetch a model bundle from Hugging Face
docs/                      cross-model engine notes — architecture.md, verification.md
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
