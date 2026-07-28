# Gemma 4 E4B Speculative — Apple Silicon Mac GPU guide

This is a hands-on guide for running the **Gemma 4 E4B speculative runtime** on an Apple Silicon Mac,
with the `corellm-chat` CLI or the macOS `DemoApp`. It is a research demo of three things on the Mac
GPU:

- a **4-chunk stateful** Core ML graph whose KV caches live in `MLState` on the GPU, so the host never
  moves KV between calls;
- **one graph in three roles** — sequence length is a range dimension (S = 1…128), so S=1 is decode,
  S=draft+1 is the batched speculation verify, and larger S is prefill. There are no separate verify
  functions and no drafter model;
- **prompt-lookup lossless speculation** — drafts come from matching the last n-gram against the
  prompt, then get verified on the main graph. Speculation changes only speed: the output is
  byte-identical with it off and on.

The graph is int4 chunks plus **int8 embedding / per-layer (PLE) sidecars** with fp32 per-row scales:
6.5 GB on disk, about 4 GB resident.

The runtime lives in the shared `Sources/CoreMLBackend` library (`CoreMLChainV2`), the same library the
CLI and the demo app link, so both run the identical engine.

---

## What you need

- An Apple Silicon Mac on macOS 26 with roughly **8 GB of free unified memory** (measured resident
  footprint is about 4 GB) and **6.5 GB of free disk**.
- The **model bundle**: the four chunk `.mlmodelc` folders (`chunk_0_12`, `chunk_12_24`, `chunk_24_36`,
  `chunk_36_42`), `lmhead.mlmodelc`, the int8 sidecars (`embed_int8.bin`, `ple_int8.bin`) with their
  `*_scale.bin` files, the tokenizer, `convert_config_v2int4_int8.json`, and a `manifest.json` whose
  `format` is `coreml-stateful-chain-v2`.

This graph is **Mac GPU only**. The `MLState` KV design does not run on the Neural Engine, and
`cpuOnly` is refused — multiple `MLState` instances loading together segfault natively. For iPhone,
use the [E2B bundle](e2b-speculative-device.md) instead.

---

## 1. Get the bundle

```bash
hf download okayuji/Gemma-4-E4B-it-coreml-speculative --local-dir ./gemma-4-e4b-speculative
```

Keep it as a single folder. The bundle is weights + config only; everything below drives it from this
repository.

## 2. Run it (CLI)

```bash
swift build -c release
BIN=.build/release/corellm-chat
MODEL=./gemma-4-e4b-speculative

"$BIN" --model "$MODEL" --stats --prompt "List three fruits, one per line."
```

The manifest carries no compute-unit override, so the CLI selects `cpuAndGPU` on its own — no
`--compute` flag is needed. Omit `--prompt` for an interactive REPL that reuses the KV cache across
turns.

## 3. Turn speculation on

Speculation is **opt-in** through the environment:

```bash
CORELLM_MTP_PLD=1 "$BIN" --model "$MODEL" --stats --prompt "…"
```

| Variable | Meaning | Default |
|---|---|---|
| `CORELLM_MTP_PLD` | enable prompt-lookup speculation | off |
| `CORELLM_MTP_DRAFT_LEN` | tokens drafted per round | 4 |
| `CORELLM_MTP_PLD_MINN` | shortest n-gram allowed to match | 3 |

`CORELLM_MTP_DRAFT_LEN=6` favors verbatim-heavy work; with mixed workloads 4 is the safer default.
Without `CORELLM_MTP_PLD` the CLI reports `this bundle has no speculation assets` and runs plain
decode — that is expected, because this bundle carries no drafter model.

## 4. Confirm it is lossless

Speculation is verified, so the two runs below must produce **identical bytes**. This is the check to
run yourself rather than take on trust:

```bash
PROMPT='Extract every person, place, and date from this note as JSON with keys people, places, dates: Maya flew from Osaka to Reykjavik on March 3, 2026, and met Dr. Chen and Lena Novak at the Harpa concert hall on March 5, 2026.'

"$BIN" --model "$MODEL" --prompt "$PROMPT" > off.txt
CORELLM_MTP_PLD=1 CORELLM_MTP_DRAFT_LEN=6 "$BIN" --model "$MODEL" --prompt "$PROMPT" > on.txt

cmp off.txt on.txt && echo "byte-identical"
```

Run each condition in **its own process**. Core ML keeps memory mapped after a model unloads, so
measuring several conditions in one process contaminates the later ones.

## 5. Where speculation pays

Speculation only helps when the continuation can be drafted from the prompt. Measured on an M4 Max
(128 GB) / macOS 26, GPU, greedy, one process per condition, draft length 6:

| Workload | Off | On | Speedup | Acceptance |
|---|---|---|---|---|
| Verbatim copy (repeat a 10-item list) | 33.2 tok/s | **126.3 tok/s** | **×3.80** | 0.86 |
| JSON extraction (99 tokens to EOS) | 32.9 tok/s | 32.7 tok/s | ×0.99 | 0.38 |

On copy, quotation, enumeration and literal-span extraction, speculation is worth several times the
decode rate. On JSON scaffolding and free-form prose it sits around break-even — the braces and key
names are not in the prompt, so there is nothing to draft from. For chat-style use, leave it off.

## 6. Run it in the demo app (macOS)

```bash
cd Examples/DemoApp
cp Local.xcconfig.template Local.xcconfig   # once: set DEVELOPMENT_TEAM = your 10-char Team ID
xcodegen generate
open DemoApp.xcodeproj
```

Select the **`DemoApp`** (macOS) scheme and Run. On the **Models** screen, *Gemma 4 E4B Speculative
(Mac GPU)* downloads the bundle from Hugging Face with progress and cancel, then **Load in Chat**
switches to the conversation screen. The entry is macOS-only: at 6.5 GB it does not fit an iPhone's
memory budget, and the graph needs the GPU.

---

## Notes

- **Greedy only.** The `lm_head` emits argmax token ids, not logits, with the tanh soft-cap baked in.
  Temperature sampling would need a logits-head variant, which this bundle does not include.
- **Context is 2,048 tokens** — a fixed property of the converted graph, not a runtime setting.
- **int4 fidelity.** Against the fp32 reference, greedy output diverges at the first generated token on
  one tested prompt — a benign style fork, not degradation. On the measured verbatim, enumeration and
  JSON workloads, int4 output is identical to the fp16 build's.
- **Rejection is safe.** When a drafted token is rejected the position rolls back, and rejected rows are
  overwritten and masked, so the `MLState` KV never diverges from a plain decode. That invariant is what
  the byte-identity gate above checks.

## Troubleshooting

**The first run of each new sequence shape is slow.** The GPU specializes kernels per shape on first
use, so the first prefill of a session can take about twice its steady latency. It is cached; later
runs at a width already seen do not pay it.

**`v2 stateful requires the GPU`.** You passed `--compute cpuOnly` or `cpuAndNeuralEngine`. This graph
runs only on `cpuAndGPU` or `all`.

**Core ML compile caches grow.** Repeated first-loads populate
`~/Library/Caches/**/com.apple.e5rt.e5bundlecache`, which reaches tens of GB across models over time.
It is safe to delete; the next first-load re-specializes.

**A literal `<end_of_turn>` appears at the end of a reply.** The bundle's `manifest.json` does not
override the chat template, so the runtime falls back to the `<start_of_turn>` / `<end_of_turn>`
spelling, which Gemma 4 tokenizes as ordinary text rather than as its turn-control tokens (105 / 106).
Generation still stops on EOS and speculation stays lossless, but the marker is echoed as text and each
turn spends about 18 extra prompt tokens. Adding `promptPrefix` (`<|turn>user\n`) and `promptSuffix`
(`<turn|>\n<|turn>model\n`) to `manifest.json` restores the canonical template.
