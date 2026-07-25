# Gemma 4 12B IT — 128K Context Ladder (Core ML)

Run **Gemma 4 12B** on Apple Silicon with **Core ML**, at a full **131,072-token** context —
using a *Context Ladder* that keeps short conversations fast and only pays the wide-context cost
once a conversation actually grows past 32K. Clone, download the bundle, and chat.

> **12B** · **131,072** ctx · **~11 tok/s** (32K mode, M4 Max) · **6.7 GB** single bundle · **zero-copy** 32K→128K promotion · **lossless** speculative decoding

[← Back to the samples index](../README.md) · [日本語版 →](gemma-4-12b-128k.ja.md) · [Article](https://medium.com/@yu.j.0513/running-gemma-4-12b-with-a-128k-context-on-core-ml-23d6918dd370)

---

## What makes it different

Most "run an LLM on Core ML" samples stop at a single fixed context and greedy decode. This one
ships the research pieces that make a 12B / 128K model usable on a Mac:

- **128K Context Ladder** — one bundle exposes two Core ML functions (`ctx32k`, `ctx128k`) over a
  *shared* KV state. Below 32K it runs the narrow, ~3× faster graph; past 32K it swaps to the wide
  graph with **no data copy** and **bit-exact** output.
- **Lossless speculative decoding** — a small drafter proposes tokens; the main graph verifies them
  in one batched pass and keeps only greedy-correct ones. The text is *identical* to non-speculative
  decoding; only the speed changes.
- **Bit-exact verification everywhere** — int4 conversion, the ring KV buffer, promotion, and
  speculation are each gated against a greedy exact-match oracle. See [docs/verification.md](../docs/verification.md).

---

## Quick start

**Requirements:** an Apple Silicon Mac, macOS 26+, and the Swift 6.2 toolchain (Xcode 26). See
[Requirements](#requirements) for memory/disk.

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

One-shot instead of a REPL:

```bash
swift run -c release corellm-chat \
  --model ./models/gemma-4-12b-it-coreml-128k \
  --prompt "What is the capital of Japan? Answer in one sentence." \
  --max-tokens 64 --stats
```

### First run is slow — this is expected

Two one-time costs stack up on the very first run:

1. **Compile.** The bundle ships the chunks and `lm_head` as `.mlpackage`, not pre-compiled
   `.mlmodelc`. The first load compiles them to `.mlmodelc` **once** and caches the result inside
   the bundle directory (next to the `.mlpackage`), so every later load skips it. This step is
   fast: measured at **~1.2 s total** for the whole bundle on an M4 Max (about 0.3 s per chunk).
2. **GPU specialization.** Before the first inference the GPU kernels specialize (mostly the int8
   `lm_head`'s 262K-way argmax GEMM). A 2026-07-05 measurement put this at **~40 seconds per
   process**; a 2026-07-21 re-check on macOS 26.5.x instead saw it **cached across processes** — a
   fresh process reached its first token in **0.25–3.8 s** (CPU+GPU, drafter off/on). Budget the
   full ~40 s for a first-ever run (no cache, or after clearing the E5RT cache); later runs on the
   same machine are fast.

The CLI prints a `[warming up…]` notice before the first token. Subsequent turns in the same session
are fast, and later process launches skip the compile.

### CLI flags

| Flag | Meaning |
|---|---|
| `--model <dir>` | Path to the bundle directory (contains `manifest.json`). Required. |
| `--prompt "<text>"` | Generate one response and exit. Omit for an interactive REPL. |
| `--max-tokens <n>` | Max tokens per turn. Default 512. |
| `--no-mtp` | Disable speculative decoding. Default: ON when a drafter ships. |
| `--stats` | Print TTFT, decode ms/tok, tok/s, and draft acceptance after each turn. |

In the REPL, each line is one turn (KV is reused across turns), `/reset` starts a new conversation,
and Ctrl-D exits.

---

## Features

- **Context Ladder (multifunction, 128K).** Two functions `ctx32k` / `ctx128k` over one shared
  `[1, 131072, 512]` KV `MLState`. Promotion at 32,768 tokens is zero-copy (a function swap, no
  `read_state`/`write_state`) and verified bit-exact against a native 128K bundle.
- **Ring KV.** The 40 sliding-window layers keep their KV in a physical 1,024-slot ring buffer (the
  sliding-attention span); only the global layers hold the full 131,072-slot KV. Keeps resident KV
  bounded at 128K.
- **MTP + PLD lossless speculation.** A model drafter (`drafter_ring.mlmodelc`) is auto-detected
  from the bundle and used by default. The ladder even switches drafter width by regime (a `w32768`
  drafter below promotion, `w131072` after), because a fixed-width drafter is billed by its baked-in
  width. A prompt-lookup drafter (PLD) can be layered on with `CORELLM_MTP_PLD=1` for
  quotation/verbatim prompts.
- **KV persistence** *(library feature — not wired into this CLI)*. `CoreMLChainV2` can export the
  KV `MLState` to disk and restore it in a fresh process to continue **without re-prefilling** —
  greedy-exact across processes. This is the mechanism behind "cold-prefill a long document once".
  It was validated in the upstream research project (bit-exact cross-process restore); here it is a
  `CoreMLChainV2` API rather than a `corellm-chat` flag — treat it as a building block.

---

## Benchmarks

Measured on **M4 Max, 128 GB, macOS 26**, single process per condition, greedy decode, CPU+GPU. The
`v2mmladder` int4 bundle. `S=1` decode.

### Decode speed

| Mode | ms/tok | tok/s | When |
|---|---|---|---|
| `ctx32k` (≤ 32,768 tokens) | **90.51** | **11.0** | every conversation under 32K |
| `ctx128k` (> 32,768 tokens) | **300.96** | ~3.3 | only once a conversation grows past 32K |

The wide-context "fixed-width tax" is **3.24×** at int4 — that figure is the 128K/32K ratio measured
on the **standalone** single-mode bundles (357.04 / 110.17 ms/tok). The ladder is 15–18% faster than
standalone in *both* modes, so dividing the table above gives a slightly steeper 3.33×; same tax,
different pair of measurements. The whole point of the ladder is to *not* pay it until you have to.
*(Source: internal results 2026-07-15 build gate; 2026-07-13 promotion spike.)*

### Speculative decoding (ctx32k regime, drafter-only, draft_len = 4)

Pair-ratio median speedups over base decode, by prompt type:

| Prompt type | speedup | draft acceptance |
|---|---|---|
| Counting / enumeration | **×1.47** (75.0 ms/tok) | 0.64 |
| Quotation | **×1.27** | 0.57 |
| Verbatim copy | ×1.05 | 0.50 |
| Free-form | ×1.05 | 0.50 |

Speculation helps most on structured / repetitive text and is roughly break-even on free prose; it
is **always lossless** (output identical to `--no-mtp`).
*(Source: internal results 2026-07-19 ladder drafter regime; 2026-07-18 ring drafter MTP.)*

> Reproduced here as a smoke test (same Mac): base decode **10.2 tok/s**; with MTP on the same Q&A
> prompt **16.0 tok/s** at **0.75** acceptance — and byte-identical text. This is the two-run
> equality described in [docs/verification.md](../docs/verification.md).

### Promotion, memory, size

| | |
|---|---|
| 32K → 128K promotion | **bit-exact** (greedy 32/32), zero-copy shared `MLState`, ~0 switch cost |
| Bundle size | **6.71 GB** (int4 chunks + int8 lm_head), single bundle |
| Runtime footprint | ~10 GB RSS (the full-context KV `MLState`, ~2.1 GB, is resident from token 1) |

*(Source: internal results 2026-07-15 build gate.)*

---

## Requirements

- **Apple Silicon Mac** (M-series). Not supported on Intel or on iOS/iPadOS (see Limitations).
- **macOS 26+**.
- **Swift 6.2 toolchain** (Xcode 26 or a matching Swift toolchain).
- **Memory:** ~10–12 GB used at runtime. **24 GB+ recommended** so the OS isn't under pressure.
- **Disk:** ~11 GB for the model bundle.

---

## Limitations (read these)

- **128K mode is slow by design.** The wide graph runs at ~300 ms/tok (~3.3 tok/s) versus ~90 ms/tok
  in 32K mode — a fixed ~3.2–3.3× tax at int4 (3.24× measured on the standalone bundles). The ladder
  avoids it for sub-32K conversations, but a genuinely 128K-deep context is not fast.
- **Greedy argmax only.** The `lm_head` emits an argmax token id, not logits. There is **no
  temperature / top-k / top-p sampling** — outputs are deterministic. Sampling would require
  re-converting the model with a logits head.
- **Mac only.** The v2 stateful chain keeps its KV in GPU-resident `MLState`; running CPU-only or on
  the ANE segfaults with multiple `MLState` instances. There is no iOS build of this pipeline.
- **Speculation's edge shrinks with context.** The fixed-width verify tax means the largest wins are
  on short, structured prompts. Deeper contexts still benefit but by less; free-form prose is near
  break-even.
- **First-ever run pays ~40 s** of one-time GPU kernel specialization (measured 2026-07-05). On
  macOS 26.5.x (re-checked 2026-07-21) this is cached across processes, so later runs reach the first
  token in seconds; budget ~40 s only for the first run or after clearing the E5RT cache.

---

## Troubleshooting

- **"It hangs on the first token."** The one-time `lm_head` GPU specialization runs before the first
  token. On a first-ever run (no cache) budget ~40 s (measured 2026-07-05); on macOS 26.5.x the
  specialization is cached across processes (re-checked 2026-07-21), so a fresh process is back to a
  few seconds. Warm turns in the same session are fast. Pre-warm at app launch if you embed the library.
- **Disk fills up / random segfaults after many runs.** Core ML's E5RT cache
  (`~/Library/Caches/com.apple.e5rt.e5bundlecache` and the swiftpm-testing-helper variant) grows
  several GB per run. Clear it if you're low on space (it regenerates; next load is just slower).
- **Out of memory / heavy swapping.** The model needs ~10–12 GB resident. Close other large apps;
  prefer a 24 GB+ machine.
- **"bundle has no drafter — running without speculation."** The drafter file
  (`drafter_ring.mlmodelc`) wasn't found in the bundle directory. Re-run the download script.

---

## How it works

- [docs/architecture.md](../docs/architecture.md) — the pipeline, the Context Ladder, ring KV, and the
  drafter path.
- [docs/verification.md](../docs/verification.md) — the bit-exact / lossless gates and the margin rule.

## Repository layout

```
Package.swift              library CoreLLMKit (LLMCore + CoreMLBackend) + executable corellm-chat
Sources/LLMCore/           pure-Swift types & protocols (no Core ML dependency)
Sources/CoreMLBackend/     the Core ML engine, chains, host inputs, drafter, tokenizer wrapper
Sources/corellm-chat/      the streaming chat CLI
scripts/download-model.sh  fetch the model bundle from Hugging Face
docs/                      architecture & verification notes
```

Only dependency: [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers)
(tokenizer only, hidden behind the `Tokenizing` protocol).

## License

- **Code:** MIT — see [LICENSE](../LICENSE).
- **Model weights** (distributed separately on Hugging Face): derived from
  [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) — converted to a Core ML
  graph and quantized (int4 AWQ matmul / int8 lm_head) — and distributed under the **Apache License
  2.0**. **Built with Gemma.**

The following applies to the model weights only — the code in this repository stays MIT.
Gemma 4 is released by Google under the [Apache License 2.0](https://ai.google.dev/gemma/docs/gemma_4_license), and the derived weights in this bundle carry the same license. The full license text ships with the model bundle on Hugging Face.

"Gemma" is a trademark of Google LLC; "Core ML" and "Apple Silicon" are trademarks of Apple Inc.
Not affiliated with, endorsed by, or sponsored by Google or Apple.
