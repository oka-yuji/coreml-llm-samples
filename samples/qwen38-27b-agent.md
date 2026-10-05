# Qwen3.8 27B Agent (Core ML)

Run **Qwen3.8 27B** on an Apple Silicon Mac with **Core ML** as a web-browsing agent: it searches,
reads pages, writes notes and answers with sources, fully on the machine in front of you. The bundle
is int8 with a 16,384-token context and the model's own multi-token-prediction head for lossless
self-speculative decoding.

> **27B** · **16,384** ctx · **16.7 tok/s** decode (M4 Max, median of 148 generations) · **55.7 GB** peak · **48/48** agent runs on a frozen 12-task benchmark

[← Back to the samples index](../README.md) · [日本語版 →](qwen38-27b-agent.ja.md) · [Model on Hugging Face](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent) · Article: coming soon

---

## What makes it different

- **A hybrid model on `MLState`** — 48 gated-delta-rule linear-attention layers carry a recurrent
  state that cannot be rolled back, and 16 full-attention layers carry a KV cache. The runtime keeps
  the two in step across agent turns by reusing the cache by **text prefix**: only the new suffix of
  a conversation is tokenised, so a generated turn is never re-tokenised and never re-prefilled
  (0 of 148 generations in the benchmark).
- **Lossless self-speculation without a drafter model** — the MTP head drafts up to three tokens; a
  static four-row `verify` function accepts the greedy-correct prefix and writes the accepted
  recurrent state back. Speculation off and on produce byte-identical text.
- **An agent you can audit** — every step's TTFT, tok/s and reused tokens are shown under the step
  and logged to `metrics.jsonl`; every transcript is saved as Markdown; a `Sources:` section is
  checked against the pages the agent actually saw.

---

## Quick start

**Requirements:** an Apple Silicon Mac with 96 GB or more of unified memory, macOS 26+, Xcode 26 /
Swift 6.2, about 30 GB of disk. See [Requirements](#requirements).

```bash
# 1. Clone
git clone https://github.com/oka-yuji/coreml-llm-samples.git
cd coreml-llm-samples

# 2. Download the bundle (~30 GB)
hf download okayuji/Qwen3.8-27B-coreml-agent --local-dir ./models/qwen38-27b-agent

# 3. Chat from the CLI
swift run -c release --package-path CoreLLMKit corellm-chat --model ./models/qwen38-27b-agent --stats \
  --prompt "List three fruits, one per line."
```

For the agent, open the demo app (`CoreMLSamples.xcodeproj`, build for `arm64`), download the bundle from
**Models** (or register the folder from step 2 as described under **Agent** in the
[README](../README.md)), press **Load in Chat**, then switch to **Agent**. The loop, the tools, the
settings and the logs are described in [docs/agent-demo.md](../docs/agent-demo.md).

---

## Key numbers

Apple M4 Max (128 GB), macOS 26.6.2, GPU, greedy, one process at a time. The agent benchmark, its
task list and grading script are in this repository:
[docs/results/2026-09-13-qwen38-agent-benchmark.md](../docs/results/2026-09-13-qwen38-agent-benchmark.md).
The CLI speed, load-time and reference-agreement numbers come from the author's conversion
records, which are not published; they are reproduced with their conditions on the
[model card](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent).

| Measurement | Value |
|---|---|
| Agent benchmark, 12 frozen tasks × (3 rounds `low` + 1 round `off`) | 48 / 48 |
| Same tasks with file operations enabled (4 tools), 1 round | 12 / 12 |
| Decode, median of 148 generations (speculation on) | 16.65 tok/s (12.4–19.7) |
| Time to first token, median | 6.9 s (max 46.4 s at a session's first prefill) |
| Peak footprint, median | 55.7 GB (max 55.7 GB) |
| Speculation acceptance, median | 0.86 |
| CLI decode, speculation off → on (Japanese / English / code) | 5.9 → 10.9 / 5.8 → 13.7 / 6.0 → 16.8 tok/s |
| Bundle load, demo app | 47.4 s median (48 runs); CLI 8–9 s after the first compile (94 s) |

---

## Requirements

- Apple Silicon Mac, macOS 26 or later, **96 GB or more** unified memory. Measured on 128 GB; the
  55.7 GB peak leaves 64 GB Macs out; 96 GB is untested.
- About 30 GB of disk for the bundle (30.3 GB, 28.2 GiB), plus the OS's compiled-model cache.
- Build the demo app with `ARCHS=arm64` (the package does not build for x86_64).

---

## Limitations

- 96 GB+ Macs only; no iPhone / iPad.
- 16,384-token context: two or three long pages fill it, after which the agent drops the oldest tool
  output.
- Web search uses DuckDuckGo's HTML endpoint by default (a Brave Search API key is optional);
  results vary with the endpoint.
- Thinking is in English; answers follow the user's language.
- Numbers are from one machine and one OS version.
- The 55.7 GB peak is the OS's footprint metric on macOS 26.6.2. On macOS 27.0 the same load shows
  about 50 GiB resident but only a few GB of footprint, because the weights are mapped as clean
  file-backed pages, so the app's peak figure is not comparable across OS versions.

---

## Verification

- Top-1 agreement with the bf16 reference: 633 / 640 positions (98.9%), 0 hard flips; a 32-token
  teacher-forced sequence matched exactly.
- Lossless one-token decode through the verify function: hidden state and recurrent states bit-equal
  over 200 positions; accepted-prefix write-back bit-equal for every accepted length.
- Speculation off = on: byte-identical on the recorded probe prompts.
- Agent benchmark: 48 / 48 with the task list and grading script frozen by SHA-256.
- GUI end-to-end: 14 checks driven inside the demo app with screenshots.

---

## License

Weights: Apache License 2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B);
the license text is in the model repository. Code in this repository: see [LICENSE](../LICENSE).
