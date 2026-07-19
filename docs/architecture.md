# Architecture

A short tour of how the engine runs Gemma 4 12B under Core ML. For the full research
write-ups (with every measured number and every wrong prediction), see the source-of-truth
project the numbers in the README are drawn from.

## Pipeline

```
tokenizer.json ─┐
                ▼
prompt ──► HFTokenizer ──► CoreMLEngine ──► CoreMLChainV2 ──► chunk_0_12 ─► chunk_12_24 ─► chunk_24_36 ─► chunk_36_48 ─► lmhead ─► argmax
                              │  (actor)        (stateful)         (int4 matmul GEMM, AWQ p999)                (int8)
                              ▼
                         MTP drafter (optional, lossless speculation)
```

- **`ModelBundle`** reads `manifest.json` (name, context length, prompt template, `format`).
  `format = "coreml-stateful-chain-v2"` selects the stateful chain below.
- **`CoreMLEngine`** is a Swift `actor`. It owns the tokenizer and chain, applies the chat
  template from the manifest, drives prefill + decode, and yields an `AsyncThrowingStream` of
  `GenerationEvent`s (`loadCompleted`, `prefillCompleted`, `token`, `finished`).
- **`CoreMLChainV2`** is the stateful graph. The 48 transformer layers are split into four
  `.mlmodelc` chunks that pass hidden states down the chain; the KV cache lives in Core ML
  `MLState` (GPU-resident) rather than in host buffers. The `lmhead` chunk emits an argmax
  token id (not logits).

## The Context Ladder (128K)

The bundle ships a single set of chunks that expose **two functions**, `ctx32k` and `ctx128k`,
via `MLModelConfiguration.functionName`. Both functions share one physical KV `MLState`:

- The full-attention `MLState` is declared at the maximum shape `[1, 131072, 512]` and is shared
  by both functions. `ctx32k` writes/reads only the first 32,768 slots (a masked window); the
  tail stays zero and is hidden by the attention mask.
- Below 32,768 tokens the engine runs the `ctx32k` function — a narrower GEMM that is ~3× faster.
- When the write position reaches 32,768, the engine switches to the `ctx128k` function. Because
  the KV already lives in the shared `MLState`, **promotion is zero-copy**: no `read_state` /
  `write_state`, just a function swap. It has been verified bit-exact (greedy 32/32 identical to
  a native 128K bundle).

This is how the model keeps ~90 ms/tok for every conversation that stays under 32K, and only pays
the wide-context tax (~300 ms/tok) once a conversation actually grows past 32K.

## Ring KV

Gemma 4 12B mixes sliding-window attention layers with a few global layers. The 40 sliding-window
layers only ever attend to a 1,024-token span, so their KV is kept in a **physical 1,024-slot ring
buffer** (`[8, 1024, 256]`) with wrap-around addressing. Only the global layers hold the full
`[1, 131072, 512]` KV. This keeps the resident KV footprint bounded even at 128K.

## Speculative decoding (MTP)

When a drafter (`drafter_ring.mlmodelc`) ships in the bundle, the engine can run lossless
speculative decoding: the small drafter proposes several tokens, the main graph verifies them in a
single batched call, and only the tokens that match greedy decoding are kept. Output is **identical**
to non-speculative decoding by construction — verification is what makes it lossless, so speculation
only changes speed, never text.

For the ladder, the drafter also switches by regime: a `w32768` drafter below promotion and a
`w131072` drafter after, because a fixed-width drafter is billed by its baked-in KV width, not by
the target's effective context.

A **prompt-lookup drafter (PLD)** can be enabled alongside the model drafter with
`CORELLM_MTP_PLD=1`; it copies n-gram continuations straight from the prompt for quotation/verbatim
prompts. It is off by default.

## Files

- `Sources/LLMCore` — pure-Swift types and protocols (`LLMEngine`, `ModelBundle`, generation
  types, `Tokenizing`). No Core ML dependency.
- `Sources/CoreMLBackend` — the Core ML implementation (`CoreMLEngine`, `CoreMLChainV2`, host
  input assembly, ring/ladder masks, RoPE, the drafter path, and the tokenizer wrapper).
- `Sources/corellm-chat` — the streaming chat CLI.
