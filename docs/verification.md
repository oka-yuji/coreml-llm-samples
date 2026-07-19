# Verification methodology

Every performance trick in this engine — int4 quantization, the ring KV buffer, the 32K→128K
promotion, speculative decoding — is only allowed to ship if it is **numerically transparent**:
it may change speed or memory, never the tokens produced. This page describes the gates that
enforce that.

## The rule: greedy top-1 exact match

The engine decodes greedily (argmax). So the correctness oracle is simple and strict: for a fixed
prompt, the token id sequence must match the reference. We compare against:

- the **Hugging Face reference** (PyTorch) for the base model conversion, and
- the **non-speculative path** for anything that adds speculation.

A feature passes only when `#diff = 0` over a fixed greedy window (typically 32 tokens), or, in the
few genuine floating-point ties, when it passes an explicit margin rule (below).

## near-tie / margin rule

int4 GEMM at fp16 compute precision can land two candidate tokens within floating-point rounding of
each other (a "near-tie"). When the top-2 logits are within a small margin, a different but equally
valid argmax is not a bug. These cases are judged by the **chain's own effective logit margin**, not
by a higher-precision oracle — an fp32/bf16 oracle can itself pick the "wrong" side of a tie the
fp16 pipeline never sees. In practice this affects a handful of positions and never the overall
meaning.

## The gates

| Property | How it is verified | Result |
|---|---|---|
| int4 conversion vs HF | teacher-forced greedy, `#mismatch = 0` (one pos36 near-tie under margin rule) | exact |
| Ring KV (32K) | Swift ↔ Python token sequence exact, including 1024-window wrap | exact |
| 32K → 128K promotion | greedy 32/32 identical to a native 128K bundle; zero near-tie | **bit-exact** |
| Speculative decoding (MTP) | drafter ON output == OFF output, token-for-token, incl. deep / window-crossing prompts | **lossless** |
| KV persistence | restore in a fresh process, continue with no prefill, greedy 32/32 identical | exact |
| Multi-turn KV reuse | "continue == full re-prefill" — identical token sequence, text, and count | exact |

## Why speculation is lossless by construction

The drafter proposes tokens; the main graph then verifies them in a single batched forward pass and
keeps only the tokens whose argmax matches greedy decoding. Rejected tokens are discarded and the
position rewinds. So the *drafter's* quality only affects the acceptance rate (speed) — the text is
always exactly what greedy decoding would have produced. A weak drafter makes it slower, never wrong.

## The zero-copy promotion argument

When the KV is promoted from the 32K window to the full 128K, no data moves: both functions share
one physical `MLState` sized at the maximum shape. The mathematical reason it is bit-exact: the KV
slots above the written region are zero and masked with `-inf`, so `exp(-inf) = 0` contributes
nothing to the softmax denominator. Widening the visible window therefore cannot change any logit.

## Reproducing the smoke test

The two runs below are exactly what this repository was validated with (M4 Max, macOS 26):

```bash
# base decode (in the documented ~11 tok/s band for 32K mode)
corellm-chat --model <bundle> --prompt "…" --max-tokens 64 --no-mtp --stats

# speculative decode — identical text, higher tok/s, prints draft acceptance
corellm-chat --model <bundle> --prompt "…" --max-tokens 64 --stats
```

The generated text is identical between the two; only `tok/s` (and the `draft acceptance` line)
differ. That equality is the lossless guarantee in action.
