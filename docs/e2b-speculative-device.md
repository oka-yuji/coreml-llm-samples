# Gemma 4 E2B Speculative (pal6) — on-device test guide

This is a hands-on guide for running the **Gemma 4 E2B speculative runtime** on an iPhone with the
`DemoApp-iOS` app. It is a research demo of three things on Apple's Neural Engine:

- a **3-chunk multifunction** Core ML graph (decode + offset prefill) with host-side KV, `pal6`
  weights and int8 embedding / per-layer sidecars;
- **prompt-lookup lossless speculation** — verify width 4 only (`prefill4` + a batched `lmhead_v4`
  head), the memory-minimal single-width policy, with the batched verify head pinned to `cpuOnly`
  when the engine runs on the ANE;
- **KV save / restore** — the host KV cache dumps to disk and restores decode-only (no prefill
  functions resident), so a long prompt can be prefilled once and resumed later without paying
  prefill again.

> **Provisional naming.** The bundle format id is `coreml-corellm-r1` and is **not final**. If it is
> renamed later, update `manifest.json` (`format`) and `ChunkedSpeculativeChain.format` together.

The runtime lives in the shared `Sources/CoreMLBackend` library (`ChunkedSpeculativeChain`), the same
library the CLI and the demo app link, so the CLI on a Mac and the app on a phone run the identical
engine.

---

## What you need

- An iPhone on iOS 26 (the campaign target is iPhone 15 / 6 GB) and a Mac with Xcode 26.
- The **model bundle** (about 4.7 GB): the three `pal6` chunk `.mlmodelc` folders, `lmhead.mlmodelc`,
  `lmhead_v4.mlmodelc`, the int8 sidecars (`embed_int8.bin`, `ple_int8.bin`) and their `*_scale_f32.bin`
  files, the tokenizer, `convert_config.json`, and a `manifest.json` whose `format` is
  `coreml-corellm-r1`, `sidecarStage` is `int8`, and `computeUnits` is `all` (ANE).

The bundle is distributed separately (it is far too large for git). Keep it as a single folder, for
example `gemma-4-e2b-speculative-pal6/`.

---

## 1. Build the app

```bash
cd Examples/DemoApp
xcodegen generate          # regenerates DemoApp.xcodeproj from project.yml
open DemoApp.xcodeproj
```

In Xcode, select the **`DemoApp-iOS`** scheme and your iPhone as the run destination. On the
`DemoApp-iOS` target's **Signing & Capabilities**, pick your development team (the project ships with
automatic signing). Press Run to install the app.

The macOS **`DemoApp`** scheme still builds and runs unchanged; it is the same app for the Mac.

## 2. Put the bundle on the phone

The app reads bundles from its own **Documents** folder (file sharing is enabled), so either:

- **Files app:** copy `gemma-4-e2b-speculative-pal6/` into *On My iPhone → DemoApp*, or
- **devicectl:** with the phone attached,

  ```bash
  BID=com.coreml-llm-samples.DemoApp
  xcrun devicectl device copy to --device <UDID> \
    --domain-type appDataContainer --domain-identifier "$BID" \
    --source /path/to/gemma-4-e2b-speculative-pal6 \
    --destination Documents/gemma-4-e2b-speculative-pal6
  ```

## 3. Load, chat, speculate, checkpoint

1. Launch the app. On the **Chat** screen's empty state, any bundle found in Documents is listed
   under **On-device bundles (Documents)** — tap `gemma-4-e2b-speculative-pal6` to load it. The first
   reply specializes ANE kernels (a one-time cost); later replies are fast.
2. **Short chat:** type a prompt and send. Greedy (temperature 0) decoding streams the reply.
3. **Speculation on / off:** the **Speculation** switch in the status bar toggles prompt-lookup
   speculation. It only changes speed — the text is identical either way (verification makes it
   lossless). It fires on quotation / verbatim / repetitive spans and stays out of the way otherwise.
4. **KV save / restore:** the **KV** menu has *Save Checkpoint* (prefill the current prompt once and
   dump the KV cache) and *Restore + Continue* (reload that KV and keep decoding with no prefill).
   The status line reports the checkpoint size and that no wide prefill functions were resident during
   restore.

---

## Run it on a Mac (CLI)

The `corellm-chat` CLI runs the same bundle on a Mac. `cpuOnly` is a good way to sanity-check output
without paying ANE specialization time:

```bash
swift build -c release
BIN=.build/release/corellm-chat
MODEL=/path/to/gemma-4-e2b-speculative-pal6

# short chat
"$BIN" --model "$MODEL" --compute cpuOnly --max-tokens 40 --stats \
  --prompt "List three fruits, one per line."

# speculation is lossless: these two produce identical text
"$BIN" --model "$MODEL" --compute cpuOnly --max-tokens 90 --no-mtp \
  --prompt "Repeat this line exactly five times, one per line: alpha bravo charlie delta."
"$BIN" --model "$MODEL" --compute cpuOnly --max-tokens 90 --stats \
  --prompt "Repeat this line exactly five times, one per line: alpha bravo charlie delta."

# KV save, then restore-and-continue (no prefill) — the continuation matches an uninterrupted run
"$BIN" --model "$MODEL" --compute cpuOnly --prompt "List three fruits, one per line." --kv-save /tmp/e2b-kv
"$BIN" --model "$MODEL" --compute cpuOnly --max-tokens 40 --kv-restore /tmp/e2b-kv
"$BIN" --model "$MODEL" --compute cpuOnly --max-tokens 90 --kv-restore /tmp/e2b-kv --kv-spec
```

`--compute` overrides the manifest's compute units; on a Mac use `cpuOnly` (or `all` for the ANE).

---

## Notes

- **Verify width is 4 only.** Speculation loads exactly one verify width (`prefill4` + `lmhead_v4`) to
  keep the resident footprint minimal, which is what survives on a 6 GB phone. The batched verify head
  runs on `cpuOnly` under the ANE because an int8 batched argmax head misbehaves there; the forward
  pass stays on the ANE, and the result is still bit-exact against plain decoding.
- **Restore is decode-only.** Restoring a checkpoint never loads the wide prefill functions, so it
  avoids the prefill working set that is the memory ceiling for long prompts on a small phone.
- **KV portability.** A checkpoint carries an identity key (config hash, shapes, sidecar stage, layer
  head-dims). Restoring into a different or altered bundle is refused; the bundle's folder name is a
  record-only field and does not have to match.

## Reading the metrics

Each reply shows a one-line caption, for example:

```
TTFT 1.2s  |  12.5 tok/s  |  prompt 36 (+reused 24)  |  180 tok  |  draft 74%  |  mem 310MB  |  eos
```

- **reused** is the KV cache reused from earlier turns (LCP prefix). The second turn of a conversation
  only prefills the new tokens, so its TTFT is much smaller than the first turn's.
- **finish reason** is one of `eos` (the model stopped), `cap` (an explicit token limit), or
  `contextFull` (the context window filled). Generation is **unlimited by default** now: it runs until
  `eos` or the context is full, not a fixed token count.
- **mem** is the app's own `phys_footprint` (its accounting). On the Neural Engine the model weights are
  wired by the OS **outside** this number (about 1.5 GB for E2B), so the whole-device memory use is
  higher than the `mem` shown here.

The first reply after loading pays a one-time ANE kernel specialization (tens of seconds on the phone);
later replies are fast, and later turns reuse the KV cache. If replies feel slow "every time", it is
this one-time specialization per launch, not re-processing the conversation.

A full record for every message, checkpoint, and launch is appended as JSON lines to
`Documents/metrics.jsonl` (identity, token counts, `finishReason`, TTFT / prefill / per-token latencies,
speculation stats, KV op timings, staged `phys_footprint` and available memory, thermal state, battery).
Retrieve it with the Files app (*On My iPhone -> DemoApp -> metrics.jsonl*) or `devicectl device copy from`.
