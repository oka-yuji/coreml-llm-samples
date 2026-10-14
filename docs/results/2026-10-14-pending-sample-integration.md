# Pending sample integration verification — 2026-10-14

The restored work is checked against GitHub main `0f7f8fe`. The source branches are `demo/gliner-mapper` (`ae7c6446`), `exp/mlstate-ios` (`bd53677d`) and `feat/live-b70` (`508eec22`). Local main is included by the Mapper history, and the old worktree prompts-file patch is equivalent to `4b5400f`.

The integration preserves the current CoreLLMKit/CoreMLSamples layout and the newer cached camera-mode selection and 4096px image bound. Private research copies and raw logs from the public repository's working folder are excluded.

## Corrections

- The GLiNER fixture reader resolves the fixture from the current layout.
- The default Xcode project has no local Mapper model references. `project.mapper.yml` explicitly adds downloaded assets to both targets.
- Specialization hints belong to each chain and are passed to every immediate or delayed model load.
- Tap-based dialog input works without speech availability. Cancellation during audio-route settling returns before microphone startup.
- The self-test keeps current prompt-file/RAG options and adds the explicit compute-unit override.
- The old camera branch cannot replace the newer cached mode selection or image bound.
- Probe input clearing uses the actual byte buffer supplied by Core ML, including Int8 inputs.
- The format wrapper rejects abnormal formatter exits, mixed index/worktree Swift changes, symlinks, unknown diagnostics and increased legacy-name counts.
- Current reproduction paths replace obsolete Examples/DemoApp paths. Output-equality claims remain limited to the recorded input/device cases.

## Validation

- Swift unit suite: 22 model-free tests executed, 10 model-dependent tests skipped. The 30-sentence GLiNER number-window fixture has zero mismatches.
- SwiftLint: 83 files, zero diagnostics beyond the documented baseline. Initial adoption recorded 207 existing diagnostics; removal of shared unsafe state leaves 206. Formatting migration adds no file/rule count.
- Strict swift-format: changed Swift files pass except the exact existing mathematical/fixture name diagnostics listed in `Tools/Format/legacy-identifiers.json` (36 occurrences maximum).
- macOS and iOS Release builds succeed with `ARCHS=arm64 CODE_SIGNING_ALLOWED=NO` and no downloaded Mapper assets.
- `swift Tools/Verification/CheckProbeBuffers.swift` reproduces the old Int8 size error inside a deliberately oversized allocation: 96 canary bytes changed. The bounded replacement changes zero canary bytes and clears all 32 input bytes. This is a CPU allocation check, not model inference.
- Independent review covers source changes, conflict resolutions, the generated project, frozen fixtures and validation-tool behavior. The formatter failure-path mock covers seven positive/negative cases.

The checks do not establish new model accuracy, device-memory limits, speech recognition quality or network retrieval behavior. Earlier measurements retain their original conditions. No model inference, model download or private raw-data publication was performed for this integration.
