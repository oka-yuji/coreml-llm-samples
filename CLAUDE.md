# coreml-llm-samples

## Swift conventions

Use `~/claude-config/conventions/swift/architecture.md` and `testing.md`.
The existing implementation is captured by the initial SwiftLint baseline; new diagnostics must not be added.
The initial baseline records 207 existing diagnostics. The formatting-only migration keeps each file/rule count unchanged or lower, with 206 remaining after removing the shared unsafe specialization setting.

Before a PR:

```sh
swiftlint lint --config .swiftlint.yml --baseline .swiftlint-baseline.json --strict --no-cache
python3 Tools/Format/check_format.py --base origin/main
swift test --package-path CoreLLMKit
xcodebuild -project CoreMLSamples.xcodeproj -scheme CoreMLSamples -configuration Release -destination 'platform=macOS,arch=arm64' CODE_SIGNING_ALLOWED=NO build
xcodebuild -project CoreMLSamples.xcodeproj -scheme CoreMLSamples-iOS -configuration Release -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
```

### Differences from the common conventions

- Existing mathematical dimensions (`H`, `PLE`, `CTX`, `N`, `L`, `S`, `W`) in `ChunkedSpeculativeChain` and the four Python fixture fields in the two GLiNER fixture readers retain their original names. `Tools/Format/legacy-identifiers.json` lists the exact diagnostic/count ceiling; every other strict swift-format diagnostic fails. Do not extend the list for new declarations.
- Test display names use English in this public sample. Each name describes the expected behavior.
- There is no dedicated pure-logic target yet. `LLMCore` also contains runtime contracts. When adding a dedicated logic target, update the three pure-logic lint paths; the initial migration does not split existing targets.
- Initial adoption preserves the existing architecture and freezes existing violations. Any new asynchronous feature must follow the common Feature/Store rules.

## Model-dependent checks

The default unit tests require no downloaded model. GLiNER, round5, Qwen and Gemma integration tests are opt-in through their `CORELLM_*` environment variables. A skipped model test is not evidence of model verification.
Model execution must use the shared CoreLLM heavy-job lock. Build and CPU-only unit checks do not load model weights.

## Generated project

`project.yml` is the source of the Xcode project. Run `xcodegen generate` after changing sources or local model resources. The optional Mapper assets are not committed. A clean checkout must build without them.
