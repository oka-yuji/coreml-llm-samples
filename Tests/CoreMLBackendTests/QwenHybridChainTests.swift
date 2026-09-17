import CoreML
import Foundation
import LLMCore
import Testing
@testable import CoreMLBackend

private let qwen35Env = "CORELLM_QWEN35_BUNDLE"
private let qwen38Env = "CORELLM_QWEN38_BUNDLE"
private let gemmaEnv = "CORELLM_GEMMA_BUNDLE"

private func bundleURL(_ env: String) -> URL? {
    guard let path = ProcessInfo.processInfo.environment[env], !path.isEmpty else { return nil }
    return URL(filePath: path)
}

private let qwen35URL = bundleURL(qwen35Env)
private let qwen38URL = bundleURL(qwen38Env)
private let gemmaURL = bundleURL(gemmaEnv)

private func exists(_ url: URL?, _ name: String) -> Bool {
    guard let url else { return false }
    return FileManager.default.fileExists(atPath: url.appending(path: name).path(percentEncoded: false))
}

private var qwen35Exists: Bool {
    exists(qwen35URL, "manifest.json") && exists(qwen35URL, "convert_config_v2int4.json")
        && exists(qwen35URL, "chunk_0_8.mlmodelc")
}

private var qwen38TokenizerExists: Bool { exists(qwen38URL, "tokenizer.json") }
private var gemmaTokenizerExists: Bool { exists(gemmaURL, "tokenizer.json") }

private let qwen35Skip: Comment = "set \(qwen35Env) to a qwen3_5 v2 bundle directory to run this test"
private let qwen38Skip: Comment = "set \(qwen38Env) or \(qwen35Env) to a bundle directory to run this test"
private let gemmaSkip: Comment = "set \(gemmaEnv) to a Gemma bundle directory to run this test"

private let jaGreeting = "\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}"
private let jaCapitalQuestion =
    "\u{65E5}\u{672C}\u{306E}\u{9996}\u{90FD}\u{306F}\u{3069}\u{3053}\u{3067}\u{3059}\u{304B}?"
private let jaFollowUp =
    "\u{305D}\u{306E}\u{8857}\u{306E}\u{89B3}\u{5149}\u{5730}\u{3092}1\u{3064}\u{6559}\u{3048}"
    + "\u{3066}\u{304F}\u{3060}\u{3055}\u{3044}\u{3002}"
private let jaSelfIntro =
    "\u{65E5}\u{672C}\u{8A9E}\u{3067}\u{81EA}\u{5DF1}\u{7D39}\u{4ECB}\u{3057}\u{3066}\u{3001}"
    + "Core ML \u{306B}\u{3064}\u{3044}\u{3066}3\u{6587}\u{3067}\u{8AAC}\u{660E}\u{3057}\u{3066}"
    + "\u{304F}\u{3060}\u{3055}\u{3044}\u{3002}"
private let emojiRepeat = "Repeat this emoji 40 times: \u{1F381}"
private let emojiFollowUp = "How many did you write? Answer with one number."
private let eosQuestion = "Answer in one short sentence: what is the capital of Japan?"
private let eosFollowUp = "Name one tourist spot there in one short sentence."

private func capturingStandardError<T>(_ body: () async throws -> T) async throws -> (T, String) {
    let url = URL.temporaryDirectory.appending(path: "corellm-stderr-\(UUID().uuidString).log")
    FileManager.default.createFile(atPath: url.path(percentEncoded: false), contents: nil)
    let sink = try FileHandle(forWritingTo: url)
    let saved = dup(2)
    dup2(sink.fileDescriptor, 2)
    defer {
        dup2(saved, 2)
        close(saved)
        try? sink.close()
        try? FileManager.default.removeItem(at: url)
    }
    let value = try await body()
    return (value, (try? String(contentsOf: url, encoding: .utf8)) ?? "")
}

@Suite
struct SpeculationBudgetSuite {

    @Test
    func speculationNeedsRoomForEveryTokenItCanCommit() {
        #expect(!SpeculationPolicy.allowsRound(remaining: 3, maxAccept: 4))
        #expect(!SpeculationPolicy.allowsRound(remaining: 1, maxAccept: 2))
        #expect(SpeculationPolicy.allowsRound(remaining: 4, maxAccept: 4))
        #expect(SpeculationPolicy.allowsRound(remaining: 9, maxAccept: 4))
        #expect(SpeculationPolicy.allowsRound(remaining: 1, maxAccept: 1))
    }

    @Test
    func emptyPromptThrowsInsteadOfTrippingAPrecondition() {
        #expect(throws: (any Error).self) { try CoreMLEngine.requireNonEmptyPrompt([]) }
        #expect(throws: Never.self) { try CoreMLEngine.requireNonEmptyPrompt([1]) }
    }
}

@Suite(.serialized)
struct QwenHybridChainSuite {

    @Test(.enabled(if: qwen35Exists, qwen35Skip), .timeLimit(.minutes(10)))
    func hybridEnablesStaticVerifyMTPAndGuardsRewind() async throws {
        let url = try #require(qwen35URL)
        let chain = try await CoreMLChainV2(bundleURL: url, computeUnits: .cpuAndGPU)
        #expect(chain.config.isHybrid, "the qwen3_5 bundle is not detected as hybrid")
        #expect(chain.config.storeLayers.isEmpty, "qwen3_5 configs are not expected to list store_layers")
        #expect(chain.config.hasStaticVerify, "the verify block was not read from the config")
        #expect(chain.config.verifyMeta?.mode == "gdn", "unexpected verify mode: \(chain.config.verifyMeta?.mode ?? "nil")")
        #expect(chain.staticVerifyReady, "the verify function was not loaded next to main")
        #expect(chain.verifyWidth == 4, "verify width is \(chain.verifyWidth), expected 4")
        #expect(chain.maxAcceptedPerRound == chain.verifyWidth,
                "a hybrid round can commit up to verifyWidth tokens")
        #expect(chain.mtpHeadURL != nil, "mtp.mlmodelc was not found in the bundle")
        #expect(chain.supportsMTP, "a hybrid bundle with mtp.mlmodelc + verify should support speculation")

        try chain.reset()
        _ = try chain.prefill([9906, 1917, 0], blockSize: chain.config.maxS)
        let pos = chain.position
        #expect(pos == 3)
        try chain.rewind(to: pos)
        #expect(chain.position == pos)
        #expect(throws: (any Error).self) { try chain.rewind(to: pos - 1) }
    }

    @Test(.enabled(if: qwen35Exists, qwen35Skip), .timeLimit(.minutes(15)))
    func hybridEngineReusesCacheAcrossTurns() async throws {
        let url = try #require(qwen35URL)
        let bundle = try ModelBundle(contentsOf: url)
        #expect(bundle.manifest.assistantSuffix == "<|im_end|>\n", "manifest assistantSuffix was not read")
        #expect(bundle.manifest.eos == [248044, 248046], "manifest eos was not read")
        let engine = CoreMLEngine()
        try await engine.load(bundle, options: LoadOptions(computeUnits: .cpuAndGPU))

        let (text, count) = try await engine.canonicalPrompt(prompt: jaGreeting)
        let tokenizer = try await HFTokenizer(modelFolder: url, eosTokenIDs: [])
        let rawIDs = try tokenizer.encode(text)
        #expect(count == rawIDs.count, "a BOS token was inserted for a ChatML bundle")

        func turn(
            _ prompt: String, history: [ChatTurn] = [], raw: String? = nil, maxTokens: Int,
            emitSpecialTokens: Bool = false, speculative: Bool = false
        ) async throws -> (
            text: String, ids: [Int], reused: Int, promptTokens: Int, generated: Int,
            finish: FinishReason?
        ) {
            var out = ""
            var ids: [Int] = []
            var reused = 0
            var promptTokens = 0
            var generated = 0
            var finish: FinishReason?
            let request = GenerationRequest(
                prompt: prompt,
                config: GenerationConfig(
                    maxNewTokens: maxTokens, multiTokenPrediction: speculative,
                    emitSpecialTokens: emitSpecialTokens),
                history: history, reuseCache: true, rawPrompt: raw)
            for try await event in engine.generate(request) {
                switch event {
                case .token(let chunk): out += chunk.text; ids.append(chunk.tokenID)
                case .prefillCompleted(let p): reused = p.reusedTokens; promptTokens = p.promptTokens
                case .finished(let m): generated = m.generatedTokens; finish = m.finishReason
                default: break
                }
            }
            return (out, ids, reused, promptTokens, generated, finish)
        }

        await engine.resetConversation()
        let t1 = try await turn(jaCapitalQuestion, history: [], maxTokens: 16)
        #expect(!t1.text.isEmpty, "turn 1 produced no text")
        let history = [
            ChatTurn(role: .user, text: jaCapitalQuestion),
            ChatTurn(role: .assistant, text: t1.text),
        ]
        let t2 = try await turn(jaFollowUp, history: history, maxTokens: 16)
        #expect(!t2.text.isEmpty, "turn 2 produced no text")
        #expect(t2.reused > 0, "turn 2 reused no KV (the rewind no-op path was not taken)")
        #expect(t2.reused == t1.promptTokens + t1.generated,
                "turn 2 reused \(t2.reused) of the \(t1.promptTokens + t1.generated) tokens in the KV cache")
        print("[qwen 2turn] t1(reused=\(t1.reused)): \(t1.text)")
        print("[qwen 2turn] t2(reused=\(t2.reused)): \(t2.text)")

        await engine.resetConversation()
        let cut = try await turn(emojiRepeat, maxTokens: 20)
        let split = cut.text.unicodeScalars.last == "\u{FFFD}"
        let next = try await turn(
            emojiFollowUp,
            history: [
                ChatTurn(role: .user, text: emojiRepeat),
                ChatTurn(role: .assistant, text: cut.text),
            ],
            maxTokens: 8)
        print("[qwen split] cap cut a multi-byte character: \(split), "
              + "reused=\(next.reused)/\(cut.promptTokens + cut.generated)")
        #expect(next.reused == cut.promptTokens + cut.generated,
                "reused \(next.reused) of \(cut.promptTokens + cut.generated) after a split \(split) reply")

        await engine.resetConversation()
        let stopped = try await turn(eosQuestion, maxTokens: 160, speculative: true)
        #expect(stopped.finish == .eos,
                "turn 1 finished as \(stopped.finish?.rawValue ?? "?"); this gate needs an EOS ending")
        let cached = stopped.promptTokens + stopped.generated + 1
        let (resumed, diagnostics) = try await capturingStandardError {
            try await turn(
                eosFollowUp,
                history: [
                    ChatTurn(role: .user, text: eosQuestion),
                    ChatTurn(role: .assistant, text: stopped.text),
                ],
                maxTokens: 8, speculative: true)
        }
        #expect(resumed.reused == cached,
                "the EOS turn left \(cached) tokens in the KV cache, the next turn reused \(resumed.reused)")
        #expect(diagnostics.contains("reuse=text"),
                "the turn after an EOS turn did not continue from the text prefix: \(diagnostics)")
        #expect(!diagnostics.contains("cache reuse unavailable"),
                "the turn after an EOS turn re-prefilled the whole conversation: \(diagnostics)")
        let eosText = String((bundle.manifest.assistantSuffix ?? "").dropLast())
        let head = try await engine.canonicalPrompt(prompt: eosQuestion).text + stopped.text + eosText
        let whole = try await engine.canonicalPrompt(
            prompt: eosFollowUp,
            history: [
                ChatTurn(role: .user, text: eosQuestion),
                ChatTurn(role: .assistant, text: stopped.text),
            ]).text
        #expect(whole.utf8.starts(with: head.utf8), "the EOS text is not a prefix of the next prompt")
        let tail = try tokenizer.encode(
            String(decoding: whole.utf8.dropFirst(head.utf8.count), as: UTF8.self))
        let wholeIDs = try tokenizer.encode(whole)
        #expect(Array(wholeIDs.suffix(tail.count)) == tail,
                "encoding the suffix alone does not match the same span of the whole prompt")
        #expect(resumed.promptTokens == wholeIDs.count,
                "the continued prompt is \(resumed.promptTokens) tokens, the whole one \(wholeIDs.count)")
        print("[qwen eos] reused=\(resumed.reused)/\(cached) prompt=\(resumed.promptTokens)"
              + "/\(wholeIDs.count) suffix=\(tail.count) "
              + diagnostics.split(separator: "\n").filter { $0.hasPrefix("[v2]") }.joined())

        await engine.resetConversation()
        let raw = try await turn("ignored", raw: text, maxTokens: 1)
        #expect(raw.promptTokens == count,
                "rawPrompt tokenized to \(raw.promptTokens) tokens, the template path to \(count)")
        #expect(raw.reused == 0, "resetConversation() did not clear the KV state")

        await engine.resetConversation()
        let plain = try await turn(jaCapitalQuestion, maxTokens: 8)
        await engine.resetConversation()
        let special = try await turn(jaCapitalQuestion, maxTokens: 8, emitSpecialTokens: true)
        #expect(special.ids == plain.ids, "emitSpecialTokens changed which tokens were generated")
        #expect(special.text == (try tokenizer.decode(special.ids, skipSpecialTokens: false)),
                "GenerationConfig.emitSpecialTokens did not reach the tokenizer: \(special.text)")
        #expect(plain.text == (try tokenizer.decode(plain.ids, skipSpecialTokens: true)),
                "the default path did not skip special tokens: \(plain.text)")

        #expect(await engine.setThinking(true), "setThinking failed on a bundle with promptSuffixThinking")
        let (thinkingText, _) = try await engine.canonicalPrompt(prompt: jaGreeting)
        #expect(thinkingText.hasSuffix("<think>\n"), "thinking suffix not applied: \(thinkingText)")
        #expect(await engine.setThinking(false))
        let (restored, _) = try await engine.canonicalPrompt(prompt: jaGreeting)
        #expect(restored == text, "setThinking(false) did not restore the default prompt suffix")
        await engine.unload()
    }

    @Test(.enabled(if: qwen38TokenizerExists || qwen35Exists, qwen38Skip))
    func emitSpecialTokensKeepsChatMLMarkers() async throws {
        let url = try #require(qwen38TokenizerExists ? qwen38URL : qwen35URL)
        let tokenizer = try await HFTokenizer(modelFolder: url, eosTokenIDs: [])
        let ids = try tokenizer.encode("<|im_start|>assistant\n<think>hm</think>\n<tool_call>x</tool_call>")
        let skipped = try tokenizer.decode(ids, skipSpecialTokens: true)
        let kept = try tokenizer.decode(ids, skipSpecialTokens: false)

        #expect(!skipped.contains("<|im_start|>"), "skipSpecialTokens did not drop <|im_start|>")
        #expect(kept.contains("<|im_start|>"), "emitSpecialTokens did not keep <|im_start|>")
        for tag in ["<think>", "</think>", "<tool_call>", "</tool_call>"] {
            #expect(kept.contains(tag), "\(tag) is missing with skipSpecialTokens: false")
            #expect(skipped.contains(tag),
                    "\(tag) is dropped by skipSpecialTokens: true (it is a special token after all)")
        }
        print("[qwen special] skipped=\(skipped)")
        print("[qwen special] kept=\(kept)")
    }

    @Test(.enabled(if: qwen35Exists, qwen35Skip), .timeLimit(.minutes(60)))
    func speculationOffEqualsOnByteIdentical() async throws {
        let url = try #require(qwen35URL)
        let prompts = [
            ("ja", jaSelfIntro),
            ("en", "Explain in three sentences why speculative decoding is lossless."),
            ("code", "Write a Python function that reverses a string, then explain it in one sentence."),
        ]

        func run(speculative: Bool) async throws -> [(text: String, tokens: Int, acceptance: Double?)] {
            let engine = CoreMLEngine()
            try await engine.load(
                try ModelBundle(contentsOf: url), options: LoadOptions(computeUnits: .cpuAndGPU))
            #expect(await engine.supportsSpeculation, "the bundle does not advertise speculation")
            var out: [(text: String, tokens: Int, acceptance: Double?)] = []
            for (_, prompt) in prompts {
                var text = ""
                var tokens = 0
                var acceptance: Double?
                let request = GenerationRequest(
                    prompt: prompt,
                    config: GenerationConfig(maxNewTokens: 64, multiTokenPrediction: speculative),
                    history: [], reuseCache: false)
                for try await event in engine.generate(request) {
                    switch event {
                    case .token(let chunk): text += chunk.text
                    case .finished(let m): tokens = m.generatedTokens; acceptance = m.draftAcceptanceRate
                    default: break
                    }
                }
                out.append((text, tokens, acceptance))
            }
            await engine.unload()
            return out
        }

        let off = try await run(speculative: false)
        let on = try await run(speculative: true)
        for (i, (name, _)) in prompts.enumerated() {
            print("[gate \(name)] off=\(off[i].tokens) tok  on=\(on[i].tokens) tok  "
                  + "acceptance=\(on[i].acceptance.map { String(format: "%.2f", $0) } ?? "-")")
            #expect(Array(on[i].text.utf8) == Array(off[i].text.utf8),
                    "speculation ON and OFF diverged on \(name)\nOFF: \(off[i].text)\nON:  \(on[i].text)")
            #expect(on[i].tokens == off[i].tokens,
                    "speculation ON emitted \(on[i].tokens) tokens, OFF emitted \(off[i].tokens)")
            let rate = try #require(on[i].acceptance, "no draft acceptance was reported for \(name)")
            #expect(rate > 0, "no draft was ever accepted on \(name)")
            #expect(rate < 1, "every draft was accepted on \(name); the write-back path was never exercised")
        }
    }

    @Test(.enabled(if: qwen35Exists, qwen35Skip), .timeLimit(.minutes(30)))
    func staticVerifySlotWriteBackMatchesSequential() async throws {
        let url = try #require(qwen35URL)
        let chain = try await CoreMLChainV2(bundleURL: url, computeUnits: .cpuAndGPU)
        try #require(chain.staticVerifyReady, "this gate is meaningless without a static verify function")
        let tokenizer = try await HFTokenizer(modelFolder: url, eosTokenIDs: [])
        let prompt = try tokenizer.encode(jaSelfIntro)
        let width = chain.verifyWidth
        let cont = 6

        func greedy(_ count: Int) throws -> [Int] {
            try chain.reset()
            var next = try chain.prefill(prompt, blockSize: chain.config.maxS)
            var out: [Int] = []
            for _ in 0..<count {
                out.append(next)
                next = try chain.decodeStep(tokenID: next)
            }
            return out
        }
        let seq = try greedy(width + cont)

        for accepted in 1..<width {
            let want = Array(seq[(accepted + 1)...(accepted + cont)])
            func run(writeBack: Bool) throws -> [Int] {
                try chain.reset()
                _ = try chain.prefill(prompt, blockSize: chain.config.maxS)
                _ = try chain.verifyForward(tokens: Array(seq[0..<width]), basePosition: chain.position)
                chain.commitVerified(accepted, writeBackState: writeBack)
                var token = seq[accepted]
                var got: [Int] = []
                for _ in 0..<cont {
                    token = try chain.decodeStep(tokenID: token)
                    got.append(token)
                }
                return got
            }
            let got = try run(writeBack: true)
            #expect(got == want,
                    "slot \(accepted - 1) write-back diverged from the sequential trace: \(got) != \(want)")
            let control = try run(writeBack: false)
            print("[slot accepted=\(accepted)] writeback=\(got == want) control=\(control == want)")
            #expect(control != want,
                    "skipping the write-back still matched; this gate cannot detect a no-op write-back")
        }
    }

    @Test(.enabled(if: gemmaTokenizerExists, gemmaSkip))
    func gemmaStillDeclaresBOS() async throws {
        let url = try #require(gemmaURL)
        let gemma = try await HFTokenizer(modelFolder: url, eosTokenIDs: [])
        #expect(gemma.bosTokenID == 2, "Gemma's bosTokenID is not 2; BOS insertion would regress")
        if let qwen38URL, qwen38TokenizerExists {
            let qwen = try await HFTokenizer(modelFolder: qwen38URL, eosTokenIDs: [])
            #expect(qwen.bosTokenID == nil, "Qwen declares a BOS token; it would be inserted")
        }
    }
}
