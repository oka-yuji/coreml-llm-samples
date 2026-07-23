import CoreML
import Foundation
import LLMCore

public actor CoreMLEngine: LLMEngine {
    public nonisolated let descriptor = EngineDescriptor(
        name: "Core ML",
        backend: .coreML,
        supportsMultiTokenPrediction: true
    )

    private var chain: (any GenerationChain)?

    private var speculative: (any SpeculativeDecoding)?
    private var tokenizer: HFTokenizer?
    private var pendingLoadMetrics: LoadMetrics?
    private var promptPrefix = "<start_of_turn>user\n"
    private var promptSuffix = "<end_of_turn>\n<start_of_turn>model\n"

    private var assistantSuffix = "<end_of_turn>\n"

    private var processedTokens: [Int] = []
    private var loadedBundleURL: URL?
    private var loadedUnits: MLComputeUnits = .cpuOnly

    public init() {}

    public func load(_ model: ModelBundle, options: LoadOptions) async throws {
        guard chain == nil else { return }
        let clock = ContinuousClock()
        let start = clock.now

        let units: MLComputeUnits = switch options.computeUnits {
        case .all: .all
        case .cpuAndGPU: .cpuAndGPU
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        case .cpuOnly: .cpuOnly
        }

        if model.manifest.format == "coreml-stateful-chain-v2" {

            guard units == .cpuAndGPU || units == .all else {
                throw LLMEngineError.incompatibleBundle(
                    reason: "v2 stateful bundles require the GPU (computeUnits=\(options.computeUnits.rawValue)). "
                        + "CPU_ONLY / ANE crash with a native segfault when multiple MLState instances load together. "
                        + "Specify cpuAndGPU or all")
            }
            let v2 = try await CoreMLChainV2(bundleURL: model.directoryURL, computeUnits: units)
            tokenizer = try await HFTokenizer(
                modelFolder: model.directoryURL, eosTokenIDs: Set(v2.config.eosIDs))
            chain = v2
            speculative = v2
        } else {
            let v1 = try await CoreMLChain(bundleURL: model.directoryURL, computeUnits: units)
            tokenizer = try await HFTokenizer(
                modelFolder: model.directoryURL, eosTokenIDs: Set(v1.config.eosIDs))
            chain = v1
            speculative = v1
        }
        loadedBundleURL = model.directoryURL
        loadedUnits = units
        if let prefix = model.manifest.promptPrefix { promptPrefix = prefix }
        if let suffix = model.manifest.promptSuffix { promptSuffix = suffix }
        assistantSuffix = Self.deriveAssistantSuffix(prefix: promptPrefix, suffix: promptSuffix)
        processedTokens = []
        pendingLoadMetrics = LoadMetrics(duration: clock.now - start)
    }

    public func unload() {
        chain = nil
        speculative = nil
        tokenizer = nil
        pendingLoadMetrics = nil
        processedTokens = []
    }

    public func resetConversation() {
        try? chain?.reset()
        processedTokens = []
    }

    func setPromptTemplate(prefix: String, suffix: String) {
        promptPrefix = prefix
        promptSuffix = suffix
        assistantSuffix = Self.deriveAssistantSuffix(prefix: prefix, suffix: suffix)
    }

    struct SkeletonExtractionResult: Sendable {
        let json: String
        let promptText: String
        let promptTokens: Int
        let injectedTokens: Int
        let valueTokens: Int
        let fieldDemotions: Int
        let headSkips: Int
        let genMillis: Double
        let fields: [FieldRender]
        let parseOK: Bool
        let keysComplete: Bool
    }

    struct FieldRender: Sendable {
        let key: String
        let rendered: String
    }

    func extractSkeletonJSON(
        prompt: String,
        schema: SkeletonSchema,
        history: [ChatTurn] = [],
        maxStringSpanTokens: Int = 48,
        maxNumberSpanTokens: Int = 16,
        useHeadSkip: Bool = true
    ) throws -> SkeletonExtractionResult {
        guard let chain, let tokenizer else { throw LLMEngineError.notLoaded }
        let promptText = conversationText(history: history, prompt: prompt)
        let promptIDs = try encodeConversation(history: history, prompt: prompt)
        guard promptIDs.count < chain.contextLength else {
            throw LLMEngineError.generationFailed(
                reason: "prompt (\(promptIDs.count) tokens) exceeds context length \(chain.contextLength)")
        }
        let decoder = try SkeletonJSONDecoder(
            schema: schema, tokenizer: tokenizer,
            maxStringSpanTokens: maxStringSpanTokens, maxNumberSpanTokens: maxNumberSpanTokens,
            useHeadSkip: useHeadSkip)
        let clock = ContinuousClock()
        let start = clock.now
        try chain.reset()
        let firstPending = try chain.prefill(promptIDs)
        let (values, stats) = try decoder.decode(chain: chain, firstPending: firstPending)
        let json = decoder.assembleJSON(values)
        let genMs = (clock.now - start) / .seconds(1) * 1000
        let (parseOK, keysComplete) = SkeletonJSONDecoder.selfCheck(json: json, schema: schema)
        let rendered: [FieldRender] = values.map { key, value in
            let text: String
            switch value {
            case .null: text = "null"
            case .string(let s): text = s
            case .enumeration(let s): text = s
            case .number(let n): text = n
            }
            return FieldRender(key: key, rendered: text)
        }
        return SkeletonExtractionResult(
            json: json, promptText: promptText, promptTokens: promptIDs.count,
            injectedTokens: stats.injectedTokens, valueTokens: stats.valueTokens,
            fieldDemotions: stats.fieldDemotions, headSkips: stats.headSkips, genMillis: genMs,
            fields: rendered, parseOK: parseOK, keysComplete: keysComplete)
    }

    func canonicalPrompt(prompt: String, history: [ChatTurn] = []) throws -> (text: String, tokenCount: Int) {
        let text = conversationText(history: history, prompt: prompt)
        let ids = try encodeConversation(history: history, prompt: prompt)
        return (text, ids.count)
    }

    func pldStatsSnapshot() -> CoreMLChainV2.PLDStats? {
        (chain as? CoreMLChainV2)?.pldStatsSnapshot()
    }

    func treeStatsSnapshot() -> CoreMLChainV2.TreeStats? {
        (chain as? CoreMLChainV2)?.treeStatsSnapshot()
    }

    public nonisolated func generate(
        _ request: GenerationRequest
    ) -> AsyncThrowingStream<GenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(request, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(
        _ request: GenerationRequest,
        continuation: AsyncThrowingStream<GenerationEvent, any Error>.Continuation
    ) async throws {
        guard let chain, let tokenizer else { throw LLMEngineError.notLoaded }

        let clock = ContinuousClock()
        let start = clock.now
        if let metrics = pendingLoadMetrics {
            continuation.yield(.loadCompleted(metrics))
            pendingLoadMetrics = nil
        }

        let promptIDs = try encodeConversation(history: request.history, prompt: request.prompt)

        guard promptIDs.count < chain.contextLength else {
            throw LLMEngineError.generationFailed(
                reason: "prompt (\(promptIDs.count) tokens) exceeds context length \(chain.contextLength)"
            )
        }
        let maxNewTokens = min(request.config.maxNewTokens, chain.contextLength - promptIDs.count)

        let useMTP = request.config.multiTokenPrediction && (speculative?.supportsMTP ?? false)

        let reusedTokens: Int
        if request.reuseCache {

            let lcp = max(0, min(Self.commonPrefixLength(promptIDs, processedTokens), promptIDs.count - 1))
            chain.rewind(to: lcp)

            processedTokens = Array(promptIDs[0..<lcp])
            reusedTokens = lcp
        } else {
            try chain.reset()
            processedTokens = []
            reusedTokens = 0
        }
        if useMTP { try await installMTPIfNeeded() }
        let diff = reusedTokens > 0 ? Array(promptIDs[reusedTokens...]) : promptIDs
        var nextToken = try chain.prefill(diff)

        processedTokens = promptIDs
        continuation.yield(.prefillCompleted(
            PrefillMetrics(
                promptTokens: promptIDs.count, reusedTokens: reusedTokens, duration: clock.now - start)
        ))

        let decodeStart = clock.now
        var firstTokenAt: ContinuousClock.Instant?
        var generated: [Int] = []
        var emittedText = ""
        var acceptedTotal = 0
        var draftedTotal = 0

        func emit(_ token: Int) throws {
            generated.append(token)
            let now = clock.now
            if firstTokenAt == nil { firstTokenAt = now }
            let fullText = try tokenizer.decode(generated)
            let delta = fullText.hasPrefix(emittedText) ? String(fullText.dropFirst(emittedText.count)) : fullText
            emittedText = fullText
            guard !delta.isEmpty else { return }
            let elapsed = (now - decodeStart) / .seconds(1)
            continuation.yield(.token(TokenChunk(
                text: delta,
                tokenID: token,
                tokensPerSecond: Double(generated.count) / max(elapsed, 0.001)
            )))
        }

        decodeLoop: while generated.count < maxNewTokens {
            try Task.checkCancellation()
            if tokenizer.eosTokenIDs.contains(nextToken) { break }

            if useMTP, let spec = speculative {

                let round = try spec.mtpRound(prediction: nextToken, context: processedTokens)
                acceptedTotal += round.accepted
                draftedTotal += round.drafted

                processedTokens.append(contentsOf: round.emitted)
                for token in round.emitted {
                    try emit(token)
                    if tokenizer.eosTokenIDs.contains(token) || generated.count >= maxNewTokens {
                        break decodeLoop
                    }
                }
                nextToken = round.next
            } else {
                let current = nextToken
                try emit(current)
                nextToken = try chain.decodeStep(tokenID: current)
                processedTokens.append(current)
            }
        }

        let decodeSeconds = (clock.now - decodeStart) / .seconds(1)
        continuation.yield(.finished(GenerationMetrics(
            promptTokens: promptIDs.count,
            generatedTokens: generated.count,
            timeToFirstToken: (firstTokenAt ?? decodeStart) - start,
            decodeTokensPerSecond: Double(generated.count) / max(decodeSeconds, 0.001),
            peakMemoryBytes: Self.memoryFootprint(),
            draftAcceptanceRate: draftedTotal > 0 ? Double(acceptedTotal) / Double(draftedTotal) : nil
        )))
    }

    private func conversationText(history: [ChatTurn], prompt: String) -> String {
        var text = ""
        for turn in history {
            switch turn.role {
            case .user: text += promptPrefix + turn.text + promptSuffix
            case .assistant: text += turn.text + assistantSuffix
            }
        }
        text += promptPrefix + prompt + promptSuffix
        return text
    }

    private func encodeConversation(history: [ChatTurn], prompt: String) throws -> [Int] {
        guard let tokenizer else { throw LLMEngineError.notLoaded }
        var ids = try tokenizer.encode(conversationText(history: history, prompt: prompt))
        let bos = tokenizer.bosTokenID ?? 2
        if ids.first != bos { ids.insert(bos, at: 0) }
        return ids
    }

    static func deriveAssistantSuffix(prefix: String, suffix: String) -> String {
        guard let userRange = prefix.range(of: "user") else { return suffix }
        let roleOpen = String(prefix[..<userRange.lowerBound])
        guard !roleOpen.isEmpty, let modelRange = suffix.range(of: roleOpen + "model") else { return suffix }
        return String(suffix[..<modelRange.lowerBound])
    }

    static func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }

    private func installMTPIfNeeded() async throws {
        guard let bundleURL = loadedBundleURL else { return }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = loadedUnits

        if let v1 = speculative as? CoreMLChain, !v1.mtpLoaded, let mtp = v1.config.mtp {
            let drafter = try await CoreMLChain.loadCompiled(bundleURL: bundleURL, name: mtp.drafter, configuration: cfg)
            var verify: [MLModel] = []
            for name in mtp.verifyChunks {
                verify.append(try await CoreMLChain.loadCompiled(bundleURL: bundleURL, name: name, configuration: cfg))
            }
            let verifyHead = try await CoreMLChain.loadCompiled(bundleURL: bundleURL, name: mtp.verifyLmhead, configuration: cfg)
            try v1.installMTP(drafter: drafter, verifyChunks: verify, verifyHead: verifyHead)
        } else if let v2 = speculative as? CoreMLChainV2, !v2.mtpLoaded, let drafterURL = v2.drafterURL {

            let drafterCfg = MLModelConfiguration()
            drafterCfg.computeUnits = Self.resolveDrafterComputeUnits(
                default: loadedUnits == .all ? .cpuAndGPU : loadedUnits)
            let drafter = try await CoreMLChain.loadCompiled(
                bundleURL: drafterURL.deletingLastPathComponent(),
                name: drafterURL.lastPathComponent, configuration: drafterCfg)

            var drafterPre: MLModel? = nil
            if let preURL = v2.drafterPreURL {
                drafterPre = try await CoreMLChain.loadCompiled(
                    bundleURL: preURL.deletingLastPathComponent(),
                    name: preURL.lastPathComponent, configuration: drafterCfg)
            }
            try v2.installMTP(drafter: drafter, drafterPre: drafterPre)
        }
    }

    private static func resolveDrafterComputeUnits(default fallback: MLComputeUnits) -> MLComputeUnits {
        switch ProcessInfo.processInfo.environment["CORELLM_MTP_DRAFTER_CU"]?.lowercased() {
        case "all": return .all
        case "gpu": return .cpuAndGPU
        case "ane": return .cpuAndNeuralEngine
        case "cpu": return .cpuOnly
        default: return fallback
        }
    }

    private static func memoryFootprint() -> Int? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) : nil
    }
}
