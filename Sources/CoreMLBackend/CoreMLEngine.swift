import CoreML
import Foundation
import LLMCore
import os

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

    public var supportsSpeculation: Bool { speculative?.supportsMTP ?? false }

    public var speculationLoaded: Bool { speculative?.mtpLoaded ?? false }

    public func prepareSpeculation() async throws {
        guard let chunked = chain as? ChunkedSpeculativeChain else { return }
        try await loadChunkedVerifyIfNeeded(chunked)
    }

    private func loadChunkedVerifyIfNeeded(_ chunked: ChunkedSpeculativeChain) async throws {
        guard chunked.supportsMTP, !chunked.mtpLoaded,
              let bundleURL = loadedBundleURL,
              let pkg = chunked.verifyHeadPackageName else { return }
        let compiled = try await ChunkedSpeculativeChain.compileIfNeeded(bundleURL: bundleURL, name: pkg)
        let model = try ChunkedSpeculativeChain.loadFunction(
            compiledURL: compiled, functionName: nil, computeUnits: chunked.verifyHeadCU)
        try chunked.installVerifyHead(model)
    }

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

        if model.manifest.format == ChunkedSpeculativeChain.format {
            let r = try await ChunkedSpeculativeChain(
                bundleURL: model.directoryURL, computeUnits: units,
                sidecarStage: model.manifest.sidecarStage,
                preloadVerifyAssets: options.preloadSpeculation)
            tokenizer = try await HFTokenizer(
                modelFolder: model.directoryURL, eosTokenIDs: Set(r.config.eosIDs))
            chain = r
            speculative = r
        } else if model.manifest.format == "coreml-stateful-chain-v2" {

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

    public func kvSave(to url: URL, prompt: String, warmupDecodes: Int = 0) async throws -> KVCheckpointInfo {
        guard let chain = chain as? ChunkedSpeculativeChain else {
            throw LLMEngineError.incompatibleBundle(reason: "KV persistence requires a \(ChunkedSpeculativeChain.format) bundle")
        }
        let ids = try encodeConversation(history: [], prompt: prompt)
        let clock = ContinuousClock()
        try chain.reset()
        let t0 = clock.now
        var pending = try chain.prefillScheduled(ids)
        let prefillSec = (clock.now - t0) / .seconds(1)
        var processed = ids
        for _ in 0..<max(0, warmupDecodes) {
            processed.append(pending)
            pending = try chain.decodeStep(tokenID: pending)
        }
        let e0 = clock.now
        let manifest = try chain.exportKV(to: url, pendingNextToken: pending, processedTokens: processed)
        let exportSec = (clock.now - e0) / .seconds(1)
        processedTokens = processed
        let bytes = manifest.layers.reduce(0) { $0 + $1.kBytes + $1.vBytes }
        return KVCheckpointInfo(
            position: manifest.position, tokenCount: manifest.tokenCount, pendingNextToken: pending,
            fileBytes: bytes, layerCount: manifest.layers.count,
            prefillSeconds: prefillSec, exportSeconds: exportSec, importSeconds: nil,
            residentPrefillWidths: chain.residentPrefillWidths(), continuation: nil, continuationText: nil,
            peakMemoryBytes: Self.memoryFootprint())
    }

    public func kvRestoreAndContinue(
        from url: URL, verifyPrompt: String?, maxNew: Int, speculative useSpec: Bool = false
    ) async throws -> KVCheckpointInfo {
        guard let chain = chain as? ChunkedSpeculativeChain, let tokenizer else {
            throw LLMEngineError.incompatibleBundle(reason: "KV persistence requires a \(ChunkedSpeculativeChain.format) bundle")
        }
        chain.setBlockScheduledPrefill(true)
        defer { chain.setBlockScheduledPrefill(false) }
        let expected: [Int]? = try verifyPrompt.map { try encodeConversation(history: [], prompt: $0) }
        let clock = ContinuousClock()
        let i0 = clock.now
        let manifest = try chain.importKV(from: url, expectedContext: expected)
        let importSec = (clock.now - i0) / .seconds(1)
        processedTokens = manifest.processedTokens
        let seed = manifest.pendingNextToken ?? 0
        let eos = Set(tokenizer.eosTokenIDs)
        let budget = max(0, chain.contextLength - manifest.position - 8)
        let effectiveMaxNew = maxNew > 0 ? min(maxNew, budget) : budget
        let cont = useSpec
            ? try chain.continueSpeculative(seed: seed, context: manifest.processedTokens, maxNew: effectiveMaxNew, eos: eos)
            : try chain.continueGreedy(seed: seed, maxNew: effectiveMaxNew, eos: eos)
        let bytes = manifest.layers.reduce(0) { $0 + $1.kBytes + $1.vBytes }
        let pld = useSpec ? chain.pldStatsSnapshot() : nil
        return KVCheckpointInfo(
            position: manifest.position, tokenCount: manifest.tokenCount, pendingNextToken: seed,
            fileBytes: bytes, layerCount: manifest.layers.count,
            prefillSeconds: nil, exportSeconds: nil, importSeconds: importSec,
            residentPrefillWidths: chain.residentPrefillWidths(),
            continuation: cont, continuationText: try? tokenizer.decode(cont),
            peakMemoryBytes: Self.memoryFootprint(),
            pldRounds: pld?.pldRounds, pldFallbackRounds: pld?.fallbackRounds,
            pldDraftedTokens: pld?.draftedTokens, pldAcceptedTokens: pld?.acceptedTokens)
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
            throw LLMEngineError.contextOverflow(
                promptTokens: promptIDs.count, contextLength: chain.contextLength)
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
            throw LLMEngineError.contextOverflow(
                promptTokens: promptIDs.count, contextLength: chain.contextLength)
        }
        let thermalStart = Self.thermalName()
        let margin = 8
        let budget = max(0, chain.contextLength - promptIDs.count - margin)
        let requested = request.config.maxNewTokens
        let cap = requested > 0 ? min(requested, budget) : budget
        let capIsContextBound = !(requested > 0 && requested <= budget)

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
        let prefillSeconds = (clock.now - start) / .seconds(1)
        let footprintAfterPrefill = Self.memoryFootprint()

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
        var tokenInstants: [ContinuousClock.Instant] = []
        var sawEOS = false

        func emit(_ token: Int) throws {
            generated.append(token)
            let now = clock.now
            tokenInstants.append(now)
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

        decodeLoop: while generated.count < cap {
            try Task.checkCancellation()
            if tokenizer.eosTokenIDs.contains(nextToken) { sawEOS = true; break }

            if useMTP, let spec = speculative {

                let round = try spec.mtpRound(prediction: nextToken, context: processedTokens)
                acceptedTotal += round.accepted
                draftedTotal += round.drafted

                processedTokens.append(contentsOf: round.emitted)
                for token in round.emitted {
                    try emit(token)
                    if tokenizer.eosTokenIDs.contains(token) { sawEOS = true; break decodeLoop }
                    if generated.count >= cap { break decodeLoop }
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
        let finishReason: FinishReason = sawEOS ? .eos : (capIsContextBound ? .contextFull : .cap)
        let footprintAtEnd = Self.memoryFootprint()
        let pld = useMTP ? (chain as? ChunkedSpeculativeChain)?.pldStatsSnapshot() : nil
        let widths = (chain as? ChunkedSpeculativeChain)?.residentPrefillWidths()
        var perTokenMillis: [Double] = []
        var previous = decodeStart
        for instant in tokenInstants {
            perTokenMillis.append((instant - previous) / .seconds(1) * 1000)
            previous = instant
        }
        let peak = [footprintAfterPrefill, footprintAtEnd].compactMap { $0 }.max()
        continuation.yield(.finished(GenerationMetrics(
            promptTokens: promptIDs.count,
            generatedTokens: generated.count,
            timeToFirstToken: (firstTokenAt ?? decodeStart) - start,
            decodeTokensPerSecond: Double(generated.count) / max(decodeSeconds, 0.001),
            peakMemoryBytes: peak,
            draftAcceptanceRate: draftedTotal > 0 ? Double(acceptedTotal) / Double(draftedTotal) : nil,
            reusedTokens: reusedTokens,
            finishReason: finishReason,
            prefillSeconds: prefillSeconds,
            perTokenMillis: perTokenMillis,
            footprintAfterPrefillBytes: footprintAfterPrefill,
            footprintAtEndBytes: footprintAtEnd,
            availableMemoryBytes: Self.availableMemory(),
            thermalStateStart: thermalStart,
            thermalStateEnd: Self.thermalName(),
            specEnabled: useMTP,
            specRounds: pld?.pldRounds,
            specDrafted: draftedTotal > 0 ? draftedTotal : nil,
            specAccepted: draftedTotal > 0 ? acceptedTotal : nil,
            specFallbackRounds: pld?.fallbackRounds,
            feedWidths: widths
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
        if let chunked = chain as? ChunkedSpeculativeChain {
            try await loadChunkedVerifyIfNeeded(chunked)
            return
        }
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

    private static func availableMemory() -> Int? {
        #if os(iOS)
        return Int(os_proc_available_memory())
        #else
        return nil
        #endif
    }

    private static func thermalName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
