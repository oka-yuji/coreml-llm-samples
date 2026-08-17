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
    private var liveVisionEncoder: VisionEncoder?

    public static let defaultLiveVisionComputeUnits: ComputeUnitPreference = {
        #if os(macOS)
        return .cpuAndGPU
        #else
        return .all
        #endif
    }()

    private var liveVisionUnits: ComputeUnitPreference = CoreMLEngine.defaultLiveVisionComputeUnits

    public init() {}

    public func setLiveVisionComputeUnits(_ preference: ComputeUnitPreference) async {
        guard preference != liveVisionUnits else { return }
        liveVisionUnits = preference
        await liveVisionEncoder?.unload()
        liveVisionEncoder = nil
    }

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

        let units = Self.mlComputeUnits(options.computeUnits)

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
        liveVisionEncoder = nil
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

    private var multimodalChain: (any MultimodalChain)? { chain as? any MultimodalChain }

    private func requireMultimodalChain() throws -> any MultimodalChain {
        guard let mm = multimodalChain else {
            throw LLMEngineError.incompatibleBundle(
                reason: "image input requires a \(ChunkedSpeculativeChain.format) or "
                    + "coreml-stateful-chain-v2 bundle")
        }
        return mm
    }

    public func supportsImageInput() -> Bool {
        multimodalChain != nil && visionModelURL() != nil
    }

    private func visionModelURL() -> URL? {
        Self.sidecarModelURL(bundleURL: loadedBundleURL, name: "vision_fp16")
    }

    private enum VLMImageSource {
        case url(URL)
        case frame(LiveFrameImage)
        case encoded(LiveEncodedFrame)
    }

    private func softTokens(
        for image: VLMImageSource
    ) async throws -> (soft: SoftTokenRows, seconds: Double) {
        let held = liveVisionEncoder
        let clock = ContinuousClock()
        let t0 = clock.now
        switch image {
        case .encoded(let ready):
            return (ready.soft, ready.encodeSeconds)
        case .url(let url):
            let encoder = try held ?? makeVisionEncoder()
            let soft = try await encoder.encode(imageAt: url, releaseAfter: held == nil)
            return (soft, (clock.now - t0) / .seconds(1))
        case .frame(let frame):
            let encoder = try held ?? makeVisionEncoder()
            let soft = try await encoder.encode(image: frame.cgImage, releaseAfter: held == nil)
            return (soft, (clock.now - t0) / .seconds(1))
        }
    }

    private func prepareVLMSegments(
        image: VLMImageSource, question: String
    ) async throws -> (segments: [PromptSegment], flatIDs: [Int], imageRows: Int, visionSeconds: Double) {
        guard let tokenizer else { throw LLMEngineError.notLoaded }
        let hidden = try requireMultimodalChain().hiddenSize
        let (soft, visionSeconds) = try await softTokens(for: image)
        guard soft.hidden == hidden else {
            throw LLMEngineError.incompatibleBundle(
                reason: "the vision encoder emits \(soft.hidden)-wide soft tokens but this bundle's "
                    + "hidden size is \(hidden) — vision_fp16 does not belong to this model")
        }
        let bos = tokenizer.bosTokenID ?? 2
        let userTokens = try tokenizer.encode("user\n")
        let questionTokens = try tokenizer.encode(question)
        let modelTokens = try tokenizer.encode("model\n")
        let segments = VLMPrompt.segments(
            bos: bos, userTokens: userTokens, questionTokens: questionTokens,
            modelTokens: modelTokens, image: soft)
        let flatIDs = VLMPrompt.flatIDs(
            bos: bos, userTokens: userTokens, questionTokens: questionTokens,
            modelTokens: modelTokens, imageRows: soft.rows)
        return (segments, flatIDs, soft.rows, visionSeconds)
    }

    private static func mlComputeUnits(_ preference: ComputeUnitPreference) -> MLComputeUnits {
        switch preference {
        case .all: .all
        case .cpuAndGPU: .cpuAndGPU
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        case .cpuOnly: .cpuOnly
        }
    }

    private func makeVisionEncoder(computeUnits: MLComputeUnits) throws -> VisionEncoder {
        guard let visionURL = visionModelURL() else {
            throw LLMEngineError.modelNotFound(
                path: "vision_fp16.mlpackage (absent next to the loaded bundle)")
        }
        return VisionEncoder(packageURL: visionURL, computeUnits: computeUnits)
    }

    private func makeVisionEncoder() throws -> VisionEncoder {
        try makeVisionEncoder(computeUnits: loadedUnits)
    }

    public func loadLiveVisionEncoder() async throws -> LiveVisionEncoderInfo {
        _ = try requireMultimodalChain()
        let clock = ContinuousClock()
        let t0 = clock.now
        let encoder = try liveVisionEncoder
            ?? makeVisionEncoder(computeUnits: Self.mlComputeUnits(liveVisionUnits))
        try await encoder.loadIfNeeded()
        liveVisionEncoder = encoder
        guard let rows = await encoder.softTokenRowCount() else {
            throw LLMEngineError.incompatibleBundle(
                reason: "the vision encoder does not declare a fixed soft_tokens row count")
        }
        let warmUpSeconds = try await encoder.warmUpIfNeeded()
        return LiveVisionEncoderInfo(
            imageRows: rows, seconds: (clock.now - t0) / .seconds(1),
            warmUpSeconds: warmUpSeconds, computeUnits: await encoder.computeUnitsLabel)
    }

    public func prepareLivePrefill(
        question: String, imageRows: Int
    ) throws -> LiveVisionPrefillInfo {
        let mm = try requireMultimodalChain()
        guard let tokenizer else { throw LLMEngineError.notLoaded }
        let flatIDs = VLMPrompt.flatIDs(
            bos: tokenizer.bosTokenID ?? 2,
            userTokens: try tokenizer.encode("user\n"),
            questionTokens: try tokenizer.encode(question),
            modelTokens: try tokenizer.encode("model\n"),
            imageRows: imageRows)
        guard flatIDs.count < mm.contextLength else {
            throw LLMEngineError.contextOverflow(
                promptTokens: flatIDs.count, contextLength: mm.contextLength)
        }
        let widths = mm.plannedPrefillWidths(promptLength: flatIDs.count, from: 0)
        let clock = ContinuousClock()
        let t0 = clock.now
        try mm.materializePrefill(widths: widths)
        return LiveVisionPrefillInfo(
            promptTokens: flatIDs.count, prefillWidths: widths,
            seconds: (clock.now - t0) / .seconds(1))
    }

    public func beginLiveVisionSession(question: String) async throws -> LiveVisionPrewarm {
        let encoder = try await loadLiveVisionEncoder()
        let prefill = try prepareLivePrefill(question: question, imageRows: encoder.imageRows)
        return LiveVisionPrewarm(encoder: encoder, prefill: prefill)
    }

    public func liveVisionEncodeHandle() async throws -> LiveVisionEncodeHandle {
        _ = try await loadLiveVisionEncoder()
        guard let encoder = liveVisionEncoder else {
            throw LLMEngineError.generationFailed(
                reason: "the live vision encoder is not resident")
        }
        return LiveVisionEncodeHandle(encoder: encoder)
    }

    public func endLiveVisionSession() async {
        await liveVisionEncoder?.unload()
        liveVisionEncoder = nil
    }

    public func residentPrefillWidths() -> [Int] {
        multimodalChain?.residentPrefillWidths() ?? []
    }

    public func generateWithImage(
        imageURL: URL, question: String, maxNew: Int, speculative useSpec: Bool
    ) async throws -> VLMGenerationInfo {
        try await runImageTurn(
            image: .url(imageURL), question: question, maxNew: maxNew,
            speculative: useSpec, onPhase: nil)
    }

    public func generateWithImage(
        frame: LiveFrameImage, question: String, maxNew: Int, speculative useSpec: Bool,
        onPhase: (@Sendable (VLMPhase) -> Void)? = nil,
        onPartial: (@Sendable (String, Int) -> Void)? = nil
    ) async throws -> VLMGenerationInfo {
        try await runImageTurn(
            image: .frame(frame), question: question, maxNew: maxNew,
            speculative: useSpec, onPhase: onPhase, onPartial: onPartial)
    }

    public func generateWithEncodedFrame(
        _ encoded: LiveEncodedFrame, question: String, maxNew: Int, speculative useSpec: Bool,
        onPhase: (@Sendable (VLMPhase) -> Void)? = nil,
        onPartial: (@Sendable (String, Int) -> Void)? = nil
    ) async throws -> VLMGenerationInfo {
        try await runImageTurn(
            image: .encoded(encoded), question: question, maxNew: maxNew,
            speculative: useSpec, onPhase: onPhase, onPartial: onPartial)
    }

    public static let partialUpdateTokens = 4

    private func runImageTurn(
        image: VLMImageSource, question: String, maxNew: Int, speculative useSpec: Bool,
        onPhase: (@Sendable (VLMPhase) -> Void)?,
        onPartial: (@Sendable (String, Int) -> Void)? = nil
    ) async throws -> VLMGenerationInfo {
        let mm = try requireMultimodalChain()
        guard let tokenizer else { throw LLMEngineError.notLoaded }
        onPhase?(.encode)
        let prep = try await prepareVLMSegments(image: image, question: question)
        guard prep.flatIDs.count < mm.contextLength else {
            throw LLMEngineError.contextOverflow(
                promptTokens: prep.flatIDs.count, contextLength: mm.contextLength)
        }
        let speculating = (mm as? ChunkedSpeculativeChain).map { useSpec && $0.supportsMTP } ?? false
        if speculating { try await installMTPIfNeeded() }
        let eos = Set(tokenizer.eosTokenIDs)
        let clock = ContinuousClock()
        try mm.reset()
        onPhase?(.feed)
        let p0 = clock.now
        let seed = try mm.prefillSegments(prep.segments)
        let prefillSeconds = (clock.now - p0) / .seconds(1)
        let cap = Self.tokenBudget(
            maxNew: maxNew, used: prep.flatIDs.count, contextLength: mm.contextLength)
        onPhase?(.generate)
        let d0 = clock.now
        var streamed: [Int] = []
        var streamedAt = 0
        let onToken: ((Int) -> Void)? = onPartial.map { partial in
            { token in
                streamed.append(token)
                guard streamed.count - streamedAt >= Self.partialUpdateTokens else { return }
                streamedAt = streamed.count
                guard let text = try? tokenizer.decode(streamed) else { return }
                partial(text, streamedAt)
            }
        }
        let out: [Int]
        if speculating, let chunked = mm as? ChunkedSpeculativeChain {
            out = try chunked.continueSpeculative(
                seed: seed, context: prep.flatIDs, maxNew: cap, eos: eos, onToken: onToken)
        } else {
            out = try mm.continueGreedy(seed: seed, maxNew: cap, eos: eos, onToken: onToken)
        }
        let decodeSeconds = (clock.now - d0) / .seconds(1)
        processedTokens = prep.flatIDs + out
        return VLMGenerationInfo(
            text: (try? tokenizer.decode(out)) ?? "", generatedTokens: out.count,
            promptTokens: prep.flatIDs.count, imageRows: prep.imageRows,
            visionEncodeSeconds: prep.visionSeconds, prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            decodeTokensPerSecond: decodeSeconds > 0 ? Double(out.count) / decodeSeconds : 0,
            peakMemoryBytes: Self.memoryFootprint())
    }

    public func continueWithImageContext(
        question: String, maxNew: Int, speculative useSpec: Bool
    ) async throws -> VLMGenerationInfo {
        let mm = try requireMultimodalChain()
        guard let tokenizer else { throw LLMEngineError.notLoaded }
        guard !processedTokens.isEmpty else {
            throw LLMEngineError.generationFailed(reason: "no image context to continue from")
        }
        let ids = VLMPrompt.followUpTokens(
            userTokens: try tokenizer.encode("user\n"),
            questionTokens: try tokenizer.encode(question),
            modelTokens: try tokenizer.encode("model\n"))
        let used = mm.position + ids.count
        guard used < mm.contextLength else {
            throw LLMEngineError.contextOverflow(
                promptTokens: used, contextLength: mm.contextLength)
        }
        let speculating = (mm as? ChunkedSpeculativeChain).map { useSpec && $0.supportsMTP } ?? false
        if speculating { try await installMTPIfNeeded() }
        let eos = Set(tokenizer.eosTokenIDs)
        let clock = ContinuousClock()
        let p0 = clock.now
        let seed = try mm.prefill(ids)
        let prefillSeconds = (clock.now - p0) / .seconds(1)
        var context = processedTokens + ids
        let cap = Self.tokenBudget(maxNew: maxNew, used: used, contextLength: mm.contextLength)
        let d0 = clock.now
        let out: [Int]
        if speculating, let chunked = mm as? ChunkedSpeculativeChain {
            out = try chunked.continueSpeculative(seed: seed, context: context, maxNew: cap, eos: eos)
        } else {
            out = try mm.continueGreedy(seed: seed, maxNew: cap, eos: eos, onToken: nil)
        }
        let decodeSeconds = (clock.now - d0) / .seconds(1)
        context.append(contentsOf: out)
        processedTokens = context
        return VLMGenerationInfo(
            text: (try? tokenizer.decode(out)) ?? "", generatedTokens: out.count,
            promptTokens: ids.count, imageRows: 0,
            visionEncodeSeconds: 0, prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds,
            decodeTokensPerSecond: decodeSeconds > 0 ? Double(out.count) / decodeSeconds : 0,
            peakMemoryBytes: Self.memoryFootprint())
    }

    public nonisolated static var maxAudioSeconds: Double { AudioPreprocess.maxSeconds }

    public nonisolated static var defaultTranscriptionInstruction: String { ASRPrompt.instruction() }

    public func supportsAudioInput() -> Bool {
        chain is ChunkedSpeculativeChain && audioModelURL() != nil
    }

    private func audioModelURL() -> URL? {
        Self.sidecarModelURL(bundleURL: loadedBundleURL, name: "audio_fp16")
    }

    private static func sidecarModelURL(bundleURL: URL?, name: String) -> URL? {
        guard let bundleURL else { return nil }
        let fileManager = FileManager.default
        let package = bundleURL.appending(path: "\(name).mlpackage")
        let compiled = bundleURL.appending(path: "\(name).mlmodelc")
        let hasPackage = fileManager.fileExists(atPath: package.path(percentEncoded: false))
        let hasCompiled = fileManager.fileExists(atPath: compiled.path(percentEncoded: false))
        return hasPackage || hasCompiled ? package : nil
    }

    private func prepareASRSegments(
        samples: [Float], instruction: String
    ) async throws -> (segments: [PromptSegment], flatIDs: [Int], audioRows: Int, encodeSeconds: Double) {
        guard let tokenizer else { throw LLMEngineError.notLoaded }
        guard let audioURL = audioModelURL() else {
            throw LLMEngineError.modelNotFound(
                path: "audio_fp16.mlpackage (absent next to the loaded bundle)")
        }
        let encoder = AudioEncoder(packageURL: audioURL)
        let clock = ContinuousClock()
        let t0 = clock.now
        let soft = try await encoder.encode(samples: samples, releaseAfter: true)
        let encodeSeconds = (clock.now - t0) / .seconds(1)
        let bos = tokenizer.bosTokenID ?? 2
        let userTokens = try tokenizer.encode("user\n")
        let instructionTokens = try tokenizer.encode(instruction)
        let modelTokens = try tokenizer.encode("model\n")
        let segments = ASRPrompt.segments(
            bos: bos, userTokens: userTokens, instructionTokens: instructionTokens,
            modelTokens: modelTokens, audio: soft)
        let flatIDs = ASRPrompt.flatIDs(
            bos: bos, userTokens: userTokens, instructionTokens: instructionTokens,
            modelTokens: modelTokens, audioRows: soft.rows)
        return (segments, flatIDs, soft.rows, encodeSeconds)
    }

    public func generateWithAudio(
        samples: [Float], instruction: String, maxNew: Int, speculative useSpec: Bool
    ) async throws -> ASRGenerationInfo {
        guard let chunked = chain as? ChunkedSpeculativeChain, let tokenizer else {
            throw LLMEngineError.incompatibleBundle(
                reason: "audio input requires a \(ChunkedSpeculativeChain.format) bundle")
        }
        let prep = try await prepareASRSegments(samples: samples, instruction: instruction)
        guard prep.flatIDs.count < chunked.contextLength else {
            throw LLMEngineError.contextOverflow(
                promptTokens: prep.flatIDs.count, contextLength: chunked.contextLength)
        }
        let spec = useSpec && chunked.supportsMTP
        if spec { try await installMTPIfNeeded() }
        let eos = Set(tokenizer.eosTokenIDs)
        let clock = ContinuousClock()
        try chunked.reset()
        let p0 = clock.now
        let seed = try chunked.prefillSegments(prep.segments)
        let prefillSeconds = (clock.now - p0) / .seconds(1)
        let cap = Self.tokenBudget(
            maxNew: maxNew, used: prep.flatIDs.count, contextLength: chunked.contextLength)
        let d0 = clock.now
        let out = spec
            ? try chunked.continueSpeculative(seed: seed, context: prep.flatIDs, maxNew: cap, eos: eos)
            : try chunked.continueGreedy(seed: seed, maxNew: cap, eos: eos)
        let decodeSeconds = (clock.now - d0) / .seconds(1)
        processedTokens = prep.flatIDs + out
        return ASRGenerationInfo(
            text: (try? tokenizer.decode(out)) ?? "", generatedTokens: out.count,
            promptTokens: prep.flatIDs.count, audioRows: prep.audioRows,
            audioSeconds: Double(samples.count) / Double(AudioPreprocess.sampleRate),
            audioEncodeSeconds: prep.encodeSeconds, prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            decodeTokensPerSecond: decodeSeconds > 0 ? Double(out.count) / decodeSeconds : 0,
            peakMemoryBytes: Self.memoryFootprint())
    }

    public func continueWithAudioContext(
        question: String, maxNew: Int, speculative useSpec: Bool
    ) async throws -> ASRGenerationInfo {
        guard let chunked = chain as? ChunkedSpeculativeChain, let tokenizer else {
            throw LLMEngineError.incompatibleBundle(
                reason: "audio input requires a \(ChunkedSpeculativeChain.format) bundle")
        }
        guard !processedTokens.isEmpty else {
            throw LLMEngineError.generationFailed(reason: "no audio context to continue from")
        }
        let ids = ASRPrompt.followUpTokens(
            userTokens: try tokenizer.encode("user\n"),
            questionTokens: try tokenizer.encode(question),
            modelTokens: try tokenizer.encode("model\n"))
        let used = chunked.position + ids.count
        guard used < chunked.contextLength else {
            throw LLMEngineError.contextOverflow(
                promptTokens: used, contextLength: chunked.contextLength)
        }
        let spec = useSpec && chunked.supportsMTP
        if spec { try await installMTPIfNeeded() }
        let eos = Set(tokenizer.eosTokenIDs)
        let clock = ContinuousClock()
        let p0 = clock.now
        let seed = try chunked.prefillScheduled(ids)
        let prefillSeconds = (clock.now - p0) / .seconds(1)
        var context = processedTokens + ids
        let cap = Self.tokenBudget(maxNew: maxNew, used: used, contextLength: chunked.contextLength)
        let d0 = clock.now
        let out = spec
            ? try chunked.continueSpeculative(seed: seed, context: context, maxNew: cap, eos: eos)
            : try chunked.continueGreedy(seed: seed, maxNew: cap, eos: eos)
        let decodeSeconds = (clock.now - d0) / .seconds(1)
        context.append(contentsOf: out)
        processedTokens = context
        return ASRGenerationInfo(
            text: (try? tokenizer.decode(out)) ?? "", generatedTokens: out.count,
            promptTokens: ids.count, audioRows: 0, audioSeconds: 0, audioEncodeSeconds: 0,
            prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds,
            decodeTokensPerSecond: decodeSeconds > 0 ? Double(out.count) / decodeSeconds : 0,
            peakMemoryBytes: Self.memoryFootprint())
    }

    private static func tokenBudget(maxNew: Int, used: Int, contextLength: Int) -> Int {
        let budget = max(0, contextLength - used - 8)
        return maxNew > 0 ? min(maxNew, budget) : budget
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
