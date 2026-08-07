import CoreML
import CryptoKit
import Foundation
import LLMCore

final class ChunkedSpeculativeChain {
    static let format = "coreml-corellm-r1"
    static let verifyWidth = 4
    static let pldMinN = 3
    static let pldFireThreshold = 3

    let config: ChunkedChainConfig
    private(set) var position = 0

    private(set) var pldRounds = 0
    private(set) var pldFallbackRounds = 0
    private(set) var pldDraftedTokens = 0
    private(set) var pldAcceptedTokens = 0

    struct PLDStats: Sendable {
        let pldRounds: Int
        let fallbackRounds: Int
        let draftedTokens: Int
        let acceptedTokens: Int
        var hitRate: Double? { let t = pldRounds + fallbackRounds; return t > 0 ? Double(pldRounds) / Double(t) : nil }
        var draftAcceptRate: Double? { draftedTokens > 0 ? Double(acceptedTokens) / Double(draftedTokens) : nil }
    }

    private let bundleURL: URL
    private let computeUnits: MLComputeUnits
    let verifyHeadCU: MLComputeUnits
    private let numLayers: Int
    private let H: Int
    private let PLE: Int
    private let firstShared: Int
    private let CTX: Int
    private let neg: Float16
    private let sliding: Int
    private let storeS: Int
    private let storeF: Int
    private let sidecarStage: String

    private let decodeModels: [MLModel]
    private let head: MLModel
    private var batchedVerifyHead: MLModel?
    private let bundleSupportsVerify: Bool
    private let embedSidecar: QuantizedSidecar
    private let pleSidecar: QuantizedSidecar
    private let chunkCompiledURLs: [URL]

    private var prefillModels: [Int: [MLModel]] = [:]

    private(set) var blockScheduledPrefill = false

    private var kbuf: [MLMultiArray] = []
    private var vbuf: [MLMultiArray] = []

    private let hiddenBuffer: MLMultiArray
    private let embedsBuffer: MLMultiArray
    private let tokidBuffer: MLMultiArray
    private let rowBuffer: MLMultiArray

    private let dOnehot, dCosS, dSinS, dCosF, dSinF, dMaskS, dMaskF: MLMultiArray

    private var prefillBufs: [Int: PrefillBuffers] = [:]

    private struct PrefillBuffers {
        let ie: MLMultiArray
        let tk: MLMultiArray
        let onehot: MLMultiArray
        let cosS, sinS, cosF, sinF: MLMultiArray
        let maskPlainS, maskPlainF: MLMultiArray
        let maskOffS, maskOffF: MLMultiArray
    }

    init(bundleURL: URL, computeUnits: MLComputeUnits, sidecarStage: String? = nil,
         verifyHeadComputeUnits: MLComputeUnits? = nil, preloadVerifyAssets: Bool = true) async throws {
        let config = try ChunkedChainConfig.load(from: bundleURL.appending(path: "convert_config.json"))
        self.config = config
        self.bundleURL = bundleURL
        self.computeUnits = computeUnits
        numLayers = config.numLayers
        H = config.hidden
        PLE = config.pleDim
        firstShared = config.firstShared
        CTX = config.CTX
        neg = Float16(config.NEG)
        sliding = config.sliding
        guard let sS = config.storeLayers["sliding_attention"], let sF = config.storeLayers["full_attention"] else {
            throw LLMEngineError.incompatibleBundle(reason: "chunked config: store_layers invalid")
        }
        storeS = sS
        storeF = sF
        let stage = sidecarStage ?? config.defaultSidecarStage
        self.sidecarStage = stage

        var models: [MLModel] = []
        var compiledURLs: [URL] = []
        for name in config.chunkPackages {
            let compiled = try await Self.compileIfNeeded(bundleURL: bundleURL, name: name)
            compiledURLs.append(compiled)
            models.append(try Self.loadFunction(
                compiledURL: compiled, functionName: config.decodeFunction, computeUnits: computeUnits))
        }
        decodeModels = models
        chunkCompiledURLs = compiledURLs
        let headCompiled = try await Self.compileIfNeeded(bundleURL: bundleURL, name: config.lmheadPackage)
        head = try Self.loadFunction(compiledURL: headCompiled, functionName: nil, computeUnits: computeUnits)

        let headCU = verifyHeadComputeUnits ?? Self.resolveVerifyHeadComputeUnits(engine: computeUnits)
        verifyHeadCU = headCU
        bundleSupportsVerify = config.batchedHeadPackages[Self.verifyWidth] != nil
            && config.verifyFunctions[Self.verifyWidth] != nil
        if preloadVerifyAssets, bundleSupportsVerify,
           let pkg = config.batchedHeadPackages[Self.verifyWidth] {
            let compiled = try await Self.compileIfNeeded(bundleURL: bundleURL, name: pkg)
            batchedVerifyHead = try Self.loadFunction(compiledURL: compiled, functionName: nil, computeUnits: headCU)
        } else {
            batchedVerifyHead = nil
        }

        guard let sc = config.sidecarStages[stage] else {
            throw LLMEngineError.incompatibleBundle(reason: "chunked config: sidecar stage \(stage) is absent")
        }
        if stage == "int8" {
            embedSidecar = try QuantizedSidecar(
                int8: bundleURL.appending(path: sc.embed.file),
                scale: bundleURL.appending(path: sc.embed.scaleFile ?? "embed_scale_f32.bin"),
                rows: sc.embed.rows, cols: sc.embed.cols, runtimeScaleSqrt: sc.embed.runtimeScaleSqrt ?? Double(H))
            pleSidecar = try QuantizedSidecar(
                int8: bundleURL.appending(path: sc.ple.file),
                scale: bundleURL.appending(path: sc.ple.scaleFile ?? "ple_scale_f32.bin"),
                rows: sc.ple.rows, cols: sc.ple.cols, runtimeScaleSqrt: sc.ple.runtimeScaleSqrt ?? Double(PLE))
        } else {
            embedSidecar = try QuantizedSidecar(fp16: bundleURL.appending(path: sc.embed.file), rows: sc.embed.rows, cols: sc.embed.cols)
            pleSidecar = try QuantizedSidecar(fp16: bundleURL.appending(path: sc.ple.file), rows: sc.ple.rows, cols: sc.ple.cols)
        }

        func alloc(_ dims: [Int]) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: dims.map { NSNumber(value: $0) }, dataType: .float16)
            a.withF16 { $0.update(repeating: 0) }
            return a
        }
        hiddenBuffer = try alloc([H])
        embedsBuffer = try alloc([H])
        tokidBuffer = try alloc([numLayers, PLE])
        rowBuffer = try alloc([H])
        dOnehot = try alloc([CTX])
        dCosS = try alloc([config.HD]); dSinS = try alloc([config.HD])
        dCosF = try alloc([config.GHD]); dSinF = try alloc([config.GHD])
        dMaskS = try alloc([CTX]); dMaskF = try alloc([CTX])

        try reset()

        if batchedVerifyHead != nil {
            let block = Array(repeating: 0, count: Self.verifyWidth)
            let hidden = try runPrefillOffset(block, p: 0, N: Self.verifyWidth)
            _ = try lmheadBatched(hidden, width: Self.verifyWidth, count: Self.verifyWidth)
            try reset()
        }
    }

    var verifyHeadPackageName: String? {
        bundleSupportsVerify ? config.batchedHeadPackages[Self.verifyWidth] : nil
    }

    func installVerifyHead(_ model: MLModel) throws {
        guard batchedVerifyHead == nil else { return }
        batchedVerifyHead = model
        if position == 0 {
            let block = Array(repeating: 0, count: Self.verifyWidth)
            let hidden = try runPrefillOffset(block, p: 0, N: Self.verifyWidth)
            _ = try lmheadBatched(hidden, width: Self.verifyWidth, count: Self.verifyWidth)
            try reset()
        }
    }

    static func cuName(_ cu: MLComputeUnits) -> String {
        switch cu {
        case .cpuOnly: "cpuOnly"
        case .cpuAndGPU: "cpuAndGPU"
        case .cpuAndNeuralEngine: "cpuAndNeuralEngine"
        case .all: "all"
        @unknown default: "unknown"
        }
    }

    static func compileIfNeeded(bundleURL: URL, name: String) async throws -> URL {
        try await CompiledModelStore.compiledModelURL(bundleURL: bundleURL, name: name)
    }

    static func resolveVerifyHeadComputeUnits(engine: MLComputeUnits) -> MLComputeUnits {
        switch ProcessInfo.processInfo.environment["CORELLM_VERIFY_HEAD_CU"]?.lowercased() {
        case "cpu": return .cpuOnly
        case "gpu": return .cpuAndGPU
        case "ane": return .cpuAndNeuralEngine
        case "all": return .all
        default:
            switch engine {
            case .all, .cpuAndNeuralEngine: return .cpuOnly
            default: return engine
            }
        }
    }

    static func loadFunction(compiledURL: URL, functionName: String?, computeUnits: MLComputeUnits) throws -> MLModel {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        if let functionName { cfg.functionName = functionName }
        return try MLModel(contentsOf: compiledURL, configuration: cfg)
    }

    private func prefillChain(N: Int) throws -> [MLModel] {
        if let m = prefillModels[N] { return m }
        guard let fn = config.allPrefillFunctions[N] else {
            throw LLMEngineError.incompatibleBundle(reason: "chunked config: prefill N=\(N) has no function")
        }
        let models = try chunkCompiledURLs.map {
            try Self.loadFunction(compiledURL: $0, functionName: fn, computeUnits: computeUnits)
        }
        prefillModels[N] = models
        return models
    }

    func reset() throws {
        position = 0
        pldRounds = 0; pldFallbackRounds = 0; pldDraftedTokens = 0; pldAcceptedTokens = 0
        kbuf = try (0..<firstShared).map { try zeroKV(layer: $0) }
        vbuf = try (0..<firstShared).map { try zeroKV(layer: $0) }
    }

    func rewind(to newPosition: Int) {
        position = max(0, min(newPosition, CTX))
    }

    private func zeroKV(layer i: Int) throws -> MLMultiArray {
        let a = try MLMultiArray(
            shape: [1, NSNumber(value: CTX), NSNumber(value: config.headDim(ofLayer: i))], dataType: .float16)
        a.withF16 { $0.update(repeating: 0) }
        return a
    }

    func decodeStep(tokenID: Int) throws -> Int {
        try runDecodeChunks(tokenID: tokenID)
        return try lmhead(hiddenBuffer)
    }

    private func runDecodeChunks(tokenID: Int, softRow: SoftRowRef? = nil) throws {
        guard position < CTX else {
            throw LLMEngineError.generationFailed(reason: "context length \(CTX) exceeded")
        }
        fillDecodeHost(pos: position)
        if let ov = softRow {
            embedsBuffer.withF16 { ov.tokens.readRow(ov.row, into: $0.baseAddress!) }
            tokidBuffer.withF16 { pleSidecar.read(row: 0, into: $0.baseAddress!) }
        } else {
            embedsBuffer.withF16 { embedSidecar.read(row: tokenID, into: $0.baseAddress!) }
            tokidBuffer.withF16 { pleSidecar.read(row: tokenID, into: $0.baseAddress!) }
        }
        copyFlat(from: embedsBuffer, to: hiddenBuffer, count: H)

        for (ci, bounds) in config.chunkBounds.enumerated() {
            let (a, b) = (bounds[0], bounds[1])
            var features: [String: Any] = [
                "hidden_in": hiddenBuffer, "inputs_embeds": embedsBuffer, "tokid": tokidBuffer,
                "onehot": dOnehot, "cos_s": dCosS, "sin_s": dSinS, "cos_f": dCosF, "sin_f": dSinF,
                "mask_s": dMaskS, "mask_f": dMaskF,
            ]
            if a < firstShared {
                for i in a..<b { features["k_\(i)_in"] = kbuf[i]; features["v_\(i)_in"] = vbuf[i] }
            } else {
                features["sks"] = kbuf[storeS]; features["svs"] = vbuf[storeS]
                features["skf"] = kbuf[storeF]; features["svf"] = vbuf[storeF]
            }
            let out = try decodeModels[ci].prediction(from: MLDictionaryFeatureProvider(dictionary: features))
            guard let hidden = out.featureValue(for: "hidden")?.multiArrayValue else {
                throw LLMEngineError.generationFailed(reason: "chunked decode chunk \(ci) returned no hidden")
            }
            copyFlat(from: hidden, to: hiddenBuffer, count: H)
            if a < firstShared {
                for i in a..<b {
                    guard let k = out.featureValue(for: "k_\(i)_out")?.multiArrayValue,
                          let v = out.featureValue(for: "v_\(i)_out")?.multiArrayValue else {
                        throw LLMEngineError.generationFailed(reason: "chunked decode chunk \(ci) layer \(i) KV missing")
                    }
                    kbuf[i] = k; vbuf[i] = v
                }
            }
        }
        position += 1
    }

    func prefillScheduled(_ ids: [Int]) throws -> Int {
        try feedScheduled(ids: ids, rowRefs: Array(repeating: nil, count: ids.count))
    }

    private func feedScheduled(ids: [Int], rowRefs: [SoftRowRef?]) throws -> Int {
        guard !ids.isEmpty else { throw LLMEngineError.generationFailed(reason: "empty prompt") }
        precondition(rowRefs.count == ids.count)
        if blockScheduledPrefill {
            throw LLMEngineError.generationFailed(
                reason: "prefill invoked on a decode-only (restore) chain; restore replaces prefill so the wide prefill functions are never loaded")
        }
        let offsetNs = config.prefillNs.filter { $0 != config.plainN }.sorted(by: >)
        var idx = 0
        var last = 0
        while idx < ids.count {
            try Task.checkCancellation()
            let remaining = ids.count - idx
            if position == 0 && idx == 0 && remaining >= config.plainN {
                let block = Array(ids[idx..<(idx + config.plainN)])
                let hidden = try runPrefillPlain(
                    block, N: config.plainN, softRows: softSlice(rowRefs, idx, idx + config.plainN))
                last = try lmheadRow(hidden, row: config.plainN - 1)
                idx += config.plainN
            } else if let N = offsetNs.first(where: { remaining >= $0 && position + $0 <= CTX }) {
                let block = Array(ids[idx..<(idx + N)])
                let hidden = try runPrefillOffset(
                    block, p: position, N: N, softRows: softSlice(rowRefs, idx, idx + N))
                last = try lmheadRow(hidden, row: N - 1)
                idx += N
            } else {
                for i in idx..<ids.count {
                    try Task.checkCancellation()
                    try runDecodeChunks(tokenID: ids[i], softRow: rowRefs[i])
                    last = try lmhead(hiddenBuffer)
                }
                idx = ids.count
            }
        }
        return last
    }

    func plannedPrefillWidths(promptLength: Int, from startPosition: Int = 0) -> [Int] {
        let offsetNs = config.prefillNs.filter { $0 != config.plainN }.sorted(by: >)
        var widths: Set<Int> = []
        var pos = startPosition
        var idx = 0
        while idx < promptLength {
            let remaining = promptLength - idx
            if pos == 0 && idx == 0 && remaining >= config.plainN {
                widths.insert(config.plainN)
                idx += config.plainN
                pos = config.plainN
            } else if let N = offsetNs.first(where: { remaining >= $0 && pos + $0 <= CTX }) {
                widths.insert(N)
                idx += N
                pos += N
            } else {
                break
            }
        }
        return widths.sorted(by: >)
    }

    func materializePrefill(widths: [Int]) throws {
        for n in widths {
            _ = try prefillChain(N: n)
            _ = try prefillBuffers(N: n)
        }
    }

    private func softSlice(_ rowRefs: [SoftRowRef?], _ lo: Int, _ hi: Int) -> [SoftRowRef?]? {
        for k in lo..<hi where rowRefs[k] != nil { return Array(rowRefs[lo..<hi]) }
        return nil
    }

    @discardableResult
    func runPrefillPlain(_ ids: [Int], N: Int, softRows: [SoftRowRef?]? = nil) throws -> MLMultiArray {
        let models = try prefillChain(N: N)
        let bufs = try prefillBuffers(N: N)
        let L = ids.count
        precondition(L <= N)
        fillTokenInputs(bufs: bufs, ids: ids, N: N, softRows: softRows)
        writeRoPE(cos: bufs.cosS, sin: bufs.sinS, inv: config.invSlide, positions: (0..<N).map { $0 })
        writeRoPE(cos: bufs.cosF, sin: bufs.sinF, inv: config.invFull, positions: (0..<N).map { $0 })
        fillPlainMask(bufs.maskPlainS, sliding: true, N: N, L: L)
        fillPlainMask(bufs.maskPlainF, sliding: false, N: N, L: L)

        var hidden = bufs.ie
        var store: [Int: (MLMultiArray, MLMultiArray)] = [:]
        for (ci, bounds) in config.chunkBounds.enumerated() {
            let (a, b) = (bounds[0], bounds[1])
            var features: [String: Any] = [
                "hidden_in": hidden, "inputs_embeds": bufs.ie, "tokid": bufs.tk,
                "cos_s": bufs.cosS, "sin_s": bufs.sinS, "cos_f": bufs.cosF, "sin_f": bufs.sinF,
                "mask_s": bufs.maskPlainS, "mask_f": bufs.maskPlainF,
            ]
            if a >= firstShared {
                features["sks"] = store[storeS]!.0; features["svs"] = store[storeS]!.1
                features["skf"] = store[storeF]!.0; features["svf"] = store[storeF]!.1
            }
            let out = try models[ci].prediction(from: MLDictionaryFeatureProvider(dictionary: features))
            guard let h = out.featureValue(for: "hidden")?.multiArrayValue else {
                throw LLMEngineError.generationFailed(reason: "chunked prefill\(N) chunk \(ci) returned no hidden")
            }
            hidden = h
            if a < firstShared {
                for i in a..<b {
                    guard let k = out.featureValue(for: "k_\(i)_out")?.multiArrayValue,
                          let v = out.featureValue(for: "v_\(i)_out")?.multiArrayValue else {
                        throw LLMEngineError.generationFailed(reason: "chunked prefill\(N) chunk \(ci) layer \(i) KV missing")
                    }
                    if i == storeS || i == storeF { store[i] = (k, v) }
                    writebackKV(dst: kbuf[i], src: k, p: 0, L: L, hd: config.headDim(ofLayer: i))
                    writebackKV(dst: vbuf[i], src: v, p: 0, L: L, hd: config.headDim(ofLayer: i))
                }
            }
        }
        position = L
        return hidden
    }

    @discardableResult
    func runPrefillOffset(_ block: [Int], p: Int, N: Int, softRows: [SoftRowRef?]? = nil) throws -> MLMultiArray {
        let models = try prefillChain(N: N)
        let bufs = try prefillBuffers(N: N)
        let L = block.count
        precondition(L <= N && p + L <= CTX)
        fillTokenInputs(bufs: bufs, ids: block, N: N, softRows: softRows)
        writeRoPE(cos: bufs.cosS, sin: bufs.sinS, inv: config.invSlide, positions: (0..<N).map { p + $0 })
        writeRoPE(cos: bufs.cosF, sin: bufs.sinF, inv: config.invFull, positions: (0..<N).map { p + $0 })
        fillOffsetOnehot(bufs.onehot, p: p, N: N, L: L)
        fillOffsetMask(bufs.maskOffS, sliding: true, p: p, N: N, L: L)
        fillOffsetMask(bufs.maskOffF, sliding: false, p: p, N: N, L: L)

        var hidden = bufs.ie
        for (ci, bounds) in config.chunkBounds.enumerated() {
            let (a, b) = (bounds[0], bounds[1])
            var features: [String: Any] = [
                "hidden_in": hidden, "inputs_embeds": bufs.ie, "tokid": bufs.tk,
                "cos_s": bufs.cosS, "sin_s": bufs.sinS, "cos_f": bufs.cosF, "sin_f": bufs.sinF,
                "mask_s": bufs.maskOffS, "mask_f": bufs.maskOffF, "onehot": bufs.onehot,
            ]
            if a < firstShared {
                for i in a..<b { features["k_\(i)_in"] = kbuf[i]; features["v_\(i)_in"] = vbuf[i] }
            } else {
                features["sks"] = kbuf[storeS]; features["svs"] = vbuf[storeS]
                features["skf"] = kbuf[storeF]; features["svf"] = vbuf[storeF]
            }
            let out = try models[ci].prediction(from: MLDictionaryFeatureProvider(dictionary: features))
            guard let h = out.featureValue(for: "hidden")?.multiArrayValue else {
                throw LLMEngineError.generationFailed(reason: "chunked prefill\(N) offset chunk \(ci) returned no hidden")
            }
            hidden = h
            if a < firstShared {
                for i in a..<b {
                    guard let k = out.featureValue(for: "k_\(i)_out")?.multiArrayValue,
                          let v = out.featureValue(for: "v_\(i)_out")?.multiArrayValue else {
                        throw LLMEngineError.generationFailed(reason: "chunked prefill\(N) offset chunk \(ci) layer \(i) KV missing")
                    }
                    writebackKV(dst: kbuf[i], src: k, p: p, L: L, hd: config.headDim(ofLayer: i))
                    writebackKV(dst: vbuf[i], src: v, p: p, L: L, hd: config.headDim(ofLayer: i))
                }
            }
        }
        position = p + L
        return hidden
    }

    private func lmhead(_ hidden: MLMultiArray) throws -> Int {
        let out = try head.prediction(from: MLDictionaryFeatureProvider(dictionary: ["hidden": hidden]))
        guard let token = out.featureValue(for: "token")?.multiArrayValue else {
            throw LLMEngineError.generationFailed(reason: "chunked lm_head returned no token")
        }
        return token[0].intValue
    }

    private func lmheadRow(_ hidden: MLMultiArray, row: Int) throws -> Int {
        copyRow(from: hidden, row: row, to: rowBuffer)
        return try lmhead(rowBuffer)
    }

    func lmheadBatched(_ hidden: MLMultiArray, width W: Int, count: Int) throws -> [Int] {
        guard W == Self.verifyWidth, let h = batchedVerifyHead else {
            throw LLMEngineError.incompatibleBundle(reason: "chunked config: batched verify head width \(W) is absent")
        }
        let out = try h.prediction(from: MLDictionaryFeatureProvider(dictionary: ["hidden": hidden]))
        guard let token = out.featureValue(for: "token")?.multiArrayValue else {
            throw LLMEngineError.generationFailed(reason: "chunked batched head\(W) returned no token")
        }
        return (0..<count).map { token[$0].intValue }
    }

    private func fillDecodeHost(pos p: Int) {
        dOnehot.withF16 { buf in
            for s in 0..<CTX { buf[s] = 0 }
            buf[p] = 1
        }
        writeRoPE(cos: dCosS, sin: dSinS, inv: config.invSlide, positions: [p])
        writeRoPE(cos: dCosF, sin: dSinF, inv: config.invFull, positions: [p])
        dMaskS.withF16 { buf in
            for s in 0..<CTX { buf[s] = (s <= p && p - s < sliding) ? 0 : neg }
        }
        dMaskF.withF16 { buf in
            for s in 0..<CTX { buf[s] = s <= p ? 0 : neg }
        }
    }

    private func writeRoPE(cos: MLMultiArray, sin: MLMultiArray, inv: [Double], positions: [Int]) {
        let half = inv.count
        cos.withF16 { cbuf in
            sin.withF16 { sbuf in
                for (row, p) in positions.enumerated() {
                    let off = row * half * 2
                    let pd = Double(p)
                    for i in 0..<half {
                        let angle = inv[i] * pd
                        let c = Float16(Foundation.cos(angle))
                        let sv = Float16(Foundation.sin(angle))
                        cbuf[off + i] = c
                        cbuf[off + i + half] = c
                        sbuf[off + i] = sv
                        sbuf[off + i + half] = sv
                    }
                }
            }
        }
    }

    private func fillPlainMask(_ mask: MLMultiArray, sliding s: Bool, N: Int, L: Int) {
        mask.withF16 { buf in
            for i in 0..<N {
                for j in 0..<N {
                    let visible = j <= i && j < L && (!s || (i - j < sliding))
                    buf[i * N + j] = visible ? 0 : neg
                }
            }
        }
    }

    private func fillOffsetMask(_ mask: MLMultiArray, sliding s: Bool, p: Int, N: Int, L: Int) {
        mask.withF16 { buf in
            for i in 0..<N {
                let g = p + i
                let validRow = i < L
                for slot in 0..<CTX {
                    let base = slot <= g && slot < p + L && validRow
                    let visible = base && (!s || (g - slot < sliding))
                    buf[i * CTX + slot] = visible ? 0 : neg
                }
            }
        }
    }

    private func fillOffsetOnehot(_ onehot: MLMultiArray, p: Int, N: Int, L: Int) {
        onehot.withF16 { buf in
            for k in 0..<(N * CTX) { buf[k] = 0 }
            for i in 0..<min(L, N) { buf[i * CTX + (p + i)] = 1 }
        }
    }

    private func fillTokenInputs(bufs: PrefillBuffers, ids: [Int], N: Int, softRows: [SoftRowRef?]? = nil) {
        precondition(softRows == nil || softRows!.count == ids.count)
        bufs.ie.withF16 { buf in
            for k in 0..<(N * H) { buf[k] = 0 }
            for (i, t) in ids.enumerated() {
                if let ov = softRows?[i] {
                    ov.tokens.readRow(ov.row, into: buf.baseAddress! + i * H)
                } else {
                    embedSidecar.read(row: t, into: buf.baseAddress! + i * H)
                }
            }
        }
        let plecols = numLayers * PLE
        bufs.tk.withF16 { buf in
            for k in 0..<(N * plecols) { buf[k] = 0 }
            for (i, t) in ids.enumerated() {
                let row = (softRows?[i] != nil) ? MultimodalSlot.perLayerInputRow : t
                pleSidecar.read(row: row, into: buf.baseAddress! + i * plecols)
            }
        }
    }

    private func prefillBuffers(N: Int) throws -> PrefillBuffers {
        if let b = prefillBufs[N] { return b }
        func alloc(_ dims: [Int]) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: dims.map { NSNumber(value: $0) }, dataType: .float16)
            a.withF16 { $0.update(repeating: 0) }
            return a
        }
        let b = PrefillBuffers(
            ie: try alloc([N, H]),
            tk: try alloc([N, numLayers, PLE]),
            onehot: try alloc([N, CTX]),
            cosS: try alloc([N, config.HD]), sinS: try alloc([N, config.HD]),
            cosF: try alloc([N, config.GHD]), sinF: try alloc([N, config.GHD]),
            maskPlainS: try alloc([N, N]), maskPlainF: try alloc([N, N]),
            maskOffS: try alloc([N, CTX]), maskOffF: try alloc([N, CTX]))
        prefillBufs[N] = b
        return b
    }

    private func writebackKV(dst: MLMultiArray, src: MLMultiArray, p: Int, L: Int, hd: Int) {
        dst.withF16 { d in
            src.withF16 { s in
                for r in 0..<L {
                    (d.baseAddress! + (p + r) * hd).update(from: s.baseAddress! + r * hd, count: hd)
                }
            }
        }
    }

    private func copyFlat(from source: MLMultiArray, to destination: MLMultiArray, count: Int) {
        source.withF16 { src in
            destination.withF16 { dst in
                dst.baseAddress!.update(from: src.baseAddress!, count: count)
            }
        }
    }

    private func copyRow(from source: MLMultiArray, row: Int, to destination: MLMultiArray) {
        source.withF16 { src in
            destination.withF16 { dst in
                dst.baseAddress!.update(from: src.baseAddress! + row * H, count: H)
            }
        }
    }
}

extension ChunkedSpeculativeChain: GenerationChain {
    var contextLength: Int { CTX }

    func prefill(_ promptIDs: [Int]) throws -> Int {
        try prefillScheduled(promptIDs)
    }
}

extension ChunkedSpeculativeChain: SpeculativeDecoding {
    var supportsMTP: Bool { bundleSupportsVerify }
    var mtpLoaded: Bool { batchedVerifyHead != nil }

    func mtpRound(prediction: Int, context: [Int]) throws -> MTPRound {
        let base = position
        guard supportsMTP, batchedVerifyHead != nil else {
            let next = try decodeStep(tokenID: prediction)
            return MTPRound(emitted: [prediction], next: next, accepted: 0, drafted: 0)
        }
        let maxDraft = Self.verifyWidth - 1
        let cap = min(maxDraft, CTX - 1 - base)
        let pldCopied = cap >= 1
            ? PromptLookup.draft(context: context, last: prediction, maxDraft: cap, maxN: 3, minN: Self.pldMinN)
            : []
        let S = 1 + pldCopied.count
        guard !pldCopied.isEmpty, S >= Self.pldFireThreshold else {
            pldFallbackRounds += 1
            let next = try decodeStep(tokenID: prediction)
            return MTPRound(emitted: [prediction], next: next, accepted: 0, drafted: 0)
        }
        let drafts = [prediction] + pldCopied
        let hidden = try runPrefillOffset(drafts, p: base, N: Self.verifyWidth)
        let targets = try lmheadBatched(hidden, width: Self.verifyWidth, count: S)
        var accepted = 1
        while accepted < S && drafts[accepted] == targets[accepted - 1] { accepted += 1 }
        position = base + accepted
        pldRounds += 1
        pldDraftedTokens += pldCopied.count
        pldAcceptedTokens += (accepted - 1)
        return MTPRound(
            emitted: Array(drafts[0..<accepted]), next: targets[accepted - 1],
            accepted: accepted, drafted: S)
    }
}

extension ChunkedSpeculativeChain {
    func pldStatsSnapshot() -> PLDStats {
        PLDStats(pldRounds: pldRounds, fallbackRounds: pldFallbackRounds,
                 draftedTokens: pldDraftedTokens, acceptedTokens: pldAcceptedTokens)
    }

    func setBlockScheduledPrefill(_ b: Bool) { blockScheduledPrefill = b }

    func residentPrefillWidths() -> [Int] { prefillModels.keys.sorted() }

    func continueGreedy(
        seed: Int, maxNew: Int, eos: Set<Int>, onToken: ((Int) -> Void)? = nil
    ) throws -> [Int] {
        var out: [Int] = []
        var next = seed
        while out.count < maxNew {
            try Task.checkCancellation()
            if eos.contains(next) { break }
            out.append(next)
            onToken?(next)
            if out.count >= maxNew { break }
            next = try decodeStep(tokenID: next)
        }
        return out
    }

    func continueSpeculative(
        seed: Int, context: [Int], maxNew: Int, eos: Set<Int>, onToken: ((Int) -> Void)? = nil
    ) throws -> [Int] {
        var processed = context
        var next = seed
        var out: [Int] = []
        loop: while out.count < maxNew {
            try Task.checkCancellation()
            if eos.contains(next) { break }
            let round = try mtpRound(prediction: next, context: processed)
            processed.append(contentsOf: round.emitted)
            for token in round.emitted {
                if eos.contains(token) { break loop }
                out.append(token)
                onToken?(token)
                if out.count >= maxNew { break loop }
            }
            next = round.next
        }
        return out
    }
}

final class SoftTokenRows: Sendable {
    let rows: Int
    let hidden: Int
    private let data: [Float16]

    init(rows: Int, hidden: Int, data: [Float16]) {
        precondition(data.count == rows * hidden,
                     "SoftTokenRows: data.count \(data.count) != rows*hidden \(rows * hidden)")
        self.rows = rows
        self.hidden = hidden
        self.data = data
    }

    func readRow(_ r: Int, into dst: UnsafeMutablePointer<Float16>) {
        precondition(r >= 0 && r < rows)
        data.withUnsafeBufferPointer { dst.update(from: $0.baseAddress! + r * hidden, count: hidden) }
    }
}

enum MultimodalSlot {
    static let imagePlaceholderID = 258880
    static let audioPlaceholderID = 258881
    static let perLayerInputRow = 0
}

enum PromptSegment {
    case tokens([Int])
    case image(SoftTokenRows)
    case audio(SoftTokenRows)
}

struct SoftRowRef {
    let tokens: SoftTokenRows
    let row: Int
}

extension ChunkedSpeculativeChain {
    static var imagePlaceholderID: Int { MultimodalSlot.imagePlaceholderID }
    static var audioPlaceholderID: Int { MultimodalSlot.audioPlaceholderID }

    var hiddenSize: Int { H }

    private func flattenSegments(_ segments: [PromptSegment]) -> (ids: [Int], rowRefs: [SoftRowRef?]) {
        var ids: [Int] = []
        var rowRefs: [SoftRowRef?] = []
        func appendSoft(_ soft: SoftTokenRows, placeholder: Int, kind: String) {
            precondition(soft.hidden == H, "\(kind) soft tokens hidden \(soft.hidden) != chain H \(H)")
            for r in 0..<soft.rows {
                ids.append(placeholder)
                rowRefs.append(SoftRowRef(tokens: soft, row: r))
            }
        }
        for seg in segments {
            switch seg {
            case .tokens(let ts):
                for t in ts { ids.append(t); rowRefs.append(nil) }
            case .image(let soft):
                appendSoft(soft, placeholder: Self.imagePlaceholderID, kind: "image")
            case .audio(let soft):
                appendSoft(soft, placeholder: Self.audioPlaceholderID, kind: "audio")
            }
        }
        return (ids, rowRefs)
    }

    @discardableResult
    func prefillSegments(_ segments: [PromptSegment]) throws -> Int {
        let (ids, rowRefs) = flattenSegments(segments)
        guard !ids.isEmpty else { throw LLMEngineError.generationFailed(reason: "empty prompt (segments)") }
        return try feedScheduled(ids: ids, rowRefs: rowRefs)
    }
}

struct ChunkedKVIdentity: Codable, Equatable, Sendable {
    var convertConfigSHA256: String
    var numLayers: Int
    var hidden: Int
    var pleDim: Int
    var firstShared: Int
    var ctx: Int
    var hd: Int
    var ghd: Int
    var sliding: Int
    var sidecarStage: String
    var ownLayerHeadDims: [Int]
    var storeLayers: [String: Int]
}

struct ChunkedKVLayerEntry: Codable, Sendable {
    var layer: Int
    var type: String
    var headDim: Int
    var rows: Int
    var kFile: String
    var vFile: String
    var kBytes: Int
    var vBytes: Int
}

struct ChunkedKVManifest: Codable, Sendable {
    var schemaVersion: Int
    var identity: ChunkedKVIdentity
    var position: Int
    var pendingNextToken: Int?
    var tokenCount: Int
    var tokenSeqSHA256: String
    var processedTokens: [Int]
    var layers: [ChunkedKVLayerEntry]
    var bundleName: String?
    var createdOnOS: String
    var createdWithComputeUnits: String
    var createdAtISO8601: String
}

enum ChunkedKVError: Error, CustomStringConvertible {
    case badManifest(String)
    case schemaMismatch(expected: Int, found: Int)
    case identityMismatch(expected: ChunkedKVIdentity, found: ChunkedKVIdentity)
    case tokenMismatch(expectedCount: Int, manifestCount: Int)
    case missingFile(String)
    case sizeMismatch(name: String, expected: Int, found: Int)

    var description: String {
        switch self {
        case .badManifest(let m): return "KV manifest invalid: \(m)"
        case .schemaMismatch(let e, let f): return "KV schema mismatch (target=\(e) file=\(f))"
        case .identityMismatch(let e, let f):
            return "KV restore target bundle does not match the saved source (expected=\(e), found=\(f))"
        case .tokenMismatch(let e, let m):
            return "KV restore committed-token sequence mismatch (requested=\(e)tok, manifest=\(m)tok)"
        case .missingFile(let n): return "KV state file not found: \(n)"
        case .sizeMismatch(let n, let e, let f):
            return "KV state \(n) byte-count mismatch (expected=\(e), file=\(f))"
        }
    }
}

extension ChunkedSpeculativeChain {
    static var kvSchemaVersion: Int { 1 }

    func kvIdentity() -> ChunkedKVIdentity {
        ChunkedKVIdentity(
            convertConfigSHA256: convertConfigSHA256(),
            numLayers: numLayers, hidden: H, pleDim: PLE, firstShared: firstShared,
            ctx: CTX, hd: config.HD, ghd: config.GHD, sliding: sliding,
            sidecarStage: sidecarStage,
            ownLayerHeadDims: (0..<firstShared).map { config.headDim(ofLayer: $0) },
            storeLayers: config.storeLayers)
    }

    @discardableResult
    func exportKV(to url: URL, pendingNextToken: Int?, processedTokens: [Int]) throws -> ChunkedKVManifest {
        let fm = FileManager.default
        try fm.createDirectory(at: url, withIntermediateDirectories: true)
        let rows = position
        var entries: [ChunkedKVLayerEntry] = []
        for i in 0..<firstShared {
            let hd = config.headDim(ofLayer: i)
            let kData = kvPrefixData(kbuf[i], rows: rows, hd: hd)
            let vData = kvPrefixData(vbuf[i], rows: rows, hd: hd)
            let kFile = "k_\(i).bin"
            let vFile = "v_\(i).bin"
            try kData.write(to: url.appending(path: kFile), options: .atomic)
            try vData.write(to: url.appending(path: vFile), options: .atomic)
            entries.append(ChunkedKVLayerEntry(
                layer: i, type: config.layerTypes[i], headDim: hd, rows: rows,
                kFile: kFile, vFile: vFile, kBytes: kData.count, vBytes: vData.count))
        }
        let manifest = ChunkedKVManifest(
            schemaVersion: Self.kvSchemaVersion,
            identity: kvIdentity(),
            position: position,
            pendingNextToken: pendingNextToken,
            tokenCount: processedTokens.count,
            tokenSeqSHA256: Self.tokenSeqSHA256(processedTokens),
            processedTokens: processedTokens,
            layers: entries,
            bundleName: bundleURL.lastPathComponent,
            createdOnOS: ProcessInfo.processInfo.operatingSystemVersionString,
            createdWithComputeUnits: Self.cuName(computeUnits),
            createdAtISO8601: ISO8601DateFormatter().string(from: Date()))
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(manifest).write(to: url.appending(path: "manifest.json"), options: .atomic)
        excludeFromBackup(url)
        return manifest
    }

    @discardableResult
    func importKV(from url: URL, expectedContext: [Int]?) throws -> ChunkedKVManifest {
        guard let mData = try? Data(contentsOf: url.appending(path: "manifest.json")) else {
            throw ChunkedKVError.badManifest("manifest.json unreadable: \(url.path())")
        }
        let manifest: ChunkedKVManifest
        do { manifest = try JSONDecoder().decode(ChunkedKVManifest.self, from: mData) }
        catch { throw ChunkedKVError.badManifest("manifest.json decode failed: \(error)") }

        guard manifest.schemaVersion == Self.kvSchemaVersion else {
            throw ChunkedKVError.schemaMismatch(expected: Self.kvSchemaVersion, found: manifest.schemaVersion)
        }
        let current = kvIdentity()
        guard manifest.identity == current else {
            throw ChunkedKVError.identityMismatch(expected: current, found: manifest.identity)
        }
        if let saved = manifest.bundleName, saved != bundleURL.lastPathComponent {
            let warn = "chunked KV WARN: bundleName differs (manifest=\(saved) current=\(bundleURL.lastPathComponent)); "
                + "record-only field, restore continues (identity matches)\n"
            FileHandle.standardError.write(Data(warn.utf8))
        }
        if let want = expectedContext {
            guard want.count == manifest.position,
                  Self.tokenSeqSHA256(want) == manifest.tokenSeqSHA256 else {
                throw ChunkedKVError.tokenMismatch(expectedCount: want.count, manifestCount: manifest.position)
            }
        }
        guard manifest.layers.count == firstShared else {
            throw ChunkedKVError.badManifest("own layer count mismatch: manifest=\(manifest.layers.count) chain=\(firstShared)")
        }

        try reset()
        let byLayer = Dictionary(manifest.layers.map { ($0.layer, $0) }, uniquingKeysWith: { a, _ in a })
        for i in 0..<firstShared {
            guard let e = byLayer[i] else {
                throw ChunkedKVError.badManifest("layer \(i) entry missing")
            }
            let hd = config.headDim(ofLayer: i)
            guard e.headDim == hd, e.rows == manifest.position else {
                throw ChunkedKVError.badManifest(
                    "layer \(i) shape mismatch (hd=\(e.headDim)/\(hd) rows=\(e.rows)/\(manifest.position))")
            }
            try loadKVPrefix(into: kbuf[i], from: url.appending(path: e.kFile),
                             rows: e.rows, hd: hd, expectBytes: e.kBytes, name: e.kFile)
            try loadKVPrefix(into: vbuf[i], from: url.appending(path: e.vFile),
                             rows: e.rows, hd: hd, expectBytes: e.vBytes, name: e.vFile)
        }
        position = manifest.position
        return manifest
    }

    private func convertConfigSHA256() -> String {
        guard let data = try? Data(contentsOf: bundleURL.appending(path: "convert_config.json")) else { return "" }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func tokenSeqSHA256(_ tokens: [Int]) -> String {
        var hasher = SHA256()
        for t in tokens {
            var le = Int64(t).littleEndian
            withUnsafeBytes(of: &le) { hasher.update(bufferPointer: $0) }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func kvPrefixData(_ arr: MLMultiArray, rows: Int, hd: Int) -> Data {
        let count = rows * hd
        var data = Data(count: count * MemoryLayout<Float16>.size)
        guard count > 0 else { return data }
        let strides = arr.strides.map(\.intValue)
        let rowStride = strides.count >= 2 ? strides[1] : hd
        let colStride = strides.count >= 3 ? strides[2] : 1
        arr.withF16 { src in
            data.withUnsafeMutableBytes { dstRaw in
                let dst = dstRaw.bindMemory(to: Float16.self)
                if rowStride == hd && colStride == 1 {
                    dst.baseAddress!.update(from: src.baseAddress!, count: count)
                } else {
                    for r in 0..<rows {
                        let base = src.baseAddress! + r * rowStride
                        for d in 0..<hd { dst[r * hd + d] = base[d * colStride] }
                    }
                }
            }
        }
        return data
    }

    private func loadKVPrefix(into arr: MLMultiArray, from fileURL: URL,
                             rows: Int, hd: Int, expectBytes: Int, name: String) throws {
        guard FileManager.default.fileExists(atPath: fileURL.path()) else {
            throw ChunkedKVError.missingFile(name)
        }
        let data = try Data(contentsOf: fileURL, options: [.alwaysMapped])
        let want = rows * hd * MemoryLayout<Float16>.size
        guard data.count == want, want == expectBytes else {
            throw ChunkedKVError.sizeMismatch(name: name, expected: want, found: data.count)
        }
        guard rows > 0 else { return }
        arr.withF16 { dst in
            data.withUnsafeBytes { srcRaw in
                _ = memcpy(dst.baseAddress!, srcRaw.baseAddress!, want)
            }
        }
    }

    private func excludeFromBackup(_ url: URL) {
        #if !os(macOS)
        var u = url
        var rv = URLResourceValues()
        rv.isExcludedFromBackup = true
        try? u.setResourceValues(rv)
        #endif
    }
}
