import Accelerate
import CoreML
import Foundation
import LLMCore

private struct UncheckedBox<T>: @unchecked Sendable {
    let value: T
}

private final class Int8Sidecar {
    private let data: Data
    private let scaleData: Data
    let rows: Int
    let cols: Int
    private let scratch: UnsafeMutablePointer<Float>

    init(int8 url: URL, scale scaleURL: URL, rows: Int, cols: Int) throws {
        self.data = try Data(contentsOf: url, options: [.alwaysMapped])
        let expected = rows * cols
        guard data.count == expected else {
            throw LLMEngineError.incompatibleBundle(
                reason: "\(url.lastPathComponent): int8 size mismatch (actual \(data.count), expected \(expected))")
        }
        let s = try Data(contentsOf: scaleURL, options: [.alwaysMapped])
        guard s.count == rows * MemoryLayout<Float32>.size else {
            throw LLMEngineError.incompatibleBundle(
                reason: "\(scaleURL.lastPathComponent): scale size mismatch (actual \(s.count), expected \(rows * 4))")
        }
        self.scaleData = s
        self.rows = rows
        self.cols = cols
        self.scratch = .allocate(capacity: cols)
    }

    deinit { scratch.deallocate() }

    func read(row: Int, offset: Int, count: Int, into destination: UnsafeMutablePointer<Float16>) {
        precondition(row >= 0 && row < rows && offset >= 0 && offset + count <= cols)
        var factor = scaleData.withUnsafeBytes {
            $0.load(fromByteOffset: row * 4, as: Float32.self)
        } / 127.0
        let base = row * cols + offset
        data.withUnsafeBytes { raw in
            let p = raw.bindMemory(to: Int8.self).baseAddress! + base
            vDSP_vflt8(p, 1, scratch, 1, vDSP_Length(count))
            vDSP_vsmul(scratch, 1, &factor, scratch, 1, vDSP_Length(count))
            var src = vImage_Buffer(data: scratch, height: 1,
                                    width: vImagePixelCount(count), rowBytes: count * 4)
            var dst = vImage_Buffer(data: destination, height: 1,
                                    width: vImagePixelCount(count), rowBytes: count * 2)
            _ = vImageConvert_PlanarFtoPlanar16F(&src, &dst, vImage_Flags(kvImageNoFlags))
        }
    }
}

private enum V2Sidecar {
    case fp16(Sidecar)
    case int8(Int8Sidecar)

    var rows: Int { switch self { case .fp16(let s): s.rows; case .int8(let s): s.rows } }
    var cols: Int { switch self { case .fp16(let s): s.cols; case .int8(let s): s.cols } }

    static func make(bundleURL: URL, spec: ChainConfigV2.Sidecars.File) throws -> V2Sidecar {
        let rows = spec.shape[0]
        let cols = spec.shape[1]
        if spec.dtype == "int8" {
            guard let scaleName = spec.scale else {
                throw LLMEngineError.incompatibleBundle(
                    reason: "\(spec.file): an int8 sidecar requires a scale entry")
            }
            return .int8(try Int8Sidecar(
                int8: bundleURL.appending(path: spec.file),
                scale: bundleURL.appending(path: scaleName),
                rows: rows, cols: cols))
        }
        return .fp16(try Sidecar(url: bundleURL.appending(path: spec.file), rows: rows, cols: cols))
    }

    func read(row: Int, offset: Int = 0, count: Int? = nil, into destination: UnsafeMutablePointer<Float16>) {
        switch self {
        case .fp16(let s):
            s.read(row: row, offset: offset, count: count, into: destination)
        case .int8(let s):
            s.read(row: row, offset: offset, count: count ?? (s.cols - offset), into: destination)
        }
    }
}

final class CoreMLChainV2 {
    let config: ChainConfigV2
    private(set) var position = 0

    private(set) var loadModelsSeconds: Double = 0
    private(set) var prewarmSeconds: Double = 0

    private(set) var headComputeUnits: MLComputeUnits = .cpuAndGPU

    private let bundleURL: URL
    private let chunks: [MLModel]
    private let head: MLModel
    private let embedSidecar: V2Sidecar
    private let pleSidecar: V2Sidecar?
    private let usesPLE: Bool
    private var tokidBuffers: [String: MLMultiArray] = [:]

    private let needsSharedKV: [Bool]
    private let statefulChunk: [Bool]
    private let hasSharedKV: Bool
    private let host: HostInputsV2

    var chunkStatefulFlags: [Bool] { statefulChunk }
    var chunkSharedKVFlags: [Bool] { needsSharedKV }

    private let chunks128: [MLModel]?

    private let hostCtx32k: HostInputsV2?
    private let hostCtx128k: HostInputsV2?

    private var promoted = false

    var isPromoted: Bool { promoted }

    func forcePromotion(_ on: Bool) { if config.isLadder { promoted = on } }

    private let lmheadInput: MLMultiArray

    private var states: [MLState] = []

    private var hiddenBuffers: [Int: MLMultiArray] = [:]

    let drafterURL: URL?

    let drafterPreURL: URL?

    var supportsMTP: Bool { drafterURL != nil || pldEnabled }
    private(set) var mtpLoaded = false

    private(set) var draftLen: Int
    private let baseDraftLen: Int
    private let maxDraftLen: Int
    private let adaptiveDraft: Bool
    private var recentAccept: [Double] = []

    let pldEnabled: Bool

    let pldMinN: Int

    private(set) var pldRounds = 0
    private(set) var drafterFallbackRounds = 0
    private(set) var pldDraftedTokens = 0
    private(set) var pldAcceptedTokens = 0

    struct PLDStats {
        let enabled: Bool
        let pldRounds: Int
        let fallbackRounds: Int
        let draftedTokens: Int
        let acceptedTokens: Int

        var hitRate: Double? {
            let total = pldRounds + fallbackRounds
            return total > 0 ? Double(pldRounds) / Double(total) : nil
        }

        var draftAcceptRate: Double? {
            draftedTokens > 0 ? Double(acceptedTokens) / Double(draftedTokens) : nil
        }
    }

    func pldStatsSnapshot() -> PLDStats {
        PLDStats(
            enabled: pldEnabled, pldRounds: pldRounds, fallbackRounds: drafterFallbackRounds,
            draftedTokens: pldDraftedTokens, acceptedTokens: pldAcceptedTokens)
    }

    let treeEnabled: Bool

    var treeActive: Bool
    private let treeHost: TreeHostInputs

    private(set) var treeAltFired = 0
    private(set) var treeAltRecovered = 0

    struct TreeStats {
        let enabled: Bool
        let altFired: Int
        let altRecovered: Int
    }

    func treeStatsSnapshot() -> TreeStats {
        TreeStats(enabled: treeEnabled, altFired: treeAltFired, altRecovered: treeAltRecovered)
    }

    private var drafter: MLModel?
    private var drafterHost: DrafterHostInputs?

    private var drafterPre: MLModel?
    private var drafterHostPre: DrafterHostInputs?

    private(set) var drafterPreRounds = 0
    private(set) var drafterFullRounds = 0
    private var draftEmbed: MLMultiArray?
    private var draftHidden: MLMultiArray?
    private var lastHiddenBuffer: MLMultiArray?

    private var lastToken = 0

    private var storeSKS: MLMultiArray?
    private var storeSVS: MLMultiArray?
    private var storeSKF: MLMultiArray?
    private var storeSVF: MLMultiArray?

    private let storeKName: (sliding: String, full: String)
    private let storeVName: (sliding: String, full: String)

    private var ladderStoreSlidingK: MLMultiArray?
    private var ladderStoreSlidingV: MLMultiArray?
    private var ladderStoreFullK: MLMultiArray?
    private var ladderStoreFullV: MLMultiArray?

    private var ladderStoreFullKPre: MLMultiArray?
    private var ladderStoreFullVPre: MLMultiArray?

    private var ladderCapturePre = false

    private var ladderStoreStateNames: (slidingK: String, slidingV: String, fullK: String, fullV: String)?

    private var ladderStoreSyncedRows = 0

    init(bundleURL: URL, computeUnits: MLComputeUnits) async throws {
        guard computeUnits == .cpuAndGPU || computeUnits == .all else {
            throw LLMEngineError.incompatibleBundle(
                reason: "v2 stateful requires the GPU (computeUnits=\(computeUnits.rawValue)). "
                    + "CPU_ONLY / ANE crash with a native segfault when multiple MLState instances load together. "
                    + "Specify cpuAndGPU or all")
        }
        let config = try ChainConfigV2.loadFromBundle(bundleURL)
        self.config = config
        self.bundleURL = bundleURL

        let headCU = Self.resolveHeadComputeUnits(chunkUnits: computeUnits)
        headComputeUnits = headCU
        let clock = ContinuousClock()
        let loadStart = clock.now
        if config.isLadder {

            func loadSet(function: String) async throws -> [MLModel] {
                var out: [MLModel] = []
                for name in config.chunks {
                    let cfg = MLModelConfiguration()
                    cfg.computeUnits = computeUnits
                    cfg.functionName = function
                    out.append(try await CoreMLChain.loadCompiled(
                        bundleURL: bundleURL, name: name, configuration: cfg))
                }
                return out
            }
            let c32 = try await loadSet(function: "ctx32k")
            let c128 = try await loadSet(function: "ctx128k")
            let headCfg = MLModelConfiguration()
            headCfg.computeUnits = headCU
            let h = try await CoreMLChain.loadCompiled(
                bundleURL: bundleURL, name: config.lmhead, configuration: headCfg)
            chunks = c32
            chunks128 = c128
            head = h
        } else {

            let names = config.chunks + [config.lmhead]
            let headIndex = config.chunks.count
            var loaded = [MLModel?](repeating: nil, count: names.count)
            try await withThrowingTaskGroup(of: (Int, UncheckedBox<MLModel>).self) { group in
                for (i, name) in names.enumerated() {
                    let units = (i == headIndex) ? headCU : computeUnits
                    group.addTask {
                        let cfg = MLModelConfiguration()
                        cfg.computeUnits = units
                        let model = try await CoreMLChain.loadCompiled(
                            bundleURL: bundleURL, name: name, configuration: cfg)
                        return (i, UncheckedBox(value: model))
                    }
                }
                for try await (i, box) in group { loaded[i] = box.value }
            }
            let models = loaded.map { $0! }
            chunks = Array(models[0..<config.chunks.count])
            head = models[config.chunks.count]
            chunks128 = nil
        }
        loadModelsSeconds = (clock.now - loadStart) / .seconds(1)

        embedSidecar = try V2Sidecar.make(bundleURL: bundleURL, spec: config.sidecars.embed)
        usesPLE = config.usesPLE
        if let ple = config.sidecars.ple {
            pleSidecar = try V2Sidecar.make(bundleURL: bundleURL, spec: ple)
        } else {
            pleSidecar = nil
        }
        let sharedFlags = chunks.map { $0.modelDescription.inputDescriptionsByName.keys.contains("sks") }
        needsSharedKV = sharedFlags
        statefulChunk = sharedFlags.map { !$0 }
        hasSharedKV = sharedFlags.contains(true)
        host = HostInputsV2(config: config)
        if config.isLadder {

            hostCtx32k = HostInputsV2(
                config: config, splitOnehot: true,
                windowSlide: config.windowSlide, ctxFull: config.ladderCtx32kWindow)
            hostCtx128k = HostInputsV2(
                config: config, splitOnehot: true,
                windowSlide: config.windowSlide, ctxFull: config.ladderCtx128kWindow)
        } else {
            hostCtx32k = nil
            hostCtx128k = nil
        }
        treeHost = TreeHostInputs(config: config)
        lmheadInput = try MLMultiArray(shape: [1, NSNumber(value: config.H)], dataType: .float16)

        let fm = FileManager.default
        if config.usesSplitOnehot {

            let ringDrafter = bundleURL.appending(path: "drafter_ring.mlmodelc")
            drafterURL = fm.fileExists(atPath: ringDrafter.path(percentEncoded: false)) ? ringDrafter : nil

            if config.isLadder {
                let preDrafter = bundleURL.appending(path: "drafter_ring32k.mlmodelc")
                drafterPreURL = fm.fileExists(atPath: preDrafter.path(percentEncoded: false)) ? preDrafter : nil
            } else {
                drafterPreURL = nil
            }
        } else {

            let drafterCandidates = [
                bundleURL.appending(path: "drafter.mlmodelc"),
                bundleURL.deletingLastPathComponent().appending(path: "drafter.mlmodelc"),
            ]
            drafterURL = drafterCandidates.first { fm.fileExists(atPath: $0.path(percentEncoded: false)) }
            drafterPreURL = nil
        }

        let env = ProcessInfo.processInfo.environment

        let maxDraft = min(16, config.maxS)
        maxDraftLen = maxDraft

        let requested = env["CORELLM_MTP_DRAFT_LEN"].flatMap(Int.init) ?? 4
        baseDraftLen = min(max(requested, 4), maxDraft)
        draftLen = baseDraftLen
        adaptiveDraft = ["1", "true", "yes"].contains((env["CORELLM_MTP_ADAPTIVE"] ?? "").lowercased())

        pldEnabled = ["1", "true", "yes"].contains((env["CORELLM_MTP_PLD"] ?? "").lowercased())

        pldMinN = max(1, env["CORELLM_MTP_PLD_MINN"].flatMap(Int.init) ?? 3)

        treeEnabled = ["1", "true", "yes"].contains((env["CORELLM_MTP_TREE"] ?? "").lowercased())
        treeActive = treeEnabled

        let storeS = config.storeLayers["sliding_attention"] ?? 46
        let storeF = config.storeLayers["full_attention"] ?? 47
        storeKName = (sliding: "k_\(storeS)_out", full: "k_\(storeF)_out")
        storeVName = (sliding: "v_\(storeS)_out", full: "v_\(storeF)_out")

        states = chunks.map { $0.makeState() }
        position = 0

        let warmStart = clock.now
        _ = try decodeStep(tokenID: 2)
        try reset()
        prewarmSeconds = (clock.now - warmStart) / .seconds(1)
    }

    static func resolveHeadComputeUnits(chunkUnits: MLComputeUnits) -> MLComputeUnits {
        switch (ProcessInfo.processInfo.environment["CORELLM_V2_HEAD_CU"] ?? "").lowercased() {
        case "cpu": return .cpuOnly
        case "gpu": return .cpuAndGPU
        case "ane": return .cpuAndNeuralEngine
        case "all": return .all
        case "chunks": return chunkUnits
        default: return chunkUnits
        }
    }

    func reset() throws {
        states = chunks.map { $0.makeState() }
        position = 0
        lastToken = 0
        promoted = false
        draftLen = baseDraftLen
        recentAccept.removeAll(keepingCapacity: true)
        storeSKS = nil; storeSVS = nil; storeSKF = nil; storeSVF = nil
        ladderStoreSyncedRows = 0
        ladderCapturePre = false
        drafterPreRounds = 0; drafterFullRounds = 0
        pldRounds = 0; drafterFallbackRounds = 0; pldDraftedTokens = 0; pldAcceptedTokens = 0
        treeAltFired = 0; treeAltRecovered = 0

    }

    func rewind(to newPosition: Int) {
        position = max(0, min(newPosition, config.effectiveContextLength))
        storeSKS = nil; storeSVS = nil; storeSKF = nil; storeSVF = nil

        ladderStoreSyncedRows = min(ladderStoreSyncedRows, position)
    }

    func installMTP(drafter: MLModel, drafterPre: MLModel? = nil) throws {
        guard !mtpLoaded, supportsMTP else { return }
        self.drafter = drafter
        drafterHost = try DrafterHostInputs(config: config)

        if config.isLadder, let drafterPre {
            self.drafterPre = drafterPre
            drafterHostPre = try DrafterHostInputs(
                config: config, maskFWidthOverride: config.ladderCtx32kWindow)
        }
        draftEmbed = try MLMultiArray(shape: [NSNumber(value: config.H)], dataType: .float16)
        draftHidden = try MLMultiArray(shape: [NSNumber(value: config.H)], dataType: .float16)
        lastHiddenBuffer = try MLMultiArray(shape: [NSNumber(value: config.H)], dataType: .float16)

        if config.isLadder { try allocateLadderStoreBuffers() }
        mtpLoaded = true
    }

    private func allocateLadderStoreBuffers() throws {
        let last = chunks.count - 1
        let sIdx = config.layerTypes.lastIndex(of: "sliding_attention")
            ?? config.storeLayers["sliding_attention"] ?? 46
        let fIdx = config.layerTypes.lastIndex(of: "full_attention")
            ?? config.storeLayers["full_attention"] ?? 47
        let names = (slidingK: "k_\(sIdx)", slidingV: "v_\(sIdx)", fullK: "k_\(fIdx)", fullV: "v_\(fIdx)")
        ladderStoreStateNames = names

        func allocLike(_ name: String) throws -> MLMultiArray {
            guard chunks[last].modelDescription.stateDescriptionsByName[name] != nil else {
                throw LLMEngineError.generationFailed(
                    reason: "ladder store state \(name) is absent from the final chunk (index \(last)); MTP capture is impossible")
            }
            var out: MLMultiArray?
            var caught: Error?
            states[last].withMultiArray(for: name) { arr in
                do { out = try MLMultiArray(shape: arr.shape, dataType: arr.dataType) }
                catch { caught = error }
            }
            if let caught { throw caught }
            guard let out else {
                throw LLMEngineError.generationFailed(reason: "failed to allocate a buffer for ladder store state \(name)")
            }
            return out
        }
        ladderStoreSlidingK = try allocLike(names.slidingK)
        ladderStoreSlidingV = try allocLike(names.slidingV)
        ladderStoreFullK = try allocLike(names.fullK)
        ladderStoreFullV = try allocLike(names.fullV)

        zeroBuffer(ladderStoreFullK)
        zeroBuffer(ladderStoreFullV)

        if drafterPre != nil, let big = ladderStoreFullK {
            let d = big.shape.count >= 3 ? big.shape[2].intValue : 512
            let w32 = config.ladderCtx32kWindow
            func allocSmall() throws -> MLMultiArray {
                try MLMultiArray(
                    shape: [1, NSNumber(value: w32), NSNumber(value: d)], dataType: big.dataType)
            }
            let pk = try allocSmall(); let pv = try allocSmall()
            zeroBuffer(pk); zeroBuffer(pv)
            ladderStoreFullKPre = pk
            ladderStoreFullVPre = pv
        }
        ladderStoreSyncedRows = 0
        ladderCapturePre = false
    }

    private func zeroBuffer(_ arr: MLMultiArray?) {
        arr?.withUnsafeMutableBytes { dst, _ in
            if let base = dst.baseAddress { memset(base, 0, dst.count) }
        }
    }

    private func captureLadderStoreKV() throws {
        guard let names = ladderStoreStateNames,
              let sk = ladderStoreSlidingK, let sv = ladderStoreSlidingV,
              let fkBig = ladderStoreFullK, let fvBig = ladderStoreFullV else { return }
        let last = chunks.count - 1

        try copyState(chunk: last, name: names.slidingK, into: sk)
        try copyState(chunk: last, name: names.slidingV, into: sv)

        let usePre = (drafterPre != nil) && !promoted
        let fk = usePre ? (ladderStoreFullKPre ?? fkBig) : fkBig
        let fv = usePre ? (ladderStoreFullVPre ?? fvBig) : fvBig

        if usePre != ladderCapturePre {
            ladderStoreSyncedRows = 0
            ladderCapturePre = usePre
        }

        let numRows = fk.shape.count >= 2 ? fk.shape[1].intValue : Int.max
        let upTo = min(position, numRows)
        try copyFullStateRows(chunk: last, name: names.fullK, into: fk, from: ladderStoreSyncedRows, upTo: upTo)
        try copyFullStateRows(chunk: last, name: names.fullV, into: fv, from: ladderStoreSyncedRows, upTo: upTo)
        ladderStoreSyncedRows = upTo
        storeSKS = sk; storeSVS = sv; storeSKF = fk; storeSVF = fv
    }

    private func copyFullStateRows(chunk ci: Int, name: String, into dest: MLMultiArray,
                                   from: Int, upTo: Int) throws {
        var didIncremental = false
        states[ci].withMultiArray(for: name) { arr in

            guard Self.isCContiguous(arr), Self.isCContiguous(dest),
                  arr.shape.count == 3, arr.shape[0].intValue == 1,
                  dest.shape.count == 3, dest.shape[0].intValue == 1 else { return }
            let srcRows = arr.shape[1].intValue
            let destRows = dest.shape[1].intValue
            guard srcRows > 0, destRows > 0 else { return }
            arr.withUnsafeBytes { src in
                dest.withUnsafeMutableBytes { dst, _ in
                    guard src.count % srcRows == 0, dst.count % destRows == 0 else { return }
                    let rowBytes = src.count / srcRows
                    guard rowBytes == dst.count / destRows else { return }

                    let bound = min(srcRows, destRows)
                    let lo = max(0, min(from, bound))
                    let hi = max(lo, min(upTo, bound))
                    didIncremental = true
                    guard hi > lo else { return }
                    let offset = lo * rowBytes
                    let length = (hi - lo) * rowBytes
                    _ = memcpy(dst.baseAddress! + offset, src.baseAddress! + offset, length)
                }
            }
        }
        if !didIncremental {

            try copyState(chunk: ci, name: name, into: dest)
        }
    }

    private static func isCContiguous(_ arr: MLMultiArray) -> Bool {
        let shape = arr.shape.map { $0.intValue }
        let strides = arr.strides.map { $0.intValue }
        guard shape.count == strides.count, !shape.isEmpty else { return false }
        var expected = 1
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            guard strides[i] == expected else { return false }
            expected *= shape[i]
        }
        return true
    }

    private func copyState(chunk ci: Int, name: String, into dest: MLMultiArray) throws {
        var caught: Error?
        states[ci].withMultiArray(for: name) { arr in
            arr.withUnsafeBytes { src in
                dest.withUnsafeMutableBytes { dst, _ in
                    guard src.count == dst.count else {
                        caught = LLMEngineError.generationFailed(
                            reason: "ladder store state \(name) byte-count mismatch (state=\(src.count) buf=\(dst.count))")
                        return
                    }
                    _ = memcpy(dst.baseAddress!, src.baseAddress!, src.count)
                }
            }
        }
        if let caught { throw caught }
    }

    @discardableResult
    func prefill(_ promptIDs: [Int], blockSize: Int) throws -> Int {
        try prefillRows(promptIDs, blockSize: blockSize, softRows: nil)
    }

    @discardableResult
    private func prefillRows(
        _ promptIDs: [Int], blockSize: Int, softRows: [SoftRowRef?]?
    ) throws -> Int {
        precondition(!promptIDs.isEmpty, "prefill needs at least 1 token")
        precondition(softRows == nil || softRows!.count == promptIDs.count)

        let start = position
        let ctxLimit = config.effectiveContextLength
        guard start + promptIDs.count <= ctxLimit else {
            throw LLMEngineError.generationFailed(
                reason: "position after prefill \(start + promptIDs.count) exceeds context length \(ctxLimit)")
        }
        let block = max(1, min(blockSize, config.maxS))

        let clampToWindow = config.usesSplitOnehot
        var pos = 0
        var lastHidden: MLMultiArray?
        var lastRow = 0
        while pos < promptIDs.count {
            try Task.checkCancellation()
            let base = start + pos
            var s = min(block, promptIDs.count - pos)
            if clampToWindow {
                let toBoundary = config.windowSlide - (base % config.windowSlide)
                s = min(s, toBoundary)
            }
            lastHidden = try forwardChunks(
                tokens: Array(promptIDs[pos..<pos + s]), basePosition: base,
                softRows: softSlice(softRows, pos, pos + s))
            lastRow = s - 1
            pos += s
        }
        position = start + promptIDs.count

        guard let lastHidden else {
            throw LLMEngineError.generationFailed(reason: "prefill: hidden is empty")
        }

        copyRow(from: lastHidden, row: lastRow, into: lmheadInput)

        if mtpLoaded, let lastHiddenBuffer {
            lastToken = promptIDs[promptIDs.count - 1]
            copyRow(from: lastHidden, row: lastRow, into: lastHiddenBuffer)
        }
        return try lmheadArgmax(lmheadInput)
    }

    func decodeStep(tokenID: Int) throws -> Int {
        guard position < config.effectiveContextLength else {
            throw LLMEngineError.generationFailed(
                reason: "context length \(config.effectiveContextLength) exceeded")
        }
        let hidden = try forwardChunks(tokens: [tokenID], basePosition: position)
        let next = try lmheadArgmax(hidden)
        if mtpLoaded, let lastHiddenBuffer {
            lastToken = tokenID
            copyRow(from: hidden, row: 0, into: lastHiddenBuffer)
        }
        position += 1
        return next
    }

    private func forwardChunks(
        tokens: [Int], basePosition: Int, softRows: [SoftRowRef?]? = nil
    ) throws -> MLMultiArray {
        let s = tokens.count
        let hin = try hiddenBuffer(for: s)

        if let softRows {
            precondition(softRows.count == s)
            hin.withF16 { buf in
                for (r, tok) in tokens.enumerated() {
                    if let ov = softRows[r] {
                        ov.tokens.readRow(ov.row, into: buf.baseAddress! + r * config.H)
                    } else {
                        embedSidecar.read(row: tok, into: buf.baseAddress! + r * config.H)
                    }
                }
            }
        } else {
            hin.withF16 { buf in
                for (r, tok) in tokens.enumerated() {
                    embedSidecar.read(row: tok, into: buf.baseAddress! + r * config.H)
                }
            }
        }
        if config.isLadder {

            let maxPos = basePosition + s - 1
            if !promoted && maxPos >= config.ladderCtx32kWindow { promoted = true }
            let activeChunks = promoted ? chunks128! : chunks
            let activeHost = promoted ? hostCtx128k! : hostCtx32k!
            let feats = try activeHost.filled(base: basePosition, count: s)
            return try runChunks(
                chunks: activeChunks, hidden: hin, feats: feats, splitOnehot: true, tokens: tokens,
                softRows: softRows)
        }
        let feats = try host.filled(base: basePosition, count: s)
        return try runChunks(
            chunks: chunks, hidden: hin, feats: feats, splitOnehot: config.usesSplitOnehot, tokens: tokens,
            softRows: softRows)
    }

    private func forwardChunksTree(
        mainTokens: [Int], altToken: Int, basePosition: Int, altSlot: Int
    ) throws -> MLMultiArray {
        let mainCount = mainTokens.count
        let hin = try hiddenBuffer(for: mainCount + 1)
        hin.withF16 { buf in
            for (r, tok) in mainTokens.enumerated() {
                embedSidecar.read(row: tok, into: buf.baseAddress! + r * config.H)
            }
            embedSidecar.read(row: altToken, into: buf.baseAddress! + mainCount * config.H)
        }
        let feats = try treeHost.filled(base: basePosition, mainCount: mainCount, altSlot: altSlot)

        return try runChunks(
            chunks: chunks, hidden: hin, feats: feats, splitOnehot: false, tokens: mainTokens + [altToken])
    }

    private func runChunks(
        chunks activeChunks: [MLModel], hidden hin: MLMultiArray,
        feats: HostInputsV2.Buffers, splitOnehot: Bool, tokens: [Int],
        softRows: [SoftRowRef?]? = nil
    ) throws -> MLMultiArray {
        let inputsEmbeds = hin
        var hidden: MLMultiArray = hin
        var sharedKV: [String: MLMultiArray] = [:]
        for ci in activeChunks.indices {

            var dictionary: [String: Any] = [
                "hidden_in": hidden,
                "cos_s": feats.cosS, "sin_s": feats.sinS,
                "cos_f": feats.cosF, "sin_f": feats.sinF,
                "mask_s": feats.maskS, "mask_f": feats.maskF,
            ]
            if splitOnehot {
                dictionary["onehot_s"] = feats.onehotS!
                dictionary["onehot_f"] = feats.onehotF!
            } else {
                dictionary["onehot"] = feats.onehot!
            }
            if usesPLE {
                dictionary["inputs_embeds"] = inputsEmbeds
                dictionary["tokid"] = try fillTokid(
                    chunkIndex: ci, tokens: tokens, softRows: softRows)
            }
            if hasSharedKV && needsSharedKV[ci] {
                dictionary["sks"] = sharedKV["sks"]
                dictionary["svs"] = sharedKV["svs"]
                dictionary["skf"] = sharedKV["skf"]
                dictionary["svf"] = sharedKV["svf"]
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: dictionary)
            let output = statefulChunk[ci]
                ? try activeChunks[ci].prediction(from: provider, using: states[ci])
                : try activeChunks[ci].prediction(from: provider)
            if hasSharedKV {
                if let k = output.featureValue(for: storeKName.sliding)?.multiArrayValue { sharedKV["sks"] = k }
                if let v = output.featureValue(for: storeVName.sliding)?.multiArrayValue { sharedKV["svs"] = v }
                if let k = output.featureValue(for: storeKName.full)?.multiArrayValue { sharedKV["skf"] = k }
                if let v = output.featureValue(for: storeVName.full)?.multiArrayValue { sharedKV["svf"] = v }
            }
            guard let h = output.featureValue(for: "hidden")?.multiArrayValue else {
                throw LLMEngineError.generationFailed(reason: "v2 chunk \(ci) did not return hidden")
            }
            hidden = h

            if mtpLoaded && ci == activeChunks.count - 1 {
                storeSKS = output.featureValue(for: storeKName.sliding)?.multiArrayValue
                storeSVS = output.featureValue(for: storeVName.sliding)?.multiArrayValue
                storeSKF = output.featureValue(for: storeKName.full)?.multiArrayValue
                storeSVF = output.featureValue(for: storeVName.full)?.multiArrayValue
            }
        }
        return hidden
    }

    private func lmheadArgmax(_ hidden: MLMultiArray) throws -> Int {
        let output = try head.prediction(from: MLDictionaryFeatureProvider(dictionary: ["hidden": hidden]))
        guard let token = output.featureValue(for: "token")?.multiArrayValue else {
            throw LLMEngineError.generationFailed(reason: "v2 lm_head did not return a token")
        }
        return token[0].intValue
    }

    private func lmheadArgmaxRows(_ hidden: MLMultiArray, count s: Int) throws -> [Int] {
        let output = try head.prediction(from: MLDictionaryFeatureProvider(dictionary: ["hidden": hidden]))
        guard let token = output.featureValue(for: "token")?.multiArrayValue else {
            throw LLMEngineError.generationFailed(reason: "v2 lm_head did not return a token")
        }
        return (0..<s).map { token[$0].intValue }
    }

    func batchedVerifyArgmaxes(tokens: [Int], basePosition: Int) throws -> [Int] {
        let hidden = try forwardChunks(tokens: tokens, basePosition: basePosition)
        return try lmheadArgmaxRows(hidden, count: tokens.count)
    }

    func mtpRound(prediction: Int, context: [Int] = []) throws -> MTPRound {

        if config.isLadder, mtpLoaded, drafter != nil, ladderStoreStateNames != nil {
            try captureLadderStoreKV()
        }

        let hasDrafter = mtpLoaded && drafter != nil && drafterHost != nil
            && draftEmbed != nil && draftHidden != nil && lastHiddenBuffer != nil
            && storeSKS != nil && storeSVS != nil && storeSKF != nil && storeSVF != nil
        guard pldEnabled || hasDrafter, position + draftLen < config.effectiveContextLength else {
            let next = try decodeStep(tokenID: prediction)
            return MTPRound(emitted: [prediction], next: next, accepted: 0, drafted: 0)
        }

        let pldCopied: [Int] = pldEnabled
            ? PromptLookup.draft(
                context: context, last: prediction,
                maxDraft: min(draftLen, config.effectiveContextLength - 1 - position),
                minN: pldMinN)
            : []
        let usedPLD = !pldCopied.isEmpty
        var drafts: [Int]
        if usedPLD {
            drafts = [prediction] + pldCopied
        } else if hasDrafter, let drafter, let drafterHost, let draftEmbed, let draftHidden,
                  let lastHiddenBuffer,
                  let sks = storeSKS, let svs = storeSVS, let skf = storeSKF, let svf = storeSVF {
            if pldEnabled { drafterFallbackRounds += 1 }

            let usePre = config.isLadder && !promoted && drafterPre != nil && drafterHostPre != nil
            let activeDrafter = usePre ? (drafterPre ?? drafter) : drafter
            let activeHost = usePre ? (drafterHostPre ?? drafterHost) : drafterHost
            if usePre { drafterPreRounds += 1 } else { drafterFullRounds += 1 }

            activeHost.update(position: position - 1)
            copyFlat(from: lastHiddenBuffer, to: draftHidden, count: config.H)

            _ = try draftStep(token: lastToken, drafter: activeDrafter, host: activeHost,
                              sks: sks, svs: svs, skf: skf, svf: svf, embed: draftEmbed, hidden: draftHidden)
            drafts = [prediction]
            var tok = prediction
            for _ in 1..<draftLen {
                tok = try draftStep(token: tok, drafter: activeDrafter, host: activeHost,
                                    sks: sks, svs: svs, skf: skf, svf: svf, embed: draftEmbed, hidden: draftHidden)
                drafts.append(tok)
            }
        } else {

            if pldEnabled { drafterFallbackRounds += 1 }
            let next = try decodeStep(tokenID: prediction)
            return MTPRound(emitted: [prediction], next: next, accepted: 0, drafted: 0)
        }

        let base = position
        let drafted = drafts.count

        var altToken: Int?
        var altSlot = 0

        if treeActive, !config.usesSplitOnehot, usedPLD, drafts.count >= 2, hasDrafter,
           let drafter, let drafterHost, let draftEmbed, let draftHidden, let lastHiddenBuffer,
           let sks = storeSKS, let svs = storeSVS, let skf = storeSKF, let svf = storeSVF {
            let slot = base + drafts.count
            if slot <= config.effectiveContextLength - 1 {
                drafterHost.update(position: position - 1)
                copyFlat(from: lastHiddenBuffer, to: draftHidden, count: config.H)
                _ = try draftStep(token: lastToken, drafter: drafter, host: drafterHost,
                                  sks: sks, svs: svs, skf: skf, svf: svf, embed: draftEmbed, hidden: draftHidden)
                let a1 = try draftStep(token: prediction, drafter: drafter, host: drafterHost,
                                       sks: sks, svs: svs, skf: skf, svf: svf, embed: draftEmbed, hidden: draftHidden)
                if a1 != drafts[1] { altToken = a1; altSlot = slot }
            }
        }

        func recordPLD(mainAccepted: Int) {
            guard usedPLD else { return }
            pldRounds += 1
            pldDraftedTokens += pldCopied.count
            pldAcceptedTokens += (mainAccepted - 1)
        }

        if let altToken {

            treeAltFired += 1
            let treeHidden = try forwardChunksTree(
                mainTokens: drafts, altToken: altToken, basePosition: base, altSlot: altSlot)
            let targets = try lmheadArgmaxRows(treeHidden, count: drafts.count + 1)

            var accepted = 1
            while accepted < drafts.count && drafts[accepted] == targets[accepted - 1] { accepted += 1 }

            if accepted == 1 && altToken == targets[0] {

                treeAltRecovered += 1
                let altHidden = try forwardChunks(tokens: [altToken], basePosition: base + 1)
                let next = try lmheadArgmax(altHidden)
                position = base + 2
                lastToken = altToken
                if let lastHiddenBuffer { copyRow(from: altHidden, row: 0, into: lastHiddenBuffer) }
                recordPLD(mainAccepted: accepted)
                if adaptiveDraft { adaptDraftLength(accepted: accepted, drafted: drafted) }
                return MTPRound(emitted: [prediction, altToken], next: next, accepted: 2, drafted: drafted)
            }

            position = base + accepted
            lastToken = drafts[accepted - 1]
            if let lastHiddenBuffer { copyRow(from: treeHidden, row: accepted - 1, into: lastHiddenBuffer) }
            recordPLD(mainAccepted: accepted)
            if adaptiveDraft { adaptDraftLength(accepted: accepted, drafted: drafted) }
            return MTPRound(
                emitted: Array(drafts[0..<accepted]), next: targets[accepted - 1],
                accepted: accepted, drafted: drafted)
        }

        let verifyHidden = try forwardChunks(tokens: drafts, basePosition: base)
        let targets = try lmheadArgmaxRows(verifyHidden, count: drafts.count)

        var accepted = 1
        while accepted < drafts.count && drafts[accepted] == targets[accepted - 1] {
            accepted += 1
        }

        position = base + accepted
        lastToken = drafts[accepted - 1]
        if let lastHiddenBuffer { copyRow(from: verifyHidden, row: accepted - 1, into: lastHiddenBuffer) }
        recordPLD(mainAccepted: accepted)
        if adaptiveDraft { adaptDraftLength(accepted: accepted, drafted: drafted) }
        return MTPRound(
            emitted: Array(drafts[0..<accepted]), next: targets[accepted - 1],
            accepted: accepted, drafted: drafted)
    }

    private func draftStep(
        token tok: Int, drafter: MLModel, host: DrafterHostInputs,
        sks: MLMultiArray, svs: MLMultiArray, skf: MLMultiArray, svf: MLMultiArray,
        embed: MLMultiArray, hidden: MLMultiArray
    ) throws -> Int {
        embed.withF16 { buf in embedSidecar.read(row: tok, into: buf.baseAddress!) }
        let out = try drafter.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "embed": embed, "hidden": hidden,
            "cos_s": host.cosS, "sin_s": host.sinS, "cos_f": host.cosF, "sin_f": host.sinF,
            "mask_s": host.maskS, "mask_f": host.maskF,
            "sks": sks, "svs": svs, "skf": skf, "svf": svf,
        ]))
        guard let t = out.featureValue(for: "token")?.multiArrayValue,
              let h = out.featureValue(for: "hidden_next")?.multiArrayValue else {
            throw LLMEngineError.generationFailed(reason: "drafter produced invalid output")
        }
        copyFlat(from: h, to: hidden, count: config.H)
        return t[0].intValue
    }

    private func adaptDraftLength(accepted: Int, drafted: Int) {
        recentAccept.append(Double(accepted) / Double(max(drafted, 1)))
        guard recentAccept.count >= 4 else { return }
        let mean = recentAccept.reduce(0, +) / Double(recentAccept.count)
        if mean > 0.9 { draftLen = min(draftLen + 2, maxDraftLen) }
        else if mean < 0.5 { draftLen = max(draftLen - 2, 4) }
        recentAccept.removeAll(keepingCapacity: true)
    }

    private func hiddenBuffer(for s: Int) throws -> MLMultiArray {
        if let b = hiddenBuffers[s] { return b }
        let b = try MLMultiArray(shape: [NSNumber(value: s), NSNumber(value: config.H)], dataType: .float16)
        hiddenBuffers[s] = b
        return b
    }

    private func fillTokid(
        chunkIndex ci: Int, tokens: [Int], softRows: [SoftRowRef?]? = nil
    ) throws -> MLMultiArray {
        let a = config.chunkBounds[ci][0]
        let b = config.chunkBounds[ci][1]
        let nLayers = b - a
        let ple = config.pleDim
        let buf = try tokidBuffer(chunkIndex: ci, s: tokens.count, nLayers: nLayers)
        if let softRows {
            buf.withF16 { dst in
                for (r, tok) in tokens.enumerated() {
                    let row = (softRows[r] != nil) ? MultimodalSlot.perLayerInputRow : tok
                    pleSidecar!.read(
                        row: row, offset: a * ple, count: nLayers * ple,
                        into: dst.baseAddress! + r * nLayers * ple)
                }
            }
            return buf
        }
        buf.withF16 { dst in
            for (r, tok) in tokens.enumerated() {
                pleSidecar!.read(
                    row: tok, offset: a * ple, count: nLayers * ple,
                    into: dst.baseAddress! + r * nLayers * ple)
            }
        }
        return buf
    }

    private func tokidBuffer(chunkIndex ci: Int, s: Int, nLayers: Int) throws -> MLMultiArray {
        let key = "\(ci)_\(s)"
        if let b = tokidBuffers[key] { return b }
        let b = try MLMultiArray(
            shape: [NSNumber(value: s), NSNumber(value: nLayers), NSNumber(value: config.pleDim)],
            dataType: .float16)
        tokidBuffers[key] = b
        return b
    }

    private func copyRow(from source: MLMultiArray, row: Int, into destination: MLMultiArray) {
        source.withF16 { src in
            destination.withF16 { dst in
                dst.baseAddress!.update(from: src.baseAddress! + row * config.H, count: config.H)
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
}

extension CoreMLChainV2 {
    var hiddenSize: Int { config.H }

    func softSlice(_ softRows: [SoftRowRef?]?, _ lo: Int, _ hi: Int) -> [SoftRowRef?]? {
        guard let softRows else { return nil }
        for k in lo..<hi where softRows[k] != nil { return Array(softRows[lo..<hi]) }
        return nil
    }

    private func flattenSegments(_ segments: [PromptSegment]) -> (ids: [Int], rowRefs: [SoftRowRef?]) {
        var ids: [Int] = []
        var rowRefs: [SoftRowRef?] = []
        func appendSoft(_ soft: SoftTokenRows, placeholder: Int, kind: String) {
            precondition(soft.hidden == config.H,
                         "\(kind) soft tokens hidden \(soft.hidden) != chain H \(config.H)")
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
                appendSoft(soft, placeholder: MultimodalSlot.imagePlaceholderID, kind: "image")
            case .audio(let soft):
                appendSoft(soft, placeholder: MultimodalSlot.audioPlaceholderID, kind: "audio")
            }
        }
        return (ids, rowRefs)
    }

    @discardableResult
    func prefillSegments(_ segments: [PromptSegment], blockSize: Int) throws -> Int {
        let (ids, rowRefs) = flattenSegments(segments)
        guard !ids.isEmpty else {
            throw LLMEngineError.generationFailed(reason: "empty prompt (segments)")
        }
        return try prefillRows(ids, blockSize: blockSize, softRows: rowRefs)
    }

    @discardableResult
    func prefillSegments(_ segments: [PromptSegment]) throws -> Int {
        try prefillSegments(segments, blockSize: config.maxS)
    }

    func plannedPrefillWidths(promptLength: Int, from startPosition: Int = 0) -> [Int] {
        let block = max(1, config.maxS)
        var widths: Set<Int> = []
        var pos = 0
        while pos < promptLength {
            var s = min(block, promptLength - pos)
            if config.usesSplitOnehot {
                let base = startPosition + pos
                s = min(s, config.windowSlide - (base % config.windowSlide))
            }
            widths.insert(s)
            pos += s
        }
        return widths.sorted(by: >)
    }

    func materializePrefill(widths: [Int]) throws {
        for n in widths + [1] {
            _ = try hiddenBuffer(for: n)
            guard usesPLE else { continue }
            for (ci, bounds) in config.chunkBounds.enumerated() {
                _ = try tokidBuffer(chunkIndex: ci, s: n, nLayers: bounds[1] - bounds[0])
            }
        }
    }

    func residentPrefillWidths() -> [Int] { hiddenBuffers.keys.sorted() }
}

struct StateBundleIdentity: Codable, Equatable, Sendable {
    var bundleName: String
    var ctx: Int
    var nLayers: Int
    var h: Int
    var hd: Int
    var ghd: Int
    var sliding: Int
    var chunks: [String]
    var lmhead: String
    var stateCount: Int
}

struct StateEntry: Codable, Sendable {
    var name: String
    var chunkIndex: Int
    var shape: [Int]
    var dataType: Int
    var byteCount: Int
    var contiguous: Bool
    var file: String
}

struct StateManifest: Codable, Sendable {
    var formatVersion: Int
    var identity: StateBundleIdentity
    var position: Int

    var pendingNextToken: Int?

    var processedTokens: [Int]
    var states: [StateEntry]
}

enum StatePersistenceError: Error, CustomStringConvertible {
    case identityMismatch(expected: StateBundleIdentity, found: StateBundleIdentity)
    case missingStateFile(String)
    case sizeMismatch(name: String, expected: Int, found: Int)
    case unknownState(String)

    var description: String {
        switch self {
        case .identityMismatch(let expected, let found):
            return "KV restore target bundle does not match the saved source (expected=\(expected), found=\(found))"
        case .missingStateFile(let name):
            return "state file not found: \(name)"
        case .sizeMismatch(let name, let expected, let found):
            return "state \(name) byte-count mismatch (target=\(expected), file=\(found))"
        case .unknownState(let name):
            return "state name not in manifest: \(name)"
        }
    }
}

extension CoreMLChainV2 {

    private func stateNames(of model: MLModel) -> [String] {
        model.modelDescription.stateDescriptionsByName.keys.sorted()
    }

    func stateExportIdentity() -> StateBundleIdentity {
        let total = chunks.reduce(0) { $0 + stateNames(of: $1).count }
        return StateBundleIdentity(
            bundleName: bundleURL.lastPathComponent,
            ctx: config.CTX, nLayers: config.NLAYERS, h: config.H, hd: config.HD,
            ghd: config.GHD, sliding: config.SLIDING,
            chunks: config.chunks, lmhead: config.lmhead, stateCount: total)
    }

    private func isCContiguous(shape: [Int], strides: [Int]) -> Bool {
        guard shape.count == strides.count else { return false }
        var expected = 1
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[i] != expected { return false }
            expected *= shape[i]
        }
        return true
    }

    @discardableResult
    func exportStates(to url: URL, pendingNextToken: Int?, processedTokens: [Int]) throws -> StateManifest {
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path(percentEncoded: false)) { try fm.removeItem(at: url) }
        try fm.createDirectory(at: url, withIntermediateDirectories: true)

        var entries: [StateEntry] = []
        for ci in chunks.indices {
            for name in stateNames(of: chunks[ci]) {
                var caught: Error?
                var entry: StateEntry?
                states[ci].withMultiArray(for: name) { arr in
                    let shape = arr.shape.map(\.intValue)
                    let strides = arr.strides.map(\.intValue)
                    let contiguous = self.isCContiguous(shape: shape, strides: strides)
                    let file = "\(name).bin"
                    arr.withUnsafeBytes { raw in
                        let data = Data(bytes: raw.baseAddress!, count: raw.count)
                        do {
                            try data.write(to: url.appending(path: file), options: .atomic)
                            entry = StateEntry(
                                name: name, chunkIndex: ci, shape: shape,
                                dataType: arr.dataType.rawValue, byteCount: raw.count,
                                contiguous: contiguous, file: file)
                        } catch { caught = error }
                    }
                }
                if let caught { throw caught }
                if let entry { entries.append(entry) }
            }
        }

        let manifest = StateManifest(
            formatVersion: 1, identity: stateExportIdentity(), position: position,
            pendingNextToken: pendingNextToken, processedTokens: processedTokens, states: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: url.appending(path: "manifest.json"), options: .atomic)
        return manifest
    }

    @discardableResult
    func importStates(from url: URL) throws -> StateManifest {
        let fm = FileManager.default
        let manifest = try JSONDecoder().decode(
            StateManifest.self, from: try Data(contentsOf: url.appending(path: "manifest.json")))

        let current = stateExportIdentity()
        guard manifest.identity == current else {
            throw StatePersistenceError.identityMismatch(expected: current, found: manifest.identity)
        }

        let byName = Dictionary(manifest.states.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        for ci in chunks.indices {
            for name in stateNames(of: chunks[ci]) {
                guard let entry = byName[name] else { throw StatePersistenceError.unknownState(name) }
                let fileURL = url.appending(path: entry.file)
                guard fm.fileExists(atPath: fileURL.path(percentEncoded: false)) else {
                    throw StatePersistenceError.missingStateFile(entry.file)
                }
                let data = try Data(contentsOf: fileURL, options: [.alwaysMapped])
                var caught: Error?
                states[ci].withMultiArray(for: name) { arr in
                    arr.withUnsafeMutableBytes { raw, _ in
                        guard raw.count == entry.byteCount, data.count == entry.byteCount else {
                            caught = StatePersistenceError.sizeMismatch(
                                name: name, expected: raw.count, found: data.count)
                            return
                        }
                        data.withUnsafeBytes { src in
                            _ = memcpy(raw.baseAddress!, src.baseAddress!, entry.byteCount)
                        }
                    }
                }
                if let caught { throw caught }
            }
        }

        position = manifest.position
        lastToken = manifest.processedTokens.last ?? 0
        storeSKS = nil; storeSVS = nil; storeSKF = nil; storeSVF = nil
        ladderStoreSyncedRows = 0
        return manifest
    }
}
