import Foundation
import LLMCore

struct ChunkedChainConfig: Sendable {
    let numLayers: Int
    let hidden: Int
    let pleDim: Int
    let firstShared: Int
    let storeLayers: [String: Int]
    let chunkBounds: [[Int]]
    let eosIDs: [Int]

    let CTX: Int
    let HD: Int
    let GHD: Int
    let sliding: Int
    let NEG: Double
    let layerTypes: [String]
    let invSlide: [Double]
    let invFull: [Double]

    let chunkPackages: [String]
    let lmheadPackage: String
    let decodeFunction: String
    let prefillNs: [Int]
    let prefillFunctions: [Int: String]
    let plainN: Int
    let defaultSidecarStage: String
    let sidecarStages: [String: SidecarSet]

    let verifyWidths: [Int]
    let verifyFunctions: [Int: String]
    let batchedHeadPackages: [Int: String]

    var allPrefillFunctions: [Int: String] { prefillFunctions.merging(verifyFunctions) { a, _ in a } }

    struct SidecarSet: Sendable {
        let embed: SidecarFile
        let ple: SidecarFile
    }

    struct SidecarFile: Sendable {
        let file: String
        let rows: Int
        let cols: Int
        let scaleFile: String?
        let runtimeScaleSqrt: Double?
    }

    func headDim(ofLayer i: Int) -> Int { layerTypes[i] == "full_attention" ? GHD : HD }

    var kvHeads: Int { 1 }

    static func load(from url: URL) throws -> ChunkedChainConfig {
        let data = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        return try ChunkedChainConfig(root: root)
    }

    init(root: [String: Any]) throws {
        func req<T>(_ v: Any?, _ what: String) throws -> T {
            guard let x = v as? T else {
                throw LLMEngineError.incompatibleBundle(reason: "chunked config: \(what) missing or invalid")
            }
            return x
        }
        numLayers = try req(root["num_layers"], "num_layers")
        hidden = try req(root["hidden"], "hidden")
        pleDim = try req(root["ple_dim"], "ple_dim")
        firstShared = try req(root["first_shared"], "first_shared")
        let store: [String: Any] = try req(root["store_layers"], "store_layers")
        storeLayers = store.compactMapValues { $0 as? Int }
        let bounds: [[Int]] = try req(root["chunk_bounds"], "chunk_bounds")
        chunkBounds = bounds
        eosIDs = (root["eos_token_ids"] as? [Int]) ?? []

        let io: [String: Any] = try req(root["hostio"], "hostio")
        CTX = try req(io["CTX"], "hostio.CTX")
        HD = try req(io["HD"], "hostio.HD")
        GHD = try req(io["GHD"], "hostio.GHD")
        sliding = try req(io["SLIDING"], "hostio.SLIDING")
        guard let negV = Self.num(io["NEG"]) else {
            throw LLMEngineError.incompatibleBundle(reason: "chunked config: hostio.NEG invalid")
        }
        NEG = negV
        layerTypes = try req(io["layer_types"], "hostio.layer_types")
        invSlide = try Self.doubles(io["inv_slide"], "inv_slide")
        invFull = try Self.doubles(io["inv_full"], "inv_full")

        let cl: [String: Any] = try req(root["corellm"], "corellm")
        chunkPackages = try req(cl["chunks"], "corellm.chunks")
        lmheadPackage = try req(cl["lmhead"], "corellm.lmhead")
        decodeFunction = (cl["decode_function"] as? String) ?? "decode"
        let pf: [String: Any] = try req(cl["prefill"], "corellm.prefill")
        let ns: [Int] = try req(pf["N"], "corellm.prefill.N")
        prefillNs = ns.sorted(by: >)
        let fns: [String: String] = (pf["functions"] as? [String: String]) ?? [:]
        var fmap: [Int: String] = [:]
        for (k, v) in fns { if let n = Int(k) { fmap[n] = v } }
        prefillFunctions = fmap
        plainN = (pf["plain_N"] as? Int) ?? 512

        if let vf = cl["verify"] as? [String: Any] {
            verifyWidths = ((vf["widths"] as? [Int]) ?? []).sorted(by: >)
            var vmap: [Int: String] = [:]
            if let vfns = vf["functions"] as? [String: String] {
                for (k, v) in vfns { if let n = Int(k) { vmap[n] = v } }
            }
            verifyFunctions = vmap
            var hmap: [Int: String] = [:]
            if let lb = vf["lmhead_batched"] as? [String: String] {
                for (k, v) in lb { if let n = Int(k) { hmap[n] = v } }
            }
            batchedHeadPackages = hmap
        } else {
            verifyWidths = []
            verifyFunctions = [:]
            batchedHeadPackages = [:]
        }

        let sc: [String: Any] = try req(cl["sidecars"], "corellm.sidecars")
        defaultSidecarStage = (sc["default_stage"] as? String) ?? "fp16"
        var stages: [String: SidecarSet] = [:]
        for stage in ["fp16", "int8"] {
            guard let s = sc[stage] as? [String: Any] else { continue }
            stages[stage] = SidecarSet(
                embed: try Self.sidecarFile(s["embed"], "sidecars.\(stage).embed"),
                ple: try Self.sidecarFile(s["ple"], "sidecars.\(stage).ple"))
        }
        sidecarStages = stages
    }

    private static func num(_ v: Any?) -> Double? {
        if let d = v as? Double { return d }
        if let n = v as? NSNumber { return n.doubleValue }
        if let i = v as? Int { return Double(i) }
        return nil
    }

    private static func doubles(_ v: Any?, _ what: String) throws -> [Double] {
        guard let arr = v as? [Any] else {
            throw LLMEngineError.incompatibleBundle(reason: "chunked config: \(what) is not an array")
        }
        return arr.map { num($0) ?? 0 }
    }

    private static func sidecarFile(_ v: Any?, _ what: String) throws -> SidecarFile {
        guard let d = v as? [String: Any], let file = d["file"] as? String,
              let shape = d["shape"] as? [Int], shape.count == 2 else {
            throw LLMEngineError.incompatibleBundle(reason: "chunked config: \(what) invalid")
        }
        return SidecarFile(
            file: file, rows: shape[0], cols: shape[1],
            scaleFile: d["scale"] as? String,
            runtimeScaleSqrt: (d["runtime_scale_sqrt"] as? Double)
                ?? (d["runtime_scale_sqrt"] as? Int).map(Double.init))
    }
}
