import Foundation

struct ChainConfigV2: Codable, Sendable {
    var H: Int
    var HD: Int
    var GHD: Int
    var NH: Int
    var NKV: Int

    private var nkvFullStored: Int?
    var NLAYERS: Int

    private var ctxStored: Int?
    private var pleStored: Int?
    var SLIDING: Int
    var SOFTCAP: Double
    var NEG: Double
    var sRange: SRange
    var chunkBounds: [[Int]]
    var layerTypes: [String]

    private var storeLayersStored: [String: Int]?

    var chunks: [String]
    var lmhead: String
    var hostio: HostIO
    var sidecars: Sidecars
    var eosIDs: [Int]

    var ctx32k: Ctx32K?

    var kind: String?

    var functions: [String: LadderFunction]?

    var defaultFunction: String?

    var nMax: Int?

    var wslide: Int?

    struct LadderFunction: Codable, Sendable {
        var fullWindow: Int
        enum CodingKeys: String, CodingKey { case fullWindow = "full_window" }
    }

    struct Ctx32K: Codable, Sendable {
        var mode: String
        var ctxFull: Int
        var windowSlide: Int

        enum CodingKeys: String, CodingKey {
            case mode
            case ctxFull = "ctx_full"
            case windowSlide = "window_slide"
        }
    }

    struct SRange: Codable, Sendable {
        var lower: Int
        var upper: Int
        var defaultValue: Int

        enum CodingKeys: String, CodingKey {
            case lower, upper
            case defaultValue = "default"
        }
    }

    struct HostIO: Codable, Sendable {
        var invSlide: [Double]
        var invFull: [Double]

        enum CodingKeys: String, CodingKey {
            case invSlide = "inv_slide"
            case invFull = "inv_full"
        }
    }

    struct Sidecars: Codable, Sendable {
        var embed: File
        var ple: File?

        struct File: Codable, Sendable {
            var file: String
            var shape: [Int]
            var dtype: String
            var scale: String?
        }
    }

    enum CodingKeys: String, CodingKey {
        case H, HD, GHD, NH, NKV, NLAYERS, SLIDING, SOFTCAP, NEG
        case nkvFullStored = "NKV_FULL"
        case ctxStored = "CTX"
        case pleStored = "PLE"
        case sRange = "S_range"
        case chunkBounds = "chunk_bounds"
        case layerTypes = "layer_types"
        case storeLayersStored = "store_layers"
        case chunks, lmhead, hostio, sidecars
        case eosIDs = "eos_ids"
        case ctx32k
        case kind, functions, nMax = "N_MAX", wslide = "WSLIDE"
        case defaultFunction = "default_function"
    }

    var CTX: Int { ctxStored ?? nMax ?? SLIDING }

    var storeLayers: [String: Int] { storeLayersStored ?? [:] }

    var maxS: Int { sRange.upper }

    var NKV_FULL: Int { nkvFullStored ?? NKV }

    var pleDim: Int { pleStored ?? 0 }

    var usesPLE: Bool { sidecars.ple != nil }

    var isRing: Bool { ctx32k?.mode == "ctx32k-ring" }

    var isLadder: Bool { kind == "multifunction_ctx_ladder" }

    var usesSplitOnehot: Bool { isRing || isLadder }

    var ladderCtx32kWindow: Int { functions?["ctx32k"]?.fullWindow ?? 32768 }

    var ladderCtx128kWindow: Int { functions?["ctx128k"]?.fullWindow ?? (nMax ?? 131072) }

    var ctxFull: Int { ctx32k?.ctxFull ?? CTX }

    var drafterCtxFull: Int { isLadder ? (nMax ?? 131072) : ctxFull }

    var windowSlide: Int { ctx32k?.windowSlide ?? (isLadder ? (wslide ?? SLIDING) : SLIDING) }

    var effectiveContextLength: Int {
        if isLadder { return ladderCtx128kWindow }
        return isRing ? ctxFull : CTX
    }

    static func load(from url: URL) throws -> ChainConfigV2 {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ChainConfigV2.self, from: data)
    }

    static func loadFromBundle(_ bundleURL: URL) throws -> ChainConfigV2 {
        let fm = FileManager.default

        let candidates = [
            bundleURL.appending(path: "convert_config_v2int4_int8.json"),
            bundleURL.appending(path: "convert_config_v2.json"),
            bundleURL.appending(path: "convert_config_v2int4.json"),
            bundleURL.appending(path: "convert_config_ladder.json"),
        ]
        let url = candidates.first { fm.fileExists(atPath: $0.path(percentEncoded: false)) } ?? candidates[0]
        return try load(from: url)
    }
}
