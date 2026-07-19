import Foundation

/// 変換パイプライン(tools/e4b)が出力する `convert_config.json`。
/// チェーン実行に必要な全メタデータ(形状・層構成・RoPE 逆周波数など)。
struct ChainConfig: Codable, Sendable {
    var model: String
    var H: Int
    var HD: Int
    var GHD: Int
    var NH: Int
    var NKV: Int
    /// full 層の KV ヘッド数(gemma4_unified の MQA)。nil なら NKV と同じ(E 系)。
    var NKV_FULL: Int?
    /// per-layer embeddings の次元。0 なら PLE なし(12B / gemma4_unified)。
    var PLE: Int
    var NLAYERS: Int
    var CTX: Int
    var SLIDING: Int
    var SOFTCAP: Double
    var NEG: Double
    var firstShared: Int
    var storeLayers: [String: Int]
    var chunkBounds: [[Int]]
    var layerTypes: [String]
    var eosIDs: [Int]
    var chunks: [String]
    var lmhead: String
    var sidecars: Sidecars
    var hostio: HostIO
    /// MTP(投機的デコード)用アセット。nil なら MTP 非対応バンドル。
    var mtp: MTP?

    struct MTP: Codable, Sendable {
        var drafter: String
        var verifyChunks: [String]
        var verifyLmhead: String
        var draftLen: Int

        enum CodingKeys: String, CodingKey {
            case drafter
            case verifyChunks = "verify_chunks"
            case verifyLmhead = "verify_lmhead"
            case draftLen = "draft_len"
        }
    }

    struct Sidecars: Codable, Sendable {
        var embed: File
        /// PLE サイドカー。PLE なしのモデル(12B)には存在しない。
        var ple: File?

        struct File: Codable, Sendable {
            var file: String
            var shape: [Int]
            var dtype: String
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

    enum CodingKeys: String, CodingKey {
        case model, H, HD, GHD, NH, NKV, NKV_FULL, PLE, NLAYERS, CTX, SLIDING, SOFTCAP, NEG
        case firstShared = "first_shared"
        case storeLayers = "store_layers"
        case chunkBounds = "chunk_bounds"
        case layerTypes = "layer_types"
        case eosIDs = "eos_ids"
        case chunks, lmhead, sidecars, hostio, mtp
    }

    var hasPLE: Bool { PLE > 0 }

    /// 層 i の head_dim(full 層は GHD=512、sliding 層は HD=256)。
    func headDim(ofLayer i: Int) -> Int {
        layerTypes[i] == "full_attention" ? GHD : HD
    }

    /// 層 i の KV ヘッド数(gemma4_unified は full 層のみ MQA=1)。
    func kvHeads(ofLayer i: Int) -> Int {
        layerTypes[i] == "full_attention" ? (NKV_FULL ?? NKV) : NKV
    }

    static func load(from url: URL) throws -> ChainConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ChainConfig.self, from: data)
    }
}
