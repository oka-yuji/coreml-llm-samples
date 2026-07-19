import Foundation

/// v2 stateful 変換パイプライン(`tools/12b/u12b_v2_stateful.py`)が出力する
/// `convert_config_v2.json`。KV を `MLState`(デバイス常駐)に載せ、クエリ長 S を
/// `RangeDim(1..128)` にした「KV を運ばない」チェーンのメタデータ。
///
/// 既存ステートレス版の `ChainConfig`(convert_config.json)とはスキーマが異なるため
/// 別型にしている。runtime で必要なキーのみ復号し、`state_shapes` / `inputs` /
/// `last_chunk_extra_outputs` などの情報用キーは無視する(MLState はモデルが自前で確保する)。
struct ChainConfigV2: Codable, Sendable {
    var H: Int
    var HD: Int
    var GHD: Int
    var NH: Int
    var NKV: Int
    /// full 層の KV ヘッド数(gemma4_unified の MQA=1)。
    var NKV_FULL: Int
    var NLAYERS: Int
    /// 焼き込み CTX。ladder(multifunction)バンドルには `CTX` キーが無い(N_MAX 駆動)ため optional 化。
    /// 既定 2048 / ring バンドルは `CTX` を持つので従来どおり(2048)。ladder は `CTX` 経由で N_MAX に落ちる。
    private var ctxStored: Int?
    var SLIDING: Int
    var SOFTCAP: Double
    var NEG: Double
    var sRange: SRange
    var chunkBounds: [[Int]]
    var layerTypes: [String]
    /// store 層 KV(MTP ドラフター用)。ladder は MTP をガード無効化するため `store_layers` を持たない → optional。
    private var storeLayersStored: [String: Int]?
    /// チャンクの `.mlmodelc` 名(順に chunk_0_12 … chunk_36_48)。
    var chunks: [String]
    var lmhead: String
    var hostio: HostIO
    var sidecars: Sidecars
    var eosIDs: [Int]
    /// ctx32k リングモード(タスク B1/B2)のメタ。既定 2048 バンドルには無いので optional
    /// (キー欠落 = nil = 従来経路)。ring では sliding 層がリング窓・full 層が 32K 絶対になる。
    var ctx32k: Ctx32K?

    // --- C-final: multifunction CTX ラダー(v2mmladder。ctx32k / ctx128k の 2 function を 1 バンドルに同梱)---
    /// バンドル種別。`multifunction_ctx_ladder` のとき ladder(関数指定ロード + 共有 MLState 昇格)。既定/ring は nil。
    var kind: String?
    /// 各 function の full 層窓幅(ctx32k=32768 / ctx128k=131072)。ladder のみ。
    var functions: [String: LadderFunction]?
    /// 既定 function 名(ladder。既定 "ctx32k")。
    var defaultFunction: String?
    /// full 層の最大絶対長 N_MAX(ladder=131072)。ring/2048 バンドルには無いので optional。
    var nMax: Int?
    /// sliding リング窓 WSLIDE(ladder=1024)。ring/2048 には無いので optional(= SLIDING に落ちる)。
    var wslide: Int?

    /// ladder の function ごとの full 層窓幅メタ(convert_config_ladder.json の `functions.<name>.full_window`)。
    struct LadderFunction: Codable, Sendable {
        var fullWindow: Int
        enum CodingKeys: String, CodingKey { case fullWindow = "full_window" }
    }

    /// リングモードのメタ(config の `ctx32k` ブロック。情報キー state_dim_rule / hostio_ring は無視)。
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

        struct File: Codable, Sendable {
            var file: String
            var shape: [Int]
            var dtype: String
        }
    }

    enum CodingKeys: String, CodingKey {
        case H, HD, GHD, NH, NKV, NKV_FULL, NLAYERS, SLIDING, SOFTCAP, NEG
        case ctxStored = "CTX"
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

    /// 焼き込み CTX。`CTX` キーがある(2048/ring)ならそれ、無い(ladder)なら N_MAX → SLIDING に落ちる。
    var CTX: Int { ctxStored ?? nMax ?? SLIDING }

    /// store 層 KV マップ(ladder は空 = MTP ガード対象)。
    var storeLayers: [String: Int] { storeLayersStored ?? [:] }

    /// prefill ブロックの最大 S(RangeDim の上限、= 128)。
    var maxS: Int { sRange.upper }

    /// ctx32k リングモードか(sliding = リング窓 / full = 32K 絶対)。既定バンドルは false。
    var isRing: Bool { ctx32k?.mode == "ctx32k-ring" }

    /// multifunction CTX ラダー(v2mmladder。関数指定ロード + 共有 MLState でコピーゼロ昇格)か。
    var isLadder: Bool { kind == "multifunction_ctx_ladder" }

    /// onehot を sliding(リング窓)/ full(絶対)に分離するモードか(ring または ladder)。
    /// 既定 2048 は単一 onehot = false。
    var usesSplitOnehot: Bool { isRing || isLadder }

    /// ladder ctx32k function の full 層窓幅(32768)。ladder 以外では参照しない。
    var ladderCtx32kWindow: Int { functions?["ctx32k"]?.fullWindow ?? 32768 }
    /// ladder ctx128k function の full 層窓幅(131072 = N_MAX)。
    var ladderCtx128kWindow: Int { functions?["ctx128k"]?.fullWindow ?? (nMax ?? 131072) }

    /// full 層の絶対 KV 長 = `onehot_f` / `mask_f` の列数。非 ring は CTX(2048)。
    var ctxFull: Int { ctx32k?.ctxFull ?? CTX }

    /// ドラフター用 full 層絶対幅(= drafter の skf/svf/mask_f 列数)。
    /// ladder は N_MAX(131072): full state は昇格に依らず常に物理 [1,131072,512] で、ctx32k regime では
    /// 先頭 32768 のみ書かれ tail=0 を mask_f(131072 幅)が不可視化する → 単一ドラフター + 単一 mask 規則で正しい。
    /// ring は ctxFull(32K=32768 / 128K=131072)。非 ring は CTX(未使用)。
    var drafterCtxFull: Int { isLadder ? (nMax ?? 131072) : ctxFull }

    /// sliding 層リング窓 = `onehot_s` / `mask_s` の列数。ladder は WSLIDE(1024)、非 ring は SLIDING。
    var windowSlide: Int { ctx32k?.windowSlide ?? (isLadder ? (wslide ?? SLIDING) : SLIDING) }

    /// 実効コンテキスト長(= 位置上限)。ladder は ctx128k 窓(131072)、ring は ctxFull(32768)、非 ring は CTX。
    var effectiveContextLength: Int {
        if isLadder { return ladderCtx128kWindow }
        return isRing ? ctxFull : CTX
    }

    static func load(from url: URL) throws -> ChainConfigV2 {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ChainConfigV2.self, from: data)
    }

    /// バンドル直下の convert_config を解決してロードする。32K int4 バンドル(v2mm32k)は
    /// `convert_config_v2int4.json` のみを持つため、`convert_config_v2.json` が無ければ int4 名へ
    /// フォールバックする。既定バンドル(両ファイル在 or v2.json 在)は従来どおり v2.json を使う = 不変。
    static func loadFromBundle(_ bundleURL: URL) throws -> ChainConfigV2 {
        let fm = FileManager.default
        // 探索順: convert_config_v2.json(既定 2048)→ v2int4.json(32K/128K ring int4)→
        // convert_config_ladder.json(C-final multifunction ladder)。既存 2 バンドルの解決は不変。
        let candidates = [
            bundleURL.appending(path: "convert_config_v2.json"),
            bundleURL.appending(path: "convert_config_v2int4.json"),
            bundleURL.appending(path: "convert_config_ladder.json"),
        ]
        let url = candidates.first { fm.fileExists(atPath: $0.path()) } ?? candidates[0]
        return try load(from: url)
    }
}
