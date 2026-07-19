import CoreML
import Foundation
import LLMCore

/// Core ML の型(MLModel 等)は Sendable ではないため、並列ロードの集約(task group)を
/// 跨ぐときだけ隔離検査を無効化する箱。エンジン actor 内で生成・直列消費する。
private struct UncheckedBox<T>: @unchecked Sendable {
    let value: T
}

/// 12B v2 **stateful** チェーン実行(4 chunk + lm_head)。
///
/// 既存の `CoreMLChain`(ステートレス・KV を入出力テンソルで運ぶ)と違い、KV は各チャンクの
/// `MLState`(デバイス常駐)に載る。ホストが運ぶのはチャンク間の `hidden`(7.7KB)だけ。
/// クエリ長 S は `RangeDim(1..128)` で可変 — decode は S=1、prefill は S=128 ブロック + 端数。
///
/// - KV バッファ / addKVFeatures / adoptKVOutputs は不要(state 化で消滅)。
/// - reset は `makeState()` の作り直し + position=0。
/// - **GPU 専用**(CPU_ONLY / ANE では複数 state 同時ロードで native segfault)。
///   詳細: docs/results/2026-07-04-12b-v2-stateful-fp16.md、models/gemma4-12b-work/v2/README_v2.md。
final class CoreMLChainV2 {
    let config: ChainConfigV2
    private(set) var position = 0

    // 計測用(t3 / LoadMetrics の内訳)。
    private(set) var loadModelsSeconds: Double = 0
    private(set) var prewarmSeconds: Double = 0
    /// lmhead を実際にロードした compute units(既定はチャンクと同一 = GPU。env で CPU に切替可能)。
    private(set) var headComputeUnits: MLComputeUnits = .cpuAndGPU

    private let bundleURL: URL
    private let chunks: [MLModel]
    private let head: MLModel
    private let embedSidecar: Sidecar
    private let host: HostInputsV2

    // MARK: - C-final: multifunction CTX ラダー(v2mmladder)

    /// ladder の ctx128k function チャンク集合。**共有 MLState**(`states`、ctx32k と同一)で駆動する
    /// = コピーゼロ昇格(read/write_state ゼロ)。非 ladder バンドルでは nil。
    private let chunks128: [MLModel]?
    /// ladder の function 別ホスト入力(full 層幅 32768 / 131072)。sliding は両者同一計算。非 ladder は nil。
    private let hostCtx32k: HostInputsV2?
    private let hostCtx128k: HostInputsV2?
    /// ladder が ctx128k へ昇格済みか。書込位置が ctx32k 窓(32768)以上に達した時点で一方向に true。
    /// reset で false。rewind では戻さない(昇格は一方向 = 仕様)。非 ladder では常に false。
    private var promoted = false

    /// (テスト/診断)ladder が ctx128k function に昇格しているか。
    var isPromoted: Bool { promoted }
    /// (テスト/診断)ladder の function を強制切替する(G2 の ctx128k 直接 / G3 の強制昇格 regime 再現用)。
    /// on=true で以降の forward は位置に依らず ctx128k を使い、一方向に維持される。非 ladder では無効。
    func forcePromotion(_ on: Bool) { if config.isLadder { promoted = on } }

    /// lm_head を最後の 1 位置だけ S=1 で呼ぶための入力バッファ (1, H)。
    private let lmheadInput: MLMultiArray

    /// チャンクごとの KV state(`makeState()`)。reset で作り直す。
    private var states: [MLState] = []
    /// S ごとの hidden_in バッファ(S=1 / 128 / 端数を使い回す)。
    private var hiddenBuffers: [Int: MLMultiArray] = [:]

    // MARK: - MTP(投機的デコード。タスク #5)

    /// 解決済みドラフターの場所。nil なら(同梱)ドラフター投機 非対応バンドル。
    /// - 非 ring(2048): v1 変換物 `drafter.mlmodelc`、探索順 <bundle> → <bundle>/..(従来どおり)。
    /// - ring/ladder(`usesSplitOnehot`): リング対応 `drafter_ring.mlmodelc` を **バンドル直下のみ**で
    ///   検出(親フォールバックなし = 2048 資産 drafter.mlmodelc の誤検出防止)。実在しなければ nil。
    let drafterURL: URL?
    /// ladder regime 切替用の第 2(非昇格 = w32768)ドラフターの場所。`drafter_ring32k.mlmodelc` が
    /// ladder バンドル直下に実在するときだけ非 nil。非 ladder / 不在では nil = 従来どおり w131072 単独
    /// (後方互換)。エンジン / テストヘルパはこれが非 nil のとき第 2 ドラフターもロードして installMTP へ渡す。
    let drafterPreURL: URL?
    /// MTP(投機的デコード)を提供できるか。ドラフター同梱バンドル(2048=drafter / ring・ladder=drafter_ring)、
    /// または PLD 有効(env `CORELLM_MTP_PLD`)なら true。ring/ladder は `drafter_ring.mlmodelc` 実在時のみ
    /// ドラフター投機が有効になり、不在でも PLD(追加モデル不要の n-gram コピー)は単独で MTP を解禁できる。
    var supportsMTP: Bool { drafterURL != nil || pldEnabled }
    private(set) var mtpLoaded = false

    /// ドラフト長 S(既定 4。env `CORELLM_MTP_DRAFT_LEN` で 4..min(12,maxS) に固定可)。
    /// adaptive(env `CORELLM_MTP_ADAPTIVE`)有効時はラウンドごとに [4, maxDraftLen] で自動調整。
    private(set) var draftLen: Int
    private let baseDraftLen: Int
    private let maxDraftLen: Int
    private let adaptiveDraft: Bool
    private var recentAccept: [Double] = []

    // MARK: - PLD(prompt-lookup ハイブリッドドラフト。タスク A1)

    /// PLD(現在列の n-gram 一致から draft をコピー)を有効にするか。env `CORELLM_MTP_PLD`(1|true|yes)。
    /// 無効時は draft 生成が従来ドラフター 100% = 既存経路とバイト等価の挙動。
    let pldEnabled: Bool
    /// PLD の末尾 n-gram 下限 minN(maxN は 3 固定)。既定 3。env `CORELLM_MTP_PLD_MINN`(整数)で上書き。
    /// A1 追試(2026-07-11)で決定: minN=2 は列挙系で区切り(", ")の 2-gram が採択 0.00 の draft を割り込ませ
    /// verify を空費し 0.86× に劣化していた。minN=3 で列挙の PLD ヒットが 0.27→0.00 になり劣化解消(1.00×)、
    /// 逐語の利得は長一致が n=3 で拾えるため維持(1.72×→1.70×)、引用 1.03×/自由 非劣化。8 round・clean 中央値で
    /// 全基準 PASS。pldEnabled=false 時は未使用(PLD OFF の既定経路はバイト等価のまま)。
    let pldMinN: Int
    /// PLD 統計(reset でクリア。直近の生成分)。ベンチログ・ロスレスゲート用。
    private(set) var pldRounds = 0              // PLD が draft を供給したラウンド数。
    private(set) var drafterFallbackRounds = 0  // PLD 有効だが n-gram 不一致でドラフターへ落ちたラウンド数。
    private(set) var pldDraftedTokens = 0       // PLD が出した投機トークン合計(prediction を除く)。
    private(set) var pldAcceptedTokens = 0      // うち採択された投機トークン数(prediction を除く)。

    /// PLD 統計のスナップショット(ヒット率・PLD draft 採択率を含む)。
    struct PLDStats {
        let enabled: Bool
        let pldRounds: Int
        let fallbackRounds: Int
        let draftedTokens: Int
        let acceptedTokens: Int
        /// PLD が draft を供給したラウンド / 全 MTP ラウンド(ドラフターフォールバック含む)。
        var hitRate: Double? {
            let total = pldRounds + fallbackRounds
            return total > 0 ? Double(pldRounds) / Double(total) : nil
        }
        /// PLD 由来の投機トークンのうち採択された割合(prediction を除く分母/分子)。
        var draftAcceptRate: Double? {
            draftedTokens > 0 ? Double(acceptedTokens) / Double(draftedTokens) : nil
        }
    }

    /// 直近の生成分の PLD 統計を返す(reset を跨がない範囲)。
    func pldStatsSnapshot() -> PLDStats {
        PLDStats(
            enabled: pldEnabled, pldRounds: pldRounds, fallbackRounds: drafterFallbackRounds,
            draftedTokens: pldDraftedTokens, acceptedTokens: pldAcceptedTokens)
    }

    // MARK: - ツリー MTP(A2b。主鎖 + 先頭 draft 位置の代替候補 1 ノード)

    /// ツリー MTP を有効にするか。env `CORELLM_MTP_TREE`(1|true|yes)。既定 off = 既存経路とバイト等価。
    /// A2b: PLD ヒットラウンドで、主鎖(PLD 鎖)に「先頭 draft 位置の代替候補 1 ノード(= ドラフター先頭予測)」
    /// を scratch スロットで足して 1 回の verify に載せる。主鎖先頭 draft が棄却され代替候補が正解のとき +1 回収。
    /// ドラフターは argmax `token` しか出さないため drafter top-2(設計 (a))は不可 → 代替候補 = ドラフター先頭予測(設計 (b))。
    /// PLD ミス時は従来ドラフター鎖(代替ノードなし)= PLD 有効時の既存挙動と一致。
    let treeEnabled: Bool
    /// ラウンドごとの実効ツリー ON/OFF。既定 = `treeEnabled`(env)。**ベンチの interleaved A/B 専用**の上書き点で、
    /// production(エンジン経由)はこれを触らない → env 無指定なら常に false = 既定経路とバイト等価。
    var treeActive: Bool
    private let treeHost: TreeHostInputs
    /// ツリー統計(reset でクリア。直近の生成分)。ベンチログ・ロスレスゲート用。
    private(set) var treeAltFired = 0        // 代替ノードを verify に足したラウンド数。
    private(set) var treeAltRecovered = 0    // うち +1 回収(主鎖先頭棄却 → 代替候補採択)したラウンド数。

    /// ツリー統計のスナップショット(reset を跨がない範囲)。
    struct TreeStats {
        let enabled: Bool
        let altFired: Int
        let altRecovered: Int
    }

    /// 直近の生成分のツリー統計を返す。
    func treeStatsSnapshot() -> TreeStats {
        TreeStats(enabled: treeEnabled, altFired: treeAltFired, altRecovered: treeAltRecovered)
    }

    private var drafter: MLModel?
    private var drafterHost: DrafterHostInputs?
    // MARK: - ladder regime 切替: 非昇格(position<32768)用の w32768 ドラフター経路
    //
    // STEP 1 マイクロベンチ(2026-07-19)で w131072 ドラフターの per-call が w32768 の 3.3×
    // (CPU 122 vs 37ms、GPU 62 vs 18ms)= 呼び出しコスト(skf/svf ステージング + W 幅アテンション)が
    // 窓幅にほぼ比例と確認。非昇格中(可視域 ≤position-1 < 32768)は 32768 幅で完全被覆できるため、
    // w32768 ドラフター + 32768 幅ホスト + 小 skf/svf バッファに切り替えて draft レグのコストを ~70% 削減する。
    // 昇格後(promoted)は従来の w131072 経路。drafterPre 不在(後方互換)なら常に w131072。
    private var drafterPre: MLModel?          // 非昇格用 w32768 ドラフター(drafterPreURL 実在時のみ)。
    private var drafterHostPre: DrafterHostInputs?  // mask_f 幅 32768 のホスト(sliding は共有計算)。
    /// 直近ラウンドで非昇格(w32768)経路を踏んだ回数 / 昇格(w131072)経路を踏んだ回数。reset でクリア。
    /// RD2 ゲートの「両経路踏破」証跡・ベンチの regime 確認に使う。
    private(set) var drafterPreRounds = 0
    private(set) var drafterFullRounds = 0
    private var draftEmbed: MLMultiArray?     // (H,) ドラフター embed 入力
    private var draftHidden: MLMultiArray?    // (H,) ドラフター hidden 入力(チェーンで更新)
    private var lastHiddenBuffer: MLMultiArray? // (H,) 直前確定位置の本体最終 hidden
    /// 直前に処理した確定トークン(ドラフトの hidden 準備に使う)。
    private var lastToken = 0

    /// store 層 KV(ドラフターの sks/svs/skf/svf 入力へ渡すポインタ)。sliding=46、full=47。
    /// - 非 ladder(2048/ring): 最終チャンク出力(`k_46_out` 等)の参照をそのまま持つ(ゼロコピー。
    ///   ドラフターは別モデルなので次の本体推論まで生存する)。形状は 2048=[8,2048,256]/[1,2048,512]、
    ///   ring=[8,1024,256]/[1,ctxFull,512]。
    /// - ladder: チャンクは store 出力を持たない(KV は state 常駐)ため、最終チャンク state から毎ラウンド
    ///   コピーした所有バッファ(`ladderStore*`)を指す。
    private var storeSKS: MLMultiArray?
    private var storeSVS: MLMultiArray?
    private var storeSKF: MLMultiArray?
    private var storeSVF: MLMultiArray?
    /// 最終チャンクが出す store 層 KV の出力特徴名(config 駆動。非 ladder の出力 capture 用)。
    private let storeKName: (sliding: String, full: String)
    private let storeVName: (sliding: String, full: String)

    // MARK: - ladder(v2mmladder)の store 層 KV コピー(state → 所有バッファ)
    //
    // ladder のチャンクは store 出力を持たず、KV は各チャンクの MLState に載る。ドラフターは state を
    // 直接読めない(withMultiArray の closure 外へ参照を持ち出せない)ため、最終チャンク state を毎ラウンド
    // 事前確保バッファへ memcpy し、storeSKS/SVS/SKF/SVF をそこへ向ける。full は物理 [1,131072,512]
    // (ctx32k regime でも tail=0 を mask_f が隠す)。非 ladder では nil(出力 capture のみ)。

    /// ladder 専用: store 層 KV の所有バッファ(fp16、state 実形状に一致)。installMTP で確保、mtpRound で更新。
    private var ladderStoreSlidingK: MLMultiArray?
    private var ladderStoreSlidingV: MLMultiArray?
    private var ladderStoreFullK: MLMultiArray?
    private var ladderStoreFullV: MLMultiArray?
    /// ladder regime 切替: 非昇格用の full 層小バッファ [1,32768,512](drafterPre 実在時のみ確保)。
    /// sliding バッファ(ladderStoreSlidingK/V [8,1024,256])は窓幅 1024 が両 regime 共通なので共用する。
    private var ladderStoreFullKPre: MLMultiArray?
    private var ladderStoreFullVPre: MLMultiArray?
    /// captureLadderStoreKV が現在「小バッファ(非昇格)」へ同期しているか。regime 遷移(小⇄大)を検出して
    /// 同期水位 `ladderStoreSyncedRows` をリセットし、切替先バッファへ確定 prefix [0,position) を全量再同期する。
    private var ladderCapturePre = false
    /// ladder の store 層 state 名(`k_<lastSliding>` 等。layerTypes から導出、storeLayers が空でも動く)。非 ladder は nil。
    private var ladderStoreStateNames: (slidingK: String, slidingV: String, fullK: String, fullV: String)?
    /// ladder full 層(絶対レイアウト [1,131072,512])の所有バッファに同期済みの行数(水位)。
    /// captureLadderStoreKV は確定 prefix [0,position) が verify 後不変であることを使い、
    /// 前回同期済み `syncedRows` から `min(position, numRows)` までの増分行だけをコピーする。
    /// 無効化点: reset/importStates=0、rewind=min(self,position)、installMTP(バッファ確保時)=0。
    /// rewind の縮小を忘れると多ターン継続で prefix 書換分が再コピーされずロスレスが壊れる。
    private var ladderStoreSyncedRows = 0

    init(bundleURL: URL, computeUnits: MLComputeUnits) async throws {
        guard computeUnits == .cpuAndGPU || computeUnits == .all else {
            throw LLMEngineError.incompatibleBundle(
                reason: "v2 stateful は GPU が必須です(computeUnits=\(computeUnits.rawValue))。"
                    + "CPU_ONLY / ANE では複数 MLState 同時ロードで native segfault になります。"
                    + "cpuAndGPU か all を指定してください")
        }
        let config = try ChainConfigV2.loadFromBundle(bundleURL)
        self.config = config
        self.bundleURL = bundleURL

        // --- チャンク + lmhead を並列ロード(withThrowingTaskGroup)---
        // ロード内訳(docs/results/2026-07-05-12b-v2-load-breakdown.md): ~40s ロードのほぼ全量が lmhead
        // (int8・262K vocab の argmax GEMM)の GPU/ANE MPSGraph 特殊化。チャンク 4 個は各 ~1s。
        // 特殊化はプロセス跨ぎでも同一プロセス再ロードでもキャッシュされない(.all は 40s を初回 predict
        // へ遅延するだけ)。lmhead は stateless なので CPU_ONLY ロードが可能で、それだと ~0.8s(特殊化回避)。
        // ただし CPU 実行は decode を 20→8.5 tok/s に落とす(int8 262K matmul が per-token で重い)ため、
        // **既定はチャンクと同一 CU(GPU)= decode 優先**とし、ロードは先行ロードで隠す方針。
        // CPU への切替は env `CORELLM_V2_HEAD_CU=cpu`(ロード優先/decode 少ない用途)。正しさは CU 不変
        // (fp16 は 32tok EXACT、int4 は自然域 EXACT + 既知 near-tie 1 箇所を実測で確認)。
        let headCU = Self.resolveHeadComputeUnits(chunkUnits: computeUnits)
        headComputeUnits = headCU
        let clock = ContinuousClock()
        let loadStart = clock.now
        if config.isLadder {
            // ladder: 各チャンクを `functionName = "ctx32k" | "ctx128k"` で **2 回** 関数指定ロードする。
            // .mlpackage は初回 loadCompiled で .mlmodelc 化(1 回だけ、両 function を含む)。以降は同一
            // .mlmodelc を functionName 違いで開くだけ = 再コンパイルなし・重み dedup。
            // メモリ安全のため直列ロード(初回コンパイルの並列スパイクで watchdog を踏まないため)。
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
            // 既存経路(2048 / ring 単関数)は並列ロードで不変。
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

        embedSidecar = try Sidecar(
            url: bundleURL.appending(path: config.sidecars.embed.file),
            rows: config.sidecars.embed.shape[0], cols: config.sidecars.embed.shape[1]
        )
        host = HostInputsV2(config: config)
        if config.isLadder {
            // full 幅を function ごとに固定した 2 ホスト(sliding は同一計算)。onehot_f/mask_f 幅が異なる。
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

        // --- MTP 設定(ドラフター探索・draftLen・store 層出力名)---
        let fm = FileManager.default
        if config.usesSplitOnehot {
            // ring / ladder: リング対応ドラフター `drafter_ring.mlmodelc` を **バンドル直下のみ**で検出する。
            // 親ディレクトリへのフォールバック探索はしない(2048 資産 `drafter.mlmodelc` の誤検出防止)。
            // 実在しなければ nil = 同梱ドラフター投機は無効(PLD は追加モデル不要で supportsMTP を別途立て得る)。
            // DrafterHostInputs / store KV は ring/ladder のリング KV レイアウトに対応済み。
            let ringDrafter = bundleURL.appending(path: "drafter_ring.mlmodelc")
            drafterURL = fm.fileExists(atPath: ringDrafter.path()) ? ringDrafter : nil
            // ladder のみ: 非昇格 regime 用の第 2 ドラフター `drafter_ring32k.mlmodelc`(w32768)を直下で検出。
            // 実在しなければ nil = 従来どおり w131072 単独(後方互換)。ring(32k/128k 単一 regime)では resolve しない。
            if config.isLadder {
                let preDrafter = bundleURL.appending(path: "drafter_ring32k.mlmodelc")
                drafterPreURL = fm.fileExists(atPath: preDrafter.path()) ? preDrafter : nil
            } else {
                drafterPreURL = nil
            }
        } else {
            // 既定 2048: 従来の探索(<bundle>/drafter.mlmodelc → <bundle>/../drafter.mlmodelc)= 不変。
            let drafterCandidates = [
                bundleURL.appending(path: "drafter.mlmodelc"),
                bundleURL.deletingLastPathComponent().appending(path: "drafter.mlmodelc"),
            ]
            drafterURL = drafterCandidates.first { fm.fileExists(atPath: $0.path()) }
            drafterPreURL = nil
        }

        let env = ProcessInfo.processInfo.environment
        // 固定 draft_len の上限。既定 4 は不変(byte 等価)。C3 の draft_len スイープで 16 まで測るため 16 に緩めた
        // (verify は本体グラフの S=drafts.count で maxS=128 まで許容 = グラフ制約は無い。上限は adaptive 天井も兼ねる)。
        let maxDraft = min(16, config.maxS)
        maxDraftLen = maxDraft
        // 既定 draft_len = 4(タスク #9 で決定)。多様 8 プロンプト分布で median/min tok/s が最良
        // (採択 ~0.5 帯は短いドラフトほど verify の無駄が減る)。固定 6/8/10・adaptive はいずれも
        // median か min で 4 に劣る。詳細: docs/results/2026-07-05-12b-adaptive-draft.md。
        // env `CORELLM_MTP_DRAFT_LEN`(4..maxDraft 固定)/ `CORELLM_MTP_ADAPTIVE`(自動調整)で上書き可。
        let requested = env["CORELLM_MTP_DRAFT_LEN"].flatMap(Int.init) ?? 4
        baseDraftLen = min(max(requested, 4), maxDraft)
        draftLen = baseDraftLen
        adaptiveDraft = ["1", "true", "yes"].contains((env["CORELLM_MTP_ADAPTIVE"] ?? "").lowercased())
        // PLD(prompt-lookup ハイブリッドドラフト、タスク A1)。既定 off = 既存経路とバイト等価。
        pldEnabled = ["1", "true", "yes"].contains((env["CORELLM_MTP_PLD"] ?? "").lowercased())
        // PLD minN 既定 3(A1 追試 2026-07-11 で決定: 列挙系の区切り 2-gram 誤一致による 0.86× 劣化を解消、
        // 逐語利得は維持)。env `CORELLM_MTP_PLD_MINN`(整数)で上書き可(minN=2 の旧挙動も再現できる)。
        // maxN=3 固定なので minN>3 は PLD 実質 OFF、minN<1 は無意味 → 1 で下限クランプ。
        pldMinN = max(1, env["CORELLM_MTP_PLD_MINN"].flatMap(Int.init) ?? 3)
        // ツリー MTP(A2b)。既定 off = 既存経路とバイト等価。PLD とは独立に組み合わせ可(発火は PLD ヒット時のみ)。
        treeEnabled = ["1", "true", "yes"].contains((env["CORELLM_MTP_TREE"] ?? "").lowercased())
        treeActive = treeEnabled

        let storeS = config.storeLayers["sliding_attention"] ?? 46
        let storeF = config.storeLayers["full_attention"] ?? 47
        storeKName = (sliding: "k_\(storeS)_out", full: "k_\(storeF)_out")
        storeVName = (sliding: "v_\(storeS)_out", full: "v_\(storeF)_out")

        // 全 stored property が初期化済みになったので self メソッドを呼べる。
        states = chunks.map { $0.makeState() }
        position = 0

        // プリウォーム: S=1 を 1 回流して Metal 形状特殊化を先に済ませ、state/position を戻す。
        let warmStart = clock.now
        _ = try decodeStep(tokenID: 2)
        try reset()
        prewarmSeconds = (clock.now - warmStart) / .seconds(1)
    }

    /// lmhead(stateless)の実効 compute units。**既定はチャンクと同一 CU(= GPU、decode 優先)**。
    /// env `CORELLM_V2_HEAD_CU`(cpu|gpu|ane|all|chunks)で上書き可能:
    /// `cpu` はロードを ~40s→~7s に短縮する代わりに decode を落とす(ロード優先用途/計測)。
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

    /// KV state を作り直して位置を巻き戻す(v2 は state 化のためゼロ埋め不要)。
    func reset() throws {
        states = chunks.map { $0.makeState() }
        position = 0
        lastToken = 0
        promoted = false   // ladder: 新規会話は ctx32k から。
        draftLen = baseDraftLen
        recentAccept.removeAll(keepingCapacity: true)
        storeSKS = nil; storeSVS = nil; storeSKF = nil; storeSVF = nil
        ladderStoreSyncedRows = 0   // 新規会話は full 層バッファを全行再同期。
        ladderCapturePre = false    // 新規会話は非昇格(小バッファ)regime から捕捉し直す。
        drafterPreRounds = 0; drafterFullRounds = 0
        pldRounds = 0; drafterFallbackRounds = 0; pldDraftedTokens = 0; pldAcceptedTokens = 0
        treeAltFired = 0; treeAltRecovered = 0
        // treeActive は reset で戻さない(ベンチが 1 世代ごとに設定する上書き点。既定は env 由来の treeEnabled)。
    }

    /// KV state は保持したまま位置だけを巻き戻す(マルチターン継続 / prefix cache)。
    /// state 内 KV の巻き戻し位置以降は後続の forwardChunks(prefill / decode)が onehot 書き込みで
    /// 上書きし、mask が未確定位置を不可視化する(MTP 部分採択で実証済みの原理と同一)。
    /// MTP の store 層 KV ポインタは無効化する(直後の prefill が最新を捕捉し直す)。
    func rewind(to newPosition: Int) {
        position = max(0, min(newPosition, config.effectiveContextLength))
        storeSKS = nil; storeSVS = nil; storeSKF = nil; storeSVF = nil
        // 巻き戻し位置以降は後続 prefill/decode で書き換わる → 同期水位を position まで縮めて再コピー対象に戻す
        // (縮めないと [position,oldSynced) の古い KV が再同期されず可視域に混入 = 多ターンのロスレス破壊)。
        ladderStoreSyncedRows = min(ladderStoreSyncedRows, position)
    }

    /// MTP アセット(ドラフター)の注入。ロード済み MLModel はエンジン(actor)側で
    /// 用意して渡す(Swift 6 の隔離境界のため)。verify は本体 v2 グラフの S=draftLen 呼び出しで
    /// 行うため、v1 のような専用 verify チェーンは不要。
    /// - `drafter`: 昇格後(および後方互換の単独)ドラフター。ladder=w131072、ring=ctxFull 幅、2048=v1 変換物。
    /// - `drafterPre`: ladder regime 切替用の非昇格 w32768 ドラフター(`drafterPreURL` 実在時にエンジン/
    ///   テストがロードして渡す)。nil(既定)なら従来どおり `drafter` 単独 = 全 regime で w131072(後方互換)。
    func installMTP(drafter: MLModel, drafterPre: MLModel? = nil) throws {
        guard !mtpLoaded, supportsMTP else { return }
        self.drafter = drafter
        drafterHost = try DrafterHostInputs(config: config)
        // ladder regime 切替: drafterPre が渡された(= drafter_ring32k.mlmodelc 実在)ときだけ、非昇格用の
        // w32768 ドラフターと 32768 幅ホストを用意する。非 ladder ではエンジン/テストが drafterPre=nil を渡す。
        if config.isLadder, let drafterPre {
            self.drafterPre = drafterPre
            drafterHostPre = try DrafterHostInputs(
                config: config, maskFWidthOverride: config.ladderCtx32kWindow)
        }
        draftEmbed = try MLMultiArray(shape: [NSNumber(value: config.H)], dataType: .float16)
        draftHidden = try MLMultiArray(shape: [NSNumber(value: config.H)], dataType: .float16)
        lastHiddenBuffer = try MLMultiArray(shape: [NSNumber(value: config.H)], dataType: .float16)
        // ladder は store 出力を持たない → 最終チャンク state を毎ラウンドコピーするための所有バッファを確保。
        if config.isLadder { try allocateLadderStoreBuffers() }
        mtpLoaded = true
    }

    /// ladder の store 層 KV バッファを最終チャンク state の実形状に合わせて確保する(installMTP 内で 1 回)。
    /// store 層 index は `config.layerTypes` の「最後の sliding」「最後の full」から導出(storeLayers が空でも動く)。
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
                    reason: "ladder store state \(name) が最終チャンク(index \(last))に存在しない = MTP capture 不可")
            }
            var out: MLMultiArray?
            var caught: Error?
            states[last].withMultiArray(for: name) { arr in
                do { out = try MLMultiArray(shape: arr.shape, dataType: arr.dataType) }
                catch { caught = error }
            }
            if let caught { throw caught }
            guard let out else {
                throw LLMEngineError.generationFailed(reason: "ladder store state \(name) のバッファ確保に失敗")
            }
            return out
        }
        ladderStoreSlidingK = try allocLike(names.slidingK)
        ladderStoreSlidingV = try allocLike(names.slidingV)
        ladderStoreFullK = try allocLike(names.fullK)
        ladderStoreFullV = try allocLike(names.fullV)
        // full 層は増分コピーで行 [position,numRows) を一度も書かない(mask_f が可視域を position-1 で切る)。
        // MLMultiArray(shape:) のバッキングは未初期化 = NaN の可能性があり、マスク前スコア計算(q·k を全行)に
        // NaN が混ざると softmax が NaN 汚染する。確保時に一度だけゼロ埋めして「stale/ゼロでも安全」を担保する
        // (現行の全量コピーは state 側のゼロ埋め領域をそのまま反映していたため、その挙動と等価に保つ)。
        zeroBuffer(ladderStoreFullK)
        zeroBuffer(ladderStoreFullV)
        // regime 切替(drafterPre 実在)のときのみ、非昇格用 full 層小バッファ [1,32768,512] を確保する。
        // D(=512)と dtype は大バッファ(= state 実形状 [1,131072,512])から取り、行数だけ ctx32k 窓(32768)。
        // 大バッファ同様ゼロ埋め(増分コピーが [position,32768) を書かない NaN 事故防止)。sliding は共用のため確保しない。
        if drafterPre != nil, let big = ladderStoreFullK {
            let d = big.shape.count >= 3 ? big.shape[2].intValue : 512
            let w32 = config.ladderCtx32kWindow   // 32768
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

    /// MLMultiArray の raw バッキングを 0 で埋める(未初期化バッキングの NaN 事故防止)。
    private func zeroBuffer(_ arr: MLMultiArray?) {
        arr?.withUnsafeMutableBytes { dst, _ in
            if let base = dst.baseAddress { memset(base, 0, dst.count) }
        }
    }

    /// ladder: 最終チャンクの store 層 state を所有バッファへ反映し storeSKS/SVS/SKF/SVF をそこへ向ける(毎ラウンド)。
    /// - sliding(k_46/v_46 [8,1024,256]): リング窓の in-place 巡回更新のため増分不可 → 毎ラウンド全量コピー(4MB×2)。
    /// - full(k_47/v_47 [1,131072,512]): 絶対レイアウトで確定 prefix [0,position) は verify 後不変
    ///   (棄却行は position 以降にしか無く次ラウンドの verify が上書きする)。よって前回同期済み
    ///   `ladderStoreSyncedRows` から `upTo=min(position,numRows)` までの増分行だけを memcpy する。
    ///   ドラフターの mask_f は slot ≤ position-1 しか開かないため [position,numRows) は stale/ゼロでも安全。
    private func captureLadderStoreKV() throws {
        guard let names = ladderStoreStateNames,
              let sk = ladderStoreSlidingK, let sv = ladderStoreSlidingV,
              let fkBig = ladderStoreFullK, let fvBig = ladderStoreFullV else { return }
        let last = chunks.count - 1
        // sliding: リング巡回のため増分不可 → 全量(小/大 regime 共通の共有バッファ)。
        try copyState(chunk: last, name: names.slidingK, into: sk)
        try copyState(chunk: last, name: names.slidingV, into: sv)

        // full: regime に応じて小(非昇格 [1,32768,512])/ 大(昇格後 [1,131072,512])バッファへ増分コピー。
        // 非昇格中は position-1 < 32768 が可視域 → 32768 幅で完全被覆。昇格を跨ぐと大バッファへ切替。
        // src state は常に物理 [1,131072,512](ctx32k regime でも先頭 32768 のみ書かれ tail=0 は mask_f が隠す)。
        let usePre = (drafterPre != nil) && !promoted
        let fk = usePre ? (ladderStoreFullKPre ?? fkBig) : fkBig
        let fv = usePre ? (ladderStoreFullVPre ?? fvBig) : fvBig
        // regime 遷移(小⇄大)を検出したら同期水位を 0 に戻し、切替先バッファへ確定 prefix [0,position) を
        // 全量再同期する(切替先は前 regime 中に書かれていない = zero-init のまま。昇格時 position≈32768+ で ~34MB)。
        if usePre != ladderCapturePre {
            ladderStoreSyncedRows = 0
            ladderCapturePre = usePre
        }
        // numRows は「同期先バッファ」の物理行数(小=32768 / 大=131072)。upTo は state 側 prefix 上限で clamp。
        let numRows = fk.shape.count >= 2 ? fk.shape[1].intValue : Int.max
        let upTo = min(position, numRows)
        try copyFullStateRows(chunk: last, name: names.fullK, into: fk, from: ladderStoreSyncedRows, upTo: upTo)
        try copyFullStateRows(chunk: last, name: names.fullV, into: fv, from: ladderStoreSyncedRows, upTo: upTo)
        ladderStoreSyncedRows = upTo
        storeSKS = sk; storeSVS = sv; storeSKF = fk; storeSVF = fv
    }

    /// full 層(絶対レイアウト)の store state を所有バッファへ反映する。src/dest がともに C 連続かつ
    /// [1,N,D] shape なら行 [from,upTo) のバイト範囲だけを memcpy(確定 prefix [0,from) は不変ゆえ再コピー不要)。
    /// 非 C 連続 / 想定外 shape / バイト数不一致は従来の全量 `copyState` へフォールバックする(正しさ優先)。
    /// コピー 0 行(from==upTo)は memcpy をスキップ(ポインタのセットは呼び出し側が毎回行う)。
    private func copyFullStateRows(chunk ci: Int, name: String, into dest: MLMultiArray,
                                   from: Int, upTo: Int) throws {
        var didIncremental = false
        states[ci].withMultiArray(for: name) { arr in
            // strides が [.., D, 1] 型(行優先・パディングなし)かを src/dest 双方で検証。
            // dest は src と同行数(大バッファ)でも少行数(regime 切替の小バッファ [1,32768,512])でもよい。
            // src state は絶対レイアウト [1,131072,512] で、非昇格の可視域 [0,position)(< 32768)は行番号が
            // src/dest で一致するため、同一バイトオフセットへ行 [lo,hi) を memcpy できる。
            guard Self.isCContiguous(arr), Self.isCContiguous(dest),
                  arr.shape.count == 3, arr.shape[0].intValue == 1,
                  dest.shape.count == 3, dest.shape[0].intValue == 1 else { return }
            let srcRows = arr.shape[1].intValue
            let destRows = dest.shape[1].intValue
            guard srcRows > 0, destRows > 0 else { return }
            arr.withUnsafeBytes { src in
                dest.withUnsafeMutableBytes { dst, _ in
                    guard src.count % srcRows == 0, dst.count % destRows == 0 else { return }
                    let rowBytes = src.count / srcRows                 // = D × 要素バイト(shape[0]==1)。
                    guard rowBytes == dst.count / destRows else { return }  // 同一 D×要素バイトのみ。
                    // dest 行数を超えて書かない(小バッファは先頭 destRows 行のみ保持)。src 側も同上限で clamp。
                    let bound = min(srcRows, destRows)
                    let lo = max(0, min(from, bound))
                    let hi = max(lo, min(upTo, bound))
                    didIncremental = true
                    guard hi > lo else { return }                      // 0 行はスキップ。
                    let offset = lo * rowBytes
                    let length = (hi - lo) * rowBytes
                    _ = memcpy(dst.baseAddress! + offset, src.baseAddress! + offset, length)
                }
            }
        }
        if !didIncremental {
            // 非 C 連続 / 想定外 shape / per-row バイト不一致: 従来挙動(全量 memcpy。バイト数不一致は throw)。
            // 小バッファ(dest≠src サイズ)でこの分岐に落ちると copyState がバイト不一致で throw = 異常検知。
            try copyState(chunk: ci, name: name, into: dest)
        }
    }

    /// MLMultiArray が C 連続(行優先・パディングなし)かを strides で検証する。
    /// shape [d0,…,dn] に対し stride[i] == Π_{j>i} d_j(最内 = 1)なら真。
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

    /// チャンク `ci` の state `name` の raw バイト列を `dest` へ memcpy する(バイト数一致を検証)。
    private func copyState(chunk ci: Int, name: String, into dest: MLMultiArray) throws {
        var caught: Error?
        states[ci].withMultiArray(for: name) { arr in
            arr.withUnsafeBytes { src in
                dest.withUnsafeMutableBytes { dst, _ in
                    guard src.count == dst.count else {
                        caught = LLMEngineError.generationFailed(
                            reason: "ladder store state \(name) バイト数不一致(state=\(src.count) buf=\(dst.count))")
                        return
                    }
                    _ = memcpy(dst.baseAddress!, src.baseAddress!, src.count)
                }
            }
        }
        if let caught { throw caught }
    }

    // MARK: - 生成

    /// プロンプトを S ブロックで流して KV を張り(lm_head は呼ばない)、最後の 1 位置だけ
    /// S=1 で lm_head を 1 回叩いて最初の生成トークン(argmax)を返す。位置はプロンプト長へ進む。
    ///
    /// `blockSize` は既定 128(RangeDim 上限)。端数はそのまま S=端数 で 1 回(パディング不要)。
    /// `blockSize=1` にすると S=1 逐次 prefill(= 既存ステートレス経路 / python G2 と同一レジーム)。
    @discardableResult
    func prefill(_ promptIDs: [Int], blockSize: Int) throws -> Int {
        precondition(!promptIDs.isEmpty, "prefill には最低 1 トークン必要")
        // **現在位置 `start` から**張る(reset 直後は 0、rewind 後は巻き戻し位置 = マルチターン継続)。
        let start = position
        let ctxLimit = config.effectiveContextLength
        guard start + promptIDs.count <= ctxLimit else {
            throw LLMEngineError.generationFailed(
                reason: "prefill 後の位置 \(start + promptIDs.count) がコンテキスト長 \(ctxLimit) を超過")
        }
        let block = max(1, min(blockSize, config.maxS))
        // ring / ladder: sliding が物理リング窓(windowSlide=1024)なので、1 回の predict が窓境界を跨ぐと
        // リングスロット衝突で早行の最古位置が失われる(tail-loss)。prefill は最終行 hidden のみ消費し、
        // KV 書込は行ごと独立なので、各 predict を単一の windowSlide 整列窓に収めれば最終行と KV はロスレス。
        // 32768(ladder ctx32k 窓)も 1024 の倍数 → この境界も跨がず、昇格が block 境界と自然に一致する。
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
            lastHidden = try forwardChunks(tokens: Array(promptIDs[pos..<pos + s]), basePosition: base)
            lastRow = s - 1
            pos += s
        }
        position = start + promptIDs.count

        guard let lastHidden else {
            throw LLMEngineError.generationFailed(reason: "prefill: hidden が空")
        }
        // 最後の位置の hidden だけ (1, H) に取り出し、S=1 で lm_head を 1 回。
        copyRow(from: lastHidden, row: lastRow, into: lmheadInput)
        // MTP: 最初のラウンドのドラフト起点(直前確定トークンとその最終 hidden)を保持。
        if mtpLoaded, let lastHiddenBuffer {
            lastToken = promptIDs[promptIDs.count - 1]
            copyRow(from: lastHidden, row: lastRow, into: lastHiddenBuffer)
        }
        return try lmheadArgmax(lmheadInput)
    }

    /// S=1 逐次 decode。トークン 1 つを処理し、次トークン(greedy argmax)を返す。位置は 1 進む。
    func decodeStep(tokenID: Int) throws -> Int {
        guard position < config.effectiveContextLength else {
            throw LLMEngineError.generationFailed(
                reason: "コンテキスト長 \(config.effectiveContextLength) を超過")
        }
        let hidden = try forwardChunks(tokens: [tokenID], basePosition: position)  // (1, H)
        let next = try lmheadArgmax(hidden)
        if mtpLoaded, let lastHiddenBuffer {
            lastToken = tokenID
            copyRow(from: hidden, row: 0, into: lastHiddenBuffer)
        }
        position += 1
        return next
    }

    // MARK: - 内部

    /// S 個のトークンを 4 チャンクへ通し、最終チャンクの hidden (S, H) を返す(標準経路)。
    /// 各チャンクは自層の KV を state に in-place 書き込みする(onehot が書き込み位置)。
    private func forwardChunks(tokens: [Int], basePosition: Int) throws -> MLMultiArray {
        let s = tokens.count
        let hin = try hiddenBuffer(for: s)
        // hidden_in = embed[token](√H はサイドカーに焼き込み済み)。
        hin.withF16 { buf in
            for (r, tok) in tokens.enumerated() {
                embedSidecar.read(row: tok, into: buf.baseAddress! + r * config.H)
            }
        }
        if config.isLadder {
            // ladder: 書込最大位置が ctx32k 窓(32768)以上になる時点で ctx128k へ昇格(一方向)。
            // 共有 MLState(`states`)を別 function のチャンクで駆動 = コピーゼロ昇格。
            let maxPos = basePosition + s - 1
            if !promoted && maxPos >= config.ladderCtx32kWindow { promoted = true }
            let activeChunks = promoted ? chunks128! : chunks
            let activeHost = promoted ? hostCtx128k! : hostCtx32k!
            let feats = try activeHost.filled(base: basePosition, count: s)
            return try runChunks(chunks: activeChunks, hidden: hin, feats: feats, splitOnehot: true)
        }
        let feats = try host.filled(base: basePosition, count: s)
        return try runChunks(
            chunks: chunks, hidden: hin, feats: feats, splitOnehot: config.usesSplitOnehot)
    }

    /// ツリー MTP(A2b)の verify: 主鎖 `mainTokens`(連番スロット)+ 代替候補ノード `altToken`
    /// (scratch=`altSlot`・RoPE 位置 basePosition+1・可視 prefix+prediction+self)を S=mainCount+1 で 1 回通す。
    /// 主鎖行は標準 verify とビット等価のホスト値。代替行は `TreeHostInputs` が兄弟マスクを作る。
    /// 返り値 (mainCount+1, H): 行 0..<mainCount = 主鎖 hidden、行 mainCount = 代替ノード hidden。
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
        // ツリー MTP は 2048 単関数専用(ring / ladder では MTP をガード無効化)= 単一 onehot。
        return try runChunks(chunks: chunks, hidden: hin, feats: feats, splitOnehot: false)
    }

    /// 埋め込み済み hidden_in (S, H) と 7 本のホスト入力を `activeChunks` へ通し、最終 hidden (S, H) を返す。
    /// `forwardChunks`(標準 / ladder)と `forwardChunksTree`(ツリー)の共通コア。store 層 KV も同一条件で捕捉する。
    /// `splitOnehot`: onehot を onehot_s / onehot_f に分離するか(ring / ladder = true、2048 = false)。
    private func runChunks(
        chunks activeChunks: [MLModel], hidden hin: MLMultiArray,
        feats: HostInputsV2.Buffers, splitOnehot: Bool
    ) throws -> MLMultiArray {
        var hidden: MLMultiArray = hin
        for ci in activeChunks.indices {
            // ring / ladder: onehot を onehot_s(1024 リング)/ onehot_f(絶対)に分離。
            // 既定 2048: 単一 onehot(= CTX 列)。mask_s/mask_f の列数も HostInputsV2 が各モードで用意する。
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
            // ladder は activeChunks(ctx32k/ctx128k)を切り替えるが state は共有(states[ci])= コピーゼロ昇格。
            let output = try activeChunks[ci].prediction(
                from: MLDictionaryFeatureProvider(dictionary: dictionary),
                using: states[ci]
            )
            guard let h = output.featureValue(for: "hidden")?.multiArrayValue else {
                throw LLMEngineError.generationFailed(reason: "v2 chunk \(ci) が hidden を返さなかった")
            }
            hidden = h
            // MTP: 最終チャンクは store 層(46/47)の更新後 KV を通常出力する。最新をゼロコピーで保持し、
            // 次ラウンドのドラフター(sks/svs/skf/svf)へ渡す。棄却位置は次ラウンドの mask で不可視。
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
            throw LLMEngineError.generationFailed(reason: "v2 lm_head が token を返さなかった")
        }
        return token[0].intValue
    }

    /// (S, H) hidden を lm_head に通し、S 行それぞれの greedy トークンを返す(verify 用)。
    private func lmheadArgmaxRows(_ hidden: MLMultiArray, count s: Int) throws -> [Int] {
        let output = try head.prediction(from: MLDictionaryFeatureProvider(dictionary: ["hidden": hidden]))
        guard let token = output.featureValue(for: "token")?.multiArrayValue else {
            throw LLMEngineError.generationFailed(reason: "v2 lm_head が token を返さなかった")
        }
        return (0..<s).map { token[$0].intValue }
    }

    /// テスト/診断専用: `basePosition` から S=`tokens.count` のバッチ verify を 1 回流し、各行の greedy
    /// argmax を返す。`position` / `lastToken` は変更しない(store 層 KV は mtpLoaded 時のみ副次更新される
    /// が、本ゲートは installMTP しない前提で使う)。リングの S>1 バッチ verify(窓跨ぎの tail-loss)を
    /// S=1 逐次と突き合わせるための純関数的フック。
    func batchedVerifyArgmaxes(tokens: [Int], basePosition: Int) throws -> [Int] {
        let hidden = try forwardChunks(tokens: tokens, basePosition: basePosition)
        return try lmheadArgmaxRows(hidden, count: tokens.count)
    }

    // MARK: - MTP(投機的デコード)

    /// MTP 1 ラウンド: ドラフター(または PLD)が先読み → 本体 v2 グラフ(S=drafts.count)で一括 verify → 採択。
    ///
    /// A1(PLD ハイブリッド): env `CORELLM_MTP_PLD` 有効時、現在列 `context + [prediction]` の末尾
    /// n-gram 一致から draft をコピーできればドラフター呼び出しゼロで先読みする(引用・要約系で高採択)。
    /// 一致しなければ従来ドラフターへ完全フォールバック。ロスレス性は verify(本体 argmax 照合)が
    /// 保証するため、draft の出所を変えても品質は不変(採択率が変わるだけ)。
    ///
    /// `prediction` は直前ステップで得た本体の確定トークン(位置 `position` の正解)。
    /// P0-4(ドラフト起点を確定トークンに揃える): 最初のドラフター呼び出しは (embed(lastToken), hidden)
    /// で hidden_next を得るが、`drafts[0] := prediction` に固定してチェーンの次入力 embed も prediction を使う。
    /// これで「drafts[0] != prediction」の死にラウンドが構造的に消える(verify 判定は不変なのでロスレス)。
    ///
    /// 棄却位置の state 内 KV は「後続の onehot 書き込みで上書き」で v1 と同じ理屈で正しい
    /// (onehot が書き込み位置を、mask が未確定位置の不可視化を担う)。
    func mtpRound(prediction: Int, context: [Int] = []) throws -> MTPRound {
        // ladder: チャンクは store 出力を持たないため、ドラフター判定の前に最終チャンク state から
        // store 層 KV を所有バッファへコピーして storeSKS/SVS/SKF/SVF を最新化する(非 ladder は
        // runChunks の出力 capture が担う)。直前ラウンドの forwardChunks が書いた KV(pos < position)を反映。
        if config.isLadder, mtpLoaded, drafter != nil, ladderStoreStateNames != nil {
            try captureLadderStoreKV()
        }
        // MTP を回せる最低条件。PLD は drafter 不要(ring では同梱 drafter が非対応でも PLD 単独で回る)。
        // drafter フォールバック(および tree)は drafter + store 層 KV が揃うときだけ有効。
        // 位置上限は effectiveContextLength(非 ring は == CTX なのでバイト等価、ring は 32768/131072)。
        let hasDrafter = mtpLoaded && drafter != nil && drafterHost != nil
            && draftEmbed != nil && draftHidden != nil && lastHiddenBuffer != nil
            && storeSKS != nil && storeSVS != nil && storeSKF != nil && storeSVF != nil
        guard pldEnabled || hasDrafter, position + draftLen < config.effectiveContextLength else {
            let next = try decodeStep(tokenID: prediction)
            return MTPRound(emitted: [prediction], next: next, accepted: 0, drafted: 0)
        }

        // 1) draft 生成。PLD 有効(env `CORELLM_MTP_PLD`)かつ現在列 `context + [prediction]` の末尾
        //    n-gram 一致が見つかれば、その直後のトークン列をコピーして draft にする(ドラフター呼び出しゼロ)。
        //    一致しなければ従来ドラフターで先読み(完全フォールバック)。どちらも drafts[0] は必ず
        //    prediction(P0-4)。maxDraft は CTX 端で position が CTX へ到達しないよう clamp
        //    (guard 下では常に = draftLen。ドラフター経路と同じ「round 後 position < CTX」不変条件)。
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
            // ladder regime 切替: 非昇格中(position-1 < 32768)は w32768 ドラフター + 32768 幅ホストを使う
            // (skf/svf は captureLadderStoreKV が既に小バッファ [1,32768,512] へ向けている)。昇格後 / drafterPre
            // 不在 / 非 ladder は従来の w131072(drafter, drafterHost)= 後方互換。tree は usesSplitOnehot ガードで不発。
            let usePre = config.isLadder && !promoted && drafterPre != nil && drafterHostPre != nil
            let activeDrafter = usePre ? (drafterPre ?? drafter) : drafter
            let activeHost = usePre ? (drafterHostPre ?? drafterHost) : drafterHost
            if usePre { drafterPreRounds += 1 } else { drafterFullRounds += 1 }
            // 直前確定位置(position-1)固定で先読み。ドラフターは store 層 KV へクロスアテンション。
            activeHost.update(position: position - 1)
            copyFlat(from: lastHiddenBuffer, to: draftHidden, count: config.H)
            // 呼び出し 0: lastToken で hidden を進める(token 出力は捨て、drafts[0] := prediction / P0-4)。
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
            // PLD ミス かつ drafter フォールバック不可(ring の PLD 単独経路)→ S=1 decode に落とす。
            if pldEnabled { drafterFallbackRounds += 1 }
            let next = try decodeStep(tokenID: prediction)
            return MTPRound(emitted: [prediction], next: next, accepted: 0, drafted: 0)
        }

        let base = position
        let drafted = drafts.count

        // 1.5) ツリー MTP(A2b): PLD ヒットラウンドのみ、主鎖(PLD 鎖)に「先頭 draft 位置(base+1)の
        //     代替候補 1 ノード」を scratch スロットで足す。代替候補 = ドラフターの先頭予測(warm-up + 1 step。
        //     主鎖 KV には触れない)。ドラフターは argmax token しか出さない(top-2 不可 = 設計 (a) 不成立)ため、
        //     設計 (b): PLD 鎖 = 主鎖 / drafter 先頭 = 代替。代替が主鎖先頭 draft と同じ・scratch が CTX 端超え・
        //     treeActive=false のときは足さない(= 既存経路)。
        var altToken: Int?
        var altSlot = 0
        // ツリー MTP(代替ノード)は 2048 単関数専用。ring/ladder(`usesSplitOnehot`)では TreeHostInputs /
        // forwardChunksTree が単一 onehot 前提(分離 onehot_s/onehot_f を作らない)のため不正入力になる → ガード。
        if treeActive, !config.usesSplitOnehot, usedPLD, drafts.count >= 2, hasDrafter,
           let drafter, let drafterHost, let draftEmbed, let draftHidden, let lastHiddenBuffer,
           let sks = storeSKS, let svs = storeSVS, let skf = storeSKF, let svf = storeSVF {
            let slot = base + drafts.count               // 主鎖使用域 base..base+drafts.count-1 の直後 = scratch。
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

        // PLD 統計を積む共通処理(usedPLD 時のみ。pldAccepted は「PLD draft の採択数」= 主鎖採択分 accepted-1)。
        func recordPLD(mainAccepted: Int) {
            guard usedPLD else { return }
            pldRounds += 1
            pldDraftedTokens += pldCopied.count
            pldAcceptedTokens += (mainAccepted - 1)
        }

        if let altToken {
            // 2') ツリー verify: S=drafts.count+1(主鎖 + 代替ノード)を 1 回。末尾行 = 代替ノードの argmax。
            treeAltFired += 1
            let treeHidden = try forwardChunksTree(
                mainTokens: drafts, altToken: altToken, basePosition: base, altSlot: altSlot)
            let targets = try lmheadArgmaxRows(treeHidden, count: drafts.count + 1)

            // 3') 採択: 主鎖を従来どおり逐次照合(代替ノードは主鎖の判定に一切影響しない = 別スロット・別マスク)。
            var accepted = 1
            while accepted < drafts.count && drafts[accepted] == targets[accepted - 1] { accepted += 1 }

            if accepted == 1 && altToken == targets[0] {
                // +1 回収: 主鎖先頭 draft は棄却(drafts[1] != targets[0])、かつ代替候補 = 本体の正解トークン。
                // 採択トークン(altToken)の KV は scratch にあるため、正規スロット base+1 へ S=1 で書き直す
                // (= MTP-OFF の decodeStep(altToken) と同一計算 → next もロスレス。スパイクの batch 再書き込み)。
                treeAltRecovered += 1
                let altHidden = try forwardChunks(tokens: [altToken], basePosition: base + 1)
                let next = try lmheadArgmax(altHidden)
                position = base + 2
                lastToken = altToken
                if let lastHiddenBuffer { copyRow(from: altHidden, row: 0, into: lastHiddenBuffer) }
                recordPLD(mainAccepted: accepted)     // PLD draft は不採択(accepted-1 = 0。回収は drafter 由来)。
                if adaptiveDraft { adaptDraftLength(accepted: accepted, drafted: drafted) }
                return MTPRound(emitted: [prediction, altToken], next: next, accepted: 2, drafted: drafted)
            }

            // 回収せず(主鎖先頭が採択された / 代替候補も外れた): 主鎖を通常採択。代替ノードの scratch KV は
            // 後続 decode の前進上書き + causal 隠蔽で自己修復する(明示クリア不要)。
            position = base + accepted
            lastToken = drafts[accepted - 1]
            if let lastHiddenBuffer { copyRow(from: treeHidden, row: accepted - 1, into: lastHiddenBuffer) }
            recordPLD(mainAccepted: accepted)
            if adaptiveDraft { adaptDraftLength(accepted: accepted, drafted: drafted) }
            return MTPRound(
                emitted: Array(drafts[0..<accepted]), next: targets[accepted - 1],
                accepted: accepted, drafted: drafted)
        }

        // 2) verify(標準 / ツリー無し / PLD ミス / 代替候補なし): 本体 v2 グラフを S=drafts.count で 1 回。
        //    forwardChunks が state KV(棄却位置含む)を書き、store 層 KV も最新に更新する。既存挙動と完全一致。
        let verifyHidden = try forwardChunks(tokens: drafts, basePosition: base)  // (drafts.count, H)
        let targets = try lmheadArgmaxRows(verifyHidden, count: drafts.count)

        // 3) 採択: drafts[0](= prediction)は必ず正しい。以降は本体 argmax と一致する限り伸ばす。
        var accepted = 1
        while accepted < drafts.count && drafts[accepted] == targets[accepted - 1] {
            accepted += 1
        }

        // 4) 確定: 採択分だけ位置を進め、lastToken / lastHidden を更新(store 層 KV は verify で更新済み)。
        position = base + accepted
        lastToken = drafts[accepted - 1]
        if let lastHiddenBuffer { copyRow(from: verifyHidden, row: accepted - 1, into: lastHiddenBuffer) }
        recordPLD(mainAccepted: accepted)
        if adaptiveDraft { adaptDraftLength(accepted: accepted, drafted: drafted) }
        return MTPRound(
            emitted: Array(drafts[0..<accepted]), next: targets[accepted - 1],
            accepted: accepted, drafted: drafted)
    }

    /// ドラフター 1 ステップ: embed(token) と現在の hidden から次トークンを得て、hidden をチェーン更新する。
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
            throw LLMEngineError.generationFailed(reason: "drafter の出力が不正")
        }
        copyFlat(from: h, to: hidden, count: config.H)
        return t[0].intValue
    }

    /// adaptive draft length(実験オプション): 直近 4 ラウンドの平均採択率 >0.9 で +2 / <0.5 で -2
    /// (4..maxDraftLen で clamp)。env `CORELLM_MTP_ADAPTIVE` 有効時のみ呼ばれる。
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

    /// (S, H) の行 `row` を (1, H) / (H,) バッファへコピーする。
    private func copyRow(from source: MLMultiArray, row: Int, into destination: MLMultiArray) {
        source.withF16 { src in
            destination.withF16 { dst in
                dst.baseAddress!.update(from: src.baseAddress! + row * config.H, count: config.H)
            }
        }
    }

    /// 先頭 `count` 要素を平坦コピーする(ランク不定の hidden 更新用)。
    private func copyFlat(from source: MLMultiArray, to destination: MLMultiArray, count: Int) {
        source.withF16 { src in
            destination.withF16 { dst in
                dst.baseAddress!.update(from: src.baseAddress!, count: count)
            }
        }
    }
}

// MARK: - KV state のディスク永続化(A3 スパイク: 文書 KV キャッシュ)
//
// v2 の KV は各チャンクの MLState(デバイス常駐)に載る。これをディスクへ保存し、別プロセスで
// fresh ロードした同型チェーンへ復元することで「cold prefill を初回だけ」にする(128K 計画 A3)。
// 往復は **raw バックングストアのバイト列**で行う: 保存元と復元先が同一バンドル(= 同一 state
// バッファレイアウト)である限り、連続性やメモリ順の解釈に依存せずバイト等価で戻る。この不変条件を
// `StateBundleIdentity` で復元時に検証し、食い違えば state を一切触らずにエラーを投げる。

/// 保存元バンドルと復元先バンドルが構造的に一致するかを検証する同一性キー。
/// convert_config の主要形状 + チャンク/lmhead 名 + バンドル名 + state 総数で構成する。
/// 1 つでも食い違えば `importStates` は state ファイルを読む前に明確なエラーを投げる。
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

/// 1 state バッファのメタデータ。`byteCount` は withUnsafeBytes が返す raw バックングストアの
/// バイト数(往復はこのバイト列で行うため、`contiguous` は情報用の記録にすぎない)。
struct StateEntry: Codable, Sendable {
    var name: String
    var chunkIndex: Int
    var shape: [Int]
    var dataType: Int      // MLMultiArrayDataType.rawValue
    var byteCount: Int
    var contiguous: Bool
    var file: String
}

/// export ディレクトリの manifest(この JSON + state 毎の raw バイナリ群)。
struct StateManifest: Codable, Sendable {
    var formatVersion: Int
    var identity: StateBundleIdentity
    var position: Int
    /// prefill 直後の argmax(= 保存点からの継続シード)。prefill なしに継続を再開するために持つ。
    var pendingNextToken: Int?
    /// ホスト側の処理済みトークン列(エンジンの LCP prefix 継続 / lastToken 復元用)。
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
            return "KV 復元先バンドルが保存元と一致しません(expected=\(expected), found=\(found))"
        case .missingStateFile(let name):
            return "state ファイルが見つかりません: \(name)"
        case .sizeMismatch(let name, let expected, let found):
            return "state \(name) のバイト数不一致(復元先=\(expected), ファイル=\(found))"
        case .unknownState(let name):
            return "manifest に無い state 名: \(name)"
        }
    }
}

extension CoreMLChainV2 {
    /// チャンク `model` が所有する state 名(決定的順序 = 辞書順。export/import で同一順序を使う)。
    private func stateNames(of model: MLModel) -> [String] {
        model.modelDescription.stateDescriptionsByName.keys.sorted()
    }

    /// 現在のチェーン構成の同一性キー(保存/復元の互換判定に使う)。
    func stateExportIdentity() -> StateBundleIdentity {
        let total = chunks.reduce(0) { $0 + stateNames(of: $1).count }
        return StateBundleIdentity(
            bundleName: bundleURL.lastPathComponent,
            ctx: config.CTX, nLayers: config.NLAYERS, h: config.H, hd: config.HD,
            ghd: config.GHD, sliding: config.SLIDING,
            chunks: config.chunks, lmhead: config.lmhead, stateCount: total)
    }

    /// 期待 C 連続ストライドか(情報記録用。往復自体は raw バイトなので連続性に依存しない)。
    private func isCContiguous(shape: [Int], strides: [Int]) -> Bool {
        guard shape.count == strides.count else { return false }
        var expected = 1
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[i] != expected { return false }
            expected *= shape[i]
        }
        return true
    }

    /// 全チャンクの KV state を `url` ディレクトリへ保存する(manifest.json + state 毎 raw バイナリ)。
    /// `pendingNextToken`(prefill 直後の argmax = 継続シード)と `processedTokens`(ホスト処理済み列)も
    /// manifest に載せる。ディレクトリは作り直す(既存があれば削除)。
    @discardableResult
    func exportStates(to url: URL, pendingNextToken: Int?, processedTokens: [Int]) throws -> StateManifest {
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path()) { try fm.removeItem(at: url) }
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

    /// `url` の KV state を現在のチェーンへ復元する。まずバンドル同一性を検証し(不一致なら state を
    /// 一切触らずエラー)、次に各 state バッファへ raw バイトを書き戻し、最後に position とホスト補助状態を
    /// manifest に合わせる。復元後は prefill なしで `pendingNextToken` から継続できる。manifest を返す。
    @discardableResult
    func importStates(from url: URL) throws -> StateManifest {
        let fm = FileManager.default
        let manifest = try JSONDecoder().decode(
            StateManifest.self, from: try Data(contentsOf: url.appending(path: "manifest.json")))

        // 1) 同一性ガード(state ファイルを一切読む前に検証する)。
        let current = stateExportIdentity()
        guard manifest.identity == current else {
            throw StatePersistenceError.identityMismatch(expected: current, found: manifest.identity)
        }

        // 2) state バッファへ raw バイトを書き戻す(export と同一のチャンク×名前順)。
        let byName = Dictionary(manifest.states.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        for ci in chunks.indices {
            for name in stateNames(of: chunks[ci]) {
                guard let entry = byName[name] else { throw StatePersistenceError.unknownState(name) }
                let fileURL = url.appending(path: entry.file)
                guard fm.fileExists(atPath: fileURL.path()) else {
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

        // 3) 位置とホスト補助状態を復元。MTP の store 層 KV ポインタは無効化する
        //    (復元後の初回 mtpRound は decodeStep へフォールバックして最新を捕捉し直す = 自己修復)。
        position = manifest.position
        lastToken = manifest.processedTokens.last ?? 0
        storeSKS = nil; storeSVS = nil; storeSKF = nil; storeSVF = nil
        ladderStoreSyncedRows = 0   // 復元後は full 層バッファを全行再同期(初回 mtpRound が [0,position) を捕捉)。
        return manifest
    }
}
