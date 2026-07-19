import Foundation

/// エンジンの生成ループが必要とする最小インターフェース。
///
/// ステートレスな `CoreMLChain` と v2 stateful な `CoreMLChainV2` の両方が適合する。
/// MTP(投機的デコード)は `CoreMLChain` 固有なのでここには含めない — エンジンは MTP 時のみ
/// 具象 `CoreMLChain` を直接使い、それ以外の prefill / decode はこの protocol 経由で回す。
protocol GenerationChain: AnyObject {
    /// 焼き込み済みの最大コンテキスト長(トークン)。
    var contextLength: Int { get }

    /// 現在の書き込み位置(= KV に処理済みのトークン数)。
    var position: Int { get }

    /// 生成状態を初期化する(KV バッファ / MLState と位置を巻き戻す)。
    func reset() throws

    /// KV は保持したまま位置だけを `position` へ巻き戻す(マルチターン継続 / prefix cache 用)。
    /// 巻き戻し位置以降の KV は後続の prefill / decode が onehot 書き込みで上書きし、
    /// 未確定位置は mask が不可視化するため、状態は構成的に正しく保たれる(MTP 棄却と同一原理)。
    func rewind(to position: Int)

    /// プロンプト全トークンを処理し、最初の生成トークン(greedy argmax)を返す。
    /// **現在位置から**処理を続ける(reset 直後は 0、rewind 後は巻き戻し位置)。位置はその分進む。
    func prefill(_ promptIDs: [Int]) throws -> Int

    /// トークン 1 つを処理し、次トークン(greedy argmax)を返す。位置は 1 進む。
    func decodeStep(tokenID: Int) throws -> Int
}

// MARK: - MTP(投機的デコード)

/// MTP 1 ラウンドの結果。ロスレスに逐次デコードと一致する。
struct MTPRound {
    /// 確定して emit すべきトークン列(先頭は必ず `prediction`)。
    let emitted: [Int]
    /// 次の未処理予測トークン(次ラウンドの `prediction`)。
    let next: Int
    /// このラウンドで採択したトークン数(採択率の分子)。
    let accepted: Int
    /// このラウンドでドラフトしたトークン数(採択率の分母)。
    let drafted: Int
}

/// ドラフター投機的デコード(MTP)を提供するチェーン。
///
/// v1(`CoreMLChain`・ステートレス、専用 S=6 verify チェーン)と
/// v2(`CoreMLChainV2`・stateful、verify は本体グラフの S=draftLen 呼び出し)の両方が適合する。
/// エンジンは MTP のホットパス(`mtpRound`)をこの protocol 経由で回し、
/// 資産のロード(ドラフター等)だけ具象型で分岐する。
protocol SpeculativeDecoding: AnyObject {
    /// ドラフター資産がバンドルに存在するか(実ロードは遅延)。
    var supportsMTP: Bool { get }
    /// ドラフター資産がロード済みか。
    var mtpLoaded: Bool { get }
    /// MTP 1 ラウンド: ドラフター(または PLD)が先読み → 本体が一括 verify → 採択。
    ///
    /// `context` は確定済みトークン列(位置 `[0, position)`)= prompt + これまでに emit した全トークン。
    /// v2 の PLD(prompt-lookup、env `CORELLM_MTP_PLD`)が `context + [prediction]` の末尾 n-gram 一致
    /// からドラフトをコピーするために使う。v1 と PLD 無効時は未使用(挙動不変)。
    func mtpRound(prediction: Int, context: [Int]) throws -> MTPRound
}

extension CoreMLChain: SpeculativeDecoding {}
extension CoreMLChainV2: SpeculativeDecoding {}

// MARK: - ステートレス版(既存経路。step() の薄いラッパで振る舞いは不変)

extension CoreMLChain: GenerationChain {
    var contextLength: Int { config.CTX }

    func prefill(_ promptIDs: [Int]) throws -> Int {
        var next = 0
        for id in promptIDs {
            try Task.checkCancellation()
            next = try step(tokenID: id)
        }
        return next
    }

    func decodeStep(tokenID: Int) throws -> Int {
        try step(tokenID: tokenID)
    }
}

// MARK: - v2 stateful 版(既定 prefill は S=128 ブロック)

extension CoreMLChainV2: GenerationChain {
    /// ctx32k リングは full 層の絶対長(32768)が実効コンテキスト長。既定 2048 は CTX で不変。
    var contextLength: Int { config.effectiveContextLength }

    /// 実運用の prefill は S=128 ブロック(+ 端数)。テストは `prefill(_:blockSize:)` を直接呼ぶ。
    func prefill(_ promptIDs: [Int]) throws -> Int {
        try prefill(promptIDs, blockSize: config.maxS)
    }
}
