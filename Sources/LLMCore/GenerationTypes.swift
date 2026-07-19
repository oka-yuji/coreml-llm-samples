public struct GenerationRequest: Sendable {
    public var prompt: String
    public var config: GenerationConfig
    /// マルチターン継続時の、これまでの会話履歴(古い順)。空なら単一ターン(従来挙動)。
    /// エンジンはこれを manifest の prefix/suffix で整形し、`prompt` を末尾の user ターンとして続ける。
    public var history: [ChatTurn]
    /// KV prefix cache を使うか。true なら前ターンまでの KV を保持したまま、会話全体のトークン列と
    /// 処理済みトークン列の最長共通接頭辞(LCP)以降の差分だけを prefill する。
    /// false(既定)は毎回リセットして会話全体を再 prefill(従来挙動・後方互換)。
    public var reuseCache: Bool

    public init(
        prompt: String,
        config: GenerationConfig = GenerationConfig(),
        history: [ChatTurn] = [],
        reuseCache: Bool = false
    ) {
        self.prompt = prompt
        self.config = config
        self.history = history
        self.reuseCache = reuseCache
    }
}

/// 会話の 1 ターン(発話者と本文)。マルチターン継続でエンジンに履歴を渡すのに使う。
public struct ChatTurn: Sendable, Equatable {
    public enum Role: Sendable, Equatable { case user, assistant }
    public var role: Role
    public var text: String

    public init(role: Role, text: String) {
        self.role = role
        self.text = text
    }
}

public struct GenerationConfig: Sendable, Hashable, Codable {
    /// 生成する最大トークン数。
    public var maxNewTokens: Int
    /// 0 で greedy(決定的)。バックエンド間比較のデフォルト。
    public var temperature: Double
    public var topP: Double?
    public var seed: UInt64?
    /// MTP(ドラフターによる投機的デコード)を使うか。
    /// ドラフターを持たないバンドル / 非対応エンジンでは無視される。
    public var multiTokenPrediction: Bool

    public init(
        maxNewTokens: Int = 512,
        temperature: Double = 0,
        topP: Double? = nil,
        seed: UInt64? = nil,
        multiTokenPrediction: Bool = false
    ) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topP = topP
        self.seed = seed
        self.multiTokenPrediction = multiTokenPrediction
    }
}

public enum GenerationEvent: Sendable {
    /// 遅延ロード/初回コンパイルが生成開始時に走った場合に流れる(任意)。
    case loadCompleted(LoadMetrics)
    case prefillCompleted(PrefillMetrics)
    case token(TokenChunk)
    case finished(GenerationMetrics)
}

public struct TokenChunk: Sendable {
    public var text: String
    public var tokenID: Int
    /// 生成開始からの累積 decode tok/s(UI のライブ表示用)。
    public var tokensPerSecond: Double

    public init(text: String, tokenID: Int, tokensPerSecond: Double) {
        self.text = text
        self.tokenID = tokenID
        self.tokensPerSecond = tokensPerSecond
    }
}

public struct LoadMetrics: Sendable, Codable {
    public var duration: Duration

    public init(duration: Duration) {
        self.duration = duration
    }
}

public struct PrefillMetrics: Sendable, Codable {
    public var promptTokens: Int
    /// KV prefix cache から再利用した接頭辞トークン数(全再 prefill では 0)。
    /// 新規に prefill したトークン数は `promptTokens - reusedTokens`。
    public var reusedTokens: Int
    public var duration: Duration

    public init(promptTokens: Int, reusedTokens: Int = 0, duration: Duration) {
        self.promptTokens = promptTokens
        self.reusedTokens = reusedTokens
        self.duration = duration
    }
}

public struct GenerationMetrics: Sendable, Codable {
    public var promptTokens: Int
    public var generatedTokens: Int
    public var timeToFirstToken: Duration
    public var decodeTokensPerSecond: Double
    public var peakMemoryBytes: Int?
    /// MTP 有効時のみ: ドラフトトークンの採択率(0...1)。
    public var draftAcceptanceRate: Double?

    public init(
        promptTokens: Int,
        generatedTokens: Int,
        timeToFirstToken: Duration,
        decodeTokensPerSecond: Double,
        peakMemoryBytes: Int? = nil,
        draftAcceptanceRate: Double? = nil
    ) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.timeToFirstToken = timeToFirstToken
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.peakMemoryBytes = peakMemoryBytes
        self.draftAcceptanceRate = draftAcceptanceRate
    }
}
