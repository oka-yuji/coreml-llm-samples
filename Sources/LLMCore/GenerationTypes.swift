public struct GenerationRequest: Sendable {
    public var prompt: String
    public var config: GenerationConfig

    public var history: [ChatTurn]

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

    public var maxNewTokens: Int

    public var temperature: Double
    public var topP: Double?
    public var seed: UInt64?

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

    case loadCompleted(LoadMetrics)
    case prefillCompleted(PrefillMetrics)
    case token(TokenChunk)
    case finished(GenerationMetrics)
}

public struct TokenChunk: Sendable {
    public var text: String
    public var tokenID: Int

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
