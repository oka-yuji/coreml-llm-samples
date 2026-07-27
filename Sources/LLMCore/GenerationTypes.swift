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
        maxNewTokens: Int = 0,
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

public enum FinishReason: String, Sendable, Codable {
    case eos
    case cap
    case contextFull
    case cancelled
    case error
}

public struct GenerationMetrics: Sendable, Codable {
    public var promptTokens: Int
    public var reusedTokens: Int
    public var generatedTokens: Int
    public var timeToFirstToken: Duration
    public var decodeTokensPerSecond: Double
    public var peakMemoryBytes: Int?

    public var draftAcceptanceRate: Double?

    public var finishReason: FinishReason?
    public var prefillSeconds: Double?
    public var perTokenMillis: [Double]?
    public var footprintAfterPrefillBytes: Int?
    public var footprintAtEndBytes: Int?
    public var availableMemoryBytes: Int?
    public var thermalStateStart: String?
    public var thermalStateEnd: String?
    public var specEnabled: Bool?
    public var specRounds: Int?
    public var specDrafted: Int?
    public var specAccepted: Int?
    public var specFallbackRounds: Int?
    public var feedWidths: [Int]?

    public init(
        promptTokens: Int,
        generatedTokens: Int,
        timeToFirstToken: Duration,
        decodeTokensPerSecond: Double,
        peakMemoryBytes: Int? = nil,
        draftAcceptanceRate: Double? = nil,
        reusedTokens: Int = 0,
        finishReason: FinishReason? = nil,
        prefillSeconds: Double? = nil,
        perTokenMillis: [Double]? = nil,
        footprintAfterPrefillBytes: Int? = nil,
        footprintAtEndBytes: Int? = nil,
        availableMemoryBytes: Int? = nil,
        thermalStateStart: String? = nil,
        thermalStateEnd: String? = nil,
        specEnabled: Bool? = nil,
        specRounds: Int? = nil,
        specDrafted: Int? = nil,
        specAccepted: Int? = nil,
        specFallbackRounds: Int? = nil,
        feedWidths: [Int]? = nil
    ) {
        self.promptTokens = promptTokens
        self.reusedTokens = reusedTokens
        self.generatedTokens = generatedTokens
        self.timeToFirstToken = timeToFirstToken
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.peakMemoryBytes = peakMemoryBytes
        self.draftAcceptanceRate = draftAcceptanceRate
        self.finishReason = finishReason
        self.prefillSeconds = prefillSeconds
        self.perTokenMillis = perTokenMillis
        self.footprintAfterPrefillBytes = footprintAfterPrefillBytes
        self.footprintAtEndBytes = footprintAtEndBytes
        self.availableMemoryBytes = availableMemoryBytes
        self.thermalStateStart = thermalStateStart
        self.thermalStateEnd = thermalStateEnd
        self.specEnabled = specEnabled
        self.specRounds = specRounds
        self.specDrafted = specDrafted
        self.specAccepted = specAccepted
        self.specFallbackRounds = specFallbackRounds
        self.feedWidths = feedWidths
    }
}
