public protocol LLMEngine: Sendable {

    var descriptor: EngineDescriptor { get }

    func load(_ model: ModelBundle, options: LoadOptions) async throws

    func unload() async

    func generate(_ request: GenerationRequest) -> AsyncThrowingStream<GenerationEvent, any Error>

    func resetConversation() async
}

public extension LLMEngine {

    func resetConversation() async {}
}

public struct EngineDescriptor: Sendable, Hashable, Codable {
    public enum Backend: String, Sendable, Codable {
        case echo
        case coreML
        case coreAI

        case llamaCpp
    }

    public var name: String
    public var backend: Backend

    public var supportsMultiTokenPrediction: Bool

    public init(name: String, backend: Backend, supportsMultiTokenPrediction: Bool) {
        self.name = name
        self.backend = backend
        self.supportsMultiTokenPrediction = supportsMultiTokenPrediction
    }
}

public enum LLMEngineError: Error, Sendable, Equatable {
    case notLoaded
    case modelNotFound(path: String)
    case incompatibleBundle(reason: String)
    case drafterUnavailable
    case generationFailed(reason: String)
    case contextOverflow(promptTokens: Int, contextLength: Int)
}
