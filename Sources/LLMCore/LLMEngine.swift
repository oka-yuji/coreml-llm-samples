/// LLM 推論バックエンドの共通インターフェース。
///
/// Core ML / Core AI などの実装は actor として提供し、モデルのロード状態と
/// KV キャッシュを actor 隔離で保護する。出力はすべて `AsyncThrowingStream` 経由。
public protocol LLMEngine: Sendable {
    /// エンジンの静的な自己紹介(名前・バックエンド種別・対応機能)。
    var descriptor: EngineDescriptor { get }

    /// モデルバンドルをロードする。多重ロードは実装側で無視してよい。
    func load(_ model: ModelBundle, options: LoadOptions) async throws

    /// ロード済みモデルを解放する。
    func unload() async

    /// トークンストリームを返す。キャンセルは Task cancellation で伝搬する。
    /// 未ロード時は `LLMEngineError.notLoaded` をストリームに流す。
    func generate(_ request: GenerationRequest) -> AsyncThrowingStream<GenerationEvent, any Error>

    /// 会話を初期化する(KV prefix cache / 処理済みトークンを破棄)。
    /// 新規チャット開始・モデル切替時に呼ぶ。KV 継続を持たないエンジンでは no-op。
    func resetConversation() async
}

public extension LLMEngine {
    /// 既定では何もしない(KV 継続を実装しないエンジン向け)。
    func resetConversation() async {}
}

/// エンジンの種別と対応機能。UI 表示とベンチ結果のラベリングに使う。
public struct EngineDescriptor: Sendable, Hashable, Codable {
    public enum Backend: String, Sendable, Codable {
        case echo
        case coreML
        case coreAI
        /// llama.cpp(GGUF)ランタイム。実装は `Packages/LlamaCppKit` の `LlamaEngine`。
        case llamaCpp
    }

    public var name: String
    public var backend: Backend
    /// MTP(Multi-Token Prediction / ドラフター投機的デコード)に対応しているか。
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
}
