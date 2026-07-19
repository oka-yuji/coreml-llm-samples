/// トークナイザの抽象。実装(swift-transformers ラッパ等)はバックエンド側に置き、
/// LLMCore は語彙形式に依存しない。
public protocol Tokenizing: Sendable {
    func encode(_ text: String) throws -> [Int]
    func decode(_ ids: [Int]) throws -> String
    var eosTokenIDs: Set<Int> { get }
}
