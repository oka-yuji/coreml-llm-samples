public protocol Tokenizing: Sendable {
    func encode(_ text: String) throws -> [Int]
    func decode(_ ids: [Int]) throws -> String
    var eosTokenIDs: Set<Int> { get }
}
