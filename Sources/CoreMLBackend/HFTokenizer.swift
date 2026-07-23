import Foundation
import LLMCore
import Tokenizers

struct HFTokenizer: Tokenizing {
    private let tokenizer: any Tokenizer
    let eosTokenIDs: Set<Int>
    let bosTokenID: Int?

    init(modelFolder: URL, eosTokenIDs: Set<Int>) async throws {
        tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
        self.eosTokenIDs = eosTokenIDs
        bosTokenID = tokenizer.bosTokenId
    }

    func encode(_ text: String) throws -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    func decode(_ ids: [Int]) throws -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: true)
    }
}
