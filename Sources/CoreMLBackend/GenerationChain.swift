import Foundation

protocol GenerationChain: AnyObject {

    var contextLength: Int { get }

    var position: Int { get }

    func reset() throws

    func rewind(to position: Int)

    func prefill(_ promptIDs: [Int]) throws -> Int

    func decodeStep(tokenID: Int) throws -> Int
}

protocol MultimodalChain: GenerationChain {

    var hiddenSize: Int { get }

    func prefillSegments(_ segments: [PromptSegment]) throws -> Int

    func continueGreedy(
        seed: Int, maxNew: Int, eos: Set<Int>, onToken: ((Int) -> Void)?
    ) throws -> [Int]

    func plannedPrefillWidths(promptLength: Int, from startPosition: Int) -> [Int]

    func materializePrefill(widths: [Int]) throws

    func residentPrefillWidths() -> [Int]
}

extension MultimodalChain {
    func continueGreedy(
        seed: Int, maxNew: Int, eos: Set<Int>, onToken: ((Int) -> Void)? = nil
    ) throws -> [Int] {
        var out: [Int] = []
        var next = seed
        while out.count < maxNew {
            try Task.checkCancellation()
            if eos.contains(next) { break }
            out.append(next)
            onToken?(next)
            if out.count >= maxNew { break }
            next = try decodeStep(tokenID: next)
        }
        return out
    }
}

extension ChunkedSpeculativeChain: MultimodalChain {}

extension CoreMLChainV2: MultimodalChain {}

struct MTPRound {

    let emitted: [Int]

    let next: Int

    let accepted: Int

    let drafted: Int
}

protocol SpeculativeDecoding: AnyObject {

    var supportsMTP: Bool { get }

    var mtpLoaded: Bool { get }

    func mtpRound(prediction: Int, context: [Int]) throws -> MTPRound
}

extension CoreMLChain: SpeculativeDecoding {}
extension CoreMLChainV2: SpeculativeDecoding {}

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

extension CoreMLChainV2: GenerationChain {

    var contextLength: Int { config.effectiveContextLength }

    func prefill(_ promptIDs: [Int]) throws -> Int {
        try prefill(promptIDs, blockSize: config.maxS)
    }
}
