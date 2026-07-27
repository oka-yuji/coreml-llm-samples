import Foundation

public struct KVCheckpointInfo: Sendable {
    public var position: Int
    public var tokenCount: Int
    public var pendingNextToken: Int
    public var fileBytes: Int
    public var layerCount: Int
    public var prefillSeconds: Double?
    public var exportSeconds: Double?
    public var importSeconds: Double?
    public var residentPrefillWidths: [Int]
    public var continuation: [Int]?
    public var continuationText: String?
    public var peakMemoryBytes: Int?
    public var pldRounds: Int?
    public var pldFallbackRounds: Int?
    public var pldDraftedTokens: Int?
    public var pldAcceptedTokens: Int?

    public init(
        position: Int, tokenCount: Int, pendingNextToken: Int, fileBytes: Int, layerCount: Int,
        prefillSeconds: Double?, exportSeconds: Double?, importSeconds: Double?,
        residentPrefillWidths: [Int], continuation: [Int]?, continuationText: String?,
        peakMemoryBytes: Int? = nil,
        pldRounds: Int? = nil, pldFallbackRounds: Int? = nil,
        pldDraftedTokens: Int? = nil, pldAcceptedTokens: Int? = nil
    ) {
        self.position = position
        self.tokenCount = tokenCount
        self.pendingNextToken = pendingNextToken
        self.fileBytes = fileBytes
        self.layerCount = layerCount
        self.prefillSeconds = prefillSeconds
        self.exportSeconds = exportSeconds
        self.importSeconds = importSeconds
        self.residentPrefillWidths = residentPrefillWidths
        self.continuation = continuation
        self.continuationText = continuationText
        self.peakMemoryBytes = peakMemoryBytes
        self.pldRounds = pldRounds
        self.pldFallbackRounds = pldFallbackRounds
        self.pldDraftedTokens = pldDraftedTokens
        self.pldAcceptedTokens = pldAcceptedTokens
    }
}
