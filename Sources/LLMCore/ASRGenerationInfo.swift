import Foundation

public struct ASRGenerationInfo: Sendable {
    public var text: String
    public var generatedTokens: Int
    public var promptTokens: Int
    public var audioRows: Int
    public var audioSeconds: Double
    public var audioEncodeSeconds: Double
    public var prefillSeconds: Double
    public var decodeSeconds: Double
    public var decodeTokensPerSecond: Double
    public var peakMemoryBytes: Int?

    public init(
        text: String, generatedTokens: Int, promptTokens: Int, audioRows: Int,
        audioSeconds: Double, audioEncodeSeconds: Double, prefillSeconds: Double,
        decodeSeconds: Double, decodeTokensPerSecond: Double, peakMemoryBytes: Int? = nil
    ) {
        self.text = text
        self.generatedTokens = generatedTokens
        self.promptTokens = promptTokens
        self.audioRows = audioRows
        self.audioSeconds = audioSeconds
        self.audioEncodeSeconds = audioEncodeSeconds
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.peakMemoryBytes = peakMemoryBytes
    }
}
