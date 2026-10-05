import Foundation

public struct VLMGenerationInfo: Sendable {
    public var text: String
    public var generatedTokens: Int
    public var promptTokens: Int
    public var imageRows: Int
    public var visionEncodeSeconds: Double
    public var prefillSeconds: Double
    public var decodeSeconds: Double
    public var decodeTokensPerSecond: Double
    public var peakMemoryBytes: Int?

    public init(
        text: String, generatedTokens: Int, promptTokens: Int, imageRows: Int,
        visionEncodeSeconds: Double, prefillSeconds: Double, decodeSeconds: Double,
        decodeTokensPerSecond: Double, peakMemoryBytes: Int? = nil
    ) {
        self.text = text
        self.generatedTokens = generatedTokens
        self.promptTokens = promptTokens
        self.imageRows = imageRows
        self.visionEncodeSeconds = visionEncodeSeconds
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.peakMemoryBytes = peakMemoryBytes
    }
}
