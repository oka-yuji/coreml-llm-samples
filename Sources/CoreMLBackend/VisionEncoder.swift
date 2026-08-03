import CoreGraphics
import CoreML
import Foundation
import LLMCore

actor VisionEncoder {
    private let packageURL: URL
    private let computeUnits: MLComputeUnits
    private var model: MLModel?

    init(packageURL: URL, computeUnits: MLComputeUnits) {
        self.packageURL = packageURL
        self.computeUnits = computeUnits
    }

    func loadIfNeeded() async throws {
        guard model == nil else { return }
        let compiled = try await CompiledModelStore.compiledModelURL(
            bundleURL: packageURL.deletingLastPathComponent(), name: packageURL.lastPathComponent)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        model = try MLModel(contentsOf: compiled, configuration: cfg)
    }

    func unload() { model = nil }

    func softTokenRowCount() -> Int? {
        guard let constraint = model?.modelDescription
            .outputDescriptionsByName["soft_tokens"]?.multiArrayConstraint,
              constraint.shape.count == 2 else { return nil }
        return constraint.shape[0].intValue
    }

    func encode(patches: MLMultiArray, releaseAfter: Bool = false) async throws -> SoftTokenRows {
        try await loadIfNeeded()
        guard let model else {
            throw LLMEngineError.generationFailed(reason: "VisionEncoder: model is not loaded")
        }
        defer { if releaseAfter { unload() } }
        let out = try Self.predict(model, patches: patches)
        guard let soft = out.featureValue(for: "soft_tokens")?.multiArrayValue else {
            throw LLMEngineError.generationFailed(reason: "VisionEncoder: no soft_tokens output")
        }
        guard soft.shape.count == 2 else {
            throw LLMEngineError.generationFailed(
                reason: "VisionEncoder: soft_tokens must be rank 2 (actual \(soft.shape))")
        }
        let rows = soft.shape[0].intValue
        let hidden = soft.shape[1].intValue
        var data = [Float16](repeating: 0, count: rows * hidden)
        Self.copyToF16(soft, into: &data)
        return SoftTokenRows(rows: rows, hidden: hidden, data: data)
    }

    func encode(imageAt url: URL, releaseAfter: Bool = true) async throws -> SoftTokenRows {
        let patches = try VisionPreprocess.patches(fromImageAt: url)
        return try await encode(patches: patches, releaseAfter: releaseAfter)
    }

    func encode(image: CGImage, releaseAfter: Bool = true) async throws -> SoftTokenRows {
        let patches = try VisionPreprocess.patches(from: image)
        return try await encode(patches: patches, releaseAfter: releaseAfter)
    }

    private static func predict(_ model: MLModel, patches: MLMultiArray) throws -> MLFeatureProvider {
        try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["patches": patches]))
    }

    private static func copyToF16(_ array: MLMultiArray, into dst: inout [Float16]) {
        let count = dst.count
        switch array.dataType {
        case .float16:
            array.withF16 { src in
                dst.withUnsafeMutableBufferPointer { d in
                    d.baseAddress!.update(from: src.baseAddress!, count: count)
                }
            }
        case .float32:
            array.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float32.self)
                for i in 0..<count { dst[i] = Float16(src[i]) }
            }
        default:
            for i in 0..<count { dst[i] = Float16(array[i].floatValue) }
        }
    }
}

enum VLMPrompt {
    static let turnStart = 105
    static let turnEnd = 106
    static let newline = 107
    static let boi = 255999
    static let eoi = 258882

    static func segments(
        bos: Int, userTokens: [Int], questionTokens: [Int], modelTokens: [Int], image: SoftTokenRows
    ) -> [PromptSegment] {
        let pre = [bos, turnStart] + userTokens + [boi]
        let post = [eoi] + questionTokens + [turnEnd, newline, turnStart] + modelTokens
        return [.tokens(pre), .image(image), .tokens(post)]
    }

    static func flatIDs(
        bos: Int, userTokens: [Int], questionTokens: [Int], modelTokens: [Int], imageRows: Int
    ) -> [Int] {
        [bos, turnStart] + userTokens + [boi]
            + Array(repeating: ChunkedSpeculativeChain.imagePlaceholderID, count: imageRows)
            + [eoi] + questionTokens + [turnEnd, newline, turnStart] + modelTokens
    }

    static func followUpTokens(userTokens: [Int], questionTokens: [Int], modelTokens: [Int]) -> [Int] {
        [turnEnd, newline, turnStart] + userTokens + questionTokens
            + [turnEnd, newline, turnStart] + modelTokens
    }
}
