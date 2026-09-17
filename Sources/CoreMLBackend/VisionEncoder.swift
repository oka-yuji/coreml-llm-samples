import CoreGraphics
import CoreML
import Foundation
import LLMCore

actor VisionEncoder {
    private let packageURL: URL
    private let computeUnits: MLComputeUnits
    private var model: MLModel?
    private var geometry: VisionGeometry?
    private var didWarmUp = false

    init(packageURL: URL, computeUnits: MLComputeUnits) {
        self.packageURL = packageURL
        self.computeUnits = computeUnits
    }

    var computeUnitsLabel: String { Self.label(computeUnits) }

    func loadIfNeeded() async throws {
        guard model == nil else { return }
        let compiled = try await CompiledModelStore.compiledModelURL(
            bundleURL: packageURL.deletingLastPathComponent(), name: packageURL.lastPathComponent)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let loaded = try MLModel(contentsOf: compiled, configuration: cfg)
        guard let shape = Self.patchGeometry(of: loaded) else {
            throw LLMEngineError.incompatibleBundle(
                reason: "\(packageURL.lastPathComponent) does not declare a square patch grid of at "
                    + "most \(VisionGeometry.maxSide)px on its patches input")
        }
        model = loaded
        geometry = shape
    }

    func inputSide() async throws -> Int {
        try await loadIfNeeded()
        return try requireGeometry().side
    }

    private func requireGeometry() throws -> VisionGeometry {
        guard let geometry else {
            throw LLMEngineError.generationFailed(reason: "VisionEncoder: model is not loaded")
        }
        return geometry
    }

    private static func patchGeometry(of model: MLModel) -> VisionGeometry? {
        guard let constraint = model.modelDescription
            .inputDescriptionsByName["patches"]?.multiArrayConstraint,
              constraint.shape.count == 3 else { return nil }
        return VisionGeometry.forPatchCount(constraint.shape[1].intValue)
    }

    func warmUpIfNeeded() async throws -> Double {
        try await loadIfNeeded()
        guard !didWarmUp, let model else { return 0 }
        let blank = try VisionPreprocess.blankPatches(geometry: try requireGeometry())
        let clock = ContinuousClock()
        let t0 = clock.now
        _ = try Self.predict(model, patches: blank)
        didWarmUp = true
        return (clock.now - t0) / .seconds(1)
    }

    func unload() {
        model = nil
        geometry = nil
        didWarmUp = false
    }

    private static func label(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        case .all: return "all"
        @unknown default: return "unknown"
        }
    }

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
        let image = try VisionPreprocess.loadCGImage(from: url)
        return try await encode(image: image, releaseAfter: releaseAfter)
    }

    func encode(image: CGImage, releaseAfter: Bool = true) async throws -> SoftTokenRows {
        try await loadIfNeeded()
        let patches = try VisionPreprocess.patches(from: image, geometry: try requireGeometry())
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
        bos: Int?, userTokens: [Int], questionTokens: [Int], modelTokens: [Int], image: SoftTokenRows
    ) -> [PromptSegment] {
        let pre = (bos.map { [$0] } ?? []) + [turnStart] + userTokens + [boi]
        let post = [eoi] + questionTokens + [turnEnd, newline, turnStart] + modelTokens
        return [.tokens(pre), .image(image), .tokens(post)]
    }

    static func flatIDs(
        bos: Int?, userTokens: [Int], questionTokens: [Int], modelTokens: [Int], imageRows: Int
    ) -> [Int] {
        (bos.map { [$0] } ?? []) + [turnStart] + userTokens + [boi]
            + Array(repeating: MultimodalSlot.imagePlaceholderID, count: imageRows)
            + [eoi] + questionTokens + [turnEnd, newline, turnStart] + modelTokens
    }

    static func followUpTokens(userTokens: [Int], questionTokens: [Int], modelTokens: [Int]) -> [Int] {
        [turnEnd, newline, turnStart] + userTokens + questionTokens
            + [turnEnd, newline, turnStart] + modelTokens
    }
}
