import CoreML
import Foundation
import LLMCore

final class AudioEncoder {
    static let computeUnits: MLComputeUnits = .cpuAndGPU

    private let packageURL: URL
    private var model: MLModel?

    init(packageURL: URL) {
        self.packageURL = packageURL
    }

    func loadIfNeeded() async throws {
        guard model == nil else { return }
        let compiled = try await CompiledModelStore.compiledModelURL(
            bundleURL: packageURL.deletingLastPathComponent(), name: packageURL.lastPathComponent)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = Self.computeUnits
        model = try MLModel(contentsOf: compiled, configuration: cfg)
    }

    func unload() { model = nil }

    func encode(samples: [Float], releaseAfter: Bool = true) async throws -> SoftTokenRows {
        guard !samples.isEmpty else {
            throw LLMEngineError.generationFailed(reason: "AudioEncoder: the recording holds no samples")
        }
        try await loadIfNeeded()
        guard let model else {
            throw LLMEngineError.generationFailed(reason: "AudioEncoder: model is not loaded")
        }
        defer { if releaseAfter { unload() } }
        let inputs = try AudioPreprocess.fixedInputs(samples: samples)
        let out = try await model.prediction(from: MLDictionaryFeatureProvider(
            dictionary: ["mel": inputs.mel, "mask": inputs.mask]))
        guard let soft = out.featureValue(for: "soft_tokens")?.multiArrayValue else {
            throw LLMEngineError.generationFailed(reason: "AudioEncoder: no soft_tokens output")
        }
        let shape = soft.shape.map { $0.intValue }
        guard shape.count == 3, shape[0] == 1 else {
            throw LLMEngineError.generationFailed(
                reason: "AudioEncoder: soft_tokens must be [1, tokens, hidden] (actual \(shape))")
        }
        let produced = shape[1]
        let hidden = shape[2]
        let wanted = AudioPreprocess.softTokenCount(samples: samples.count)
        guard wanted > 0, wanted <= produced else {
            throw LLMEngineError.generationFailed(
                reason: "AudioEncoder: \(wanted) soft tokens requested but the model emits \(produced)")
        }
        var data = [Float16](repeating: 0, count: wanted * hidden)
        Self.copyToF16(soft, count: wanted * hidden, into: &data)
        return SoftTokenRows(rows: wanted, hidden: hidden, data: data)
    }

    private static func copyToF16(_ array: MLMultiArray, count: Int, into dst: inout [Float16]) {
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

enum ASRPrompt {
    static let turnStart = 105
    static let turnEnd = 106
    static let newline = 107
    static let boa = 256000
    static let eoa = 258883
    static let defaultLanguage = "English"

    static func instruction(language: String = defaultLanguage) -> String {
        "Transcribe the following speech segment in \(language) into \(language) text.\n\n"
            + "Follow these specific instructions for formatting the answer:\n"
            + "* Only output the transcription, with no newlines.\n"
            + "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point "
            + "seven, and write 3 instead of three."
    }

    static func segments(
        bos: Int, userTokens: [Int], instructionTokens: [Int], modelTokens: [Int], audio: SoftTokenRows
    ) -> [PromptSegment] {
        let pre = [bos, turnStart] + userTokens + instructionTokens + [boa]
        let post = [eoa, turnEnd, newline, turnStart] + modelTokens
        return [.tokens(pre), .audio(audio), .tokens(post)]
    }

    static func flatIDs(
        bos: Int, userTokens: [Int], instructionTokens: [Int], modelTokens: [Int], audioRows: Int
    ) -> [Int] {
        [bos, turnStart] + userTokens + instructionTokens + [boa]
            + Array(repeating: ChunkedSpeculativeChain.audioPlaceholderID, count: audioRows)
            + [eoa, turnEnd, newline, turnStart] + modelTokens
    }

    static func followUpTokens(userTokens: [Int], questionTokens: [Int], modelTokens: [Int]) -> [Int] {
        [turnEnd, newline, turnStart] + userTokens + questionTokens
            + [turnEnd, newline, turnStart] + modelTokens
    }
}
