import CoreML
import Foundation
import Hub
import Tokenizers

/// Core ML classification head of GLiNER2.5-multi (FluidInference's
/// `gliner2_multi_classification_fp16_L128_K8.mlpackage`).
///
/// The Python side builds its four input arrays with `gliner2`'s own
/// `SchemaTransformer` + `ExtractorCollator`. ``GLiNERSchemaEncoder`` mirrors that
/// layout exactly; `Tests/CoreMLBackendTests/GLiNERParityTests.swift` pins it to a
/// fixture generated from the Python reference.
///
/// Sequence layout for one classification task (no `[CLS]`/`[SEP]` — `gliner2`
/// calls `tokenize()`/`convert_tokens_to_ids()`, so the tokenizer's post-processor
/// never runs):
///
///     ( [P] <task> ( [L] <label0> [L] <label1> … ) ) [SEP_TEXT] <text words…>
///
/// `marker_indices` holds the subword position of each `[L]`.
public struct GLiNERInputs: Sendable, Equatable {
    public var inputIDs: [Int32]
    public var attentionMask: [Int32]
    public var markerIndices: [Int32]
    public var markerMask: [Float]
    /// Subword strings before ID lookup. Only used to explain parity failures.
    public var tokens: [String]
}

public enum GLiNERError: Error, CustomStringConvertible {
    case tokenizerDataMissing(String)
    case sequenceTooLong(subwords: Int, bucket: Int)
    case labelCountOutOfRange(Int, max: Int)
    case unknownToken(String)
    case missingOutput(String)

    public var description: String {
        switch self {
        case .tokenizerDataMissing(let path): "tokenizer data not found at \(path)"
        case .sequenceTooLong(let n, let bucket): "schema + text need \(n) subwords; bucket holds \(bucket)"
        case .labelCountOutOfRange(let n, let max): "expected 1...\(max) labels, got \(n)"
        case .unknownToken(let t): "token \(t) is not in the vocabulary"
        case .missingOutput(let name): "model output \(name) missing"
        }
    }
}

/// Reproduces `gliner2`'s classification preprocessing in Swift.
public final class GLiNERSchemaEncoder: Sendable {
    public static let sepText = "[SEP_TEXT]"
    public static let pToken = "[P]"
    public static let lToken = "[L]"

    private let tokenizer: any Tokenizer
    /// `added_tokens` from tokenizer.json. `UnigramTokenizer.convertTokenToId` only
    /// knows `model.vocab` (ids 0..<250101), so `[P]`, `[L]`, `[SEP_TEXT]` … would
    /// silently resolve to `[UNK]` without this map.
    private let addedTokenIDs: [String: Int32]
    private let padTokenID: Int32
    public let bucketLength: Int
    public let maxOptions: Int

    /// - Parameter tokenizerFolder: the model's `tokenizer/` directory
    ///   (`tokenizer.json` + `tokenizer_config.json`).
    public init(tokenizerFolder: URL, bucketLength: Int = 128, maxOptions: Int = 8) throws {
        self.bucketLength = bucketLength
        self.maxOptions = maxOptions

        let dataURL = tokenizerFolder.appending(path: "tokenizer.json")
        let configURL = tokenizerFolder.appending(path: "tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: dataURL.path(percentEncoded: false)) else {
            throw GLiNERError.tokenizerDataMissing(dataURL.path(percentEncoded: false))
        }

        let rawData = try Data(contentsOf: dataURL)
        let tokenizerData = try JSONDecoder().decode(Config.self, from: rawData)

        var configDict: [NSString: Any] =
            (try? JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any])
            .flatMap { dict in dict.reduce(into: [:]) { $0[$1.key as NSString] = $1.value } } ?? [:]
        // mDeBERTa-v3's `DebertaV2Tokenizer` is not in swift-transformers' registry, and the
        // fallback is BPE — wrong model entirely. The checkpoint is a SentencePiece Unigram
        // (tokenizer.json `model.type == "Unigram"`), which is what XLMRobertaTokenizer maps to.
        configDict["tokenizer_class"] = "XLMRobertaTokenizer"
        let tokenizerConfig = Config(configDict)

        tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: false)

        // Parse added_tokens straight from the JSON: these ids live above model.vocab.
        var added: [String: Int32] = [:]
        let parsed = try JSONSerialization.jsonObject(with: rawData) as? [String: Any]
        for entry in parsed?["added_tokens"] as? [[String: Any]] ?? [] {
            if let content = entry["content"] as? String, let id = entry["id"] as? Int {
                added[content] = Int32(id)
            }
        }
        addedTokenIDs = added
        padTokenID = added["[PAD]"] ?? 0
    }

    /// Words, lower-cased, matching `gliner2`'s `WhitespaceTokenSplitter`.
    /// Japanese has no spaces, so `\w+` runs break on punctuation — "吐出圧0.45MPa"
    /// becomes ["吐出圧0", ".", "45mpa"].
    static let wordPattern = try! NSRegularExpression(
        pattern: #"(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S"#,
        options: [.caseInsensitive])

    static func splitWords(_ text: String) -> [String] {
        let ns = text as NSString
        return Self.wordPattern
            .matches(in: text, range: NSRange(location: 0, length: ns.length))
            .map { ns.substring(with: $0.range).lowercased() }
    }

    /// `gliner2._collate_batch` appends "." when the text has no final punctuation,
    /// so every sequence ends with an extra `▁ .` pair. Parity depends on it.
    static func padPunctuation(_ text: String) -> String {
        guard !text.isEmpty else { return "." }
        return text.hasSuffix(".") || text.hasSuffix("!") || text.hasSuffix("?") ? text : text + "."
    }

    private func subwords(of token: String) -> [String] {
        // Added/special tokens stay whole; everything else goes through Unigram.
        addedTokenIDs[token] != nil ? [token] : tokenizer.tokenize(text: token)
    }

    private func id(of token: String) throws -> Int32 {
        if let added = addedTokenIDs[token] { return added }
        guard let id = tokenizer.convertTokenToId(token) else { throw GLiNERError.unknownToken(token) }
        return Int32(id)
    }

    public func encode(text: String, task: String, labels: [String]) throws -> GLiNERInputs {
        guard (1...maxOptions).contains(labels.count) else {
            throw GLiNERError.labelCountOutOfRange(labels.count, max: maxOptions)
        }

        // `_transform_schema(task, labels, "[L]")`
        var combined = ["(", Self.pToken, task, "("]
        for label in labels { combined.append(Self.lToken); combined.append(label) }
        combined.append(")")
        combined.append(")")
        // Single schema, so `_format_input_with_mapping` pops the trailing [SEP_STRUCT].
        combined.append(Self.sepText)
        let markerSlots = Set(stride(from: 4, to: 4 + 2 * labels.count, by: 2))
        combined.append(contentsOf: Self.splitWords(Self.padPunctuation(text)))

        var tokens: [String] = []
        var markerIndices: [Int32] = []
        for (index, token) in combined.enumerated() {
            if markerSlots.contains(index) { markerIndices.append(Int32(tokens.count)) }
            tokens.append(contentsOf: subwords(of: token))
        }

        guard tokens.count <= bucketLength else {
            throw GLiNERError.sequenceTooLong(subwords: tokens.count, bucket: bucketLength)
        }

        var inputIDs = [Int32](repeating: padTokenID, count: bucketLength)
        var attentionMask = [Int32](repeating: 0, count: bucketLength)
        for (index, token) in tokens.enumerated() {
            inputIDs[index] = try id(of: token)
            attentionMask[index] = 1
        }
        var paddedMarkers = [Int32](repeating: 0, count: maxOptions)
        var markerMask = [Float](repeating: 0, count: maxOptions)
        for (slot, position) in markerIndices.enumerated() {
            paddedMarkers[slot] = position
            markerMask[slot] = 1
        }

        return GLiNERInputs(
            inputIDs: inputIDs, attentionMask: attentionMask,
            markerIndices: paddedMarkers, markerMask: markerMask, tokens: tokens)
    }
}

/// Runs the Core ML classifier and returns one score per label.
///
/// `@unchecked Sendable`: every stored property is a `let`, and `MLModel.prediction`
/// is thread-safe. The mapper calls it serially regardless.
public final class GLiNERClassifier: @unchecked Sendable {
    public let encoder: GLiNERSchemaEncoder
    private let model: MLModel
    public let loadSeconds: Double

    /// - Parameters:
    ///   - modelURL: a compiled `.mlmodelc` or an `.mlpackage`.
    ///   - tokenizerFolder: the model's `tokenizer/` directory.
    public init(
        modelURL: URL, tokenizerFolder: URL,
        computeUnits: MLComputeUnits = .all, bucketLength: Int = 128, maxOptions: Int = 8
    ) throws {
        let start = Date()
        encoder = try GLiNERSchemaEncoder(
            tokenizerFolder: tokenizerFolder, bucketLength: bucketLength, maxOptions: maxOptions)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let compiled = modelURL.pathExtension == "mlpackage"
            ? try MLModel.compileModel(at: modelURL) : modelURL
        model = try MLModel(contentsOf: compiled, configuration: configuration)
        loadSeconds = Date().timeIntervalSince(start)
    }

    private static func multiArray(_ values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        values.withUnsafeBufferPointer { pointer.update(from: $0.baseAddress!, count: values.count) }
        return array
    }

    private static func multiArray(_ values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .float32)
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        values.withUnsafeBufferPointer { pointer.update(from: $0.baseAddress!, count: values.count) }
        return array
    }

    /// One score per label, in the order the labels were given.
    /// These are raw model outputs, not calibrated probabilities.
    public func scores(text: String, task: String, labels: [String]) throws -> [Float] {
        let inputs = try encoder.encode(text: text, task: task, labels: labels)
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": Self.multiArray(inputs.inputIDs),
            "attention_mask": Self.multiArray(inputs.attentionMask),
            "marker_indices": Self.multiArray(inputs.markerIndices),
            "marker_mask": Self.multiArray(inputs.markerMask),
        ])
        let output = try model.prediction(from: features)
        guard let probabilities = output.featureValue(for: "probabilities")?.multiArrayValue else {
            throw GLiNERError.missingOutput("probabilities")
        }
        return (0..<labels.count).map { probabilities[[0, NSNumber(value: $0)]].floatValue }
    }
}
