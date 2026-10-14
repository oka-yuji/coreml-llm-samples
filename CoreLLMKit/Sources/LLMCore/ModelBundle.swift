import Foundation

public struct ModelBundle: Sendable {
    public var directoryURL: URL
    public var manifest: ModelManifest

    public init(directoryURL: URL, manifest: ModelManifest) {
        self.directoryURL = directoryURL
        self.manifest = manifest
    }

    public init(contentsOf directoryURL: URL) throws {
        let manifestURL = directoryURL.appending(path: "manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path(percentEncoded: false)) else {
            throw LLMEngineError.modelNotFound(path: manifestURL.path(percentEncoded: false))
        }
        let data = try Data(contentsOf: manifestURL)
        self.directoryURL = directoryURL
        self.manifest = try JSONDecoder().decode(ModelManifest.self, from: data)
    }

    public var supportsMultiTokenPrediction: Bool {
        manifest.drafterRelativePath != nil
    }
}

public struct ModelManifest: Sendable, Codable, Hashable {

    public var name: String

    public var architecture: String

    public var contextLength: Int

    public var format: String?

    public var modelRelativePath: String

    public var tokenizerRelativePath: String

    public var drafterRelativePath: String?

    public var promptPrefix: String?

    public var promptSuffix: String?

    public var assistantSuffix: String?

    public var promptSuffixThinking: String?

    public var eos: [Int]?

    public var sidecarStage: String?

    public var computeUnits: String?

    /// `resident` (default) or `sequential`. `CORELLM_PREFILL_MODE` overrides it.
    public var prefillMode: String?

    /// Early exit: prefill the prompt only through the chunks that own KV (layers below
    /// `first_shared`), then run the KV-sharing chunks once, as a decode step on the prompt's last
    /// token. Their prefill functions are never loaded. `false`/absent is off.
    /// `CORELLM_PREFILL_EARLY_EXIT` overrides it.
    public var prefillEarlyExit: Bool?

    /// `default` (the Core ML default) or `fastPrediction`. `CORELLM_SPECIALIZATION` overrides it.
    public var specializationStrategy: String?

    /// Specialize the prefill functions in the background right after the model loads, so the
    /// first long prompt does not pay for it. `false`/absent is off, `true` prewarms every prefill
    /// width, and a comma-separated list such as `32` prewarms only those.
    /// `CORELLM_PREFILL_PREWARM` overrides it.
    public var prefillPrewarm: String?

    public init(
        name: String,
        architecture: String,
        contextLength: Int,
        format: String? = nil,
        modelRelativePath: String,
        tokenizerRelativePath: String,
        drafterRelativePath: String? = nil,
        promptPrefix: String? = nil,
        promptSuffix: String? = nil,
        assistantSuffix: String? = nil,
        promptSuffixThinking: String? = nil,
        eos: [Int]? = nil,
        sidecarStage: String? = nil,
        computeUnits: String? = nil,
        prefillMode: String? = nil,
        specializationStrategy: String? = nil,
        prefillPrewarm: String? = nil,
        prefillEarlyExit: Bool? = nil
    ) {
        self.name = name
        self.architecture = architecture
        self.contextLength = contextLength
        self.format = format
        self.modelRelativePath = modelRelativePath
        self.tokenizerRelativePath = tokenizerRelativePath
        self.drafterRelativePath = drafterRelativePath
        self.promptPrefix = promptPrefix
        self.promptSuffix = promptSuffix
        self.assistantSuffix = assistantSuffix
        self.promptSuffixThinking = promptSuffixThinking
        self.eos = eos
        self.sidecarStage = sidecarStage
        self.computeUnits = computeUnits
        self.prefillMode = prefillMode
        self.specializationStrategy = specializationStrategy
        self.prefillPrewarm = prefillPrewarm
        self.prefillEarlyExit = prefillEarlyExit
    }
}

public struct LoadOptions: Sendable, Hashable {
    public var computeUnits: ComputeUnitPreference

    public var preloadSpeculation: Bool

    public var allowUnsafeComputeUnits: Bool

    public init(
        computeUnits: ComputeUnitPreference = .all, preloadSpeculation: Bool = true,
        allowUnsafeComputeUnits: Bool = false
    ) {
        self.computeUnits = computeUnits
        self.preloadSpeculation = preloadSpeculation
        self.allowUnsafeComputeUnits = allowUnsafeComputeUnits
    }
}

public enum ComputeUnitPreference: String, Sendable, Hashable, Codable, CaseIterable {
    case all
    case cpuAndGPU
    case cpuAndNeuralEngine
    case cpuOnly
}
