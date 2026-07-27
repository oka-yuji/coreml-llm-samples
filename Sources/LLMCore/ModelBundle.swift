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

    public var sidecarStage: String?

    public var computeUnits: String?

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
        sidecarStage: String? = nil,
        computeUnits: String? = nil
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
        self.sidecarStage = sidecarStage
        self.computeUnits = computeUnits
    }
}

public struct LoadOptions: Sendable, Hashable {
    public var computeUnits: ComputeUnitPreference

    public var preloadSpeculation: Bool

    public init(computeUnits: ComputeUnitPreference = .all, preloadSpeculation: Bool = true) {
        self.computeUnits = computeUnits
        self.preloadSpeculation = preloadSpeculation
    }
}

public enum ComputeUnitPreference: String, Sendable, Hashable, Codable, CaseIterable {
    case all
    case cpuAndGPU
    case cpuAndNeuralEngine
    case cpuOnly
}
