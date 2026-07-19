import Foundation

/// 変換済みモデル一式(モデル本体 + tokenizer + manifest)を指すディレクトリ。
public struct ModelBundle: Sendable {
    public var directoryURL: URL
    public var manifest: ModelManifest

    public init(directoryURL: URL, manifest: ModelManifest) {
        self.directoryURL = directoryURL
        self.manifest = manifest
    }

    /// `manifest.json` を読んでバンドルを開く。
    public init(contentsOf directoryURL: URL) throws {
        let manifestURL = directoryURL.appending(path: "manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path()) else {
            throw LLMEngineError.modelNotFound(path: manifestURL.path())
        }
        let data = try Data(contentsOf: manifestURL)
        self.directoryURL = directoryURL
        self.manifest = try JSONDecoder().decode(ModelManifest.self, from: data)
    }

    /// MTP ドラフターを同梱しているか。
    public var supportsMultiTokenPrediction: Bool {
        manifest.drafterRelativePath != nil
    }
}

/// バンドル内の `manifest.json`。tools/ の変換スクリプトが書き出す。
public struct ModelManifest: Sendable, Codable, Hashable {
    /// 表示名(例: "Gemma 4 E4B it")。
    public var name: String
    /// アーキテクチャ識別子(例: "gemma4")。
    public var architecture: String
    /// 変換時に焼き込んだ最大コンテキスト長。
    public var contextLength: Int
    /// バンドル形式。"coreml-chunked-chain-v1" = チャンク分割チェーン
    /// (modelRelativePath は convert_config.json を指す)。nil は単一モデル。
    public var format: String?
    /// ターゲットモデル(または形式ごとの主要エントリ)の相対パス。
    public var modelRelativePath: String
    /// tokenizer.json の相対パス。
    public var tokenizerRelativePath: String
    /// MTP ドラフターの相対パス(例: "drafter.mlpackage")。nil なら MTP 非対応バンドル。
    public var drafterRelativePath: String?
    /// チャットプロンプトの前置文字列。nil なら Gemma 標準("<start_of_turn>user\n")。
    /// gemma4_unified(12B)はターン記法が異なるためバンドル側で指定する。
    public var promptPrefix: String?
    /// チャットプロンプトの後置文字列。nil なら Gemma 標準。
    public var promptSuffix: String?
    /// r4(emlink)バンドルの sidecar 量子化段の明示指定("fp16" / "int8")。nil なら
    /// `convert_config.json` の `default_stage`(既定 fp16)に従う。iOS 出荷は int8 段を明示する
    /// (embed/ple を int8 で持ち、6GB 級端末のメモリに収めるため)。Mac バンドルは未指定 = fp16。
    public var sidecarStage: String?
    /// バンドルが要求する実行ユニットの明示指定(ComputeUnitPreference の rawValue:
    /// "all"/"cpuAndGPU"/"cpuAndNeuralEngine"/"cpuOnly")。指定時はユーザー設定より優先される。
    /// 用途: r4a-int8 の iOS 出荷は **ANE 必須**(cpuOnly だと chunk 重みがアプリ常駐し ple int8
    /// 2.35GB の mmap が ENOMEM になる。ANE は重みを OS 側 wired に載せアプリ予算を空ける)。
    /// nil ならユーザー設定/既定に従う(Mac バンドルは未指定 = 既存挙動不変)。
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

    public init(computeUnits: ComputeUnitPreference = .all) {
        self.computeUnits = computeUnits
    }
}

/// 実行させたいコンピュートユニットの希望。
/// Core ML では `MLModelConfiguration.computeUnits`、Core AI では
/// `SpecializationOptions` にマップする。CU 別比較は本プロジェクトの主要検証軸。
public enum ComputeUnitPreference: String, Sendable, Hashable, Codable, CaseIterable {
    case all
    case cpuAndGPU
    case cpuAndNeuralEngine
    case cpuOnly
}
