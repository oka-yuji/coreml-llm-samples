import CoreML
import Foundation
import LLMCore

/// 4 chunk + lm_head のチェーン実行。
///
/// チャンクはステートレス(KV キャッシュは通常の入出力テンソル)で、
/// KV バッファ・書き込み位置・attention mask はすべてこのクラス(ホスト)が管理する。
/// バンドルに MTP アセット(ドラフター + S トークン verify グラフ)があれば
/// 投機的デコード(`mtpRound`)も提供する。
/// 方式の詳細は docs/models/gemma4-e4b-conversion-notes.md を参照。
final class CoreMLChain {
    let config: ChainConfig
    private(set) var position = 0
    /// 最後に処理したトークン(MTP のドラフト起点)。
    private(set) var lastToken = 0

    /// MTP アセットがバンドルに存在するか(実ロードは `loadMTP()` まで遅延)。
    var supportsMTP: Bool { config.mtp != nil }
    private(set) var mtpLoaded = false
    var draftLen: Int { config.mtp?.draftLen ?? 0 }

    private let bundleURL: URL
    private let mlConfig: MLModelConfiguration
    private let chunks: [MLModel]
    private let head: MLModel
    private let embedSidecar: Sidecar
    /// PLE サイドカー(PLE なしのモデルでは nil)。
    private let pleSidecar: Sidecar?
    private let host: HostInputs

    // MTP アセット(loadMTP() で遅延ロード。起動時ロードを約半分にする)
    private var drafter: MLModel?
    private var verifyChunks: [MLModel] = []
    private var verifyHead: MLModel?
    private var verifyHost: VerifyHostInputs?

    /// 非共有層(0..<firstShared)の KV キャッシュ。shape (NKV, CTX, headDim(i))。
    private var kbuf: [MLMultiArray] = []
    private var vbuf: [MLMultiArray] = []

    /// 使い回しの入力バッファ。
    private let hiddenBuffer: MLMultiArray       // (H,)
    private let embedsBuffer: MLMultiArray       // (H,)
    private let tokidBuffers: [MLMultiArray]     // chunk ごとに (B-A, PLE)
    private var draftEmbedBuffer: MLMultiArray?  // (H,)
    private var draftHiddenBuffer: MLMultiArray? // (H,)
    private var verifyHiddenBuffer: MLMultiArray?   // (S, H)
    private var verifyEmbedsBuffer: MLMultiArray?   // (S, H)
    private var verifyTokidBuffers: [MLMultiArray] = []  // chunk ごとに (S, B-A, PLE)

    init(bundleURL: URL, computeUnits: MLComputeUnits) async throws {
        // クロージャ内で self をキャプチャしないよう、初期化中はローカルの config を使う。
        let config = try ChainConfig.load(from: bundleURL.appending(path: "convert_config.json"))
        self.config = config
        self.bundleURL = bundleURL

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        self.mlConfig = mlConfig
        var loadedChunks: [MLModel] = []
        for name in config.chunks {
            loadedChunks.append(try await Self.loadCompiled(bundleURL: bundleURL, name: name, configuration: mlConfig))
        }
        chunks = loadedChunks
        head = try await Self.loadCompiled(bundleURL: bundleURL, name: config.lmhead, configuration: mlConfig)

        embedSidecar = try Sidecar(
            url: bundleURL.appending(path: config.sidecars.embed.file),
            rows: config.sidecars.embed.shape[0], cols: config.sidecars.embed.shape[1]
        )
        if let ple = config.sidecars.ple, config.hasPLE {
            pleSidecar = try Sidecar(
                url: bundleURL.appending(path: ple.file),
                rows: ple.shape[0], cols: ple.shape[1]
            )
        } else {
            pleSidecar = nil
        }
        host = try HostInputs(config: config)

        hiddenBuffer = try MLMultiArray(shape: [NSNumber(value: config.H)], dataType: .float16)
        embedsBuffer = try MLMultiArray(shape: [NSNumber(value: config.H)], dataType: .float16)
        if config.hasPLE {
            tokidBuffers = try config.chunkBounds.map { bounds in
                try MLMultiArray(
                    shape: [NSNumber(value: bounds[1] - bounds[0]), NSNumber(value: config.PLE)],
                    dataType: .float16
                )
            }
        } else {
            tokidBuffers = []
        }

        try reset()
    }

    /// MTP アセットの注入(遅延ロードの後半)。ロード済み MLModel は
    /// エンジン(actor)側で `loadCompiled` して渡す(Swift 6 の隔離境界のため)。
    func installMTP(drafter: MLModel, verifyChunks: [MLModel], verifyHead: MLModel) throws {
        guard !mtpLoaded, let mtp = config.mtp else { return }
        self.drafter = drafter
        self.verifyChunks = verifyChunks
        self.verifyHead = verifyHead
        verifyHost = try VerifyHostInputs(config: config, draftLen: mtp.draftLen)
        draftEmbedBuffer = try MLMultiArray(shape: [NSNumber(value: config.H)], dataType: .float16)
        draftHiddenBuffer = try MLMultiArray(shape: [NSNumber(value: config.H)], dataType: .float16)
        verifyHiddenBuffer = try MLMultiArray(
            shape: [NSNumber(value: mtp.draftLen), NSNumber(value: config.H)], dataType: .float16)
        verifyEmbedsBuffer = try MLMultiArray(
            shape: [NSNumber(value: mtp.draftLen), NSNumber(value: config.H)], dataType: .float16)
        verifyTokidBuffers = config.hasPLE ? try config.chunkBounds.map { bounds in
            try MLMultiArray(
                shape: [NSNumber(value: mtp.draftLen), NSNumber(value: bounds[1] - bounds[0]), NSNumber(value: config.PLE)],
                dataType: .float16)
        } : []
        mtpLoaded = true
    }

    /// `.mlpackage` を初回だけコンパイルし、`.mlmodelc` をキャッシュしてロードする
    /// (毎起動の再コンパイルを避ける。7GB 級では数分の差になる)。
    /// キャッシュ先は `CompiledModelStore` が OS 別に決める(macOS=バンドル内で従来不変、iOS=App Support)。
    static func loadCompiled(
        bundleURL: URL, name: String, configuration: MLModelConfiguration
    ) async throws -> MLModel {
        let compiledURL = try await CompiledModelStore.compiledModelURL(bundleURL: bundleURL, name: name)
        return try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
    }

    /// KV バッファをゼロ初期化して位置を巻き戻す。
    func reset() throws {
        position = 0
        lastToken = 0
        kbuf = try (0..<config.firstShared).map { try zeroKV(layer: $0) }
        vbuf = try (0..<config.firstShared).map { try zeroKV(layer: $0) }
    }

    /// KV バッファは保持したまま位置だけを巻き戻す(マルチターン継続 / prefix cache)。
    /// v1 は kbuf がホスト常駐なので、巻き戻し位置以降のスロットは後続 `step` の onehot 書き込みで
    /// 上書きされ、mask が未確定位置を不可視化する(全再 prefill と構成的に等価)。
    func rewind(to newPosition: Int) {
        position = max(0, min(newPosition, config.CTX))
    }

    private func zeroKV(layer i: Int) throws -> MLMultiArray {
        let a = try MLMultiArray(
            shape: [NSNumber(value: config.kvHeads(ofLayer: i)),
                    NSNumber(value: config.CTX),
                    NSNumber(value: config.headDim(ofLayer: i))],
            dataType: .float16
        )
        a.withF16 { $0.update(repeating: 0) }
        return a
    }

    /// トークン 1 つを処理し、greedy の次トークン id を返す。
    func step(tokenID: Int) throws -> Int {
        guard position < config.CTX else {
            throw LLMEngineError.generationFailed(reason: "コンテキスト長 \(config.CTX) を超過")
        }
        host.update(position: position)

        // sidecar ギャザー(スケールは焼き込み済み)。
        embedsBuffer.withF16 { buf in
            embedSidecar.read(row: tokenID, into: buf.baseAddress!)
        }
        if let pleSidecar {
            for (ci, bounds) in config.chunkBounds.enumerated() {
                tokidBuffers[ci].withF16 { buf in
                    pleSidecar.read(
                        row: tokenID,
                        offset: bounds[0] * config.PLE,
                        count: (bounds[1] - bounds[0]) * config.PLE,
                        into: buf.baseAddress!
                    )
                }
            }
        }

        // hidden_in の初期値は inputs_embeds。
        copyFlat(from: embedsBuffer, to: hiddenBuffer, count: config.H)

        for (ci, bounds) in config.chunkBounds.enumerated() {
            var features: [String: Any] = [
                "hidden_in": hiddenBuffer,
                "onehot": host.onehot,
                "cos_s": host.cosS, "sin_s": host.sinS,
                "cos_f": host.cosF, "sin_f": host.sinF,
                "mask_s": host.maskS, "mask_f": host.maskF,
            ]
            if config.hasPLE {
                features["inputs_embeds"] = embedsBuffer
                features["tokid"] = tokidBuffers[ci]
            }
            addKVFeatures(&features, bounds: bounds)
            let output = try chunks[ci].prediction(from: MLDictionaryFeatureProvider(dictionary: features))
            guard let hidden = output.featureValue(for: "hidden")?.multiArrayValue else {
                throw LLMEngineError.generationFailed(reason: "chunk \(ci) が hidden を返さなかった")
            }
            copyFlat(from: hidden, to: hiddenBuffer, count: config.H)
            try adoptKVOutputs(output, bounds: bounds, chunkIndex: ci)
        }

        let headOut = try head.prediction(from: MLDictionaryFeatureProvider(dictionary: ["hidden": hiddenBuffer]))
        guard let token = headOut.featureValue(for: "token")?.multiArrayValue else {
            throw LLMEngineError.generationFailed(reason: "lm_head が token を返さなかった")
        }
        position += 1
        lastToken = tokenID
        return token[0].intValue
    }

    // MARK: - MTP(投機的デコード)

    /// MTP 1 ラウンド: ドラフターが draftLen 個先読み → 本体が一括 verify → 採択。
    ///
    /// `prediction` は直前ステップで得た本体の次トークン(位置 `position` の正解)。
    /// 戻り値の emitted は確定したトークン列(先頭は必ず prediction と一致)、
    /// next は次の未処理予測トークン。出力は greedy の逐次デコードとロスレスで一致する。
    /// `context` は v2 の PLD 用の引数で、v1(ステートレス)では使わない(挙動不変)。
    func mtpRound(prediction: Int, context: [Int] = []) throws -> MTPRound {
        guard let drafter, let draftEmbedBuffer, let draftHiddenBuffer, let verifyHiddenBuffer,
              position + draftLen < config.CTX else {
            let next = try step(tokenID: prediction)
            return MTPRound(emitted: [prediction], next: next, accepted: 0, drafted: 0)
        }

        // 1) draft: 最後に処理したトークンとその hidden(post-norm)から先読み。
        //    ドラフターの position は「最後に処理した位置」= position - 1 で固定(HF 実装と同じ)。
        host.update(position: position - 1)
        copyFlat(from: hiddenBuffer, to: draftHiddenBuffer, count: config.H)
        let storeS = config.storeLayers["sliding_attention"]!
        let storeF = config.storeLayers["full_attention"]!
        var tok = lastToken
        var drafts: [Int] = []
        for _ in 0..<draftLen {
            draftEmbedBuffer.withF16 { buf in
                embedSidecar.read(row: tok, into: buf.baseAddress!)
            }
            let out = try drafter.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "embed": draftEmbedBuffer, "hidden": draftHiddenBuffer,
                "cos_s": host.cosS, "sin_s": host.sinS, "cos_f": host.cosF, "sin_f": host.sinF,
                "mask_s": host.maskS, "mask_f": host.maskF,
                "sks": kbuf[storeS], "svs": vbuf[storeS], "skf": kbuf[storeF], "svf": vbuf[storeF],
            ]))
            guard let t = out.featureValue(for: "token")?.multiArrayValue,
                  let h = out.featureValue(for: "hidden_next")?.multiArrayValue else {
                throw LLMEngineError.generationFailed(reason: "drafter の出力が不正")
            }
            tok = t[0].intValue
            drafts.append(tok)
            copyFlat(from: h, to: draftHiddenBuffer, count: config.H)
        }

        // 2) 先頭ドラフトが既知の正解と不一致ならラウンド不成立 → 通常ステップ。
        guard drafts[0] == prediction else {
            let next = try step(tokenID: prediction)
            return MTPRound(emitted: [prediction], next: next, accepted: 0, drafted: drafts.count)
        }

        // 3) verify: draftLen トークンを一括処理し、位置ごとの本体 argmax を得る。
        let base = position
        let targets = try verifyStep(tokens: drafts)
        var accepted = 1
        while accepted < drafts.count && drafts[accepted] == targets[accepted - 1] {
            accepted += 1
        }

        // 4) 採択分だけ位置を確定(棄却位置の KV は後続ステップの onehot 書き込みで上書きされる)。
        position = base + accepted
        lastToken = drafts[accepted - 1]
        copyRow(from: verifyHiddenBuffer, row: accepted - 1, to: hiddenBuffer)
        return MTPRound(
            emitted: Array(drafts[0..<accepted]), next: targets[accepted - 1],
            accepted: accepted, drafted: drafts.count)
    }

    /// S=draftLen トークンを一括処理し、各位置の greedy トークンを返す(位置は S 進む)。
    private func verifyStep(tokens: [Int]) throws -> [Int] {
        guard let verifyHead, let verifyHost, let verifyEmbedsBuffer, let verifyHiddenBuffer else {
            throw LLMEngineError.drafterUnavailable
        }
        precondition(tokens.count == draftLen)
        verifyHost.update(basePosition: position)

        verifyEmbedsBuffer.withF16 { buf in
            for (i, tok) in tokens.enumerated() {
                embedSidecar.read(row: tok, into: buf.baseAddress! + i * config.H)
            }
        }
        if let pleSidecar {
            for (ci, bounds) in config.chunkBounds.enumerated() {
                let n = bounds[1] - bounds[0]
                verifyTokidBuffers[ci].withF16 { buf in
                    for (i, tok) in tokens.enumerated() {
                        pleSidecar.read(
                            row: tok,
                            offset: bounds[0] * config.PLE,
                            count: n * config.PLE,
                            into: buf.baseAddress! + i * n * config.PLE
                        )
                    }
                }
            }
        }
        copyFlat(from: verifyEmbedsBuffer, to: verifyHiddenBuffer, count: draftLen * config.H)

        for (ci, bounds) in config.chunkBounds.enumerated() {
            var features: [String: Any] = [
                "hidden_in": verifyHiddenBuffer,
                "onehot": verifyHost.onehot,
                "cos_s": verifyHost.cosS, "sin_s": verifyHost.sinS,
                "cos_f": verifyHost.cosF, "sin_f": verifyHost.sinF,
                "mask_s": verifyHost.maskS, "mask_f": verifyHost.maskF,
            ]
            if config.hasPLE {
                features["inputs_embeds"] = verifyEmbedsBuffer
                features["tokid"] = verifyTokidBuffers[ci]
            }
            addKVFeatures(&features, bounds: bounds)
            let output = try verifyChunks[ci].prediction(from: MLDictionaryFeatureProvider(dictionary: features))
            guard let hidden = output.featureValue(for: "hidden")?.multiArrayValue else {
                throw LLMEngineError.generationFailed(reason: "verify chunk \(ci) が hidden を返さなかった")
            }
            copyFlat(from: hidden, to: verifyHiddenBuffer, count: draftLen * config.H)
            try adoptKVOutputs(output, bounds: bounds, chunkIndex: ci)
        }

        let headOut = try verifyHead.prediction(from: MLDictionaryFeatureProvider(dictionary: ["hidden": verifyHiddenBuffer]))
        guard let token = headOut.featureValue(for: "token")?.multiArrayValue else {
            throw LLMEngineError.generationFailed(reason: "verify lm_head が token を返さなかった")
        }
        position += draftLen
        return (0..<draftLen).map { token[$0].intValue }
    }

    // MARK: - 共通ヘルパ

    private func addKVFeatures(_ features: inout [String: Any], bounds: [Int]) {
        if bounds[0] >= config.firstShared {
            let storeS = config.storeLayers["sliding_attention"]!
            let storeF = config.storeLayers["full_attention"]!
            features["sks"] = kbuf[storeS]
            features["svs"] = vbuf[storeS]
            features["skf"] = kbuf[storeF]
            features["svf"] = vbuf[storeF]
        } else {
            for i in bounds[0]..<bounds[1] {
                features["k_\(i)_in"] = kbuf[i]
                features["v_\(i)_in"] = vbuf[i]
            }
        }
    }

    private func adoptKVOutputs(_ output: MLFeatureProvider, bounds: [Int], chunkIndex ci: Int) throws {
        guard bounds[0] < config.firstShared else { return }
        for i in bounds[0]..<bounds[1] {
            guard let k = output.featureValue(for: "k_\(i)_out")?.multiArrayValue,
                  let v = output.featureValue(for: "v_\(i)_out")?.multiArrayValue else {
                throw LLMEngineError.generationFailed(reason: "chunk \(ci) が層 \(i) の KV を返さなかった")
            }
            kbuf[i] = k
            vbuf[i] = v
        }
    }

    /// 出力(ランク不定)を平坦コピーする。
    private func copyFlat(from source: MLMultiArray, to destination: MLMultiArray, count: Int) {
        source.withF16 { src in
            destination.withF16 { dst in
                dst.baseAddress!.update(from: src.baseAddress!, count: count)
            }
        }
    }

    /// (S,H) バッファの行 row を (H,) バッファへコピーする。
    private func copyRow(from source: MLMultiArray, row: Int, to destination: MLMultiArray) {
        source.withF16 { src in
            destination.withF16 { dst in
                dst.baseAddress!.update(from: src.baseAddress! + row * config.H, count: config.H)
            }
        }
    }
}
