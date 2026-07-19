import CoreML
import Foundation
import LLMCore

/// Core ML バックエンド。チャンク分割チェーン形式のバンドル
/// (`manifest.json` の format = "coreml-chunked-chain-v1")を実行する。
public actor CoreMLEngine: LLMEngine {
    public nonisolated let descriptor = EngineDescriptor(
        name: "Core ML",
        backend: .coreML,
        supportsMultiTokenPrediction: true  // バンドルに MTP アセットがある場合に有効
    )

    /// 生成ループを回すチェーン(ステートレス v1 or stateful v2)。
    private var chain: (any GenerationChain)?
    /// MTP(投機的デコード)のホットパス。v1(`CoreMLChain`)/ v2(`CoreMLChainV2`)の両方が適合。
    /// ドラフター非同梱のバンドルでは `supportsMTP == false`。
    private var speculative: (any SpeculativeDecoding)?
    private var tokenizer: HFTokenizer?
    private var pendingLoadMetrics: LoadMetrics?
    private var promptPrefix = "<start_of_turn>user\n"
    private var promptSuffix = "<end_of_turn>\n<start_of_turn>model\n"
    /// アシスタントターンの終端(次ターンへの区切り)。manifest の prefix/suffix から load 時に導出。
    private var assistantSuffix = "<end_of_turn>\n"
    /// KV に処理済みのトークン列(prefill 分 + decode で本体に食わせた分)。
    /// マルチターン継続の LCP 計算の基準。不変条件: `chain.position == processedTokens.count`。
    private var processedTokens: [Int] = []
    private var loadedBundleURL: URL?
    private var loadedUnits: MLComputeUnits = .cpuOnly

    public init() {}

    public func load(_ model: ModelBundle, options: LoadOptions) async throws {
        guard chain == nil else { return }
        let clock = ContinuousClock()
        let start = clock.now

        let units: MLComputeUnits = switch options.computeUnits {
        case .all: .all
        case .cpuAndGPU: .cpuAndGPU
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        case .cpuOnly: .cpuOnly
        }

        if model.manifest.format == "coreml-stateful-chain-v2" {
            // v2 stateful は GPU 専用(CPU_ONLY / ANE では複数 state 同時ロードで segfault)。
            guard units == .cpuAndGPU || units == .all else {
                throw LLMEngineError.incompatibleBundle(
                    reason: "v2 stateful バンドルは GPU が必須です(computeUnits=\(options.computeUnits.rawValue))。"
                        + "CPU_ONLY / ANE では複数 MLState 同時ロードで native segfault になります。"
                        + "cpuAndGPU か all を指定してください")
            }
            let v2 = try await CoreMLChainV2(bundleURL: model.directoryURL, computeUnits: units)
            tokenizer = try await HFTokenizer(
                modelFolder: model.directoryURL, eosTokenIDs: Set(v2.config.eosIDs))
            chain = v2
            speculative = v2  // ドラフター(../drafter.mlmodelc)があれば supportsMTP=true
        } else {
            let v1 = try await CoreMLChain(bundleURL: model.directoryURL, computeUnits: units)
            tokenizer = try await HFTokenizer(
                modelFolder: model.directoryURL, eosTokenIDs: Set(v1.config.eosIDs))
            chain = v1
            speculative = v1
        }
        loadedBundleURL = model.directoryURL
        loadedUnits = units
        if let prefix = model.manifest.promptPrefix { promptPrefix = prefix }
        if let suffix = model.manifest.promptSuffix { promptSuffix = suffix }
        assistantSuffix = Self.deriveAssistantSuffix(prefix: promptPrefix, suffix: promptSuffix)
        processedTokens = []  // 新モデル = 新しい会話。
        pendingLoadMetrics = LoadMetrics(duration: clock.now - start)
    }

    public func unload() {
        chain = nil
        speculative = nil
        tokenizer = nil
        pendingLoadMetrics = nil
        processedTokens = []
    }

    /// 会話を初期化する(KV prefix cache / 処理済みトークンを破棄)。新規チャット・モデル切替時に呼ぶ。
    public func resetConversation() {
        try? chain?.reset()
        processedTokens = []
    }

    /// 測定・評価用に prompt テンプレート(turn マーカー)を明示上書きする(opt-in)。
    /// 既定パスは manifest 由来 or フォールバックのままで不変。r4a のように manifest に prefix/suffix が
    /// 無く、フォールバック(`<start_of_turn>`/`<end_of_turn>`)が当該モデルの実マーカーと食い違う場合に、
    /// 正しい Gemma 4(`<|turn>`/`<turn|>`)テンプレートを注入して正準な比較プロンプトを作るために使う。
    func setPromptTemplate(prefix: String, suffix: String) {
        promptPrefix = prefix
        promptSuffix = suffix
        assistantSuffix = Self.deriveAssistantSuffix(prefix: prefix, suffix: suffix)
    }

    // MARK: - Layer 0 スケルトン強制デコード(構文強制)

    /// スケルトン抽出(Layer 0)の 1 件分の結果。
    struct SkeletonExtractionResult: Sendable {
        let json: String              // ホストが組み立てた JSON(構文有効率は構造的に 100%)
        let promptText: String        // 正準チャットテンプレートで整形した素のプロンプト
        let promptTokens: Int         // BOS 込みのプロンプトトークン数
        let injectedTokens: Int       // 強制注入した構造トークンの predict 回数
        let valueTokens: Int          // モデルが生成した値スパントークン数
        let fieldDemotions: Int       // 非空だが null 降格したフィールド数
        let headSkips: Int            // lm_head をスキップした注入 predict 回数
        let genMillis: Double         // prefill + skeleton decode の総所要(1 件全体)
        let fields: [FieldRender]     // key -> 表示用値(目視検証用)
        let parseOK: Bool             // json.loads 相当が成立
        let keysComplete: Bool        // キー集合がスキーマと完全一致
    }

    struct FieldRender: Sendable {
        let key: String
        let rendered: String
    }

    /// Layer 0 スケルトン強制デコードで 1 件を抽出する。正準チャットテンプレート経由で prompt を整形し、
    /// 構造部はホストが強制注入、値スパンだけモデルに生成させる。会話は毎回リセット(reuseCache 相当なし)。
    func extractSkeletonJSON(
        prompt: String,
        schema: SkeletonSchema,
        history: [ChatTurn] = [],
        maxStringSpanTokens: Int = 48,
        maxNumberSpanTokens: Int = 16,
        useHeadSkip: Bool = true
    ) throws -> SkeletonExtractionResult {
        guard let chain, let tokenizer else { throw LLMEngineError.notLoaded }
        let promptText = conversationText(history: history, prompt: prompt)
        let promptIDs = try encodeConversation(history: history, prompt: prompt)
        guard promptIDs.count < chain.contextLength else {
            throw LLMEngineError.generationFailed(
                reason: "プロンプト(\(promptIDs.count) tokens)がコンテキスト長 \(chain.contextLength) を超過")
        }
        let decoder = try SkeletonJSONDecoder(
            schema: schema, tokenizer: tokenizer,
            maxStringSpanTokens: maxStringSpanTokens, maxNumberSpanTokens: maxNumberSpanTokens,
            useHeadSkip: useHeadSkip)
        let clock = ContinuousClock()
        let start = clock.now
        try chain.reset()
        let firstPending = try chain.prefill(promptIDs)
        let (values, stats) = try decoder.decode(chain: chain, firstPending: firstPending)
        let json = decoder.assembleJSON(values)
        let genMs = (clock.now - start) / .seconds(1) * 1000
        let (parseOK, keysComplete) = SkeletonJSONDecoder.selfCheck(json: json, schema: schema)
        let rendered: [FieldRender] = values.map { key, value in
            let text: String
            switch value {
            case .null: text = "null"
            case .string(let s): text = s
            case .enumeration(let s): text = s
            case .number(let n): text = n
            }
            return FieldRender(key: key, rendered: text)
        }
        return SkeletonExtractionResult(
            json: json, promptText: promptText, promptTokens: promptIDs.count,
            injectedTokens: stats.injectedTokens, valueTokens: stats.valueTokens,
            fieldDemotions: stats.fieldDemotions, headSkips: stats.headSkips, genMillis: genMs,
            fields: rendered, parseOK: parseOK, keysComplete: keysComplete)
    }

    /// 正準チャットテンプレートで整形した素のプロンプトと、そのトークン数(BOS 込み)を返す(報告用)。
    func canonicalPrompt(prompt: String, history: [ChatTurn] = []) throws -> (text: String, tokenCount: Int) {
        let text = conversationText(history: history, prompt: prompt)
        let ids = try encodeConversation(history: history, prompt: prompt)
        return (text, ids.count)
    }

    /// 直近生成の PLD 統計(v2 stateful かつ `CORELLM_MTP_PLD` 有効時のみ意味を持つ)。
    /// v2 以外・PLD 無効では enabled=false / カウンタ 0。テスト/ベンチログ用の内部 API。
    func pldStatsSnapshot() -> CoreMLChainV2.PLDStats? {
        (chain as? CoreMLChainV2)?.pldStatsSnapshot()
    }

    /// 直近生成のツリー MTP 統計(v2 stateful かつ `CORELLM_MTP_TREE` 有効時のみ意味を持つ)。
    /// 代替ノード発火回数 / +1 回収回数。テスト(ロスレスゲートの回収踏破 assert)/ベンチログ用の内部 API。
    func treeStatsSnapshot() -> CoreMLChainV2.TreeStats? {
        (chain as? CoreMLChainV2)?.treeStatsSnapshot()
    }

    public nonisolated func generate(
        _ request: GenerationRequest
    ) -> AsyncThrowingStream<GenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(request, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(
        _ request: GenerationRequest,
        continuation: AsyncThrowingStream<GenerationEvent, any Error>.Continuation
    ) async throws {
        guard let chain, let tokenizer else { throw LLMEngineError.notLoaded }

        let clock = ContinuousClock()
        let start = clock.now
        if let metrics = pendingLoadMetrics {
            continuation.yield(.loadCompleted(metrics))
            pendingLoadMetrics = nil
        }

        // 会話全体を manifest の prefix/suffix で整形してトークン化(単一ターンと同じ流儀)。
        // history 空 + reuseCache=false なら従来の単一ターンと完全に同一の列になる(後方互換)。
        let promptIDs = try encodeConversation(history: request.history, prompt: request.prompt)

        guard promptIDs.count < chain.contextLength else {
            throw LLMEngineError.generationFailed(
                reason: "プロンプト(\(promptIDs.count) tokens)がコンテキスト長 \(chain.contextLength) を超過"
            )
        }
        let maxNewTokens = min(request.config.maxNewTokens, chain.contextLength - promptIDs.count)

        // MTP アセット(ドラフター等)は初回使用時に遅延ロード(起動時ロードを約半分にするため)。
        // v2 は prefill 中に store 層 KV / 最終 hidden を捕捉する必要があるため、**prefill より前に**
        // ロードして `mtpLoaded=true` にしておく(v1 はホスト KV を常時保持するので順序不問)。
        let useMTP = request.config.multiTokenPrediction && (speculative?.supportsMTP ?? false)

        // prefill。reuseCache=true(マルチターン継続): 前ターンまでの KV を保持し、会話全体のトークン列と
        // 処理済み列の LCP を求めて位置をそこへ巻き戻し、LCP 以降の差分だけを prefill する。
        // false(既定): reset して会話全体を再 prefill(従来挙動)。どちらも生成結果は構成的に等価。
        // v1 は S=1 逐次(step ループ)、v2 は S=128 ブロック + 端数(state 化)。
        let reusedTokens: Int
        if request.reuseCache {
            // 最低 1 トークンは prefill する(最終位置の hidden から次トークンの argmax を得るため)。
            let lcp = max(0, min(Self.commonPrefixLength(promptIDs, processedTokens), promptIDs.count - 1))
            chain.rewind(to: lcp)
            // 巻き戻し直後の確定 KV は再利用接頭辞のみ。prefill が途中で失敗(キャンセル等)しても
            // processedTokens が実際の確定 KV と矛盾しないよう、ここで接頭辞へ縮めておく
            // (次ターンの LCP は最大でも lcp まで = 上書きされていない正しい位置しか再利用しない)。
            processedTokens = Array(promptIDs[0..<lcp])
            reusedTokens = lcp
        } else {
            try chain.reset()
            processedTokens = []
            reusedTokens = 0
        }
        if useMTP { try await installMTPIfNeeded() }
        let diff = reusedTokens > 0 ? Array(promptIDs[reusedTokens...]) : promptIDs
        var nextToken = try chain.prefill(diff)
        // prefill 成功: 位置 [0, promptIDs.count) の KV は promptIDs と一致している。
        processedTokens = promptIDs
        continuation.yield(.prefillCompleted(
            PrefillMetrics(
                promptTokens: promptIDs.count, reusedTokens: reusedTokens, duration: clock.now - start)
        ))

        // decode(greedy。temperature 等のサンプリングは logits 出力ヘッド導入後)。
        // MTP ON: ドラフターが draftLen 個先読みし本体が一括 verify(出力はロスレスで OFF と一致)。
        let decodeStart = clock.now
        var firstTokenAt: ContinuousClock.Instant?
        var generated: [Int] = []
        var emittedText = ""
        var acceptedTotal = 0
        var draftedTotal = 0

        func emit(_ token: Int) throws {
            generated.append(token)
            let now = clock.now
            if firstTokenAt == nil { firstTokenAt = now }
            let fullText = try tokenizer.decode(generated)
            let delta = fullText.hasPrefix(emittedText) ? String(fullText.dropFirst(emittedText.count)) : fullText
            emittedText = fullText
            guard !delta.isEmpty else { return }
            let elapsed = (now - decodeStart) / .seconds(1)
            continuation.yield(.token(TokenChunk(
                text: delta,
                tokenID: token,
                tokensPerSecond: Double(generated.count) / max(elapsed, 0.001)
            )))
        }

        decodeLoop: while generated.count < maxNewTokens {
            try Task.checkCancellation()
            if tokenizer.eosTokenIDs.contains(nextToken) { break }

            if useMTP, let spec = speculative {
                // context = 確定済みトークン列(prompt + これまで emit した全トークン)。
                // 不変条件: processedTokens.count == chain.position。v2 の PLD はこの列の末尾 n-gram を探す。
                let round = try spec.mtpRound(prediction: nextToken, context: processedTokens)
                acceptedTotal += round.accepted
                draftedTotal += round.drafted
                // 採択分は EOS で途中 break しても全て KV に確定済み(position が accepted 分進んでいる)。
                processedTokens.append(contentsOf: round.emitted)
                for token in round.emitted {
                    try emit(token)
                    if tokenizer.eosTokenIDs.contains(token) || generated.count >= maxNewTokens {
                        break decodeLoop
                    }
                }
                nextToken = round.next
            } else {
                let current = nextToken
                try emit(current)
                nextToken = try chain.decodeStep(tokenID: current)
                processedTokens.append(current)  // decodeStep が current を KV に確定
            }
        }

        let decodeSeconds = (clock.now - decodeStart) / .seconds(1)
        continuation.yield(.finished(GenerationMetrics(
            promptTokens: promptIDs.count,
            generatedTokens: generated.count,
            timeToFirstToken: (firstTokenAt ?? decodeStart) - start,
            decodeTokensPerSecond: Double(generated.count) / max(decodeSeconds, 0.001),
            peakMemoryBytes: Self.memoryFootprint(),
            draftAcceptanceRate: draftedTotal > 0 ? Double(acceptedTotal) / Double(draftedTotal) : nil
        )))
    }

    /// 会話履歴 + 現在の user プロンプトを、単一ターンと同じ流儀(manifest の prefix/suffix)で
    /// 整形した「素の文字列」にする(トークン化前)。継続と全再 prefill が同一列を作ることを保証する。
    /// history 空なら `promptPrefix + prompt + promptSuffix`(従来の単一ターンと同一)。
    private func conversationText(history: [ChatTurn], prompt: String) -> String {
        var text = ""
        for turn in history {
            switch turn.role {
            case .user: text += promptPrefix + turn.text + promptSuffix
            case .assistant: text += turn.text + assistantSuffix
            }
        }
        text += promptPrefix + prompt + promptSuffix
        return text
    }

    /// 会話履歴 + 現在の user プロンプトを整形してトークン列にする。先頭に BOS を付与。
    private func encodeConversation(history: [ChatTurn], prompt: String) throws -> [Int] {
        guard let tokenizer else { throw LLMEngineError.notLoaded }
        var ids = try tokenizer.encode(conversationText(history: history, prompt: prompt))
        let bos = tokenizer.bosTokenID ?? 2
        if ids.first != bos { ids.insert(bos, at: 0) }
        return ids
    }

    /// アシスタントターンの終端文字列を manifest の prefix/suffix から導出する(models/ を触らないため)。
    /// promptPrefix="<X>user\n" から role タグ "<X>" を取り、promptSuffix 中の "<X>model" より前
    /// (= ターン終端 + 改行)を返す。12B: "<turn|>\n"、gemma 標準: "<end_of_turn>\n"。
    /// 導出できなければ suffix をそのまま返す(継続/全再 prefill 間で一貫していれば正しさは不変)。
    static func deriveAssistantSuffix(prefix: String, suffix: String) -> String {
        guard let userRange = prefix.range(of: "user") else { return suffix }
        let roleOpen = String(prefix[..<userRange.lowerBound])
        guard !roleOpen.isEmpty, let modelRange = suffix.range(of: roleOpen + "model") else { return suffix }
        return String(suffix[..<modelRange.lowerBound])
    }

    /// 2 つのトークン列の最長共通接頭辞(LCP)の長さ。
    static func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }

    /// MTP アセットを(未ロードなら)ロードして注入する。資産構成が違うため具象型で分岐する:
    /// v1 はドラフター + S=6 verify チェーン(chunk×4 + lm_head)、v2 はドラフターのみ
    /// (verify は本体 v2 グラフの S=draftLen 呼び出しで賄う)。
    private func installMTPIfNeeded() async throws {
        guard let bundleURL = loadedBundleURL else { return }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = loadedUnits

        if let v1 = speculative as? CoreMLChain, !v1.mtpLoaded, let mtp = v1.config.mtp {
            let drafter = try await CoreMLChain.loadCompiled(bundleURL: bundleURL, name: mtp.drafter, configuration: cfg)
            var verify: [MLModel] = []
            for name in mtp.verifyChunks {
                verify.append(try await CoreMLChain.loadCompiled(bundleURL: bundleURL, name: name, configuration: cfg))
            }
            let verifyHead = try await CoreMLChain.loadCompiled(bundleURL: bundleURL, name: mtp.verifyLmhead, configuration: cfg)
            try v1.installMTP(drafter: drafter, verifyChunks: verify, verifyHead: verifyHead)
        } else if let v2 = speculative as? CoreMLChainV2, !v2.mtpLoaded, let drafterURL = v2.drafterURL {
            // 既にコンパイル済みの `.mlmodelc` を直ロード(loadCompiled は .mlpackage 名を .mlmodelc に
            // 読み替えるだけなので、そのまま渡せば再コンパイルは走らない)。
            //
            // ドラフター(v1 変換物、rank-1 stateless)の compute units はメイングラフとは独立に決める。
            // エンジンが `.all` のとき同じ `.all` を渡すと、この小型グラフが ANE に配置され
            // 初回コンパイルに数十秒 + draft ステップ毎のディスパッチ往復でスループットが退行する
            // (アプリ CU=all で MTP ON が OFF より遅い問題の原因)。既定はエンジン `.all` のとき
            // `.cpuAndGPU` に固定。env `CORELLM_MTP_DRAFTER_CU`(all|gpu|ane|cpu)で明示上書き可。
            let drafterCfg = MLModelConfiguration()
            drafterCfg.computeUnits = Self.resolveDrafterComputeUnits(
                default: loadedUnits == .all ? .cpuAndGPU : loadedUnits)
            let drafter = try await CoreMLChain.loadCompiled(
                bundleURL: drafterURL.deletingLastPathComponent(),
                name: drafterURL.lastPathComponent, configuration: drafterCfg)
            // ladder regime 切替: 第 2 ドラフター(`drafter_ring32k.mlmodelc` = w32768)が実在すればロードして
            // 渡す。非 ladder / 不在なら drafterPreURL=nil → drafterPre=nil = 従来どおり w131072 単独(後方互換)。
            var drafterPre: MLModel? = nil
            if let preURL = v2.drafterPreURL {
                drafterPre = try await CoreMLChain.loadCompiled(
                    bundleURL: preURL.deletingLastPathComponent(),
                    name: preURL.lastPathComponent, configuration: drafterCfg)
            }
            try v2.installMTP(drafter: drafter, drafterPre: drafterPre)
        }
    }

    /// v2 MTP ドラフターの compute units を決める。env `CORELLM_MTP_DRAFTER_CU`(all|gpu|ane|cpu)を
    /// 最優先し、未指定なら `fallback` を返す。A/B 用のフック兼、既定の配置固定に使う。
    private static func resolveDrafterComputeUnits(default fallback: MLComputeUnits) -> MLComputeUnits {
        switch ProcessInfo.processInfo.environment["CORELLM_MTP_DRAFTER_CU"]?.lowercased() {
        case "all": return .all
        case "gpu": return .cpuAndGPU
        case "ane": return .cpuAndNeuralEngine
        case "cpu": return .cpuOnly
        default: return fallback
        }
    }

    /// task_vm_info の phys_footprint(jetsam 判定に使われる実メモリ)。
    private static func memoryFootprint() -> Int? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) : nil
    }
}
