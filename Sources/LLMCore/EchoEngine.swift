import Foundation

/// モデル不要のダミーエンジン。プロンプトを語単位でストリーム返しする。
///
/// 用途:
/// - UI / ViewModel の開発・プレビュー(実モデルなしでストリーミング挙動を再現)
/// - LLMEngine プロトコル準拠の参照実装(actor + nonisolated generate のパターン)
/// - MTP トグルの動作確認(ON 時は擬似的に高速化し採択率を報告する)
public actor EchoEngine: LLMEngine {
    public nonisolated let descriptor = EngineDescriptor(
        name: "Echo",
        backend: .echo,
        supportsMultiTokenPrediction: true
    )

    private let tokenDelay: Duration
    private var isLoaded = false

    public init(tokenDelay: Duration = .milliseconds(40)) {
        self.tokenDelay = tokenDelay
    }

    public func load(_ model: ModelBundle, options: LoadOptions) async throws {
        isLoaded = true
    }

    public func unload() {
        isLoaded = false
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
        guard isLoaded else { throw LLMEngineError.notLoaded }

        let clock = ContinuousClock()
        let start = clock.now
        let mtp = request.config.multiTokenPrediction

        // 擬似 prefill。
        let promptTokens = max(request.prompt.count / 3, 1)
        try await clock.sleep(for: .milliseconds(120))
        continuation.yield(.prefillCompleted(
            PrefillMetrics(promptTokens: promptTokens, duration: clock.now - start)
        ))

        // 語単位の擬似トークン列。MTP ON では 1/3 の遅延で流す(≈ 3x 相当の見た目)。
        let words = Self.pseudoTokens(echoing: request.prompt, mtp: mtp)
        let delay = mtp ? tokenDelay / 3 : tokenDelay
        let decodeStart = clock.now
        var firstTokenAt: ContinuousClock.Instant?
        var emitted = 0

        for (index, word) in words.enumerated() {
            guard emitted < request.config.maxNewTokens else { break }
            try Task.checkCancellation()
            try await clock.sleep(for: delay)

            let now = clock.now
            if firstTokenAt == nil { firstTokenAt = now }
            emitted += 1
            let elapsed = (now - decodeStart) / .seconds(1)
            continuation.yield(.token(TokenChunk(
                text: word,
                tokenID: index,
                tokensPerSecond: Double(emitted) / max(elapsed, 0.001)
            )))
        }

        let totalSeconds = (clock.now - decodeStart) / .seconds(1)
        continuation.yield(.finished(GenerationMetrics(
            promptTokens: promptTokens,
            generatedTokens: emitted,
            timeToFirstToken: (firstTokenAt ?? decodeStart) - start,
            decodeTokensPerSecond: Double(emitted) / max(totalSeconds, 0.001),
            peakMemoryBytes: nil,
            draftAcceptanceRate: mtp ? 0.78 : nil
        )))
    }

    private static func pseudoTokens(echoing prompt: String, mtp: Bool) -> [String] {
        let mode = mtp ? "MTP ON" : "MTP OFF"
        let header = "(echo / \(mode)) "
        let body = prompt.split(separator: " ", omittingEmptySubsequences: false)
            .flatMap { word -> [String] in
                // 日本語などスペース区切りでない入力はある程度の長さで刻む。
                guard word.count > 8 else { return [String(word) + " "] }
                return word.chunked(into: 4).map(String.init)
            }
        return [header] + body
    }
}

extension Substring {
    fileprivate func chunked(into size: Int) -> [Substring] {
        var result: [Substring] = []
        var index = startIndex
        while index < endIndex {
            let next = self.index(index, offsetBy: size, limitedBy: endIndex) ?? endIndex
            result.append(self[index..<next])
            index = next
        }
        return result
    }
}
