import CoreMLBackend
import Foundation
import LLMCore
import Observation

/// チャット画面の状態と生成ループを持つ ViewModel。
///
/// エンジンの生成経路は corellm-chat CLI と同一(`CoreMLEngine` の actor +
/// `AsyncThrowingStream<GenerationEvent>` を消費)。UI はこの ViewModel だけを見て描画し、
/// selftest(ヘッドレス)も同じ `loadModel` / `send` を通る。
@MainActor
@Observable
final class ChatViewModel {
    /// 画面全体のフェーズ。付随文字列は進捗/エラーの表示に使う。
    enum Phase: Equatable {
        case idle
        case loading(String)
        case ready
        case generating
        case failed(String)
    }

    /// 1 メッセージ(user / assistant)。assistant は streaming で `text` が伸びる。
    struct Message: Identifiable, Equatable {
        enum Role { case user, assistant }
        let id: UUID
        let role: Role
        var text: String

        init(id: UUID = UUID(), role: Role, text: String) {
            self.id = id
            self.role = role
            self.text = text
        }
    }

    // MARK: - Observed UI state

    var messages: [Message] = []
    var input: String = ""
    var phase: Phase = .idle
    /// 直近ターンの 1 行サマリ(tok/s・TTFT・採択率)。CLI の --stats の最小版。
    var statusLine: String = ""
    /// ロード/prefill の段階表示。
    var loadStatus: String = ""
    /// 初回推論の GPU カーネル特殊化(~40s)の最中かどうか。
    var warming: Bool = false
    /// ロード済みバンドルの表示名。
    var modelName: String = ""
    /// 1 ターンの最大生成トークン数(selftest で上書きする)。
    var maxTokens: Int = 512

    // MARK: - Engine state (non-observed)

    @ObservationIgnored private var engine: CoreMLEngine?
    @ObservationIgnored private var history: [ChatTurn] = []
    @ObservationIgnored private var hasWarmedUp = false
    /// 実行中の生成タスク。Stop でキャンセルし、selftest で完了を待つ。
    @ObservationIgnored var generation: Task<Void, Never>?

    init() {}

    // MARK: - Derived flags

    var isGenerating: Bool { if case .generating = phase { return true } else { return false } }
    var isLoading: Bool { if case .loading = phase { return true } else { return false } }
    var isModelLoaded: Bool { engine != nil }
    var trimmedInput: String { input.trimmingCharacters(in: .whitespacesAndNewlines) }

    var canSend: Bool { isModelLoaded && !isGenerating && !isLoading && !trimmedInput.isEmpty }
    var canReset: Bool { isModelLoaded && !isGenerating && !isLoading && !messages.isEmpty }
    var canLoad: Bool { !isGenerating && !isLoading }

    var phaseDescription: String {
        switch phase {
        case .idle: return "idle"
        case .loading(let s): return "loading: \(s)"
        case .ready: return "ready"
        case .generating: return "generating"
        case .failed(let s): return "failed: \(s)"
        }
    }

    // MARK: - Load

    /// バンドルディレクトリをロードする。CLI と同じ CPU+GPU 固定(v2 stateful は GPU 必須)。
    func loadModel(path: String) async {
        guard canLoad else { return }
        let url = URL(fileURLWithPath: path, isDirectory: true)
        let manifestURL = url.appending(path: "manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path()) else {
            phase = .failed("no manifest.json under \(path) — is this a model bundle directory?")
            return
        }
        phase = .loading("Reading manifest…")
        do {
            let bundle = try ModelBundle(contentsOf: url)
            modelName = bundle.manifest.name
            phase = .loading("Compiling / loading Core ML chunks (CPU+GPU)…")
            let newEngine = CoreMLEngine()
            try await newEngine.load(bundle, options: LoadOptions(computeUnits: .cpuAndGPU))
            engine = newEngine
            history = []
            messages = []
            hasWarmedUp = false
            warming = false
            statusLine = ""
            loadStatus = "Loaded. The first reply specializes GPU kernels (~40s)."
            phase = .ready
        } catch {
            engine = nil
            phase = .failed(String(describing: error))
        }
    }

    // MARK: - Send / Stop / Reset

    /// 入力を 1 ターン送信する。streaming で assistant メッセージを更新する。
    /// 送信できたら true(selftest が完了待ちの前段で使う)。
    @discardableResult
    func send() -> Bool {
        guard canSend, let engine else { return false }
        let userText = trimmedInput
        input = ""
        messages.append(Message(role: .user, text: userText))
        let assistantID = UUID()
        messages.append(Message(id: assistantID, role: .assistant, text: ""))
        statusLine = ""
        phase = .generating
        if !hasWarmedUp { warming = true }

        let config = GenerationConfig(maxNewTokens: maxTokens, temperature: 0, multiTokenPrediction: true)
        let request = GenerationRequest(prompt: userText, config: config, history: history, reuseCache: true)
        generation = Task { [self] in
            await consume(engine: engine, request: request, userText: userText, assistantID: assistantID)
        }
        return true
    }

    /// 生成をキャンセルする(部分出力は残す)。
    func stop() { generation?.cancel() }

    /// 会話をリセットする(KV / 履歴 / メッセージを破棄。CLI の /reset 相当)。
    /// カーネル特殊化はプロセス内で保持されるため hasWarmedUp は維持する。
    func reset() {
        guard let engine, canReset else { return }
        Task { await engine.resetConversation() }
        messages = []
        history = []
        statusLine = ""
        loadStatus = ""
    }

    // MARK: - Stream consumption

    private func consume(
        engine: CoreMLEngine,
        request: GenerationRequest,
        userText: String,
        assistantID: UUID
    ) async {
        var assistantText = ""
        var metrics: GenerationMetrics?
        do {
            for try await event in engine.generate(request) {
                switch event {
                case .loadCompleted(let m):
                    loadStatus = String(format: "Model loaded in %.1fs", seconds(m.duration))
                case .prefillCompleted(let p):
                    loadStatus = "Prefilled \(p.promptTokens) prompt tokens"
                    + (p.reusedTokens > 0 ? " (reused \(p.reusedTokens) from KV cache)" : "")
                case .token(let chunk):
                    warming = false
                    hasWarmedUp = true
                    assistantText += chunk.text
                    setAssistantText(id: assistantID, assistantText)
                case .finished(let m):
                    metrics = m
                }
            }
        } catch is CancellationError {
            // 消費側の for-await は cancel で nil 終了するのが基本経路だが、明示 throw も拾う。
            finishStopped(userText: userText, assistantText: assistantText)
            return
        } catch {
            warming = false
            if assistantText.isEmpty {
                setAssistantText(id: assistantID, "(generation failed)")
            }
            statusLine = "Error: \(error)"
            phase = .failed(String(describing: error))
            return
        }

        warming = false
        if Task.isCancelled {
            finishStopped(userText: userText, assistantText: assistantText)
            return
        }
        history.append(ChatTurn(role: .user, text: userText))
        history.append(ChatTurn(role: .assistant, text: assistantText))
        if let metrics { statusLine = statsLine(metrics) }
        phase = .ready
    }

    /// Stop で打ち切ったときの後始末。部分出力を履歴に残し、KV を破棄して次ターンを健全化する。
    private func finishStopped(userText: String, assistantText: String) {
        history.append(ChatTurn(role: .user, text: userText))
        history.append(ChatTurn(role: .assistant, text: assistantText))
        if let engine { Task { await engine.resetConversation() } }
        statusLine = "Stopped"
        phase = .ready
    }

    // MARK: - Helpers

    private func setAssistantText(id: UUID, _ text: String) {
        if let i = messages.firstIndex(where: { $0.id == id }) { messages[i].text = text }
    }

    private func seconds(_ d: Duration) -> Double { d / .seconds(1) }

    private func statsLine(_ m: GenerationMetrics) -> String {
        let msPerTok = m.decodeTokensPerSecond > 0 ? 1000.0 / m.decodeTokensPerSecond : 0
        var s = String(
            format: "%.1f tok/s  ·  %.0f ms/tok  ·  TTFT %.2fs  ·  %d→%d tok",
            m.decodeTokensPerSecond, msPerTok, seconds(m.timeToFirstToken),
            m.promptTokens, m.generatedTokens)
        if let acc = m.draftAcceptanceRate {
            s += String(format: "  ·  draft %.0f%%", acc * 100)
        }
        return s
    }
}
