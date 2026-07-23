import CoreMLBackend
import Foundation
import LLMCore
import Observation

@MainActor
@Observable
final class ChatViewModel {

    enum Phase: Equatable {
        case idle
        case loading(String)
        case ready
        case generating
        case failed(String)
    }

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

    var messages: [Message] = []
    var input: String = ""
    var phase: Phase = .idle

    var statusLine: String = ""

    var loadStatus: String = ""

    var warming: Bool = false

    var modelName: String = ""

    var loadedPath: String = ""

    var maxTokens: Int = 512

    @ObservationIgnored private var engine: CoreMLEngine?
    @ObservationIgnored private var history: [ChatTurn] = []
    @ObservationIgnored private var hasWarmedUp = false

    @ObservationIgnored var generation: Task<Void, Never>?

    init() {}

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

    func loadModel(path: String) async {
        guard canLoad else { return }
        let url = URL(fileURLWithPath: path, isDirectory: true)
        let manifestURL = url.appending(path: "manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path(percentEncoded: false)) else {
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
            loadedPath = path
            history = []
            messages = []
            hasWarmedUp = false
            warming = false
            statusLine = ""
            loadStatus = "Loaded. The first reply specializes GPU kernels (~40s)."
            phase = .ready
        } catch {
            engine = nil
            loadedPath = ""
            phase = .failed(String(describing: error))
        }
    }

    func unload() {
        guard canLoad else { return }
        generation?.cancel()
        generation = nil
        engine = nil
        history = []
        messages = []
        modelName = ""
        loadedPath = ""
        statusLine = ""
        loadStatus = ""
        warming = false
        hasWarmedUp = false
        phase = .idle
    }

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

    func stop() { generation?.cancel() }

    func reset() {
        guard let engine, canReset else { return }
        Task { await engine.resetConversation() }
        messages = []
        history = []
        statusLine = ""
        loadStatus = ""
    }

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

    private func finishStopped(userText: String, assistantText: String) {
        history.append(ChatTurn(role: .user, text: userText))
        history.append(ChatTurn(role: .assistant, text: assistantText))
        if let engine { Task { await engine.resetConversation() } }
        statusLine = "Stopped"
        phase = .ready
    }

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
