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

    var localBundles: [URL] = []

    var maxTokens: Int = 512

    var speculative: Bool = true

    var kvStatus: String = ""

    @ObservationIgnored private var engine: CoreMLEngine?
    @ObservationIgnored private var history: [ChatTurn] = []
    @ObservationIgnored private var hasWarmedUp = false
    @ObservationIgnored private var lastCheckpointPrompt: String?

    @ObservationIgnored private var presenter = StreamPresenter()
    @ObservationIgnored private var displayTask: Task<Void, Never>?

    @ObservationIgnored var generation: Task<Void, Never>?

    init() {}

    var isGenerating: Bool { if case .generating = phase { return true } else { return false } }
    var isLoading: Bool { if case .loading = phase { return true } else { return false } }
    var isModelLoaded: Bool { engine != nil }
    var trimmedInput: String { input.trimmingCharacters(in: .whitespacesAndNewlines) }

    var canSend: Bool { isModelLoaded && !isGenerating && !isLoading && !trimmedInput.isEmpty }
    var canReset: Bool { isModelLoaded && !isGenerating && !isLoading && !messages.isEmpty }
    var canLoad: Bool { !isGenerating && !isLoading }
    var canCheckpoint: Bool { isModelLoaded && !isGenerating && !isLoading }

    private var checkpointURL: URL {
        let base = (try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appending(path: "CoreLLMCheckpoints/chat-kv", directoryHint: .isDirectory)
    }

    var phaseDescription: String {
        switch phase {
        case .idle: return "idle"
        case .loading(let s): return "loading: \(s)"
        case .ready: return "ready"
        case .generating: return "generating"
        case .failed(let s): return "failed: \(s)"
        }
    }

    func refreshLocalBundles() {
        let fm = FileManager.default
        let docs = (try? fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? URL(fileURLWithPath: NSHomeDirectory()).appending(path: "Documents")
        localBundles = LLMModels.supported().compactMap { model -> URL? in
            let url = docs.appending(path: model.bundleFolderName, directoryHint: .isDirectory)
            let manifest = url.appending(path: "manifest.json")
            return fm.fileExists(atPath: manifest.path(percentEncoded: false)) ? url : nil
        }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
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
            let preference = bundle.manifest.computeUnits
                .flatMap(ComputeUnitPreference.init(rawValue:)) ?? .cpuAndGPU
            phase = .loading("Compiling / loading Core ML models (\(preference.rawValue))…")
            let newEngine = CoreMLEngine()
            try await newEngine.load(bundle, options: LoadOptions(computeUnits: preference))
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
        stopDisplayLoop()
        presenter.reset()
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
        kvStatus = ""
        phase = .generating
        if !hasWarmedUp { warming = true }

        presenter.reset()
        startDisplayLoop(assistantID: assistantID)

        let config = GenerationConfig(maxNewTokens: maxTokens, temperature: 0, multiTokenPrediction: speculative)
        let request = GenerationRequest(prompt: userText, config: config, history: history, reuseCache: true)
        generation = Task { [self] in
            await consume(engine: engine, request: request, userText: userText, assistantID: assistantID)
        }
        return true
    }

    func stop() { generation?.cancel() }

    func reset() {
        guard let engine, canReset else { return }
        stopDisplayLoop()
        presenter.reset()
        Task { await engine.resetConversation() }
        messages = []
        history = []
        statusLine = ""
        loadStatus = ""
        kvStatus = ""
    }

    func saveKV() {
        guard let engine, canCheckpoint else { return }
        let prompt = trimmedInput.isEmpty ? (messages.last { $0.role == .user }?.text ?? "") : trimmedInput
        guard !prompt.isEmpty else {
            kvStatus = "Type a prompt in the box, then Save Checkpoint."
            return
        }
        let dir = checkpointURL
        lastCheckpointPrompt = prompt
        kvStatus = "Saving KV checkpoint…"
        Task { [self] in
            do {
                let info = try await engine.kvSave(to: dir, prompt: prompt)
                kvStatus = String(
                    format: "Saved KV: %d tokens, %.0f KB, resident prefill %@",
                    info.tokenCount, Double(info.fileBytes) / 1024, "\(info.residentPrefillWidths)")
            } catch {
                kvStatus = "KV save failed: \(error)"
            }
        }
    }

    func restoreKV() {
        guard let engine, canCheckpoint else { return }
        guard let prompt = lastCheckpointPrompt else {
            kvStatus = "Save a checkpoint first, then Restore + Continue."
            return
        }
        let dir = checkpointURL
        let useSpec = speculative
        kvStatus = "Restoring KV (no prefill)…"
        Task { [self] in
            do {
                let info = try await engine.kvRestoreAndContinue(
                    from: dir, verifyPrompt: prompt, maxNew: maxTokens, speculative: useSpec)
                let text = info.continuationText ?? ""
                messages.append(Message(role: .user, text: prompt))
                messages.append(Message(role: .assistant, text: text))
                let seconds = info.importSeconds ?? 0
                kvStatus = String(
                    format: "Restored KV in %.3fs, continued %d tokens (no prefill), resident prefill %@",
                    seconds, info.continuation?.count ?? 0, "\(info.residentPrefillWidths)")
            } catch {
                kvStatus = "KV restore failed: \(error)"
            }
        }
    }

    private func startDisplayLoop(assistantID: UUID) {
        displayTask?.cancel()
        displayTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(33))
                guard let self, !Task.isCancelled else { return }
                if self.presenter.tick() {
                    self.setAssistantText(id: assistantID, self.presenter.displayed)
                }
            }
        }
    }

    private func stopDisplayLoop() {
        displayTask?.cancel()
        displayTask = nil
    }

    private func flushDisplay(id: UUID) {
        stopDisplayLoop()
        presenter.flush()
        setAssistantText(id: id, presenter.displayed)
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
                    presenter.append(chunk.text)
                case .finished(let m):
                    metrics = m
                }
            }
        } catch is CancellationError {

            finishStopped(userText: userText, assistantText: assistantText, assistantID: assistantID)
            return
        } catch {
            warming = false
            flushDisplay(id: assistantID)
            if assistantText.isEmpty {
                setAssistantText(id: assistantID, "(generation failed)")
            }
            statusLine = "Error: \(error)"
            phase = .failed(String(describing: error))
            return
        }

        warming = false
        if Task.isCancelled {
            finishStopped(userText: userText, assistantText: assistantText, assistantID: assistantID)
            return
        }
        flushDisplay(id: assistantID)
        history.append(ChatTurn(role: .user, text: userText))
        history.append(ChatTurn(role: .assistant, text: assistantText))
        if let metrics { statusLine = statsLine(metrics) }
        phase = .ready
    }

    private func finishStopped(userText: String, assistantText: String, assistantID: UUID) {
        flushDisplay(id: assistantID)
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
            format: "%.1f tok/s  |  %.0f ms/tok  |  TTFT %.2fs  |  %d->%d tok",
            m.decodeTokensPerSecond, msPerTok, seconds(m.timeToFirstToken),
            m.promptTokens, m.generatedTokens)
        if let acc = m.draftAcceptanceRate {
            s += String(format: "  |  draft %.0f%%", acc * 100)
        }
        return s
    }
}
