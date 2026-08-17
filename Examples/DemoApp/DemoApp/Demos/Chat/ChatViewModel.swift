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
        var attachedImageURL: URL?
        var attachedAudioSeconds: Double?

        init(
            id: UUID = UUID(), role: Role, text: String, attachedImageURL: URL? = nil,
            attachedAudioSeconds: Double? = nil
        ) {
            self.id = id
            self.role = role
            self.text = text
            self.attachedImageURL = attachedImageURL
            self.attachedAudioSeconds = attachedAudioSeconds
        }
    }

    struct AttachedAudio: Equatable {
        var samples: [Float]
        var seconds: Double
    }

    var messages: [Message] = []
    var input: String = ""
    var phase: Phase = .idle

    var statusLine: String = ""

    var loadStatus: String = ""

    var isCompilingLongLoad: Bool = false

    var warming: Bool = false

    var modelName: String = ""

    var loadedPath: String = ""

    var localBundles: [URL] = []

    var maxTokens: Int = 0

    var speculative: Bool = SpeculationPolicy.defaultEnabled()

    var preparingSpeculation: Bool = false

    var kvStatus: String = ""

    var isConversationFull: Bool = false

    var supportsImageAttachment: Bool = false

    var attachedImageURL: URL?

    var supportsAudioAttachment: Bool = false

    var attachedAudio: AttachedAudio?

    var isRecording: Bool = false

    var recordingSeconds: Double = 0

    let maxRecordingSeconds: Double = CoreMLEngine.maxAudioSeconds

    @ObservationIgnored private var imageContextActive = false

    @ObservationIgnored private var audioContextActive = false

    @ObservationIgnored private lazy var recorder = AudioRecorder(maxSeconds: maxRecordingSeconds)
    @ObservationIgnored private var recordingTimer: Task<Void, Never>?

    @ObservationIgnored private var loadedContextLength: Int = 0

    @ObservationIgnored private var engine: CoreMLEngine?
    @ObservationIgnored private var history: [ChatTurn] = []
    @ObservationIgnored private var hasWarmedUp = false
    @ObservationIgnored private var lastCheckpointPrompt: String?
    @ObservationIgnored private var loadedModelID: String?
    @ObservationIgnored private var loadedRevision: String?
    @ObservationIgnored private var loadedComputeUnits: String?
    @ObservationIgnored private var loadedFolder: String?
    @ObservationIgnored private var loadedModelLoadSeconds: Double?

    @ObservationIgnored private var presenter = StreamPresenter()
    @ObservationIgnored private var displayTask: Task<Void, Never>?
    @ObservationIgnored private var loadProgressTimer: Task<Void, Never>?
    @ObservationIgnored private var hasAttemptedAutoLoad = false

    @ObservationIgnored var generation: Task<Void, Never>?

    init() {}

    struct EngineHandle {
        let engine: CoreMLEngine
        let modelID: String?
        let hfRevision: String?
        let computeUnits: String?
        let bundleFolder: String?
        let speculative: Bool
    }

    var engineHandle: EngineHandle? {
        guard let engine else { return nil }
        return EngineHandle(
            engine: engine, modelID: loadedModelID, hfRevision: loadedRevision,
            computeUnits: loadedComputeUnits, bundleFolder: loadedFolder, speculative: speculative)
    }

    var isGenerating: Bool { if case .generating = phase { return true } else { return false } }
    var isLoading: Bool { if case .loading = phase { return true } else { return false } }
    var isModelLoaded: Bool { engine != nil }
    var trimmedInput: String { input.trimmingCharacters(in: .whitespacesAndNewlines) }

    var canSend: Bool {
        isModelLoaded && !isGenerating && !isLoading && !preparingSpeculation && !isRecording
            && (!trimmedInput.isEmpty || attachedAudio != nil) && !isConversationFull
    }
    var canReset: Bool { isModelLoaded && !isGenerating && !isLoading && !messages.isEmpty }

    var conversationFullMessage: String {
        if loadedContextLength > 0 {
            return "Context window is full (\(loadedContextLength.formatted()) tokens). "
                + "Reset to start a new conversation."
        }
        return "Context window is full. Reset to start a new conversation."
    }
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

    var loadingMessage: String {
        isCompilingLongLoad
            ? "Compiling for the Neural Engine — one-time on this device (about 90 s). "
                + "Future loads take a few seconds."
            : "Loading model…"
    }

    func refreshLocalBundles() {
        localBundles = LLMModels.supported()
            .compactMap { ModelStorage.locateBundle(folderName: $0.bundleFolderName) }
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
        let startedAt = Date()
        startLoadProgressTimer()
        do {
            let bundle = try ModelBundle(contentsOf: url)
            modelName = bundle.manifest.name
            loadedContextLength = bundle.manifest.contextLength
            let preference = bundle.manifest.computeUnits
                .flatMap(ComputeUnitPreference.init(rawValue:)) ?? .cpuAndGPU
            phase = .loading("Compiling / loading Core ML models (\(preference.rawValue))…")
            let newEngine = CoreMLEngine()
            try await newEngine.load(
                bundle, options: LoadOptions(computeUnits: preference, preloadSpeculation: speculative))
            engine = newEngine
            supportsImageAttachment = await newEngine.supportsImageInput()
            supportsAudioAttachment = await newEngine.supportsAudioInput()
            attachedImageURL = nil
            imageContextActive = false
            discardRecording()
            loadedPath = path
            loadedFolder = url.lastPathComponent
            let catalogModel = LLMModels.all.first { $0.bundleFolderName == url.lastPathComponent }
            loadedModelID = catalogModel?.id ?? bundle.manifest.name
            loadedRevision = catalogModel?.hfRevision
            loadedComputeUnits = preference.rawValue
            loadedModelLoadSeconds = nil
            history = []
            messages = []
            hasWarmedUp = false
            warming = false
            isConversationFull = false
            statusLine = ""
            stopLoadProgressTimer()
            let loadSeconds = Date().timeIntervalSince(startedAt)
            loadStatus = loadSeconds >= 30
                ? "Loaded (compiled for this device — future loads take a few seconds)."
                : "Loaded."
            rememberLastLoadedBundle(folderName: url.lastPathComponent)
            phase = .ready
        } catch {
            stopLoadProgressTimer()
            engine = nil
            supportsImageAttachment = false
            supportsAudioAttachment = false
            attachedImageURL = nil
            imageContextActive = false
            discardRecording()
            loadedPath = ""
            phase = .failed(String(describing: error))
        }
    }

    func autoLoadLastBundleOnce() async {
        guard !hasAttemptedAutoLoad else { return }
        hasAttemptedAutoLoad = true
        guard !isModelLoaded, canLoad else { return }
        let defaults = UserDefaults.standard
        guard let folderName = defaults.string(forKey: Self.lastLoadedBundleDefaultsKey),
              !folderName.isEmpty else { return }
        guard let url = ModelStorage.locateBundle(folderName: folderName) else {
            defaults.removeObject(forKey: Self.lastLoadedBundleDefaultsKey)
            return
        }
        await loadModel(path: url.path(percentEncoded: false))
    }

    private static let lastLoadedBundleDefaultsKey = "chat.lastLoadedBundleFolderName"

    private func rememberLastLoadedBundle(folderName: String) {
        UserDefaults.standard.set(folderName, forKey: Self.lastLoadedBundleDefaultsKey)
    }

    private func startLoadProgressTimer() {
        isCompilingLongLoad = false
        loadProgressTimer?.cancel()
        loadProgressTimer = Task { [weak self] in
            try? await Task.sleep(for: .seconds(10))
            guard let self, !Task.isCancelled, self.isLoading else { return }
            self.isCompilingLongLoad = true
        }
    }

    private func stopLoadProgressTimer() {
        loadProgressTimer?.cancel()
        loadProgressTimer = nil
        isCompilingLongLoad = false
    }

    func unload() {
        guard canLoad else { return }
        generation?.cancel()
        generation = nil
        stopLoadProgressTimer()
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
        isConversationFull = false
        supportsImageAttachment = false
        supportsAudioAttachment = false
        attachedImageURL = nil
        imageContextActive = false
        discardRecording()
        loadedContextLength = 0
        phase = .idle
    }

    func attachImage(_ url: URL) {
        guard supportsImageAttachment else { return }
        attachedImageURL = url
    }

    func attachImage(data: Data, fileExtension: String) {
        guard supportsImageAttachment, !data.isEmpty else { return }
        let name = "picked-\(UUID().uuidString).\(fileExtension)"
        let url = FileManager.default.temporaryDirectory.appending(path: name)
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            statusLine = "Cannot stage the picked image: \(error)"
            return
        }
        attachedImageURL = url
    }

    func clearAttachedImage() { attachedImageURL = nil }

    func attachAudio(samples: [Float]) {
        guard supportsAudioAttachment, !samples.isEmpty else { return }
        attachedAudio = AttachedAudio(
            samples: samples, seconds: AudioFileLoader.seconds(sampleCount: samples.count))
    }

    func clearAttachedAudio() { attachedAudio = nil }

    var recordingLabel: String {
        String(format: "%.0f / %.0f s", recordingSeconds, maxRecordingSeconds)
    }

    func toggleRecording() {
        if isRecording {
            finishRecording()
            return
        }
        guard supportsAudioAttachment, !isGenerating, !isLoading else { return }
        statusLine = "Waiting for microphone permission…"
        Task { [self] in
            guard await AudioRecorder.requestPermission() else {
                statusLine = AudioRecorder.permissionDeniedMessage
                return
            }
            do {
                try recorder.start()
            } catch {
                statusLine = "Recording failed: \(error)"
                return
            }
            attachedAudio = nil
            recordingSeconds = 0
            isRecording = true
            statusLine = "Recording. Press the microphone again to stop."
            recordingTimer = Task { [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(100))
                    guard let self, !Task.isCancelled, self.isRecording else { return }
                    self.recordingSeconds = AudioFileLoader.seconds(
                        sampleCount: self.recorder.sampleCount)
                    if self.recorder.isFull {
                        self.finishRecording()
                        return
                    }
                }
            }
        }
    }

    private func finishRecording() {
        recordingTimer?.cancel()
        recordingTimer = nil
        let samples = recorder.stop()
        isRecording = false
        recordingSeconds = AudioFileLoader.seconds(sampleCount: samples.count)
        if samples.isEmpty {
            statusLine = "Nothing was recorded."
        } else {
            attachedAudio = AttachedAudio(samples: samples, seconds: recordingSeconds)
            statusLine = ""
        }
    }

    private func discardRecording() {
        recordingTimer?.cancel()
        recordingTimer = nil
        if isRecording { recorder.stop() }
        isRecording = false
        recordingSeconds = 0
        attachedAudio = nil
        audioContextActive = false
    }

    @discardableResult
    func send() -> Bool {
        guard canSend, let engine else { return false }
        let userText = trimmedInput
        let imageURL = attachedImageURL
        let audio = attachedAudio
        input = ""
        attachedImageURL = nil
        attachedAudio = nil
        messages.append(Message(
            role: .user, text: userText, attachedImageURL: imageURL,
            attachedAudioSeconds: audio?.seconds))
        let assistantID = UUID()
        messages.append(Message(id: assistantID, role: .assistant, text: ""))
        statusLine = ""
        kvStatus = ""
        phase = .generating
        if !hasWarmedUp { warming = true }

        presenter.reset()
        startDisplayLoop(assistantID: assistantID)

        if audio != nil { imageContextActive = false }
        if imageURL != nil { audioContextActive = false }

        if audio != nil || (audioContextActive && imageURL == nil) {
            let instruction = userText.isEmpty
                ? CoreMLEngine.defaultTranscriptionInstruction : userText
            let useSpeculation = speculative
            let cap = maxTokens
            generation = Task { [self] in
                await consumeAudioTurn(
                    engine: engine, audio: audio, instruction: instruction, maxNew: cap,
                    speculative: useSpeculation, userText: userText, assistantID: assistantID)
            }
            return true
        }

        if imageURL != nil || imageContextActive {
            let useSpeculation = speculative
            let cap = maxTokens
            generation = Task { [self] in
                await consumeImageTurn(
                    engine: engine, imageURL: imageURL, question: userText, maxNew: cap,
                    speculative: useSpeculation, userText: userText, assistantID: assistantID)
            }
            return true
        }

        let config = GenerationConfig(maxNewTokens: maxTokens, temperature: 0, multiTokenPrediction: speculative)
        let request = GenerationRequest(prompt: userText, config: config, history: history, reuseCache: true)
        generation = Task { [self] in
            await consume(engine: engine, request: request, userText: userText, assistantID: assistantID)
        }
        return true
    }

    func stop() { generation?.cancel() }

    func speculationToggleChanged(to enabled: Bool) {
        guard enabled, let engine, isModelLoaded, !isGenerating, !isLoading, !preparingSpeculation else { return }
        Task { [self] in
            if await engine.speculationLoaded { return }
            preparingSpeculation = true
            statusLine = "Enabling speculation: one-time verify load, a few seconds…"
            do {
                try await engine.prepareSpeculation()
                statusLine = "Speculation ready."
            } catch {
                statusLine = "Speculation load failed: \(error)"
            }
            preparingSpeculation = false
        }
    }

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
        isConversationFull = false
        attachedImageURL = nil
        imageContextActive = false
        discardRecording()
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
                MetricsLog.kv(
                    info: info, op: "save", modelID: loadedModelID, hfRevision: loadedRevision,
                    computeUnits: loadedComputeUnits, bundleFolder: loadedFolder)
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
                MetricsLog.kv(
                    info: info, op: "restore", modelID: loadedModelID, hfRevision: loadedRevision,
                    computeUnits: loadedComputeUnits, bundleFolder: loadedFolder)
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
                    loadedModelLoadSeconds = seconds(m.duration)
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
            if case LLMEngineError.contextOverflow(let promptTokens, let contextLength) = error {
                isConversationFull = true
                if assistantText.isEmpty {
                    setAssistantText(id: assistantID, conversationFullMessage)
                }
                statusLine = conversationFullMessage
                phase = .ready
                MetricsLog.error(
                    phase: "preflight", reason: "context overflow",
                    modelID: loadedModelID, hfRevision: loadedRevision,
                    computeUnits: loadedComputeUnits, bundleFolder: loadedFolder,
                    promptTokens: promptTokens, contextLength: contextLength)
            } else {
                if assistantText.isEmpty {
                    setAssistantText(id: assistantID, "(generation failed)")
                }
                statusLine = "Error: \(error)"
                phase = .failed(String(describing: error))
                MetricsLog.error(
                    phase: "generation", reason: String(describing: error),
                    modelID: loadedModelID, hfRevision: loadedRevision,
                    computeUnits: loadedComputeUnits, bundleFolder: loadedFolder,
                    promptTokens: nil, contextLength: loadedContextLength > 0 ? loadedContextLength : nil)
            }
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
        if let metrics {
            statusLine = statsLine(metrics)
            MetricsLog.message(
                metrics: metrics, modelID: loadedModelID, hfRevision: loadedRevision,
                computeUnits: loadedComputeUnits, bundleFolder: loadedFolder,
                modelLoadSeconds: loadedModelLoadSeconds)
            loadedModelLoadSeconds = nil
            if metrics.finishReason == .contextFull { isConversationFull = true }
        }
        phase = .ready
    }

    private func consumeImageTurn(
        engine: CoreMLEngine,
        imageURL: URL?,
        question: String,
        maxNew: Int,
        speculative useSpeculation: Bool,
        userText: String,
        assistantID: UUID
    ) async {
        do {
            let info: VLMGenerationInfo
            if let imageURL {
                info = try await engine.generateWithImage(
                    imageURL: imageURL, question: question, maxNew: maxNew, speculative: useSpeculation)
            } else {
                info = try await engine.continueWithImageContext(
                    question: question, maxNew: maxNew, speculative: useSpeculation)
            }
            warming = false
            hasWarmedUp = true
            presenter.append(info.text)
            if Task.isCancelled {
                imageContextActive = false
                finishStopped(userText: userText, assistantText: info.text, assistantID: assistantID)
                return
            }
            imageContextActive = true
            flushDisplay(id: assistantID)
            history.append(ChatTurn(role: .user, text: userText))
            history.append(ChatTurn(role: .assistant, text: info.text))
            statusLine = imageStatsLine(info)
            phase = .ready
        } catch is CancellationError {
            warming = false
            imageContextActive = false
            finishStopped(
                userText: userText, assistantText: presenter.displayed, assistantID: assistantID)
        } catch {
            warming = false
            flushDisplay(id: assistantID)
            let failed = presenter.displayed.isEmpty
            if case LLMEngineError.contextOverflow(let promptTokens, let contextLength) = error {
                isConversationFull = true
                if failed { setAssistantText(id: assistantID, conversationFullMessage) }
                statusLine = conversationFullMessage
                phase = .ready
                MetricsLog.error(
                    phase: "preflight", reason: "context overflow",
                    modelID: loadedModelID, hfRevision: loadedRevision,
                    computeUnits: loadedComputeUnits, bundleFolder: loadedFolder,
                    promptTokens: promptTokens, contextLength: contextLength)
            } else {
                if failed { setAssistantText(id: assistantID, "(generation failed)") }
                statusLine = "Error: \(error)"
                phase = .failed(String(describing: error))
                MetricsLog.error(
                    phase: "generation", reason: String(describing: error),
                    modelID: loadedModelID, hfRevision: loadedRevision,
                    computeUnits: loadedComputeUnits, bundleFolder: loadedFolder,
                    promptTokens: nil, contextLength: loadedContextLength > 0 ? loadedContextLength : nil)
            }
        }
    }

    private func consumeAudioTurn(
        engine: CoreMLEngine,
        audio: AttachedAudio?,
        instruction: String,
        maxNew: Int,
        speculative useSpeculation: Bool,
        userText: String,
        assistantID: UUID
    ) async {
        do {
            let info: ASRGenerationInfo
            if let audio {
                info = try await engine.generateWithAudio(
                    samples: audio.samples, instruction: instruction, maxNew: maxNew,
                    speculative: useSpeculation)
            } else {
                info = try await engine.continueWithAudioContext(
                    question: instruction, maxNew: maxNew, speculative: useSpeculation)
            }
            warming = false
            hasWarmedUp = true
            presenter.append(info.text)
            if Task.isCancelled {
                audioContextActive = false
                finishStopped(userText: userText, assistantText: info.text, assistantID: assistantID)
                return
            }
            audioContextActive = true
            flushDisplay(id: assistantID)
            history.append(ChatTurn(role: .user, text: instruction))
            history.append(ChatTurn(role: .assistant, text: info.text))
            statusLine = audioStatsLine(info)
            phase = .ready
        } catch is CancellationError {
            warming = false
            audioContextActive = false
            finishStopped(
                userText: userText, assistantText: presenter.displayed, assistantID: assistantID)
        } catch {
            warming = false
            flushDisplay(id: assistantID)
            let failed = presenter.displayed.isEmpty
            if case LLMEngineError.contextOverflow(let promptTokens, let contextLength) = error {
                isConversationFull = true
                if failed { setAssistantText(id: assistantID, conversationFullMessage) }
                statusLine = conversationFullMessage
                phase = .ready
                MetricsLog.error(
                    phase: "preflight", reason: "context overflow",
                    modelID: loadedModelID, hfRevision: loadedRevision,
                    computeUnits: loadedComputeUnits, bundleFolder: loadedFolder,
                    promptTokens: promptTokens, contextLength: contextLength)
            } else {
                if failed { setAssistantText(id: assistantID, "(generation failed)") }
                statusLine = "Error: \(error)"
                phase = .failed(String(describing: error))
                MetricsLog.error(
                    phase: "generation", reason: String(describing: error),
                    modelID: loadedModelID, hfRevision: loadedRevision,
                    computeUnits: loadedComputeUnits, bundleFolder: loadedFolder,
                    promptTokens: nil, contextLength: loadedContextLength > 0 ? loadedContextLength : nil)
            }
        }
    }

    private func audioStatsLine(_ info: ASRGenerationInfo) -> String {
        var parts: [String] = []
        if info.audioSeconds > 0 { parts.append(String(format: "audio %.1fs", info.audioSeconds)) }
        if info.audioEncodeSeconds > 0 {
            parts.append(String(format: "encode %.2fs", info.audioEncodeSeconds))
        }
        parts.append(String(format: "TTFT %.2fs", info.audioEncodeSeconds + info.prefillSeconds))
        parts.append(String(format: "%.1f tok/s", info.decodeTokensPerSecond))
        parts.append("prompt \(info.promptTokens)")
        if info.audioRows > 0 { parts.append("audio \(info.audioRows) rows") }
        parts.append("\(info.generatedTokens) tok")
        if let mb = info.peakMemoryBytes { parts.append(String(format: "mem %.0fMB", Double(mb) / 1_048_576)) }
        return parts.joined(separator: "  |  ")
    }

    private func imageStatsLine(_ info: VLMGenerationInfo) -> String {
        var parts: [String] = []
        if info.visionEncodeSeconds > 0 {
            parts.append(String(format: "vision %.2fs", info.visionEncodeSeconds))
        }
        parts.append(String(format: "TTFT %.2fs", info.visionEncodeSeconds + info.prefillSeconds))
        parts.append(String(format: "%.1f tok/s", info.decodeTokensPerSecond))
        parts.append("prompt \(info.promptTokens)")
        if info.imageRows > 0 { parts.append("image \(info.imageRows) rows") }
        parts.append("\(info.generatedTokens) tok")
        if let mb = info.peakMemoryBytes { parts.append(String(format: "mem %.0fMB", Double(mb) / 1_048_576)) }
        return parts.joined(separator: "  |  ")
    }

    private func finishStopped(userText: String, assistantText: String, assistantID: UUID) {
        flushDisplay(id: assistantID)
        history.append(ChatTurn(role: .user, text: userText))
        history.append(ChatTurn(role: .assistant, text: assistantText))
        if let engine { Task { await engine.resetConversation() } }
        MetricsLog.cancelledMessage(
            modelID: loadedModelID, hfRevision: loadedRevision,
            computeUnits: loadedComputeUnits, bundleFolder: loadedFolder)
        statusLine = "Stopped"
        phase = .ready
    }

    private func setAssistantText(id: UUID, _ text: String) {
        if let i = messages.firstIndex(where: { $0.id == id }) { messages[i].text = text }
    }

    private func seconds(_ d: Duration) -> Double { d / .seconds(1) }

    private func statsLine(_ m: GenerationMetrics) -> String {
        var parts = [
            String(format: "TTFT %.2fs", seconds(m.timeToFirstToken)),
            String(format: "%.1f tok/s", m.decodeTokensPerSecond),
        ]
        if m.reusedTokens > 0 {
            parts.append("prompt \(m.promptTokens) (+reused \(m.reusedTokens))")
        } else {
            parts.append("prompt \(m.promptTokens)")
        }
        parts.append("\(m.generatedTokens) tok")
        if let acc = m.draftAcceptanceRate { parts.append(String(format: "draft %.0f%%", acc * 100)) }
        if let mb = m.peakMemoryBytes { parts.append(String(format: "mem %.0fMB", Double(mb) / 1_048_576)) }
        if let reason = m.finishReason { parts.append(reason.rawValue) }
        return parts.joined(separator: "  |  ")
    }
}
