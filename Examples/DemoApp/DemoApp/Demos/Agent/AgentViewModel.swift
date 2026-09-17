import CoreMLBackend
import Foundation
import LLMCore
import Observation
#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

@MainActor
@Observable
final class AgentViewModel {

    struct Entry: Identifiable {
        enum Kind { case user, thinking, answer, toolCall, toolResult, failure, note }
        let id: UUID
        var kind: Kind
        var title: String
        var text: String

        var stats: String?

        init(id: UUID = UUID(), kind: Kind, title: String, text: String, stats: String? = nil) {
            self.id = id
            self.kind = kind
            self.title = title
            self.text = text
            self.stats = stats
        }
    }

    static let braveKeyDefaultsKey = "agent.braveAPIKey"
    static let effortDefaultsKey = "agent.reasoningEffort"
    static let notesFolderDefaultsKey = "agent.notesFolder"
    static let fileOpsDefaultsKey = "agent.allowFileOps"
    static let maxStepsDefaultsKey = "agent.maxSteps"
    static let pageBudgetDefaultsKey = "agent.pageBudget"

    nonisolated static let noAnswerText = "(no answer text)"
    nonisolated static let toolRunningText = "Running…"

    var persistsSettings = true

    var entries: [Entry] = []
    var input = ""
    var statusLine = ""
    var running = false
    var maxIterations = 8 {
        didSet { store(maxIterations, forKey: Self.maxStepsDefaultsKey) }
    }
    var maxNewTokens = 1_536
    var pageCharacterBudget = 10_000 {
        didSet { store(pageCharacterBudget, forKey: Self.pageBudgetDefaultsKey) }
    }

    var currentDate: String? = AgentPromptBuilder.today()

    private(set) var divergeCount = 0
    private(set) var capSteps = 0
    private(set) var lastFinishReason: FinishReason?

    private(set) var sessionID: String?
    @ObservationIgnored private var turnIndex = 0
    @ObservationIgnored private var modelName = ""

    var effort: AgentReasoningEffort {
        didSet { store(effort.rawValue, forKey: Self.effortDefaultsKey) }
    }

    var braveAPIKey: String {
        didSet { store(braveAPIKey, forKey: Self.braveKeyDefaultsKey) }
    }

    var notesFolderPath: String {
        didSet { store(notesFolderPath, forKey: Self.notesFolderDefaultsKey) }
    }

    var allowFileOperations: Bool {
        didSet { store(allowFileOperations, forKey: Self.fileOpsDefaultsKey) }
    }

    private func store(_ value: Any, forKey key: String) {
        guard persistsSettings else { return }
        UserDefaults.standard.set(value, forKey: key)
    }

    var notesDirectory: URL {
        notesFolderPath.isEmpty
            ? AgentTools.defaultNotesDirectory()
            : URL(fileURLWithPath: (notesFolderPath as NSString).expandingTildeInPath, isDirectory: true)
    }

    @ObservationIgnored private var identity: String?

    @ObservationIgnored private var seenPages: [String: String] = [:]

    @ObservationIgnored private var turns: [AgentTurn] = []
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var presenter = StreamPresenter()
    @ObservationIgnored private var displayTask: Task<Void, Never>?

    init() {
        let defaults = UserDefaults.standard
        effort = defaults.string(forKey: Self.effortDefaultsKey)
            .flatMap(AgentReasoningEffort.init(rawValue:)) ?? .low
        braveAPIKey = defaults.string(forKey: Self.braveKeyDefaultsKey) ?? ""
        notesFolderPath = defaults.string(forKey: Self.notesFolderDefaultsKey) ?? ""
        allowFileOperations = defaults.bool(forKey: Self.fileOpsDefaultsKey)
        if let steps = defaults.object(forKey: Self.maxStepsDefaultsKey) as? Int, steps > 0 {
            maxIterations = steps
        }
        if let budget = defaults.object(forKey: Self.pageBudgetDefaultsKey) as? Int, budget > 0 {
            pageCharacterBudget = budget
        }
    }

    var trimmedInput: String { input.trimmingCharacters(in: .whitespacesAndNewlines) }
    var canSend: Bool { !running && !trimmedInput.isEmpty }
    func canSend(chat: ChatViewModel) -> Bool {
        canSend && !chat.isGenerating && !chat.isLoading
    }
    var canReset: Bool { !running && !entries.isEmpty }
    var searchProvider: String { braveAPIKey.isEmpty ? "DuckDuckGo" : "Brave Search API" }

    @discardableResult
    func send(chat: ChatViewModel) -> Bool {
        guard canSend(chat: chat), let handle = chat.engineHandle else { return false }
        let text = trimmedInput
        input = ""
        if turns.isEmpty {
            sessionID = Self.newSessionID()
            currentDate = AgentPromptBuilder.today()
        }
        turnIndex += 1
        modelName = chat.modelName
        identity = handle.bundleFolder
            .flatMap { folder in LLMModels.all.first { $0.bundleFolderName == folder } }?.identity
        entries.append(Entry(kind: .user, title: "You", text: text))
        turns.append(.user(text))
        running = true
        lastFinishReason = nil
        chat.externallyBusy = true
        statusLine = "Step 1…"
        task = Task { [self] in
            await runLoop(handle: handle, chat: chat)
            chat.externallyBusy = false
            saveTranscript()
        }
        return true
    }

    func stop() { task?.cancel() }

    func reset(chat: ChatViewModel) {
        guard !running else { return }
        entries = []
        turns = []
        seenPages = [:]
        divergeCount = 0
        capSteps = 0
        sessionID = nil
        turnIndex = 0
        statusLine = ""
        if let engine = chat.engineHandle?.engine { Task { await engine.resetConversation() } }
    }

    private func runLoop(handle: ChatViewModel.EngineHandle, chat: ChatViewModel) async {
        let settings = AgentToolSettings(
            pageCharacterBudget: pageCharacterBudget, braveAPIKey: braveAPIKey,
            notesDirectory: notesDirectory, allowFileOperations: allowFileOperations)
        var step = 0
        while step < maxIterations {
            step += 1
            if Task.isCancelled { break }
            guard chat.engineHandle?.engine === handle.engine else {
                entries.append(Entry(
                    kind: .failure, title: "Generation failed", text: "Model was unloaded"))
                statusLine = "Stopped at step \(step): the model was unloaded."
                running = false
                return
            }
            statusLine = "Step \(step) of \(maxIterations): generating…"
            let entryID = UUID()
            entries.append(Entry(id: entryID, kind: .answer, title: "Assistant", text: ""))
            let prompt = AgentPromptBuilder.render(
                turns, effort: effort, currentDate: currentDate, identity: identity,
                fileOperations: allowFileOperations)
            let outcome = await stream(engine: handle.engine, prompt: prompt, entryID: entryID)
            switch outcome {
            case .cancelled(let raw):
                markStopped(entryID: entryID, raw: raw)
                await handle.engine.resetConversation()
                statusLine = "Stopped after \(step) step(s)."
                running = false
                return
            case .overflow(let promptTokens, let contextLength):
                guard recoverFromFullContext(
                    entryID: entryID, handle: handle, step: step,
                    promptTokens: promptTokens, contextLength: contextLength) else { return }
                continue
            case .failed(let reason):
                entries.removeAll { $0.id == entryID }
                entries.append(Entry(kind: .failure, title: "Generation failed", text: reason))
                statusLine = "Failed at step \(step)."
                logError(handle: handle, reason: reason, promptTokens: nil, contextLength: nil)
                await handle.engine.resetConversation()
                running = false
                return
            case .completed(let raw, let metrics):
                lastFinishReason = metrics?.finishReason
                if metrics?.finishReason == .contextFull {
                    guard recoverFromFullContext(
                        entryID: entryID, handle: handle, step: step,
                        promptTokens: metrics?.promptTokens, contextLength: nil) else { return }
                    continue
                }
                let parsed = AgentToolCallParser.parse(raw)
                let lastEntryID = present(parsed, entryID: entryID)
                turns.append(.assistant(
                    reasoning: parsed.reasoning, text: parsed.text, toolCalls: parsed.toolCalls))
                let divergesBefore = divergeCount
                warnIfRerenderDiverges(from: prompt, raw: raw)
                let context = AgentMetricsContext(
                    session: sessionID, turn: turnIndex, step: step, effort: effort.rawValue,
                    toolCalls: parsed.toolCalls.map(\.name),
                    rerenderDiverged: divergeCount > divergesBefore,
                    thinkingChars: parsed.reasoning.count, answerChars: parsed.text.count)
                let citations = parsed.toolCalls.isEmpty ? await checkCitations(parsed.text) : nil
                if let citations {
                    entries.append(Entry(kind: .note, title: "Citations", text: citations.note))
                }
                if let metrics {
                    if metrics.finishReason == .cap {
                        capSteps += 1
                        if parsed.toolCalls.isEmpty {
                            entries.append(Entry(
                                kind: .failure,
                                title: "Answer cut at the token limit (\(maxNewTokens))",
                                text: "The answer above stops where generation ran out of tokens. "
                                    + "Raise Max tokens or ask for something shorter."))
                        }
                    }
                    let stats = statsLine(step: step, metrics: metrics, calls: parsed.toolCalls.count)
                    statusLine = stats
                    if let lastEntryID, let index = entries.firstIndex(where: { $0.id == lastEntryID }) {
                        entries[index].stats = stats
                    }
                    MetricsLog.message(
                        metrics: metrics, modelID: handle.modelID, hfRevision: handle.hfRevision,
                        computeUnits: handle.computeUnits, bundleFolder: handle.bundleFolder,
                        modelLoadSeconds: nil, citationsTotal: citations?.total,
                        citationsSeen: citations?.seenCount,
                        citationMeanOverlap: citations?.meanOverlap, agent: context)
                }
                if parsed.toolCalls.isEmpty {
                    running = false
                    return
                }

                let resultIDs = parsed.toolCalls.map { call -> UUID in
                    let resultID = UUID()
                    entries.append(Entry(
                        id: resultID, kind: .toolResult, title: "\(call.name) result",
                        text: Self.toolRunningText))
                    return resultID
                }
                statusLine = "Step \(step): running "
                    + parsed.toolCalls.map(\.name).joined(separator: " + ") + "…"
                let outputs = await AgentTools.runAll(parsed.toolCalls, settings: settings)
                for ((resultID, call), output) in zip(zip(resultIDs, parsed.toolCalls), outputs) {
                    setText(id: resultID, output.text)
                    seenPages.merge(output.seen) { _, new in new }
                    MetricsLog.agentTool(
                        context: context, name: call.name,
                        arg: call.value("url") ?? call.value("query") ?? call.value("title"),
                        seconds: output.seconds, resultChars: output.text.count,
                        isError: output.text.hasPrefix("error:"),
                        modelID: handle.modelID, bundleFolder: handle.bundleFolder)
                }
                turns.append(.toolResponses(outputs.map(\.text)))
                if Task.isCancelled {
                    await handle.engine.resetConversation()
                    statusLine = "Stopped after \(step) step(s)."
                    running = false
                    return
                }
            }
        }
        if Task.isCancelled {
            await handle.engine.resetConversation()
            statusLine = "Stopped."
        } else {
            entries.append(Entry(
                kind: .failure, title: "Step limit reached",
                text: "The agent used all \(maxIterations) steps without finishing. "
                    + "Raise the limit in Settings or ask for something narrower."))
            statusLine = "Stopped at the \(maxIterations)-step limit."
        }
        running = false
    }

    private enum Outcome {
        case completed(String, GenerationMetrics?)
        case cancelled(String)
        case overflow(Int, Int)
        case failed(String)
    }

    private func markStopped(entryID: UUID, raw: String) {
        let visible = AgentToolCallParser.parse(raw).text
        guard let index = entries.firstIndex(where: { $0.id == entryID }) else { return }
        entries[index].kind = .failure
        entries[index].title = "Stopped"
        entries[index].text = visible.isEmpty ? "(stopped before any answer)" : visible
    }

    private func recoverFromFullContext(
        entryID: UUID, handle: ChatViewModel.EngineHandle, step: Int,
        promptTokens: Int?, contextLength: Int?
    ) -> Bool {
        entries.removeAll { $0.id == entryID }
        let droppedOldest = truncateOldestToolResponse()
        guard droppedOldest || halveLastToolResponse() else {
            let sizes = promptTokens.map { tokens in
                contextLength.map { "The conversation needs \(tokens) tokens but this bundle only has \($0). " }
                    ?? "The conversation needs \(tokens) tokens, more than this bundle can hold. "
            } ?? "The conversation no longer fits in this bundle's context window. "
            entries.append(Entry(
                kind: .failure, title: "Context window is full",
                text: sizes + "Lower the page budget or ask for something narrower."))
            statusLine = "Context overflow at step \(step)."
            running = false
            return false
        }
        statusLine = "Context full — "
            + (droppedOldest ? "dropped the oldest tool output" : "halved the newest tool output")
            + " and retried."
        logError(handle: handle, reason: "context overflow", promptTokens: promptTokens,
                 contextLength: contextLength)
        return true
    }

    private func stream(engine: CoreMLEngine, prompt: String, entryID: UUID) async -> Outcome {
        let config = GenerationConfig(
            maxNewTokens: maxNewTokens, temperature: 0, multiTokenPrediction: true)
        let request = GenerationRequest(
            prompt: "", config: config, history: [], reuseCache: true, rawPrompt: prompt)
        var raw = ""
        var metrics: GenerationMetrics?
        presenter.reset()
        startDisplayLoop(entryID: entryID)
        do {
            for try await event in engine.generate(request) {
                switch event {
                case .token(let chunk):
                    raw += chunk.text
                    presenter.append(chunk.text)
                case .finished(let done):
                    metrics = done
                default:
                    break
                }
            }
        } catch is CancellationError {
            flushDisplay(id: entryID)
            return .cancelled(raw)
        } catch {
            flushDisplay(id: entryID)
            if case LLMEngineError.contextOverflow(let promptTokens, let contextLength) = error {
                return .overflow(promptTokens, contextLength)
            }
            return .failed(String(describing: error))
        }
        flushDisplay(id: entryID)
        if Task.isCancelled { return .cancelled(raw) }
        return .completed(raw, metrics)
    }

    @discardableResult
    private func present(_ parsed: AgentToolCallParser.Output, entryID: UUID) -> UUID? {
        var replacement: [Entry] = []
        if !parsed.reasoning.isEmpty {
            replacement.append(Entry(kind: .thinking, title: "Thinking", text: parsed.reasoning))
        }
        if !parsed.text.isEmpty || parsed.toolCalls.isEmpty {
            replacement.append(Entry(
                id: entryID, kind: .answer, title: "Assistant",
                text: parsed.text.isEmpty ? Self.noAnswerText : parsed.text))
        }
        for call in parsed.toolCalls {
            replacement.append(Entry(kind: .toolCall, title: call.name, text: call.summary))
        }
        guard let index = entries.firstIndex(where: { $0.id == entryID }) else {
            entries.append(contentsOf: replacement)
            return replacement.last?.id
        }
        entries.replaceSubrange(index...index, with: replacement)
        return replacement.last?.id
    }

    private func warnIfRerenderDiverges(from prompt: String, raw: String) {
        let expected = prompt + raw
        let actual = AgentPromptBuilder.render(
            turns, effort: effort, currentDate: currentDate, identity: identity,
            fileOperations: allowFileOperations)
        guard !actual.hasPrefix(expected) else { return }
        divergeCount += 1
        let index = zip(actual, expected).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            ?? min(actual.count, expected.count)
        let window = { (text: String) in
            String(text.dropFirst(max(0, index - 40)).prefix(120)).debugDescription
        }
        FileHandle.standardError.write(Data(("[agent] re-render diverges at \(index): "
            + "got \(window(actual)) want \(window(expected))\n").utf8))
    }

    struct CitationCheck {
        var urls: [(url: String, seen: Bool, overlap: Double)] = []
        var total: Int { urls.count }
        var seenCount: Int { urls.count { $0.seen } }
        var meanOverlap: Double {
            urls.isEmpty ? 0 : urls.map(\.overlap).reduce(0, +) / Double(urls.count)
        }
        var note: String {
            String(format: "%d/%d cited URLs were retrieved in this task · mean 8-gram overlap %.2f",
                   seenCount, total, meanOverlap)
        }
    }

    nonisolated static let citationSourceLimit = 200_000

    private func checkCitations(_ answer: String) async -> CitationCheck? {
        let (body, urls) = Self.splitSources(answer)
        guard !urls.isEmpty else { return nil }
        var lookup: [String: String] = [:]
        for (url, text) in seenPages { lookup[Self.normalized(url)] = text }
        var check = CitationCheck()
        for url in urls {
            let source = lookup[Self.normalized(url)]
            var overlap = 0.0
            if let source {
                let clipped = String(source.prefix(Self.citationSourceLimit))
                overlap = await Task.detached { Self.gramOverlap(body, clipped) }.value
            }
            check.urls.append((url, source != nil, overlap))
            FileHandle.standardError.write(Data(String(
                format: "[cite] url=%@ seen=%@ overlap=%.2f\n",
                url, source == nil ? "no" : "yes", overlap).utf8))
        }
        return check
    }

    nonisolated static func splitSources(_ answer: String) -> (body: String, urls: [String]) {
        let lines = answer.split(separator: "\n", omittingEmptySubsequences: false)
        guard let start = lines.lastIndex(where: { line in
            let head = line.trimmingCharacters(in: CharacterSet(charactersIn: " \t*#-")).lowercased()
            return head.hasPrefix("sources") || head.hasPrefix("\u{51FA}\u{5178}")
                || head.hasPrefix("\u{53C2}\u{8003}")
        }) else { return (answer, []) }
        return (lines[..<start].joined(separator: "\n"), urls(in: lines[start...].joined(separator: "\n")))
    }

    nonisolated static func urls(in text: String) -> [String] {
        var found: [String] = []
        for match in text.matches(of: /https?:\/\/[^\s<>"'()\[\]]+/) {
            var url = String(match.0)
            while let last = url.last,
                  ".,;:\u{3001}\u{3002}\u{00BB}\u{201D}".contains(last) { url.removeLast() }
            if !url.isEmpty, !found.contains(url) { found.append(url) }
        }
        return found
    }

    nonisolated static func normalized(_ url: String) -> String {
        var text = url.lowercased()
        while text.hasSuffix("/") { text.removeLast() }
        return text
    }

    nonisolated static func gramOverlap(_ body: String, _ source: String, size: Int = 8) -> Double {
        func grams(_ text: String) -> [String] {
            let characters = Array(text.lowercased().filter { !$0.isWhitespace })
            guard characters.count >= size else { return [] }
            return (0...(characters.count - size)).map { String(characters[$0..<($0 + size)]) }
        }
        let needles = grams(body)
        guard !needles.isEmpty else { return 0 }
        let haystack = Set(grams(source))
        return Double(needles.count { haystack.contains($0) }) / Double(needles.count)
    }

    nonisolated static let protectedToolTurns = 2

    nonisolated static func truncatingOldestToolResponse(_ turns: [AgentTurn]) -> [AgentTurn]? {
        let toolTurns = turns.indices.filter {
            if case .toolResponses = turns[$0] { return true } else { return false }
        }
        for index in toolTurns.dropLast(protectedToolTurns) {
            guard case .toolResponses(let bodies) = turns[index],
                  bodies.contains(where: { $0 != "[truncated]" }) else { continue }
            var dropped = turns
            dropped[index] = .toolResponses(bodies.map { _ in "[truncated]" })
            return dropped
        }
        return nil
    }

    private func truncateOldestToolResponse() -> Bool {
        guard let dropped = Self.truncatingOldestToolResponse(turns) else { return false }
        turns = dropped
        return true
    }

    nonisolated static let minimumToolBody = 500

    nonisolated static func halvingLastToolResponse(_ turns: [AgentTurn]) -> [AgentTurn]? {
        let index = turns.lastIndex {
            if case .toolResponses = $0 { return true } else { return false }
        }
        guard let index, case .toolResponses(let bodies) = turns[index],
              bodies.contains(where: { $0.count > minimumToolBody }) else { return nil }
        var halved = turns
        halved[index] = .toolResponses(bodies.map { body in
            guard body.count > minimumToolBody else { return body }
            return String(body.prefix(max(minimumToolBody, body.count / 2)))
        })
        return halved
    }

    private func halveLastToolResponse() -> Bool {
        guard let halved = Self.halvingLastToolResponse(turns) else { return false }
        turns = halved
        return true
    }

    private func statsLine(step: Int, metrics: GenerationMetrics, calls: Int) -> String {
        var parts = [
            "step \(step)",
            String(format: "TTFT %.2fs", metrics.timeToFirstToken / .seconds(1)),
            String(format: "%.1f tok/s", metrics.decodeTokensPerSecond),
            "prompt \(metrics.promptTokens)"
        ]
        if metrics.reusedTokens > 0 { parts.append("reused \(metrics.reusedTokens)") }
        parts.append("\(metrics.generatedTokens) tok")
        parts.append(calls == 0 ? "answered" : "\(calls) tool call(s)")
        if let reason = metrics.finishReason { parts.append(reason.rawValue) }
        return parts.joined(separator: "  |  ")
    }

    private func logError(
        handle: ChatViewModel.EngineHandle, reason: String, promptTokens: Int?, contextLength: Int?
    ) {
        MetricsLog.error(
            phase: "agent", reason: reason, modelID: handle.modelID, hfRevision: handle.hfRevision,
            computeUnits: handle.computeUnits, bundleFolder: handle.bundleFolder,
            promptTokens: promptTokens, contextLength: contextLength)
    }

    private func setText(id: UUID, _ text: String) {
        if let index = entries.firstIndex(where: { $0.id == id }) { entries[index].text = text }
    }

    private func startDisplayLoop(entryID: UUID) {
        displayTask?.cancel()
        displayTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(33))
                guard let self, !Task.isCancelled else { return }
                if self.presenter.tick() { self.setText(id: entryID, self.presenter.displayed) }
            }
        }
    }

    private func flushDisplay(id: UUID) {
        displayTask?.cancel()
        displayTask = nil
        presenter.flush()
        setText(id: id, presenter.displayed)
    }
}

extension AgentViewModel {

    static func newSessionID() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }

    static func transcriptsDirectory() -> URL {
        ModelStorage.applicationSupportDirectory()
            .appending(path: "agent-transcripts", directoryHint: .isDirectory)
    }

    func transcriptMarkdown() -> String {
        func fence(_ text: String) -> String { "````\n\(text)\n````" }
        var lines = [
            "# Agent transcript",
            "",
            "- model: \(modelName.isEmpty ? "(none)" : modelName)",
            "- session: \(sessionID ?? "(none)")",
            "- effort: \(effort.rawValue)",
            "- max steps: \(maxIterations)",
            "- page budget: \(pageCharacterBudget)",
            "- search: \(searchProvider)",
            "- date: \(currentDate ?? "(none)")",
            "- app build: \(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?")",
            ""
        ]
        for entry in entries {
            switch entry.kind {
            case .user:
                lines += ["## You", "", entry.text, ""]
            case .thinking:
                lines += ["### Thinking (\(entry.text.count) chars)", "", fence(entry.text), ""]
            case .toolCall:
                lines += ["### Calling \(entry.title)", "", fence(entry.text), ""]
            case .toolResult:
                lines += ["### \(entry.title) (\(entry.text.count) chars)", "", fence(entry.text), ""]
            case .answer:
                lines += ["## Assistant", "", entry.text, ""]
            case .failure:
                lines += ["> **\(entry.title)**: \(entry.text)", ""]
            case .note:
                lines += ["_\(entry.title): \(entry.text)_", ""]
            }
            if let stats = entry.stats { lines += ["_\(stats)_", ""] }
        }
        return lines.joined(separator: "\n")
    }

    func copyTranscript() {
        let text = transcriptMarkdown()
        #if canImport(AppKit)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #elseif canImport(UIKit)
        UIPasteboard.general.string = text
        #endif
    }

    func saveTranscript() {
        guard let sessionID, !entries.isEmpty else { return }
        let directory = Self.transcriptsDirectory()
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data(transcriptMarkdown().utf8)
                .write(to: directory.appending(path: "\(sessionID).md"), options: .atomic)
        } catch {
            FileHandle.standardError.write(Data("[agent] transcript save failed: \(error)\n".utf8))
        }
    }
}
