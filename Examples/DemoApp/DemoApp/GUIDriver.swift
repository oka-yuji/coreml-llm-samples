#if os(macOS)
import AppKit
import Foundation
import LLMCore
import ScreenCaptureKit

@MainActor
final class GUIDriver {

    static let shared = GUIDriver()

    struct StepResult: Encodable {
        let step: Int
        let title: String
        let pass: Bool
        let detail: String
        let screenshot: String?
        let seconds: Double
    }

    static func argument(_ flag: String) -> String? {
        let args = CommandLine.arguments
        if let index = args.firstIndex(of: flag), index + 1 < args.count { return args[index + 1] }
        return args.first { $0.hasPrefix(flag + "=") }.map { String($0.dropFirst(flag.count + 1)) }
    }

    static var outputDirectory: URL? {
        argument("--gui-e2e").map { URL(fileURLWithPath: $0, isDirectory: true) }
    }

    static var isRequested: Bool { outputDirectory != nil }

    static var modelFolder: String { argument("--model") ?? "qwen38-27b-agent" }

    private var attached = false
    private var results: [StepResult] = []
    private var lines: [String] = []
    private var sessions: [String] = []
    private var outDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)

    private let searchTask = "Core ML \u{306E}\u{6700}\u{65B0}\u{60C5}\u{5831}(2026 \u{5E74})"
        + "\u{3092}\u{691C}\u{7D22}\u{3057}\u{3066} 3 \u{884C}\u{3067}\u{8981}\u{7D04}\u{3057}\u{3066}"

    func attach(
        navigator: DemoNavigator, chat: ChatViewModel, models: ModelsViewModel, agent: AgentViewModel
    ) {
        guard let directory = Self.outputDirectory, !attached else { return }
        attached = true
        outDir = directory
        agent.persistsSettings = false
        agent.allowFileOperations = true
        agent.effort = .low
        Task { await run(navigator: navigator, chat: chat, models: models, agent: agent) }
    }

    private func note(_ text: String) {
        let line = "[gui] " + text
        lines.append(line)
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    private func settle() async { try? await Task.sleep(for: .milliseconds(400)) }

    private func window() -> NSWindow? { NSApp.windows.first { $0.isVisible } }

    private func wait(seconds: Double, until condition: () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition() {
            if Date() > deadline { return false }
            try? await Task.sleep(for: .milliseconds(250))
        }
        return true
    }

    private static func head(_ text: String, _ limit: Int = 300) -> String {
        text.count <= limit ? text : String(text.prefix(limit)) + "\u{2026}"
    }

    private static func uniqueColors(_ png: Data) -> Int {
        guard let rep = NSBitmapImageRep(data: png) else { return -1 }
        var seen = Set<UInt32>()
        var y = 0
        while y < rep.pixelsHigh {
            var x = 0
            while x < rep.pixelsWide {
                if let color = rep.colorAt(x: x, y: y) {
                    let red = UInt32(color.redComponent * 255)
                    let green = UInt32(color.greenComponent * 255)
                    let blue = UInt32(color.blueComponent * 255)
                    seen.insert(red << 16 | green << 8 | blue)
                }
                x += 8
            }
            y += 8
        }
        return seen.count
    }

    private static func png(_ image: CGImage) -> Data? {
        NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
    }

    private func captureWithScreenCaptureKit(_ window: NSWindow) async -> Data? {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true)
            let id = CGWindowID(window.windowNumber)
            guard let target = content.windows.first(where: { $0.windowID == id }) else {
                note("capture: window \(id) is not in the shareable list")
                return nil
            }
            let configuration = SCStreamConfiguration()
            let scale = window.backingScaleFactor
            configuration.width = Int(target.frame.width * scale)
            configuration.height = Int(target.frame.height * scale)
            configuration.showsCursor = false
            let image = try await SCScreenshotManager.captureImage(
                contentFilter: SCContentFilter(desktopIndependentWindow: target),
                configuration: configuration)
            return Self.png(image)
        } catch {
            note("capture: ScreenCaptureKit failed: \(error)")
            return nil
        }
    }

    private static func captureWithCacheDisplay(_ window: NSWindow) -> Data? {
        guard let view = window.contentView,
              let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return nil }
        view.cacheDisplay(in: view.bounds, to: rep)
        return rep.representation(using: .png, properties: [:])
    }

    private func shoot(_ index: Int, _ slug: String) async -> (name: String?, detail: String) {
        guard let window = window() else { return (nil, "no visible window") }
        var method = "screenCaptureKit"
        var data = await captureWithScreenCaptureKit(window)
        if data == nil {
            method = "cacheDisplay"
            data = Self.captureWithCacheDisplay(window)
        }
        guard let data else { return (nil, "capture failed") }
        let name = String(format: "%02d-%@.png", index, slug)
        do {
            try data.write(to: outDir.appending(path: name), options: .atomic)
        } catch {
            return (nil, "write failed: \(error)")
        }
        let colors = Self.uniqueColors(data)
        let detail = "\(method) \(data.count)B colors=\(colors)"
        note("shot \(name) \(detail)")
        return (name, detail)
    }

    private func step(
        _ index: Int, _ slug: String, _ title: String, screenshot: Bool = true,
        _ body: () async -> (Bool, String)
    ) async {
        note("step \(index) \(slug) start")
        let started = Date()
        let (pass, detail) = await body()
        await settle()
        var shot: String?
        var shotDetail = ""
        if screenshot {
            let taken = await shoot(index, slug)
            shot = taken.name
            shotDetail = " | shot: " + taken.detail
        }
        let seconds = Date().timeIntervalSince(started)
        results.append(StepResult(
            step: index, title: title, pass: pass, detail: detail + shotDetail,
            screenshot: shot, seconds: (seconds * 100).rounded() / 100))
        note(String(format: "step %d %@ %@ %.1fs %@", index, slug, pass ? "PASS" : "FAIL",
                    seconds, Self.head(detail, 400)))
    }

    private func bundlePath(_ chat: ChatViewModel) -> String? {
        if !chat.loadedPath.isEmpty { return chat.loadedPath }
        return ModelStorage.locateBundle(folderName: Self.modelFolder)?.path(percentEncoded: false)
    }

    private func ask(
        _ agent: AgentViewModel, _ chat: ChatViewModel, _ text: String, timeout: Double = 600
    ) async -> Bool {
        agent.input = text
        guard agent.send(chat: chat) else {
            note("send refused: running=\(agent.running) loaded=\(chat.isModelLoaded)")
            return false
        }
        if let id = agent.sessionID, !sessions.contains(id) { sessions.append(id) }
        return await wait(seconds: timeout) { !agent.running }
    }

    private func newEntries(_ agent: AgentViewModel, from mark: Int) -> [AgentViewModel.Entry] {
        guard mark < agent.entries.count else { return [] }
        return Array(agent.entries[mark...])
    }

    private func run(
        navigator: DemoNavigator, chat: ChatViewModel, models: ModelsViewModel, agent: AgentViewModel
    ) async {
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        note("out=\(outDir.path(percentEncoded: false))")
        note("effort=\(agent.effort.rawValue) steps=\(agent.maxIterations) "
            + "tokens=\(agent.maxNewTokens) budget=\(agent.pageCharacterBudget) "
            + "search=\(agent.searchProvider) fileOps=\(agent.allowFileOperations) "
            + "notes=\(agent.notesDirectory.path(percentEncoded: false))")
        NSApp.activate(ignoringOtherApps: true)
        _ = await wait(seconds: 30) { self.window() != nil }
        await settle()

        let folder = Self.modelFolder
        await step(1, "models-row", "Models list shows the \(folder) bundle as downloaded") {
            navigator.selection = .models
            await self.settle()
            models.refresh()
            await self.settle()
            let row = models.rows.first { $0.model.bundleFolderName == folder && $0.isDownloaded }
            guard let row else {
                return (false, "no downloaded row for \(folder) among "
                    + "\(models.rows.map(\.model.id))")
            }
            return (true, "id=\(row.model.id) folder=\(row.model.bundleFolderName) "
                + "downloaded=\(row.isDownloaded) diskSize=\(row.diskSize)")
        }

        await step(2, "agent-loading", "Agent screen while the bundle is loading") {
            navigator.selection = .agent
            await self.settle()
            _ = await self.wait(seconds: 20) { chat.isLoading || chat.isModelLoaded }
            if !chat.isLoading {
                let path = self.bundlePath(chat)
                guard let path else { return (false, "no bundle to load") }
                if chat.isModelLoaded {
                    self.note("already loaded; unloading to capture the loading state")
                    chat.unload()
                    await self.settle()
                }
                Task { await chat.loadModel(path: path) }
                _ = await self.wait(seconds: 30) { chat.isLoading }
            }
            return (chat.isLoading, "isLoading=\(chat.isLoading) phase=\(chat.phaseDescription) "
                + "loadingMessage=\(chat.loadingMessage)")
        }

        await step(3, "agent-ready", "Bundle is loaded and file operations are enabled") {
            let loaded = await self.wait(seconds: 240) { chat.isModelLoaded && !chat.isLoading }
            return (loaded && agent.allowFileOperations,
                    "loaded=\(chat.isModelLoaded) name=\(chat.modelName) "
                    + "path=\(chat.loadedPath) fileOps=\(agent.allowFileOperations) "
                    + "chatML=\(AgentView.isChatMLBundle(chat.loadedPath)) status=\(chat.loadStatus)")
        }

        await step(4, "identity", "The agent names its own model") {
            let mark = agent.entries.count
            let done = await self.ask(
                agent, chat,
                "\u{3042}\u{306A}\u{305F}\u{306F}\u{4F55}\u{306E}\u{30E2}\u{30C7}\u{30EB}\u{3067}\u{3059}\u{304B}")
            let answer = self.newEntries(agent, from: mark).last { $0.kind == .answer }?.text ?? ""
            return (done && answer.contains("Qwen3.8"),
                    "finished=\(done) answer=\(Self.head(answer))")
        }

        await step(5, "task1-tense", "Search task answers with a Sources section") {
            let mark = agent.entries.count
            let done = await self.ask(agent, chat, self.searchTask)
            let fresh = self.newEntries(agent, from: mark)
            let answer = fresh.last { $0.kind == .answer }?.text ?? ""
            let stats = fresh.compactMap(\.stats)
            let reason = agent.lastFinishReason
            let pass = done && answer.contains("Sources") && !stats.isEmpty && reason == .eos
            return (pass, "finished=\(done) sources=\(answer.contains("Sources")) "
                + "statsEntries=\(stats.count) finishReason=\(reason?.rawValue ?? "none") "
                + "stats=\(stats.joined(separator: " ;; ")) answer=\(answer)")
        }

        await step(6, "task3-note", "Weather task saves a note through write_note") {
            let mark = agent.entries.count
            let done = await self.ask(
                agent, chat,
                "\u{6771}\u{4EAC}\u{306E}\u{660E}\u{65E5}\u{306E}\u{5929}\u{6C17}\u{3092}"
                + "\u{8ABF}\u{3079}\u{3066}\u{3001}\u{8981}\u{7D04}\u{3092}\u{30E1}\u{30E2}"
                + "\u{306B}\u{4FDD}\u{5B58}\u{3057}\u{3066}")
            let fresh = self.newEntries(agent, from: mark)
            let saved = fresh.first { $0.kind == .toolResult && $0.text.contains("saved to ") }
            return (done && saved != nil,
                    "finished=\(done) saved=\(saved?.text ?? "none") "
                    + "toolResults=\(fresh.filter { $0.kind == .toolResult }.count) "
                    + "answer=\(Self.head(fresh.last { $0.kind == .answer }?.text ?? ""))")
        }

        await step(7, "move-note", "move_note puts the file under Desktop/agent") {
            let mark = agent.entries.count
            let done = await self.ask(
                agent, chat,
                "\u{305D}\u{306E}\u{30E1}\u{30E2}\u{3092} Desktop/agent "
                + "\u{306B}\u{79FB}\u{52D5}\u{3057}\u{3066}")
            let fresh = self.newEntries(agent, from: mark)
            let moved = fresh.first { $0.kind == .toolResult && $0.text.contains("moved to ") }
            let path = moved?.text.components(separatedBy: "moved to ").last?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let root = FileManager.default.homeDirectoryForCurrentUser
                .appending(path: "Desktop/agent", directoryHint: .notDirectory)
                .path(percentEncoded: false) + "/"
            let exists = !path.isEmpty && FileManager.default.fileExists(atPath: path)
            let inside = path.hasPrefix(root)
            return (done && exists && inside,
                    "finished=\(done) exists=\(exists) insideDesktopAgent=\(inside) "
                    + "path=\(path) result=\(moved?.text ?? "none")")
        }

        await step(8, "stop", "Stop halts a running task within three seconds") {
            let mark = agent.entries.count
            agent.input = self.searchTask
            guard agent.send(chat: chat) else { return (false, "send refused") }
            if let id = agent.sessionID, !self.sessions.contains(id) { self.sessions.append(id) }
            let grew = await self.wait(seconds: 180) {
                guard let last = agent.entries.last else { return false }
                return last.kind != .user && last.text.count > 20
            }
            agent.stop()
            let halted = await self.wait(seconds: 3) { !agent.running }
            _ = await self.wait(seconds: 120) { !agent.running }
            let stopped = self.newEntries(agent, from: mark)
                .first { $0.kind == .failure && $0.title == "Stopped" }
            return (grew && halted && stopped != nil,
                    "grew=\(grew) haltedWithin3s=\(halted) running=\(agent.running) "
                    + "stoppedEntry=\(stopped == nil ? "none" : Self.head(stopped?.text ?? "", 200)) "
                    + "status=\(agent.statusLine)")
        }

        await step(9, "after-stop", "A new question works after a stop") {
            let mark = agent.entries.count
            let done = await self.ask(agent, chat, "\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}")
            let answer = self.newEntries(agent, from: mark).last { $0.kind == .answer }?.text ?? ""
            return (done && !answer.isEmpty && !agent.running,
                    "finished=\(done) running=\(agent.running) answer=\(Self.head(answer))")
        }

        await step(10, "new-task", "New task clears the transcript") {
            agent.reset(chat: chat)
            await self.settle()
            return (agent.entries.isEmpty,
                    "entries=\(agent.entries.count) session=\(agent.sessionID ?? "none") "
                    + "status=\(agent.statusLine)")
        }

        await step(11, "switch-away", "Chat is blocked while the agent runs") {
            agent.input = self.searchTask
            guard agent.send(chat: chat) else { return (false, "send refused") }
            if let id = agent.sessionID, !self.sessions.contains(id) { self.sessions.append(id) }
            try? await Task.sleep(for: .seconds(2))
            navigator.selection = .chat
            await self.settle()
            return (!chat.canSend && chat.externallyBusy,
                    "chatCanSend=\(chat.canSend) externallyBusy=\(chat.externallyBusy) "
                    + "agentRunning=\(agent.running)")
        }

        await step(12, "switch-back", "The agent keeps running and keeps its entries") {
            navigator.selection = .agent
            await self.settle()
            let running = agent.running
            let count = agent.entries.count
            let done = await self.wait(seconds: 600) { !agent.running }
            return (running && count > 0 && done,
                    "runningOnReturn=\(running) entriesOnReturn=\(count) finished=\(done) "
                    + "entriesNow=\(agent.entries.count) "
                    + "answer=\(Self.head(agent.entries.last { $0.kind == .answer }?.text ?? ""))")
        }

        await step(13, "copy", "Copy transcript puts markdown on the pasteboard", screenshot: false) {
            NSPasteboard.general.clearContents()
            agent.copyTranscript()
            await self.settle()
            let text = NSPasteboard.general.string(forType: .string) ?? ""
            return (text.hasPrefix("# Agent transcript") && text.count > 1000,
                    "chars=\(text.count) head=\(Self.head(text, 120))")
        }

        await step(14, "unload", "Unload leaves the agent unable to send") {
            navigator.selection = .chat
            await self.settle()
            chat.unload()
            await self.settle()
            navigator.selection = .agent
            await self.settle()
            return (!chat.isModelLoaded && !agent.canSend(chat: chat),
                    "modelLoaded=\(chat.isModelLoaded) agentCanSend=\(agent.canSend(chat: chat)) "
                    + "entries=\(agent.entries.count) phase=\(chat.phaseDescription)")
        }

        await finish()
    }

    private func finish() async {
        let fileManager = FileManager.default
        for id in sessions {
            let source = AgentViewModel.transcriptsDirectory().appending(path: "\(id).md")
            let target = outDir.appending(path: "transcript-\(id).md")
            try? fileManager.removeItem(at: target)
            do {
                try fileManager.copyItem(at: source, to: target)
                note("transcript copied: \(target.lastPathComponent)")
            } catch {
                note("transcript missing for \(id): \(error)")
            }
        }
        await MetricsLog.flush()
        let text = (try? String(contentsOf: MetricsLog.fileURL, encoding: .utf8)) ?? ""
        let matched = text.split(separator: "\n").filter { line in
            sessions.contains { line.contains($0) }
        }
        try? Data((matched.joined(separator: "\n") + "\n").utf8)
            .write(to: outDir.appending(path: "metrics-agent.jsonl"), options: .atomic)
        note("metrics rows: \(matched.count) for sessions \(sessions)")

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        if let data = try? encoder.encode(results) {
            try? data.write(to: outDir.appending(path: "results.json"), options: .atomic)
        }
        let passed = results.count { $0.pass }
        let verdict = passed == results.count ? "PASS" : "FAIL"
        note("GUI E2E \(verdict) \(passed)/\(results.count)")
        try? Data((lines.joined(separator: "\n") + "\n").utf8)
            .write(to: outDir.appending(path: "driver.log"), options: .atomic)
        exit(passed == results.count ? 0 : 1)
    }
}
#endif
