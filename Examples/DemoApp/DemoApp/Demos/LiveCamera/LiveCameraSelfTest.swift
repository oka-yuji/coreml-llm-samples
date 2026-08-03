import CoreMLBackend
import Foundation

enum LiveCameraSelfTest {

    struct Options {
        var model: String?
        var images: [String]
        var cycles: Int
        var useCamera: Bool
        var cancelGate: Bool
        var stopBudgetSeconds: Double
        var cancelPhase: LiveCyclePhase
        var cancelWarmupCycles: Int
    }

    static let frameSampleName = "live-frame-sample.jpg"

    static var isRequested: Bool { CommandLine.arguments.contains("--live-selftest") }

    static func options(arguments args: [String] = CommandLine.arguments) -> Options {
        func value(_ flag: String) -> String? {
            if let i = args.firstIndex(of: flag), i + 1 < args.count { return args[i + 1] }
            return args.first { $0.hasPrefix(flag + "=") }.map { String($0.dropFirst(flag.count + 1)) }
        }
        let images = value("--images")?
            .split(separator: ",")
            .map { String($0).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty } ?? []
        return Options(
            model: value("--model"),
            images: images,
            cycles: value("--cycles").flatMap { Int($0) } ?? 10,
            useCamera: args.contains("--camera"),
            cancelGate: args.contains("--cancel-gate"),
            stopBudgetSeconds: value("--stop-budget").flatMap { Double($0) } ?? 1.0,
            cancelPhase: value("--cancel-phase").flatMap { LiveCyclePhase(rawValue: $0) } ?? .feed,
            cancelWarmupCycles: value("--cancel-warmup").flatMap { Int($0) } ?? 0)
    }

    static func run() -> Never {
        let opts = options()
        let sink = SelfTest.Sink(
            out: { FileHandle.standardOutput.write(Data(($0 + "\n").utf8)) },
            err: { FileHandle.standardError.write(Data(($0 + "\n").utf8)) })
        Task { @MainActor in
            exit(await execute(opts, sink: sink))
        }
        dispatchMain()
    }

    @MainActor
    static func execute(_ opts: Options, sink: SelfTest.Sink) async -> Int32 {
        let errln = sink.err
        guard opts.useCamera || !opts.images.isEmpty else {
            errln("live-selftest: either --images a.jpg,b.jpg,c.jpg or --camera is required")
            return 2
        }
        let urls = opts.images
            .map { SelfTest.resolve($0) }
            .map { URL(fileURLWithPath: $0) }
        for url in urls where !FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
            errln("live-selftest: missing image \(url.path(percentEncoded: false))")
            return 2
        }

        let chat = ChatViewModel()
        chat.maxTokens = LiveCameraViewModel.maxNewTokens
        if let model = opts.model.map(SelfTest.resolve) {
            errln("[live] preloading \(model)")
            await chat.loadModel(path: model)
            errln("[live] preloaded phase=\(chat.phaseDescription) image=\(chat.supportsImageAttachment)")
        } else {
            errln("[live] no --model given: starting from an unloaded engine")
        }
        let handle: ChatViewModel.EngineHandle
        switch await LiveEngineProvision.ensureVisionEngine(
            chat: chat, status: { errln("[live] \($0)") })
        {
        case .unavailable(let message):
            errln("live-selftest: \(message)")
            return 1
        case .ready(let ready):
            handle = ready
        }
        errln("[live] engine bundle=\(handle.bundleFolder ?? "?") path=\(chat.loadedPath)")
        errln("[live] model=\(chat.modelName) speculative=\(handle.speculative) cycles=\(opts.cycles) "
            + "source=\(opts.useCamera ? "camera" : "\(urls.count) images") "
            + "thermal=\(LiveCameraViewModel.thermalName())")

        let widthsBeforePrewarm = await handle.engine.residentPrefillWidths()
        let prewarm: LiveVisionPrewarm
        do {
            prewarm = try await handle.engine.beginLiveVisionSession(
                question: LiveCameraViewModel.question)
        } catch {
            errln("live-selftest: prewarm failed (\(error))")
            return 1
        }
        let widthsAfterPrewarm = await handle.engine.residentPrefillWidths()
        errln("[live] prewarm \(prewarm.summary)")
        errln("[live] resident prefill widths \(widthsBeforePrewarm) -> \(widthsAfterPrewarm)")

        let source: any LiveFrameSource
        if opts.useCamera {
            errln("[live] requesting camera permission…")
            guard await CameraCapture.requestPermission() else {
                errln("live-selftest: \(CameraError.permissionDenied.description)")
                return 1
            }
            let capture = CameraCapture()
            do {
                try capture.configure()
            } catch {
                errln("live-selftest: \(error)")
                return 1
            }
            capture.start()
            var waited = 0.0
            while !capture.hasFrame, waited < 10 {
                try? await Task.sleep(for: .milliseconds(200))
                waited += 0.2
            }
            guard capture.hasFrame else {
                errln("live-selftest: no camera frame arrived within 10s")
                return 1
            }
            errln(String(format: "[live] camera ready after %.1fs", waited))
            let sample = ModelStorage.documentsDirectory().appending(path: Self.frameSampleName)
            do {
                let label = try capture.stageFrame(to: sample)
                errln("[live] orientation sample \(Self.frameSampleName) (\(label))")
            } catch {
                errln("[live] could not save an orientation sample: \(error)")
            }
            source = capture
        } else {
            source = FileSequenceFrameSource(urls: urls)
        }

        let live = LiveCameraViewModel()
        var reports: [LiveCycleReport] = []
        live.onCycle = { report in
            reports.append(report)
            sink.out("cycle \(report.index) [\(report.frameLabel)] \(report.caption)")
            errln("[live] \(report.detailLine)")
        }

        if opts.cancelGate {
            return await runCancelGate(
                live: live, handle: handle, source: source, opts: opts,
                reports: { reports }, sink: sink)
        }

        live.start(
            engine: handle.engine, source: source,
            speculative: handle.speculative, cycleLimit: opts.cycles,
            modelID: handle.modelID, hfRevision: handle.hfRevision,
            computeUnits: handle.computeUnits, bundleFolder: handle.bundleFolder)
        await live.loop?.value

        let widthsAfterRun = await handle.engine.residentPrefillWidths()
        errln("[live] resident prefill widths after the run \(widthsAfterRun)")
        guard widthsAfterRun == widthsAfterPrewarm else {
            errln("live-selftest: the prewarm plan missed a prefill width "
                + "(prewarm \(widthsAfterPrewarm), after run \(widthsAfterRun))")
            return 1
        }

        guard live.errorText.isEmpty else {
            errln("live-selftest: cycle failed (\(live.errorText))")
            return 1
        }
        guard reports.count == opts.cycles else {
            errln("live-selftest: expected \(opts.cycles) cycles, completed \(reports.count)")
            return 1
        }
        if let empty = reports.first(where: { $0.caption.isEmpty }) {
            errln("live-selftest: cycle \(empty.index) produced no text")
            return 1
        }

        let prompts = Set(reports.map(\.promptTokens))
        let cycleSeconds = reports.map(\.cycleSeconds)
        let mean = cycleSeconds.reduce(0, +) / Double(cycleSeconds.count)
        errln(String(
            format: "[live] cycles=%d promptTokens=%@ cycle min/mean/max %.2f/%.2f/%.2fs",
            reports.count, "\(prompts.sorted())",
            cycleSeconds.min() ?? 0, mean, cycleSeconds.max() ?? 0))
        guard prompts.count == 1 else {
            errln("live-selftest: prompt length changed across cycles \(prompts.sorted()) "
                + "— the per-cycle context reset is not holding")
            return 1
        }
        errln("[live] context reset holds: every cycle prefilled \(prompts.first ?? 0) tokens")
        return 0
    }

    @MainActor
    private static func runCancelGate(
        live: LiveCameraViewModel,
        handle: ChatViewModel.EngineHandle,
        source: any LiveFrameSource,
        opts: Options,
        reports: () -> [LiveCycleReport],
        sink: SelfTest.Sink
    ) async -> Int32 {
        let errln = sink.err
        let clock = ContinuousClock()
        let target = opts.cancelPhase
        errln("[cancel] open-ended loop, warmup=\(opts.cancelWarmupCycles) cycles, "
            + "cancelling in the \(target.label) phase")
        live.start(
            engine: handle.engine, source: source, speculative: handle.speculative,
            modelID: handle.modelID, hfRevision: handle.hfRevision,
            computeUnits: handle.computeUnits, bundleFolder: handle.bundleFolder)
        guard let running = live.loop else {
            errln("live-selftest: the loop did not start")
            return 1
        }

        let waitStart = clock.now
        while reports().count < opts.cancelWarmupCycles {
            guard live.isRunning, (clock.now - waitStart) / .seconds(1) < 120 else { break }
            try? await Task.sleep(for: .milliseconds(10))
        }
        guard reports().count >= opts.cancelWarmupCycles else {
            errln("live-selftest: only \(reports().count) of \(opts.cancelWarmupCycles) warmup cycles ran")
            return 1
        }
        let warmedAt = clock.now
        while live.cyclePhase != target, (clock.now - warmedAt) / .seconds(1) < 60 {
            guard live.isRunning else { break }
            try? await Task.sleep(for: .milliseconds(5))
        }
        guard live.cyclePhase == target else {
            errln("live-selftest: never observed the \(target.label) phase "
                + "(phase=\(live.cyclePhase.label))")
            return 1
        }
        errln(String(
            format: "[cancel] reached the %@ phase %.2fs after the warmup",
            target.label, (clock.now - warmedAt) / .seconds(1)))

        let t0 = clock.now
        live.stop()
        let uiSeconds = (clock.now - t0) / .seconds(1)
        let uiStopped = !live.isRunning && live.cyclePhase == .idle
        await running.value
        let stopSeconds = (clock.now - t0) / .seconds(1)
        errln(String(
            format: "[cancel] ui state settled in %.4fs (synchronous=%@), compute stopped in %.3fs",
            uiSeconds, uiStopped ? "yes" : "no", stopSeconds))
        guard uiStopped else {
            errln("live-selftest: stop() did not update the UI state synchronously")
            return 1
        }
        guard stopSeconds <= opts.stopBudgetSeconds else {
            errln(String(
                format: "live-selftest: compute took %.3fs to stop, budget is %.3fs",
                stopSeconds, opts.stopBudgetSeconds))
            return 1
        }
        let cancelledDuring = reports().count
        errln("[cancel] completed cycles before the cancel: \(cancelledDuring)")

        errln("[cancel] restarting to prove the chain repairs itself")
        live.start(
            engine: handle.engine, source: source, speculative: handle.speculative,
            cycleLimit: 2,
            modelID: handle.modelID, hfRevision: handle.hfRevision,
            computeUnits: handle.computeUnits, bundleFolder: handle.bundleFolder)
        await live.loop?.value

        guard live.errorText.isEmpty else {
            errln("live-selftest: the restart failed (\(live.errorText))")
            return 1
        }
        let after = Array(reports().dropFirst(cancelledDuring))
        guard after.count == 2 else {
            errln("live-selftest: expected 2 cycles after the restart, got \(after.count)")
            return 1
        }
        for report in after where report.caption.isEmpty {
            errln("live-selftest: cycle \(report.index) produced no text after the restart")
            return 1
        }
        let prompts = Set(reports().map(\.promptTokens))
        guard prompts.count == 1 else {
            errln("live-selftest: prompt length changed across cycles \(prompts.sorted())")
            return 1
        }
        for report in after {
            sink.out("restart cycle \(report.index) [\(report.frameLabel)] \(report.caption)")
            errln("[cancel] \(report.detailLine)")
        }
        errln("[cancel] restart is clean: \(after.count) cycles, promptTokens \(prompts.sorted())")
        return 0
    }
}
