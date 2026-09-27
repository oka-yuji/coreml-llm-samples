import CoreML
import CoreMLBackend
import Foundation
import LLMCore

enum SelfTest {

    struct Options {
        var model: String?
        var prompt: String?
        var image: String?
        var audio: String?
        var followUp: String?
        var maxTokens: Int
        var stopAfterSeconds: Double?
        /// `--prompts "a|b|c"` runs several turns in one launch (one process, one model load).
        var prompts: [String] = []
        /// `--wired-tail <seconds>`: after the first turn whose prompt reaches the plain prefill
        /// width, keep sampling wired for this long to see how fast it comes back.
        var wiredTailSeconds: Double = 0
        /// `--tag <name>`: echoed as the first log line so a harness can tell this run's log from
        /// the previous run's, which is still on disk until the app rewrites it.
        var tag: String?
        /// `--fresh`: every `--prompts` turn starts a new conversation (no history, no KV reuse),
        /// so a long list of independent prompts fits in one launch.
        var fresh = false

        /// `--rag "質問1|質問2"` answers each question in RAG mode. `--rag-web` / `--rag-local`
        /// pick the sources (`0`/`no`/`off` turns one off); both are on by default.
        var rag: [String] = []
        var ragWeb = true
        var ragLocal = true

        var turns: [String] {
            if !rag.isEmpty { return rag }
            return prompts.isEmpty ? [prompt ?? ""] + (followUp.map { [$0] } ?? []) : prompts
        }
    }

    struct Sink {
        var out: (String) -> Void
        var err: (String) -> Void
    }

    static var isRequested: Bool { CommandLine.arguments.contains("--selftest") }

    static func options(arguments args: [String] = CommandLine.arguments) -> Options {
        func value(_ flag: String) -> String? {
            if let i = args.firstIndex(of: flag), i + 1 < args.count { return args[i + 1] }
            return args.first { $0.hasPrefix(flag + "=") }.map { String($0.dropFirst(flag.count + 1)) }
        }
        return Options(
            model: value("--model"),
            prompt: value("--prompt"),
            image: value("--image"),
            audio: value("--audio"),
            followUp: value("--followup"),
            maxTokens: value("--max-tokens").flatMap { Int($0) } ?? 128,
            stopAfterSeconds: value("--stop-after").flatMap { Double($0) },
            prompts: value("--prompts").map { $0.split(separator: "|").map(String.init) } ?? [],
            wiredTailSeconds: value("--wired-tail").flatMap { Double($0) } ?? 0,
            tag: value("--tag"),
            fresh: args.contains("--fresh"),
            rag: value("--rag").map { $0.split(separator: "|").map(String.init) } ?? [],
            ragWeb: !["0", "no", "off", "false"].contains(value("--rag-web") ?? ""),
            ragLocal: !["0", "no", "off", "false"].contains(value("--rag-local") ?? ""))
    }

    static func run() -> Never {
        let opts = options()
        let sink = Sink(
            out: { FileHandle.standardOutput.write(Data(($0 + "\n").utf8)) },
            err: { FileHandle.standardError.write(Data(($0 + "\n").utf8)) })
        Task { @MainActor in
            exit(await execute(opts, sink: sink))
        }
        dispatchMain()
    }

    static func resolve(_ path: String) -> String {
        #if os(iOS)
        guard !path.hasPrefix("/") else { return path }
        return ModelStorage.documentsDirectory()
            .appending(path: path).path(percentEncoded: false)
        #else
        return path
        #endif
    }

    @MainActor
    static func execute(_ opts: Options, sink: Sink) async -> Int32 {
        let errln = sink.err
        if AneProbe.isRequested { return await AneProbe.run(sink: sink) }
        let hasPrompt = opts.turns.contains { !$0.isEmpty }
        guard let model = opts.model.map(resolve), hasPrompt || opts.audio != nil else {
            errln("selftest: --model plus --prompt / --prompts (or --audio) are required")
            return 2
        }
        errln("[selftest] tag=\(opts.tag ?? "(none)")")
        for line in E5RTCache.startupLines { errln(line) }
        let vm = ChatViewModel()
        vm.maxTokens = opts.maxTokens
        if !opts.rag.isEmpty {
            vm.ragEnabled = true
            vm.ragWeb = opts.ragWeb
            vm.ragLocal = opts.ragLocal
        }
        let clock = ContinuousClock()
        let loadStart = clock.now
        await vm.loadModel(path: model)
        let loadSeconds = (clock.now - loadStart) / .seconds(1)
        guard vm.isModelLoaded, case .ready = vm.phase else {
            errln("selftest: load failed (\(vm.phaseDescription))")
            return 1
        }
        errln(String(format: "[selftest] modelLoadSeconds=%.2f", loadSeconds))
        for line in vm.loadBreakdown { errln(line) }
        for line in E5RTCache.afterLoadLines() { errln(line) }
        errln("[selftest] imageAttachment=\(vm.supportsImageAttachment) "
            + "audioAttachment=\(vm.supportsAudioAttachment)")
        if vm.ragEnabled {
            let settings = RAGSettings.fromDefaults()
            errln("[rag] mode=on web=\(vm.ragWeb) local=\(vm.ragLocal) "
                + "search=\(settings.provider) results=\(settings.searchResults) "
                + "pages=\(settings.pagesToRead) pageBudget=\(settings.pageCharacterBudget)chars "
                + "sourceBudget=\(RAGContext.sourceTokenBudget)tok today=\(RAGContext.today())")
            errln("[rag] documents=" + settings.documentsDirectory.path(percentEncoded: false))
        }
        errln("[selftest] prefillMode="
            + (ProcessInfo.processInfo.environment["CORELLM_PREFILL_MODE"] ?? "(manifest default)")
            + " specialization="
            + (ProcessInfo.processInfo.environment["CORELLM_SPECIALIZATION"] ?? "(manifest default)")
            + " prewarm="
            + (ProcessInfo.processInfo.environment["CORELLM_PREFILL_PREWARM"] ?? "(manifest default)")
            + " earlyExit="
            + (ProcessInfo.processInfo.environment["CORELLM_PREFILL_EARLY_EXIT"] ?? "(manifest default)"))
        if let prewarm = vm.prewarm {
            let wiredBefore = MetricsLog.sysWiredBytes().map { Double($0) / 1_048_576 }
            let sampler = WiredSampler()
            await prewarm.value
            sampler.stop()
            for line in vm.prewarmLog { errln(line) }
            let peak = sampler.snapshot().map(\.mib).max()
            errln(String(format: "[prewarm] %@ wiredBefore=%@ wiredPeak=%@ MiB",
                         vm.prewarmSummary.isEmpty ? "(off)" : vm.prewarmSummary,
                         wiredBefore.map { String(format: "%.1f", $0) } ?? "n/a",
                         peak.map { String(format: "%.1f", $0) } ?? "n/a"))
        }
        if let image = opts.image.map(resolve) {
            vm.attachImage(URL(fileURLWithPath: image))
            guard vm.attachedImageURL != nil else {
                errln("selftest: this bundle does not take image input")
                return 1
            }
        }
        if let audio = opts.audio.map(resolve) {
            do {
                let samples = try AudioFileLoader.monoSamples(at: URL(fileURLWithPath: audio))
                vm.attachAudio(samples: samples)
            } catch {
                errln("selftest: cannot read \(audio) (\(error))")
                return 1
            }
            guard let attached = vm.attachedAudio else {
                errln("selftest: this bundle does not take audio input")
                return 1
            }
            errln(String(format: "[selftest] audio %.2fs (%d samples)",
                         attached.seconds, attached.samples.count))
        }
        if let after = opts.stopAfterSeconds {
            return await stopTurn(vm: vm, prompt: opts.prompt ?? "", after: after, sink: sink)
        }
        var reply = ""
        var tailTaken = false
        for (index, turn) in opts.turns.enumerated() {
            let mib = 1_048_576.0
            if opts.fresh, index > 0 {
                vm.reset()
                // `reset()` clears the engine from an unawaited task; wait for it, so this turn
                // cannot reuse the previous turn's KV.
                await vm.engineHandle?.engine.resetConversation()
            }
            let wiredBefore = MetricsLog.sysWiredBytes()
            let sampler = WiredSampler()
            vm.input = turn
            guard vm.send() else {
                sampler.stop()
                errln("selftest: send rejected (\(vm.phaseDescription))")
                return 1
            }
            await vm.generation?.value
            sampler.stop()
            reply = vm.messages.last { $0.role == .assistant }?.text ?? ""
            if vm.ragEnabled {
                errln("[rag] turn=\(index) question=\(turn)")
                errln("[rag] turn=\(index) \(vm.ragStatus)")
                for line in vm.messages.last(where: { $0.role == .assistant })?.sources ?? [] {
                    errln("[rag] turn=\(index) source \(line)")
                }
                errln("[rag] turn=\(index) answer=\(reply.replacing("\n", with: " "))")
            }
            sink.out(reply)
            // One line per turn, JSON-quoted, so replies with newlines can be diffed turn by turn.
            errln("[reply] turn=\(index) "
                + ((try? JSONEncoder().encode(reply)).map { String(decoding: $0, as: UTF8.self) } ?? "null"))
            let wiredAfter = MetricsLog.sysWiredBytes()
            let series = sampler.snapshot()
            if let before = wiredBefore, let after = wiredAfter {
                let base = Double(before) / mib
                let peak = series.map(\.mib).max()
                // The wired reading closest to TTFT is "just after prefill finished". TTFT comes
                // from the status line the app already prints, so no engine hook is needed.
                let ttft = Self.firstNumber(in: vm.statusLine, after: "TTFT ")
                let atTTFT = ttft.flatMap { t in
                    series.min(by: { abs($0.seconds - t) < abs($1.seconds - t) })?.mib
                }
                errln(String(
                    format: "[wired] turn=%d chars=%d before=%.1f ttft=%@ peak=%@ after=%.1f "
                        + "delta=%+.1f peakDelta=%@ MiB samples=%d",
                    index, turn.count, base,
                    atTTFT.map { String(format: "%.1f", $0) } ?? "n/a",
                    peak.map { String(format: "%.1f", $0) } ?? "n/a",
                    Double(after) / mib, Double(after - before) / mib,
                    peak.map { String(format: "%+.1f", $0 - base) } ?? "n/a",
                    series.count))
            } else {
                errln("[wired] turn=\(index) unavailable (host_statistics64 refused)")
            }
            errln("[selftest] \(vm.statusLine)")
            if let engine = vm.engineHandle?.engine {
                errln("[load] prefill functions so far: \(await engine.prefillLoadSummary)")
            }
            if !tailTaken, opts.wiredTailSeconds > 0,
               let prompt = Self.firstNumber(in: vm.statusLine, after: "prompt "), prompt >= 512 {
                tailTaken = true
                errln(String(format: "[wired-tail] turn=%d sampling %.0fs after a plain-prefill turn",
                             index, opts.wiredTailSeconds))
                let tail = await Self.sampleWired(seconds: opts.wiredTailSeconds)
                for row in tail where Int((row.seconds * 2).rounded()) % 10 == 0 {
                    errln(String(format: "[wired-tail] t=%.1fs wired=%.1f MiB", row.seconds, row.mib))
                }
                if let low = tail.map(\.mib).min(), let last = tail.last?.mib {
                    errln(String(format: "[wired-tail] min=%.1f last=%.1f MiB", low, last))
                }
            }
            if case .failed(let why) = vm.phase {
                errln("selftest: generation failed (\(why))")
                return 1
            }
        }
        return reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 1 : 0
    }

    @MainActor
    private static func stopTurn(
        vm: ChatViewModel, prompt: String, after: Double, sink: Sink
    ) async -> Int32 {
        let errln = sink.err
        vm.input = prompt
        guard vm.send() else {
            errln("selftest: send rejected (\(vm.phaseDescription))")
            return 1
        }
        guard let generation = vm.generation else {
            errln("selftest: no generation task")
            return 1
        }
        let clock = ContinuousClock()
        try? await Task.sleep(for: .seconds(after))
        guard vm.isGenerating else {
            errln("selftest: the turn already finished before the stop (\(vm.phaseDescription))")
            return 1
        }
        let t0 = clock.now
        vm.stop()
        await generation.value
        let stopSeconds = (clock.now - t0) / .seconds(1)
        let reply = vm.messages.last { $0.role == .assistant }?.text ?? ""
        errln(String(
            format: "[selftest] stop after %.1fs settled in %.3fs phase=%@ status=%@ replyChars=%d",
            after, stopSeconds, vm.phaseDescription, vm.statusLine, reply.count))
        sink.out(reply)
        guard case .ready = vm.phase else {
            errln("selftest: stop left the chat in \(vm.phaseDescription), expected ready")
            return 1
        }
        guard vm.statusLine == "Stopped" else {
            errln("selftest: stop status was \"\(vm.statusLine)\", expected \"Stopped\"")
            return 1
        }
        return 0
    }
}

/// Samples system wired memory on a background queue so a turn's **peak** is visible, not just its
/// endpoints. Same 0.5 s cadence as the ANE probe.
final class WiredSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var series: [(seconds: Double, mib: Double)] = []
    private var stopped = false
    private let origin = Date()

    init() {
        DispatchQueue.global(qos: .utility).async { [self] in
            while true {
                lock.lock()
                let done = stopped
                lock.unlock()
                if done { return }
                let value = MetricsLog.sysWiredBytes().map { Double($0) / 1_048_576 } ?? -1
                let at = Date().timeIntervalSince(origin)
                lock.lock()
                series.append((at, value))
                lock.unlock()
                Thread.sleep(forTimeInterval: 0.5)
            }
        }
    }

    func stop() {
        lock.lock()
        stopped = true
        lock.unlock()
    }

    func snapshot() -> [(seconds: Double, mib: Double)] {
        lock.lock()
        defer { lock.unlock() }
        return series
    }
}

extension SelfTest {
    /// Pulls the first number that follows `label` out of the app's status line.
    static func firstNumber(in text: String, after label: String) -> Double? {
        guard let range = text.range(of: label) else { return nil }
        let digits = text[range.upperBound...].prefix { $0.isNumber || $0 == "." }
        return Double(digits)
    }

    static func sampleWired(seconds: Double) async -> [(seconds: Double, mib: Double)] {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                var out: [(seconds: Double, mib: Double)] = []
                let start = Date()
                while Date().timeIntervalSince(start) < seconds {
                    let value = MetricsLog.sysWiredBytes().map { Double($0) / 1_048_576 } ?? -1
                    out.append((Date().timeIntervalSince(start), value))
                    Thread.sleep(forTimeInterval: 0.5)
                }
                continuation.resume(returning: out)
            }
        }
    }
}

// MARK: - ANE residency probe
//
// Ported from ~/Desktop/md/coreml-e2b-aneprobe-20260917/aneprobe.swift so the same measurement
// runs inside the app on a phone, where ANE weight residency actually shows up in wired memory.
//
//   --selftest --aneprobe "chunk_0_8.mlmodelc#decode,chunk_0_8.mlmodelc#prefill32"
//
// Paths are Documents-relative on iOS (see `SelfTest.resolve`). Each stage loads the model,
// settles, samples wired / footprint, runs three zero-filled predictions and samples again;
// the models stay held until the end, so a stage's delta is what that stage added.
enum AneProbe {
    private static let mib = 1_048_576.0

    private static func argumentStages(_ args: [String] = CommandLine.arguments) -> [String] {
        let raw: String?
        if let i = args.firstIndex(of: "--aneprobe"), i + 1 < args.count, !args[i + 1].hasPrefix("--") {
            raw = args[i + 1]
        } else {
            raw = args.first { $0.hasPrefix("--aneprobe=") }.map { String($0.dropFirst("--aneprobe=".count)) }
        }
        return (raw ?? "").split(separator: ",").map(String.init).filter { !$0.isEmpty }
    }

    static var isRequested: Bool { !argumentStages().isEmpty }

    private static func flag(_ name: String, _ args: [String] = CommandLine.arguments) -> String? {
        if let i = args.firstIndex(of: name), i + 1 < args.count { return args[i + 1] }
        return args.first { $0.hasPrefix(name + "=") }.map { String($0.dropFirst(name.count + 1)) }
    }

    private enum Event: Sendable {
        case line(String)
        case done(Int32)
    }

    /// The measurement blocks its thread for minutes (settle windows, 40-sample averages), so it
    /// runs on a background queue and streams lines back to the caller's actor.
    @MainActor
    static func run(sink: SelfTest.Sink) async -> Int32 {
        let stages = argumentStages()
        let units: MLComputeUnits = flag("--probe-units")?.lowercased() == "cpu" ? .cpuOnly : .cpuAndNeuralEngine
        let timeout = flag("--probe-timeout").flatMap { Double($0) } ?? 300
        let (stream, continuation) = AsyncStream<Event>.makeStream()
        DispatchQueue.global(qos: .userInitiated).async {
            let code = measure(stages: stages, units: units, timeout: timeout) {
                continuation.yield(.line($0))
            }
            continuation.yield(.done(code))
            continuation.finish()
        }
        var code: Int32 = 0
        for await event in stream {
            switch event {
            case .line(let line): sink.out(line)
            case .done(let value): code = value
            }
        }
        return code
    }

    // MARK: measurement primitives

    private static func wiredBytes() -> Double {
        MetricsLog.sysWiredBytes().map(Double.init) ?? -1
    }

    /// Same counter the engine reports (`task_vm_info.phys_footprint`), and available on iOS.
    private static func footprintBytes() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Double(info.phys_footprint) : -1
    }

    /// Headroom before jetsam. This is the number the 6 GB campaign is actually about: if ANE
    /// weights are charged to the app, they come out of here even when wired does not move.
    private static func availableBytes() -> Double {
        #if os(iOS)
        return Double(os_proc_available_memory())
        #else
        return -1
        #endif
    }

    /// Settle, then 40 samples at 0.5 s. The 20 s window keeps system-wired noise near +/-35 MiB.
    private static func sample(settle: Double = 3.0, n: Int = 40, gap: Double = 0.5)
        -> (wired: Double, wiredMin: Double, footprint: Double, available: Double) {
        Thread.sleep(forTimeInterval: settle)
        var wired: [Double] = []
        var footprint: [Double] = []
        var available: [Double] = []
        for i in 0..<n {
            wired.append(wiredBytes())
            footprint.append(footprintBytes())
            available.append(availableBytes())
            if i < n - 1 { Thread.sleep(forTimeInterval: gap) }
        }
        return (wired.sorted()[n / 2], wired.min()!, footprint.sorted()[n / 2], available.sorted()[n / 2])
    }

    private static func elementSize(_ type: MLMultiArrayDataType) -> Int {
        switch type {
        case .float16: 2
        case .float32: 4
        case .double: 8
        case .int32: 4
        @unknown default: 4
        }
    }

    private static func zeroInputs(_ model: MLModel) throws -> MLFeatureProvider {
        var values: [String: MLFeatureValue] = [:]
        for (name, description) in model.modelDescription.inputDescriptionsByName {
            guard let constraint = description.multiArrayConstraint else {
                throw NSError(domain: "aneprobe", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "input \(name) is not a multiArray"])
            }
            let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
            memset(array.dataPointer, 0, array.count * elementSize(constraint.dataType))
            values[name] = MLFeatureValue(multiArray: array)
        }
        return try MLDictionaryFeatureProvider(dictionary: values)
    }

    private final class Box<T>: @unchecked Sendable {
        var value: T
        init(_ value: T) { self.value = value }
    }

    /// Returns milliseconds, or nil when the prediction did not come back inside `timeout`.
    private static func predictTimed(
        _ model: MLModel, _ input: MLFeatureProvider, timeout: Double
    ) -> (ms: Double?, error: String?) {
        let elapsed = Box<Double>(-1)
        let failure = Box<String?>(nil)
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .userInitiated).async {
            let start = DispatchTime.now()
            do { _ = try model.prediction(from: input) } catch { failure.value = "\(error)" }
            elapsed.value = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e6
            done.signal()
        }
        if done.wait(timeout: .now() + timeout) == .timedOut { return (nil, "HUNG") }
        return (elapsed.value, failure.value)
    }

    // MARK: the run

    private static func measure(
        stages: [String], units: MLComputeUnits, timeout: Double, out: @escaping @Sendable (String) -> Void
    ) -> Int32 {
        typealias Sample = (wired: Double, wiredMin: Double, footprint: Double, available: Double)
        var previous: Sample?
        func emit(_ phase: String, _ stage: String, _ s: Sample,
                  load: String = "-", predict: String = "-", note: String = "-") {
            let dWired = previous.map { (s.wired - $0.wired) / mib } ?? 0
            let dWiredMin = previous.map { (s.wiredMin - $0.wiredMin) / mib } ?? 0
            let dFootprint = previous.map { (s.footprint - $0.footprint) / mib } ?? 0
            let dAvailable = previous.map { ($0.available - s.available) / mib } ?? 0
            out(String(
                format: "%@\t%@\t%.1f\t%+.1f\t%.1f\t%+.1f\t%.1f\t%+.1f\t%.1f\t%+.1f\t%@\t%@\t%@",
                phase, stage, s.wired / mib, dWired, s.wiredMin / mib, dWiredMin,
                s.footprint / mib, dFootprint, s.available / mib, dAvailable, load, predict, note))
            previous = s
        }

        out("# aneprobe units=\(ChunkedSpeculativeChainUnits.name(units)) stages=\(stages.count) timeout=\(Int(timeout))s")
        out("phase\tstage\twired_MiB\tdWired_MiB\twiredMin_MiB\tdWiredMin_MiB\tfootprint_MiB"
            + "\tdFoot_MiB\tavail_MiB\tusedFromAvail_MiB\tload_s\tpredict_ms\tnote")

        // 30 s idle baseline: the raw drift bounds how much of a delta is real.
        var baseline: [Double] = []
        for i in 0..<60 {
            baseline.append(wiredBytes())
            if i < 59 { Thread.sleep(forTimeInterval: 0.5) }
        }
        if let low = baseline.min(), let high = baseline.max() {
            out(String(format: "# baseline_drift_30s_MiB\t%.1f\tmin=%.1f\tmax=%.1f",
                       (high - low) / mib, low / mib, high / mib))
        }
        emit("baseline", "-", sample(settle: 0))

        var held: [MLModel] = []
        var exitCode: Int32 = 0
        for (index, stage) in stages.enumerated() {
            let parts = stage.split(separator: "#", maxSplits: 1).map(String.init)
            let url = URL(fileURLWithPath: SelfTest.resolve(parts[0]))
            let function = parts.count > 1 ? parts[1] : nil
            let tag = "\(url.lastPathComponent)#\(function ?? "(default)")"

            let configuration = MLModelConfiguration()
            configuration.computeUnits = units
            if let function { configuration.functionName = function }

            do {
                let start = DispatchTime.now()
                let model = try MLModel(contentsOf: url, configuration: configuration)
                let loadSeconds = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
                held.append(model)
                emit("load#\(index)", tag, sample(), load: String(format: "%.2f", loadSeconds))

                let input = try zeroInputs(model)
                var timings: [String] = []
                for label in ["warmup", "p1", "p2"] {
                    let (ms, error) = predictTimed(model, input, timeout: timeout)
                    guard let ms else {
                        out("HUNG\t\(tag)\t\(label)\ttimeout=\(Int(timeout))s")
                        return 3
                    }
                    if let error { out("PREDICT_ERROR\t\(tag)\t\(label)\t\(error)"); exitCode = 4 }
                    timings.append(String(format: "%@=%.1f", label, ms))
                }
                emit("predict#\(index)", tag, sample(),
                     predict: timings.last ?? "-", note: timings.joined(separator: " "))
            } catch {
                out("LOAD_ERROR\t\(tag)\t\(error)")
                exitCode = 4
            }
        }

        held.removeAll()
        emit("released", "-", sample(settle: 5))
        out("# aneprobe done")
        return exitCode
    }
}

/// `ChunkedSpeculativeChain.cuName` lives inside the backend module; this is the app-side echo.
private enum ChunkedSpeculativeChainUnits {
    static func name(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly: "cpuOnly"
        case .cpuAndGPU: "cpuAndGPU"
        case .cpuAndNeuralEngine: "cpuAndNeuralEngine"
        case .all: "all"
        @unknown default: "unknown"
        }
    }
}
