import CoreMLBackend
import Foundation

enum SelfTest {

    struct Options {
        var model: String?
        var prompt: String?
        var image: String?
        var audio: String?
        var followUp: String?
        var maxTokens: Int
        var stopAfterSeconds: Double?
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
            stopAfterSeconds: value("--stop-after").flatMap { Double($0) })
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
        guard let model = opts.model.map(resolve), opts.prompt != nil || opts.audio != nil else {
            errln("selftest: --model plus --prompt (or --audio) are required")
            return 2
        }
        let vm = ChatViewModel()
        vm.maxTokens = opts.maxTokens
        await vm.loadModel(path: model)
        guard vm.isModelLoaded, case .ready = vm.phase else {
            errln("selftest: load failed (\(vm.phaseDescription))")
            return 1
        }
        errln("[selftest] imageAttachment=\(vm.supportsImageAttachment) "
            + "audioAttachment=\(vm.supportsAudioAttachment)")
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
        for turn in [opts.prompt ?? ""] + (opts.followUp.map { [$0] } ?? []) {
            vm.input = turn
            guard vm.send() else {
                errln("selftest: send rejected (\(vm.phaseDescription))")
                return 1
            }
            await vm.generation?.value
            reply = vm.messages.last { $0.role == .assistant }?.text ?? ""
            sink.out(reply)
            errln("[selftest] \(vm.statusLine)")
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
