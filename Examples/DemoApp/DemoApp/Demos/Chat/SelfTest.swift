import CoreMLBackend
import Foundation

enum SelfTest {
    static func run() -> Never {
        let args = CommandLine.arguments
        func value(_ flag: String) -> String? {
            if let i = args.firstIndex(of: flag), i + 1 < args.count { return args[i + 1] }
            return args.first { $0.hasPrefix(flag + "=") }.map { String($0.dropFirst(flag.count + 1)) }
        }
        let model = value("--model")
        let prompt = value("--prompt")
        let image = value("--image")
        let audio = value("--audio")
        let followUp = value("--followup")
        let maxTokens = value("--max-tokens").flatMap { Int($0) } ?? 128

        Task { @MainActor in
            exit(await execute(
                model: model, prompt: prompt, image: image, audio: audio, followUp: followUp,
                maxTokens: maxTokens))
        }
        dispatchMain()
    }

    @MainActor
    private static func execute(
        model: String?, prompt: String?, image: String?, audio: String?, followUp: String?,
        maxTokens: Int
    ) async -> Int32 {
        func errln(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
        guard let model, prompt != nil || audio != nil else {
            errln("selftest: --model plus --prompt (or --audio) are required")
            return 2
        }
        let vm = ChatViewModel()
        vm.maxTokens = maxTokens
        await vm.loadModel(path: model)
        guard vm.isModelLoaded, case .ready = vm.phase else {
            errln("selftest: load failed (\(vm.phaseDescription))")
            return 1
        }
        errln("[selftest] imageAttachment=\(vm.supportsImageAttachment) "
            + "audioAttachment=\(vm.supportsAudioAttachment)")
        if let image {
            vm.attachImage(URL(fileURLWithPath: image))
            guard vm.attachedImageURL != nil else {
                errln("selftest: this bundle does not take image input")
                return 1
            }
        }
        if let audio {
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
        var reply = ""
        for turn in [prompt ?? ""] + (followUp.map { [$0] } ?? []) {
            vm.input = turn
            guard vm.send() else {
                errln("selftest: send rejected (\(vm.phaseDescription))")
                return 1
            }
            await vm.generation?.value
            reply = vm.messages.last { $0.role == .assistant }?.text ?? ""
            FileHandle.standardOutput.write(Data((reply + "\n").utf8))
            errln("[selftest] \(vm.statusLine)")
            if case .failed(let why) = vm.phase {
                errln("selftest: generation failed (\(why))")
                return 1
            }
        }
        return reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 1 : 0
    }
}
