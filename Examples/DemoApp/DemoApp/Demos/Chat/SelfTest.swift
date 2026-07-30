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
        let followUp = value("--followup")
        let maxTokens = value("--max-tokens").flatMap { Int($0) } ?? 128

        Task { @MainActor in
            exit(await execute(
                model: model, prompt: prompt, image: image, followUp: followUp, maxTokens: maxTokens))
        }
        dispatchMain()
    }

    @MainActor
    private static func execute(
        model: String?, prompt: String?, image: String?, followUp: String?, maxTokens: Int
    ) async -> Int32 {
        func errln(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
        guard let model, let prompt else {
            errln("selftest: --model and --prompt are required")
            return 2
        }
        let vm = ChatViewModel()
        vm.maxTokens = maxTokens
        await vm.loadModel(path: model)
        guard vm.isModelLoaded, case .ready = vm.phase else {
            errln("selftest: load failed (\(vm.phaseDescription))")
            return 1
        }
        errln("[selftest] imageAttachment=\(vm.supportsImageAttachment)")
        if let image {
            vm.attachImage(URL(fileURLWithPath: image))
            guard vm.attachedImageURL != nil else {
                errln("selftest: this bundle does not take image input")
                return 1
            }
        }
        var reply = ""
        for turn in [prompt] + (followUp.map { [$0] } ?? []) {
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
