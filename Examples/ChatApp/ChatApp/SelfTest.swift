import Foundation

/// GUI 経路をヘッドレスで検証する開発用フック。ViewModel の通常送信経路(UI と同一コード)で
/// 1 往復生成し、応答を stdout に、tok/s を含む stats を stderr に出して exit する。
enum SelfTest {
    static func run() -> Never {
        let args = CommandLine.arguments
        func value(_ flag: String) -> String? {
            if let i = args.firstIndex(of: flag), i + 1 < args.count { return args[i + 1] }
            return args.first { $0.hasPrefix(flag + "=") }.map { String($0.dropFirst(flag.count + 1)) }
        }
        let model = value("--model")
        let prompt = value("--prompt")
        let maxTokens = value("--max-tokens").flatMap { Int($0) } ?? 128

        Task { @MainActor in
            exit(await execute(model: model, prompt: prompt, maxTokens: maxTokens))
        }
        dispatchMain()  // メインキューを回して @MainActor Task を実行させる(exit で終了)。
    }

    @MainActor
    private static func execute(model: String?, prompt: String?, maxTokens: Int) async -> Int32 {
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
        vm.input = prompt
        guard vm.send() else {
            errln("selftest: send rejected (\(vm.phaseDescription))")
            return 1
        }
        await vm.generation?.value
        let reply = vm.messages.last { $0.role == .assistant }?.text ?? ""
        FileHandle.standardOutput.write(Data((reply + "\n").utf8))
        errln("[selftest] \(vm.statusLine)")
        if case .failed(let why) = vm.phase {
            errln("selftest: generation failed (\(why))")
            return 1
        }
        return reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 1 : 0
    }
}
