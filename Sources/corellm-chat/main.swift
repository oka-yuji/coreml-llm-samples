import CoreMLBackend
import Foundation
import LLMCore

let ctx32kWindow = 32768

struct Options {
    var modelPath: String?
    var prompt: String?
    var maxTokens = 0
    var mtp = true
    var stats = false
    var compute: String?
    var kvSave: String?
    var kvRestore: String?
    var kvSpec = false
}

func parseArguments(_ argv: [String]) -> Options {
    var opts = Options()
    var i = 0
    func value(after flag: String, inline: String?) -> String {
        if let inline { return inline }
        i += 1
        guard i < argv.count else {
            fail("missing value for \(flag)")
        }
        return argv[i]
    }
    while i < argv.count {
        let arg = argv[i]

        let (name, inline): (String, String?) = {
            if let eq = arg.firstIndex(of: "="), arg.hasPrefix("--") {
                return (String(arg[..<eq]), String(arg[arg.index(after: eq)...]))
            }
            return (arg, nil)
        }()
        switch name {
        case "--model": opts.modelPath = value(after: name, inline: inline)
        case "--prompt": opts.prompt = value(after: name, inline: inline)
        case "--max-tokens":
            let raw = value(after: name, inline: inline)
            guard let n = Int(raw), n >= 0 else { fail("--max-tokens expects a non-negative integer, got \(raw)") }
            opts.maxTokens = n
        case "--no-mtp": opts.mtp = false
        case "--stats": opts.stats = true
        case "--compute": opts.compute = value(after: name, inline: inline)
        case "--kv-save": opts.kvSave = value(after: name, inline: inline)
        case "--kv-restore": opts.kvRestore = value(after: name, inline: inline)
        case "--kv-spec": opts.kvSpec = true
        case "--help", "-h": printUsage(); exit(0)
        default: fail("unknown argument: \(arg)")
        }
        i += 1
    }
    return opts
}

func printUsage() {
    let usage = """
    corellm-chat — Core ML streaming chat (Gemma 4 12B, 128K Context Ladder)

    USAGE:
      corellm-chat --model <bundle-dir> [--prompt "…"] [options]

    OPTIONS:
      --model <dir>       Path to the model bundle directory (contains manifest.json). Required.
      --prompt "<text>"   Generate one response for <text> and exit. Omit for an interactive REPL.
      --max-tokens <n>    Max tokens per turn. Default: 0 = unlimited (until EOS or context is full).
      --no-mtp            Disable speculative decoding. Default: ON when the bundle supports it.
      --compute <units>   all | cpuAndGPU | cpuAndNeuralEngine | cpuOnly. Overrides the manifest.
      --stats             Print TTFT, decode ms/tok, tok/s, and draft acceptance after generation.
      --kv-save <dir>     Prefill --prompt once, dump the KV cache to <dir>, and exit.
      --kv-restore <dir>  Restore a KV cache from <dir> and continue decoding (no prefill), then exit.
      --kv-spec           Use speculative decoding for the --kv-restore continuation.
      -h, --help          Show this help.

    In interactive mode, each line you type is one turn; KV is reused across turns.
    Type /reset to start a fresh conversation, Ctrl-D to exit.
    """
    err(usage + "\n")
}

func err(_ s: String) { FileHandle.standardError.write(Data(s.utf8)) }
func out(_ s: String) { FileHandle.standardOutput.write(Data(s.utf8)) }
func fail(_ message: String) -> Never {
    err("error: \(message)\n")
    exit(2)
}

func fmtSeconds(_ d: Duration) -> Double { d / .seconds(1) }

func printKVInfo(_ info: KVCheckpointInfo, mode: String) {
    var lines = [
        "--- kv \(mode) ---",
        "position:          \(info.position)",
        "committed tokens:  \(info.tokenCount)",
        "kv bytes:          \(info.fileBytes)",
        "own layers:        \(info.layerCount)",
        "resident prefill:  \(info.residentPrefillWidths)",
    ]
    if let s = info.prefillSeconds { lines.append("prefill:           \(String(format: "%.3f", s))s") }
    if let s = info.exportSeconds { lines.append("export:            \(String(format: "%.3f", s))s") }
    if let s = info.importSeconds { lines.append("import:            \(String(format: "%.3f", s))s") }
    if let n = info.continuation?.count { lines.append("continued tokens:  \(n)") }
    if let r = info.pldRounds { lines.append("pld rounds:        \(r)") }
    if let peak = info.peakMemoryBytes {
        lines.append("peak memory:       \(String(format: "%.0f", Double(peak) / 1_048_576)) MB")
    }
    err(lines.joined(separator: "\n") + "\n")
}

func consume(
    _ engine: CoreMLEngine,
    _ request: GenerationRequest,
    stats: Bool,
    warmupHint: Bool
) async throws -> (text: String, metrics: GenerationMetrics?) {
    var assistantText = ""
    var promptTokens = 0
    var emittedApprox = 0
    var promotionAnnounced = false
    var finalMetrics: GenerationMetrics?

    if warmupHint {
        err("[warming up: first inference specializes GPU kernels, this can take ~40s]\n")
    }

    for try await event in engine.generate(request) {
        switch event {
        case .loadCompleted(let m):
            err("[model loaded in \(String(format: "%.1f", fmtSeconds(m.duration)))s]\n")
        case .prefillCompleted(let p):
            promptTokens = p.promptTokens
            if stats {
                let reused = p.reusedTokens > 0 ? " (reused \(p.reusedTokens) from KV cache)" : ""
                err("[prefill: \(p.promptTokens) prompt tokens\(reused) in "
                    + "\(String(format: "%.2f", fmtSeconds(p.duration)))s]\n")
            }
            if !promotionAnnounced, promptTokens >= ctx32kWindow {
                err("[context \(promptTokens) ≥ \(ctx32kWindow): promoted to ctx128k mode "
                    + "(zero-copy, ~3 tok/s)]\n")
                promotionAnnounced = true
            }
        case .token(let chunk):
            out(chunk.text)
            assistantText += chunk.text
            emittedApprox += 1
            if !promotionAnnounced, promptTokens + emittedApprox >= ctx32kWindow {
                err("\n[context reached \(ctx32kWindow): promoted to ctx128k mode "
                    + "(zero-copy, ~3 tok/s)]\n")
                promotionAnnounced = true
            }
        case .finished(let m):
            finalMetrics = m
        }
    }
    return (assistantText, finalMetrics)
}

func printStats(_ m: GenerationMetrics, mtp: Bool) {
    let msPerTok = m.decodeTokensPerSecond > 0 ? 1000.0 / m.decodeTokensPerSecond : 0
    var lines = [
        "--- stats ---",
        "prompt tokens:     \(m.promptTokens)\(m.reusedTokens > 0 ? " (reused \(m.reusedTokens))" : "")",
        "generated tokens:  \(m.generatedTokens)",
        "finish reason:     \(m.finishReason?.rawValue ?? "?")",
        "TTFT:              \(String(format: "%.2f", fmtSeconds(m.timeToFirstToken)))s",
        "decode:            \(String(format: "%.1f", msPerTok)) ms/tok  "
            + "(\(String(format: "%.1f", m.decodeTokensPerSecond)) tok/s)",
    ]
    if mtp {
        if let acc = m.draftAcceptanceRate {
            lines.append("draft acceptance:  \(String(format: "%.2f", acc))")
        } else {
            lines.append("draft acceptance:  n/a (no speculation this turn)")
        }
    }
    if let peak = m.peakMemoryBytes {
        lines.append("peak memory:       \(String(format: "%.2f", Double(peak) / 1_073_741_824)) GB")
    }
    err(lines.joined(separator: "\n") + "\n")
}

func runMain() async throws {
    let opts = parseArguments(Array(CommandLine.arguments.dropFirst()))
    guard let modelPath = opts.modelPath else {
        printUsage()
        fail("--model is required")
    }

    let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
    guard FileManager.default.fileExists(atPath: modelURL.appending(path: "manifest.json").path(percentEncoded: false)) else {
        fail("no manifest.json under \(modelURL.path(percentEncoded: false)) — is this a model bundle directory?")
    }

    err("Loading model bundle: \(modelURL.path(percentEncoded: false))\n")
    let bundle = try ModelBundle(contentsOf: modelURL)
    err("  \(bundle.manifest.name)\n")
    err("  context length: \(bundle.manifest.contextLength)\n")

    let preference: ComputeUnitPreference = opts.compute.flatMap(ComputeUnitPreference.init(rawValue:))
        ?? bundle.manifest.computeUnits.flatMap(ComputeUnitPreference.init(rawValue:))
        ?? .cpuAndGPU
    let engine = CoreMLEngine()
    let loadStart = ContinuousClock().now
    err("  loading Core ML models (\(preference.rawValue))…\n")
    try await engine.load(bundle, options: LoadOptions(computeUnits: preference))
    err("  models loaded in \(String(format: "%.1f", fmtSeconds(ContinuousClock().now - loadStart)))s\n")

    let specAvailable = await engine.supportsSpeculation

    if let dir = opts.kvSave {
        guard let prompt = opts.prompt else { fail("--kv-save requires --prompt") }
        let info = try await engine.kvSave(to: URL(fileURLWithPath: dir, isDirectory: true), prompt: prompt)
        printKVInfo(info, mode: "save")
        return
    }
    if let dir = opts.kvRestore {
        let info = try await engine.kvRestoreAndContinue(
            from: URL(fileURLWithPath: dir, isDirectory: true),
            verifyPrompt: opts.prompt, maxNew: opts.maxTokens, speculative: opts.kvSpec && specAvailable)
        printKVInfo(info, mode: "restore")
        if let text = info.continuationText, !text.isEmpty { out(text + "\n") }
        return
    }

    let mtp = opts.mtp && specAvailable
    if opts.mtp && !specAvailable {
        err("  note: this bundle has no speculation assets — running without speculation\n")
    }
    let config = GenerationConfig(
        maxNewTokens: opts.maxTokens, temperature: 0, multiTokenPrediction: mtp)
    err("  speculative decoding: \(mtp ? "ON" : "OFF")\n\n")

    if let prompt = opts.prompt {

        let request = GenerationRequest(prompt: prompt, config: config, history: [], reuseCache: false)
        let (_, metrics) = try await consume(engine, request, stats: opts.stats, warmupHint: true)
        out("\n")
        if opts.stats, let metrics { printStats(metrics, mtp: mtp) }
        return
    }

    err("Interactive chat. Each line is one turn; KV is reused across turns.\n")
    err("Commands: /reset (new conversation), Ctrl-D to exit.\n")
    var history: [ChatTurn] = []
    var firstTurn = true
    while true {
        err("\nYou> ")
        guard let line = readLine(strippingNewline: true) else {
            err("\n")
            break
        }
        let userText = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if userText.isEmpty { continue }
        if userText == "/reset" {
            await engine.resetConversation()
            history = []
            err("[conversation reset]\n")
            continue
        }
        err("Assistant> ")
        let request = GenerationRequest(
            prompt: userText, config: config, history: history, reuseCache: true)
        let (assistant, metrics) = try await consume(
            engine, request, stats: opts.stats, warmupHint: firstTurn)
        firstTurn = false
        out("\n")
        history.append(ChatTurn(role: .user, text: userText))
        history.append(ChatTurn(role: .assistant, text: assistant))
        if opts.stats, let metrics { printStats(metrics, mtp: mtp) }
    }
}

do {
    try await runMain()
} catch {
    err("error: \(error)\n")
    exit(1)
}
