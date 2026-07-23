import CoreMLBackend
import Foundation
import LLMCore

let ctx32kWindow = 32768

struct Options {
    var modelPath: String?
    var prompt: String?
    var maxTokens = 512
    var mtp = true
    var stats = false
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
            guard let n = Int(raw), n > 0 else { fail("--max-tokens expects a positive integer, got \(raw)") }
            opts.maxTokens = n
        case "--no-mtp": opts.mtp = false
        case "--stats": opts.stats = true
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
      --max-tokens <n>    Maximum tokens to generate per turn. Default: 512.
      --no-mtp            Disable speculative decoding (MTP). Default: ON when a drafter ships.
      --stats             Print TTFT, decode ms/tok, tok/s, and draft acceptance after generation.
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

func drafterPresent(in bundleURL: URL) -> Bool {
    let fm = FileManager.default
    for name in ["drafter_ring.mlmodelc", "drafter.mlmodelc"] {
        if fm.fileExists(atPath: bundleURL.appending(path: name).path()) { return true }
    }
    return false
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
        "prompt tokens:     \(m.promptTokens)",
        "generated tokens:  \(m.generatedTokens)",
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
    guard FileManager.default.fileExists(atPath: modelURL.appending(path: "manifest.json").path()) else {
        fail("no manifest.json under \(modelURL.path()) — is this a model bundle directory?")
    }

    err("Loading model bundle: \(modelURL.path())\n")
    let bundle = try ModelBundle(contentsOf: modelURL)
    let drafterDetected = drafterPresent(in: modelURL)
    err("  \(bundle.manifest.name)\n")
    err("  context length: \(bundle.manifest.contextLength) | "
        + "MTP drafter: \(drafterDetected ? "yes" : "no")\n")

    let engine = CoreMLEngine()
    let loadStart = ContinuousClock().now
    err("  loading Core ML chunks (CPU+GPU)…\n")
    try await engine.load(bundle, options: LoadOptions(computeUnits: .cpuAndGPU))
    err("  chunks loaded in \(String(format: "%.1f", fmtSeconds(ContinuousClock().now - loadStart)))s\n")

    let mtp = opts.mtp && drafterDetected
    if opts.mtp && !drafterDetected {
        err("  note: bundle has no drafter — running without speculation\n")
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
