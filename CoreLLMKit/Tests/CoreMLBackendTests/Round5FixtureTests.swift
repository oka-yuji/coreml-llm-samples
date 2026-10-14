import CoreML
import Foundation
import LLMCore
import Testing
@testable import CoreMLBackend

// Teacher-forced greedy gate for the round5 op-mix bundle: feed fixture `ids` one token at a
// time through the chunked chain's decode function and require every argmax to equal the
// fixture's `mac_token_id`. Set CORELLM_ROUND5_BUNDLE to the bundle directory; the fixture
// defaults to `fixture_testio_h6.json` inside it (override with CORELLM_ROUND5_FIXTURE).

private let round5BundleEnv = "CORELLM_ROUND5_BUNDLE"
private let round5FixtureEnv = "CORELLM_ROUND5_FIXTURE"
private let round5UnitsEnv = "CORELLM_ROUND5_UNITS"

private let round5URL: URL? = ProcessInfo.processInfo.environment[round5BundleEnv]
    .flatMap { $0.isEmpty ? nil : URL(filePath: $0) }

private var round5Exists: Bool {
    guard let round5URL else { return false }
    return FileManager.default.fileExists(
        atPath: round5URL.appending(path: "convert_config.json").path(percentEncoded: false))
}

private let round5Skip: Comment = "set \(round5BundleEnv) to the round5 bundle directory to run this test"

@Suite
struct Round5FixtureSuite {

    @Test("round5 chunked chain reproduces the fixture's 21 teacher-forced positions", .enabled(if: round5Exists))
    func teacherForcedPositions() async throws {
        let bundle = try #require(round5URL, round5Skip)
        let fixtureURL = ProcessInfo.processInfo.environment[round5FixtureEnv].flatMap {
            $0.isEmpty ? nil : URL(filePath: $0)
        } ?? bundle.appending(path: "fixture_testio_h6.json")

        let json = try JSONSerialization.jsonObject(
            with: Data(contentsOf: fixtureURL)) as? [String: Any] ?? [:]
        let ids = try #require(json["ids"] as? [Int], "fixture has no ids")
        let expected = try #require(json["mac_token_id"] as? [Int], "fixture has no mac_token_id")
        #expect(ids.count == expected.count)

        let units: MLComputeUnits
        switch ProcessInfo.processInfo.environment[round5UnitsEnv]?.lowercased() {
        case "cpu": units = .cpuOnly
        case "gpu": units = .cpuAndGPU
        case "all": units = .all
        default: units = .cpuAndNeuralEngine
        }

        let chain = try await ChunkedSpeculativeChain(
            bundleURL: bundle, computeUnits: units, preloadVerifyAssets: false)
        try chain.reset()

        var got: [Int] = []
        for id in ids { got.append(try chain.decodeStep(tokenID: id)) }

        let diffs = zip(got.indices, zip(got, expected)).filter { $0.1.0 != $0.1.1 }
        let report = diffs.map { "pos\($0.0): got \($0.1.0) want \($0.1.1)" }.joined(separator: ", ")
        print("[round5] units=\(ChunkedSpeculativeChain.cuName(units)) positions=\(ids.count) "
            + "mismatches=\(diffs.count) \(report)")
        print("[round5] got=\(got)")
        #expect(diffs.isEmpty, "teacher-forced mismatches: \(report)")
    }
}

// Early exit (`prefillEarlyExit`) against the full prefill, on the same bundle and compute units:
// every prompt length must give the same first token and greedy continuation, and the KV-sharing
// chunks' prefill functions must never load. The lengths cover each way a prompt can end: decode
// steps only (21), 32-row blocks ending exactly on a block (224), blocks plus a padded 32-row block
// for the remainder (250), the plain 512 block alone (512), plain plus a padded 32-row block (530),
// and plain plus one 32-row block (544). Same CORELLM_ROUND5_BUNDLE gate as above; leave
// CORELLM_PREFILL_EARLY_EXIT unset.
@Suite
struct Round5EarlyExitSuite {

    @Test("early exit matches the full prefill at every way a prompt can end", .enabled(if: round5Exists))
    func earlyExitMatchesFullPrefill() async throws {
        let bundle = try #require(round5URL, round5Skip)
        try #require(ProcessInfo.processInfo.environment["CORELLM_PREFILL_EARLY_EXIT"] == nil,
                     "CORELLM_PREFILL_EARLY_EXIT would override both chains")
        let units: MLComputeUnits = ProcessInfo.processInfo.environment[round5UnitsEnv]?.lowercased() == "cpu"
            ? .cpuOnly : .cpuAndNeuralEngine

        let tokenizer = try await HFTokenizer(modelFolder: bundle, eosTokenIDs: [])
        let head = [try #require(tokenizer.bosTokenID)] + (try tokenizer.encode("<|turn>user\n"))
        let tail = try tokenizer.encode("<turn|>\n<|turn>model\n")
        let body = try tokenizer.encode(String(repeating:
            "オンデバイス推論では、計算速度よりも先にメモリの上限が効いてくることが多い。"
            + "重みを 4 bit まで落とせば容量は半分になるが、展開時に LUT を引くコストがかかる。", count: 16))
        let lengths = [21, 224, 250, 512, 530, 544]
        let prompts = lengths.map { head + body.prefix($0 - head.count - tail.count) + tail }
        #expect(prompts.map(\.count) == lengths)

        func run(earlyExit: Bool) async throws -> (tokens: [[Int]], loads: String) {
            let chain = try await ChunkedSpeculativeChain(
                bundleURL: bundle, computeUnits: units, preloadVerifyAssets: false,
                prefillMode: "sequential", prefillEarlyExit: earlyExit)
            var tokens: [[Int]] = []
            for ids in prompts {
                try chain.reset()
                let seed = try chain.prefillScheduled(ids)
                #expect(chain.position == ids.count)
                tokens.append(try chain.continueGreedy(seed: seed, maxNew: 16, eos: []))
            }
            return (tokens, chain.prefillLoadSummary)
        }
        let full = try await run(earlyExit: false)
        let early = try await run(earlyExit: true)

        for (i, n) in lengths.enumerated() {
            print("[earlyexit] L=\(n) \(full.tokens[i] == early.tokens[i] ? "same" : "DIFF") "
                + "full=\(full.tokens[i]) early=\(early.tokens[i])")
        }
        print("[earlyexit] loads full: \(full.loads)")
        print("[earlyexit] loads early: \(early.loads)")
        #expect(early.tokens == full.tokens)
        #expect(!early.loads.contains("chunk_15_25") && !early.loads.contains("chunk_25_35"))
    }
}
