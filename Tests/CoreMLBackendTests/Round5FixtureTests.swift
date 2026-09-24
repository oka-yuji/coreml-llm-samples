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
