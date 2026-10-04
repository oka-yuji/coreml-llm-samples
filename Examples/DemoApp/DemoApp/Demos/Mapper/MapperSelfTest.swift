import CoreMLBackend
import Foundation

/// The 30-sentence fixture: expectations for the accuracy run, plus the Python
/// arrays and scores for the on-device parity check.
struct MapperFixture: Decodable, Sendable {
    struct Call: Decodable, Sendable {
        var text: String
        var task: String
        var labels: [String]
        var input_ids: [Int32]
        var attention_mask: [Int32]
        var marker_indices: [Int32]
        var marker_mask: [Float]
        var probabilities: [Float]
    }
    struct Case: Decodable, Sendable {
        var text: String
        var expected: [String: String]
    }
    var length: Int
    var maxOptions: Int
    var calls: [Call]
    var cases: [Case]

    static func load() throws -> MapperFixture {
        guard let url = Bundle.main.url(forResource: "gliner_parity", withExtension: "json") else {
            throw MapperError.missingResource("gliner_parity.json")
        }
        return try JSONDecoder().decode(MapperFixture.self, from: Data(contentsOf: url))
    }
}

enum MapperError: Error, CustomStringConvertible {
    case missingResource(String)

    var description: String {
        switch self {
        case .missingResource(let name): "\(name) is not in the app bundle"
        }
    }
}

/// Per-field accuracy over the 30 sentences, scored with the thresholds exactly as
/// specified — never tuned to the expectations.
struct MapperSelfTestReport: Sendable {
    struct FieldScore: Sendable {
        var key: String
        var name: String
        var matched: Int
        var total: Int
        var rate: Double { total == 0 ? 0 : Double(matched) / Double(total) }
    }

    var fieldScores: [FieldScore] = []
    var allFieldsMatched = 0
    var sentenceCount = 0
    /// Reference only: of list fields that do have a mention, how often the expected
    /// value is one of the top two labels. Not a rule change — the status rules are
    /// untouched; this just shows how often the right answer was in contention.
    var top2Matched = 0
    var top2Total = 0
    /// Max |swift − python| over every fixture score, measured on this device.
    var worstScoreDelta: Float = 0
    var arrayMismatches = 0
    var parityCalls = 0
    var scoresOverBudget = 0
    var argmaxDisagreements = 0
    var failures: [String] = []

    var allFieldsRate: Double {
        sentenceCount == 0 ? 0 : Double(allFieldsMatched) / Double(sentenceCount)
    }
    var top2Rate: Double { top2Total == 0 ? 0 : Double(top2Matched) / Double(top2Total) }
}

/// Timings and memory for the demo's measurement row.
struct MapperMetrics: Sendable {
    var modelLoadSeconds: Double = 0
    var secondLoadSeconds: Double?
    var footprintBeforeLoad: Int?
    var footprintAfterLoad: Int?
    var footprintPeak: Int?
    var perSentenceMedianMs: Double?
    var perSentenceP95Ms: Double?
    var repetitions = 0
}

struct MapperSelfTest: Sendable {
    let mapper: GLiNERMapper
    let fixture: MapperFixture

    struct ParityResult: Sendable {
        var calls = 0
        var arrayMismatches = 0
        var worstDelta: Float = 0
        var worstDetail = ""
        var scoresCompared = 0
        /// Scores further than 0.02 from Python — the budget the brief set.
        var scoresOverBudget = 0
        /// The decision-level question: does the winning label still agree?
        var argmaxDisagreements = 0
        var meanDelta: Float = 0
    }

    /// Compares every fixture call's input arrays and Core ML scores against the
    /// Python reference. This is how the iPhone↔Python agreement is checked.
    func runParity() -> ParityResult {
        var result = ParityResult()
        result.calls = fixture.calls.count
        var deltaSum: Float = 0
        let encoder = mapper.classifier.encoder
        for call in fixture.calls {
            guard let inputs = try? encoder.encode(
                text: call.text, task: call.task, labels: call.labels) else {
                result.arrayMismatches += 1
                continue
            }
            if inputs.inputIDs != call.input_ids || inputs.attentionMask != call.attention_mask
                || inputs.markerIndices != call.marker_indices || inputs.markerMask != call.marker_mask {
                result.arrayMismatches += 1
            }
            guard let scores = try? mapper.classifier.scores(
                text: call.text, task: call.task, labels: call.labels) else { continue }
            for (index, pair) in zip(scores, call.probabilities).enumerated() {
                let delta = abs(pair.0 - pair.1)
                result.scoresCompared += 1
                deltaSum += delta
                if delta > 0.02 { result.scoresOverBudget += 1 }
                if delta > result.worstDelta {
                    result.worstDelta = delta
                    result.worstDetail = "\(call.task)/\(call.labels[index]) "
                        + "device=\(pair.0) python=\(pair.1) text=「\(call.text)」"
                }
            }
            let deviceBest = scores.firstIndex(of: scores.max() ?? 0)
            let pythonBest = call.probabilities.firstIndex(of: call.probabilities.max() ?? 0)
            if deviceBest != pythonBest { result.argmaxDisagreements += 1 }
        }
        result.meanDelta = result.scoresCompared == 0 ? 0 : deltaSum / Float(result.scoresCompared)
        return result
    }

    /// Runs all 30 sentences through the real mapper and scores them per field.
    func runAccuracy() -> MapperSelfTestReport {
        var report = MapperSelfTestReport()
        var matched: [String: Int] = [:]
        report.sentenceCount = fixture.cases.count

        for testCase in fixture.cases {
            guard let outcome = try? mapper.map(text: testCase.text) else {
                report.failures.append("\(testCase.text): mapping failed")
                continue
            }
            var allMatch = true
            for field in outcome.fields {
                let expected = testCase.expected[field.key]
                let isMatch: Bool
                if let expected {
                    isMatch = field.status != .absent && field.value == expected
                    // Reference metric: list fields only — a number field's labels are
                    // field names, not values, so "top-2 label == expected value"
                    // is not defined for them.
                    if case .list = MapperForm.fields.first(where: { $0.key == field.key })?.kind {
                        report.top2Total += 1
                        if field.topTwo.contains(where: { $0.label == expected }) {
                            report.top2Matched += 1
                        }
                    }
                } else {
                    isMatch = field.status == .absent
                }
                if isMatch {
                    matched[field.key, default: 0] += 1
                } else {
                    allMatch = false
                    report.failures.append(
                        "\(testCase.text) / \(field.name): got "
                        + "\(field.value.isEmpty ? "未言及" : field.value)(\(String(format: "%.2f", field.score)))"
                        + " expected \(expected ?? "未言及")")
                }
            }
            if allMatch { report.allFieldsMatched += 1 }
        }

        report.fieldScores = MapperForm.fields.map {
            .init(key: $0.key, name: $0.name, matched: matched[$0.key] ?? 0,
                  total: fixture.cases.count)
        }
        return report
    }
}
