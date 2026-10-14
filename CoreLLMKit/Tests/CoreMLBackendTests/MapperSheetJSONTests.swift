import Foundation
import Testing

@testable import CoreMLBackend

// The generated-JSON route to the U1 sheet. No model needed, so these always run.
// Matching is exact by design: these tests pin that a near-miss stays a miss.

private func field(_ fields: [MapperFieldResult], _ key: String) -> MapperFieldResult {
    fields.first { $0.key == key }!
}

@Suite
struct MapperSheetJSONSuite {

    @Test("a clean sheet parses, nulls become 未言及")
    func parsesCleanSheet() throws {
        let reply = """
            {"discharge_pressure": "0.45", "temperature": "72", "oil_leak": null,
             "abnormal_sound": "正常", "vibration": null, "next_action": "給油"}
            """
        let fields = try #require(MapperSheetJSON.parse(reply))
        #expect(fields.count == 6)
        #expect(field(fields, "discharge_pressure").value == "0.45")
        #expect(field(fields, "discharge_pressure").status == .autoFilled)
        #expect(field(fields, "oil_leak").status == .absent)
        #expect(field(fields, "oil_leak").value.isEmpty)
        #expect(field(fields, "vibration").status == .absent)
        #expect(field(fields, "next_action").value == "給油")
        // A number field still shows its unit once it has a value.
        #expect(field(fields, "temperature").unit == "℃")
    }

    @Test("JSON wrapped in prose or a fenced block is still found")
    func findsEmbeddedObject() throws {
        let reply = """
            はい、わかりました。

            ```json
            {"discharge_pressure": null, "temperature": "85", "oil_leak": "なし",
             "abnormal_sound": null, "vibration": null, "next_action": "交換"}
            ```

            以上です。
            """
        let fields = try #require(MapperSheetJSON.parse(reply))
        #expect(field(fields, "temperature").value == "85")
        #expect(field(fields, "next_action").value == "交換")
        #expect(field(fields, "discharge_pressure").status == .absent)
    }

    @Test("a brace inside a string does not end the object")
    func ignoresBracesInStrings() throws {
        let reply =
            #"{"discharge_pressure": "0.5}", "temperature": null, "oil_leak": null, "abnormal_sound": null, "vibration": null, "next_action": null}"#
        let fields = try #require(MapperSheetJSON.parse(reply))
        #expect(field(fields, "discharge_pressure").value == "0.5}")
    }

    @Test("bare JSON numbers render minimally, not as 72.0")
    func rendersNumbers() throws {
        let reply =
            #"{"discharge_pressure": 0.45, "temperature": 72, "oil_leak": null, "abnormal_sound": null, "vibration": null, "next_action": null}"#
        let fields = try #require(MapperSheetJSON.parse(reply))
        #expect(field(fields, "temperature").value == "72")
        #expect(field(fields, "discharge_pressure").value == "0.45")
    }

    @Test("no JSON object at all is a parse failure, not six wrong fields")
    func reportsParseFailure() {
        #expect(MapperSheetJSON.parse("すみません、わかりませんでした。") == nil)
        #expect(MapperSheetJSON.parse("") == nil)
        #expect(MapperSheetJSON.parse("{\"unterminated\": ") == nil)
    }

    @Test("missing keys are 未言及 rather than a parse failure")
    func toleratesMissingKeys() throws {
        let fields = try #require(MapperSheetJSON.parse(#"{"temperature": "72"}"#))
        #expect(field(fields, "temperature").value == "72")
        #expect(field(fields, "oil_leak").status == .absent)
    }

    @Test("scoring is exact: a near miss is a miss")
    func scoringIsExact() throws {
        let reply =
            #"{"discharge_pressure": "0.45MPa", "temperature": "72", "oil_leak": "漏れなし", "abnormal_sound": "異常あり", "vibration": null, "next_action": null}"#
        let fields = try #require(MapperSheetJSON.parse(reply))
        // Right value, wrong spelling — no rescue layer, so these do not count.
        #expect(!MapperScoring.matches(field(fields, "discharge_pressure"), expected: "0.45"))
        #expect(!MapperScoring.matches(field(fields, "oil_leak"), expected: "なし"))
        #expect(!MapperScoring.matches(field(fields, "abnormal_sound"), expected: "異常"))
        // Exactly right.
        #expect(MapperScoring.matches(field(fields, "temperature"), expected: "72"))
        // Not mentioned and not filled.
        #expect(MapperScoring.matches(field(fields, "vibration"), expected: nil))
        // Not filled but it should have been.
        #expect(!MapperScoring.matches(field(fields, "next_action"), expected: "給油"))
        // Filled when nothing was said.
        #expect(!MapperScoring.matches(field(fields, "temperature"), expected: nil))
    }

    @Test("the prompt names every field, every candidate, and the null rule")
    func promptCoversTheForm() {
        let prompt = MapperSheetJSON.prompt(for: "吐出圧0.45MPa")
        for formField in MapperForm.fields {
            #expect(prompt.contains(formField.key))
            #expect(prompt.contains(formField.name))
            if case .list(let options) = formField.kind {
                for option in options { #expect(prompt.contains(option)) }
            }
        }
        #expect(prompt.contains("null"))
        #expect(prompt.contains("吐出圧0.45MPa"))
    }
}
