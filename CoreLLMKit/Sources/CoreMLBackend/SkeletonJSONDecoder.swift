import Foundation
import LLMCore

protocol ForcedInjectionChain: AnyObject {

    func feedNoPredict(tokenID: Int) throws
}

enum SkeletonFieldType: Sendable, Equatable {
    case string
    case date
    case number
    case enumeration([String])
}

struct SkeletonField: Sendable {
    let key: String
    let type: SkeletonFieldType
}

struct SkeletonSchema: Sendable {
    let fields: [SkeletonField]

    init(fields: [SkeletonField]) { self.fields = fields }

    static func from(json: [String: Any]) throws -> SkeletonSchema {
        guard let order = json["key_order"] as? [String] else {
            throw LLMEngineError.incompatibleBundle(reason: "skeleton schema: key_order is missing")
        }
        let defs = (json["fields"] as? [String: Any]) ?? [:]
        var fields: [SkeletonField] = []
        for key in order {
            let def = defs[key] as? [String: Any]
            let typeStr = (def?["type"] as? String) ?? "string"
            let type: SkeletonFieldType
            switch typeStr {
            case "number": type = .number
            case "date": type = .date
            case "enum":
                let values = (def?["values"] as? [String]) ?? []
                type = .enumeration(values)
            default: type = .string
            }
            fields.append(SkeletonField(key: key, type: type))
        }
        return SkeletonSchema(fields: fields)
    }
}

enum SkeletonValue: Sendable, Equatable {
    case null
    case string(String)
    case number(String)
    case enumeration(String)
}

struct SkeletonStats: Sendable {
    var injectedTokens = 0
    var valueTokens = 0
    var fieldDemotions = 0
    var headSkips = 0
}

final class SkeletonJSONDecoder {
    let schema: SkeletonSchema
    let maxStringSpanTokens: Int
    let maxNumberSpanTokens: Int
    let useHeadSkip: Bool

    private let tokenizer: HFTokenizer
    private let startSegment: [Int]
    private let sepWithComma: [[Int]]
    private let sepNoComma: [[Int]]
    private let endSegment: [Int]

    init(
        schema: SkeletonSchema,
        tokenizer: HFTokenizer,
        maxStringSpanTokens: Int = 48,
        maxNumberSpanTokens: Int = 16,
        useHeadSkip: Bool = true
    ) throws {
        self.schema = schema
        self.tokenizer = tokenizer
        self.maxStringSpanTokens = maxStringSpanTokens
        self.maxNumberSpanTokens = maxNumberSpanTokens
        self.useHeadSkip = useHeadSkip

        let keys = schema.fields.map { $0.key }
        guard let first = keys.first else {
            throw LLMEngineError.incompatibleBundle(reason: "skeleton schema: fields are empty")
        }

        startSegment = try tokenizer.encode("{\"\(first)\":")

        var withC: [[Int]] = []
        var noC: [[Int]] = []
        for key in keys.dropFirst() {
            let seg = try tokenizer.encode(", \"\(key)\":")
            withC.append(seg)

            if seg.count > 1 { noC.append(Array(seg.dropFirst())) } else { noC.append(seg) }
        }
        sepWithComma = withC
        sepNoComma = noC
        endSegment = try tokenizer.encode("}")
    }

    func decode(
        chain: any GenerationChain,
        firstPending: Int
    ) throws -> (values: [(String, SkeletonValue)], stats: SkeletonStats) {
        var stats = SkeletonStats()
        var pending = firstPending
        var values: [(String, SkeletonValue)] = []
        var prevEndedWithComma = false

        for (idx, field) in schema.fields.enumerated() {
            let segment: [Int]
            if idx == 0 {
                segment = startSegment
            } else {
                segment = prevEndedWithComma ? sepNoComma[idx - 1] : sepWithComma[idx - 1]
            }

            pending = try inject(chain: chain, segment: segment, stats: &stats)

            let outcome = try generateValueSpan(chain: chain, field: field, firstPending: pending, stats: &stats)
            pending = outcome.pending
            values.append((field.key, outcome.value))
            prevEndedWithComma = outcome.endedWithComma
            if outcome.demoted { stats.fieldDemotions += 1 }
        }

        return (values, stats)
    }

    private func inject(chain: any GenerationChain, segment: [Int], stats: inout SkeletonStats) throws -> Int {
        var pending = 0
        let injector = useHeadSkip ? (chain as? ForcedInjectionChain) : nil
        for (i, tok) in segment.enumerated() {
            if i < segment.count - 1, let injector {
                try injector.feedNoPredict(tokenID: tok)
                stats.headSkips += 1
            } else {
                pending = try chain.decodeStep(tokenID: tok)
            }
            stats.injectedTokens += 1
        }
        return pending
    }

    private struct SpanOutcome {
        let value: SkeletonValue
        let endedWithComma: Bool
        let pending: Int
        let demoted: Bool
    }

    private func generateValueSpan(
        chain: any GenerationChain,
        field: SkeletonField,
        firstPending: Int,
        stats: inout SkeletonStats
    ) throws -> SpanOutcome {
        let maxTok = (field.type == .number) ? maxNumberSpanTokens : maxStringSpanTokens
        var spanTokens: [Int] = []
        var pending = firstPending

        for _ in 0..<maxTok {
            if tokenizer.eosTokenIDs.contains(pending) { break }
            let fed = pending
            pending = try chain.decodeStep(tokenID: fed)
            spanTokens.append(fed)
            stats.valueTokens += 1
            let text = try tokenizer.decode(spanTokens)
            switch Self.evaluate(text: text, type: field.type) {
            case .incomplete:
                continue
            case .complete(let v):
                return SpanOutcome(value: v, endedWithComma: Self.endsWithComma(text), pending: pending, demoted: false)
            case .demote:
                return SpanOutcome(value: .null, endedWithComma: Self.endsWithComma(text), pending: pending, demoted: true)
            }
        }

        let text = spanTokens.isEmpty ? "" : (try tokenizer.decode(spanTokens))
        let (value, demoted) = Self.finalize(text: text, type: field.type)
        return SpanOutcome(value: value, endedWithComma: Self.endsWithComma(text), pending: pending, demoted: demoted)
    }

    enum SpanStatus {
        case incomplete
        case complete(SkeletonValue)
        case demote
    }

    static func evaluate(text: String, type: SkeletonFieldType) -> SpanStatus {
        let t = trimLeadingWhitespace(text)
        if t.isEmpty { return .incomplete }

        if t.hasPrefix("null") { return .complete(.null) }
        if "null".hasPrefix(t) { return .incomplete }

        switch type {
        case .number:
            return evaluateNumber(t)
        case .string, .date, .enumeration:
            return evaluateStringLike(t, type: type)
        }
    }

    private static func evaluateStringLike(_ t: String, type: SkeletonFieldType) -> SpanStatus {
        let chars = Array(t)
        guard chars.first == "\"" else { return .demote }
        guard let close = closingQuoteIndex(chars, from: 1) else { return .incomplete }
        let content = unescapeJSON(String(chars[1..<close]))
        switch type {
        case .enumeration(let allowed):
            let norm = content.trimmingCharacters(in: .whitespacesAndNewlines)
            return allowed.contains(norm) ? .complete(.enumeration(norm)) : .demote
        default:
            return .complete(.string(content))
        }
    }

    private static func evaluateNumber(_ t: String) -> SpanStatus {
        let chars = Array(t)

        if chars.first == "\"" {
            guard let close = closingQuoteIndex(chars, from: 1) else { return .incomplete }
            let inner = String(chars[1..<close]).trimmingCharacters(in: .whitespacesAndNewlines)
            if let num = canonicalNumber(inner) { return .complete(.number(num)) }
            return .demote
        }

        let (run, terminated) = scanNumberRun(chars)
        if run.isEmpty { return .demote }
        if terminated {
            if let num = canonicalNumber(run) { return .complete(.number(num)) }
            return .demote
        }
        return .incomplete
    }

    static func finalize(text: String, type: SkeletonFieldType) -> (SkeletonValue, Bool) {
        let t = trimLeadingWhitespace(text)
        if t.isEmpty { return (.null, false) }
        if t.hasPrefix("null") || "null".hasPrefix(t) { return (.null, false) }
        switch type {
        case .number:
            let (run, _) = scanNumberRun(Array(t))
            if let num = canonicalNumber(run) { return (.number(num), false) }
            return (.null, true)
        case .string, .date, .enumeration:

            return (.null, true)
        }
    }

    func assembleJSON(_ values: [(String, SkeletonValue)]) -> String {
        var parts: [String] = []
        for (key, value) in values {
            let vs: String
            switch value {
            case .null: vs = "null"
            case .string(let s): vs = Self.jsonStringLiteral(s)
            case .enumeration(let s): vs = Self.jsonStringLiteral(s)
            case .number(let n): vs = n
            }
            parts.append(Self.jsonStringLiteral(key) + ": " + vs)
        }
        return "{" + parts.joined(separator: ", ") + "}"
    }

    static func selfCheck(json: String, schema: SkeletonSchema) -> (parseOK: Bool, keysComplete: Bool) {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return (false, false)
        }
        let keys = Set(obj.keys)
        let expected = Set(schema.fields.map { $0.key })
        return (true, keys == expected)
    }

    static func trimLeadingWhitespace(_ s: String) -> String {
        let chars = Array(s)
        var i = 0
        while i < chars.count, chars[i] == " " || chars[i] == "\n" || chars[i] == "\t" || chars[i] == "\r" { i += 1 }
        return String(chars[i...])
    }

    static func endsWithComma(_ s: String) -> Bool {
        for ch in s.reversed() {
            if ch == " " || ch == "\n" || ch == "\t" || ch == "\r" { continue }
            return ch == ","
        }
        return false
    }

    private static func closingQuoteIndex(_ chars: [Character], from start: Int) -> Int? {
        var i = start
        var escaped = false
        while i < chars.count {
            let c = chars[i]
            if escaped { escaped = false }
            else if c == "\\" { escaped = true }
            else if c == "\"" { return i }
            i += 1
        }
        return nil
    }

    private static func scanNumberRun(_ chars: [Character]) -> (String, Bool) {
        let numeric: Set<Character> = ["-", "+", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "e", "E"]
        var i = 0
        while i < chars.count, numeric.contains(chars[i]) { i += 1 }
        let run = String(chars[0..<i])
        let terminated = i < chars.count
        return (run, terminated)
    }

    static func canonicalNumber(_ raw: String) -> String? {
        var s = raw
        while s.hasSuffix(".") { s.removeLast() }
        if s.hasPrefix("+") { s.removeFirst() }
        if s.hasPrefix(".") { s = "0" + s }
        if s.hasPrefix("-.") { s = "-0" + s.dropFirst() }
        if s.isEmpty || Double(s) == nil { return nil }
        let pattern = "^-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$"
        if s.range(of: pattern, options: .regularExpression) != nil { return s }

        guard let d = Double(s) else { return nil }
        if d == d.rounded() && abs(d) < 1e15 { return String(Int(d)) }
        return String(d)
    }

    static func jsonStringLiteral(_ s: String) -> String {
        var out = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
        return out
    }

    private static func unescapeJSON(_ s: String) -> String {
        guard s.contains("\\") else { return s }
        var out = ""
        let chars = Array(s)
        var i = 0
        while i < chars.count {
            if chars[i] == "\\", i + 1 < chars.count {
                let n = chars[i + 1]
                switch n {
                case "\"": out += "\""
                case "\\": out += "\\"
                case "n": out += "\n"
                case "r": out += "\r"
                case "t": out += "\t"
                case "/": out += "/"
                default: out.append(n)
                }
                i += 2
            } else {
                out.append(chars[i]); i += 1
            }
        }
        return out
    }
}
