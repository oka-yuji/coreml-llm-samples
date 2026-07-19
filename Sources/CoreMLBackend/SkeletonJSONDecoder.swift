import Foundation
import LLMCore

// MARK: - 任意最適化フック: lm_head を呼ばない注入ステップ

/// 構造セグメント注入を「lm_head 抜き」で回せるチェーンが適合する任意プロトコル。
///
/// スケルトンデコードでは構造トークン(キー・括弧・カンマ)を強制注入するが、その argmax は捨てる
/// ため lm_head の計算は無駄になる。適合チェーンはセグメント途中トークンで lm_head をスキップでき、
/// predict/token を削減できる。適合チェーンが無い場合は `decodeStep` に素直にフォールバックする
/// (挙動不変。このサンプルの v2 stateful チェーンは適合しないため常にフォールバック経路)。
protocol ForcedInjectionChain: AnyObject {
    /// トークン 1 つを KV に食わせて位置を 1 進めるが、lm_head(次トークン予測)は計算しない。
    func feedNoPredict(tokenID: Int) throws
}

// MARK: - スキーマ

/// 値スパンの型。構造は同じでも終端規則・降格規則が型ごとに異なる。
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

/// 固定順・固定キーの抽出スキーマ。`tools/compare/assets/schema.json` 形式から構築できる。
struct SkeletonSchema: Sendable {
    let fields: [SkeletonField]

    init(fields: [SkeletonField]) { self.fields = fields }

    /// schema.json(`key_order` + `fields[key].type`(+ enum は `values`))形式の dict から構築。
    static func from(json: [String: Any]) throws -> SkeletonSchema {
        guard let order = json["key_order"] as? [String] else {
            throw LLMEngineError.incompatibleBundle(reason: "skeleton schema: key_order が無い")
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

/// 抽出済みフィールド値(ホストが JSON へ組み立てる素材)。
enum SkeletonValue: Sendable, Equatable {
    case null
    case string(String)
    case number(String)        // 検証済みの生数値テキスト(JSON 数値として妥当)
    case enumeration(String)
}

/// Layer 0 の内訳計測。
struct SkeletonStats: Sendable {
    var injectedTokens = 0      // 強制注入した構造トークンの predict 回数
    var valueTokens = 0         // モデルが生成した値スパントークン数
    var fieldDemotions = 0      // 非空だが採用できず null 降格したフィールド数
    var headSkips = 0           // lm_head をスキップした注入 predict の回数(最適化効果の内訳)
}

// MARK: - スケルトン JSON デコーダ(GenerationChain 汎用)

/// 「構造部はホストが強制注入し、モデルには値スパンだけ生成させる」Layer 0 デコーダ。
///
/// 出力 JSON はホストがフィールド値から組み立てるため、構文有効率は構造的に 100% になる。
/// 値スパンは型別マイクロオートマトンで監視し、逸脱は打ち切り + null 降格する。
/// チェーン非依存(`GenerationChain`)で、`ForcedInjectionChain` 適合時のみ lm_head スキップ最適化を使う。
final class SkeletonJSONDecoder {
    let schema: SkeletonSchema
    let maxStringSpanTokens: Int
    let maxNumberSpanTokens: Int
    let useHeadSkip: Bool

    private let tokenizer: HFTokenizer
    private let startSegment: [Int]
    private let sepWithComma: [[Int]]   // schema.fields[1...] の separator(先頭カンマ有り)
    private let sepNoComma: [[Int]]     // 直前の値が既にカンマを吐いた場合に使う(先頭カンマ無し)
    private let endSegment: [Int]

    /// - Parameters:
    ///   - maxStringSpanTokens: string/date/enum の値スパン最大長。超過で null 降格。
    ///   - maxNumberSpanTokens: number の値スパン最大長。超過は妥当な数値 prefix を採る。
    ///   - useHeadSkip: 構造注入で lm_head をスキップする最適化(ForcedInjectionChain 適合時のみ有効)。
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
            throw LLMEngineError.incompatibleBundle(reason: "skeleton schema: フィールドが空")
        }
        // 開始: `{"<key0>":`
        startSegment = try tokenizer.encode("{\"\(first)\":")
        // separator: `, "<key_i>":`(先頭カンマ有り)と、その先頭カンマを外した版。
        var withC: [[Int]] = []
        var noC: [[Int]] = []
        for key in keys.dropFirst() {
            let seg = try tokenizer.encode(", \"\(key)\":")
            withC.append(seg)
            // 先頭がカンマトークンなら外す(モデルが直前値で `",` / `,` を吐いた場合の in-distribution 継続)。
            if seg.count > 1 { noC.append(Array(seg.dropFirst())) } else { noC.append(seg) }
        }
        sepWithComma = withC
        sepNoComma = noC
        endSegment = try tokenizer.encode("}")
    }

    // MARK: - デコード本体

    /// prefill 済みチェーンに Layer 0 を適用する。`firstPending` は prefill が返したモデル予測。
    /// 構造は強制注入(argmax は捨てる)、値スパンだけモデル argmax を採用して監視する。
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
            // 構造セグメント注入: 最後のトークンだけ lm_head 込み(次の値の先頭予測が要る)。
            pending = try inject(chain: chain, segment: segment, stats: &stats)

            let outcome = try generateValueSpan(chain: chain, field: field, firstPending: pending, stats: &stats)
            pending = outcome.pending
            values.append((field.key, outcome.value))
            prevEndedWithComma = outcome.endedWithComma
            if outcome.demoted { stats.fieldDemotions += 1 }
        }
        // 終端 `}` は出力(ホスト組立)側だけで足りるため KV へ食わせない(predict 節約)。
        return (values, stats)
    }

    /// 構造セグメントを強制注入し、セグメント直後のモデル予測(値の先頭候補)を返す。
    private func inject(chain: any GenerationChain, segment: [Int], stats: inout SkeletonStats) throws -> Int {
        var pending = 0
        let injector = useHeadSkip ? (chain as? ForcedInjectionChain) : nil
        for (i, tok) in segment.enumerated() {
            if i < segment.count - 1, let injector {
                try injector.feedNoPredict(tokenID: tok)   // 途中トークンは lm_head スキップ
                stats.headSkips += 1
            } else {
                pending = try chain.decodeStep(tokenID: tok)  // 最後のトークンは次予測が要る
            }
            stats.injectedTokens += 1
        }
        return pending
    }

    // MARK: - 値スパン監視

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
            if tokenizer.eosTokenIDs.contains(pending) { break }   // EOS/特殊は食わせずスパン終了
            let fed = pending
            pending = try chain.decodeStep(tokenID: fed)           // モデル予測を食わせて前進
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
        // maxTok 超過 or EOS で未完 → 最終確定。
        let text = spanTokens.isEmpty ? "" : (try tokenizer.decode(spanTokens))
        let (value, demoted) = Self.finalize(text: text, type: field.type)
        return SpanOutcome(value: value, endedWithComma: Self.endsWithComma(text), pending: pending, demoted: demoted)
    }

    // MARK: - 型別オートマトン(純関数)

    enum SpanStatus {
        case incomplete
        case complete(SkeletonValue)
        case demote                    // 非空だが妥当でない → null 降格(内訳計測用)
    }

    /// スパンの decode テキストを評価する。先頭空白は許容(`▁` 由来の先頭スペースを吸収)。
    static func evaluate(text: String, type: SkeletonFieldType) -> SpanStatus {
        let t = trimLeadingWhitespace(text)
        if t.isEmpty { return .incomplete }

        // null 分岐(全型共通)。バレ null(引用符なし)のみ。
        if t.hasPrefix("null") { return .complete(.null) }
        if "null".hasPrefix(t) { return .incomplete }   // まだ "n"/"nu"/"nul" と入力中

        switch type {
        case .number:
            return evaluateNumber(t)
        case .string, .date, .enumeration:
            return evaluateStringLike(t, type: type)
        }
    }

    private static func evaluateStringLike(_ t: String, type: SkeletonFieldType) -> SpanStatus {
        let chars = Array(t)
        guard chars.first == "\"" else { return .demote }   // 引用符で開かねば garbage
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
        // モデルが数値を引用符で括った場合の寛容な回収。
        if chars.first == "\"" {
            guard let close = closingQuoteIndex(chars, from: 1) else { return .incomplete }
            let inner = String(chars[1..<close]).trimmingCharacters(in: .whitespacesAndNewlines)
            if let num = canonicalNumber(inner) { return .complete(.number(num)) }
            return .demote
        }
        // バレ数値: 先頭の数値様 run を採る。
        let (run, terminated) = scanNumberRun(chars)
        if run.isEmpty { return .demote }
        if terminated {
            if let num = canonicalNumber(run) { return .complete(.number(num)) }
            return .demote
        }
        return .incomplete   // まだ伸びうる(例: "36." → "36.5")
    }

    /// maxTok 超過 / EOS 未完時の best-effort 確定。
    static func finalize(text: String, type: SkeletonFieldType) -> (SkeletonValue, Bool) {
        let t = trimLeadingWhitespace(text)
        if t.isEmpty { return (.null, false) }                 // 空 = クリーンな null
        if t.hasPrefix("null") || "null".hasPrefix(t) { return (.null, false) }
        switch type {
        case .number:
            let (run, _) = scanNumberRun(Array(t))
            if let num = canonicalNumber(run) { return (.number(num), false) }
            return (.null, true)
        case .string, .date, .enumeration:
            // 閉じ引用符が来ずに打ち切り → null 降格(未完文字列は採らない)。
            return (.null, true)
        }
    }

    // MARK: - JSON 組み立て

    /// 抽出済み値から JSON 文字列を組み立てる(全キー・key_order 順・正しいエスケープ)。
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

    /// 組み立て結果のセルフチェック: 素の JSON としてパース可能 + キー集合がスキーマと完全一致。
    static func selfCheck(json: String, schema: SkeletonSchema) -> (parseOK: Bool, keysComplete: Bool) {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return (false, false)
        }
        let keys = Set(obj.keys)
        let expected = Set(schema.fields.map { $0.key })
        return (true, keys == expected)
    }

    // MARK: - 文字列ユーティリティ

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

    /// chars[start...] から最初の「エスケープされていない `"`」の index を返す。
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

    /// 先頭から数値様(`-+.0-9eE`)の run を採り、(run, その後に非数値文字が来たか)を返す。
    private static func scanNumberRun(_ chars: [Character]) -> (String, Bool) {
        let numeric: Set<Character> = ["-", "+", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "e", "E"]
        var i = 0
        while i < chars.count, numeric.contains(chars[i]) { i += 1 }
        let run = String(chars[0..<i])
        let terminated = i < chars.count      // run の後に非数値文字がある = 数値スパン終了
        return (run, terminated)
    }

    /// 数値 run を JSON 妥当な数値文字列へ正規化(末尾ドット除去・先頭 + 除去・`.5`→`0.5`)。妥当でなければ nil。
    static func canonicalNumber(_ raw: String) -> String? {
        var s = raw
        while s.hasSuffix(".") { s.removeLast() }
        if s.hasPrefix("+") { s.removeFirst() }
        if s.hasPrefix(".") { s = "0" + s }
        if s.hasPrefix("-.") { s = "-0" + s.dropFirst() }
        if s.isEmpty || Double(s) == nil { return nil }
        let pattern = "^-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$"
        if s.range(of: pattern, options: .regularExpression) != nil { return s }
        // 非正準(例: 先頭 0 埋め)は Double 経由で正準化。
        guard let d = Double(s) else { return nil }
        if d == d.rounded() && abs(d) < 1e15 { return String(Int(d)) }
        return String(d)
    }

    /// JSON 文字列リテラル(必要なエスケープ付き)。
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

    /// スパン内容の最小限アンエスケープ(モデルは値内でめったにエスケープしないが安全側に処理)。
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
