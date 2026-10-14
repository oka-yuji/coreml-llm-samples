import Foundation

/// The U1 inspection sheet and the rules that turn one Japanese sentence into it.
///
/// Numbers are cut out by code; the model only decides *which* field a number
/// belongs to and *which* option a list field takes. Every number shown in the UI
/// is called a "score" — the model's outputs are uncalibrated, so they are not
/// probabilities and not an accuracy.
public enum MapperForm {

    public enum Kind: Sendable, Equatable {
        case number(unit: String)
        case list(options: [String])
    }

    public struct Field: Sendable, Equatable {
        public let key: String
        public let name: String
        public let kind: Kind
    }

    /// Demo sheet: generic plant-inspection items. No customer data.
    public static let fields: [Field] = [
        Field(key: "discharge_pressure", name: "吐出圧", kind: .number(unit: "MPa")),
        Field(key: "temperature", name: "温度", kind: .number(unit: "℃")),
        Field(key: "oil_leak", name: "油漏れ", kind: .list(options: ["あり", "なし"])),
        Field(key: "abnormal_sound", name: "異音", kind: .list(options: ["正常", "異常", "未点検"])),
        Field(key: "vibration", name: "振動", kind: .list(options: ["正常", "異常", "未点検"])),
        Field(key: "next_action", name: "次回作業", kind: .list(options: ["給油", "交換", "清掃", "なし"])),
    ]

    public static let noMention = "言及なし"
    public static let numberTask = "数値の項目"
    public static let numberOther = "その他"

    // MARK: - Mention gate (experiment, off by default)

    /// Asks, as its own binary classification, whether the sentence talks about a
    /// field at all — separating "was it mentioned?" from "what was the value?".
    ///
    /// The measured failure mode is that the model fills in fields nobody mentioned,
    /// with high scores, and that a Japanese negation ("異音なし") lands on 言及なし
    /// rather than 正常. The single classification head has to answer both questions
    /// at once with one set of labels; this splits them. It costs one extra model
    /// call per list field.
    ///
    /// Off by default: the demo's behaviour is unchanged unless it is switched on.
    public static let mentionLabels = ["述べている", "述べていない"]
    public static func mentionTask(for field: String) -> String {
        "この文は「\(field)」について述べているか"
    }

    /// Provisional demo thresholds — not tuned against the self-test expectations.
    public static let autoFillThreshold: Float = 0.75
    public static let mentionThreshold: Float = 0.50

    public static var numberFields: [Field] {
        fields.filter { if case .number = $0.kind { true } else { false } }
    }

    /// Labels for the "which field does this number belong to" call.
    public static var numberLabels: [String] {
        numberFields.map(\.name) + [numberOther]
    }

    // MARK: - Number candidates (code, not the model)

    public struct NumberCandidate: Sendable, Equatable {
        /// NFKC-normalised digits, e.g. "0.45".
        public var value: String
        /// Unit that immediately followed the digits, or "".
        public var unit: String
        /// 8 code points before … 4 after, the text handed to the classifier.
        public var window: String
    }

    /// Units recognised directly after the digits. Order matters: it mirrors the
    /// Python reference so the two sides cut identical windows.
    public static let units = ["MPa", "kPa", "℃", "度", "mm", "%", "％"]
    static let windowBefore = 8
    static let windowAfter = 4

    /// Digits (half or full width) with an optional decimal part, plus a trailing unit.
    ///
    /// Kana numerals ("れいてんよんご") are deliberately out of scope for the demo.
    ///
    /// Scans code points directly rather than using a regex: Python's `re` indexes
    /// by code point, and matching that exactly is what keeps the windows — and so
    /// the model inputs — identical on both sides.
    public static func numberCandidates(in text: String) -> [NumberCandidate] {
        func isDigit(_ scalar: Unicode.Scalar) -> Bool {
            ("0"..."9").contains(scalar) || ("\u{FF10}"..."\u{FF19}").contains(scalar)
        }
        func isDot(_ scalar: Unicode.Scalar) -> Bool { scalar == "." || scalar == "\u{FF0E}" }
        func slice(_ chars: [Unicode.Scalar], _ range: Range<Int>) -> String {
            String(String.UnicodeScalarView(chars[range]))
        }

        let chars = Array(text.unicodeScalars)
        var candidates: [NumberCandidate] = []
        var index = 0
        while index < chars.count {
            guard isDigit(chars[index]) else {
                index += 1
                continue
            }
            let start = index
            while index < chars.count, isDigit(chars[index]) { index += 1 }
            // A decimal point only counts when a digit follows it.
            if index + 1 < chars.count, isDot(chars[index]), isDigit(chars[index + 1]) {
                index += 1
                while index < chars.count, isDigit(chars[index]) { index += 1 }
            }
            let digitsEnd = index
            var end = digitsEnd
            var matched = ""
            for unit in units {
                let width = unit.unicodeScalars.count
                guard end + width <= chars.count else { continue }
                if slice(chars, end..<(end + width)).lowercased() == unit.lowercased() {
                    matched = unit
                    end += width
                    break
                }
            }
            candidates.append(
                NumberCandidate(
                    value: slice(chars, start..<digitsEnd).precomposedStringWithCompatibilityMapping,
                    unit: matched,
                    window: slice(
                        chars, max(0, start - windowBefore)..<min(chars.count, end + windowAfter))))
        }
        return candidates
    }
}

/// One row of the filled-in sheet.
public struct MapperFieldResult: Sendable, Equatable {
    public enum Status: String, Sendable {
        case autoFilled = "自動入力"
        case review = "要確認"
        case absent = "未言及"
    }

    public var key: String
    public var name: String
    /// Filled value, or "" when the field came out as 未言及.
    public var value: String
    public var unit: String
    public var score: Float
    public var status: Status
    /// Top two labels with their scores, always shown so the model's hesitation is
    /// visible. Display only — it never changes `status`.
    public var topTwo: [(label: String, score: Float)]
    /// e.g. "訂正の可能性" when two numbers landed on the same field.
    public var note: String

    public init(
        key: String, name: String, value: String, unit: String, score: Float,
        status: Status, topTwo: [(label: String, score: Float)], note: String
    ) {
        self.key = key
        self.name = name
        self.value = value
        self.unit = unit
        self.score = score
        self.status = status
        self.topTwo = topTwo
        self.note = note
    }

    public var displayValue: String {
        guard !value.isEmpty else { return "—" }
        if case .number = MapperForm.fields.first(where: { $0.key == key })?.kind { return value + unit }
        return value
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.key == rhs.key && lhs.value == rhs.value && lhs.status == rhs.status
    }
}

/// One scoring rule, used for every engine so the comparison is like-for-like:
/// a mentioned field must carry exactly the expected value, and a field nobody
/// mentioned must come out 未言及.
public enum MapperScoring {
    public static func matches(_ field: MapperFieldResult, expected: String?) -> Bool {
        guard let expected else { return field.status == .absent }
        return field.status != .absent && field.value == expected
    }
}

public struct MapperOutcome: Sendable {
    public var fields: [MapperFieldResult]
    public var seconds: Double
    public var modelCalls: Int
}

/// Fills the U1 sheet from one sentence: four list calls plus one call per numeric
/// candidate, run serially.
public struct GLiNERMapper: Sendable {
    public let classifier: GLiNERClassifier
    /// Experiment: ask "is this field mentioned?" separately. Off by default, so the
    /// demo path is exactly what was measured today.
    public var useMentionGate: Bool

    public init(classifier: GLiNERClassifier, useMentionGate: Bool = false) {
        self.classifier = classifier
        self.useMentionGate = useMentionGate
    }

    public func map(text: String) throws -> MapperOutcome {
        let started = Date()
        var calls = 0
        var results: [String: MapperFieldResult] = [:]

        // Numbers: code finds the candidates, the model assigns each to a field.
        let numberLabels = MapperForm.numberLabels
        for candidate in MapperForm.numberCandidates(in: text) {
            let scores = try classifier.scores(
                text: candidate.window, task: MapperForm.numberTask, labels: numberLabels)
            calls += 1
            let ranked = Self.rank(labels: numberLabels, scores: scores)
            guard let best = ranked.first,
                best.label != MapperForm.numberOther,
                best.score >= MapperForm.mentionThreshold,
                let field = MapperForm.numberFields.first(where: { $0.name == best.label })
            else { continue }

            var note = ""
            // Two candidates on one field: the later one wins and gets flagged.
            // "圧力0.45、違った、0.54MPa" is the case this is for.
            var status: MapperFieldResult.Status =
                best.score >= MapperForm.autoFillThreshold ? .autoFilled : .review
            if results[field.key] != nil {
                note = "訂正の可能性"
                status = .review
            }
            var unit = candidate.unit
            if case .number(let defaultUnit) = field.kind, unit.isEmpty { unit = defaultUnit }
            results[field.key] = MapperFieldResult(
                key: field.key, name: field.name, value: candidate.value, unit: unit,
                score: best.score, status: status, topTwo: Array(ranked.prefix(2)), note: note)
        }

        // List fields: one call each, labels + 言及なし.
        for field in MapperForm.fields {
            guard case .list(let options) = field.kind else { continue }

            // Optional extra call: decide mention separately from value.
            var gatedOut = false
            if useMentionGate {
                let gate = try classifier.scores(
                    text: text, task: MapperForm.mentionTask(for: field.name),
                    labels: MapperForm.mentionLabels)
                calls += 1
                gatedOut =
                    Self.rank(labels: MapperForm.mentionLabels, scores: gate)
                    .first?.label == "述べていない"
            }

            let labels = options + [MapperForm.noMention]
            let scores = try classifier.scores(text: text, task: field.name, labels: labels)
            calls += 1
            let ranked = Self.rank(labels: labels, scores: scores)
            let best = ranked[0]
            let status: MapperFieldResult.Status
            if gatedOut || best.label == MapperForm.noMention
                || best.score < MapperForm.mentionThreshold
            {
                status = .absent
            } else if best.score >= MapperForm.autoFillThreshold {
                status = .autoFilled
            } else {
                status = .review
            }
            results[field.key] = MapperFieldResult(
                key: field.key, name: field.name,
                value: status == .absent ? "" : best.label, unit: "",
                score: best.score, status: status, topTwo: Array(ranked.prefix(2)), note: "")
        }

        let ordered = MapperForm.fields.map { field in
            results[field.key]
                ?? MapperFieldResult(
                    key: field.key, name: field.name, value: "", unit: "", score: 0,
                    status: .absent, topTwo: [], note: "")
        }
        return MapperOutcome(
            fields: ordered, seconds: Date().timeIntervalSince(started), modelCalls: calls)
    }

    public static func rank(labels: [String], scores: [Float]) -> [(label: String, score: Float)] {
        zip(labels, scores).map { (label: $0.0, score: $0.1) }.sorted { $0.score > $1.score }
    }
}

/// `phys_footprint` from `task_info`, the number Xcode's memory gauge shows.
public func mapperMemoryFootprint() -> Int? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? Int(info.phys_footprint) : nil
}
