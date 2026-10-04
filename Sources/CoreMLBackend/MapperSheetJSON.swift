import Foundation

/// The generative route to the same U1 sheet: hand a model the form and the
/// sentence, make it emit JSON, parse it.
///
/// This exists to compare a general LLM against the GLiNER2.5 classifier under
/// identical conditions — same 30 sentences, same six fields, same scoring. The
/// prompt is fixed (one prompt, no per-sentence tuning) and reproduced verbatim in
/// `docs/results/2026-10-04-gliner25-mapper-demo.md`.
///
/// Matching is exact and there is no rescue layer: a value the model spells
/// differently from the candidate ("異常あり" for "異常", "0.45MPa" for "0.45") is
/// kept as written and therefore counts as a miss. That is the point — it measures
/// what the model actually produces, not what a post-processor could salvage.
public enum MapperSheetJSON {

    public static func prompt(for text: String) -> String {
        """
        設備の点検メモを点検票に変換します。下の JSON だけを出力してください。説明は書かないでください。

        項目と書き方:
        - discharge_pressure(吐出圧): 数字だけ。単位は書かない。例 "0.45"
        - temperature(温度): 数字だけ。単位は書かない。例 "72"
        - oil_leak(油漏れ): "あり" か "なし"
        - abnormal_sound(異音): "正常" か "異常" か "未点検"
        - vibration(振動): "正常" か "異常" か "未点検"
        - next_action(次回作業): "給油" か "交換" か "清掃" か "なし"

        規則:
        - 文中で言及されていない項目は null にする。
        - 値は上に書いた候補の文字列と完全に一致させる。
        - JSON 以外は出力しない。

        文: 「\(text)」

        JSON:
        """
    }

    /// First balanced `{...}` in the reply. Models like to wrap JSON in prose or a
    /// fenced block, so the object is located rather than assuming the whole reply
    /// is JSON. Quoted braces are ignored.
    static func firstJSONObject(in reply: String) -> String? {
        var depth = 0
        var start: String.Index?
        var inString = false
        var escaped = false
        for index in reply.indices {
            let character = reply[index]
            if inString {
                if escaped { escaped = false } else if character == "\\" { escaped = true } else if character == "\"" { inString = false }
                continue
            }
            switch character {
            case "\"": inString = true
            case "{":
                if depth == 0 { start = index }
                depth += 1
            case "}":
                depth -= 1
                if depth == 0, let start {
                    return String(reply[start...index])
                }
                if depth < 0 { return nil }
            default: break
            }
        }
        return nil
    }

    /// A JSON scalar as the text the sheet should show. Numbers are rendered
    /// minimally (72 -> "72", 0.45 -> "0.45") so a model that emits a bare JSON
    /// number is not punished for the quoting style — the *value* still has to be
    /// right. Anything else (object, array, bool) is rendered as-is and will miss.
    static func text(from value: Any) -> String? {
        switch value {
        case is NSNull: return nil
        case let string as String:
            let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "true" : "false" }
            let double = number.doubleValue
            if double == double.rounded(), abs(double) < 1e15 {
                return String(Int64(double))
            }
            return String(describing: number)
        default: return nil
        }
    }

    /// `nil` means the reply held no parsable JSON object — counted separately as a
    /// parse failure rather than as six wrong fields.
    public static func parse(_ reply: String) -> [MapperFieldResult]? {
        guard let object = firstJSONObject(in: reply),
            let data = object.data(using: .utf8),
            let dictionary = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }

        return MapperForm.fields.map { field in
            let value = dictionary[field.key].flatMap(text(from:)) ?? ""
            var unit = ""
            if case .number(let defaultUnit) = field.kind, !value.isEmpty { unit = defaultUnit }
            return MapperFieldResult(
                key: field.key, name: field.name, value: value, unit: unit,
                // A generated sheet carries no score; the status is simply filled or not.
                score: 0, status: value.isEmpty ? .absent : .autoFilled,
                topTwo: [], note: "")
        }
    }
}
