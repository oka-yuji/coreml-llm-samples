import Foundation

enum AgentToolCallParser {

    struct Output: Equatable {
        var reasoning: String
        var text: String
        var toolCalls: [AgentToolCall]
    }

    static func parse(_ raw: String) -> Output {
        let callPattern = /^\s*<function=([^>\n]*)>(.*?)<\/function>\s*<\/tool_call>/
            .dotMatchesNewlines()
        let parameterPattern = /<parameter=([^>\n]*)>(.*?)<\/parameter>/.dotMatchesNewlines()
        var body = raw[...]
        var reasoning = ""
        if let close = body.range(of: "</think>") {
            var head = body[body.startIndex..<close.lowerBound]
            if let open = head.range(of: "<think>") { head = head[open.upperBound...] }
            reasoning = head.trimmingCharacters(in: .whitespacesAndNewlines)
            body = body[close.upperBound...]
        }
        var calls: [AgentToolCall] = []
        let blocks = String(body).components(separatedBy: "<tool_call>")
        var text = blocks[0]
        for block in blocks.dropFirst() {
            guard let match = block.firstMatch(of: callPattern) else { continue }
            let arguments = match.2.matches(of: parameterPattern).map {
                AgentArgument(name: String($0.1), value: unwrapParameter(String($0.2)))
            }
            calls.append(AgentToolCall(name: String(match.1), arguments: arguments))
            text += block[match.range.upperBound...]
        }
        return Output(
            reasoning: reasoning,
            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
            toolCalls: calls)
    }

    static func unwrapParameter(_ raw: String) -> String {
        var value = raw[...]
        if value.hasPrefix("\n") { value = value.dropFirst() }
        if value.hasSuffix("\n") { value = value.dropLast() }
        return String(value)
    }
}
