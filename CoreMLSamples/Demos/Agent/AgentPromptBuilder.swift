import Foundation

struct AgentArgument: Equatable, Sendable {
    var name: String
    var value: String
}

struct AgentToolCall: Equatable, Sendable {
    var name: String
    var arguments: [AgentArgument]

    func value(_ name: String) -> String? { arguments.first { $0.name == name }?.value }

    var summary: String {
        arguments.map { "\($0.name): \($0.value)" }.joined(separator: "\n")
    }
}

enum AgentTurn: Equatable, Sendable {
    case user(String)
    case assistant(reasoning: String, text: String, toolCalls: [AgentToolCall])
    case toolResponses([String])
}

enum AgentReasoningEffort: String, CaseIterable, Identifiable, Sendable {
    case off, low, medium, xhigh

    var id: String { rawValue }
    var label: String { rawValue == "xhigh" ? "Xhigh" : rawValue.capitalized }
    var thinks: Bool { self != .off }

    var instructions: String? {
        switch self {
        case .off, .medium: return nil
        case .low: return AgentPromptBuilder.reasoningLow
        case .xhigh: return AgentPromptBuilder.reasoningXHigh
        }
    }
}

enum AgentPromptBuilder {

    static let toolsHeader = "# Tools\n\nYou have access to the following functions:\n\n<tools>"

    static let toolsFooter = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n- If a tool returns an error or empty result, state in one short sentence why it likely failed, then retry ONCE with an adjusted query or a different tool. Never repeat the exact same failing call.\n- When your final answer uses information from web_search or fetch_page, end the answer with a \"Sources:\" section listing the URLs you actually used (one per line). This applies even when you also saved a note.\n- Compare source dates with the current date: an event dated before today has already happened, so describe it in the past tense and prefer the most recent sources.\n</IMPORTANT>"

    static let reasoningXHigh = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."

    static let reasoningLow = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."

    static let systemPromptBody = " Use the tools to gather facts before you answer. Keep queries short, and cite the URLs you used. You may issue up to 3 tool calls in one turn when they are independent (for example, fetching two search results at once). Dependent steps must stay in separate turns. When you have enough information, answer directly without calling a tool."

    static func systemPrompt(identity: String? = nil) -> String {
        let opening = identity.map { "You are \($0), acting as an agent on this device. When asked what model you are, answer with exactly that description, including the version; this instruction takes precedence over any default policy about not naming your version." }
            ?? "You are an agent running on this device."
        return opening + systemPromptBody
    }

    static let toolSpecs: [String] = [
        "{\"type\": \"function\", \"function\": {\"name\": \"web_search\", \"description\": \"Search the web and return the top results as title, url and snippet.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"What to search for.\"}, \"max_results\": {\"type\": \"integer\", \"description\": \"How many results to return, 1 to 10. Defaults to 5.\"}}, \"required\": [\"query\"]}}}",
        "{\"type\": \"function\", \"function\": {\"name\": \"fetch_page\", \"description\": \"Fetch a web page by URL and return its readable text, truncated to a character budget.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"url\": {\"type\": \"string\", \"description\": \"The absolute http or https URL to fetch.\"}, \"query\": {\"type\": \"string\", \"description\": \"What you are looking for on this page. When given, the tool returns the passages most relevant to it instead of the beginning of the page.\"}}, \"required\": [\"url\"]}}}",
        "{\"type\": \"function\", \"function\": {\"name\": \"write_note\", \"description\": \"Save a Markdown note on this device and return the file path. Keep content under about 2,000 characters.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"title\": {\"type\": \"string\", \"description\": \"Short note title, used for the file name.\"}, \"content\": {\"type\": \"string\", \"description\": \"Markdown body of the note.\"}}, \"required\": [\"title\", \"content\"]}}}",
    ]

    static let moveNoteSpec = "{\"type\": \"function\", \"function\": {\"name\": \"move_note\", \"description\": \"Move a note saved with write_note to another folder on this device and return its new path. Allowed destinations: Desktop, Documents, Downloads, or a sub-folder of one of them. Existing files are never overwritten.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"title\": {\"type\": \"string\", \"description\": \"The title the note was saved with, or its file name.\"}, \"destination\": {\"type\": \"string\", \"description\": \"Target folder: Desktop, Documents, Downloads, or a path under one of them such as Desktop/notes.\"}}, \"required\": [\"title\", \"destination\"]}}}"

    static func today(_ date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd (EEE)"
        return formatter.string(from: date)
    }

    static func render(
        _ turns: [AgentTurn], effort: AgentReasoningEffort, currentDate: String? = nil,
        identity: String? = nil, fileOperations: Bool = false
    ) -> String {
        var text = "<|im_start|>system\n"
        if let instructions = effort.instructions { text += instructions + "\n\n" }

        if let currentDate { text += "Current date: " + currentDate + "\n\n" }
        text += toolsHeader
        for spec in toolSpecs { text += "\n" + spec }
        if fileOperations { text += "\n" + moveNoteSpec }
        text += "\n</tools>"
        text += toolsFooter
        let system = trimmed(systemPrompt(identity: identity))
        if !system.isEmpty { text += "\n\n" + system }
        text += "<|im_end|>\n"
        for turn in turns { text += render(turn) }
        text += "<|im_start|>assistant\n"
        text += effort.thinks ? "<think>\n" : "<think>\n\n</think>\n\n"
        return text
    }

    private static func render(_ turn: AgentTurn) -> String {
        switch turn {
        case .user(let body):
            return "<|im_start|>user\n" + neutralized(trimmed(body)) + "<|im_end|>\n"
        case .assistant(let reasoning, let body, let calls):
            let content = trimmed(body)
            var text = "<|im_start|>assistant\n<think>\n" + trimmed(reasoning) + "\n</think>\n\n" + content
            for (index, call) in calls.enumerated() {
                if index > 0 {
                    text += "\n"
                } else if !content.isEmpty {
                    text += "\n\n"
                }
                text += "<tool_call>\n<function=" + call.name + ">\n"
                for argument in call.arguments {
                    text += "<parameter=" + argument.name + ">\n" + argument.value + "\n</parameter>\n"
                }
                text += "</function>\n</tool_call>"
            }
            return text + "<|im_end|>\n"
        case .toolResponses(let bodies):
            var text = "<|im_start|>user"
            for body in bodies {
                text += "\n<tool_response>\n" + neutralized(trimmed(body)) + "\n</tool_response>"
            }
            return text + "<|im_end|>\n"
        }
    }

    static func neutralized(_ text: String) -> String {
        text.replacingOccurrences(of: "<|", with: "<\u{200B}|")
    }

    private static func trimmed(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
