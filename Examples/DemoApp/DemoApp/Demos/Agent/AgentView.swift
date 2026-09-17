import LLMCore
import SwiftUI
#if os(macOS)
import AppKit
#endif

struct AgentView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(DemoNavigator.self) private var navigator
    @Environment(AgentViewModel.self) private var agent
    @State private var bundleIsSupported = false
    @State private var showSettings = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if chat.isModelLoaded, bundleIsSupported {
                transcript
                Divider()
                if showSettings {
                    settings
                    Divider()
                }
                status
                composer
            } else {
                emptyState
            }
        }
        .navigationTitle("Agent")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .task(id: chat.loadedPath) { bundleIsSupported = AgentView.isChatMLBundle(chat.loadedPath) }
    }

    static func isChatMLBundle(_ path: String) -> Bool {
        guard !path.isEmpty,
              let bundle = try? ModelBundle(contentsOf: URL(fileURLWithPath: path, isDirectory: true))
        else { return false }
        return bundle.manifest.promptPrefix?.contains("<|im_start|>") == true
    }

    private var header: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(chat.isModelLoaded ? chat.modelName : "No model loaded")
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("web_search · fetch_page · write_note"
                    + (agent.allowFileOperations ? " · move_note" : "")
                    + " — search via \(agent.searchProvider)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Spacer()
            Button("Copy") { agent.copyTranscript() }
                .disabled(agent.entries.isEmpty)
                .help("Copy the whole transcript as Markdown")
            Button("New Task") { agent.reset(chat: chat) }
                .disabled(!agent.canReset)
            Button {
                showSettings.toggle()
            } label: {
                Image(systemName: "gearshape")
            }
            .accessibilityLabel("Agent settings")
        }
        .padding(10)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer()
            switch chat.phase {
            case .loading:
                ProgressView()
                Text("Loading model…").font(.headline)
                if chat.isCompilingLongLoad {
                    Text(chat.loadingMessage)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
            case .failed(let why) where !chat.isModelLoaded:
                Image(systemName: "exclamationmark.triangle")
                    .font(.largeTitle)
                    .foregroundStyle(.orange)
                Text("Model failed to load").font(.headline)
                Text(why)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(4)
                Button("Open Chat") { navigator.selection = .chat }
            default:
                Image(systemName: "sparkles")
                    .font(.largeTitle)
                    .foregroundStyle(.secondary)
                if !chat.isModelLoaded {
                    Text("No model loaded").font(.headline)
                    Text("Load a Qwen3.8 Core ML bundle in Chat, then come back here.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                    Button("Open Chat") { navigator.selection = .chat }
                    Button("Open Models") { navigator.selection = .models }
                } else {
                    Text("This demo needs a Qwen bundle").font(.headline)
                    Text("\(chat.modelName) does not use the ChatML tool-calling template. "
                        + "Load a Qwen3.8 bundle in Chat to run the agent.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                    Button("Open Chat") { navigator.selection = .chat }
                }
            }
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if agent.entries.isEmpty { placeholder }
                    ForEach(agent.entries) { entry in
                        AgentEntryRow(entry: entry).id(entry.id)
                    }
                    Color.clear.frame(height: 1).id(bottomAnchor)
                }
                .padding(12)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: agent.entries.last?.text) { _, _ in
                withAnimation(.easeOut(duration: 0.1)) { proxy.scrollTo(bottomAnchor, anchor: .bottom) }
            }
            .onChange(of: agent.entries.count) { _, _ in proxy.scrollTo(bottomAnchor, anchor: .bottom) }
        }
    }

    private let bottomAnchor = "agent-bottom-anchor"

    private var placeholder: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Ask for something that needs the web:")
                .font(.callout)
                .foregroundStyle(.secondary)
            ForEach(AgentView.examples, id: \.self) { example in
                Button(example) { agent.input = example }
                    .buttonStyle(.plain)
                    .font(.callout)
                    .foregroundStyle(Color.accentColor)
                    .multilineTextAlignment(.leading)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    static let examples = [
        "Search for the latest Core ML news this year and summarise it in three lines.",
        "Read https://developer.apple.com/documentation/coreml and give me three key points.",
        "Look up tomorrow's weather in Tokyo, then save a summary as a note."
    ]

    private var settings: some View {
        @Bindable var agent = agent
        return VStack(alignment: .leading, spacing: 8) {
            Picker("Reasoning effort", selection: $agent.effort) {
                ForEach(AgentReasoningEffort.allCases) { Text($0.label).tag($0) }
            }
            .pickerStyle(.segmented)
            .controlSize(.mini)
            .frame(maxWidth: 320)
            .disabled(agent.running)
            HStack(spacing: 8) {
                Text("Max steps: \(agent.maxIterations)")
                    .monospacedDigit()
                    .frame(width: 190, alignment: .leading)
                Slider(value: sliderBinding($agent.maxIterations), in: 1...16, step: 1)
                    .controlSize(.small)
                    .frame(maxWidth: 320)
            }
            .disabled(agent.running)
            HStack(spacing: 8) {
                Text("Page budget: \(agent.pageCharacterBudget) chars")
                    .monospacedDigit()
                    .frame(width: 190, alignment: .leading)
                Slider(value: sliderBinding($agent.pageCharacterBudget), in: 500...20_000, step: 500)
                    .controlSize(.small)
                    .frame(maxWidth: 320)
            }
            .disabled(agent.running)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text("Brave Search API key")
                    SecureField("optional — empty uses DuckDuckGo", text: $agent.braveAPIKey)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 280)
                }
                Text("Stored in this app's preferences on this Mac")
                    .foregroundStyle(.secondary)
            }
            .disabled(agent.running)
            #if os(macOS)
            Toggle("Allow file operations (move_note)", isOn: $agent.allowFileOperations)
                .disabled(agent.running)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text("Notes folder")
                    Text(agent.notesDirectory.path(percentEncoded: false))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.head)
                    Button("Choose…") { chooseNotesFolder() }
                    Button("Default") { agent.notesFolderPath = "" }
                        .disabled(agent.notesFolderPath.isEmpty)
                }
                Text("Notes outside Desktop, Documents or Downloads cannot be moved by the agent")
                    .foregroundStyle(.secondary)
            }
            .disabled(agent.running)
            #endif
        }
        .font(.caption)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func sliderBinding(_ value: Binding<Int>) -> Binding<Double> {
        Binding(get: { Double(value.wrappedValue) }, set: { value.wrappedValue = Int($0.rounded()) })
    }

    #if os(macOS)
    private func chooseNotesFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose"
        panel.message = "Where the agent saves its notes"
        panel.directoryURL = agent.notesDirectory
        guard panel.runModal() == .OK, let url = panel.url else { return }
        agent.notesFolderPath = url.path(percentEncoded: false)
    }
    #endif

    private var status: some View {
        HStack(spacing: 8) {
            if agent.running { ProgressView().controlSize(.small) }
            Text(agent.statusLine.isEmpty ? "Ready." : agent.statusLine)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var composer: some View {
        @Bindable var agent = agent
        return HStack(alignment: .bottom, spacing: 8) {
            TextField("Task", text: $agent.input, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...6)
                .padding(8)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color.fieldBackground))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.fieldBorder))
                .focused($inputFocused)
                .disabled(agent.running)
                .onSubmit(sendIfPossible)
            if agent.running {
                Button(action: agent.stop) {
                    Image(systemName: "stop.circle.fill").font(.system(size: 28))
                }
                .buttonStyle(.plain)
                .foregroundStyle(Color.accentColor)
                .keyboardShortcut(".", modifiers: .command)
                .accessibilityLabel("Stop")
            } else {
                Button(action: sendIfPossible) {
                    Image(systemName: "arrow.up.circle.fill").font(.system(size: 28))
                }
                .buttonStyle(.plain)
                .foregroundStyle(agent.canSend(chat: chat) ? Color.accentColor : Color.secondary)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!agent.canSend(chat: chat))
                .accessibilityLabel("Send")
            }
        }
        .padding(10)
    }

    private func sendIfPossible() {
        guard agent.canSend(chat: chat) else { return }
        inputFocused = false
        agent.send(chat: chat)
    }
}

struct AgentEntryRow: View {
    let entry: AgentViewModel.Entry
    @State private var expanded = false

    var body: some View {
        switch entry.kind {
        case .user:
            VStack(alignment: .leading, spacing: 2) {
                label("You", "person")
                Text(entry.text).textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .answer:
            VStack(alignment: .leading, spacing: 2) {
                label("Assistant", "sparkles")
                Text(entry.text).textSelection(.enabled)
                statsCaption
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .failure:
            VStack(alignment: .leading, spacing: 2) {
                label(entry.title, "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                Text(entry.text)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .note:
            Label("\(entry.title): \(entry.text)", systemImage: "checkmark.seal")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        case .toolCall:
            VStack(alignment: .leading, spacing: 4) {
                label("Calling \(entry.title)", "wrench.and.screwdriver")
                Text(entry.text)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                statsCaption
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color.fieldBackground))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.fieldBorder))
        case .thinking, .toolResult:
            collapsible
        }
    }

    @ViewBuilder
    private var statsCaption: some View {
        if let stats = entry.stats {
            Text(stats)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.tertiary)
        }
    }

    private var collapsible: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button {
                expanded.toggle()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                    Text(entry.kind == .thinking ? "Thinking" : entry.title)
                    Text("(\(entry.text.count) chars)").foregroundStyle(.secondary)
                }
                .font(.caption)
            }
            .buttonStyle(.plain)
            Text(expanded ? entry.text : preview)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var preview: String {
        let flat = entry.text.replacingOccurrences(of: "\n", with: " ")
        return flat.count > 300 ? String(flat.prefix(300)) + "…" : flat
    }

    private func label(_ text: String, _ systemImage: String) -> some View {
        Label(text, systemImage: systemImage)
            .font(.caption)
            .foregroundStyle(.secondary)
    }
}
