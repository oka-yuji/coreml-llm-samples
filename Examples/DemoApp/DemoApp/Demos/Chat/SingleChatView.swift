import SwiftUI
import UniformTypeIdentifiers

struct SingleChatView: View {
    @State private var vm = ChatViewModel()
    @AppStorage("lastModelPath") private var selectedPath = ""
    @State private var importing = false

    var body: some View {
        VStack(spacing: 0) {
            transcript
            Divider()
            statusBar
            composer
        }
        .navigationTitle(vm.isModelLoaded ? vm.modelName : "Chat")
        .toolbar {
            ToolbarItemGroup {
                Button("Choose") { importing = true }
                    .disabled(!vm.canLoad)
                Button("Load") { Task { await vm.loadModel(path: selectedPath) } }
                    .disabled(!vm.canLoad || !pathHasManifest)
                Button("Reset", action: vm.reset)
                    .disabled(!vm.canReset)
            }
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result {
                _ = url.startAccessingSecurityScopedResource()
                selectedPath = url.path
            }
        }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(vm.messages) { message in
                        ChatBubble(message: message).id(message.id)
                    }
                    Color.clear.frame(height: 1).id(bottomAnchor)
                }
                .padding(12)
            }
            .onChange(of: vm.messages.last?.text) { _, _ in
                withAnimation(.easeOut(duration: 0.1)) { proxy.scrollTo(bottomAnchor, anchor: .bottom) }
            }
            .onChange(of: vm.messages.count) { _, _ in
                proxy.scrollTo(bottomAnchor, anchor: .bottom)
            }
        }
    }

    private let bottomAnchor = "bottom-anchor"

    private var statusBar: some View {
        HStack(spacing: 8) {
            if vm.isLoading || vm.isGenerating { ProgressView().controlSize(.small) }
            Text(statusText)
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

    private var statusText: String {
        switch vm.phase {
        case .idle:
            return "Choose a model bundle directory, then Load."
        case .loading(let s):
            return s
        case .failed(let s):
            return s
        case .generating:
            if vm.warming { return "Warming up: the first reply specializes GPU kernels (~40s)…" }
            return vm.loadStatus.isEmpty ? "Generating…" : vm.loadStatus
        case .ready:
            if !vm.statusLine.isEmpty { return vm.statusLine }
            return vm.loadStatus.isEmpty ? "Ready." : vm.loadStatus
        }
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message", text: $vm.input, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...6)
                .padding(8)
                .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary))
                .disabled(!vm.isModelLoaded || vm.isLoading)
                .onSubmit(sendIfPossible)

            if vm.isGenerating {
                Button("Stop", action: vm.stop)
            } else {
                Button("Send", action: sendIfPossible)
                    .disabled(!vm.canSend)
            }
        }
        .padding(10)
    }

    private var pathHasManifest: Bool {
        guard !selectedPath.isEmpty else { return false }
        return FileManager.default.fileExists(
            atPath: URL(fileURLWithPath: selectedPath).appending(path: "manifest.json").path())
    }

    private func sendIfPossible() {
        guard vm.canSend else { return }
        vm.send()
    }
}

private struct ChatBubble: View {
    let message: ChatViewModel.Message

    private var isUser: Bool { message.role == .user }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(isUser ? "You" : "Assistant")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(message.text.isEmpty && !isUser ? " " : message.text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(isUser ? AnyShapeStyle(Color.accentColor.opacity(0.15)) : AnyShapeStyle(.quaternary)))
        }
    }
}
