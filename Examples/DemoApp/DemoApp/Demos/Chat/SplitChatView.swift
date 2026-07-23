import AppKit
import SwiftUI

struct SplitChatView: View {
    @State private var vm = ChatViewModel()

    @AppStorage("lastModelPath") private var selectedPath = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            Divider()
            footer
            composer
        }
        .frame(minWidth: 560, minHeight: 480)
    }

    private var header: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(vm.isModelLoaded ? vm.modelName : "No model loaded")
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if !selectedPath.isEmpty {
                    Text(selectedPath)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            Spacer()
            Button("Choose…", action: chooseModel)
                .disabled(!vm.canLoad)
            Button("Load") { Task { await vm.loadModel(path: selectedPath) } }
                .keyboardShortcut("l", modifiers: .command)
                .disabled(!vm.canLoad || !pathHasManifest)
            Button("Reset", action: vm.reset)
                .disabled(!vm.canReset)
        }
        .padding(10)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(vm.messages) { message in
                        MessageRow(message: message)
                            .id(message.id)
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

    private var footer: some View {
        HStack(spacing: 8) {
            statusIcon
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

    @ViewBuilder private var statusIcon: some View {
        switch vm.phase {
        case .loading, .generating:
            ProgressView().controlSize(.small)
        case .failed:
            Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
        default:
            EmptyView()
        }
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
                .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .textBackgroundColor)))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor)))
                .disabled(!vm.isModelLoaded || vm.isLoading)
                .onSubmit(sendIfPossible)

            if vm.isGenerating {
                Button("Stop", action: vm.stop)
                    .keyboardShortcut(".", modifiers: .command)
            } else {
                Button("Send", action: sendIfPossible)
                    .keyboardShortcut(.return, modifiers: .command)
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

    private func chooseModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Select"
        panel.message = "Choose a model bundle directory (the folder that contains manifest.json)"
        if !selectedPath.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: selectedPath)
        }
        if panel.runModal() == .OK, let url = panel.url {
            selectedPath = url.path
        }
    }
}

private struct MessageRow: View {
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
                        .fill(isUser ? Color.accentColor.opacity(0.12) : Color(nsColor: .controlBackgroundColor)))
        }
    }
}
