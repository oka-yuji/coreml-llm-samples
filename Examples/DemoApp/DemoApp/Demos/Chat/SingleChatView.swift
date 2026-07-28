import SwiftUI

struct SingleChatView: View {
    @Environment(ChatViewModel.self) private var vm
    @Environment(DemoNavigator.self) private var navigator
    @FocusState private var inputFocused: Bool

    var body: some View {
        Group {
            if vm.isModelLoaded {
                VStack(spacing: 0) {
                    transcript
                    Divider()
                    statusBar
                    composer
                }
            } else {
                emptyState
            }
        }
        .navigationTitle(vm.isModelLoaded ? vm.modelName : "Chat")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .onChange(of: vm.speculative) { _, on in vm.speculationToggleChanged(to: on) }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button("Reset", action: vm.reset)
                    .disabled(!vm.canReset)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            switch vm.phase {
            case .loading:
                ProgressView()
                Text(statusText).font(.callout).foregroundStyle(.secondary)
            case .failed(let why):
                Image(systemName: "exclamationmark.triangle")
                    .font(.largeTitle)
                    .foregroundStyle(.orange)
                Text("Failed to load model").font(.headline)
                Text(why)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                Button("Open Models") { navigator.selection = .models }
            default:
                Image(systemName: "square.and.arrow.down")
                    .font(.largeTitle)
                    .foregroundStyle(.secondary)
                Text("No model loaded").font(.headline)
                Text("Open Models to download a bundle, or load one placed in Documents.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                Button("Open Models") { navigator.selection = .models }
                localBundlesList
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
        .task { vm.refreshLocalBundles() }
    }

    @ViewBuilder private var localBundlesList: some View {
        if !vm.localBundles.isEmpty {
            Divider()
            Text("On-device bundles (Documents)")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(vm.localBundles, id: \.self) { url in
                Button {
                    Task { await vm.loadModel(path: url.path(percentEncoded: false)) }
                } label: {
                    Label(url.lastPathComponent, systemImage: "shippingbox")
                }
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
            .scrollDismissesKeyboard(.interactively)
            .onTapGesture { inputFocused = false }
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
        @Bindable var vm = vm
        return HStack(spacing: 8) {
            if vm.isLoading || vm.isGenerating || vm.preparingSpeculation { ProgressView().controlSize(.small) }
            Text(vm.kvStatus.isEmpty ? statusText : vm.kvStatus)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer()
            Toggle("Spec", isOn: $vm.speculative)
                .toggleStyle(.switch)
                .controlSize(.mini)
                .labelsHidden()
                .disabled(vm.isGenerating || vm.preparingSpeculation)
            Menu {
                Button("Save context checkpoint", action: vm.saveKV)
                Button("Restore checkpoint", action: vm.restoreKV)
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuIndicator(.hidden)
            .disabled(!vm.canCheckpoint)
            .fixedSize()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var statusText: String {
        switch vm.phase {
        case .idle:
            return "Idle."
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
        @Bindable var vm = vm
        return VStack(spacing: 8) {
            if vm.isConversationFull {
                contextFullBanner
            }
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Message", text: $vm.input, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...6)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary))
                    .disabled(!vm.isModelLoaded || vm.isLoading)
                    .focused($inputFocused)
                    .onSubmit(sendIfPossible)

                if vm.isGenerating {
                    Button(action: vm.stop) {
                        Image(systemName: "stop.circle.fill")
                            .font(.system(size: 30))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.accentColor)
                    .accessibilityLabel("Stop")
                } else {
                    Button(action: sendIfPossible) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 30))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(vm.canSend ? Color.accentColor : Color.secondary)
                    .disabled(!vm.canSend)
                    .accessibilityLabel("Send")
                }
            }
        }
        .padding(10)
    }

    private var contextFullBanner: some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.circle")
            Text(vm.conversationFullMessage)
            Spacer(minLength: 0)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func sendIfPossible() {
        guard vm.canSend else { return }
        inputFocused = false
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
