import SwiftUI
import UniformTypeIdentifiers

struct SplitChatView: View {
    @Environment(ChatViewModel.self) private var vm
    @Environment(DemoNavigator.self) private var navigator
    @FocusState private var inputFocused: Bool
    @State private var showImageImporter = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if vm.isModelLoaded {
                transcript
                Divider()
                footer
                composer
            } else {
                emptyState
            }
        }
        .frame(minWidth: 560, minHeight: 480)
        .onChange(of: vm.speculative) { _, on in vm.speculationToggleChanged(to: on) }
    }

    private var header: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(vm.isModelLoaded ? vm.modelName : "No model loaded")
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if vm.isModelLoaded, !vm.loadedPath.isEmpty {
                    Text(vm.loadedPath)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            Spacer()
            Button("Reset", action: vm.reset)
                .disabled(!vm.canReset)
        }
        .padding(10)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer()
            switch vm.phase {
            case .loading:
                ProgressView()
                Text(statusText)
                    .font(.callout)
                    .foregroundStyle(.secondary)
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
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
        .task { vm.refreshLocalBundles() }
    }

    @ViewBuilder private var localBundlesList: some View {
        if !vm.localBundles.isEmpty {
            Divider().frame(maxWidth: 320)
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
                        MessageRow(message: message)
                            .id(message.id)
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

    private var footer: some View {
        @Bindable var vm = vm
        return HStack(spacing: 8) {
            statusIcon
            Text(vm.kvStatus.isEmpty ? statusText : vm.kvStatus)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer()
            Toggle("Speculation", isOn: $vm.speculative)
                .toggleStyle(.switch)
                .controlSize(.mini)
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

    @ViewBuilder private var statusIcon: some View {
        if vm.preparingSpeculation {
            ProgressView().controlSize(.small)
        } else {
            switch vm.phase {
            case .loading, .generating:
                ProgressView().controlSize(.small)
            case .failed:
                Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
            default:
                EmptyView()
            }
        }
    }

    private var statusText: String {
        switch vm.phase {
        case .idle:
            return "Idle."
        case .loading:
            return vm.loadingMessage
        case .failed(let s):
            return s
        case .generating:
            if vm.warming { return "Warming up: prefilling your prompt; a long prompt takes longer…" }
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
            if let url = vm.attachedImageURL {
                attachmentChip(url)
            }
            HStack(alignment: .bottom, spacing: 8) {
                if vm.supportsImageAttachment {
                    Button { showImageImporter = true } label: {
                        Image(systemName: "photo.on.rectangle")
                            .font(.system(size: 20))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.secondary)
                    .disabled(vm.isGenerating || vm.isLoading)
                    .accessibilityLabel("Attach image")
                    .padding(.bottom, 4)
                }
                TextField("Message", text: $vm.input, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...6)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.fieldBackground))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.fieldBorder))
                    .disabled(!vm.isModelLoaded || vm.isLoading)
                    .focused($inputFocused)
                    .onSubmit(sendIfPossible)

                if vm.isGenerating {
                    Button(action: vm.stop) {
                        Image(systemName: "stop.circle.fill")
                            .font(.system(size: 28))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.accentColor)
                    .keyboardShortcut(".", modifiers: .command)
                    .accessibilityLabel("Stop")
                } else {
                    Button(action: sendIfPossible) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 28))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(vm.canSend ? Color.accentColor : Color.secondary)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!vm.canSend)
                    .accessibilityLabel("Send")
                }
            }
        }
        .padding(10)
        .fileImporter(isPresented: $showImageImporter, allowedContentTypes: [.image]) { result in
            if case .success(let url) = result { vm.attachImage(url) }
        }
    }

    private func attachmentChip(_ url: URL) -> some View {
        HStack(spacing: 8) {
            ImageThumbnail(url: url, side: 40)
            Text(url.lastPathComponent)
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 0)
            Button(action: vm.clearAttachedImage) {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .accessibilityLabel("Remove image")
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.bubbleBackground))
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

private struct ImageThumbnail: View {
    let url: URL
    let side: CGFloat

    var body: some View {
        AsyncImage(url: url) { image in
            image.resizable().scaledToFill()
        } placeholder: {
            Color.secondary.opacity(0.15)
        }
        .frame(width: side, height: side)
        .clipShape(RoundedRectangle(cornerRadius: 6))
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
            if let url = message.attachedImageURL {
                ImageThumbnail(url: url, side: 120)
            }
            Text(message.text.isEmpty && !isUser ? " " : message.text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(isUser ? Color.accentColor.opacity(0.12) : Color.bubbleBackground))
        }
    }
}
