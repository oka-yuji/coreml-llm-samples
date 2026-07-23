import AppKit
import SwiftUI

struct SplitModelsView: View {
    @Environment(ModelsViewModel.self) private var vm
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(DemoNavigator.self) private var navigator

    @State private var pendingDeleteID: String?
    @State private var showDeleteAll = false
    @State private var localBundleError: String?

    var body: some View {
        List {
            Section("Models") {
                ForEach(vm.rows, id: \.model.id) { row in
                    ModelRow(
                        row: row,
                        onDownload: { vm.startDownload(row.model.id) },
                        onCancel: { vm.cancelDownload(row.model.id) },
                        onLoad: { loadInChat(row) },
                        onDelete: { pendingDeleteID = row.model.id }
                    )
                    .padding(.vertical, 4)
                }
            }
            storageSection
            localSection
        }
        .navigationTitle("Models")
        .onAppear { vm.refresh() }
        .alert("Cannot open bundle", isPresented: localBundleErrorBinding) {
            Button("OK", role: .cancel) { localBundleError = nil }
        } message: {
            Text(localBundleError ?? "")
        }
        .alert("Delete this model?", isPresented: deleteAlertBinding) {
            Button("Delete", role: .destructive) {
                if let id = pendingDeleteID, let row = vm.row(for: id) { performDelete(row) }
                pendingDeleteID = nil
            }
            Button("Cancel", role: .cancel) { pendingDeleteID = nil }
        } message: {
            Text("This removes the downloaded files from disk.")
        }
        .alert("Delete all models?", isPresented: $showDeleteAll) {
            Button("Delete All", role: .destructive) { deleteAll() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes every downloaded model from disk.")
        }
    }

    private var deleteAlertBinding: Binding<Bool> {
        Binding(
            get: { pendingDeleteID != nil },
            set: { if !$0 { pendingDeleteID = nil } }
        )
    }

    private var localBundleErrorBinding: Binding<Bool> {
        Binding(
            get: { localBundleError != nil },
            set: { if !$0 { localBundleError = nil } }
        )
    }

    @ViewBuilder
    private var localSection: some View {
        Section("Local") {
            Button {
                openLocalBundle()
            } label: {
                Label("Open local bundle…", systemImage: "folder")
            }
            .help("Load a model bundle from disk without downloading")
            Text("Point at a folder that contains manifest.json to run a bundle you already have.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var storageSection: some View {
        Section("Storage") {
            LabeledContent("Model storage") {
                Text(ByteFormatting.formatBytes(vm.modelsDirectorySize)).monospacedDigit()
            }
            LabeledContent("Disk available") {
                Text(ByteFormatting.formatBytes(vm.availableDiskSpace)).monospacedDigit()
            }
            Button(role: .destructive) {
                showDeleteAll = true
            } label: {
                Label("Delete All", systemImage: "trash")
            }
            .disabled(vm.modelsDirectorySize == 0 || vm.isBusy)
        }
    }

    private func loadInChat(_ row: ModelRowState) {
        loadPath(row.bundleDirectory.path)
    }

    private func loadPath(_ path: String) {
        Task { await chatVM.loadModel(path: path) }
        navigator.selection = .chat
    }

    private func openLocalBundle() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Load"
        panel.message = "Choose a model bundle directory (the folder that contains manifest.json)"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard FileManager.default.fileExists(atPath: url.appending(path: "manifest.json").path(percentEncoded: false)) else {
            localBundleError = "That folder has no manifest.json. Choose a model bundle directory."
            return
        }
        loadPath(url.path)
    }

    private func performDelete(_ row: ModelRowState) {
        if chatVM.loadedPath == row.bundleDirectory.path {
            chatVM.unload()
        }
        vm.delete(row.model.id)
    }

    private func deleteAll() {
        for row in vm.rows where chatVM.loadedPath == row.bundleDirectory.path {
            chatVM.unload()
        }
        vm.deleteAll()
    }
}

struct ModelRow: View {
    let row: ModelRowState
    var onDownload: () -> Void
    var onCancel: () -> Void
    var onLoad: () -> Void
    var onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(row.model.displayName)
                .fontWeight(.medium)
                .lineLimit(1)
                .truncationMode(.middle)

            if row.isDownloading {
                downloadingView
            } else if row.isDownloaded {
                downloadedView
            } else {
                notDownloadedView
            }

            if let error = row.error, !row.isDownloading {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
                    Text(error).foregroundStyle(.red).lineLimit(2)
                    Spacer()
                }
                .font(.caption)
            }
        }
    }

    @ViewBuilder
    private var downloadingView: some View {
        VStack(alignment: .leading, spacing: 3) {
            ProgressView(value: row.progress)
            HStack {
                if row.bytesPerSecond > 0 {
                    Text(ByteFormatting.formatSpeed(row.bytesPerSecond)).monospacedDigit()
                } else {
                    Text("Preparing…")
                }
                if let remaining = row.estimatedTimeRemaining {
                    Text("· \(ByteFormatting.formatDuration(remaining)) left").monospacedDigit()
                }
                Spacer()
                if row.totalBytes > 0 {
                    Text("\(ByteFormatting.formatBytes(row.downloadedBytes)) / \(ByteFormatting.formatBytes(row.totalBytes))")
                        .monospacedDigit()
                }
                Button("Cancel", action: onCancel).controlSize(.small)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var downloadedView: some View {
        HStack {
            Label(ByteFormatting.formatBytes(row.diskSize), systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.caption)
            Spacer()
            Button("Load in Chat", action: onLoad).controlSize(.small)
            Button(role: .destructive, action: onDelete) {
                Image(systemName: "trash")
            }
            .controlSize(.small)
        }
    }

    @ViewBuilder
    private var notDownloadedView: some View {
        HStack {
            Text("Not downloaded · \(row.model.approxSizeText)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button(row.error == nil ? "Download" : "Retry", action: onDownload)
                .controlSize(.small)
        }
    }
}
