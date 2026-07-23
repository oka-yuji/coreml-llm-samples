import SwiftUI
import UniformTypeIdentifiers

struct SingleModelsView: View {
    @Environment(ModelsViewModel.self) private var vm
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(DemoNavigator.self) private var navigator

    @State private var pendingDeleteID: String?
    @State private var importing = false

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
            Section("Storage") {
                LabeledContent("Model storage") {
                    Text(ByteFormatting.formatBytes(vm.modelsDirectorySize)).monospacedDigit()
                }
                LabeledContent("Disk available") {
                    Text(ByteFormatting.formatBytes(vm.availableDiskSpace)).monospacedDigit()
                }
            }
            Section("Local") {
                Button {
                    importing = true
                } label: {
                    Label("Open local bundle…", systemImage: "folder")
                }
            }
        }
        .navigationTitle("Models")
        .onAppear { vm.refresh() }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
            guard case .success(let url) = result else { return }
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }
            guard FileManager.default.fileExists(atPath: url.appending(path: "manifest.json").path(percentEncoded: false)) else { return }
            loadPath(url.path)
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
    }

    private var deleteAlertBinding: Binding<Bool> {
        Binding(
            get: { pendingDeleteID != nil },
            set: { if !$0 { pendingDeleteID = nil } }
        )
    }

    private func loadInChat(_ row: ModelRowState) {
        loadPath(row.bundleDirectory.path)
    }

    private func loadPath(_ path: String) {
        Task { await chatVM.loadModel(path: path) }
        navigator.selection = .chat
    }

    private func performDelete(_ row: ModelRowState) {
        if chatVM.loadedPath == row.bundleDirectory.path {
            chatVM.unload()
        }
        vm.delete(row.model.id)
    }
}
