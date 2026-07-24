import SwiftUI

struct SingleModelsView: View {
    @Environment(ModelsViewModel.self) private var vm
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(DemoNavigator.self) private var navigator

    @State private var pendingDeleteID: String?

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
        }
        .navigationTitle("Models")
        .onAppear { vm.refresh() }
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
