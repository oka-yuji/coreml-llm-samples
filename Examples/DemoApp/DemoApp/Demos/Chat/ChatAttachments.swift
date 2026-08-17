import SwiftUI
#if os(iOS)
import PhotosUI
#else
import UniformTypeIdentifiers
#endif

struct ImageThumbnail: View {
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

struct ImageAttachmentChip: View {
    @Environment(ChatViewModel.self) private var vm
    let url: URL

    var body: some View {
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
}

struct AudioAttachmentChip: View {
    @Environment(ChatViewModel.self) private var vm

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: vm.isRecording ? "record.circle" : "waveform")
                .foregroundStyle(vm.isRecording ? .red : .secondary)
            if vm.isRecording {
                Text("Recording \(vm.recordingLabel)")
                    .font(.caption)
                    .monospacedDigit()
            } else if let audio = vm.attachedAudio {
                Text(String(format: "Audio %.1fs ready. Send transcribes it.", audio.seconds))
                    .font(.caption)
            }
            Spacer(minLength: 0)
            if !vm.isRecording {
                Button(action: vm.clearAttachedAudio) {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .accessibilityLabel("Remove audio")
            }
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.bubbleBackground))
    }
}

struct AttachImageButton: View {
    @Environment(ChatViewModel.self) private var vm
    var glyphSize: CGFloat = 20

    #if os(iOS)
    @State private var pickedItem: PhotosPickerItem?
    #else
    @State private var showImporter = false
    #endif

    private var isDisabled: Bool { vm.isGenerating || vm.isLoading || vm.isRecording }

    var body: some View {
        #if os(iOS)
        PhotosPicker(selection: $pickedItem, matching: .images, photoLibrary: .shared()) {
            glyph
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .accessibilityLabel("Attach image")
        .onChange(of: pickedItem) { _, item in
            guard let item else { return }
            Task { await stage(item) }
        }
        #else
        Button { showImporter = true } label: { glyph }
            .buttonStyle(.plain)
            .disabled(isDisabled)
            .accessibilityLabel("Attach image")
            .fileImporter(isPresented: $showImporter, allowedContentTypes: [.image]) { result in
                if case .success(let url) = result { vm.attachImage(url) }
            }
        #endif
    }

    private var glyph: some View {
        Image(systemName: "photo.on.rectangle")
            .font(.system(size: glyphSize))
            .foregroundStyle(Color.secondary)
    }

    #if os(iOS)
    private func stage(_ item: PhotosPickerItem) async {
        let fileExtension = item.supportedContentTypes.first?.preferredFilenameExtension ?? "jpg"
        do {
            guard let data = try await item.loadTransferable(type: Data.self) else {
                vm.statusLine = "The selected photo could not be read."
                pickedItem = nil
                return
            }
            vm.attachImage(data: data, fileExtension: fileExtension)
        } catch {
            vm.statusLine = "The selected photo could not be read: \(error)"
        }
        pickedItem = nil
    }
    #endif
}

struct RecordAudioButton: View {
    @Environment(ChatViewModel.self) private var vm
    var glyphSize: CGFloat = 20

    var body: some View {
        Button(action: vm.toggleRecording) {
            Image(systemName: vm.isRecording ? "stop.circle" : "mic")
                .font(.system(size: glyphSize))
        }
        .buttonStyle(.plain)
        .foregroundStyle(vm.isRecording ? Color.red : Color.secondary)
        .disabled(vm.isGenerating || vm.isLoading)
        .accessibilityLabel(vm.isRecording ? "Stop recording" : "Record audio")
    }
}
