import SwiftUI

struct LiveCameraView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(DemoNavigator.self) private var navigator
    @Environment(\.scenePhase) private var scenePhase

    @State private var live = LiveCameraViewModel()
    @State private var camera = CameraCapture()
    @State private var cameraReady = false
    @State private var cameraError = ""

    var body: some View {
        Group {
            if chat.supportsImageAttachment {
                liveScreen
            } else {
                unavailableState
            }
        }
        .navigationTitle("Live Camera")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .onDisappear { teardown() }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { teardown() }
        }
    }

    private var liveScreen: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .bottom) {
                if cameraReady {
                    CameraPreview(session: camera.session)
                } else {
                    Color.black.overlay {
                        VStack(spacing: 8) {
                            if cameraError.isEmpty {
                                ProgressView()
                                Text("Starting the camera…")
                            } else {
                                Image(systemName: "video.slash")
                                    .font(.largeTitle)
                                Text(cameraError)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal)
                            }
                        }
                        .font(.callout)
                        .foregroundStyle(.white)
                    }
                }
                captionOverlay
            }
            .clipped()
            Divider()
            controlBar
        }
        .task {
            live.startThermalWatch()
            await prepareCamera()
        }
    }

    private var captionOverlay: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(live.caption.isEmpty ? placeholderCaption : live.caption)
                .font(.headline)
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity, alignment: .leading)
            if !live.breakdown.isEmpty {
                Text(live.breakdown)
                    .font(.caption2)
                    .monospacedDigit()
                    .foregroundStyle(.white.opacity(0.75))
            }
        }
        .padding(12)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
        .padding(12)
    }

    private var placeholderCaption: String {
        if live.isPausedByThermal { return "Paused while the device cools down." }
        return live.isRunning ? "Reading the first frame…" : "Press Start to describe what the camera sees."
    }

    private var controlBar: some View {
        VStack(spacing: 6) {
            HStack(spacing: 12) {
                Button(live.isRunning ? "Stop" : "Start") {
                    live.isRunning ? live.stop() : startLoop()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canToggle)

                Spacer()

                Label(String(format: "%.1fs", live.lastCycleSeconds), systemImage: "timer")
                    .monospacedDigit()
                Label(live.thermalState, systemImage: thermalGlyph)
                    .foregroundStyle(live.isPausedByThermal ? Color.orange : Color.secondary)
                Text("\(live.cycleCount)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            .font(.callout)

            Text(live.statusLine.isEmpty ? "Ready." : live.statusLine)
                .font(.caption)
                .foregroundStyle(live.errorText.isEmpty ? Color.secondary : Color.orange)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
    }

    private var thermalGlyph: String {
        live.isPausedByThermal ? "thermometer.high" : "thermometer.medium"
    }

    private var canToggle: Bool {
        if live.isRunning { return true }
        return cameraReady && chat.engineHandle != nil && !chat.isGenerating && !chat.isLoading
    }

    private var unavailableState: some View {
        VStack(spacing: 12) {
            Image(systemName: "camera.metering.unknown")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("No image-capable model loaded").font(.headline)
            Text("This demo needs a loaded bundle that has a vision encoder next to it. "
                + "Load one from Models, then come back.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Open Models") { navigator.selection = .models }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private func prepareCamera() async {
        guard !cameraReady else { return }
        cameraError = ""
        guard await CameraCapture.requestPermission() else {
            cameraError = CameraError.permissionDenied.description
            return
        }
        do {
            try camera.configure()
        } catch {
            cameraError = String(describing: error)
            return
        }
        camera.start()
        cameraReady = true
    }

    private func startLoop() {
        guard let handle = chat.engineHandle else { return }
        live.start(
            engine: handle.engine, source: camera, speculative: handle.speculative,
            modelID: handle.modelID, hfRevision: handle.hfRevision,
            computeUnits: handle.computeUnits, bundleFolder: handle.bundleFolder)
    }

    private func teardown() {
        live.stop()
        live.stopThermalWatch()
        camera.stop()
        cameraReady = false
    }
}
