import CoreMLBackend
import SwiftUI

struct LiveCameraView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(DemoNavigator.self) private var navigator
    @Environment(\.scenePhase) private var scenePhase

    @State private var live = LiveCameraViewModel()
    @State private var camera = CameraCapture()
    @State private var cameraReady = false
    @State private var cameraError = ""
    @State private var starting = false
    @State private var loadingStage = ""
    @State private var offerModels = false

    var body: some View {
        liveScreen
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
                if starting { loadingGate }
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

    private var loadingGate: some View {
        VStack(spacing: 10) {
            ProgressView()
                .controlSize(.large)
                .tint(.white)
            Text(loadingStage.isEmpty ? "Preparing…" : loadingStage)
                .font(.callout)
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)
            Text("One-time setup for this model. The camera stays live.")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
        }
        .padding(20)
        .background(.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 16))
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var captionOverlay: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(overlayText)
                .font(.headline)
                .foregroundStyle(live.errorText.isEmpty ? Color.white : Color.orange)
                .frame(maxWidth: .infinity, alignment: .leading)
            if live.errorText.isEmpty, !live.breakdown.isEmpty {
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

    private var phaseCapsule: some View {
        Label(live.cyclePhase.label, systemImage: live.cyclePhase.glyph)
            .font(.caption2)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                live.cyclePhase == .idle ? Color.secondary.opacity(0.15) : Color.accentColor.opacity(0.2),
                in: Capsule())
            .foregroundStyle(live.cyclePhase == .idle ? Color.secondary : Color.accentColor)
    }

    private var overlayText: String {
        if !live.errorText.isEmpty { return live.errorText }
        return live.caption.isEmpty ? placeholderCaption : live.caption
    }

    private var placeholderCaption: String {
        if live.isPausedByThermal { return "Paused while the device cools down." }
        if starting { return loadingStage.isEmpty ? "Preparing the model…" : loadingStage }
        return live.isRunning ? "Reading the first frame…" : "Press Start to describe what the camera sees."
    }

    private var controlBar: some View {
        VStack(spacing: 6) {
            HStack(spacing: 12) {
                Button(startStopTitle) {
                    if live.isRunning { live.stop() } else { Task { await startLoop() } }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canToggle)

                if live.isRunning { phaseCapsule }

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
                .lineLimit(3)
                .frame(maxWidth: .infinity, alignment: .leading)

            if offerModels {
                Button("Open Models") { navigator.selection = .models }
                    .font(.caption)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(12)
    }

    private var startStopTitle: String {
        if live.isRunning { return "Stop" }
        return starting ? "Loading…" : "Start"
    }

    private var thermalGlyph: String {
        live.isPausedByThermal ? "thermometer.high" : "thermometer.medium"
    }

    private var canToggle: Bool {
        if live.isRunning { return true }
        return !starting && !chat.isGenerating && !chat.isLoading
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
        var waited = 0.0
        while !camera.hasFrame, waited < 10 {
            if Task.isCancelled { return }
            try? await Task.sleep(for: .milliseconds(200))
            waited += 0.2
        }
        guard camera.hasFrame else {
            cameraError = CameraError.noFrame.description
            return
        }
        cameraReady = true
    }

    private func startLoop() async {
        guard !starting, !live.isRunning else { return }
        starting = true
        offerModels = false
        live.errorText = ""
        stage("Starting the camera…")
        defer { starting = false; loadingStage = "" }

        if !cameraReady {
            await prepareCamera()
            guard cameraReady else {
                let message = cameraError.isEmpty ? CameraError.noCamera.description : cameraError
                live.errorText = message
                live.statusLine = message
                return
            }
        }

        stage("Checking the loaded model…")
        switch await LiveEngineProvision.ensureVisionEngine(
            chat: chat, status: { stage($0) })
        {
        case .unavailable(let message):
            live.errorText = message
            live.statusLine = message
            offerModels = true
        case .ready(let handle):
            guard cameraReady else {
                live.statusLine = "The camera stopped while the model was loading. Press Start again."
                return
            }
            stage("Loading the vision encoder…")
            do {
                let encoder = try await handle.engine.loadLiveVisionEncoder()
                stage("Preparing the prompt pipeline…")
                _ = try await handle.engine.prepareLivePrefill(
                    question: LiveCameraViewModel.question, imageRows: encoder.imageRows)
            } catch {
                let message = "Preparing the vision pipeline failed: \(error)"
                live.errorText = message
                live.statusLine = message
                return
            }
            live.start(
                engine: handle.engine, source: camera, speculative: handle.speculative,
                modelID: handle.modelID, hfRevision: handle.hfRevision,
                computeUnits: handle.computeUnits, bundleFolder: handle.bundleFolder)
        }
    }

    private func stage(_ text: String) {
        loadingStage = text
        live.statusLine = text
    }

    private func teardown() {
        live.cancel()
        camera.stop()
        cameraReady = false
        if let engine = chat.engineHandle?.engine {
            Task { await engine.endLiveVisionSession() }
        }
    }
}
