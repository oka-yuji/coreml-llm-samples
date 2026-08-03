import Observation
import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

enum Demo: String, CaseIterable, Identifiable {
    case chat
    case liveCamera
    case models

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chat: return "Chat"
        case .liveCamera: return "Live Camera"
        case .models: return "Models"
        }
    }

    var summary: String {
        switch self {
        case .chat: return "Streaming chat with a Core ML LLM bundle"
        case .liveCamera: return "Describe what the camera sees, frame after frame"
        case .models: return "Download and manage model bundles"
        }
    }

    var systemImage: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        case .liveCamera: return "camera.viewfinder"
        case .models: return "square.and.arrow.down"
        }
    }

    @MainActor @ViewBuilder var splitView: some View {
        switch self {
        case .chat: SplitChatView()
        case .liveCamera: LiveCameraView()
        case .models: SplitModelsView()
        }
    }

    @MainActor @ViewBuilder var singleView: some View {
        switch self {
        case .chat: SingleChatView()
        case .liveCamera: LiveCameraView()
        case .models: SingleModelsView()
        }
    }
}

@MainActor
@Observable
final class DemoNavigator {
    var selection: Demo? = .chat
}

enum DeviceKind {
    case iPhone, iPad, mac

    @MainActor
    static var current: DeviceKind {
        #if os(macOS)
        return .mac
        #else
        return UIDevice.current.userInterfaceIdiom == .phone ? .iPhone : .iPad
        #endif
    }
}

struct DemoRootView: View {
    var body: some View {
        Group {
            switch DeviceKind.current {
            case .mac, .iPad:
                SplitRootView()
            case .iPhone:
                SingleRootView()
            }
        }
        .task { MetricsLog.session(models: LLMModels.all.map { $0.id }) }
    }
}

struct SplitRootView: View {
    @State private var navigator = DemoNavigator()
    @State private var chatVM = ChatViewModel()
    @State private var modelsVM = ModelsViewModel()

    var body: some View {
        @Bindable var navigator = navigator
        NavigationSplitView {
            List(selection: $navigator.selection) {
                ForEach(Demo.allCases) { demo in
                    NavigationLink(value: demo) {
                        DemoRow(demo: demo)
                    }
                }
            }
            .navigationTitle("Demos")
            .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        } detail: {
            if let selection = navigator.selection {
                selection.splitView
            } else {
                ContentUnavailableView("Select a demo", systemImage: "square.grid.2x2")
            }
        }
        .environment(navigator)
        .environment(chatVM)
        .environment(modelsVM)
        .task { await chatVM.autoLoadLastBundleOnce() }
    }
}

struct SingleRootView: View {
    @State private var navigator = DemoNavigator()
    @State private var chatVM = ChatViewModel()
    @State private var modelsVM = ModelsViewModel()

    var body: some View {
        @Bindable var navigator = navigator
        TabView(selection: $navigator.selection) {
            ForEach(Demo.allCases) { demo in
                NavigationStack {
                    demo.singleView
                }
                .tabItem { Label(demo.title, systemImage: demo.systemImage) }
                .tag(demo as Demo?)
            }
        }
        .environment(navigator)
        .environment(chatVM)
        .environment(modelsVM)
        .task { await chatVM.autoLoadLastBundleOnce() }
    }
}

struct DemoRow: View {
    let demo: Demo

    var body: some View {
        Label {
            VStack(alignment: .leading, spacing: 2) {
                Text(demo.title)
                Text(demo.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } icon: {
            Image(systemName: demo.systemImage)
        }
    }
}
