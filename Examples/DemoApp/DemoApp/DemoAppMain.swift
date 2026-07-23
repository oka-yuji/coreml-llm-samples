import SwiftUI

/// このアプリが持つデモの一覧。**デモを増やすときはここに case を 1 つ足し、`Demos/` 下にフォルダを 1 つ
/// 作るだけ**でよい(サイドバーも詳細ペインもこの enum 1 つで駆動する)。今は Chat のみ。
enum Demo: String, CaseIterable, Identifiable {
    case chat

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chat: return "Chat"
        }
    }

    var summary: String {
        switch self {
        case .chat: return "Streaming chat with a Core ML LLM bundle"
        }
    }

    var systemImage: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        }
    }

    /// 選択中デモの画面。詳細ペインに表示する。
    @MainActor @ViewBuilder var view: some View {
        switch self {
        case .chat: ChatView()
        }
    }
}

/// サイドバーにデモ一覧、詳細側に選択中デモの画面。ウィンドウ 1 枚。
struct DemoRootView: View {
    @State private var selection: Demo? = .chat

    var body: some View {
        NavigationSplitView {
            List(selection: $selection) {
                ForEach(Demo.allCases) { demo in
                    NavigationLink(value: demo) {
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
            }
            .navigationTitle("Demos")
            .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        } detail: {
            if let selection {
                selection.view
            } else {
                ContentUnavailableView("Select a demo", systemImage: "square.grid.2x2")
            }
        }
    }
}

/// アプリのシーン。ウィンドウ 1 枚 = デモ一覧。
struct DemoApp: App {
    var body: some Scene {
        WindowGroup("Core ML Demos") {
            DemoRootView()
        }
        .defaultSize(width: 900, height: 640)
    }
}

/// エントリポイント。`--selftest` が渡されたら UI を出さずヘッドレスで Chat デモの ViewModel を 1 往復
/// 駆動して exit する(GUI 経路の自動検証用。README には載せない開発用フック)。それ以外は通常起動。
@main
enum DemoAppMain {
    static func main() {
        if CommandLine.arguments.contains("--selftest") {
            SelfTest.run()  // Never returns.
        }
        DemoApp.main()
    }
}
