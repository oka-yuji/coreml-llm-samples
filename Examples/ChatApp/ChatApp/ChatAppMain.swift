import SwiftUI

/// アプリのシーン。ウィンドウ 1 枚(設定画面・履歴保存・複数会話は作らない)。
struct ChatApp: App {
    var body: some Scene {
        WindowGroup("Core ML Chat") {
            ChatView()
        }
        .defaultSize(width: 720, height: 640)
    }
}

/// エントリポイント。`--selftest` が渡されたら UI を出さずヘッドレスで 1 往復生成して exit する
/// (GUI 経路の自動検証用。README には載せない開発用フック)。それ以外は通常の SwiftUI アプリを起動。
@main
enum ChatAppMain {
    static func main() {
        if CommandLine.arguments.contains("--selftest") {
            SelfTest.run()  // Never returns.
        }
        ChatApp.main()
    }
}
