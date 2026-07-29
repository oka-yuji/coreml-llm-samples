import Foundation
import SwiftUI

struct DemoApp: App {
    var body: some Scene {
        WindowGroup("Core ML Demos") {
            DemoRootView()
        }
        .defaultSize(width: 900, height: 640)
    }
}

@main
enum DemoAppMain {
    static func main() {
        // Credential cleanup for builds that had a token field.
        UserDefaults.standard.removeObject(forKey: "huggingface_token")
        if CommandLine.arguments.contains("--selftest") {
            SelfTest.run()
        }
        if CommandLine.arguments.contains("--models-selftest") {
            ModelsSelfTest.run()
        }
        DemoApp.main()
    }
}
