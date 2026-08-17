import Foundation
import SwiftUI

struct DemoApp: App {
    var body: some Scene {
        WindowGroup("Core ML Demos") {
            #if os(iOS)
            if SelfTest.isRequested {
                SelfTestView()
            } else if LiveCameraSelfTest.isRequested {
                LiveSelfTestView()
            } else {
                DemoRootView()
            }
            #else
            DemoRootView()
            #endif
        }
        .defaultSize(width: 900, height: 640)
    }
}

@main
enum DemoAppMain {
    static func main() {
        // Credential cleanup for builds that had a token field.
        UserDefaults.standard.removeObject(forKey: "huggingface_token")
        #if os(macOS)
        if CommandLine.arguments.contains("--selftest") {
            SelfTest.run()
        }
        if CommandLine.arguments.contains("--models-selftest") {
            ModelsSelfTest.run()
        }
        if CommandLine.arguments.contains("--live-selftest") {
            LiveCameraSelfTest.run()
        }
        #endif
        DemoApp.main()
    }
}
