// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "CoreLLMKit",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
        .visionOS(.v26),
    ],
    products: [
        .library(name: "LLMCore", targets: ["LLMCore"]),
        .library(name: "CoreMLBackend", targets: ["CoreMLBackend"]),
        .executable(name: "corellm-chat", targets: ["corellm-chat"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
    ],
    targets: [
        .target(name: "LLMCore"),
        .target(
            name: "CoreMLBackend",
            dependencies: [
                "LLMCore",
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "corellm-chat",
            dependencies: ["CoreMLBackend", "LLMCore"]
        ),
    ]
)
