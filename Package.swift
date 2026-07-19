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
        // Embeddable libraries: the pure-Swift core (types, protocols) and the Core ML backend.
        .library(name: "LLMCore", targets: ["LLMCore"]),
        .library(name: "CoreMLBackend", targets: ["CoreMLBackend"]),
        // Ready-to-run streaming chat CLI.
        .executable(name: "corellm-chat", targets: ["corellm-chat"]),
    ],
    dependencies: [
        // Tokenizer only. Hidden behind the `Tokenizing` protocol so LLMCore stays vocab-agnostic.
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
