#if os(iOS)
import Observation
import SwiftUI
import UIKit

@MainActor
@Observable
final class SelfTestLog {
    static let fileName = "selftest-output.txt"

    private(set) var lines: [String] = []
    private(set) var finished = false

    @ObservationIgnored private let url: URL

    init() {
        url = ModelStorage.documentsDirectory().appending(path: Self.fileName)
        try? Data().write(to: url, options: .atomic)
    }

    var fileURL: URL { url }

    func record(_ line: String) {
        lines.append(line)
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        try? handle.seekToEnd()
        try? handle.write(contentsOf: Data((line + "\n").utf8))
    }

    func markFinished() { finished = true }
}

struct SelfTestView: View {
    @State private var log = SelfTestLog()

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if log.finished {
                    Image(systemName: "checkmark.circle").foregroundStyle(.green)
                } else {
                    ProgressView().controlSize(.small)
                }
                Text(log.finished ? "Self-test finished" : "Self-test running")
                    .font(.headline)
            }
            Text(log.fileURL.lastPathComponent)
                .font(.caption)
                .foregroundStyle(.secondary)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(Array(log.lines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
        .padding(12)
        .task { await run() }
    }

    private func run() async {
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }
        let sink = SelfTest.Sink(
            out: { log.record($0) },
            err: { log.record($0) })
        let code = await SelfTest.execute(SelfTest.options(), sink: sink)
        log.record("[selftest] exit \(code)")
        log.markFinished()
    }
}
#endif
