#if os(iOS)
import SwiftUI
import UIKit

struct LiveSelfTestView: View {
    @State private var log = SelfTestLog()

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if log.finished {
                    Image(systemName: "checkmark.circle").foregroundStyle(.green)
                } else {
                    ProgressView().controlSize(.small)
                }
                Text(log.finished ? "Live self-test finished" : "Live self-test running")
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
        let code = await LiveCameraSelfTest.execute(LiveCameraSelfTest.options(), sink: sink)
        log.record("[live-selftest] exit \(code)")
        log.markFinished()
    }
}
#endif
