import Foundation

struct StreamPresenter {
    private var pending: [Character] = []
    private(set) var displayed: String = ""

    var pendingCount: Int { pending.count }
    var hasPending: Bool { !pending.isEmpty }

    mutating func append(_ text: String) {
        pending.append(contentsOf: text)
    }

    @discardableResult
    mutating func tick() -> Bool {
        guard !pending.isEmpty else { return false }
        let batch = max(1, Int((Double(pending.count) / 10.0).rounded(.up)))
        let count = min(batch, pending.count)
        displayed.append(contentsOf: pending[..<count])
        pending.removeFirst(count)
        return true
    }

    mutating func flush() {
        guard !pending.isEmpty else { return }
        displayed.append(contentsOf: pending)
        pending.removeAll()
    }

    mutating func reset() {
        pending.removeAll()
        displayed.removeAll()
    }
}
