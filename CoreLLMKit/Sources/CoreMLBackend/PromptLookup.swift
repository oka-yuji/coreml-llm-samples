import Foundation

enum PromptLookup {

    static func draft(
        context: [Int], last: Int, maxDraft: Int, maxN: Int = 3, minN: Int = 2
    ) -> [Int] {
        let L = context.count + 1
        guard maxDraft > 0, minN >= 1, L >= minN + 1 else { return [] }

        @inline(__always) func tok(_ i: Int) -> Int { i == context.count ? last : context[i] }

        var n = min(maxN, L - 1)
        while n >= minN {
            if let copied = copyAfterLatestMatch(tok: tok, length: L, n: n, maxDraft: maxDraft) {
                return copied
            }
            n -= 1
        }
        return []
    }

    private static func copyAfterLatestMatch(
        tok: (Int) -> Int, length L: Int, n: Int, maxDraft: Int
    ) -> [Int]? {
        let suffixStart = L - n
        var j = suffixStart - 1
        while j >= 0 {
            var matched = true
            var k = 0
            while k < n {
                if tok(j + k) != tok(suffixStart + k) { matched = false; break }
                k += 1
            }
            if matched {
                let from = j + n
                let to = min(from + maxDraft, L)
                guard from < to else { return nil }
                var out = [Int]()
                out.reserveCapacity(to - from)
                var i = from
                while i < to { out.append(tok(i)); i += 1 }
                return out
            }
            j -= 1
        }
        return nil
    }
}
