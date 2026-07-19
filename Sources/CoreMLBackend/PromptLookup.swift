import Foundation

/// Prompt-lookup(n-gram コピー)ドラフト探索。
///
/// 現在列の末尾 n-gram と同じ並びを文脈(prompt + 生成済み)から後方探索し、その直後の
/// トークン列を draft 候補としてコピーする。要約・引用・QA では出力が文脈の再利用だらけなので、
/// このコピーが本体 argmax と一致する率(採択率)が高い。**ロスレス性は verify 側が保証する**ため、
/// この探索は「候補を出すだけ」で品質には一切影響しない(採択率が変わるだけ)。
///
/// 探索は素朴な O(L) 逆走査(2048 では <1ms)。将来 128K で suffix automaton / ハッシュ索引へ
/// 差し替えられるよう、入力(トークン列)と出力(draft 候補)だけに依存する独立関数にしてある。
/// 現在列は `context + [last]` として渡す — `last` は直前に確定した予測トークン(= 位置 `context.count`)で、
/// 呼び出し側の連結コスト(O(L))を避けるため materialize せず末尾要素として扱う。
enum PromptLookup {
    /// 末尾 n-gram(n=maxN..minN でフォールバック)の一致を後方探索し、一致直後から最大 `maxDraft`
    /// トークンをコピーして返す。一致なしなら空配列(呼び出し側は従来ドラフターへ完全フォールバック)。
    ///
    /// - Parameters:
    ///   - context: 確定済みトークン列(位置 `[0, context.count)`)。
    ///   - last: 直前に確定した予測トークン(位置 `context.count`)。末尾 n-gram はこれで終わる。
    ///   - maxDraft: コピーする最大トークン数(= 呼び出し側の draft_len)。
    ///   - maxN / minN: n-gram 長の上限・下限(既定 3 → 2 フォールバック)。
    /// - Returns: draft 候補(1..maxDraft トークン、一致なしは空)。コピー元は既存トークンのみ
    ///   (最大でも `last` まで)なので「現在位置と重なる分」は構造的に発生しない。
    static func draft(
        context: [Int], last: Int, maxDraft: Int, maxN: Int = 3, minN: Int = 2
    ) -> [Int] {
        let L = context.count + 1  // 現在列の長さ(context + [last])。
        guard maxDraft > 0, minN >= 1, L >= minN + 1 else { return [] }

        // 現在列 seq[i]: i < context.count は context[i]、i == context.count は last。materialize しない。
        @inline(__always) func tok(_ i: Int) -> Int { i == context.count ? last : context[i] }

        // n を長い順に試す(長い一致ほど信頼できる → 採択率が高い)。
        var n = min(maxN, L - 1)
        while n >= minN {
            if let copied = copyAfterLatestMatch(tok: tok, length: L, n: n, maxDraft: maxDraft) {
                return copied
            }
            n -= 1
        }
        return []
    }

    /// 末尾 n トークン `seq[L-n..<L]` と一致する最新(最大 j)の出現を `seq[0..<L-n]` から後方探索し、
    /// 一致直後 `seq[j+n ..< min(j+n+maxDraft, L)]` をコピーして返す。一致なしは nil。
    /// 探索窓 `[j, j+n)` は末尾 n-gram と重ならない(j+n <= L-n)ため自己一致は起きない。
    private static func copyAfterLatestMatch(
        tok: (Int) -> Int, length L: Int, n: Int, maxDraft: Int
    ) -> [Int]? {
        let suffixStart = L - n
        var j = suffixStart - 1  // 最新一致優先: 大きい j から下る。
        while j >= 0 {
            var matched = true
            var k = 0
            while k < n {
                if tok(j + k) != tok(suffixStart + k) { matched = false; break }
                k += 1
            }
            if matched {
                let from = j + n                       // 一致直後(<= L-1 = last のインデックス)。
                let to = min(from + maxDraft, L)       // 現在列を越える分は自然に打ち切り。
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
