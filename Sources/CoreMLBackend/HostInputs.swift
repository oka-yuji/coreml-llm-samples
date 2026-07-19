import CoreML
import Foundation

extension MLMultiArray {
    /// fp16 バッファとして書き換えるヘルパ。
    func withF16<R>(_ body: (UnsafeMutableBufferPointer<Float16>) -> R) -> R {
        withUnsafeMutableBytes { raw, _ in
            body(raw.bindMemory(to: Float16.self))
        }
    }
}

/// 位置依存のホスト計算(onehot / RoPE cos・sin / attention mask)。
/// グラフ内で position 整数から導出すると変換が壊れる(aten::Int)ため、
/// 毎ステップこちらで計算してテンソル入力として渡す(ラボ実証済みの方式)。
final class HostInputs {
    let onehot: MLMultiArray
    let cosS: MLMultiArray
    let sinS: MLMultiArray
    let cosF: MLMultiArray
    let sinF: MLMultiArray
    let maskS: MLMultiArray
    let maskF: MLMultiArray

    private let invSlide: [Float]
    private let invFull: [Float]
    private let ctx: Int
    private let sliding: Int
    private let neg: Float16
    private var lastOnehotPosition: Int?

    init(config: ChainConfig) throws {
        invSlide = config.hostio.invSlide.map(Float.init)
        invFull = config.hostio.invFull.map(Float.init)
        ctx = config.CTX
        sliding = config.SLIDING
        neg = Float16(config.NEG)

        func alloc(_ count: Int) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: [NSNumber(value: count)], dataType: .float16)
            a.withF16 { buf in buf.update(repeating: 0) }
            return a
        }
        onehot = try alloc(ctx)
        cosS = try alloc(config.HD)
        sinS = try alloc(config.HD)
        cosF = try alloc(config.GHD)
        sinF = try alloc(config.GHD)
        maskS = try alloc(ctx)
        maskF = try alloc(ctx)
    }

    /// 位置 p のホスト入力へバッファを更新する。
    func update(position p: Int) {
        onehot.withF16 { buf in
            if let last = lastOnehotPosition { buf[last] = 0 }
            buf[p] = 1
        }
        lastOnehotPosition = p

        // RoPE: 角度ベクトルは concat([inv*p, inv*p])。full は partial rotary
        // (inv_full の後半は 0 → cos=1, sin=0 = NoPE)が焼き込み済み。
        write(cos: cosS, sin: sinS, inv: invSlide, position: p)
        write(cos: cosF, sin: sinF, inv: invFull, position: p)

        maskS.withF16 { buf in
            for s in 0..<ctx {
                buf[s] = (s <= p && p - s < sliding) ? 0 : neg
            }
        }
        maskF.withF16 { buf in
            for s in 0..<ctx {
                buf[s] = s <= p ? 0 : neg
            }
        }
    }

    private func write(cos: MLMultiArray, sin: MLMultiArray, inv: [Float], position p: Int) {
        let half = inv.count
        cos.withF16 { cbuf in
            sin.withF16 { sbuf in
                for i in 0..<half {
                    let angle = inv[i] * Float(p)
                    let c = Float16(cosf(angle))
                    let s = Float16(sinf(angle))
                    cbuf[i] = c
                    cbuf[i + half] = c
                    sbuf[i] = s
                    sbuf[i + half] = s
                }
            }
        }
    }
}

/// S トークン一括 verify 用のホスト入力。形状は (S, CTX) / (S, HD) / (S, GHD)。
/// 行 s = 位置 base+s のホスト値(HostInputs と同じ数式)。
final class VerifyHostInputs {
    let onehot: MLMultiArray
    let cosS: MLMultiArray
    let sinS: MLMultiArray
    let cosF: MLMultiArray
    let sinF: MLMultiArray
    let maskS: MLMultiArray
    let maskF: MLMultiArray

    private let invSlide: [Float]
    private let invFull: [Float]
    private let ctx: Int
    private let sliding: Int
    private let neg: Float16
    private let s: Int

    init(config: ChainConfig, draftLen: Int) throws {
        invSlide = config.hostio.invSlide.map(Float.init)
        invFull = config.hostio.invFull.map(Float.init)
        ctx = config.CTX
        sliding = config.SLIDING
        neg = Float16(config.NEG)
        s = draftLen

        func alloc(_ cols: Int) throws -> MLMultiArray {
            try MLMultiArray(shape: [NSNumber(value: draftLen), NSNumber(value: cols)], dataType: .float16)
        }
        onehot = try alloc(ctx)
        cosS = try alloc(config.HD)
        sinS = try alloc(config.HD)
        cosF = try alloc(config.GHD)
        sinF = try alloc(config.GHD)
        maskS = try alloc(ctx)
        maskF = try alloc(ctx)
    }

    func update(basePosition: Int) {
        onehot.withF16 { buf in
            buf.update(repeating: 0)
            for row in 0..<s { buf[row * ctx + basePosition + row] = 1 }
        }
        write(cos: cosS, sin: sinS, inv: invSlide, base: basePosition)
        write(cos: cosF, sin: sinF, inv: invFull, base: basePosition)
        maskS.withF16 { buf in
            for row in 0..<s {
                let p = basePosition + row
                for slot in 0..<ctx {
                    buf[row * ctx + slot] = (slot <= p && p - slot < sliding) ? 0 : neg
                }
            }
        }
        maskF.withF16 { buf in
            for row in 0..<s {
                let p = basePosition + row
                for slot in 0..<ctx {
                    buf[row * ctx + slot] = slot <= p ? 0 : neg
                }
            }
        }
    }

    private func write(cos: MLMultiArray, sin: MLMultiArray, inv: [Float], base: Int) {
        let half = inv.count
        cos.withF16 { cbuf in
            sin.withF16 { sbuf in
                for row in 0..<s {
                    let offset = row * half * 2
                    for i in 0..<half {
                        let angle = inv[i] * Float(base + row)
                        let c = Float16(cosf(angle))
                        let sv = Float16(sinf(angle))
                        cbuf[offset + i] = c
                        cbuf[offset + i + half] = c
                        sbuf[offset + i] = sv
                        sbuf[offset + i + half] = sv
                    }
                }
            }
        }
    }
}

/// ドラフター(**1-D 入力**)用の単一位置ホスト入力。
/// 形状は cos_s/sin_s=(HD,)、cos_f/sin_f=(GHD,)、mask_s=(mask_s 幅,)、mask_f=(mask_f 幅,)。
///
/// v2 チェーンで MTP を回すとき、verify は v2 グラフ(S 可変・(S,・) 形状)で行うが、
/// ドラフターは入力ランク 1 の別モデル(`drafter.mlmodelc` / ring は `drafter_ring.mlmodelc`)を
/// そのまま使うため、ドラフターに渡すホスト値だけは 1-D で用意する。`onehot` は持たない
/// (ドラフターは store 層 KV へのクロスアテンションのみで、自層 KV の書き込みをしない)。
/// 位置はドラフト中ずっと固定(= 直前確定位置)。
///
/// - 非 ring(2048): mask_s/mask_f 幅 = CTX、RoPE は fp32(`HostInputs.update` とバイト等価)。
/// - ring/ladder(`usesSplitOnehot`): mask_s 幅 = windowSlide(1024)/ mask_f 幅 = drafterCtxFull
///   (ring=ctxFull、ladder=N_MAX=131072)。深部 pos の near-tie 反転を防ぐため RoPE は fp64
///   (`HostInputsV2.writeDouble` と同一数式)。mask 規則は S=1 リングと同一:
///   mask_s[j]=0 iff (j<=p or p>=windowSlide) / mask_f[j]=0 iff j<=p(絶対 causal)。
final class DrafterHostInputs {
    let cosS: MLMultiArray
    let sinS: MLMultiArray
    let cosF: MLMultiArray
    let sinF: MLMultiArray
    let maskS: MLMultiArray
    let maskF: MLMultiArray

    private let invSlide: [Float]
    private let invFull: [Float]
    // ring 用の fp64 版 inv(HostInputsV2 と同一。深部 pos で fp32 と食い違うため ring は fp64 が必須)。
    private let invSlideD: [Double]
    private let invFullD: [Double]
    private let sliding: Int
    private let neg: Float16
    /// ring/ladder(onehot 分離)モードか。true で fp64 RoPE + リング mask 規則。
    private let ring: Bool
    /// ring mask_s の全可視境界(pos>=windowSlide で全 windowSlide スロット可視)。非 ring は未使用。
    private let windowSlide: Int
    /// mask_s の列数(非 ring=CTX、ring/ladder=windowSlide)。
    private let maskSWidth: Int
    /// mask_f の列数(非 ring=CTX、ring/ladder=drafterCtxFull)。
    private let maskFWidth: Int

    /// `maskFWidthOverride`: ring/ladder で mask_f(と skf/svf の列数)を明示指定する。
    /// ladder の regime 切替(非昇格 = w32768 ドラフター / 昇格後 = w131072)で、非昇格用ホストに
    /// `config.ladderCtx32kWindow`(32768)を渡すために使う。nil(既定)= 従来どおり `config.drafterCtxFull`。
    /// 非 ring では常に CTX 幅(= override 無視)なので、非 ladder・非 ring 経路はバイト等価のまま。
    init(config: ChainConfigV2, maskFWidthOverride: Int? = nil) throws {
        invSlide = config.hostio.invSlide.map(Float.init)
        invFull = config.hostio.invFull.map(Float.init)
        invSlideD = config.hostio.invSlide
        invFullD = config.hostio.invFull
        sliding = config.SLIDING
        neg = Float16(config.NEG)
        ring = config.usesSplitOnehot
        windowSlide = config.windowSlide
        maskSWidth = ring ? config.windowSlide : config.CTX
        maskFWidth = ring ? (maskFWidthOverride ?? config.drafterCtxFull) : config.CTX

        func alloc(_ count: Int) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: [NSNumber(value: count)], dataType: .float16)
            a.withF16 { buf in buf.update(repeating: 0) }
            return a
        }
        cosS = try alloc(config.HD)
        sinS = try alloc(config.HD)
        cosF = try alloc(config.GHD)
        sinF = try alloc(config.GHD)
        maskS = try alloc(maskSWidth)
        maskF = try alloc(maskFWidth)
    }

    /// 位置 p のホスト入力へ更新する(onehot を除く)。
    /// 非 ring: `HostInputs.update` と同一数式(fp32 RoPE・CTX 幅の従来 mask)= バイト等価。
    /// ring/ladder: fp64 RoPE + リング mask(mask_s は pos>=windowSlide で全可視、mask_f は絶対 causal)。
    func update(position p: Int) {
        if ring {
            writeDouble(cos: cosS, sin: sinS, inv: invSlideD, position: p)
            writeDouble(cos: cosF, sin: sinF, inv: invFullD, position: p)
            maskS.withF16 { buf in
                for j in 0..<maskSWidth { buf[j] = (j <= p || p >= windowSlide) ? 0 : neg }
            }
            maskF.withF16 { buf in
                for j in 0..<maskFWidth { buf[j] = j <= p ? 0 : neg }
            }
        } else {
            write(cos: cosS, sin: sinS, inv: invSlide, position: p)
            write(cos: cosF, sin: sinF, inv: invFull, position: p)
            maskS.withF16 { buf in
                for s in 0..<maskSWidth { buf[s] = (s <= p && p - s < sliding) ? 0 : neg }
            }
            maskF.withF16 { buf in
                for s in 0..<maskFWidth { buf[s] = s <= p ? 0 : neg }
            }
        }
    }

    /// 非 ring 用 fp32 RoPE(従来と同一)。
    private func write(cos: MLMultiArray, sin: MLMultiArray, inv: [Float], position p: Int) {
        let half = inv.count
        cos.withF16 { cbuf in
            sin.withF16 { sbuf in
                for i in 0..<half {
                    let angle = inv[i] * Float(p)
                    let c = Float16(cosf(angle))
                    let s = Float16(sinf(angle))
                    cbuf[i] = c
                    cbuf[i + half] = c
                    sbuf[i] = s
                    sbuf[i + half] = s
                }
            }
        }
    }

    /// ring 用 fp64 RoPE(`HostInputsV2.writeDouble` と同一。深部 pos の near-tie 反転を防ぐ)。
    private func writeDouble(cos: MLMultiArray, sin: MLMultiArray, inv: [Double], position p: Int) {
        let half = inv.count
        cos.withF16 { cbuf in
            sin.withF16 { sbuf in
                let pd = Double(p)
                for i in 0..<half {
                    let angle = inv[i] * pd
                    let c = Float16(Foundation.cos(angle))
                    let s = Float16(Foundation.sin(angle))
                    cbuf[i] = c
                    cbuf[i + half] = c
                    sbuf[i] = s
                    sbuf[i + half] = s
                }
            }
        }
    }
}

/// v2 stateful チェーン用の **可変 S** ホスト入力。任意の S(1..maxS)について
/// (S, CTX) / (S, HD) / (S, GHD) 形状の onehot / RoPE cos,sin / mask を生成する。
///
/// `VerifyHostInputs`(固定 draftLen)を任意 S に一般化したもの。数式は `HostInputs` と
/// 完全に同一(python `np_host` と行単位で一致)で、先頭に S 次元を付けるだけ。
/// バッファは S ごとにキャッシュして使い回す(decode の S=1 / prefill の S=128・端数)。
final class HostInputsV2 {
    /// 1 回の予測に渡す入力バッファ(すべて (S, ・) fp16)。
    /// 既定 2048: 単一 `onehot`(列数 CTX)を使う(onehotS/onehotF は nil)。
    /// ctx32k リング: `onehotS`(列数 windowSlide=1024・リング)/ `onehotF`(列数 ctxFull=32768・絶対)を
    /// 使う(onehot は nil)。mask_s/mask_f の列数も各モードで変わる(2048=CTX / ring=1024・32768)。
    struct Buffers {
        let onehot: MLMultiArray?
        let onehotS: MLMultiArray?
        let onehotF: MLMultiArray?
        let cosS: MLMultiArray
        let sinS: MLMultiArray
        let cosF: MLMultiArray
        let sinF: MLMultiArray
        let maskS: MLMultiArray
        let maskF: MLMultiArray
    }

    private let invSlide: [Float]
    private let invFull: [Float]
    // ring 用の fp64 版 inv(python は np.float64 で角度計算 → fp16。深部 pos で float32 と fp16 が食い違う)。
    private let invSlideD: [Double]
    private let invFullD: [Double]
    private let ctx: Int
    private let sliding: Int
    private let hd: Int
    private let ghd: Int
    private let neg: Float16
    /// onehot を sliding(リング窓)/ full(絶対)に分離するか(ring / ladder = true、2048 = false)。
    private let splitOnehot: Bool
    private let windowSlide: Int
    /// full 層の絶対列数(= onehot_f / mask_f 幅)。ladder は function ごとに 32768 / 131072 と異なるため
    /// インスタンス生成時に明示する(2 インスタンスを持つ)。ring は 32768 / 131072、2048 は未使用。
    private let ctxFull: Int
    private var cache: [Int: Buffers] = [:]

    /// 既定 init: config から split / windowSlide / ctxFull を導出する。
    /// 2048 = 単一 onehot(splitOnehot=false)、ring = 分離(splitOnehot=true, ctxFull=32768/131072)。
    convenience init(config: ChainConfigV2) {
        self.init(
            config: config, splitOnehot: config.usesSplitOnehot,
            windowSlide: config.windowSlide, ctxFull: config.ctxFull)
    }

    /// ladder 用の明示 init: full 層窓幅を function ごとに指定する(ctx32k=32768 / ctx128k=131072)。
    /// sliding(onehot_s / mask_s / RoPE)は両 function 同一。
    init(config: ChainConfigV2, splitOnehot: Bool, windowSlide: Int, ctxFull: Int) {
        invSlide = config.hostio.invSlide.map(Float.init)
        invFull = config.hostio.invFull.map(Float.init)
        invSlideD = config.hostio.invSlide
        invFullD = config.hostio.invFull
        ctx = config.CTX
        sliding = config.SLIDING
        hd = config.HD
        ghd = config.GHD
        neg = Float16(config.NEG)
        self.splitOnehot = splitOnehot
        self.windowSlide = windowSlide
        self.ctxFull = ctxFull
    }

    /// 位置 `[base, base+count)` に対応する S=count のホスト入力を埋めて返す。
    /// 同じ S のバッファは再利用する(毎回 in-place で全域を書き直す)。
    /// ring/非 ring で onehot・mask の列数と規則が変わる(python `np_host_32k` / `np_host` と行単位一致)。
    func filled(base: Int, count s: Int) throws -> Buffers {
        let b = try buffers(for: s)
        if splitOnehot {
            // ring は深部 pos(>1000)を扱うため fp64 で角度・cos/sin を計算(python np.float64 と bit 一致)。
            writeDouble(cos: b.cosS, sin: b.sinS, inv: invSlideD, base: base, count: s)
            writeDouble(cos: b.cosF, sin: b.sinF, inv: invFullD, base: base, count: s)
        } else {
            // 2048 既定: 従来どおり fp32(小 pos では fp16 が fp64 と一致。バイト等価維持)。
            write(cos: b.cosS, sin: b.sinS, inv: invSlide, base: base, count: s)
            write(cos: b.cosF, sin: b.sinF, inv: invFull, base: base, count: s)
        }
        if splitOnehot {
            // sliding: リング書き込み slot = pos % windowSlide。可視 = (slot<=pos) or (pos>=windowSlide)。
            b.onehotS!.withF16 { buf in
                buf.update(repeating: 0)
                for row in 0..<s { buf[row * windowSlide + ((base + row) % windowSlide)] = 1 }
            }
            b.maskS.withF16 { buf in
                // リング窓のバッチ可視マスク。S=1 は従来と完全同一(slot<=pos、pos>=W で全可視)。
                // S>1 で窓が満杯になる(wrap する)バッチでは「このバッチで未来行 r>row が書いた
                // リングスロット」を不可視にして未来トークン KV への漏れを防ぐ。それ以外の 1024 スロットは
                // 可視(= リング内の最新 1024 位置のうち row から見て causal な分)。窓が満杯にならない
                // バッチ(base+S-1 < W)は wrap しないので従来どおり slot<=pos。物理リング == 窓(1024)の
                // ため row 0 は最古 S-1 位置を失う(tail-loss、不可避)= バッチ verify は pos>=W で厳密ロスレス
                // ではない。実効ロスレス性はゲートで実測する(docs/results C3 参照)。
                let wrapped = (base + s - 1) >= windowSlide
                for row in 0..<s {
                    let p = base + row
                    if !wrapped {
                        for slot in 0..<windowSlide { buf[row * windowSlide + slot] = slot <= p ? 0 : neg }
                    } else {
                        for slot in 0..<windowSlide { buf[row * windowSlide + slot] = 0 }
                        var r = row + 1
                        while r < s { buf[row * windowSlide + ((base + r) % windowSlide)] = neg; r += 1 }
                    }
                }
            }
            // full: 絶対書き込み slot = pos。可視 = 絶対 causal(slot<=pos)。
            b.onehotF!.withF16 { buf in
                buf.update(repeating: 0)
                for row in 0..<s { buf[row * ctxFull + (base + row)] = 1 }
            }
            b.maskF.withF16 { buf in
                for row in 0..<s {
                    let p = base + row
                    for slot in 0..<ctxFull { buf[row * ctxFull + slot] = slot <= p ? 0 : neg }
                }
            }
        } else {
            b.onehot!.withF16 { buf in
                buf.update(repeating: 0)
                for row in 0..<s { buf[row * ctx + base + row] = 1 }
            }
            b.maskS.withF16 { buf in
                for row in 0..<s {
                    let p = base + row
                    for slot in 0..<ctx { buf[row * ctx + slot] = (slot <= p && p - slot < sliding) ? 0 : neg }
                }
            }
            b.maskF.withF16 { buf in
                for row in 0..<s {
                    let p = base + row
                    for slot in 0..<ctx { buf[row * ctx + slot] = slot <= p ? 0 : neg }
                }
            }
        }
        return b
    }

    private func buffers(for s: Int) throws -> Buffers {
        if let b = cache[s] { return b }
        func alloc(_ cols: Int) throws -> MLMultiArray {
            try MLMultiArray(shape: [NSNumber(value: s), NSNumber(value: cols)], dataType: .float16)
        }
        let b: Buffers
        if splitOnehot {
            b = Buffers(
                onehot: nil,
                onehotS: try alloc(windowSlide), onehotF: try alloc(ctxFull),
                cosS: try alloc(hd), sinS: try alloc(hd),
                cosF: try alloc(ghd), sinF: try alloc(ghd),
                maskS: try alloc(windowSlide), maskF: try alloc(ctxFull)
            )
        } else {
            b = Buffers(
                onehot: try alloc(ctx),
                onehotS: nil, onehotF: nil,
                cosS: try alloc(hd), sinS: try alloc(hd),
                cosF: try alloc(ghd), sinF: try alloc(ghd),
                maskS: try alloc(ctx), maskF: try alloc(ctx)
            )
        }
        cache[s] = b
        return b
    }

    private func write(cos: MLMultiArray, sin: MLMultiArray, inv: [Float], base: Int, count s: Int) {
        let half = inv.count
        cos.withF16 { cbuf in
            sin.withF16 { sbuf in
                for row in 0..<s {
                    let offset = row * half * 2
                    for i in 0..<half {
                        let angle = inv[i] * Float(base + row)
                        let c = Float16(cosf(angle))
                        let sv = Float16(sinf(angle))
                        cbuf[offset + i] = c
                        cbuf[offset + i + half] = c
                        sbuf[offset + i] = sv
                        sbuf[offset + i + half] = sv
                    }
                }
            }
        }
    }

    /// ring 用 fp64 RoPE: 角度・cos/sin を Double(libm)で計算し fp16 へ丸める。python
    /// `np.cos((inv_f64 * float(p))).astype(float16)` と bit 一致(深部 pos の near-tie 反転を防ぐ)。
    private func writeDouble(cos: MLMultiArray, sin: MLMultiArray, inv: [Double], base: Int, count s: Int) {
        let half = inv.count
        cos.withF16 { cbuf in
            sin.withF16 { sbuf in
                for row in 0..<s {
                    let offset = row * half * 2
                    let p = Double(base + row)
                    for i in 0..<half {
                        let angle = inv[i] * p
                        let c = Float16(Foundation.cos(angle))
                        let sv = Float16(Foundation.sin(angle))
                        cbuf[offset + i] = c
                        cbuf[offset + i + half] = c
                        sbuf[offset + i] = sv
                        sbuf[offset + i + half] = sv
                    }
                }
            }
        }
    }
}

/// ツリー MTP verify(A2b、v1 = 主鎖 + 先頭 draft 位置の代替候補 1 ノード)用のホスト入力。
/// S = `mainCount + 1` 行を作る:
/// - 行 `0..<mainCount` = **主鎖**(連番スロット base+row・RoPE 位置 base+row・標準 causal)。
///   数式は `HostInputsV2` と行単位で完全一致(= 主鎖部分は標準 verify とビット等価)。
/// - 行 `mainCount` = **代替候補ノード**(scratch スロット `altSlot`・RoPE 位置 `base+1`・
///   可視 = prefix(0..base-1) + prediction(slot base) + self(altSlot))。主鎖の m1 と同 RoPE 位置の兄弟。
///
/// 設計はスパイク `tools/spikes/spike_tree_verify.py` の `tree_host_feats` に一致する:
/// onehot(書き込み先)/ cos,sin(RoPE)/ mask(可視)は独立入力で、物理スロット index は位置計算に
/// 使われない。よって「兄弟 = 同 RoPE 位置・別スロット」がホスト側だけで成立する(スパイク 18/18 exact)。
/// 標準 verify の `HostInputsV2` とは**別キャッシュ**を持ち、同じ S のバッファを共有しない(干渉回避)。
final class TreeHostInputs {
    private let invSlide: [Float]
    private let invFull: [Float]
    private let ctx: Int
    private let sliding: Int
    private let hd: Int
    private let ghd: Int
    private let neg: Float16
    private var cache: [Int: HostInputsV2.Buffers] = [:]   // key = mainCount(バッファ S = mainCount+1)

    init(config: ChainConfigV2) {
        invSlide = config.hostio.invSlide.map(Float.init)
        invFull = config.hostio.invFull.map(Float.init)
        ctx = config.CTX
        sliding = config.SLIDING
        hd = config.HD
        ghd = config.GHD
        neg = Float16(config.NEG)
    }

    /// base から始まる主鎖 `mainCount` トークン + 代替ノード(scratch=`altSlot`、RoPE 位置 base+1)の
    /// S=mainCount+1 ホスト入力を埋めて返す。`altSlot` は主鎖の使用域(base..base+mainCount-1)の外側
    /// (= base+mainCount 以降)である前提。同じ S のバッファは再利用する(毎回全域を書き直す)。
    func filled(base: Int, mainCount: Int, altSlot: Int) throws -> HostInputsV2.Buffers {
        let b = try buffers(for: mainCount)
        let altRow = mainCount
        let altPos = base + 1

        b.onehot!.withF16 { buf in
            buf.update(repeating: 0)
            for row in 0..<mainCount { buf[row * ctx + base + row] = 1 }
            buf[altRow * ctx + altSlot] = 1
        }
        writeRoPE(b.cosS, b.sinS, inv: invSlide, base: base, mainCount: mainCount, altPos: altPos)
        writeRoPE(b.cosF, b.sinF, inv: invFull, base: base, mainCount: mainCount, altPos: altPos)

        // full 可視マスク: 主鎖行は標準 causal、代替行は {0..base}(prefix+prediction)∪{altSlot}(self) のみ。
        b.maskF.withF16 { buf in
            for row in 0..<mainCount {
                let p = base + row
                for slot in 0..<ctx { buf[row * ctx + slot] = slot <= p ? 0 : neg }
            }
            for slot in 0..<ctx { buf[altRow * ctx + slot] = neg }
            for slot in 0...base { buf[altRow * ctx + slot] = 0 }
            buf[altRow * ctx + altSlot] = 0
        }
        // sliding 可視マスク: 主鎖行は標準、代替行は stored 位置(prefix/prediction は slot index、self は altPos)
        // からの距離 < sliding のみ可視。
        b.maskS.withF16 { buf in
            for row in 0..<mainCount {
                let p = base + row
                for slot in 0..<ctx { buf[row * ctx + slot] = (slot <= p && p - slot < sliding) ? 0 : neg }
            }
            for slot in 0..<ctx { buf[altRow * ctx + slot] = neg }
            for slot in 0...base where altPos - slot < sliding { buf[altRow * ctx + slot] = 0 }
            buf[altRow * ctx + altSlot] = 0   // self(距離 0 < sliding)
        }
        return b
    }

    private func buffers(for mainCount: Int) throws -> HostInputsV2.Buffers {
        if let b = cache[mainCount] { return b }
        let s = mainCount + 1
        func alloc(_ cols: Int) throws -> MLMultiArray {
            try MLMultiArray(shape: [NSNumber(value: s), NSNumber(value: cols)], dataType: .float16)
        }
        // ツリー MTP は 2048 モード専用(ctx32k では MTP をガード無効化)。単一 onehot を使う。
        let b = HostInputsV2.Buffers(
            onehot: try alloc(ctx),
            onehotS: nil, onehotF: nil,
            cosS: try alloc(hd), sinS: try alloc(hd),
            cosF: try alloc(ghd), sinF: try alloc(ghd),
            maskS: try alloc(ctx), maskF: try alloc(ctx))
        cache[mainCount] = b
        return b
    }

    /// 行 `0..<mainCount` は位置 base+row、行 `mainCount`(代替)は位置 `altPos` の cos/sin を書く。
    private func writeRoPE(
        _ cos: MLMultiArray, _ sin: MLMultiArray, inv: [Float], base: Int, mainCount: Int, altPos: Int
    ) {
        let half = inv.count
        cos.withF16 { cbuf in
            sin.withF16 { sbuf in
                func writeRow(_ row: Int, pos: Int) {
                    let offset = row * half * 2
                    for i in 0..<half {
                        let angle = inv[i] * Float(pos)
                        let c = Float16(cosf(angle))
                        let sv = Float16(sinf(angle))
                        cbuf[offset + i] = c
                        cbuf[offset + i + half] = c
                        sbuf[offset + i] = sv
                        sbuf[offset + i + half] = sv
                    }
                }
                for row in 0..<mainCount { writeRow(row, pos: base + row) }
                writeRow(mainCount, pos: altPos)
            }
        }
    }
}
