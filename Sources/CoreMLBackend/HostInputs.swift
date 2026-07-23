import CoreML
import Foundation

extension MLMultiArray {

    func withF16<R>(_ body: (UnsafeMutableBufferPointer<Float16>) -> R) -> R {
        withUnsafeMutableBytes { raw, _ in
            body(raw.bindMemory(to: Float16.self))
        }
    }
}

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

    func update(position p: Int) {
        onehot.withF16 { buf in
            if let last = lastOnehotPosition { buf[last] = 0 }
            buf[p] = 1
        }
        lastOnehotPosition = p

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

final class DrafterHostInputs {
    let cosS: MLMultiArray
    let sinS: MLMultiArray
    let cosF: MLMultiArray
    let sinF: MLMultiArray
    let maskS: MLMultiArray
    let maskF: MLMultiArray

    private let invSlide: [Float]
    private let invFull: [Float]

    private let invSlideD: [Double]
    private let invFullD: [Double]
    private let sliding: Int
    private let neg: Float16

    private let ring: Bool

    private let windowSlide: Int

    private let maskSWidth: Int

    private let maskFWidth: Int

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

final class HostInputsV2 {

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

    private let invSlideD: [Double]
    private let invFullD: [Double]
    private let ctx: Int
    private let sliding: Int
    private let hd: Int
    private let ghd: Int
    private let neg: Float16

    private let splitOnehot: Bool
    private let windowSlide: Int

    private let ctxFull: Int
    private var cache: [Int: Buffers] = [:]

    convenience init(config: ChainConfigV2) {
        self.init(
            config: config, splitOnehot: config.usesSplitOnehot,
            windowSlide: config.windowSlide, ctxFull: config.ctxFull)
    }

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

    func filled(base: Int, count s: Int) throws -> Buffers {
        let b = try buffers(for: s)
        if splitOnehot {

            writeDouble(cos: b.cosS, sin: b.sinS, inv: invSlideD, base: base, count: s)
            writeDouble(cos: b.cosF, sin: b.sinF, inv: invFullD, base: base, count: s)
        } else {

            write(cos: b.cosS, sin: b.sinS, inv: invSlide, base: base, count: s)
            write(cos: b.cosF, sin: b.sinF, inv: invFull, base: base, count: s)
        }
        if splitOnehot {

            b.onehotS!.withF16 { buf in
                buf.update(repeating: 0)
                for row in 0..<s { buf[row * windowSlide + ((base + row) % windowSlide)] = 1 }
            }
            b.maskS.withF16 { buf in

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

final class TreeHostInputs {
    private let invSlide: [Float]
    private let invFull: [Float]
    private let ctx: Int
    private let sliding: Int
    private let hd: Int
    private let ghd: Int
    private let neg: Float16
    private var cache: [Int: HostInputsV2.Buffers] = [:]

    init(config: ChainConfigV2) {
        invSlide = config.hostio.invSlide.map(Float.init)
        invFull = config.hostio.invFull.map(Float.init)
        ctx = config.CTX
        sliding = config.SLIDING
        hd = config.HD
        ghd = config.GHD
        neg = Float16(config.NEG)
    }

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

        b.maskF.withF16 { buf in
            for row in 0..<mainCount {
                let p = base + row
                for slot in 0..<ctx { buf[row * ctx + slot] = slot <= p ? 0 : neg }
            }
            for slot in 0..<ctx { buf[altRow * ctx + slot] = neg }
            for slot in 0...base { buf[altRow * ctx + slot] = 0 }
            buf[altRow * ctx + altSlot] = 0
        }

        b.maskS.withF16 { buf in
            for row in 0..<mainCount {
                let p = base + row
                for slot in 0..<ctx { buf[row * ctx + slot] = (slot <= p && p - slot < sliding) ? 0 : neg }
            }
            for slot in 0..<ctx { buf[altRow * ctx + slot] = neg }
            for slot in 0...base where altPos - slot < sliding { buf[altRow * ctx + slot] = 0 }
            buf[altRow * ctx + altSlot] = 0
        }
        return b
    }

    private func buffers(for mainCount: Int) throws -> HostInputsV2.Buffers {
        if let b = cache[mainCount] { return b }
        let s = mainCount + 1
        func alloc(_ cols: Int) throws -> MLMultiArray {
            try MLMultiArray(shape: [NSNumber(value: s), NSNumber(value: cols)], dataType: .float16)
        }

        let b = HostInputsV2.Buffers(
            onehot: try alloc(ctx),
            onehotS: nil, onehotF: nil,
            cosS: try alloc(hd), sinS: try alloc(hd),
            cosF: try alloc(ghd), sinF: try alloc(ghd),
            maskS: try alloc(ctx), maskF: try alloc(ctx))
        cache[mainCount] = b
        return b
    }

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
