import Accelerate
import AVFoundation
import CoreML
import Foundation

enum AudioPreprocess {
    static let sampleRate = 16000
    static let melBins = 128
    static let frameLength = 320
    static let hopLength = 160
    static let fftLength = 512
    static let fftLog2 = 9
    static let spectrumBins = fftLength / 2 + 1
    static let melFloor = 1e-3
    static let maxSamples = 480_000
    static let padMultiple = 128
    static let fixedFrames = 3000
    static let fixedTokens = 750

    static var maxSeconds: Double { Double(maxSamples) / Double(sampleRate) }

    enum PreprocessError: Error, CustomStringConvertible {
        case cannotDecodeAudio(String)
        case cannotMakeConverter(String)
        case emptyAudio

        var description: String {
            switch self {
            case .cannotDecodeAudio(let p): return "cannot decode audio: \(p)"
            case .cannotMakeConverter(let p): return "cannot resample to 16 kHz mono: \(p)"
            case .emptyAudio: return "the recording holds no samples"
            }
        }
    }

    static func paddedSampleCount(_ samples: Int) -> Int {
        let n = min(samples, maxSamples)
        if n % padMultiple != 0 { return ((n / padMultiple) + 1) * padMultiple }
        return n
    }

    static func melFrameCount(paddedSamples n: Int) -> Int {
        (n + hopLength - (frameLength + 1)) / hopLength + 1
    }

    static func softTokenCount(frames: Int) -> Int {
        var t = frames
        for _ in 0..<2 { t = (t - 1) / 2 + 1 }
        return t
    }

    static func softTokenCount(samples: Int) -> Int {
        softTokenCount(frames: melFrameCount(paddedSamples: paddedSampleCount(samples)))
    }

    struct LogMel {
        let features: [Float]
        let mask: [Bool]
        let frames: Int
    }

    static func logMel(samples: [Float]) -> LogMel {
        let padLeft = frameLength / 2
        let real = min(samples.count, maxSamples)
        let padded = paddedSampleCount(real)
        var wave = [Float](repeating: 0, count: padLeft + padded)
        if real > 0 {
            samples.withUnsafeBufferPointer { src in
                wave.withUnsafeMutableBufferPointer { dst in
                    (dst.baseAddress! + padLeft).update(from: src.baseAddress!, count: real)
                }
            }
        }
        let span = frameLength + 1
        let frames = max(0, (wave.count - span) / hopLength + 1)

        var features = [Float](repeating: 0, count: frames * melBins)
        var mask = [Bool](repeating: false, count: frames)
        guard frames > 0, let setup = vDSP_create_fftsetupD(vDSP_Length(fftLog2), FFTRadix(kFFTRadix2))
        else { return LogMel(features: features, mask: mask, frames: frames) }
        defer { vDSP_destroy_fftsetupD(setup) }

        let half = fftLength / 2
        var windowed = [Double](repeating: 0, count: fftLength)
        var evenPart = [Double](repeating: 0, count: half)
        var oddPart = [Double](repeating: 0, count: half)
        var magnitude = [Double](repeating: 0, count: spectrumBins)
        var bins = [Double](repeating: 0, count: melBins)

        let window = AudioMelTables.hannWindow
        let firstColumn = AudioMelTables.filterFirstColumn
        let weight0 = AudioMelTables.filterValue0
        let weight1 = AudioMelTables.filterValue1

        for frame in 0..<frames {
            let base = frame * hopLength
            for i in 0..<frameLength { windowed[i] = Double(wave[base + i] * window[i]) }

            windowed.withUnsafeBufferPointer { buffer in
                buffer.baseAddress!.withMemoryRebound(to: DSPDoubleComplex.self, capacity: half) { packed in
                    evenPart.withUnsafeMutableBufferPointer { re in
                        oddPart.withUnsafeMutableBufferPointer { im in
                            var split = DSPDoubleSplitComplex(realp: re.baseAddress!, imagp: im.baseAddress!)
                            vDSP_ctozD(packed, 2, &split, 1, vDSP_Length(half))
                            vDSP_fft_zripD(
                                setup, &split, 1, vDSP_Length(fftLog2), FFTDirection(FFT_FORWARD))
                        }
                    }
                }
            }

            magnitude[0] = abs(evenPart[0]) * 0.5
            magnitude[half] = abs(oddPart[0]) * 0.5
            for k in 1..<half { magnitude[k] = hypot(evenPart[k], oddPart[k]) * 0.5 }

            for j in 0..<melBins { bins[j] = 0 }
            for k in 0..<spectrumBins {
                let column = firstColumn[k]
                if column < 0 { continue }
                let m = magnitude[k]
                bins[column] += m * weight0[k]
                if column + 1 < melBins { bins[column + 1] += m * weight1[k] }
            }

            let voiced = base + frameLength < padLeft + real
            mask[frame] = voiced
            let keep: Float = voiced ? 1 : 0
            let row = frame * melBins
            for j in 0..<melBins { features[row + j] = Float(log(bins[j] + melFloor)) * keep }
        }
        return LogMel(features: features, mask: mask, frames: frames)
    }

    static func fixedInputs(samples: [Float]) throws -> (mel: MLMultiArray, mask: MLMultiArray) {
        let computed = logMel(samples: samples)
        let mel = try MLMultiArray(
            shape: [1, NSNumber(value: fixedFrames), NSNumber(value: melBins)], dataType: .float16)
        let mask = try MLMultiArray(
            shape: [1, NSNumber(value: fixedFrames)], dataType: .float16)
        let rows = min(computed.frames, fixedFrames)
        mel.withF16 { buffer in
            let dst = buffer.baseAddress!
            for k in 0..<(fixedFrames * melBins) { dst[k] = 0 }
            for r in 0..<rows {
                let src = r * melBins
                for j in 0..<melBins { dst[r * melBins + j] = Float16(computed.features[src + j]) }
            }
        }
        mask.withF16 { buffer in
            let dst = buffer.baseAddress!
            for k in 0..<fixedFrames { dst[k] = 0 }
            for r in 0..<rows { dst[r] = computed.mask[r] ? 1 : 0 }
        }
        return (mel, mask)
    }

    static func samples(fromAudioFileAt url: URL) throws -> [Float] {
        let path = url.path(percentEncoded: false)
        guard let file = try? AVAudioFile(forReading: url) else {
            throw PreprocessError.cannotDecodeAudio(path)
        }
        guard let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
            channels: 1, interleaved: false) else {
            throw PreprocessError.cannotMakeConverter(path)
        }
        let source = file.processingFormat
        let length = Int(file.length)
        guard length > 0 else { throw PreprocessError.emptyAudio }

        if source.sampleRate == target.sampleRate, source.channelCount == 1,
           source.commonFormat == .pcmFormatFloat32 {
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: source, frameCapacity: AVAudioFrameCount(length)) else {
                throw PreprocessError.cannotDecodeAudio(path)
            }
            try file.read(into: buffer)
            return floats(from: buffer)
        }

        guard let converter = AVAudioConverter(from: source, to: target) else {
            throw PreprocessError.cannotMakeConverter(path)
        }
        let ratio = target.sampleRate / source.sampleRate
        let capacity = AVAudioFrameCount(Double(length) * ratio) + AVAudioFrameCount(sampleRate)
        guard let input = AVAudioPCMBuffer(
                pcmFormat: source, frameCapacity: AVAudioFrameCount(length)),
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
            throw PreprocessError.cannotMakeConverter(path)
        }
        try file.read(into: input)
        let feed = SingleBufferFeed(input)
        var conversionError: NSError?
        converter.convert(to: output, error: &conversionError) { _, status in feed.next(status) }
        if let conversionError { throw conversionError }
        return floats(from: output)
    }

    final class SingleBufferFeed: @unchecked Sendable {
        private var buffer: AVAudioPCMBuffer?

        init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }

        func next(_ status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioPCMBuffer? {
            guard let pending = buffer else {
                status.pointee = .endOfStream
                return nil
            }
            buffer = nil
            status.pointee = .haveData
            return pending
        }
    }

    static func floats(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channels = buffer.floatChannelData, buffer.frameLength > 0 else { return [] }
        let count = Int(buffer.frameLength)
        return [Float](UnsafeBufferPointer(start: channels[0], count: count))
    }
}
