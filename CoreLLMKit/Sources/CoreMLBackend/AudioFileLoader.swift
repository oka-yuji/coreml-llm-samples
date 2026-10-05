import Foundation

public enum AudioFileLoader {
    public static var sampleRate: Int { AudioPreprocess.sampleRate }

    public static func monoSamples(at url: URL) throws -> [Float] {
        try AudioPreprocess.samples(fromAudioFileAt: url)
    }

    public static func seconds(sampleCount: Int) -> Double {
        Double(sampleCount) / Double(AudioPreprocess.sampleRate)
    }
}
