import Foundation

public enum SpeculationPolicy {

    public static let lowMemoryThresholdBytes: UInt64 = 7 * 1024 * 1024 * 1024

    public static func defaultEnabled(physicalMemoryBytes: UInt64) -> Bool {
        physicalMemoryBytes >= lowMemoryThresholdBytes
    }

    public static func defaultEnabled(processInfo: ProcessInfo = .processInfo) -> Bool {
        defaultEnabled(physicalMemoryBytes: processInfo.physicalMemory)
    }
}
