#if os(iOS)
import Foundation
import Vision

/// Reads a single-column "項目：値" slip with Vision, on device.
///
/// The recognised text goes into the text field and stops there — the person reads
/// it, fixes it, and presses マッピング. An OCR slip must never look like a model
/// mistake, which is the same rule the dictation path follows.
enum MapperOCR {
    static func recognize(imageData: Data) throws -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["ja-JP"]
        request.usesLanguageCorrection = true

        let handler = VNImageRequestHandler(data: imageData, options: [:])
        try handler.perform([request])

        // Vision's bounding boxes are bottom-left origin, so descending y is
        // top-to-bottom on the page.
        return (request.results ?? [])
            .sorted { $0.boundingBox.origin.y > $1.boundingBox.origin.y }
            .compactMap { $0.topCandidates(1).first?.string }
            .joined(separator: "\n")
    }
}
#endif
