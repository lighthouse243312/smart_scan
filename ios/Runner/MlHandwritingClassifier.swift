import CoreML
import CoreVideo
import Foundation

/// Runs the trained printed-vs-handwriting CNN (see beacon_smart_scan/ml/) on each word crop.
/// `HandwritingClassifier` is Xcode's auto-generated Core ML wrapper for
/// HandwritingClassifier.mlpackage — it appears once the model file is added to the Runner
/// target's Compile Sources (done via the same xcodeproj-gem technique used for the other
/// native files in this project). Input: 128x64 grayscale, matching ml/train.py's IMG_W/IMG_H —
/// the model's own preprocessing layer applies the /255 normalization, so raw 0-255 pixel bytes
/// go in directly, no manual scaling here.
enum MlHandwritingClassifier {
    private static let inputW = 128
    private static let inputH = 64

    private static let model: HandwritingClassifier? = {
        try? HandwritingClassifier(configuration: MLModelConfiguration())
    }()

    /// [textBlocks] keys: id, left, top, right, bottom (image pixel coordinates).
    static func classify(imagePath: String, textBlocks: [[String: Any]]) throws -> [[String: Any]] {
        guard let model else {
            throw NSError(domain: ImageProcessingErrorDomain, code: ImageProcessingErrorCode.processingFailed.rawValue,
                           userInfo: [NSLocalizedDescriptionKey: "Không tải được model phân loại chữ viết tay"])
        }
        let crops = try ImageProcessingOpenCV.handwritingCrops(atPath: imagePath, textBlocks: textBlocks)

        var results: [[String: Any]] = []
        results.reserveCapacity(textBlocks.count)
        for (block, cropData) in zip(textBlocks, crops) {
            let id = (block["id"] as? String) ?? ""
            let confidence = try classifyOne(model: model, cropData: cropData)
            results.append(["id": id, "mlConfidence": confidence])
        }
        return results
    }

    private static func classifyOne(model: HandwritingClassifier, cropData: Data) throws -> Double {
        guard let pixelBuffer = makePixelBuffer(from: cropData) else {
            return 0.0
        }
        let output = try model.prediction(input: pixelBuffer)
        return Double(truncating: output.Identity[0])
    }

    private static func makePixelBuffer(from grayscaleBytes: Data) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, inputW, inputH,
            kCVPixelFormatType_OneComponent8,
            nil, &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        grayscaleBytes.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            if bytesPerRow == inputW {
                memcpy(base, srcBase, inputW * inputH)
            } else {
                for row in 0..<inputH {
                    memcpy(base.advanced(by: row * bytesPerRow), srcBase.advanced(by: row * inputW), inputW)
                }
            }
        }
        return buffer
    }
}
