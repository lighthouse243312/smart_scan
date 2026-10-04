import Flutter
import Foundation
import UIKit

/// Thrown for argument problems detected on the Swift side before ever reaching OpenCV.
private struct ArgumentError: Error {
    let message: String
}

/// Bridges Dart to `ImageProcessingOpenCV` (Obj-C++/OpenCV) via a single MethodChannel. Every
/// call is dispatched onto a background queue — never the main thread — since these are
/// multi-hundred-ms to multi-second operations on full-resolution scan images.
enum ImageProcessingChannel {
    static let channelName = "beacon_smart_scan/image_processing"

    static func register(messenger: FlutterBinaryMessenger) {
        let channel = FlutterMethodChannel(name: channelName, binaryMessenger: messenger)
        channel.setMethodCallHandler { call, result in
            DispatchQueue.global(qos: .userInitiated).async {
                let response = handle(call: call)
                DispatchQueue.main.async { result(response) }
            }
        }
    }

    private static func handle(call: FlutterMethodCall) -> Any {
        guard let args = call.arguments as? [String: Any] else {
            return errorResult(code: "INVALID_ARGUMENT", message: "Missing arguments")
        }
        do {
            switch call.method {
            case "sharpen":
                let inputPath = try requireString(args, "inputPath")
                let outputPath = try requireString(args, "outputPath")
                let amount = (args["amount"] as? Double) ?? 1.5
                let radius = (args["radius"] as? Double) ?? 3.0
                try ImageProcessingOpenCV.sharpen(atPath: inputPath, outputPath: outputPath, amount: amount, radius: radius)
                return ["outputPath": outputPath]

            case "straighten":
                let inputPath = try requireString(args, "inputPath")
                let outputPath = try requireString(args, "outputPath")
                let how = try ImageProcessingOpenCV.straighten(atPath: inputPath, outputPath: outputPath)
                return ["outputPath": outputPath, "method": how]

            case "removeShadow":
                let inputPath = try requireString(args, "inputPath")
                let outputPath = try requireString(args, "outputPath")
                try ImageProcessingOpenCV.removeShadow(atPath: inputPath, outputPath: outputPath)
                return ["outputPath": outputPath]

            case "rotate":
                let inputPath = try requireString(args, "inputPath")
                let outputPath = try requireString(args, "outputPath")
                let quarterTurns = (args["quarterTurnsClockwise"] as? Int) ?? 1
                try ImageProcessingOpenCV.rotate(atPath: inputPath, outputPath: outputPath, quarterTurnsClockwise: quarterTurns)
                return ["outputPath": outputPath]

            case "inkColorMask":
                let inputPath = try requireString(args, "inputPath")
                let maskPath = try requireString(args, "maskPath")
                let colorDelta = (args["colorDelta"] as? Double) ?? 0.05
                let coverage = try ImageProcessingOpenCV.inkColorMask(atPath: inputPath, maskPath: maskPath, colorDelta: colorDelta)
                return ["maskPath": maskPath, "coverage": coverage]

            case "segmentationMask":
                let inputPath = try requireString(args, "inputPath")
                let maskPath = try requireString(args, "maskPath")
                let threshold = (args["threshold"] as? Double) ?? 0.5
                let colorDelta = (args["colorDelta"] as? Double) ?? 0.05
                let coverage = try ImageProcessingOpenCV.segmentationMask(atPath: inputPath, maskPath: maskPath, threshold: threshold, colorDelta: colorDelta)
                return ["maskPath": maskPath, "coverage": coverage]

            case "applyMaskStrokes":
                let maskPath = try requireString(args, "maskPath")
                let outputPath = try requireString(args, "outputPath")
                let strokes = (args["strokes"] as? [[String: Any]]) ?? []
                let coverage = try ImageProcessingOpenCV.applyMaskStrokes(atPath: maskPath, outputPath: outputPath, strokes: strokes)
                return ["maskPath": outputPath, "coverage": coverage]

            case "eraseWithMask":
                let inputPath = try requireString(args, "inputPath")
                let analysisPath = try requireString(args, "analysisPath")
                let maskPath = try requireString(args, "maskPath")
                let outputPath = try requireString(args, "outputPath")
                try ImageProcessingOpenCV.eraseWithMask(atPath: inputPath, analysisPath: analysisPath, maskPath: maskPath, outputPath: outputPath)
                return ["outputPath": outputPath]

            case "eraseHandwriting":
                let inputPath = try requireString(args, "inputPath")
                let analysisPath = try requireString(args, "analysisPath")
                let maskPath = try requireString(args, "maskPath")
                let outputPath = try requireString(args, "outputPath")
                let useModel = (args["useModel"] as? Bool) ?? true
                let threshold = (args["threshold"] as? Double) ?? 0.5
                let colorDelta = (args["colorDelta"] as? Double) ?? 0.05
                let passes = try ImageProcessingOpenCV.eraseHandwriting(atPath: inputPath, analysisPath: analysisPath, maskPath: maskPath, outputPath: outputPath, useModel: useModel, threshold: threshold, colorDelta: colorDelta)
                return ["outputPath": outputPath, "passes": passes]

            case "importImage":
                let inputPath = try requireString(args, "inputPath")
                let outputPath = try requireString(args, "outputPath")
                let size = try importImage(inputPath: inputPath, outputPath: outputPath)
                return ["outputPath": outputPath, "width": size.width, "height": size.height]

            default:
                return errorResult(code: "INVALID_ARGUMENT", message: "Unknown method: \(call.method)")
            }
        } catch let error as ArgumentError {
            return errorResult(code: "INVALID_ARGUMENT", message: error.message)
        } catch let error as NSError where error.domain == ImageProcessingErrorDomain {
            return errorResult(code: code(forOpenCvError: error), message: error.localizedDescription)
        } catch {
            return errorResult(code: "PROCESSING_FAILED", message: error.localizedDescription)
        }
    }

    /// Longest side of an imported picture: processing time grows with the pixel count, and this
    /// is well above what text needs.
    private static let maxImportLongSide: CGFloat = 4000

    /// Brings a picture from outside the app (photo library / files) into the pipeline's format.
    /// The document scanner hands over upright JPEGs; an arbitrary picture may instead be HEIC
    /// (which OpenCV cannot read), carry its rotation only as an EXIF orientation, be transparent
    /// (a PNG screenshot) or be huge. Decoded by UIKit, redrawn upright onto white at <= 4000 px
    /// and written as PNG: lossless, because a second JPEG pass smears the ink colours the handwriting
    /// detection separates pen from print by (measured: a pen-crossed word the PNG kept whole came
    /// out broken after re-encoding at quality 0.95).
    private static func importImage(inputPath: String, outputPath: String) throws -> (width: Int, height: Int) {
        guard FileManager.default.fileExists(atPath: inputPath) else {
            throw NSError(domain: ImageProcessingErrorDomain, code: ImageProcessingErrorCode.fileNotFound.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "Không tìm thấy ảnh: \(inputPath)"])
        }
        guard let image = UIImage(contentsOfFile: inputPath), image.size.width > 0, image.size.height > 0 else {
            throw ArgumentError(message: "Không đọc được định dạng ảnh này")
        }
        // pixel size, upright (UIImage.size already accounts for the orientation)
        let pixelW = image.size.width * image.scale, pixelH = image.size.height * image.scale
        let s = min(1, maxImportLongSide / max(pixelW, pixelH))
        let target = CGSize(width: (pixelW * s).rounded(), height: (pixelH * s).rounded())
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let upright = UIGraphicsImageRenderer(size: target, format: format).image { ctx in
            UIColor.white.setFill()
            ctx.fill(CGRect(origin: .zero, size: target))
            image.draw(in: CGRect(origin: .zero, size: target))   // draws with its orientation applied
        }
        guard let data = upright.pngData() else {
            throw ArgumentError(message: "Không ghi được ảnh")
        }
        try data.write(to: URL(fileURLWithPath: outputPath))
        return (Int(target.width), Int(target.height))
    }

    private static func requireString(_ args: [String: Any], _ key: String) throws -> String {
        guard let value = args[key] as? String else {
            throw ArgumentError(message: "Missing argument: \(key)")
        }
        return value
    }

    private static func code(forOpenCvError error: NSError) -> String {
        switch ImageProcessingErrorCode(rawValue: error.code) {
        case .fileNotFound: return "FILE_NOT_FOUND"
        case .invalidArgument: return "INVALID_ARGUMENT"
        default: return "PROCESSING_FAILED"
        }
    }

    private static func errorResult(code: String, message: String) -> FlutterError {
        FlutterError(code: code, message: message, details: nil)
    }
}
