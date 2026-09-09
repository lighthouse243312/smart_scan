import Flutter
import Foundation

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

            case "detectHandwritingRegions":
                let imagePath = try requireString(args, "imagePath")
                let textBlocks = (args["textBlocks"] as? [[String: Any]]) ?? []
                return try ImageProcessingOpenCV.detectHandwritingRegions(atPath: imagePath, textBlocks: textBlocks)

            case "classifyHandwriting":
                let imagePath = try requireString(args, "imagePath")
                let textBlocks = (args["textBlocks"] as? [[String: Any]]) ?? []
                return try MlHandwritingClassifier.classify(imagePath: imagePath, textBlocks: textBlocks)

            case "eraseRegions":
                let inputPath = try requireString(args, "inputPath")
                let outputPath = try requireString(args, "outputPath")
                let rects = (args["rects"] as? [[String: Any]]) ?? []
                let padding = (args["padding"] as? Double) ?? 6.0
                let inpaintRadius = (args["inpaintRadius"] as? Double) ?? 5.0
                try ImageProcessingOpenCV.eraseRegions(atPath: inputPath, outputPath: outputPath, rects: rects, padding: padding, inpaintRadius: inpaintRadius)
                return ["outputPath": outputPath]

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
