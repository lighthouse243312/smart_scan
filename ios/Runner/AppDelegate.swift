import Flutter
import UIKit
import UniformTypeIdentifiers

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    ImageProcessingChannel.register(messenger: engineBridge.applicationRegistrar.messenger())
    DocumentPickerChannel.register(messenger: engineBridge.applicationRegistrar.messenger())
  }
}

/// The system document picker for a PDF: answers a path to a copy in the app's temp dir, or nil if
/// the user cancelled. Mirrors the `beacon_smart_scan/documents` channel in MainActivity.kt.
final class DocumentPickerChannel: NSObject, UIDocumentPickerDelegate {
  private static let shared = DocumentPickerChannel()
  /// The Dart call waiting for the picker (one at a time).
  private var pending: FlutterResult?

  static func register(messenger: FlutterBinaryMessenger) {
    let channel = FlutterMethodChannel(name: "beacon_smart_scan/documents", binaryMessenger: messenger)
    channel.setMethodCallHandler { call, result in
      guard call.method == "pickPdf" else { return result(FlutterMethodNotImplemented) }
      shared.pickPdf(result: result)
    }
  }

  private func pickPdf(result: @escaping FlutterResult) {
    guard pending == nil else {
      return result(FlutterError(code: "BUSY", message: "Đang mở trình chọn file", details: nil))
    }
    guard let presenter = Self.topViewController() else {
      return result(FlutterError(code: "PROCESSING_FAILED", message: "Không mở được trình chọn file", details: nil))
    }
    pending = result
    let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.pdf], asCopy: true)
    picker.delegate = self
    picker.allowsMultipleSelection = false
    presenter.present(picker, animated: true)
  }

  func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
    guard let result = pending else { return }
    pending = nil
    guard let url = urls.first else { return result(nil) }
    // the picker's copy lives in a folder it may clear: copied next to the app's other temp files
    let target = FileManager.default.temporaryDirectory
      .appendingPathComponent("picked_\(Int(Date().timeIntervalSince1970 * 1000)).pdf")
    do {
      try? FileManager.default.removeItem(at: target)
      try FileManager.default.copyItem(at: url, to: target)
      result(target.path)
    } catch {
      result(FlutterError(code: "FILE_NOT_FOUND", message: "Không mở được file: \(error.localizedDescription)", details: nil))
    }
  }

  func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
    pending?(nil)
    pending = nil
  }

  private static func topViewController() -> UIViewController? {
    let window = UIApplication.shared.connectedScenes
      .compactMap { ($0 as? UIWindowScene)?.windows.first(where: \.isKeyWindow) }
      .first
    var top = window?.rootViewController
    while let presented = top?.presentedViewController { top = presented }
    return top
  }
}
