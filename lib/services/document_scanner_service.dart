import 'dart:io';

import 'package:cunning_document_scanner/cunning_document_scanner.dart';
import 'package:device_info_plus/device_info_plus.dart';

/// Wraps the native document-scanner plugin (Android ML Kit GmsDocumentScanner /
/// iOS VisionKit VNDocumentCameraViewController) — auto edge-detect, crop and perspective
/// correction happen natively, for free, before this app ever sees an image.
class DocumentScannerService {
  bool? _isPhysicalDevice;

  /// Returns cropped page image paths, or an empty list if the user cancelled.
  ///
  /// Always uses [ScannerSource.camera] on a real device — the app is a document scanner,
  /// so a camera is the whole point. On the iOS Simulator (no camera hardware) this falls
  /// back to [ScannerSource.gallery] so the rest of the pipeline stays testable there.
  ///
  /// Deliberately NOT [ScannerSource.cameraAndGallery]: that mode's "Camera or Gallery?"
  /// action sheet crashes on iOS due to a bug in cunning_document_scanner 3.0.2 (it sets
  /// `alertController.presentationController?.delegate` before the alert is presented,
  /// which trips a UIKit assertion — SIGSEGV via an uncaught NSInternalInconsistencyException
  /// in `-[_UIAlertControllerPresentationController setDelegate:]`). Confirmed via a real
  /// crash log, not a guess. Revisit if a newer plugin version fixes this upstream.
  Future<List<String>> scan({int maxPages = 10}) async {
    final useGallery = Platform.isIOS && !(await _isRunningOnPhysicalDevice());
    final paths = await CunningDocumentScanner.getPictures(
      noOfPages: maxPages,
      scannerSource: useGallery ? ScannerSource.gallery : ScannerSource.camera,
      androidScannerMode: AndroidScannerMode.full,
    );
    return paths ?? const [];
  }

  Future<bool> _isRunningOnPhysicalDevice() async {
    if (_isPhysicalDevice != null) return _isPhysicalDevice!;
    final info = await DeviceInfoPlugin().iosInfo;
    _isPhysicalDevice = info.isPhysicalDevice;
    return _isPhysicalDevice!;
  }
}
