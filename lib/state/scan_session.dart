import 'package:flutter/foundation.dart';

import '../core/models/handwriting_method.dart';
import '../core/models/scan_page.dart';
import '../services/document_scanner_service.dart';
import '../services/image_processing_service.dart';

enum ScanStep { capture, processing, review, export }

/// Owns the whole scan flow's state: pages, which step we're on, in-flight processing, and
/// the handwriting mask for the page currently under review.
class ScanSession extends ChangeNotifier {
  ScanSession({
    DocumentScannerService? scannerService,
    ImageProcessingService? imageProcessingService,
  })  : _scannerService = scannerService ?? DocumentScannerService(),
        _imageProcessingService = imageProcessingService ?? ImageProcessingService();

  final DocumentScannerService _scannerService;
  final ImageProcessingService _imageProcessingService;

  /// Which detector builds the mask, and how aggressively (0-1, higher = erase more).
  HandwritingMethod method = HandwritingMethod.segmentation;
  double sensitivity = 0.6;

  final List<ScanPage> _pages = [];
  List<ScanPage> get pages => List.unmodifiable(_pages);

  int currentPageIndex = 0;
  ScanPage? get currentPage => _pages.isEmpty ? null : _pages[currentPageIndex];

  ScanStep step = ScanStep.capture;
  bool isProcessing = false;
  String? lastError;

  Future<bool> capture() async {
    final paths = await _guarded(() => _scannerService.scan());
    if (paths == null || paths.isEmpty) return false;
    _pages.addAll(paths.map((p) => ScanPage(originalPath: p)));
    currentPageIndex = _pages.length - paths.length;
    step = ScanStep.processing;
    notifyListeners();
    return true;
  }

  /// Runs sharpen then shadow-removal (chained) on the current page.
  Future<void> processCurrentPage() async {
    final page = currentPage;
    if (page == null) return;
    final result = await _guarded(() async {
      final sharpened = await _imageProcessingService.sharpen(page.originalPath);
      final shadowRemoved = await _imageProcessingService.removeShadow(sharpened);
      return (sharpened: sharpened, shadowRemoved: shadowRemoved);
    });
    if (result == null) return;
    // A fresh ScanPage, not copyWith: any mask/erase result was computed from the previous
    // processed image and no longer lines up with this one.
    _pages[currentPageIndex] = ScanPage(
      originalPath: page.originalPath,
      sharpenedPath: result.sharpened,
      shadowRemovedPath: result.shadowRemoved,
    );
    notifyListeners();
  }

  /// Rotates the page 90° (clockwise if [clockwise], else counter-clockwise) and re-runs
  /// sharpen+shadow-removal from the newly-rotated original. Document scanners crop the page
  /// rectangle correctly but don't know which edge is "up" for reading, so this is the manual
  /// fix for a scan that came out sideways/upside-down.
  Future<void> rotateCurrentPage({bool clockwise = true}) async {
    final page = currentPage;
    if (page == null) return;
    final rotatedPath = await _guarded(
      () => _imageProcessingService.rotate(page.originalPath, quarterTurnsClockwise: clockwise ? 1 : 3),
    );
    if (rotatedPath == null) return;
    _pages[currentPageIndex] = ScanPage(originalPath: rotatedPath);
    notifyListeners();
    await processCurrentPage();
  }

  /// Builds the handwriting mask for the current page with [method] at [sensitivity]. Always
  /// runs on the processed page from before any erase, and discards manual brush edits and any
  /// previous erase result.
  Future<void> detectHandwriting() async {
    final page = currentPage;
    if (page == null) return;
    final result = await _guarded(
      () => _imageProcessingService.detectHandwritingMask(
        page.cleanPath,
        method: method,
        sensitivity: sensitivity,
      ),
    );
    if (result == null) return;
    _pages[currentPageIndex] = ScanPage(
      originalPath: page.originalPath,
      sharpenedPath: page.sharpenedPath,
      shadowRemovedPath: page.shadowRemovedPath,
      maskPath: result.maskPath,
      maskCoverage: result.coverage,
    );
    step = ScanStep.review;
    notifyListeners();
  }

  /// Switches detector (and/or sensitivity) and re-detects the current page.
  Future<void> updateDetection({HandwritingMethod? method, double? sensitivity}) async {
    if (method != null) this.method = method;
    if (sensitivity != null) this.sensitivity = sensitivity;
    await detectHandwriting();
  }

  /// Manual correction of the current mask with brush [strokes].
  Future<void> applyMaskStrokes(List<MaskStroke> strokes) async {
    final page = currentPage;
    final maskPath = page?.maskPath;
    if (page == null || maskPath == null || strokes.isEmpty) return;
    final result = await _guarded(() => _imageProcessingService.applyMaskStrokes(maskPath, strokes));
    if (result == null) return;
    _pages[currentPageIndex] = page.copyWith(maskPath: result.maskPath, maskCoverage: result.coverage);
    notifyListeners();
  }

  Future<void> eraseHandwriting() async {
    final page = currentPage;
    final maskPath = page?.maskPath;
    if (page == null || maskPath == null) return;
    final result = await _guarded(() => _imageProcessingService.eraseWithMask(page.cleanPath, maskPath));
    if (result == null) return;
    _pages[currentPageIndex] = page.copyWith(finalPath: result);
    step = ScanStep.export;
    notifyListeners();
  }

  void selectPage(int index) {
    if (index < 0 || index >= _pages.length) return;
    currentPageIndex = index;
    notifyListeners();
  }

  void reset() {
    _pages.clear();
    currentPageIndex = 0;
    step = ScanStep.capture;
    lastError = null;
    notifyListeners();
  }

  Future<T?> _guarded<T>(Future<T> Function() action) async {
    isProcessing = true;
    lastError = null;
    notifyListeners();
    try {
      return await action();
    } catch (e) {
      lastError = e.toString();
      return null;
    } finally {
      isProcessing = false;
      notifyListeners();
    }
  }
}
