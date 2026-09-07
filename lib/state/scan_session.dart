import 'dart:ui';

import 'package:flutter/foundation.dart';

import '../core/models/scan_page.dart';
import '../core/models/text_region.dart';
import '../services/document_scanner_service.dart';
import '../services/image_processing_service.dart';
import '../services/text_recognition_service.dart';

enum ScanStep { capture, processing, review, export }

/// Owns the whole scan flow's state: pages, which step we're on, in-flight processing, and
/// the handwriting-region selection for the page currently under review.
class ScanSession extends ChangeNotifier {
  ScanSession({
    DocumentScannerService? scannerService,
    ImageProcessingService? imageProcessingService,
    TextRecognitionService? textRecognitionService,
  })  : _scannerService = scannerService ?? DocumentScannerService(),
        _imageProcessingService = imageProcessingService ?? ImageProcessingService(),
        _textRecognitionService = textRecognitionService ?? TextRecognitionService();

  final DocumentScannerService _scannerService;
  final ImageProcessingService _imageProcessingService;
  final TextRecognitionService _textRecognitionService;

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
    _pages[currentPageIndex] = page.copyWith(
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

  /// Detects text regions then scores each for handwriting likelihood.
  Future<void> detectHandwriting() async {
    final page = currentPage;
    if (page == null) return;
    final result = await _guarded(() async {
      final regions = await _textRecognitionService.recognize(page.displayPath);
      return _imageProcessingService.scoreHandwriting(page.displayPath, regions);
    });
    if (result == null) return;
    _pages[currentPageIndex] = page.copyWith(textRegions: result);
    step = ScanStep.review;
    notifyListeners();
  }

  void toggleRegionSelection(String regionId) {
    final page = currentPage;
    if (page == null) return;
    final updated = page.textRegions
        .map((r) => r.id == regionId ? r.copyWith(selectedForErase: !r.selectedForErase) : r)
        .toList();
    _pages[currentPageIndex] = page.copyWith(textRegions: updated);
    notifyListeners();
  }

  void addManualRegion(Rect rect) {
    final page = currentPage;
    if (page == null) return;
    final region = TextRegion(
      id: 'manual_${DateTime.now().microsecondsSinceEpoch}',
      boundingBox: rect,
      text: '',
      isManual: true,
      selectedForErase: true,
    );
    _pages[currentPageIndex] = page.copyWith(textRegions: [...page.textRegions, region]);
    notifyListeners();
  }

  Future<void> eraseSelected() async {
    final page = currentPage;
    if (page == null) return;
    final rects = page.textRegions.where((r) => r.selectedForErase).map((r) => r.boundingBox).toList();
    if (rects.isEmpty) return;
    final result = await _guarded(() => _imageProcessingService.eraseRegions(page.displayPath, rects));
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

  @override
  void dispose() {
    _textRecognitionService.dispose();
    super.dispose();
  }
}
