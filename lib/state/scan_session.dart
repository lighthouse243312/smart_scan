import 'package:flutter/foundation.dart';

import '../core/models/handwriting_method.dart';
import '../core/models/saved_document.dart';
import '../core/models/scan_page.dart';
import '../services/document_scanner_service.dart';
import '../services/image_picker_service.dart';
import '../services/image_processing_service.dart';
import 'document_library.dart';

enum ScanStep { capture, processing, review, export }

/// Owns the whole scan flow's state: pages, which step we're on, in-flight processing, and
/// the handwriting mask for the page currently under review.
class ScanSession extends ChangeNotifier {
  ScanSession({
    this.library,
    DocumentScannerService? scannerService,
    ImagePickerService? imagePickerService,
    ImageProcessingService? imageProcessingService,
  })  : _scannerService = scannerService ?? DocumentScannerService(),
        _imagePickerService = imagePickerService ?? ImagePickerService(),
        _imageProcessingService = imageProcessingService ?? ImageProcessingService();

  /// Where captures are archived as they are taken (null: no archive, e.g. in tests).
  final DocumentLibrary? library;

  /// The saved document this session's pages belong to.
  String? documentId;

  final DocumentScannerService _scannerService;
  final ImagePickerService _imagePickerService;
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

  /// While detection / erase runs over several pages: the 1-based page being worked on, and of how
  /// many (null otherwise).
  ({int current, int total})? batchProgress;

  /// What a mask depends on besides its page: pages whose mask was detected with other settings
  /// are detected again before they are shown or erased.
  String get detectionSettings => '${method.name}:$sensitivity';

  bool _maskStale(ScanPage p) => p.maskPath == null || p.maskSettings != detectionSettings;

  Future<bool> capture() async {
    final paths = await _guarded(() => _scannerService.scan());
    if (paths == null || paths.isEmpty) return false;
    await _addPages(paths);
    currentPageIndex = _pages.length - paths.length;
    step = ScanStep.processing;
    notifyListeners();
    return true;
  }

  /// Adds pictures from outside the app (photo library): each is converted to an upright JPEG
  /// first, since the rest of the pipeline assumes what the document scanner produces.
  Future<bool> importImages() async {
    final paths = await _guarded(() async {
      final picked = await _imagePickerService.pickImages();
      final imported = <String>[];
      for (final p in picked) {
        // a photo from the library is rarely square to the page (the scanner path is)
        final upright = await _imageProcessingService.importImage(p);
        imported.add(await _imageProcessingService.straighten(upright));
      }
      return imported;
    });
    if (paths == null || paths.isEmpty) return false;
    await _addPages(paths);
    currentPageIndex = _pages.length - paths.length;
    step = ScanStep.processing;
    notifyListeners();
    return true;
  }

  /// Adds the pages of a PDF picked from the device's files: each rendered to an image natively,
  /// then levelled like a library photo (a scanned PDF is often slightly skewed; a level page is
  /// copied unchanged).
  Future<bool> importPdf() async {
    final paths = await _guarded(() async {
      final pdf = await _imagePickerService.pickPdf();
      if (pdf == null) return const <String>[];
      final rendered = await _imageProcessingService.renderPdf(pdf);
      return [for (final p in rendered) await _imageProcessingService.straighten(p)];
    });
    if (paths == null || paths.isEmpty) return false;
    await _addPages(paths);
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
      storedPageId: page.storedPageId,
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
    _pages[currentPageIndex] = ScanPage(originalPath: rotatedPath, storedPageId: page.storedPageId);
    notifyListeners();
    await processCurrentPage();
  }

  /// Builds the handwriting mask of every page that has none yet (or one detected with other
  /// settings) with [method] at [sensitivity], page after page; pages that still need it are
  /// sharpened / de-shadowed first. A re-detected page loses its brush edits and erase result.
  /// Detection reads the page as it came out of the scanner: sharpening adds bright halos and
  /// shadow removal shifts the ink colour, both of which the colour analysis relies on not
  /// happening. Pixel geometry is the same, so the mask still lines up with the processed page it
  /// is erased from.
  Future<void> detectHandwriting() async {
    if (_pages.isEmpty) return;
    lastError = null;
    await processRemainingPages();
    if (lastError != null) return;
    final pending = [for (var i = 0; i < _pages.length; i++) if (_maskStale(_pages[i])) i];
    final ok = await _guarded(() async {
      for (var n = 0; n < pending.length; n++) {
        batchProgress = (current: n + 1, total: pending.length);
        notifyListeners();
        await _detectPage(pending[n]);
      }
      return true;
    });
    if (ok == null) return;
    step = ScanStep.review;
    notifyListeners();
  }

  Future<void> _detectPage(int index) async {
    final page = _pages[index];
    final result = await _imageProcessingService.detectHandwritingMask(
      page.originalPath,
      method: method,
      sensitivity: sensitivity,
    );
    _pages[index] = ScanPage(
      originalPath: page.originalPath,
      storedPageId: page.storedPageId,
      sharpenedPath: page.sharpenedPath,
      shadowRemovedPath: page.shadowRemovedPath,
      maskPath: result.maskPath,
      maskCoverage: result.coverage,
      maskSettings: detectionSettings,
    );
    notifyListeners();
  }

  /// Switches detector (and/or sensitivity) and re-detects the current page; the other pages are
  /// re-detected when shown ([selectPage]) or erased ([eraseHandwriting]).
  Future<void> updateDetection({HandwritingMethod? method, double? sensitivity}) async {
    if (method != null) this.method = method;
    if (sensitivity != null) this.sensitivity = sensitivity;
    await _guarded(() => _detectPage(currentPageIndex));
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

  /// Erases the handwriting of every page that has some, one page after the other (each erase
  /// already keeps the device's cores busy). A page already erased with its current mask is not
  /// erased again; a page whose mask is stale is re-detected first. Each erase runs in passes on
  /// the native side: later passes catch the pen pieces one pass leaves behind, kept off the print
  /// this (reviewed) mask found — see [ImageProcessingService.eraseHandwriting].
  Future<void> eraseHandwriting() async {
    lastError = null;
    await processRemainingPages();
    if (lastError != null) return;
    final stale = [for (var i = 0; i < _pages.length; i++) if (_maskStale(_pages[i])) i];
    final ok = await _guarded(() async {
      for (final i in stale) {
        await _detectPage(i);
      }
      final pending = [
        for (var i = 0; i < _pages.length; i++)
          if (_pages[i].maskCoverage > 0 && _pages[i].finalMaskPath != _pages[i].maskPath) i,
      ];
      for (var n = 0; n < pending.length; n++) {
        batchProgress = (current: n + 1, total: pending.length);
        notifyListeners();
        final i = pending[n];
        final page = _pages[i];
        final maskPath = page.maskPath!;
        final result = await _imageProcessingService.eraseHandwriting(
          page.cleanPath,
          analysisPath: page.originalPath,
          maskPath: maskPath,
          method: method,
          sensitivity: sensitivity,
        );
        _pages[i] = page.copyWith(finalPath: result, finalMaskPath: maskPath);
        notifyListeners();
      }
      return true;
    });
    if (ok == null) return;
    step = ScanStep.export;
    notifyListeners();
  }

  /// Whether any page has handwriting to erase.
  bool get hasHandwriting => _pages.any((p) => p.maskCoverage > 0 || _maskStale(p));

  void selectPage(int index) {
    if (index < 0 || index >= _pages.length) return;
    currentPageIndex = index;
    notifyListeners();
    // in review, a page shows its mask for the current settings
    if (step == ScanStep.review && _pages[index].shadowRemovedPath != null && _maskStale(_pages[index]) && !isProcessing) {
      _guarded(() => _detectPage(index));
    }
  }

  /// Drops page [index] (from the saved document too); with no page left the session goes back
  /// to capture.
  void removePage(int index) {
    if (index < 0 || index >= _pages.length) return;
    final removed = _pages.removeAt(index);
    final docId = documentId, pageId = removed.storedPageId;
    if (library != null && docId != null && pageId != null) library!.removePage(docId, pageId);
    if (_pages.isEmpty) {
      reset();
      return;
    }
    currentPageIndex = currentPageIndex.clamp(0, _pages.length - 1);
    notifyListeners();
  }

  /// Drops every page at once, deleting the saved document as a whole (not page by page, which
  /// would race on the document's metadata), then goes back to capture.
  void removeAllPages() {
    final docId = documentId;
    if (library != null && docId != null) library!.delete(docId);
    reset();
  }

  /// Sharpens + removes shadows on every page that has not been processed yet, so an export
  /// never ships a raw capture next to cleaned pages.
  Future<void> processRemainingPages() async {
    final pending = [for (var i = 0; i < _pages.length; i++) if (_pages[i].shadowRemovedPath == null) i];
    if (pending.isEmpty) return;
    await _guarded(() async {
      for (final i in pending) {
        final page = _pages[i];
        final sharpened = await _imageProcessingService.sharpen(page.originalPath);
        final shadowRemoved = await _imageProcessingService.removeShadow(sharpened);
        _pages[i] = ScanPage(
          originalPath: page.originalPath,
          storedPageId: page.storedPageId,
          sharpenedPath: sharpened,
          shadowRemovedPath: shadowRemoved,
        );
        notifyListeners();
      }
    });
  }

  /// Moves to the export step (with or without the handwriting erase).
  void goToExport() {
    step = ScanStep.export;
    notifyListeners();
  }

  void reset() {
    _pages.clear();
    documentId = null;
    currentPageIndex = 0;
    step = ScanStep.capture;
    lastError = null;
    notifyListeners();
  }

  /// New pages from the scanner / library. Their originals are archived untouched right away:
  /// a new saved document for the session's first pages, the same document after that.
  Future<void> _addPages(List<String> paths) async {
    var stored = <SavedPage>[];
    final lib = library;
    if (lib != null) {
      try {
        if (documentId == null) {
          final doc = await lib.create(paths, title: _defaultTitle());
          documentId = doc.id;
          stored = doc.pages;
        } else {
          stored = await lib.addPages(documentId!, paths);
        }
      } catch (_) {
        // archiving must never block scanning; the session works on the capture files
      }
    }
    _pages.addAll([
      for (var i = 0; i < paths.length; i++)
        ScanPage(originalPath: paths[i], storedPageId: i < stored.length ? stored[i].id : null),
    ]);
  }

  String _defaultTitle() {
    final n = DateTime.now();
    String two(int v) => v.toString().padLeft(2, '0');
    return 'Scan ${n.year}-${two(n.month)}-${two(n.day)} ${two(n.hour)}:${two(n.minute)}';
  }

  /// Opens a saved document for editing again, from its untouched originals.
  void openDocument(SavedDocument doc) {
    _pages
      ..clear()
      ..addAll(doc.pages.map((p) => ScanPage(originalPath: p.originalPath, storedPageId: p.id)));
    documentId = doc.id;
    currentPageIndex = 0;
    step = ScanStep.processing;
    lastError = null;
    notifyListeners();
  }

  /// Keeps every page's current processed result in the saved document, next to its original.
  /// Returns the number of pages saved.
  Future<int> saveProcessedToLibrary() async {
    final lib = library;
    if (lib == null) return 0;
    if (documentId == null) {
      final doc = await lib.create([for (final p in _pages) p.originalPath], title: _defaultTitle());
      documentId = doc.id;
      for (var i = 0; i < _pages.length && i < doc.pages.length; i++) {
        _pages[i] = _pages[i].copyWith(storedPageId: doc.pages[i].id);
      }
    }
    final edits = <String, String>{
      for (final p in _pages)
        if (p.storedPageId != null && p.displayPath != p.originalPath) p.storedPageId!: p.displayPath,
    };
    await lib.saveEdited(documentId!, edits);
    return edits.length;
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
      batchProgress = null;
      notifyListeners();
    }
  }
}
