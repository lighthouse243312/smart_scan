import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';

import '../../core/models/handwriting_method.dart';
import '../../core/theme/app_theme.dart';
import '../../l10n/app_localizations.dart';
import '../../services/export_service.dart';
import '../../state/scan_session.dart';
import '../../widgets/page_strip.dart';
import '../export/export_screen.dart';

enum _BrushMode { none, add, keep }

/// Shows the detected handwriting mask (red overlay) on the page, lets the user switch detector /
/// sensitivity, correct the mask with a brush, and then erase it.
class HandwritingReviewScreen extends StatefulWidget {
  const HandwritingReviewScreen({super.key});

  @override
  State<HandwritingReviewScreen> createState() => _HandwritingReviewScreenState();
}

class _HandwritingReviewScreenState extends State<HandwritingReviewScreen> {
  /// Brush diameter in screen (logical) pixels — converted to image pixels per stroke, so it
  /// feels the same whatever the photo's resolution or zoom.
  static const double _brushScreenWidth = 24;

  _BrushMode _brushMode = _BrushMode.none;
  bool _showMask = true;
  bool _savingOriginalImage = false;
  double? _pendingSensitivity;

  /// In-progress stroke, in image pixel coordinates.
  List<Offset> _liveStroke = const [];

  // Decoded once per source path (not on every rebuild) — only its pixel size is needed, to map
  // touches onto image coordinates.
  String? _decodedPath;
  Future<ui.Image>? _decodeFuture;

  Future<ui.Image> _decodeImage(String path) async {
    final bytes = await File(path).readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  @override
  Widget build(BuildContext context) {
    final session = context.watch<ScanSession>();
    final page = session.currentPage;
    if (page != null && page.cleanPath != _decodedPath) {
      _decodedPath = page.cleanPath;
      _decodeFuture = _decodeImage(page.cleanPath);
    }

    final l = AppLocalizations.of(context);
    // closing a later step ends the scan: a new capture starts a new document
    return PopScope(
      onPopInvokedWithResult: (didPop, _) {
        if (didPop) session.reset();
      },
      child: Scaffold(
        appBar: AppBar(
          // steps only go forward: leaving here closes the scan flow, it does not reopen processing
          leading: const CloseButton(),
          title: Text(
            session.pages.length > 1
                ? '${l.handwritingTitle} · ${l.pageOf(session.currentPageIndex + 1, session.pages.length)}'
                : l.handwritingTitle,
          ),
          actions: [
            IconButton(
              tooltip: _showMask ? l.hideMask : l.showMask,
              icon: Icon(_showMask ? Icons.visibility : Icons.visibility_off),
              onPressed: () => setState(() => _showMask = !_showMask),
            ),
            // Saves the RAW capture straight out of the document scanner — before sharpen,
            // shadow-removal, or erase touch it — so a bug report can tell "the detector read this
            // wrong" apart from "the photo itself was already blurry/warped/cropped oddly".
            if (kDebugMode && page != null)
              IconButton(
                tooltip: l.saveOriginal,
                icon: _savingOriginalImage
                    ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.image_outlined),
                onPressed: _savingOriginalImage ? null : () => _saveOriginalImageToGallery(context, page.originalPath),
              ),
          ],
        ),
        body: page == null
            ? const SizedBox.shrink()
            : Column(
                children: [
                  const SizedBox(height: 8),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: SegmentedButton<HandwritingMethod>(
                      segments: [
                        ButtonSegment(
                          value: HandwritingMethod.inkColor,
                          label: Text(l.methodInk),
                          icon: const Icon(Icons.palette_outlined),
                        ),
                        ButtonSegment(
                          value: HandwritingMethod.segmentation,
                          label: Text(l.methodAi),
                          icon: const Icon(Icons.auto_awesome_outlined),
                        ),
                      ],
                      selected: {session.method},
                      onSelectionChanged: session.isProcessing ? null : (s) => session.updateDetection(method: s.first),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Row(
                      children: [
                        Text(l.sensitivity, style: const TextStyle(fontSize: 13)),
                        Expanded(
                          child: Slider(
                            value: _pendingSensitivity ?? session.sensitivity,
                            onChanged: session.isProcessing ? null : (v) => setState(() => _pendingSensitivity = v),
                            onChangeEnd: (v) async {
                              await session.updateDetection(sensitivity: v);
                              if (mounted) setState(() => _pendingSensitivity = null);
                            },
                          ),
                        ),
                      ],
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Text(
                      _hintText(l, session.method, page.maskCoverage),
                      style: const TextStyle(fontSize: 13, color: Colors.grey),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Expanded(
                    child: FutureBuilder<ui.Image>(
                      future: _decodeFuture,
                      builder: (context, snapshot) {
                        if (!snapshot.hasData) {
                          return const Center(child: CircularProgressIndicator());
                        }
                        return _buildCanvas(context, session, snapshot.data!);
                      },
                    ),
                  ),
                  if (session.lastError != null)
                    Padding(
                      padding: const EdgeInsets.all(8),
                      child: Text(session.lastError!, style: const TextStyle(color: AppColors.danger)),
                    ),
                  if (session.pages.length > 1) ...[
                    const SizedBox(height: 8),
                    PageStrip(
                      pages: session.pages,
                      selected: session.currentPageIndex,
                      onSelect: session.isProcessing ? (_) {} : session.selectPage,
                    ),
                  ],
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    child: SegmentedButton<_BrushMode>(
                      segments: [
                        ButtonSegment(
                          value: _BrushMode.none,
                          label: Text(l.brushView),
                          icon: const Icon(Icons.pan_tool_outlined),
                        ),
                        ButtonSegment(value: _BrushMode.add, label: Text(l.brushAdd), icon: const Icon(Icons.brush)),
                        ButtonSegment(
                          value: _BrushMode.keep,
                          label: Text(l.brushKeep),
                          icon: const Icon(Icons.cleaning_services_outlined),
                        ),
                      ],
                      selected: {_brushMode},
                      onSelectionChanged: (s) => setState(() => _brushMode = s.first),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                    child: Row(
                      children: [
                        Expanded(
                          flex: 2,
                          child: OutlinedButton(
                            onPressed: session.isProcessing ? null : () => _toExport(context, session),
                            child: Text(l.skip),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          flex: 3,
                          child: FilledButton.icon(
                            onPressed: session.isProcessing || !session.hasHandwriting
                                ? null
                                : () async {
                                    await session.eraseHandwriting();
                                    if (context.mounted && session.lastError == null) await _toExport(context, session);
                                  },
                            icon: session.isProcessing
                                ? const SizedBox(
                                    width: 18,
                                    height: 18,
                                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                                  )
                                : const Icon(Icons.auto_fix_high_rounded),
                            label: Text(session.pages.length > 1 ? l.eraseAllPages : l.eraseHandwriting),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
      ),
    );
  }

  /// The other pages are sharpened / de-shadowed first, so the export never mixes raw captures in.
  Future<void> _toExport(BuildContext context, ScanSession session) async {
    await session.processRemainingPages();
    if (!context.mounted || session.lastError != null) return;
    session.goToExport();
    Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => const ExportScreen()));
  }

  String _hintText(AppLocalizations l, HandwritingMethod method, double coverage) {
    if (coverage <= 0) return method == HandwritingMethod.inkColor ? l.hintNoneInk : l.hintNoneAi;
    return l.hintCoverage((coverage * 100).toStringAsFixed(coverage < 0.01 ? 2 : 1));
  }

  Widget _buildCanvas(BuildContext context, ScanSession session, ui.Image image) {
    final page = session.currentPage!;
    final brushing = _brushMode != _BrushMode.none;
    return InteractiveViewer(
      // While brushing, one-finger drags must reach the brush instead of panning the page.
      panEnabled: !brushing,
      scaleEnabled: !brushing,
      minScale: 1.0,
      maxScale: 6.0,
      child: Center(
        child: AspectRatio(
          aspectRatio: image.width / image.height,
          child: LayoutBuilder(
            builder: (context, constraints) {
              // Widget → image pixel scale (the page is drawn to fill this box exactly).
              final toImage = image.width / constraints.maxWidth;
              final brushImageWidth = _brushScreenWidth * toImage;
              return GestureDetector(
                onPanStart: brushing && !session.isProcessing
                    ? (d) => setState(() => _liveStroke = [d.localPosition * toImage])
                    : null,
                onPanUpdate: brushing && !session.isProcessing
                    ? (d) => setState(() => _liveStroke = [..._liveStroke, d.localPosition * toImage])
                    : null,
                onPanEnd: brushing && !session.isProcessing ? (_) => _commitStroke(session, brushImageWidth) : null,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.file(File(page.cleanPath), fit: BoxFit.fill),
                    if (_showMask && page.maskPath != null)
                      Image.file(File(page.maskPath!), fit: BoxFit.fill, gaplessPlayback: true),
                    if (_liveStroke.isNotEmpty)
                      CustomPaint(
                        painter: _LiveStrokePainter(
                          points: _liveStroke,
                          scale: 1 / toImage,
                          width: _brushScreenWidth,
                          erase: _brushMode == _BrushMode.keep,
                        ),
                      ),
                    if (session.isProcessing)
                      ColoredBox(
                        color: const Color(0x66FFFFFF),
                        child: Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const CircularProgressIndicator(),
                              if (session.batchProgress case final b?) ...[
                                const SizedBox(height: 12),
                                Text(
                                  AppLocalizations.of(context).erasingPages(b.current, b.total),
                                  style: Theme.of(context).textTheme.bodySmall,
                                ),
                              ],
                            ],
                          ),
                        ),
                      ),
                  ],
                ),
              );
            },
          ),
        ),
      ),
    );
  }

  Future<void> _commitStroke(ScanSession session, double brushImageWidth) async {
    final points = _liveStroke;
    if (points.isEmpty) return;
    final stroke = MaskStroke(
      points: [
        for (final p in points) ...[p.dx, p.dy],
      ],
      width: brushImageWidth,
      erase: _brushMode == _BrushMode.keep,
    );
    await session.applyMaskStrokes([stroke]);
    if (mounted) setState(() => _liveStroke = const []);
  }

  Future<void> _saveOriginalImageToGallery(BuildContext context, String originalPath) async {
    setState(() => _savingOriginalImage = true);
    try {
      // Copied to a clearly-labeled temp name first — saving originalPath directly would carry
      // the document scanner's own filename into Photos, indistinguishable there from any other
      // pipeline stage once several debug photos pile up in the same album.
      final tempDir = await getTemporaryDirectory();
      final ext = originalPath.split('.').last;
      final outPath = '${tempDir.path}/original_${DateTime.now().millisecondsSinceEpoch}.$ext';
      await File(originalPath).copy(outPath);
      await ExportService().saveToGallery([outPath]);

      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(AppLocalizations.of(context).savedOriginal)));
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(AppLocalizations.of(context).saveFailed('$e'))));
      }
    } finally {
      if (mounted) setState(() => _savingOriginalImage = false);
    }
  }
}

/// Draws the stroke being brushed right now, before the native side has applied it to the mask.
class _LiveStrokePainter extends CustomPainter {
  _LiveStrokePainter({required this.points, required this.scale, required this.width, required this.erase});

  /// Image pixel coordinates; [scale] maps them back onto this widget.
  final List<Offset> points;
  final double scale;
  final double width;
  final bool erase;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = erase ? const Color(0x9942A5F5) : const Color(0x99FF0000)
      ..strokeWidth = width
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..style = PaintingStyle.stroke;
    final path = Path()..moveTo(points.first.dx * scale, points.first.dy * scale);
    for (final p in points.skip(1)) {
      path.lineTo(p.dx * scale, p.dy * scale);
    }
    if (points.length == 1) {
      canvas.drawCircle(points.first * scale, width / 2, paint..style = PaintingStyle.fill);
    } else {
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(_LiveStrokePainter old) => old.points != points || old.erase != erase;
}
