import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';

import '../../core/models/text_region.dart';
import '../../services/export_service.dart';
import '../../state/scan_session.dart';
import '../../widgets/region_overlay_painter.dart';
import '../../widgets/step_progress_indicator.dart';
import '../export/export_screen.dart';

class HandwritingReviewScreen extends StatefulWidget {
  const HandwritingReviewScreen({super.key});

  @override
  State<HandwritingReviewScreen> createState() => _HandwritingReviewScreenState();
}

class _HandwritingReviewScreenState extends State<HandwritingReviewScreen> {
  bool _drawMode = false;
  bool _savingDebugImage = false;

  // Cached by source path rather than re-decoded on every build (the previous version called
  // _decodeImage directly inside FutureBuilder's `future:`, which re-decodes on every rebuild —
  // e.g. every region toggle). Also keeps the last decoded ui.Image around so the debug-save
  // button (in the AppBar, outside the FutureBuilder's own scope) can use it without decoding a
  // second time.
  String? _decodedPath;
  Future<ui.Image>? _decodeFuture;
  ui.Image? _lastDecodedImage;

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
    if (page != null && page.displayPath != _decodedPath) {
      _decodedPath = page.displayPath;
      _decodeFuture = _decodeImage(page.displayPath);
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Chữ viết tay'),
        actions: [
          // Debug builds only: reading every region's breakdown one long-press at a time doesn't
          // scale past a couple of regions — this dumps ALL of them as plain text onto the
          // clipboard in one tap, ready to paste straight into a message instead of transcribing
          // numbers off a popup or a screenshot by hand.
          if (kDebugMode && page != null)
            IconButton(
              tooltip: 'Copy debug (tất cả vùng)',
              icon: const Icon(Icons.bug_report_outlined),
              onPressed: () => _copyDebugDump(context, page.textRegions),
            ),
          // Saves a copy of this page WITH the debug boxes/labels burned into the actual image
          // pixels (not a screen capture) directly to Photos, so it can be sent from there like
          // any other photo — no in-app popup or clipboard step to find. Also sidesteps the
          // on-screen zoom problem entirely: the labels are drawn once at the photo's own full
          // resolution (thousands of px wide), not the phone screen's much smaller logical size,
          // so they're legible in Photos' own pinch-zoom without needing anything special here.
          if (kDebugMode && page != null)
            IconButton(
              tooltip: 'Lưu ảnh debug vào Ảnh',
              icon: _savingDebugImage
                  ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.save_alt_outlined),
              onPressed: _savingDebugImage
                  ? null
                  : () => _saveDebugImageToGallery(context, page.textRegions),
            ),
          IconButton(
            tooltip: 'Vẽ thêm vùng',
            icon: Icon(_drawMode ? Icons.edit : Icons.edit_outlined,
                color: _drawMode ? Theme.of(context).colorScheme.primary : null),
            onPressed: () => setState(() => _drawMode = !_drawMode),
          ),
        ],
      ),
      body: page == null
          ? const SizedBox.shrink()
          : Column(
              children: [
                StepProgressIndicator(currentStep: session.step),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Text(
                    'Vùng gợi ý được tô sẵn để xoá — chạm để bật/tắt. Bật chế độ vẽ (biểu tượng bút) để tự khoanh thêm vùng.'
                    '${kDebugMode ? ' Giữ (long-press) một ô để xem chi tiết điểm số của riêng ô đó.' : ''}',
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
                      final image = snapshot.data!;
                      _lastDecodedImage = image;
                      return InteractiveViewer(
                        // Pinch-zoom only (panEnabled: false) so a single-finger drag still
                        // reaches the RegionOverlay below for tap-to-toggle and draw-mode —
                        // InteractiveViewer's own pan would otherwise swallow that gesture.
                        // Added because the debug score labels (kDebugMode) pack many small
                        // boxes into a dense grid (e.g. every calendar date individually) that's
                        // unreadable at the screen's fit-to-width size; there was previously no
                        // way to get closer to one label without leaving the app to zoom a
                        // screenshot after the fact, by which point the labels are already a
                        // blurred, illegible photo of the screen.
                        panEnabled: false,
                        minScale: 1.0,
                        maxScale: 6.0,
                        child: Center(
                          child: AspectRatio(
                            aspectRatio: image.width / image.height,
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                Image.file(File(page.displayPath), fit: BoxFit.fill),
                                RegionOverlay(
                                  imageWidth: image.width,
                                  imageHeight: image.height,
                                  regions: page.textRegions,
                                  drawModeEnabled: _drawMode,
                                  onToggle: session.toggleRegionSelection,
                                  onManualRegion: session.addManualRegion,
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
                if (session.lastError != null)
                  Padding(
                    padding: const EdgeInsets.all(8),
                    child: Text(session.lastError!, style: const TextStyle(color: Colors.red)),
                  ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                  child: FilledButton.icon(
                    onPressed: session.isProcessing || !page.textRegions.any((r) => r.selectedForErase)
                        ? null
                        : () async {
                            await session.eraseSelected();
                            if (context.mounted) {
                              Navigator.of(context).push(
                                MaterialPageRoute(builder: (_) => const ExportScreen()),
                              );
                            }
                          },
                    icon: session.isProcessing
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.auto_fix_high),
                    label: const Text('Xoá vùng đã chọn'),
                  ),
                ),
              ],
            ),
    );
  }

  void _copyDebugDump(BuildContext context, List<TextRegion> regions) {
    final buffer = StringBuffer();
    // A whole-page dump (every calendar date included) runs long enough to get silently
    // truncated by the time it's pasted somewhere else — verified: a real dump lost its own
    // first several regions this way, which happened to include the one actually in question.
    // Only the regions worth looking at are the ones flagged for erase, or ones the CNN/heuristic
    // rated meaningfully close to being flagged — a confidently-correct 0.01 print score doesn't
    // need reviewing and just pushes the regions that DO need it further into truncation risk.
    final interesting = regions.where((r) => r.selectedForErase || r.confidence > 0.15).toList()
      // Smallest-area first: a single mis-erased character (a lone digit, a lone weekday
      // letter) is exactly what a small erased box looks like, whereas a genuine multi-word
      // handwriting line is a much bigger box — verified this dump kept getting cut off before
      // reaching the one small region actually in question, buried behind many larger
      // correctly-flagged handwriting boxes earlier in scan order.
      ..sort((a, b) {
        final areaA = a.boundingBox.width * a.boundingBox.height;
        final areaB = b.boundingBox.width * b.boundingBox.height;
        return areaA.compareTo(areaB);
      });
    buffer.writeln('${interesting.length}/${regions.length} regions shown (selectedForErase or confidence > 0.15), smallest-area first');
    for (final region in interesting) {
      final box = region.boundingBox;
      // selectedForErase, not isLikelyHandwriting — the latter is the PRE-neighbor-boost verdict
      // and can disagree with what actually gets erased once _boostSpatialNeighbors/
      // _unionSplitSiblings run, which was confusing to debug against the real erased output.
      final kind = region.isManual ? "manual" : region.selectedForErase ? "ERASE" : "keep";
      buffer.writeln(
        '${region.id} [${box.left.toStringAsFixed(0)},${box.top.toStringAsFixed(0)},'
        '${box.width.toStringAsFixed(0)}x${box.height.toStringAsFixed(0)}] $kind',
      );
      final debug = region.debugBreakdown;
      if (debug != null) {
        buffer.writeln(
          'f:${region.confidence.toStringAsFixed(2)} ml:${debug.mlConfidence.toStringAsFixed(2)} '
          'h:${debug.heuristic.toStringAsFixed(2)} ang:${debug.angleVariationScore.toStringAsFixed(2)} '
          'col:${debug.colorDeviation.toStringAsFixed(2)} int:${debug.intensityDeviation.toStringAsFixed(2)} '
          'strk:${debug.strokeWidthDeviation.toStringAsFixed(2)}'
          '${debug.hasWideUnderline ? " underline" : ""}${debug.matchesPageInk ? " pageInk" : ""}'
          '${debug.cappedByStraightness ? " capped" : ""}',
        );
      }
    }
    Clipboard.setData(ClipboardData(text: buffer.toString()));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Đã copy debug — dán vào tin nhắn để gửi.')),
    );
  }

  Future<void> _saveDebugImageToGallery(BuildContext context, List<TextRegion> regions) async {
    final image = _lastDecodedImage;
    if (image == null) return;
    setState(() => _savingDebugImage = true);
    try {
      final recorder = ui.PictureRecorder();
      final canvas = Canvas(recorder);
      canvas.drawImage(image, Offset.zero, Paint());
      // Scaled off the image's OWN width, not a fixed size — a phone-camera photo (thousands of
      // px wide) needs a much bigger font than a small document scan for the label to read as
      // roughly the same relative size either way, and to stay clearly bigger than the on-screen
      // 8px default (which exists for a much smaller logical-pixel canvas).
      final fontSize = (image.width * 0.012).clamp(14.0, 40.0);
      RegionPainter(regions: regions, scale: 1.0, fontSize: fontSize)
          .paint(canvas, Size(image.width.toDouble(), image.height.toDouble()));
      final picture = recorder.endRecording();
      final rendered = await picture.toImage(image.width, image.height);
      final bytes = await rendered.toByteData(format: ui.ImageByteFormat.png);
      if (bytes == null) throw Exception('Không thể tạo ảnh debug');

      final tempDir = await getTemporaryDirectory();
      final outPath = '${tempDir.path}/debug_${DateTime.now().millisecondsSinceEpoch}.png';
      await File(outPath).writeAsBytes(bytes.buffer.asUint8List());
      await ExportService().saveToGallery(outPath);

      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Đã lưu ảnh debug vào Ảnh — gửi trực tiếp từ đó.')),
        );
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Lưu thất bại: $e')));
      }
    } finally {
      if (mounted) setState(() => _savingDebugImage = false);
    }
  }
}
