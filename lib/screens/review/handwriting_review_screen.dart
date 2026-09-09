import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../../core/models/text_region.dart';
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
                    future: _decodeImage(page.displayPath),
                    builder: (context, snapshot) {
                      if (!snapshot.hasData) {
                        return const Center(child: CircularProgressIndicator());
                      }
                      final image = snapshot.data!;
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
    for (final region in regions) {
      final box = region.boundingBox;
      buffer.writeln(
        '${region.id}  box:[${box.left.toStringAsFixed(0)},${box.top.toStringAsFixed(0)},'
        '${box.width.toStringAsFixed(0)}x${box.height.toStringAsFixed(0)}]  '
        '${region.isManual ? "manual" : region.isLikelyHandwriting ? "HANDWRITING" : "print"}',
      );
      final debug = region.debugBreakdown;
      if (debug != null) {
        buffer.writeln(
          '  final:${region.confidence.toStringAsFixed(3)} '
          'ml:${debug.mlConfidence.toStringAsFixed(3)} '
          'heuristic:${debug.heuristic.toStringAsFixed(3)}',
        );
        buffer.writeln(
          '  angle:${debug.angleVariationScore.toStringAsFixed(3)} '
          'color:${debug.colorDeviation.toStringAsFixed(3)} '
          'intensity:${debug.intensityDeviation.toStringAsFixed(3)} '
          'stroke:${debug.strokeWidthDeviation.toStringAsFixed(3)}',
        );
        buffer.writeln(
          '  underline:${debug.hasWideUnderline} pageInk:${debug.matchesPageInk} capped:${debug.cappedByStraightness}',
        );
      }
    }
    Clipboard.setData(ClipboardData(text: buffer.toString()));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Đã copy debug — dán vào tin nhắn để gửi.')),
    );
  }
}
