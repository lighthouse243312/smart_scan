import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

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
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16),
                  child: Text(
                    'Vùng gợi ý được tô sẵn để xoá — chạm để bật/tắt. Bật chế độ vẽ (biểu tượng bút) để tự khoanh thêm vùng.',
                    style: TextStyle(fontSize: 13, color: Colors.grey),
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
                      return Center(
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
}
