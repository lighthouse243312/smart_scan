import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';

import '../../core/models/handwriting_method.dart';
import '../../services/export_service.dart';
import '../../state/scan_session.dart';
import '../../widgets/step_progress_indicator.dart';
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

    return Scaffold(
      appBar: AppBar(
        title: const Text('Chữ viết tay'),
        actions: [
          IconButton(
            tooltip: _showMask ? 'Ẩn vùng phát hiện' : 'Hiện vùng phát hiện',
            icon: Icon(_showMask ? Icons.visibility : Icons.visibility_off),
            onPressed: () => setState(() => _showMask = !_showMask),
          ),
          // Saves the RAW capture straight out of the document scanner — before sharpen,
          // shadow-removal, or erase touch it — so a bug report can tell "the detector read this
          // wrong" apart from "the photo itself was already blurry/warped/cropped oddly".
          if (kDebugMode && page != null)
            IconButton(
              tooltip: 'Lưu ảnh gốc vào Ảnh',
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
                StepProgressIndicator(currentStep: session.step),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: SegmentedButton<HandwritingMethod>(
                    segments: const [
                      ButtonSegment(
                        value: HandwritingMethod.inkColor,
                        label: Text('Màu mực'),
                        icon: Icon(Icons.palette_outlined),
                      ),
                      ButtonSegment(
                        value: HandwritingMethod.segmentation,
                        label: Text('Model AI'),
                        icon: Icon(Icons.auto_awesome_outlined),
                      ),
                    ],
                    selected: {session.method},
                    onSelectionChanged: session.isProcessing
                        ? null
                        : (s) => session.updateDetection(method: s.first),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Row(
                    children: [
                      const Text('Độ nhạy', style: TextStyle(fontSize: 13)),
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
                    _hintText(session.method, page.maskCoverage),
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
                    child: Text(session.lastError!, style: const TextStyle(color: Colors.red)),
                  ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: SegmentedButton<_BrushMode>(
                    segments: const [
                      ButtonSegment(value: _BrushMode.none, label: Text('Xem'), icon: Icon(Icons.pan_tool_outlined)),
                      ButtonSegment(value: _BrushMode.add, label: Text('Thêm vùng xoá'), icon: Icon(Icons.brush)),
                      ButtonSegment(value: _BrushMode.keep, label: Text('Giữ lại'), icon: Icon(Icons.cleaning_services_outlined)),
                    ],
                    selected: {_brushMode},
                    onSelectionChanged: (s) => setState(() => _brushMode = s.first),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                  child: FilledButton.icon(
                    onPressed: session.isProcessing || page.maskCoverage <= 0
                        ? null
                        : () async {
                            await session.eraseHandwriting();
                            if (context.mounted && session.lastError == null) {
                              Navigator.of(context).push(
                                MaterialPageRoute(builder: (_) => const ExportScreen()),
                              );
                            }
                          },
                    icon: session.isProcessing
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.auto_fix_high),
                    label: const Text('Xoá chữ viết tay'),
                  ),
                ),
              ],
            ),
    );
  }

  String _hintText(HandwritingMethod method, double coverage) {
    if (coverage <= 0) {
      return method == HandwritingMethod.inkColor
          ? 'Không thấy mực màu — chữ viết bằng bút đen/bút chì cần dùng "Model AI".'
          : 'Không thấy chữ viết tay — thử tăng độ nhạy hoặc tô thêm bằng cọ.';
    }
    final percent = (coverage * 100).toStringAsFixed(coverage < 0.01 ? 2 : 1);
    return 'Vùng đỏ ($percent% trang) sẽ bị xoá; chỗ tím là chữ in bị viết đè — sẽ được giữ lại. '
        'Dùng cọ để thêm vùng hoặc giữ lại chữ bị tô nhầm; đổi cách/độ nhạy sẽ phát hiện lại từ đầu.';
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
                onPanEnd: brushing && !session.isProcessing
                    ? (_) => _commitStroke(session, brushImageWidth)
                    : null,
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
                      const ColoredBox(
                        color: Color(0x66FFFFFF),
                        child: Center(child: CircularProgressIndicator()),
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
      points: [for (final p in points) ...[p.dx, p.dy]],
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
      await ExportService().saveToGallery(outPath);

      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Đã lưu ảnh gốc vào Ảnh — gửi trực tiếp từ đó.')),
        );
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Lưu thất bại: $e')));
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
