import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../core/models/text_region.dart';

/// Draws every [TextRegion] over an image and handles tap-to-toggle / drag-to-draw-a-new-region.
/// The parent must size this exactly to the displayed image's box (e.g. wrap the image + this
/// widget in an `AspectRatio` matching the image's own aspect ratio) — [imageWidth]/[imageHeight]
/// are the image's natural pixel size, used only to convert between screen and image coordinates.
class RegionOverlay extends StatefulWidget {
  const RegionOverlay({
    super.key,
    required this.imageWidth,
    required this.imageHeight,
    required this.regions,
    required this.onToggle,
    this.onManualRegion,
    this.drawModeEnabled = false,
  });

  final int imageWidth;
  final int imageHeight;
  final List<TextRegion> regions;
  final ValueChanged<String> onToggle;
  final ValueChanged<Rect>? onManualRegion;
  final bool drawModeEnabled;

  @override
  State<RegionOverlay> createState() => _RegionOverlayState();
}

class _RegionOverlayState extends State<RegionOverlay> {
  Offset? _dragStart;
  Offset? _dragCurrent;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final scale = constraints.maxWidth / widget.imageWidth;

        Rect toImageRect(Offset a, Offset b) {
          final left = (a.dx < b.dx ? a.dx : b.dx) / scale;
          final top = (a.dy < b.dy ? a.dy : b.dy) / scale;
          final right = (a.dx > b.dx ? a.dx : b.dx) / scale;
          final bottom = (a.dy > b.dy ? a.dy : b.dy) / scale;
          return Rect.fromLTRB(left, top, right, bottom);
        }

        return GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTapUp: widget.drawModeEnabled
              ? null
              : (details) {
                  final tapImagePoint = details.localPosition / scale;
                  for (final region in widget.regions.reversed) {
                    if (region.boundingBox.contains(tapImagePoint)) {
                      widget.onToggle(region.id);
                      return;
                    }
                  }
                },
          onPanStart: widget.drawModeEnabled ? (d) => setState(() => _dragStart = d.localPosition) : null,
          onPanUpdate: widget.drawModeEnabled ? (d) => setState(() => _dragCurrent = d.localPosition) : null,
          onPanEnd: widget.drawModeEnabled
              ? (_) {
                  if (_dragStart != null && _dragCurrent != null) {
                    final rect = toImageRect(_dragStart!, _dragCurrent!);
                    if (rect.width > 4 && rect.height > 4) {
                      widget.onManualRegion?.call(rect);
                    }
                  }
                  setState(() {
                    _dragStart = null;
                    _dragCurrent = null;
                  });
                }
              : null,
          child: CustomPaint(
            size: Size(constraints.maxWidth, constraints.maxHeight),
            painter: _RegionPainter(
              regions: widget.regions,
              scale: scale,
              dragStart: _dragStart,
              dragCurrent: _dragCurrent,
            ),
          ),
        );
      },
    );
  }
}

class _RegionPainter extends CustomPainter {
  _RegionPainter({required this.regions, required this.scale, this.dragStart, this.dragCurrent});

  final List<TextRegion> regions;
  final double scale;
  final Offset? dragStart;
  final Offset? dragCurrent;

  @override
  void paint(Canvas canvas, Size size) {
    for (final region in regions) {
      final rect = Rect.fromLTRB(
        region.boundingBox.left * scale,
        region.boundingBox.top * scale,
        region.boundingBox.right * scale,
        region.boundingBox.bottom * scale,
      );
      final isHandwriting = region.isLikelyHandwriting || region.isManual;
      final borderColor = region.selectedForErase
          ? const Color(0xFFE53935)
          : (isHandwriting ? const Color(0xFFFB8C00) : const Color(0xFF43A047));
      final fillColor = borderColor.withValues(alpha: region.selectedForErase ? 0.22 : 0.10);

      canvas.drawRect(rect, Paint()..color = fillColor);
      canvas.drawRect(
        rect,
        Paint()
          ..color = borderColor
          ..style = PaintingStyle.stroke
          ..strokeWidth = region.selectedForErase ? 2.5 : 1.5,
      );

      // Debug builds only: every signal that fed into the final score, next to each box — a
      // wrong verdict's exact breakdown (vs. never having been detected at all, which shows no
      // box whatsoever) is visible at a glance instead of guessed at one heuristic-weight change
      // at a time. Two rounds of blind reweighting this session each fixed one real case and
      // broke another; this exists so the next fix is aimed at the actual number, not a guess.
      if (kDebugMode) {
        final debug = region.debugBreakdown;
        final label = region.isManual
            ? 'manual'
            : debug == null
                ? region.confidence.toStringAsFixed(2)
                : 'f:${region.confidence.toStringAsFixed(2)} '
                    'ml:${debug.mlConfidence.toStringAsFixed(2)} '
                    'h:${debug.heuristic.toStringAsFixed(2)}\n'
                    'ang:${debug.angleVariationScore.toStringAsFixed(2)} '
                    'col:${debug.colorDeviation.toStringAsFixed(2)} '
                    'int:${debug.intensityDeviation.toStringAsFixed(2)}\n'
                    'stroke:${debug.strokeWidthDeviation.toStringAsFixed(2)}'
                    '${debug.hasWideUnderline ? " underline" : ""}'
                    '${debug.cappedByStraightness ? " CAPPED" : ""}';
        final painter = TextPainter(
          text: TextSpan(
            text: label,
            style: const TextStyle(fontSize: 8, color: Colors.white, fontWeight: FontWeight.bold, height: 1.2),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        final labelOrigin = Offset(rect.left, rect.top - painter.height);
        canvas.drawRect(
          Rect.fromLTWH(labelOrigin.dx, labelOrigin.dy, painter.width + 4, painter.height + 2),
          Paint()..color = borderColor.withValues(alpha: 0.9),
        );
        painter.paint(canvas, labelOrigin + const Offset(2, 1));
      }
    }

    if (dragStart != null && dragCurrent != null) {
      final dragRect = Rect.fromPoints(dragStart!, dragCurrent!);
      canvas.drawRect(dragRect, Paint()..color = const Color(0xFF1E88E5).withValues(alpha: 0.15));
      canvas.drawRect(
        dragRect,
        Paint()
          ..color = const Color(0xFF1E88E5)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _RegionPainter oldDelegate) {
    return oldDelegate.regions != regions ||
        oldDelegate.dragStart != dragStart ||
        oldDelegate.dragCurrent != dragCurrent;
  }
}
