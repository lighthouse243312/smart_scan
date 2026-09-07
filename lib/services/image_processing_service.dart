import 'dart:math';

import 'package:flutter/services.dart';

import '../core/channels/image_processing_channel.dart';
import '../core/models/text_region.dart';
import '../core/utils/temp_paths.dart';

/// App-level exception surfaced from the native pipeline (wraps [PlatformException]).
class ImageProcessingException implements Exception {
  ImageProcessingException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Typed façade over [ImageProcessingChannel] — allocates temp output paths and converts
/// [PlatformException]s from the native side into [ImageProcessingException].
class ImageProcessingService {
  Future<String> sharpen(String inputPath) => _run(() async {
        final outputPath = await TempPaths.next('sharpened.png');
        return ImageProcessingChannel.sharpen(inputPath: inputPath, outputPath: outputPath);
      });

  Future<String> removeShadow(String inputPath) => _run(() async {
        final outputPath = await TempPaths.next('shadow_removed.png');
        return ImageProcessingChannel.removeShadow(inputPath: inputPath, outputPath: outputPath);
      });

  /// Rotates by 90°-steps (1 = 90° CW, 2 = 180°, 3 = 90° CCW). Document scanners crop the page
  /// rectangle correctly but don't know which edge is "up" for reading, so this lets the user
  /// fix orientation by hand before text detection runs on it.
  Future<String> rotate(String inputPath, {required int quarterTurnsClockwise}) => _run(() async {
        final outputPath = await TempPaths.next('rotated.png');
        return ImageProcessingChannel.rotate(
          inputPath: inputPath,
          outputPath: outputPath,
          quarterTurnsClockwise: quarterTurnsClockwise,
        );
      });

  /// Gets each word's raw native pixel stats, then decides handwriting vs print by comparing
  /// every word against the PAGE'S OWN most-common stroke width and ink color — self-calibrating
  /// against whatever this particular document's printed font/ink actually looks like, rather
  /// than a fixed threshold. Most of a page is printed text sharing one font and one ink color;
  /// whichever words deviate from that majority (different stroke width, different ink color,
  /// different baseline — see TextRecognitionService) are the handwriting candidates. Regions
  /// scored as likely-handwriting come back pre-selected for erase — the user can still toggle
  /// any of them, since this is a best-effort heuristic, not a trained classifier.
  Future<List<TextRegion>> scoreHandwriting(String imagePath, List<TextRegion> regions) => _run(() async {
        if (regions.isEmpty) return regions;
        final stats = await ImageProcessingChannel.detectHandwritingRegions(
          imagePath: imagePath,
          textBlocks: regions
              .map((r) => {
                    'id': r.id,
                    'left': r.boundingBox.left,
                    'top': r.boundingBox.top,
                    'right': r.boundingBox.right,
                    'bottom': r.boundingBox.bottom,
                    'charCount': r.text.replaceAll(RegExp(r'\s'), '').length,
                  })
              .toList(),
        );

        final inked = regions.map((r) => stats[r.id]).whereType<NativeWordStats>().where((s) => s.hasInk).toList();
        final referenceStrokeWidth = _median(inked.map((s) => s.avgStrokeWidth).toList());
        final referenceColor = (
          b: _median(inked.map((s) => s.inkColorB).toList()),
          g: _median(inked.map((s) => s.inkColorG).toList()),
          r: _median(inked.map((s) => s.inkColorR).toList()),
        );

        return regions.map((r) {
          final s = stats[r.id];
          if (s == null || !s.hasInk) return r;

          final strokeWidthDeviation = referenceStrokeWidth > 0
              ? ((s.avgStrokeWidth - referenceStrokeWidth).abs() / referenceStrokeWidth).clamp(0.0, 1.0)
              : 0.0;
          final colorDistance = _colorDistance(s, referenceColor);
          // 255 * sqrt(3) is the max possible BGR distance; 60 ("noticeably different ink") is
          // a starting normalizer, not tuned against a labeled dataset yet.
          final colorDeviation = (colorDistance / 60.0).clamp(0.0, 1.0);

          final blended = (0.35 * r.baselineVarianceScore +
                  0.30 * colorDeviation +
                  0.20 * strokeWidthDeviation +
                  0.15 * s.confidence)
              .clamp(0.0, 1.0);
          final isHandwriting = blended > 0.5;
          return TextRegion(
            id: r.id,
            boundingBox: r.boundingBox,
            text: r.text,
            confidence: blended,
            isLikelyHandwriting: isHandwriting,
            selectedForErase: isHandwriting,
            isManual: r.isManual,
            baselineVarianceScore: r.baselineVarianceScore,
          );
        }).toList();
      });

  double _colorDistance(NativeWordStats s, ({double b, double g, double r}) reference) {
    final db = s.inkColorB - reference.b;
    final dg = s.inkColorG - reference.g;
    final dr = s.inkColorR - reference.r;
    return sqrt(db * db + dg * dg + dr * dr);
  }

  double _median(List<double> values) {
    if (values.isEmpty) return 0.0;
    final sorted = [...values]..sort();
    final mid = sorted.length ~/ 2;
    return sorted.length.isOdd ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  Future<String> eraseRegions(String inputPath, List<Rect> rects) => _run(() async {
        final outputPath = await TempPaths.next('erased.png');
        return ImageProcessingChannel.eraseRegions(inputPath: inputPath, outputPath: outputPath, rects: rects);
      });

  Future<T> _run<T>(Future<T> Function() action) async {
    try {
      return await action();
    } on PlatformException catch (e) {
      throw ImageProcessingException(e.message ?? 'Xử lý ảnh thất bại (${e.code})');
    }
  }
}
