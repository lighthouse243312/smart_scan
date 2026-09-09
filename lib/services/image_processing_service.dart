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

  /// Classifies each word with the trained CNN (see beacon_smart_scan/ml/) — the primary signal,
  /// since 6+ rounds of pure heuristics hit a hard accuracy ceiling on real worksheet photos.
  /// The native pixel stats (stroke width, ink color, baseline, wide-underline) are still pulled
  /// in as a secondary signal: mostly diluted into the blend, except "sits on a wide pre-printed
  /// blank line" — a structural cue the CNN can't see since it only looks at the word's own crop
  /// — which still acts as a floor. Regions scored as likely-handwriting come back pre-selected
  /// for erase; the user can still toggle any of them.
  Future<List<TextRegion>> scoreHandwriting(String imagePath, List<TextRegion> regions) => _run(() async {
        if (regions.isEmpty) return regions;
        final textBlocks = regions
            .map((r) => {
                  'id': r.id,
                  'left': r.boundingBox.left,
                  'top': r.boundingBox.top,
                  'right': r.boundingBox.right,
                  'bottom': r.boundingBox.bottom,
                  'charCount': r.text.replaceAll(RegExp(r'\s'), '').length,
                })
            .toList();

        final results = await Future.wait([
          ImageProcessingChannel.detectHandwritingRegions(imagePath: imagePath, textBlocks: textBlocks),
          ImageProcessingChannel.classifyHandwriting(imagePath: imagePath, textBlocks: textBlocks),
        ]);
        final stats = results[0] as Map<String, NativeWordStats>;
        final mlConfidence = results[1] as Map<String, double>;

        final inked = regions.map((r) => stats[r.id]).whereType<NativeWordStats>().where((s) => s.hasInk).toList();
        final referenceStrokeWidth = _median(inked.map((s) => s.avgStrokeWidth).toList());
        final referenceIntensityStdDev = _median(inked.map((s) => s.inkIntensityStdDev).toList());
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
          // Ink-darkness consistency is one-directional: printed ink IS the low-variance
          // reference, so only being NOTICEABLY MORE uneven than the page's typical print
          // counts (a word steadier than the reference isn't suspicious).
          final intensityRatio = referenceIntensityStdDev > 0
              ? s.inkIntensityStdDev / referenceIntensityStdDev
              : 1.0;
          final intensityDeviation = ((intensityRatio - 1.0)).clamp(0.0, 1.0);

          final heuristic = (0.25 * r.baselineVarianceScore +
                  0.20 * colorDeviation +
                  0.20 * s.angleVariationScore +
                  0.15 * intensityDeviation +
                  0.10 * strokeWidthDeviation +
                  0.10 * s.confidence)
              .clamp(0.0, 1.0);
          // The trained CNN is the primary signal; the heuristic (pixel stats self-calibrated
          // against this page's own printed text) stays as a secondary vote.
          final ml = mlConfidence[r.id] ?? heuristic;
          final weighted = (0.75 * ml + 0.25 * heuristic).clamp(0.0, 1.0);
          // A word sitting on a fill-in-blank's own pre-printed line is close to certain to be
          // handwriting — a floor, not just one more diluted vote among many. Also the one signal
          // here the CNN structurally can't see, since it only looks at the word's own crop.
          final blended = s.hasWideUnderline ? max(weighted, 0.8) : weighted;
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
