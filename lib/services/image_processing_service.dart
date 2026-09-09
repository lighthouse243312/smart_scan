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
        final mlKitBlocks = regions
            .map((r) => {
                  'left': r.boundingBox.left,
                  'top': r.boundingBox.top,
                  'right': r.boundingBox.right,
                  'bottom': r.boundingBox.bottom,
                })
            .toList();

        // ML Kit sometimes emits no region at all for loosely-connected cursive handwriting —
        // verified on a real photo where two lines of a handwritten note got zero boxes while a
        // clearer third line was detected fine. Find ink it left unclaimed and add those as
        // extra candidate regions before scoring, so this isn't a dead end for the classifier —
        // it only ever needs a pixel crop, not a transcription.
        final orphanMaps = await ImageProcessingChannel.detectOrphanRegions(
          imagePath: imagePath,
          existingBlocks: mlKitBlocks,
        );
        final orphanRegions = orphanMaps.map((m) => TextRegion(
              id: m['id'] as String,
              boundingBox: Rect.fromLTRB(
                (m['left'] as num).toDouble(),
                (m['top'] as num).toDouble(),
                (m['right'] as num).toDouble(),
                (m['bottom'] as num).toDouble(),
              ),
              text: '',
            ));
        final allRegions = [...regions, ...orphanRegions];
        if (allRegions.isEmpty) return allRegions;

        final textBlocks = allRegions
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

        // Stroke width scales with font size — a big bold title and a tiny calendar-grid number
        // can both be perfectly ordinary print, just at very different sizes. Comparing raw
        // pixel stroke widths made the bigger one look like a wild outlier purely because most
        // of a page's words (e.g. a calendar's ~300 date numbers) are small, dragging the
        // reference stroke width down and flagging any large heading as "different style".
        // Normalizing by each word's own height first makes the comparison scale-invariant.
        double normalizedStrokeWidth(NativeWordStats s, TextRegion region) {
          final height = region.boundingBox.height;
          return height > 0 ? s.avgStrokeWidth / height : s.avgStrokeWidth;
        }

        final inkedRegions = allRegions.where((r) => stats[r.id]?.hasInk ?? false).toList();
        final referenceStrokeWidth = _median(inkedRegions.map((r) => normalizedStrokeWidth(stats[r.id]!, r)).toList());
        final inked = inkedRegions.map((r) => stats[r.id]!).toList();
        final referenceIntensityStdDev = _median(inked.map((s) => s.inkIntensityStdDev).toList());
        final referenceColor = (
          b: _median(inked.map((s) => s.inkColorB).toList()),
          g: _median(inked.map((s) => s.inkColorG).toList()),
          r: _median(inked.map((s) => s.inkColorR).toList()),
        );

        return allRegions.map((r) {
          final s = stats[r.id];
          if (s == null || !s.hasInk) return r;

          final strokeWidthDeviation = referenceStrokeWidth > 0
              ? ((normalizedStrokeWidth(s, r) - referenceStrokeWidth).abs() / referenceStrokeWidth).clamp(0.0, 1.0)
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

          final heuristic = (0.20 * r.baselineVarianceScore +
                  0.15 * colorDeviation +
                  0.35 * s.angleVariationScore +
                  0.10 * intensityDeviation +
                  0.10 * strokeWidthDeviation +
                  0.10 * s.confidence)
              .clamp(0.0, 1.0);
          // The trained CNN is the primary signal; the heuristic (pixel stats self-calibrated
          // against this page's own printed text) stays as a secondary vote.
          final ml = mlConfidence[r.id] ?? heuristic;
          final weighted = (0.75 * ml + 0.25 * heuristic).clamp(0.0, 1.0);
          // Printed glyphs repeat the exact same stroke angle; a real hand never does — verified
          // directly on a case that fooled the CNN (several calendar date numbers merged by the
          // text recognizer into one garbled multi-digit block, which the CNN read as irregular
          // and thus handwriting-like): angle-variation score came out 0.0-0.3 for those merged-
          // but-still-printed blocks vs a consistent ~1.0 for genuine handwriting in the same
          // photo. That's a wide enough margin to treat a very low score as near-certain print,
          // overriding the CNN — a ceiling, mirroring hasWideUnderline's floor on the other side.
          final cappedByStraightness = s.angleVariationScore <= 0.15 ? min(weighted, 0.2) : weighted;
          // A word sitting on a fill-in-blank's own pre-printed line is close to certain to be
          // handwriting — a floor, not just one more diluted vote among many. Also the one signal
          // here the CNN structurally can't see, since it only looks at the word's own crop.
          final blended = s.hasWideUnderline ? max(cappedByStraightness, 0.8) : cappedByStraightness;
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
