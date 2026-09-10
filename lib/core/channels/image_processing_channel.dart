import 'package:flutter/services.dart';

/// Raw per-word pixel measurements from the native side — a stroke-shape confidence (a weak
/// fallback signal on its own), the word's average stroke width, and its average ink color
/// (BGR). None of these are a handwriting verdict by themselves; see
/// ImageProcessingService.scoreHandwriting for how they're compared against the page's own
/// most-common values to reach one.
class NativeWordStats {
  const NativeWordStats({
    required this.confidence,
    required this.angleVariationScore,
    required this.avgStrokeWidth,
    required this.inkColorB,
    required this.inkColorG,
    required this.inkColorR,
    required this.inkIntensityStdDev,
    required this.hasWideUnderline,
    required this.hasReliableAngleData,
    required this.hasInk,
  });

  final double confidence;

  /// Self-contained [0, 1] score: how much each letter's tilt varies from the next within this
  /// one word — intrinsic to the ink shape, independent of position/color. A printed font
  /// repeats the exact same glyph angle every time; a hand never repeats a stroke identically.
  final double angleVariationScore;
  final double avgStrokeWidth;
  final double inkColorB;
  final double inkColorG;
  final double inkColorR;

  /// Std-dev of pixel darkness within the word's own ink — printed ink/toner is near-uniform
  /// (low value), pen ink varies with pressure/speed/flow (higher value).
  final double inkIntensityStdDev;

  /// True when the word sits on a long, near-solid dark line spanning noticeably wider than the
  /// word itself — a fill-in-the-blank answer written on its pre-printed blank line. A printed
  /// word's own in-text underline (emphasis) hugs the word tightly instead, so it reads false.
  /// The single strongest signal on a worksheet-style document.
  final bool hasWideUnderline;

  /// False when the native side had fewer than 2 measurable stroke components to compare angles
  /// across (a tiny fragment — a single short stroke or curl, e.g. a split-off tail end of a
  /// word) — [angleVariationScore] is then a meaningless 0.0 placeholder, NOT a measurement of
  /// "this is dead straight." The straightness ceiling in ImageProcessingService must check this
  /// before treating a low angleVariationScore as evidence of print — verified: a genuine
  /// handwriting fragment this small got angle 0.0 purely from lacking enough data, and was
  /// capped to a near-zero score as if confidently straight print.
  final bool hasReliableAngleData;
  final bool hasInk;
}

/// Thin typed wrapper over the native `beacon_smart_scan/image_processing` MethodChannel.
/// The actual sharpen/shadow-removal/handwriting-detect/inpaint algorithms are implemented
/// natively (Kotlin+OpenCV on Android, Obj-C++/OpenCV on iOS) — see
/// android/app/src/main/kotlin/.../imageprocessing/ and ios/Runner/ImageProcessingOpenCV.mm.
class ImageProcessingChannel {
  ImageProcessingChannel._();

  static const MethodChannel _channel = MethodChannel('beacon_smart_scan/image_processing');

  static Future<String> sharpen({
    required String inputPath,
    required String outputPath,
    double amount = 1.5,
    double radius = 3.0,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('sharpen', {
      'inputPath': inputPath,
      'outputPath': outputPath,
      'amount': amount,
      'radius': radius,
    });
    return result!['outputPath'] as String;
  }

  static Future<String> removeShadow({
    required String inputPath,
    required String outputPath,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('removeShadow', {
      'inputPath': inputPath,
      'outputPath': outputPath,
    });
    return result!['outputPath'] as String;
  }

  static Future<String> rotate({
    required String inputPath,
    required String outputPath,
    required int quarterTurnsClockwise,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('rotate', {
      'inputPath': inputPath,
      'outputPath': outputPath,
      'quarterTurnsClockwise': quarterTurnsClockwise,
    });
    return result!['outputPath'] as String;
  }

  /// ML Kit's text recognizer sometimes emits NO region at all for loosely-connected cursive
  /// handwriting (verified on a real photo: two lines of a handwritten note got zero boxes,
  /// while a third line in clearer, more separated letters was detected fine) — its OCR-based
  /// detector has an implicit "is this legible text" confidence gate that messy cursive can
  /// fall below. This finds ink not already covered by any of [existingBlocks], merges nearby
  /// letters/words into phrase-level blobs, and returns them as extra candidate regions so the
  /// classifier — which only needs a pixel crop, not a transcription — can still score them.
  /// [existingBlocks] keys: left, top, right, bottom. Returns `{id, left, top, right, bottom}` maps.
  static Future<List<Map<String, dynamic>>> detectOrphanRegions({
    required String imagePath,
    required List<Map<String, Object>> existingBlocks,
  }) async {
    final result = await _channel.invokeListMethod<Map<Object?, Object?>>(
      'detectOrphanRegions',
      {'imagePath': imagePath, 'existingBlocks': existingBlocks},
    );
    return (result ?? const []).map((e) => e.cast<String, dynamic>()).toList();
  }

  /// [textBlocks] keys: id, left, top, right, bottom, charCount (image pixel coordinates).
  /// Returns raw per-word measurements keyed by region id — NOT a handwriting decision. Native
  /// only measures; [ImageProcessingService.scoreHandwriting] decides by comparing each word
  /// against the page's own most-common stroke width/ink color.
  static Future<Map<String, NativeWordStats>> detectHandwritingRegions({
    required String imagePath,
    required List<Map<String, Object>> textBlocks,
  }) async {
    final result = await _channel.invokeListMethod<Map<Object?, Object?>>(
      'detectHandwritingRegions',
      {'imagePath': imagePath, 'textBlocks': textBlocks},
    );
    final map = <String, NativeWordStats>{};
    for (final entry in result ?? const []) {
      final id = entry['id'] as String;
      map[id] = NativeWordStats(
        confidence: (entry['confidence'] as num).toDouble(),
        angleVariationScore: (entry['angleVariationScore'] as num).toDouble(),
        avgStrokeWidth: (entry['avgStrokeWidth'] as num).toDouble(),
        inkColorB: (entry['inkColorB'] as num).toDouble(),
        inkColorG: (entry['inkColorG'] as num).toDouble(),
        inkColorR: (entry['inkColorR'] as num).toDouble(),
        inkIntensityStdDev: (entry['inkIntensityStdDev'] as num).toDouble(),
        hasWideUnderline: entry['hasWideUnderline'] as bool,
        hasReliableAngleData: entry['hasReliableAngleData'] as bool,
        hasInk: entry['hasInk'] as bool,
      );
    }
    return map;
  }

  /// Runs the trained CNN (see beacon_smart_scan/ml/) on each word crop — a real classifier,
  /// not a heuristic. [textBlocks] keys: id, left, top, right, bottom (image pixel coordinates).
  /// Returns each region's raw sigmoid output (1.0 == handwriting) keyed by id.
  static Future<Map<String, double>> classifyHandwriting({
    required String imagePath,
    required List<Map<String, Object>> textBlocks,
  }) async {
    final result = await _channel.invokeListMethod<Map<Object?, Object?>>(
      'classifyHandwriting',
      {'imagePath': imagePath, 'textBlocks': textBlocks},
    );
    final map = <String, double>{};
    for (final entry in result ?? const []) {
      final id = entry['id'] as String;
      map[id] = (entry['mlConfidence'] as num).toDouble();
    }
    return map;
  }

  static Future<String> eraseRegions({
    required String inputPath,
    required String outputPath,
    required List<Rect> rects,
    List<Rect> keepRects = const [],
    double padding = 6.0,
    double inpaintRadius = 5.0,
  }) async {
    Map<String, double> rectToMap(Rect r) => {
          'left': r.left,
          'top': r.top,
          'right': r.right,
          'bottom': r.bottom,
        };
    final result = await _channel.invokeMapMethod<String, dynamic>('eraseRegions', {
      'inputPath': inputPath,
      'outputPath': outputPath,
      'rects': rects.map(rectToMap).toList(),
      'keepRects': keepRects.map(rectToMap).toList(),
      'padding': padding,
      'inpaintRadius': inpaintRadius,
    });
    return result!['outputPath'] as String;
  }
}
