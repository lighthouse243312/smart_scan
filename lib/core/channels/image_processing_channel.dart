import 'package:flutter/services.dart';

/// Raw per-word pixel measurements from the native side — a stroke-shape confidence (a weak
/// fallback signal on its own), the word's average stroke width, and its average ink color
/// (BGR). None of these are a handwriting verdict by themselves; see
/// ImageProcessingService.scoreHandwriting for how they're compared against the page's own
/// most-common values to reach one.
class NativeWordStats {
  const NativeWordStats({
    required this.confidence,
    required this.avgStrokeWidth,
    required this.inkColorB,
    required this.inkColorG,
    required this.inkColorR,
    required this.hasInk,
  });

  final double confidence;
  final double avgStrokeWidth;
  final double inkColorB;
  final double inkColorG;
  final double inkColorR;
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
        avgStrokeWidth: (entry['avgStrokeWidth'] as num).toDouble(),
        inkColorB: (entry['inkColorB'] as num).toDouble(),
        inkColorG: (entry['inkColorG'] as num).toDouble(),
        inkColorR: (entry['inkColorR'] as num).toDouble(),
        hasInk: entry['hasInk'] as bool,
      );
    }
    return map;
  }

  static Future<String> eraseRegions({
    required String inputPath,
    required String outputPath,
    required List<Rect> rects,
    double padding = 6.0,
    double inpaintRadius = 5.0,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('eraseRegions', {
      'inputPath': inputPath,
      'outputPath': outputPath,
      'rects': rects
          .map((r) => {
                'left': r.left,
                'top': r.top,
                'right': r.right,
                'bottom': r.bottom,
              })
          .toList(),
      'padding': padding,
      'inpaintRadius': inpaintRadius,
    });
    return result!['outputPath'] as String;
  }
}
