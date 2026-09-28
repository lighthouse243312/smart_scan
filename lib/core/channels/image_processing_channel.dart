import 'package:flutter/services.dart';

import '../models/handwriting_method.dart';

/// A mask file written by the native side: a page-sized PNG with two independent layers — alpha =
/// handwriting (drawn semi-transparent red, so the file doubles as the review overlay; magenta
/// where it crosses print) and blue = printed ink, which the erase step restores.
class HandwritingMaskResult {
  const HandwritingMaskResult({required this.maskPath, required this.coverage});

  final String maskPath;

  /// Fraction of the page covered by the mask, 0-1.
  final double coverage;

  factory HandwritingMaskResult.fromMap(Map<String, dynamic> map) => HandwritingMaskResult(
        maskPath: map['maskPath'] as String,
        coverage: (map['coverage'] as num).toDouble(),
      );
}

/// Thin typed wrapper over the native `beacon_smart_scan/image_processing` MethodChannel.
/// The algorithms are implemented natively and mirrored on both platforms — Kotlin + OpenCV +
/// TFLite on Android (android/app/src/main/kotlin/.../imageprocessing/), Obj-C++ + OpenCV +
/// Core ML on iOS (ios/Runner/ImageProcessingOpenCV.mm).
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

  /// Colour-of-ink mask. [minSaturation] is on OpenCV's 0-255 HSV saturation scale.
  static Future<HandwritingMaskResult> inkColorMask({
    required String inputPath,
    required String maskPath,
    required double minSaturation,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('inkColorMask', {
      'inputPath': inputPath,
      'maskPath': maskPath,
      'minSaturation': minSaturation,
    });
    return HandwritingMaskResult.fromMap(result!);
  }

  /// Segmentation-model mask: pixels whose handwriting probability exceeds [threshold].
  static Future<HandwritingMaskResult> segmentationMask({
    required String inputPath,
    required String maskPath,
    required double threshold,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('segmentationMask', {
      'inputPath': inputPath,
      'maskPath': maskPath,
      'threshold': threshold,
    });
    return HandwritingMaskResult.fromMap(result!);
  }

  /// Applies manual brush [strokes] to the mask at [maskPath], writing the result to [outputPath].
  static Future<HandwritingMaskResult> applyMaskStrokes({
    required String maskPath,
    required String outputPath,
    required List<MaskStroke> strokes,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('applyMaskStrokes', {
      'maskPath': maskPath,
      'outputPath': outputPath,
      'strokes': strokes.map((s) => s.toMap()).toList(),
    });
    return HandwritingMaskResult.fromMap(result!);
  }

  /// Rebuilds the page without the handwriting: handwriting pixels over print get the nearby print
  /// colour back, the rest get the paper colour (no inpainting, so crossed-over print stays sharp).
  static Future<String> eraseWithMask({
    required String inputPath,
    required String maskPath,
    required String outputPath,
    double dilatePx = 2.0,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('eraseWithMask', {
      'inputPath': inputPath,
      'maskPath': maskPath,
      'outputPath': outputPath,
      'dilatePx': dilatePx,
    });
    return result!['outputPath'] as String;
  }
}
