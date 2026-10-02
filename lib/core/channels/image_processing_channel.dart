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

  /// Straightens a photographed page: squared to its printed rules (rotation + keystone) or,
  /// without rules, its text rows levelled; an already-level page is copied unchanged.
  static Future<String> straighten({
    required String inputPath,
    required String outputPath,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('straighten', {
      'inputPath': inputPath,
      'outputPath': outputPath,
    });
    return result!['outputPath'] as String;
  }

  /// Brings a picture from outside the app (photo library / files) into the pipeline's format:
  /// decoded by the platform (HEIC too), turned upright by its EXIF orientation, flattened onto
  /// white, capped at 4000 px on the long side and written as PNG (lossless) to [outputPath].
  static Future<String> importImage({
    required String inputPath,
    required String outputPath,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('importImage', {
      'inputPath': inputPath,
      'outputPath': outputPath,
    });
    return result!['outputPath'] as String;
  }

  /// Ink-colour + layout mask. [colorDelta] = how much bluer than the local print (optical
  /// density OD_B/OD_R) a stroke must be to count as pen; smaller = more sensitive.
  static Future<HandwritingMaskResult> inkColorMask({
    required String inputPath,
    required String maskPath,
    required double colorDelta,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('inkColorMask', {
      'inputPath': inputPath,
      'maskPath': maskPath,
      'colorDelta': colorDelta,
    });
    return HandwritingMaskResult.fromMap(result!);
  }

  /// Model mask combined with the ink-colour mask: handwriting where either finds it (model
  /// probability above [threshold], or colour at [colorDelta]); print only where both agree.
  static Future<HandwritingMaskResult> segmentationMask({
    required String inputPath,
    required String maskPath,
    required double threshold,
    required double colorDelta,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('segmentationMask', {
      'inputPath': inputPath,
      'maskPath': maskPath,
      'threshold': threshold,
      'colorDelta': colorDelta,
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

  /// Rebuilds [inputPath] without the handwriting: where a stroke crosses real print the print is
  /// restored in its nearby colour, the rest is inpainted from the surrounding paper.
  /// [analysisPath] is the unprocessed page the mask was computed on (same geometry).
  static Future<String> eraseWithMask({
    required String inputPath,
    required String analysisPath,
    required String maskPath,
    required String outputPath,
  }) async {
    final result = await _channel.invokeMapMethod<String, dynamic>('eraseWithMask', {
      'inputPath': inputPath,
      'analysisPath': analysisPath,
      'maskPath': maskPath,
      'outputPath': outputPath,
    });
    return result!['outputPath'] as String;
  }
}
