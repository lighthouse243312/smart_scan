import 'package:flutter/services.dart';

import '../core/channels/image_processing_channel.dart';
import '../core/models/handwriting_method.dart';
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

  /// Converts a picture picked from outside the app into the pipeline's upright PNG (lossless: no
  /// second JPEG pass over the ink colours).
  Future<String> importImage(String inputPath) => _run(() async {
        final outputPath = await TempPaths.next('imported.png');
        return ImageProcessingChannel.importImage(inputPath: inputPath, outputPath: outputPath);
      });

  /// Rotates by 90°-steps (1 = 90° CW, 2 = 180°, 3 = 90° CCW). Document scanners crop the page
  /// rectangle correctly but don't know which edge is "up" for reading, so this lets the user
  /// fix orientation by hand.
  Future<String> rotate(String inputPath, {required int quarterTurnsClockwise}) => _run(() async {
        final outputPath = await TempPaths.next('rotated.png');
        return ImageProcessingChannel.rotate(
          inputPath: inputPath,
          outputPath: outputPath,
          quarterTurnsClockwise: quarterTurnsClockwise,
        );
      });

  /// Builds the handwriting mask for [inputPath] with [method]. [sensitivity] (0-1, higher =
  /// erase more) is mapped onto each method's own knob: the colour method's required colour
  /// difference from print, the model's probability threshold.
  Future<HandwritingMaskResult> detectHandwritingMask(
    String inputPath, {
    required HandwritingMethod method,
    required double sensitivity,
  }) =>
      _run(() async {
        final s = sensitivity.clamp(0.0, 1.0);
        final maskPath = await TempPaths.next('handwriting_mask.png');
        return switch (method) {
          HandwritingMethod.inkColor => ImageProcessingChannel.inkColorMask(
              inputPath: inputPath,
              maskPath: maskPath,
              colorDelta: 0.08 - 0.06 * s,
            ),
          HandwritingMethod.segmentation => ImageProcessingChannel.segmentationMask(
              inputPath: inputPath,
              maskPath: maskPath,
              threshold: 0.8 - 0.6 * s,
              colorDelta: 0.08 - 0.06 * s,
            ),
        };
      });

  Future<HandwritingMaskResult> applyMaskStrokes(String maskPath, List<MaskStroke> strokes) => _run(() async {
        final outputPath = await TempPaths.next('handwriting_mask_edited.png');
        return ImageProcessingChannel.applyMaskStrokes(
          maskPath: maskPath,
          outputPath: outputPath,
          strokes: strokes,
        );
      });

  /// Erases the mask's handwriting from [inputPath]; [analysisPath] is the page the mask was
  /// detected on.
  Future<String> eraseWithMask(String inputPath, {required String analysisPath, required String maskPath}) =>
      _run(() async {
        final outputPath = await TempPaths.next('erased.png');
        return ImageProcessingChannel.eraseWithMask(
          inputPath: inputPath,
          analysisPath: analysisPath,
          maskPath: maskPath,
          outputPath: outputPath,
        );
      });

  Future<T> _run<T>(Future<T> Function() action) async {
    try {
      return await action();
    } on PlatformException catch (e) {
      throw ImageProcessingException(e.message ?? 'Xử lý ảnh thất bại (${e.code})');
    }
  }
}
