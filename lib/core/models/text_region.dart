import 'dart:ui';

/// One text block detected by ML Kit on a scan page, plus this app's own handwriting
/// heuristic score and the user's final say on whether to erase it.
class TextRegion {
  TextRegion({
    required this.id,
    required this.boundingBox,
    required this.text,
    this.confidence = 0.0,
    this.isLikelyHandwriting = false,
    this.selectedForErase = false,
    this.isManual = false,
    this.baselineVarianceScore = 0.0,
    this.debugBreakdown,
  });

  /// Stable id for this region within its page (used to match ML Kit boxes back to
  /// native heuristic results, and as a Flutter [Key] for the overlay).
  final String id;

  /// Bounding box in the coordinate space of the processed page image (pixels).
  final Rect boundingBox;

  /// Recognized text, empty for a manually-drawn region.
  final String text;

  /// Native heuristic confidence in [0, 1] that this region is handwriting. Not meaningful
  /// for manually-drawn regions (isManual == true), which are always user-intent.
  final double confidence;

  final bool isLikelyHandwriting;

  /// Whether this region is currently checked to be erased. Pre-checked when
  /// [isLikelyHandwriting] is true, but always user-toggleable — the heuristic is best-effort.
  final bool selectedForErase;

  /// True for a region the user drew themselves (not from ML Kit/the heuristic).
  final bool isManual;

  /// How much each word's baseline (bottom edge) wanders relative to the others in the same
  /// line, in [0, 1] — printed text sits on a near-perfectly rigid baseline (near 0), cursive
  /// handwriting visibly undulates (higher). Computed in Dart straight from ML Kit's own
  /// per-word boxes (see TextRecognitionService); more reliable in practice than the native
  /// pixel-level stroke-width heuristic alone, which is blended in as a secondary signal.
  final double baselineVarianceScore;

  /// Debug-only breakdown of every signal that fed into [confidence] — populated by
  /// ImageProcessingService.scoreHandwriting, null otherwise (manual regions, or before
  /// scoring runs). Exists purely so a wrong verdict can be diagnosed from a single screenshot
  /// (which exact signal is driving it) instead of guessed at one heuristic-weight change at a
  /// time — two rounds of blind reweighting this session each fixed one case and broke another.
  final DebugScoreBreakdown? debugBreakdown;

  TextRegion copyWith({
    Rect? boundingBox,
    bool? selectedForErase,
  }) {
    return TextRegion(
      id: id,
      boundingBox: boundingBox ?? this.boundingBox,
      text: text,
      confidence: confidence,
      isLikelyHandwriting: isLikelyHandwriting,
      selectedForErase: selectedForErase ?? this.selectedForErase,
      isManual: isManual,
      baselineVarianceScore: baselineVarianceScore,
      debugBreakdown: debugBreakdown,
    );
  }
}

class DebugScoreBreakdown {
  const DebugScoreBreakdown({
    required this.mlConfidence,
    required this.heuristic,
    required this.angleVariationScore,
    required this.colorDeviation,
    required this.intensityDeviation,
    required this.strokeWidthDeviation,
    required this.hasWideUnderline,
    required this.matchesPageInk,
    required this.cappedByStraightness,
  });

  final double mlConfidence;
  final double heuristic;
  final double angleVariationScore;
  final double colorDeviation;
  final double intensityDeviation;
  final double strokeWidthDeviation;
  final bool hasWideUnderline;
  final bool matchesPageInk;
  final bool cappedByStraightness;
}
