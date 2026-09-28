/// How the handwriting mask is detected. Both produce a per-pixel mask that the same erase step
/// consumes, so they can be compared on the same page.
enum HandwritingMethod {
  /// Saturated (blue/red/green…) ink darker than the paper — fast and offline, but blind to black
  /// ink and pencil, and also catches coloured print.
  inkColor,

  /// The bundled InkSegmenter U-Net classifies every pixel as paper / print / handwriting —
  /// handles black ink and pencil, costs a model run (still offline).
  segmentation,
}

/// One manual brush correction on the mask, in image pixel coordinates.
class MaskStroke {
  const MaskStroke({required this.points, required this.width, required this.erase});

  /// Flattened `[x0, y0, x1, y1, ...]`.
  final List<double> points;
  final double width;

  /// `true` removes the stroke's area from the mask (keep that ink), `false` adds it (erase it).
  final bool erase;

  Map<String, Object> toMap() => {'points': points, 'width': width, 'erase': erase};
}
