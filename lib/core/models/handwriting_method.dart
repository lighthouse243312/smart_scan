/// How the handwriting mask is detected. Both produce a per-pixel mask that the same erase step
/// consumes, so they can be compared on the same page.
enum HandwritingMethod {
  /// Ink colour (optical density) against the local print colour, backed by page layout (print
  /// forms regular lines) — offline, works for pen ink that differs even slightly from the print
  /// (e.g. blue-black ballpoint); cannot separate pen ink identical in colour to the print.
  inkColor,

  /// The bundled InkSegmenter U-Net COMBINED with [inkColor]: handwriting wherever either finds
  /// it, print only where both agree — the model covers pen ink the same colour as the print
  /// (black ballpoint), colour covers neat writing the model misses. Costs a model run (offline).
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
