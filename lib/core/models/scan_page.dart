/// One page of a scan session, tracking the image path at each processing step so the user
/// can compare before/after and so a step can be re-run without re-doing earlier ones.
class ScanPage {
  ScanPage({
    required this.originalPath,
    this.sharpenedPath,
    this.shadowRemovedPath,
    this.maskPath,
    this.maskCoverage = 0,
    this.finalPath,
  });

  /// Path straight out of the document-scanner (already cropped/perspective-corrected).
  final String originalPath;

  final String? sharpenedPath;
  final String? shadowRemovedPath;

  /// Handwriting mask for [cleanPath] (alpha = handwriting); null until detection has run.
  final String? maskPath;

  /// Fraction of the page covered by [maskPath], 0-1.
  final double maskCoverage;

  /// Path after handwriting erase has been applied; null until the user erases something.
  final String? finalPath;

  /// The processed page BEFORE any erase — what handwriting detection and erase both run on.
  String get cleanPath => shadowRemovedPath ?? sharpenedPath ?? originalPath;

  /// Best available image for the current step — falls back through the pipeline.
  String get displayPath => finalPath ?? cleanPath;

  ScanPage copyWith({
    String? sharpenedPath,
    String? shadowRemovedPath,
    String? maskPath,
    double? maskCoverage,
    String? finalPath,
  }) {
    return ScanPage(
      originalPath: originalPath,
      sharpenedPath: sharpenedPath ?? this.sharpenedPath,
      shadowRemovedPath: shadowRemovedPath ?? this.shadowRemovedPath,
      maskPath: maskPath ?? this.maskPath,
      maskCoverage: maskCoverage ?? this.maskCoverage,
      finalPath: finalPath ?? this.finalPath,
    );
  }
}
