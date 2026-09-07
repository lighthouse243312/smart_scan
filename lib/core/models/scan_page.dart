import 'text_region.dart';

/// One page of a scan session, tracking the image path at each processing step so the user
/// can compare before/after and so a step can be re-run without re-doing earlier ones.
class ScanPage {
  ScanPage({
    required this.originalPath,
    this.sharpenedPath,
    this.shadowRemovedPath,
    this.finalPath,
    this.textRegions = const [],
  });

  /// Path straight out of the document-scanner (already cropped/perspective-corrected).
  final String originalPath;

  final String? sharpenedPath;
  final String? shadowRemovedPath;

  /// Path after handwriting erase has been applied; null until the user erases something.
  final String? finalPath;

  final List<TextRegion> textRegions;

  /// Best available image for the current step — falls back through the pipeline.
  String get displayPath =>
      finalPath ?? shadowRemovedPath ?? sharpenedPath ?? originalPath;

  ScanPage copyWith({
    String? sharpenedPath,
    String? shadowRemovedPath,
    String? finalPath,
    List<TextRegion>? textRegions,
  }) {
    return ScanPage(
      originalPath: originalPath,
      sharpenedPath: sharpenedPath ?? this.sharpenedPath,
      shadowRemovedPath: shadowRemovedPath ?? this.shadowRemovedPath,
      finalPath: finalPath ?? this.finalPath,
      textRegions: textRegions ?? this.textRegions,
    );
  }
}
