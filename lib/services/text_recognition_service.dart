import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

import '../core/models/text_region.dart';

/// Wraps ML Kit text recognition to get per-word bounding boxes — the candidate regions fed
/// into the handwriting heuristic. Classifying whole LINES doesn't work on real documents:
/// a fill-in-the-blank worksheet routinely has a printed question and a handwritten answer
/// sharing the same line (e.g. "My mother is not happy with ___handwritten word___"), so a
/// line-level score gets diluted/contaminated by whichever content dominates the line. Per-word
/// (ML Kit's `TextElement`) regions avoid that, and as a side benefit only erase the actual
/// handwritten word instead of the whole line's printed text around it.
///
/// ML Kit does not distinguish handwriting from printed text itself; classification happens
/// afterwards from two blended signals: how much each word's baseline deviates from its line's
/// own median (computed here), and a native pixel-level stroke-width heuristic (see
/// ImageProcessingService.scoreHandwriting). A handwritten word inserted into an otherwise
/// printed line commonly sits off-baseline — that per-word deviation is a much stronger, more
/// localized signal than a whole-line average.
class TextRecognitionService {
  final TextRecognizer _recognizer = TextRecognizer(script: TextRecognitionScript.latin);

  /// Normalizes baseline deviation-from-line-median to [0, 1]. A starting point, not tuned
  /// against a labeled dataset yet.
  static const double _baselineDeviationNormalizer = 0.14;

  /// Characters whose lowercase form extends below the baseline (a descender). A word
  /// containing one has a bounding-box bottom that's legitimately lower than a descender-free
  /// word on the exact same baseline — e.g. "young" vs "the". Ignoring this made ALL-CAPS words
  /// (which never have descenders) look artificially baseline-aligned relative to ordinary
  /// lowercase text, and vice versa: real per-word height comparison was dropped entirely for
  /// this same reason (it correlated with letter case, not with handwriting).
  static const _descenderChars = {'g', 'j', 'p', 'q', 'y'};

  Future<List<TextRegion>> recognize(String imagePath) async {
    final inputImage = InputImage.fromFilePath(imagePath);
    final result = await _recognizer.processImage(inputImage);

    final regions = <TextRegion>[];
    var index = 0;
    for (final block in result.blocks) {
      for (final line in block.lines) {
        final elements = line.elements;
        if (elements.isEmpty) continue;

        final lineHeight = line.boundingBox.height;
        // Descender allowance: lowercase descenders typically reach ~25% of line height below
        // the baseline. Subtracting it from descender-word bottoms before comparing puts every
        // word on the same "true baseline" footing regardless of which letters it contains.
        final descenderAllowance = lineHeight * 0.25;
        double compensatedBottom(TextElement e) =>
            e.boundingBox.bottom - (_hasDescender(e.text) ? descenderAllowance : 0.0);

        final medianCompensatedBottom = _median(elements.map(compensatedBottom).toList());

        for (final element in elements) {
          final deviation = lineHeight > 0
              ? (compensatedBottom(element) - medianCompensatedBottom).abs() / lineHeight
              : 0.0;
          regions.add(TextRegion(
            id: 'word_${index++}',
            boundingBox: element.boundingBox,
            text: element.text,
            baselineVarianceScore: (deviation / _baselineDeviationNormalizer).clamp(0.0, 1.0),
          ));
        }
      }
    }
    return regions;
  }

  bool _hasDescender(String text) => text.split('').any((c) => _descenderChars.contains(c));

  double _median(List<double> values) {
    if (values.isEmpty) return 0.0;
    final sorted = [...values]..sort();
    final mid = sorted.length ~/ 2;
    return sorted.length.isOdd ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  void dispose() => _recognizer.close();
}
