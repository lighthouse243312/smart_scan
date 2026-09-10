import 'dart:math';

import 'package:flutter/services.dart';

import '../core/channels/image_processing_channel.dart';
import '../core/models/text_region.dart';
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

  /// Rotates by 90°-steps (1 = 90° CW, 2 = 180°, 3 = 90° CCW). Document scanners crop the page
  /// rectangle correctly but don't know which edge is "up" for reading, so this lets the user
  /// fix orientation by hand before text detection runs on it.
  Future<String> rotate(String inputPath, {required int quarterTurnsClockwise}) => _run(() async {
        final outputPath = await TempPaths.next('rotated.png');
        return ImageProcessingChannel.rotate(
          inputPath: inputPath,
          outputPath: outputPath,
          quarterTurnsClockwise: quarterTurnsClockwise,
        );
      });

  /// Classifies each word with the trained CNN (see beacon_smart_scan/ml/) — the primary signal,
  /// since 6+ rounds of pure heuristics hit a hard accuracy ceiling on real worksheet photos.
  /// The native pixel stats (stroke width, ink color, baseline, wide-underline) are still pulled
  /// in as a secondary signal: mostly diluted into the blend, except "sits on a wide pre-printed
  /// blank line" — a structural cue the CNN can't see since it only looks at the word's own crop
  /// — which still acts as a floor. Regions scored as likely-handwriting come back pre-selected
  /// for erase; the user can still toggle any of them.
  Future<List<TextRegion>> scoreHandwriting(String imagePath, List<TextRegion> regions) => _run(() async {
        final mlKitBlocks = regions
            .map((r) => {
                  'left': r.boundingBox.left,
                  'top': r.boundingBox.top,
                  'right': r.boundingBox.right,
                  'bottom': r.boundingBox.bottom,
                })
            .toList();

        // ML Kit sometimes emits no region at all for loosely-connected cursive handwriting —
        // verified on a real photo where two lines of a handwritten note got zero boxes while a
        // clearer third line was detected fine. Find ink it left unclaimed and add those as
        // extra candidate regions before scoring, so this isn't a dead end for the classifier —
        // it only ever needs a pixel crop, not a transcription.
        final orphanMaps = await ImageProcessingChannel.detectOrphanRegions(
          imagePath: imagePath,
          existingBlocks: mlKitBlocks,
        );
        final orphanRegions = orphanMaps.map((m) => TextRegion(
              id: m['id'] as String,
              boundingBox: Rect.fromLTRB(
                (m['left'] as num).toDouble(),
                (m['top'] as num).toDouble(),
                (m['right'] as num).toDouble(),
                (m['bottom'] as num).toDouble(),
              ),
              text: '',
            ));
        final allRegions = [...regions, ...orphanRegions];
        if (allRegions.isEmpty) return allRegions;

        final textBlocks = allRegions
            .map((r) => {
                  'id': r.id,
                  'left': r.boundingBox.left,
                  'top': r.boundingBox.top,
                  'right': r.boundingBox.right,
                  'bottom': r.boundingBox.bottom,
                  'charCount': r.text.replaceAll(RegExp(r'\s'), '').length,
                })
            .toList();

        final results = await Future.wait([
          ImageProcessingChannel.detectHandwritingRegions(imagePath: imagePath, textBlocks: textBlocks),
          ImageProcessingChannel.classifyHandwriting(imagePath: imagePath, textBlocks: textBlocks),
        ]);
        final stats = results[0] as Map<String, NativeWordStats>;
        final mlConfidence = results[1] as Map<String, double>;

        // Stroke width scales with font size — a big bold title and a tiny calendar-grid number
        // can both be perfectly ordinary print, just at very different sizes. Comparing raw
        // pixel stroke widths made the bigger one look like a wild outlier purely because most
        // of a page's words (e.g. a calendar's ~300 date numbers) are small, dragging the
        // reference stroke width down and flagging any large heading as "different style".
        // Normalizing by each word's own height first makes the comparison scale-invariant.
        double normalizedStrokeWidth(NativeWordStats s, TextRegion region) {
          final height = region.boundingBox.height;
          return height > 0 ? s.avgStrokeWidth / height : s.avgStrokeWidth;
        }

        final inkedRegions = allRegions.where((r) => stats[r.id]?.hasInk ?? false).toList();
        // The page's own typical text-line height, at whatever resolution/zoom this particular
        // photo happens to be — a fixed pixel gap can't tell "touching letters" from "two
        // unrelated words three lines apart" once the same physical page is photographed at a
        // very different zoom level (see OrphanInkDetector's merge-dilation fix for the same
        // lesson, hit earlier this session, applied here too).
        final referenceRegionHeight = _median(inkedRegions.map((r) => r.boundingBox.height).toList());
        final referenceStrokeWidth = _median(inkedRegions.map((r) => normalizedStrokeWidth(stats[r.id]!, r)).toList());
        final inked = inkedRegions.map((r) => stats[r.id]!).toList();
        final referenceIntensityStdDev = _median(inked.map((s) => s.inkIntensityStdDev).toList());
        final referenceColor = (
          b: _median(inked.map((s) => s.inkColorB).toList()),
          g: _median(inked.map((s) => s.inkColorG).toList()),
          r: _median(inked.map((s) => s.inkColorR).toList()),
        );

        final scored = allRegions.map((r) {
          final s = stats[r.id];
          if (s == null || !s.hasInk) return r;

          final strokeWidthDeviation = referenceStrokeWidth > 0
              ? ((normalizedStrokeWidth(s, r) - referenceStrokeWidth).abs() / referenceStrokeWidth).clamp(0.0, 1.0)
              : 0.0;
          final colorDistance = _colorDistance(s, referenceColor);
          // 255 * sqrt(3) is the max possible BGR distance; 60 ("noticeably different ink") is
          // a starting normalizer, not tuned against a labeled dataset yet.
          final colorDeviation = (colorDistance / 60.0).clamp(0.0, 1.0);
          // Ink-darkness consistency is one-directional: printed ink IS the low-variance
          // reference, so only being NOTICEABLY MORE uneven than the page's typical print
          // counts (a word steadier than the reference isn't suspicious).
          final intensityRatio = referenceIntensityStdDev > 0
              ? s.inkIntensityStdDev / referenceIntensityStdDev
              : 1.0;
          final intensityDeviation = ((intensityRatio - 1.0)).clamp(0.0, 1.0);

          final heuristic = (0.20 * r.baselineVarianceScore +
                  0.15 * colorDeviation +
                  0.35 * s.angleVariationScore +
                  0.10 * intensityDeviation +
                  0.10 * strokeWidthDeviation +
                  0.10 * s.confidence)
              .clamp(0.0, 1.0);
          // A word-or-phrase crop is reliably WIDER than tall; a single character (a digit, a
          // lone weekday-header letter) is roughly as wide as it is tall or narrower. Computed
          // up front because it also gates how much weight the CNN gets, below.
          final isSplitChunk = RegExp(r'^orphan_\d+_\d+$').hasMatch(r.id);
          final aspectRatio = r.boundingBox.height > 0 ? r.boundingBox.width / r.boundingBox.height : 0.0;
          final isGlyphShaped = aspectRatio > 0 && aspectRatio < 1.3;
          // Only distrust the CNN's shape-based read on a glyph when the ink COLOR also fails to
          // rule out print (verified this needed adding: a real single-character handwriting
          // fragment — colorDeviation 1.00, about as far from the page's print reference as this
          // scale goes — still got under-scored and left unerased, because the blanket CNN-weight
          // cut below applied to every glyph-shaped box regardless of what its ink actually
          // looked like). A color that clearly isn't the page's print ink is real independent
          // evidence of a pen mark that a print font, however unusually shaped, cannot produce.
          final glyphColorMatchesPage = isGlyphShaped && colorDeviation <= 0.35;
          // The trained CNN is the primary signal in general — but it was trained on realistic
          // word/phrase crops, never truly isolated single characters, and it shows: verified on
          // a real calendar whose month numerals (1-12) and weekday-header letters (S M T W T F
          // S) are printed in a casual script-style font — the CNN repeatedly scored individual
          // digits and letters from that font as handwriting with near-1.0 confidence, on shape
          // alone, despite their ink matching the page's own print in every physical measurement
          // (stroke width, color, darkness). Cut the CNN's weight for anything glyph-shaped AND
          // lean more on the heuristic, which is self-calibrated against this exact page's own
          // print rather than a fixed training set that never saw this font's single characters.
          final mlWeight = glyphColorMatchesPage ? 0.35 : 0.75;
          final ml = mlConfidence[r.id] ?? heuristic;
          final weighted = (mlWeight * ml + (1 - mlWeight) * heuristic).clamp(0.0, 1.0);
          // Printed glyphs repeat the exact same stroke angle; a real hand never does — verified
          // directly on a case that fooled the CNN (several calendar date numbers merged by the
          // text recognizer into one garbled multi-digit block, which the CNN read as irregular
          // and thus handwriting-like): angle-variation score came out 0.0-0.3 for those merged-
          // but-still-printed blocks vs a consistent ~1.0 for genuine handwriting in the same
          // photo. That's a wide enough margin to treat a very low score as near-certain print,
          // overriding the CNN — a ceiling, mirroring hasWideUnderline's floor on the other side.
          //
          // Capped to 0.08, not the more conventional-looking 0.2: an orphan region (ML Kit
          // never read it at all) is held to an aggressive 0.15 bar specifically because ML
          // Kit's own failure is itself near-certain evidence of handwriting — verified this cap
          // has to clear THAT bar too, not just the normal 0.5 one, or it does nothing for
          // exactly the orphans it exists to protect: a whole un-detected row of straight-print
          // calendar digits, each one correctly identified as straight print and capped to 0.2,
          // still read as 0.2 > 0.15 and got erased anyway — the cap's own value was silently
          // higher than one of the two bars it needs to sit under.
          // Gated on hasReliableAngleData: with fewer than 2 measurable stroke components (a
          // tiny fragment — one short stroke, a single curl, often the split-off tail end of a
          // word during merging), angleVariationScore is a meaningless 0.0 placeholder, not a
          // measurement of "this is dead straight" — verified: a real handwriting fragment this
          // small got capped to near-zero purely for lacking enough data to measure, not because
          // anything about it actually looked like print.
          final cappedByStraightness = s.hasReliableAngleData && s.angleVariationScore <= 0.15
              ? min(weighted, 0.08)
              : weighted;
          // A second, independent ceiling for a shape the CNN reads as unusual but whose INK
          // physically matches this page's own established print: same stroke thickness, same
          // color, same darkness consistency as every other word already confirmed print on this
          // page. Verified on a real calendar whose template draws its month numbers (1-12) in a
          // casual script-styled print font — the CNN, going by shape alone, occasionally scored
          // one of these (a lone "9") as handwriting even though nothing about the ink itself
          // differed from the rest of the page's print. Angle-variation can't catch this case (a
          // single compact glyph, or a stacked two-size date pair, doesn't repeat a stroke angle
          // the way a whole word does) — but stroke width, color and darkness are direct physical
          // measurements of the ink itself, indifferent to the glyph's shape, and printed ink from
          // the same source (pen or printer) as the rest of the page can't drift far from the
          // page's own reference on all three at once. Real handwriting reliably differs in at
          // least one.
          // Thresholds loosened from an initial 0.15/0.15/0.10 based on a real debug dump: this
          // calendar's month-NAME titles ("July", "September"...) render their letters touching/
          // connected (a script-style print font), which starves angleVariationScore of enough
          // separate contours to be reliable (see hasReliableAngleData) — leaving THIS ceiling as
          // the only one that could still catch them, and it was missing them too. The actual
          // dump showed confirmed real handwriting consistently landing at colorDeviation >= 0.37,
          // while these print titles sat at 0.18-0.35 — a clean gap — with strokeWidthDeviation
          // and intensityDeviation showing similarly clean separation once checked against the
          // same data. Retuned to sit just past the print side of each observed gap.
          // strokeWidthDeviation raised to match colorDeviation's 0.35 after a real dump showed
          // a genuine print title ("July", colorDeviation 0.05 — about as close to the page's
          // print reference as it gets) narrowly missing this ceiling on stroke width alone
          // (0.31, just over the old 0.3), while every confirmed real handwriting sample so far
          // still sits at 0.48+ — comfortably clear of 0.35.
          final matchesPageInk = strokeWidthDeviation <= 0.35 && colorDeviation <= 0.35 && intensityDeviation <= 0.15;
          // Same 0.08 reasoning as cappedByStraightness above — must clear the orphan bar too.
          final cappedByPageInk = matchesPageInk ? min(cappedByStraightness, 0.08) : cappedByStraightness;
          // The real "is there physical evidence this is print" signal — NOT "did capping change
          // the number." Those aren't the same thing: a region whose ml+heuristic blend was
          // ALREADY very low on its own (no cap needed to get it there) has just as much of a
          // print-straightness/page-ink match as one that needed the cap, but "value changed"
          // reads false for it. Verified this exact gap let a confidently-print word (its own
          // angle already <=0.15, needing no capping since its raw score was already near zero)
          // get erased anyway by the neighbor-boost below, which used the old "did it change"
          // proxy to decide whether a region already had print evidence.
          final looksLikePrint = (s.hasReliableAngleData && s.angleVariationScore <= 0.15) || matchesPageInk;
          // A word sitting on a fill-in-blank's own pre-printed line is close to certain to be
          // handwriting — a floor, not just one more diluted vote among many. Also the one signal
          // here the CNN structurally can't see, since it only looks at the word's own crop.
          final blended = s.hasWideUnderline ? max(cappedByPageInk, 0.8) : cappedByPageInk;
          // A region only exists here as an "orphan" because ML Kit's own text recognizer —
          // which read every calendar number and header on the same real page correctly —
          // failed to recognize it as legible text at all. Handwriting is exactly the content
          // that DOESN'T follow a consistent template the way print does, so a legible-text
          // detector failing on it is itself near-certain evidence it's handwriting, not a vote
          // that still needs to clear the same bar as a word ML Kit successfully read. Verified:
          // a genuine handwritten word ("coaching") kept landing just under 0.5 across several
          // rounds of unrelated fixes, because the blend still expected the same certainty a
          // normally-recognized word can produce. Treat orphans as handwriting by default; only
          // a near-total absence of any signal for it (an oddly-shaped noise blob that slipped
          // past the size filters) should override that.
          // The low bar above is earned by ML Kit failing on a whole PHRASE — a multi-character
          // run has nowhere to hide behind "maybe it's just an unusual font." A single isolated
          // glyph doesn't get that same benefit of the doubt: verified on a real calendar whose
          // template draws its month numerals (1-12) in a casual script-style print font — ML
          // Kit failed to read one of them ("9", sitting right against actual handwriting above
          // it), and the 0.15 bar then erased it outright. A lone glyph close to square (roughly
          // as wide as it is tall) is exactly what a single stray character looks like, whereas a
          // real handwritten word or phrase is reliably wider than that — so only a
          // multi-character-shaped orphan gets the aggressive bar; a single-glyph-shaped one is
          // held to the same standard as any ML-Kit-recognized word.
          final isSingleGlyphScale = r.id.startsWith('orphan_') && !isSplitChunk && isGlyphShaped;
          final threshold = r.id.startsWith('orphan_') && !isSingleGlyphScale ? 0.15 : 0.5;
          final isHandwriting = blended > threshold;
          return TextRegion(
            id: r.id,
            boundingBox: r.boundingBox,
            text: r.text,
            confidence: blended,
            isLikelyHandwriting: isHandwriting,
            selectedForErase: isHandwriting,
            isManual: r.isManual,
            baselineVarianceScore: r.baselineVarianceScore,
            debugBreakdown: DebugScoreBreakdown(
              mlConfidence: ml,
              heuristic: heuristic,
              angleVariationScore: s.angleVariationScore,
              colorDeviation: colorDeviation,
              intensityDeviation: intensityDeviation,
              strokeWidthDeviation: strokeWidthDeviation,
              hasWideUnderline: s.hasWideUnderline,
              matchesPageInk: matchesPageInk,
              cappedByStraightness: looksLikePrint,
            ),
          );
        }).toList();

        _unionSplitSiblings(scored);
        _boostSpatialNeighbors(scored, referenceRegionHeight);
        return scored;
      });

  /// A run-on handwritten phrase too wide for one classifier crop gets split into several
  /// narrower chunks natively (see OrphanInkDetector) sharing one id prefix (`orphan_5_0`,
  /// `orphan_5_1`, ...). Each is scored independently, but a low-ink or awkwardly-cut chunk can
  /// individually miss the >0.5 bar even though its siblings clearly don't — verified: exactly
  /// this left small unerased ink fragments scattered through an otherwise-erased handwritten
  /// note. Once ANY sibling reads as handwriting, treat the merely-LOW-scoring ones as part of
  /// the same unit so a slice a coin-flip away from the threshold doesn't leave a gap in the
  /// middle of one phrase.
  ///
  /// Excludes a sibling that was actively CAPPED (cappedByStraightness in its debug breakdown) —
  /// that isn't "just below the bar," it's the straightness/page-ink ceiling having found actual
  /// physical evidence this specific chunk is print, independent of what its neighbors are.
  /// Verified: real handwriting sitting immediately next to unrelated print merged into one
  /// orphan blob (adjacent, not overlapping) split into several chunks — one genuinely
  /// handwritten (high angle variation, correctly flagged) and the rest genuinely straight print
  /// (correctly capped) — and this union used to force-erase the print chunks too just because
  /// they shared a merge blob with a real handwriting neighbor.
  void _unionSplitSiblings(List<TextRegion> regions) {
    final bySplitGroup = <String, List<int>>{};
    for (var i = 0; i < regions.length; i++) {
      final match = RegExp(r'^(orphan_\d+)_\d+$').firstMatch(regions[i].id);
      if (match != null) bySplitGroup.putIfAbsent(match.group(1)!, () => []).add(i);
    }
    for (final indices in bySplitGroup.values) {
      if (indices.any((i) => regions[i].selectedForErase)) {
        for (final i in indices) {
          final region = regions[i];
          final looksLikePrint = region.debugBreakdown?.cappedByStraightness ?? false;
          if (!region.selectedForErase && !looksLikePrint) {
            regions[i] = region.copyWith(selectedForErase: true);
          }
        }
      }
    }
  }

  /// Generalizes [_unionSplitSiblings] from "pieces of the same native merge blob" to ANY two
  /// regions that simply sit next to each other on the page — a human reading the page doesn't
  /// need two letters to have come from the same upstream merge step to see they're part of the
  /// same handwritten word; physical adjacency alone is the evidence. Verified: a single letter
  /// inside a real handwritten word (touching-distance from an already-confirmed handwriting
  /// neighbor, its own angle-variation score showing real irregularity) still got missed because
  /// the CNN, on that one small glyph crop in isolation, read it as print — exactly the class of
  /// single-character crop the CNN was never trained on and is least reliable about (see the
  /// glyph-shaped mlWeight cut above, same root cause, opposite failure direction).
  ///
  /// A region only borrows confidence from a neighbor when it isn't ALREADY confidently print
  /// (cappedByStraightness) — proximity to real handwriting is corroborating context for an
  /// otherwise-ambiguous glyph, not permission to override a ceiling that found actual physical
  /// evidence. This is what keeps it from repeating the exact bug [Inpainter]'s keepRects
  /// protection exists for (real handwriting eating an adjacent, confidently-print digit) — that
  /// protection stays in place regardless, since it acts on selectedForErase after this runs.
  void _boostSpatialNeighbors(List<TextRegion> regions, double referenceRegionHeight) {
    // A fraction of this page's OWN typical text height, not a fixed pixel count — a photo
    // zoomed in tight on the handwriting puts real letter-to-letter gaps at many times more
    // pixels than the same gap in a whole-page shot. Falls back to a small fixed value only if
    // there's no reference at all (e.g. a page with no other detected text to measure from).
    final proximityPx = referenceRegionHeight > 0 ? referenceRegionHeight * 0.4 : 20.0;
    // Only a CONFIRMED neighbor counts — one comfortably past the erase threshold on its own
    // merits, not one that itself only got there via this same boost (which could otherwise
    // chain arbitrarily far across a whole page of touching print).
    const confidentNeighborThreshold = 0.7;
    for (var i = 0; i < regions.length; i++) {
      final region = regions[i];
      if (region.selectedForErase || region.isManual) continue;
      // `cappedByStraightness` only covers the two SPECIFIC ceilings (dead-straight angle, or
      // page-ink match) — a region can fail both of those on a technicality while its own raw
      // blended score already sits confidently near zero (both the CNN and the heuristic
      // independently agreeing it's print). Verified: a region at final confidence 0.06 (ml:0.00,
      // heuristic:0.22) still got erased by this boost, because its angle (0.25) was too high to
      // trip the straightness ceiling and its stroke deviation (0.46) was just over the page-ink
      // ceiling — neither ceiling's specific condition matched, even though the region's own
      // score already said print about as clearly as this blend ever does. A flat confidence
      // floor catches that case without needing a new named ceiling for it.
      final looksLikePrint = (region.debugBreakdown?.cappedByStraightness ?? false) || region.confidence < 0.1;
      if (looksLikePrint) continue;
      final expanded = region.boundingBox.inflate(proximityPx);
      final hasConfidentHandwritingNeighbor = regions.any(
        (other) =>
            !identical(other, region) &&
            other.confidence >= confidentNeighborThreshold &&
            expanded.overlaps(other.boundingBox),
      );
      if (hasConfidentHandwritingNeighbor) {
        regions[i] = region.copyWith(selectedForErase: true);
      }
    }
  }

  double _colorDistance(NativeWordStats s, ({double b, double g, double r}) reference) {
    final db = s.inkColorB - reference.b;
    final dg = s.inkColorG - reference.g;
    final dr = s.inkColorR - reference.r;
    return sqrt(db * db + dg * dg + dr * dr);
  }

  double _median(List<double> values) {
    if (values.isEmpty) return 0.0;
    final sorted = [...values]..sort();
    final mid = sorted.length ~/ 2;
    return sorted.length.isOdd ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  Future<String> eraseRegions(String inputPath, List<Rect> rects, {List<Rect> keepRects = const []}) =>
      _run(() async {
        final outputPath = await TempPaths.next('erased.png');
        return ImageProcessingChannel.eraseRegions(
          inputPath: inputPath,
          outputPath: outputPath,
          rects: rects,
          keepRects: keepRects,
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
