package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Rect
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Best-effort "is this text block handwritten?" per-word measurement. This returns raw
 * per-word stats — it does NOT decide handwriting vs print itself. Dart (ImageProcessingService)
 * makes that call by comparing every word's stats against the PAGE'S OWN most-common values
 * (a self-calibrating "reference style/color" instead of a fixed threshold): most of a page is
 * printed text sharing one font and one ink color, so whichever words deviate from that
 * majority — different stroke width, different ink color, different baseline (see
 * TextRecognitionService) — are the handwriting candidates.
 *
 * Two native pixel-level stats are measured here, based on the Stroke Width Transform
 * text-classification idea (Epshtein et al.): printed fonts render every character with the
 * same stroke width by design, while handwriting's stroke width drifts with pen pressure/speed.
 * The key fix over a naive per-pixel measurement: aggregate stroke width **per connected
 * component** (per character/glyph cluster) first, then look at variance ACROSS components —
 * not across every ink pixel pooled together, which mixes in each letter's own thick joints and
 * corners and drowns out the real signal (verified empirically: pooling all pixels scored a
 * clean printed line *higher* than actual handwriting).
 */
object HandwritingDetector {
    private const val MIN_INK_PIXELS = 20
    private const val MIN_COMPONENT_AREA = 3
    private const val DARK_PIXEL_THRESHOLD = 150

    data class RegionStats(
        val strokeVariationScore: Double,
        val componentRatioScore: Double,
        val angleVariationScore: Double,
        val avgStrokeWidth: Double,
        val inkColorB: Double,
        val inkColorG: Double,
        val inkColorR: Double,
        val inkIntensityStdDev: Double,
        val hasWideUnderline: Boolean,
        // False when there weren't even 2 qualifying stroke components to compare angles across
        // (a tiny fragment — one short stroke, a single curl) — angleVariationScore is then a
        // meaningless 0.0 placeholder, NOT a measurement of "this is dead straight." Dart's
        // straightness ceiling must see this to avoid treating "no data" the same as "definitely
        // print" — verified: a real handwriting fragment this small (the tail end of a word,
        // split off during merging) got angle 0.0 from having only one stroke to look at, and
        // was capped to a near-zero score as if it were confidently straight print.
        val hasReliableAngleData: Boolean,
        val hasInk: Boolean,
    )

    fun detect(imagePath: String, textBlocks: List<Map<String, Any>>): List<Map<String, Any>> {
        val src = ImageIO.readOrThrow(imagePath)
        val gray = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            return textBlocks.map { block ->
                val id = block["id"] as? String ?: ""
                val charCount = (block["charCount"] as? Number)?.toInt()?.coerceAtLeast(1) ?: 1
                val rect = ImageIO.mapToClippedRect(block, gray.width(), gray.height())
                val stats = scoreRegion(src, gray, rect, charCount)
                mapOf(
                    "id" to id,
                    // Native-only fallback confidence (stroke shape + component count), used if
                    // the page-relative signals in Dart have nothing to compare against.
                    "confidence" to (0.5 * stats.strokeVariationScore + 0.5 * stats.componentRatioScore),
                    "angleVariationScore" to stats.angleVariationScore,
                    "avgStrokeWidth" to stats.avgStrokeWidth,
                    "inkColorB" to stats.inkColorB,
                    "inkColorG" to stats.inkColorG,
                    "inkColorR" to stats.inkColorR,
                    "inkIntensityStdDev" to stats.inkIntensityStdDev,
                    "hasWideUnderline" to stats.hasWideUnderline,
                    "hasReliableAngleData" to stats.hasReliableAngleData,
                    "hasInk" to stats.hasInk,
                )
            }
        } finally {
            gray.release()
            src.release()
        }
    }

    private fun scoreRegion(src: Mat, gray: Mat, rect: Rect, charCount: Int): RegionStats {
        val colorCrop = Mat(src, rect)
        val crop = Mat(gray, rect)
        val binary = Mat()
        try {
            Imgproc.threshold(crop, binary, 0.0, 255.0, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)
            if (Core.countNonZero(binary) < MIN_INK_PIXELS) {
                // too little ink in this box to say anything meaningful
                return RegionStats(
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    hasWideUnderline = false, hasReliableAngleData = false, hasInk = false,
                )
            }

            val (strokeWidthsPerComponent, componentCount) = perComponentStrokeWidths(binary)
            val avgStrokeWidth = if (strokeWidthsPerComponent.isEmpty()) 0.0 else strokeWidthsPerComponent.average()
            val strokeScore = strokeVariationAcrossComponentsScore(strokeWidthsPerComponent)
            val ratioScore = componentCountRatioScore(componentCount, charCount)
            val (angleScore, hasReliableAngleData) = componentAngleVariationScore(binary)
            val inkColor = averageInkColor(colorCrop, binary)
            val inkIntensityStdDev = inkIntensityStdDev(crop, binary)
            val hasWideUnderline = hasWideUnderlineBelow(gray, rect)

            return RegionStats(
                strokeVariationScore = strokeScore,
                componentRatioScore = ratioScore,
                angleVariationScore = angleScore,
                avgStrokeWidth = avgStrokeWidth,
                inkColorB = inkColor[0],
                inkColorG = inkColor[1],
                inkColorR = inkColor[2],
                inkIntensityStdDev = inkIntensityStdDev,
                hasWideUnderline = hasWideUnderline,
                hasReliableAngleData = hasReliableAngleData,
                hasInk = true,
            )
        } finally {
            binary.release()
            crop.release()
            colorCrop.release()
        }
    }

    /** Mean BGR of the ink pixels (the actual glyph strokes, not the paper background). */
    private fun averageInkColor(colorCrop: Mat, binaryInk: Mat): DoubleArray {
        val mean = Core.mean(colorCrop, binaryInk)
        return doubleArrayOf(mean.`val`[0], mean.`val`[1], mean.`val`[2])
    }

    /**
     * How much pixel darkness varies within the ink itself. Printed toner/ink lays down at a
     * near-uniform density, so its ink pixels cluster tightly around one dark value; pen ink
     * varies with pressure/speed/flow (skips, fades, presses darker), spreading that value out.
     */
    private fun inkIntensityStdDev(grayCrop: Mat, binaryInk: Mat): Double {
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        try {
            Core.meanStdDev(grayCrop, mean, stddev, binaryInk)
            return stddev.toArray().firstOrNull() ?: 0.0
        } finally {
            mean.release()
            stddev.release()
        }
    }

    /**
     * A fill-in-the-blank answer is written ON TOP of a pre-printed blank line — the ink usually
     * touches or overlaps it, not sitting cleanly above it with a gap — and that line is wider
     * than the answer itself (the blank was sized for a guessed-longer answer). A printed word's
     * own underline (used for in-text emphasis) hugs the word tightly instead. So: search a band
     * spanning from partway UP INSIDE the word's own box down through a generous margin below it
     * (covering both "line touches the ink" and "line has a small gap"), and look for a long,
     * near-solid dark horizontal run spanning noticeably wider than the word's own box. A fixed
     * darkness threshold is used instead of a fresh Otsu computation — Otsu on a thin, almost-
     * entirely-blank strip (a few dark line pixels among mostly paper) is not a reliable split.
     */
    private fun hasWideUnderlineBelow(gray: Mat, rect: Rect): Boolean {
        val marginX = (rect.width * 0.6).toInt().coerceAtLeast(4)
        val bandLeft = (rect.x - marginX).coerceAtLeast(0)
        val bandRight = (rect.x + rect.width + marginX).coerceAtMost(gray.width())
        val bandWidth = bandRight - bandLeft
        if (bandWidth <= 0) return false

        val bandTop = (rect.y + (rect.height * 0.6).toInt()).coerceIn(0, gray.height() - 1)
        val bandBottom = (rect.y + (rect.height * 1.6).toInt()).coerceAtMost(gray.height())
        if (bandBottom <= bandTop) return false

        val band = Mat(gray, Rect(bandLeft, bandTop, bandWidth, bandBottom - bandTop))
        try {
            val rows = band.rows()
            val cols = band.cols()
            val flat = ByteArray(rows * cols)
            band.get(0, 0, flat)

            val minRunLength = rect.width * 1.2
            var longestRun = 0
            var qualifyingRows = 0
            for (y in 0 until rows) {
                var currentRun = 0
                var rowLongestRun = 0
                val rowOffset = y * cols
                for (x in 0 until cols) {
                    // Fixed threshold, not Otsu: this strip is mostly blank paper with a thin
                    // dark line, too skewed a histogram for Otsu to split reliably.
                    if ((flat[rowOffset + x].toInt() and 0xFF) < DARK_PIXEL_THRESHOLD) {
                        currentRun++
                        if (currentRun > rowLongestRun) rowLongestRun = currentRun
                    } else {
                        currentRun = 0
                    }
                }
                if (rowLongestRun > longestRun) longestRun = rowLongestRun
                if (rowLongestRun >= minRunLength) qualifyingRows++
            }
            // A page-wide printed divider/rule (a header underline, a section separator) passes
            // right through this local band exactly like a fill-in-blank's own underline would —
            // verified: a calendar's title-divider rule forced a printed "2023 Calendar" header
            // to a floor of 0.8 this way. The distinguishing fact is absolute scale: a rule line
            // spans nearly the whole page regardless of which word happens to sit near it; a
            // real answer blank is sized for one answer and is always far short of that.
            //
            // A genuine ruled line is thin but SOLID — every row it passes through has the same
            // long dark run, because it's one continuous stroke. A dense row of separate print
            // glyphs (a dictionary's tightly-kerned digits: a calendar's date grid) can, on ONE
            // lucky scanline through several characters' mid-bodies, coincidentally produce a
            // single long run too — verified: an entire un-detected row of calendar dates,
            // treated as one merged "orphan" blob and split into per-character chunks, had this
            // fire on nearly every chunk purely because the row of digits below happened to align
            // that way on one scanline, even though the chunks' own strokes were perfectly
            // straight print. Requiring the long run to persist across MOST of the band's rows —
            // not just its single best one — keeps the real ruled-line case (solid on every row)
            // while rejecting a row of glyphs (long on at most a couple of coincidental rows).
            val sustained = qualifyingRows >= (rows * 0.6).coerceAtLeast(2.0)
            return sustained && longestRun < gray.width() * 0.7
        } finally {
            band.release()
        }
    }

    /**
     * How much each letter's own tilt varies from the next, WITHIN this one word — a signal
     * intrinsic to the ink shape itself, independent of position/underline/color, so it still
     * fires on handwriting that isn't sitting on a fill-in-blank line. A printed font renders
     * the exact same glyph outline every time a letter repeats, so every stroke sits at the same
     * angle across the whole word; a human hand never repeats a stroke at a perfectly identical
     * angle twice.
     *
     * Deliberately NOT filtered to "tall" strokes only: the fill-in-blank answers on a real
     * worksheet are mostly short 2-4 letter words ("he", "it", "us", "they"...), which often
     * have zero or one component tall enough to pass a height filter — that filter silently
     * starved this signal on exactly the majority case. Using every component with a minimum
     * area instead means even a two-letter word usually has enough data points.
     */
    private fun componentAngleVariationScore(binaryInk: Mat): Pair<Double, Boolean> {
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        val angleDeviations = ArrayList<Double>()
        try {
            Imgproc.findContours(binaryInk, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            for (contour in contours) {
                val points = contour.toArray()
                if (points.size >= 5 && Geometry.contourArea(contour) >= MIN_COMPONENT_AREA) {
                    val contour2f = MatOfPoint2f(*points)
                    val angle = Geometry.minAreaRect(contour2f).angle
                    contour2f.release()
                    // minAreaRect's angle is ambiguous mod 90° (which side is "width" flips it);
                    // fold into "deviation from the nearest axis" in [0, 45] so that's comparable.
                    val mod90 = ((angle % 90.0) + 90.0) % 90.0
                    angleDeviations.add(min(mod90, 90.0 - mod90))
                }
                contour.release()
            }
        } finally {
            hierarchy.release()
        }
        if (angleDeviations.size < 2) return 0.0 to false
        val mean = angleDeviations.average()
        val variance = angleDeviations.sumOf { (it - mean) * (it - mean) } / angleDeviations.size
        val stdDev = sqrt(variance)
        // Degrees; not tuned against a labeled dataset yet.
        return (stdDev / 12.0).coerceIn(0.0, 1.0) to true
    }

    /** Returns (one mean stroke-width value per component, total valid component count). */
    private fun perComponentStrokeWidths(binaryInk: Mat): Pair<List<Double>, Int> {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val distance = Mat()
        try {
            val numLabels = Imgproc.connectedComponentsWithStats(binaryInk, labels, stats, centroids, 8, CvType.CV_32S)
            Imgproc.distanceTransform(binaryInk, distance, Geometry.DIST_L2, 3)

            val pixelCount = binaryInk.rows() * binaryInk.cols()
            val labelFlat = IntArray(pixelCount)
            labels.get(0, 0, labelFlat)
            val distFlat = FloatArray(pixelCount)
            distance.get(0, 0, distFlat)

            val sumByLabel = DoubleArray(numLabels)
            val countByLabel = IntArray(numLabels)
            for (i in 0 until pixelCount) {
                val label = labelFlat[i]
                if (label != 0) { // 0 == background
                    sumByLabel[label] += distFlat[i].toDouble()
                    countByLabel[label]++
                }
            }

            val strokeWidths = ArrayList<Double>()
            var validComponents = 0
            for (label in 1 until numLabels) {
                if (countByLabel[label] >= MIN_COMPONENT_AREA) {
                    validComponents++
                    strokeWidths.add(2.0 * sumByLabel[label] / countByLabel[label])
                }
            }
            return Pair(strokeWidths, validComponents)
        } finally {
            labels.release()
            stats.release()
            centroids.release()
            distance.release()
        }
    }

    private fun strokeVariationAcrossComponentsScore(strokeWidthsPerComponent: List<Double>): Double {
        if (strokeWidthsPerComponent.size < 2) return 0.0
        val mean = strokeWidthsPerComponent.average()
        val variance = strokeWidthsPerComponent.sumOf { (it - mean) * (it - mean) } / strokeWidthsPerComponent.size
        val stdDev = sqrt(variance)
        val variationRatio = stdDev / (mean + 1e-6)
        return (variationRatio / 0.5).coerceIn(0.0, 1.0)
    }

    /** Fewer components than characters (letters visually joined) suggests cursive handwriting. */
    private fun componentCountRatioScore(componentCount: Int, charCount: Int): Double {
        val ratio = componentCount.toDouble() / charCount
        return (1.0 - ratio).coerceIn(0.0, 1.0)
    }
}
