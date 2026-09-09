package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

/**
 * ML Kit's text recognizer sometimes emits NO region at all for loosely-connected cursive
 * handwriting (verified on a real photo: two lines of a handwritten note got zero boxes, while
 * a third line in clearer, more separated letters was detected fine) — its OCR-based detector
 * has an implicit "is this legible text" confidence gate that messy cursive can fall below.
 * When that happens there is nothing for the per-word classifier to even look at, since it only
 * ever sees ML Kit's own boxes.
 *
 * This finds ink that ML Kit's boxes don't already cover, merges nearby letters/words into
 * phrase-level blobs (bridging normal within-line gaps the same way a human eye groups
 * handwriting into a line), and hands those back as extra candidate regions — the CNN classifier
 * only ever needs a pixel crop, not a transcription, so it can still score them.
 */
object OrphanInkDetector {
    // Bridges within-WORD letter gaps (cursive letters are usually touching or a few px apart)
    // without also bridging the larger word-to-word gap on the same line. Scaling this off the
    // INK'S OWN measured size (typical raw letter-fragment height, before any merging) rather
    // than off the image's pixel width self-calibrates to however the photo was actually framed
    // — verified: a photo of the whole page and a photo zoomed in tight on just the handwriting
    // put the very same real-world pen stroke at wildly different pixel widths, so a radius tied
    // to overall image width was still far too small to bridge cursive letters into words once
    // zoomed in (each letter was already many times wider than the old fixed 20px cap), leaving
    // fragments too small and shapeless for the classifier to read as a word at all. A radius
    // tied to the ink's own on-page size stays meaningful either way.
    private const val MERGE_DILATE_HEIGHT_FRACTION = 0.4
    private const val MERGE_DILATE_MIN_PX = 8
    private const val MERGE_DILATE_MAX_PX = 60
    // Ignore pure noise/dust when measuring typical letter size, but keep the floor low — thin
    // stroke fragments are exactly the samples this measurement needs.
    private const val GLYPH_STAT_MIN_AREA = 20
    private const val MIN_GLYPH_SAMPLES = 3
    // Only used on a page with too few raw ink fragments to measure a reliable typical size
    // (e.g. a single short word) — the original width-based estimate, as a fallback only.
    private const val FALLBACK_MERGE_DILATE_FRACTION = 0.004
    private const val FALLBACK_MERGE_DILATE_MIN_PX = 10
    private const val FALLBACK_MERGE_DILATE_MAX_PX = 20
    // A single stray dot, JPEG artifact, or thin table/gridline segment can pass a small area
    // threshold on its own — require real letter-scale bulk in BOTH dimensions, not just total
    // area (a 3px-tall, 300px-long line has plenty of "area" but is not a word).
    private const val MIN_AREA = 800
    private const val MIN_DIMENSION_PX = 15
    private const val EXISTING_BLOCK_PADDING_PX = 6.0
    // Even with a tighter merge radius, a run-on phrase can still end up wider than any single
    // word the classifier was trained on. Past this width:height ratio, slice it into roughly
    // word-sized, near-square chunks instead of handing the classifier one long, badly-squashed
    // strip — the same reasoning as not classifying whole LINES in the ML-Kit path.
    private const val MAX_ASPECT_RATIO = 2.5

    fun detect(imagePath: String, existingRects: List<Map<String, Any>>): List<Map<String, Any>> {
        val src = ImageIO.readOrThrow(imagePath)
        val gray = Mat()
        val binary = Mat()
        val claimed = Mat()
        val unclaimed = Mat()
        val dilated = Mat()
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.threshold(gray, binary, 0.0, 255.0, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)

            // Blank out anything ML Kit already boxed — only look at ink with nowhere to go.
            claimed.create(binary.size(), CvType.CV_8UC1)
            binary.copyTo(claimed)
            // Padded, not exact — an ML Kit box is accurate but not pixel-perfect down to the
            // ink's own edge; anti-aliased/blurred stroke edges bleed a few px outside it.
            // Verified: with zero padding, a row of tightly-spaced letters (a weekday header
            // "W T F") left thin unclaimed slivers around several ALREADY-correctly-detected
            // letters, which the merge step then fused into one phantom "orphan" blob of
            // jagged edge fragments — scored as handwriting purely because it isn't a real
            // letter shape, even though every character involved was ordinary straight print.
            existingRects.forEach { rectMap ->
                val rect = ImageIO.mapToClippedRect(rectMap, src.width(), src.height(), EXISTING_BLOCK_PADDING_PX)
                Imgproc.rectangle(claimed, rect.tl(), rect.br(), Scalar(0.0), -1)
            }
            unclaimed.create(binary.size(), CvType.CV_8UC1)
            claimed.copyTo(unclaimed)

            val mergeDilatePx = measureMergeDilatePx(unclaimed, src.width())
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(mergeDilatePx.toDouble(), mergeDilatePx.toDouble()))
            Imgproc.dilate(unclaimed, dilated, kernel)

            val numLabels = Imgproc.connectedComponentsWithStats(dilated, labels, stats, centroids, 8, CvType.CV_32S)

            val results = ArrayList<Map<String, Any>>()
            val w = src.width()
            val h = src.height()
            for (label in 1 until numLabels) {
                val area = stats.get(label, 4)[0]
                if (area < MIN_AREA) continue
                val lx = stats.get(label, 0)[0].toInt()
                val ly = stats.get(label, 1)[0].toInt()
                val lw = stats.get(label, 2)[0].toInt()
                val lh = stats.get(label, 3)[0].toInt()
                if (lw < MIN_DIMENSION_PX || lh < MIN_DIMENSION_PX) continue
                // Skip anything spanning half the page or more in either direction — a big
                // scanned graphic/illustration or a merged run of unrelated table gridlines the
                // OCR engine also skipped, not one line of handwriting.
                if (lw > w * 0.5 || lh > h * 0.5) continue

                // Tighten the box back down to the ACTUAL unclaimed ink inside this dilated
                // region, so the merge-dilation doesn't leave a bloated box around the real word.
                val regionRect = Rect(max(0, lx), max(0, ly), min(w - lx, lw), min(h - ly, lh))
                val inkInRegion = Mat(unclaimed, regionRect)
                val tight = tightBoundingBox(inkInRegion)
                inkInRegion.release()
                if (tight == null) continue
                val (tx, ty, tw, th) = tight
                val absLeft = lx + tx
                val absTop = ly + ty

                // Still a run-on phrase (several words the merge step above couldn't cleanly
                // separate) — split it into near-square chunks so each one resembles the single-
                // word crops the classifier was actually trained on, rather than one long strip
                // that gets squashed into an unrecognizable shape at the model's 128x64 input.
                val aspectRatio = tw.toDouble() / max(1, th)
                if (aspectRatio > MAX_ASPECT_RATIO) {
                    val chunkCount = aspectRatio.toInt().coerceAtLeast(2)
                    val chunkWidth = tw / chunkCount
                    for (i in 0 until chunkCount) {
                        val chunkLeft = absLeft + i * chunkWidth
                        val chunkRight = if (i == chunkCount - 1) absLeft + tw else chunkLeft + chunkWidth
                        results.add(
                            mapOf(
                                "id" to "orphan_${label}_$i",
                                "left" to chunkLeft.toDouble(),
                                "top" to absTop.toDouble(),
                                "right" to chunkRight.toDouble(),
                                "bottom" to (absTop + th).toDouble(),
                            )
                        )
                    }
                } else {
                    results.add(
                        mapOf(
                            "id" to "orphan_${label}",
                            "left" to absLeft.toDouble(),
                            "top" to absTop.toDouble(),
                            "right" to (absLeft + tw).toDouble(),
                            "bottom" to (absTop + th).toDouble(),
                        )
                    )
                }
            }
            return results
        } finally {
            gray.release()
            binary.release()
            claimed.release()
            unclaimed.release()
            dilated.release()
            labels.release()
            stats.release()
            centroids.release()
            src.release()
        }
    }

    /** Derives the merge-dilation kernel size from the RAW (undilated) unclaimed ink's own
     * measured letter-fragment height, so it self-calibrates to however the photo was framed —
     * see the class-level doc for why a width-based radius doesn't. Falls back to the old
     * width-based estimate only when there's too little raw ink to measure a reliable typical
     * size from. */
    private fun measureMergeDilatePx(unclaimed: Mat, imageWidth: Int): Int {
        val rawLabels = Mat()
        val rawStats = Mat()
        val rawCentroids = Mat()
        try {
            val rawNumLabels = Imgproc.connectedComponentsWithStats(unclaimed, rawLabels, rawStats, rawCentroids, 8, CvType.CV_32S)
            val heights = ArrayList<Int>()
            for (label in 1 until rawNumLabels) {
                val area = rawStats.get(label, 4)[0]
                if (area < GLYPH_STAT_MIN_AREA) continue
                heights.add(rawStats.get(label, 3)[0].toInt())
            }
            if (heights.size >= MIN_GLYPH_SAMPLES) {
                heights.sort()
                val medianHeight = heights[heights.size / 2]
                return (medianHeight * MERGE_DILATE_HEIGHT_FRACTION).toInt().coerceIn(MERGE_DILATE_MIN_PX, MERGE_DILATE_MAX_PX)
            }
            return (imageWidth * FALLBACK_MERGE_DILATE_FRACTION).toInt()
                .coerceIn(FALLBACK_MERGE_DILATE_MIN_PX, FALLBACK_MERGE_DILATE_MAX_PX)
        } finally {
            rawLabels.release()
            rawStats.release()
            rawCentroids.release()
        }
    }

    private data class TightBox(val x: Int, val y: Int, val w: Int, val h: Int)
    private operator fun TightBox.component1() = x
    private operator fun TightBox.component2() = y
    private operator fun TightBox.component3() = w
    private operator fun TightBox.component4() = h

    private fun tightBoundingBox(binaryInk: Mat): TightBox? {
        val nz = Mat()
        try {
            Core.findNonZero(binaryInk, nz)
            if (nz.empty()) return null
            val rect = Geometry.boundingRect(nz)
            return TightBox(rect.x, rect.y, rect.width, rect.height)
        } finally {
            nz.release()
        }
    }
}
