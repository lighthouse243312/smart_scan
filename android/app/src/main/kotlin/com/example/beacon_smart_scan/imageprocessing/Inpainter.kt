package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.max
import kotlin.math.min

/** Erases the given regions via OpenCV inpainting (Telea) — fills them in from surrounding pixels. */
object Inpainter {
    // Grading ink is a distinct color (commonly red) from printed black/gray text. A mask built
    // from color deviation ("redness"), not plain darkness, can never include a black-print
    // pixel no matter how close or how it's grown — verified: growing on plain darkness bridged
    // into unrelated words only ~7px away on a dense worksheet, but a color-gated mask left
    // print untouched even where a stroke crosses directly over it.
    private const val REDNESS_THRESHOLD = 30.0
    private const val INK_DILATE_PX = 5
    // How close a colored-ink connected component must be to a flagged word's box to count as
    // "its" mark. Since inclusion is gated by color, not distance, there is no risk of ever
    // marking a black-print pixel this way — so the WHOLE component is taken once any part of
    // it is this close, however far the component itself runs (a long diagonal strike-through
    // can extend 100px+ from the word it crosses out; clipping the mask to a fixed-size window
    // around the word left such strokes half-erased).
    private const val PROXIMITY_PX = 20

    fun erase(
        inputPath: String,
        outputPath: String,
        rects: List<Map<String, Any>>,
        padding: Double,
        inpaintRadius: Double,
    ): Map<String, Any> {
        val src = ImageIO.readOrThrow(inputPath)
        val gray = Mat()
        val mask = Mat.zeros(src.size(), CvType.CV_8UC1)
        val dst = Mat()
        val inkMask = Mat()
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            val numLabels = buildInkComponents(src, gray, inkMask, labels, stats, centroids)

            rects.forEach { rectMap ->
                val rect = ImageIO.mapToClippedRect(rectMap, src.width(), src.height(), padding)
                paintInkMask(rect, labels, stats, numLabels, mask)
            }
            Photo.inpaint(src, mask, dst, inpaintRadius, Photo.INPAINT_TELEA)
            ImageIO.writeOrThrow(dst, outputPath)
        } finally {
            inkMask.release()
            labels.release()
            stats.release()
            centroids.release()
            gray.release()
            mask.release()
            dst.release()
            src.release()
        }
        return mapOf("outputPath" to outputPath)
    }

    /** Computes the whole-page colored-ink mask and its connected components ONCE, reused for
     * every flagged word (cheaper than re-deriving a local mask per word, and is what lets a
     * component's full extent be found regardless of which word ends up near which part of it). */
    private fun buildInkComponents(colorSrc: Mat, graySrc: Mat, inkMaskOut: Mat, labels: Mat, stats: Mat, centroids: Mat): Int {
        val channels = ArrayList<Mat>()
        val bg = Mat()
        val redness = Mat()
        val rednessMask = Mat()
        val darkMask = Mat()
        val dilated = Mat()
        try {
            Core.split(colorSrc, channels)
            val b = channels[0]
            val g = channels[1]
            val r = channels[2]
            Core.addWeighted(b, 0.5, g, 0.5, 0.0, bg)
            Core.subtract(r, bg, redness)
            Core.compare(redness, Scalar(REDNESS_THRESHOLD), rednessMask, Core.CMP_GT)
            Core.compare(graySrc, Scalar(220.0), darkMask, Core.CMP_LT)
            Core.bitwise_and(rednessMask, darkMask, inkMaskOut)

            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(INK_DILATE_PX.toDouble(), INK_DILATE_PX.toDouble()))
            Imgproc.dilate(inkMaskOut, dilated, kernel)
            return Imgproc.connectedComponentsWithStats(dilated, labels, stats, centroids, 8, CvType.CV_32S)
        } finally {
            channels.forEach { it.release() }
            bg.release()
            redness.release()
            rednessMask.release()
            darkMask.release()
            dilated.release()
        }
    }

    /** Paints every colored-ink component near [seed] into [mask] in full (not clipped to a
     * local window), plus the seed's own tight box (handles plain composed handwriting glyphs,
     * e.g. fill-in-blank answers, which aren't a distinct color from print). */
    private fun paintInkMask(seed: Rect, labels: Mat, stats: Mat, numLabels: Int, mask: Mat) {
        val sx0 = max(0, seed.x - PROXIMITY_PX)
        val sy0 = max(0, seed.y - PROXIMITY_PX)
        val sx1 = min(labels.width(), seed.x + seed.width + PROXIMITY_PX)
        val sy1 = min(labels.height(), seed.y + seed.height + PROXIMITY_PX)
        if (sx1 > sx0 && sy1 > sy0) {
            val window = Mat(labels, Rect(sx0, sy0, sx1 - sx0, sy1 - sy0))
            val nearbyLabels = HashSet<Int>()
            try {
                val flat = IntArray(window.rows() * window.cols())
                window.get(0, 0, flat)
                for (v in flat) if (v != 0) nearbyLabels.add(v)
            } finally {
                window.release()
            }
            for (label in nearbyLabels) {
                if (label < 1 || label >= numLabels) continue
                if (stats.get(label, 2)[0] * stats.get(label, 3)[0] < 4) continue
                val componentMask = Mat()
                try {
                    Core.compare(labels, Scalar(label.toDouble()), componentMask, Core.CMP_EQ)
                    Core.bitwise_or(mask, componentMask, mask)
                } finally {
                    componentMask.release()
                }
            }
        }
        Imgproc.rectangle(mask, seed.tl(), seed.br(), Scalar(255.0), -1)
    }
}
