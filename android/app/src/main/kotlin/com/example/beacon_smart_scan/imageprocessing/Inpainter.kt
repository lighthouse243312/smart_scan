package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.max
import kotlin.math.min

/** Erases the given regions via OpenCV inpainting (Telea) — fills them in from surrounding pixels. */
object Inpainter {
    // A hand-drawn mark (circling a multiple-choice letter, a checkmark) often extends well
    // beyond the flagged word's own tight box — verified against a real annotated worksheet:
    // erasing only the word's box left the outer ring of the circle behind. Grow the erase
    // region to the actual connected ink blob near the word instead of just its bounding box.
    private const val GROW_SEARCH_PAD = 30
    private const val GROW_DILATE_PX = 7
    private const val GROW_REDNESS_THRESHOLD = 30.0
    // A long diagonal strike-through can connect to marks far outside this one word's own
    // circle — verified: uncapped growth on a dense worksheet swallowed multiple rows via one
    // continuous diagonal stroke, leaving giant inpaint smears. Cap growth to a bounded margin
    // around the seed box (bigger than a typical circle's ~15-20px overshoot, small enough
    // that a stroke merely passing through the search window can't run away with it).
    private const val GROW_MAX_PX = 22

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
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            rects.forEach { rectMap ->
                val rect = ImageIO.mapToClippedRect(rectMap, src.width(), src.height(), padding)
                val grown = growToInkRegion(src, gray, rect)
                Imgproc.rectangle(mask, grown.tl(), grown.br(), Scalar(255.0), -1)
            }
            Photo.inpaint(src, mask, dst, inpaintRadius, Photo.INPAINT_TELEA)
            ImageIO.writeOrThrow(dst, outputPath)
        } finally {
            gray.release()
            mask.release()
            dst.release()
            src.release()
        }
        return mapOf("outputPath" to outputPath)
    }

    /**
     * Grading ink is a distinct color (commonly red) from printed black/gray text. Growing on
     * a color-deviation ("redness") mask — rather than plain darkness — bridges through more
     * of the SAME ink without eating into nearby black print (verified: a plain-darkness grow
     * bridged into unrelated words only ~7px away on a dense worksheet).
     */
    private fun growToInkRegion(colorSrc: Mat, graySrc: Mat, seed: Rect): Rect {
        val sx0 = max(0, seed.x - GROW_SEARCH_PAD)
        val sy0 = max(0, seed.y - GROW_SEARCH_PAD)
        val sx1 = min(colorSrc.width(), seed.x + seed.width + GROW_SEARCH_PAD)
        val sy1 = min(colorSrc.height(), seed.y + seed.height + GROW_SEARCH_PAD)
        if (sx1 <= sx0 || sy1 <= sy0) return seed
        val windowRect = Rect(sx0, sy0, sx1 - sx0, sy1 - sy0)

        val colorWindow = Mat(colorSrc, windowRect)
        val grayWindow = Mat(graySrc, windowRect)
        val channels = ArrayList<Mat>()
        val bg = Mat()
        val redness = Mat()
        val rednessMask = Mat()
        val darkMask = Mat()
        val inkMask = Mat()
        val dilated = Mat()
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        try {
            Core.split(colorWindow, channels)
            val b = channels[0]
            val g = channels[1]
            val r = channels[2]
            Core.addWeighted(b, 0.5, g, 0.5, 0.0, bg)
            Core.subtract(r, bg, redness)
            Core.compare(redness, Scalar(GROW_REDNESS_THRESHOLD), rednessMask, Core.CMP_GT)
            Core.compare(grayWindow, Scalar(220.0), darkMask, Core.CMP_LT)
            Core.bitwise_and(rednessMask, darkMask, inkMask)

            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, org.opencv.core.Size(GROW_DILATE_PX.toDouble(), GROW_DILATE_PX.toDouble()))
            Imgproc.dilate(inkMask, dilated, kernel)

            val numLabels = Imgproc.connectedComponentsWithStats(dilated, labels, stats, centroids, 8, CvType.CV_32S)

            // Always include the original text box; union in every red component found in the
            // search window (the window itself already scopes "near this word").
            var gx0 = seed.x
            var gy0 = seed.y
            var gx1 = seed.x + seed.width
            var gy1 = seed.y + seed.height
            for (label in 1 until numLabels) {
                val lx = stats.get(label, 0)[0].toInt()
                val ly = stats.get(label, 1)[0].toInt()
                val lw = stats.get(label, 2)[0].toInt()
                val lh = stats.get(label, 3)[0].toInt()
                if (lw * lh < 4) continue
                gx0 = min(gx0, sx0 + lx)
                gy0 = min(gy0, sy0 + ly)
                gx1 = max(gx1, sx0 + lx + lw)
                gy1 = max(gy1, sy0 + ly + lh)
            }

            gx0 = max(gx0, seed.x - GROW_MAX_PX)
            gy0 = max(gy0, seed.y - GROW_MAX_PX)
            gx1 = min(gx1, seed.x + seed.width + GROW_MAX_PX)
            gy1 = min(gy1, seed.y + seed.height + GROW_MAX_PX)

            val clippedX0 = max(0, gx0)
            val clippedY0 = max(0, gy0)
            val clippedX1 = min(colorSrc.width(), gx1)
            val clippedY1 = min(colorSrc.height(), gy1)
            return Rect(clippedX0, clippedY0, max(1, clippedX1 - clippedX0), max(1, clippedY1 - clippedY0))
        } finally {
            channels.forEach { it.release() }
            bg.release()
            redness.release()
            rednessMask.release()
            darkMask.release()
            inkMask.release()
            dilated.release()
            labels.release()
            stats.release()
            centroids.release()
            colorWindow.release()
            grayWindow.release()
        }
    }
}
