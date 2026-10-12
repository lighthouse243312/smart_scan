package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

/**
 * Shared handwriting-mask helpers — mirrors the "Handwriting mask helpers" section of
 * ios/Runner/ImageProcessingOpenCV.mm so both platforms produce the same mask for a given page.
 *
 * A mask file is a BGRA PNG the same size as the page holding two INDEPENDENT layers:
 *  - alpha > 0   → handwriting (drawn semi-transparent red, so the file doubles as the review
 *                  overlay);
 *  - blue  > 127 → printed ink, stored even where alpha is 0 (invisible in the overlay).
 * A pixel can be both — handwriting written over print — and then shows magenta in the overlay.
 * The erase step repaints handwriting pixels from these layers instead of inpainting, so print
 * crossed by a pen stroke is restored rather than smeared.
 */
object HandwritingMask {
    private const val OVERLAY_ALPHA = 160.0

    // "Ink" = noticeably darker than the local paper — used to snap the model's soft,
    // reduced-scale masks onto the page's actual strokes.
    const val INK_CONTRAST_LOOSE = 8.0

    /** Releases every [mats] once [block] finishes, however it exits. */
    inline fun <T> releasing(vararg mats: Mat, block: () -> T): T {
        try {
            return block()
        } finally {
            mats.forEach { it.release() }
        }
    }

    /**
     * How much darker than the paper each pixel is (0 where lighter). Paper brightness comes from
     * a large median over a 4x-downscaled copy, which wipes out text strokes and keeps shading.
     */
    fun inkContrast(gray: Mat): Mat {
        val small = Mat()
        val paper = Mat()
        return releasing(small, paper) {
            Imgproc.resize(gray, small, Size(), 0.25, 0.25, Imgproc.INTER_AREA)
            val kernel = min(21, (min(small.cols(), small.rows()) - 1) or 1)
            if (kernel >= 3) Imgproc.medianBlur(small, small, kernel)
            Imgproc.resize(small, paper, gray.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            val contrast = Mat()
            Core.subtract(paper, gray, contrast)
            contrast
        }
    }

    /**
     * Isolated specks (sensor noise, paper texture) are never worth erasing; scaled to the image
     * so a 12 MP photo and a small scan drop roughly the same physical size.
     */
    fun minSpeckleArea(image: Mat): Int = max(12, (image.total() * 0.000004).toInt())

    fun removeSmallComponents(mask: Mat, minArea: Int) {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        releasing(labels, stats, centroids) {
            val count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids, 8, CvType.CV_32S)
            val keep = ByteArray(count)
            for (i in 1 until count) {
                val area = stats.get(i, Imgproc.CC_STAT_AREA)[0].toInt()
                keep[i] = if (area >= minArea) 255.toByte() else 0
            }
            val labelData = IntArray(labels.total().toInt())
            labels.get(0, 0, labelData)
            val maskData = ByteArray(labelData.size)
            for (i in labelData.indices) maskData[i] = keep[labelData[i]]
            mask.put(0, 0, maskData)
        }
    }

    /**
     * The layers of a mask file, each CV_8UC1 with 255 = set: handwriting (alpha), print (blue) and
     * overlap = print hidden under handwriting, restored on erase (green — red + green shows it
     * yellow in the overlay). Caller releases them.
     */
    class Layers(val handwriting: Mat, val print: Mat, val overlap: Mat) {
        fun release() {
            handwriting.release()
            print.release()
            overlap.release()
        }
    }

    /** Dev switch (InkHarness `-e consistent 0`): the old print / overlap layers, for comparison. */
    @JvmStatic var consistent = true
    /** Dev switch (InkHarness `-e touch 0`). */
    @JvmStatic var touchCheck = true

    /** Print within this many pixels of handwriting may be its rim (see [strokesTouchingPrint]). */
    private const val RIM_CLEAR = 2

    fun write(handwriting: Mat, print: Mat, overlap: Mat, path: String) {
        val red = Mat.zeros(handwriting.size(), CvType.CV_8UC1)
        val alpha = Mat.zeros(handwriting.size(), CvType.CV_8UC1)
        val blue = Mat()
        val notHw = Mat()
        val under = Mat()
        val file = Mat()
        releasing(red, alpha, blue, notHw, under, file) {
            red.setTo(Scalar(255.0), handwriting)
            alpha.setTo(Scalar(OVERLAY_ALPHA), handwriting)
            // print layer = print with no handwriting on it, pixel by pixel: a pixel both layers
            // claim is print only as print under the pen (the overlap layer, kept apart so it shows
            // yellow, not white). Upstream steps that vote per ink component can leave a pen pixel
            // flagged print too — pen digits fused with a table's grid make one huge mostly-print
            // component — and the erase would then keep and restore it.
            if (consistent) {
                Core.bitwise_not(handwriting, notHw)
                Core.bitwise_and(print, notHw, blue)
                printThroughPen(overlap, blue, handwriting, InkAnalysis.strokeUnit(handwriting.cols(), handwriting.rows()), under)
            } else {
                Core.bitwise_not(overlap, notHw); Core.bitwise_and(print, notHw, blue); overlap.copyTo(under)
            }
            Core.merge(listOf(blue, under, red, alpha), file)
            ImageIO.writeOrThrow(file, path)
        }
    }

    /**
     * Handwriting strokes (connected pieces) with print near them that is not their own rim:
     * pure print more than [RIM_CLEAR] px from any handwriting, within 2 stroke units of the stroke.
     */
    private fun strokesTouchingPrint(purePrint: Mat, handwriting: Mat, k: Int): Mat {
        val rimK = Mat.ones(2 * RIM_CLEAR + 1, 2 * RIM_CLEAR + 1, CvType.CV_8U)
        val near = Mat(); Imgproc.dilate(handwriting, near, rimK)
        val far = Mat(); Core.bitwise_not(near, near); Core.bitwise_and(purePrint, near, far)
        removeSmallComponents(far, k * k)   // specks of noise on the paper are not print
        val reach = Mat(); val reachK = Mat.ones(4 * k + 1, 4 * k + 1, CvType.CV_8U)
        Imgproc.dilate(far, reach, reachK)
        val labels = Mat()
        val n = Imgproc.connectedComponents(handwriting, labels, 8, CvType.CV_32S)
        val w = handwriting.cols(); val h = handwriting.rows()
        val lab = IntArray(w); val rr = ByteArray(w)
        val touches = BooleanArray(n)
        for (y in 0 until h) {
            labels.get(y, 0, lab); reach.get(y, 0, rr)
            for (x in 0 until w) if (lab[x] > 0 && rr[x].toInt() != 0) touches[lab[x]] = true
        }
        val out = Mat(h, w, CvType.CV_8U); val row = ByteArray(w)
        for (y in 0 until h) {
            labels.get(y, 0, lab)
            for (x in 0 until w) row[x] = if (touches[lab[x]]) 255.toByte() else 0
            out.put(y, 0, row)
        }
        listOf(rimK, near, far, reach, reachK, labels).forEach { it.release() }
        return out
    }

    /**
     * [overlap] (print under the pen) kept only where the print runs THROUGH the pen: pure print
     * within 2 stroke units on both sides along the row or along the column, pixel by pixel. A rule
     * or a printed letter a pen crosses has print on either side; the body of a pen digit written
     * on a rule does not — the model's print layer bleeds onto it, and restoring it left dashes and
     * pieces of the digits behind. Mirrors PrintThroughPen in ios/Runner/ImageProcessingOpenCV.mm.
     */
    private fun printThroughPen(overlap: Mat, purePrint: Mat, handwriting: Mat, k: Int, out: Mat) {
        val d = 2 * k
        val sides = listOf(
            Pair(Size((d + 1).toDouble(), 1.0), org.opencv.core.Point(d.toDouble(), 0.0)),   // print to the left
            Pair(Size((d + 1).toDouble(), 1.0), org.opencv.core.Point(0.0, 0.0)),            // ... to the right
            Pair(Size(1.0, (d + 1).toDouble()), org.opencv.core.Point(0.0, d.toDouble())),   // ... above
            Pair(Size(1.0, (d + 1).toDouble()), org.opencv.core.Point(0.0, 0.0)),            // ... below
        ).map { (size, anchor) ->
            val kernel = Mat.ones(size, CvType.CV_8U)
            Mat().also { Imgproc.dilate(purePrint, it, kernel, anchor); kernel.release() }
        }
        val row = Mat(); val col = Mat()
        Core.bitwise_and(sides[0], sides[1], row)
        Core.bitwise_and(sides[2], sides[3], col)
        Core.bitwise_or(row, col, out)
        // ...or it lies on a rule: a long (6 stroke units), thin horizontal run of print and print
        // under the pen — a blank line written along its whole length has no print beside the pen
        // close by, but no digit or letter body is a run that long and thin
        val rule = Mat(); val thick = Mat(); val any = Mat()
        Core.bitwise_or(overlap, purePrint, any)
        val runK = Mat.ones(1, 6 * k + 1, CvType.CV_8U); val thickK = Mat.ones(k, 1, CvType.CV_8U)
        Imgproc.morphologyEx(any, rule, Imgproc.MORPH_OPEN, runK)
        Imgproc.morphologyEx(rule, thick, Imgproc.MORPH_OPEN, thickK)
        Core.bitwise_not(thick, thick); Core.bitwise_and(rule, thick, rule)
        Core.bitwise_or(out, rule, out)
        Core.bitwise_and(out, overlap, out)
        // a pen stroke that touches no print at all (a "1800" alone in an answer box) has nothing
        // under it: the print layer only carries its own soft rim, which would bracket the stroke
        // and bring it back. Kept only on strokes with print clear of the pen within 2 units.
        val touching = if (touchCheck) strokesTouchingPrint(purePrint, handwriting, k) else Mat(out.size(), CvType.CV_8U, Scalar(255.0))
        Core.bitwise_and(out, touching, out)
        (sides + listOf(row, col, rule, thick, any, runK, thickK, touching)).forEach { it.release() }
    }

    fun read(path: String): Layers {
        val raw = Imgcodecs.imread(path, Imgcodecs.IMREAD_UNCHANGED)
        if (raw.empty() || raw.channels() != 4) {
            raw.release()
            throw ImageProcessingException("FILE_NOT_FOUND", "Không đọc được mask tại: $path")
        }
        val alpha = Mat()
        val blue = Mat()
        val green = Mat()
        return releasing(raw, alpha, blue, green) {
            Core.extractChannel(raw, alpha, 3)
            Core.extractChannel(raw, blue, 0)
            Core.extractChannel(raw, green, 1)
            val handwriting = Mat()
            val print = Mat()
            val overlap = Mat()
            Imgproc.threshold(alpha, handwriting, 0.0, 255.0, Imgproc.THRESH_BINARY)
            Imgproc.threshold(blue, print, 127.0, 255.0, Imgproc.THRESH_BINARY)
            Imgproc.threshold(green, overlap, 127.0, 255.0, Imgproc.THRESH_BINARY)
            Layers(handwriting, print, overlap)
        }
    }

    fun coverage(mask: Mat): Double = Core.countNonZero(mask).toDouble() / max(1L, mask.total()).toDouble()

    fun result(maskPath: String, handwriting: Mat): Map<String, Any> =
        mapOf("maskPath" to maskPath, "coverage" to coverage(handwriting))
}
