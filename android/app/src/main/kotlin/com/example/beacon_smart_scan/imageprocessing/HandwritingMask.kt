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

    fun write(handwriting: Mat, print: Mat, overlap: Mat, path: String) {
        val red = Mat.zeros(handwriting.size(), CvType.CV_8UC1)
        val alpha = Mat.zeros(handwriting.size(), CvType.CV_8UC1)
        val blue = Mat()
        val notOverlap = Mat()
        val file = Mat()
        releasing(red, alpha, blue, notOverlap, file) {
            red.setTo(Scalar(255.0), handwriting)
            alpha.setTo(Scalar(OVERLAY_ALPHA), handwriting)
            // overlap kept apart from print so it shows yellow, not white
            Core.bitwise_not(overlap, notOverlap)
            Core.bitwise_and(print, notOverlap, blue)
            Core.merge(listOf(blue, overlap, red, alpha), file)
            ImageIO.writeOrThrow(file, path)
        }
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
