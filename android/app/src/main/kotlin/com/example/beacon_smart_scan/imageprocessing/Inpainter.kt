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
    // Grading/pen ink is COLORED — red, blue, green, whatever pen was on hand — while printed
    // text is black/gray (R≈G≈B, near-zero saturation). A mask built from saturation, not plain
    // darkness, can never include a black-print pixel no matter how close or how it's grown —
    // verified: growing on plain darkness bridged into unrelated words only ~7px away on a dense
    // worksheet, but a color-gated mask left print untouched even where a stroke crosses
    // directly over it. Originally gated on "redness" specifically (this app's first real test
    // photos all happened to use red pen) — verified on a later real photo written in blue ink
    // that redness-only growth found nothing there at all, leaving only each word's own tight
    // box erased. Saturation generalizes to any ink color without needing to special-case each one.
    private const val SATURATION_THRESHOLD = 40.0
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
                paintInkMask(rect, gray, inkMask, labels, stats, numLabels, mask)
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
        val hsv = Mat()
        val hsvChannels = ArrayList<Mat>()
        val satMask = Mat()
        val darkMask = Mat()
        val dilated = Mat()
        try {
            Imgproc.cvtColor(colorSrc, hsv, Imgproc.COLOR_BGR2HSV)
            Core.split(hsv, hsvChannels)
            val saturation = hsvChannels[1]
            Core.compare(saturation, Scalar(SATURATION_THRESHOLD), satMask, Core.CMP_GT)
            Core.compare(graySrc, Scalar(220.0), darkMask, Core.CMP_LT)
            Core.bitwise_and(satMask, darkMask, inkMaskOut)

            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(INK_DILATE_PX.toDouble(), INK_DILATE_PX.toDouble()))
            Imgproc.dilate(inkMaskOut, dilated, kernel)
            return Imgproc.connectedComponentsWithStats(dilated, labels, stats, centroids, 8, CvType.CV_32S)
        } finally {
            hsv.release()
            hsvChannels.forEach { it.release() }
            satMask.release()
            darkMask.release()
            dilated.release()
        }
    }

    /** Paints every colored-ink component near [seed] into [mask] in full (not clipped to a
     * local window), plus [seed]'s own ink specifically — not the whole rectangle solid. A
     * handwritten word's bounding box is axis-aligned but the writing itself rarely is (slanted,
     * uneven letter heights), so a solid rectangle fill reaches into its own corners — verified:
     * this erased a nearby PRINTED word that happened to sit inside a handwriting box's corner
     * but was never actually part of the handwriting's own ink.
     *
     * Within the seed itself, prefer the already-computed COLORED-ink mask over a fresh Otsu
     * darkness threshold: Otsu just splits the crop's own pixels into "darker half" / "lighter
     * half" with no idea which dark pixels are the handwriting and which are a printed word
     * sharing the same crop — verified: a handwriting box that happened to reach right up
     * against an adjacent printed word's edge had Otsu darken both, erasing part of the print.
     * Color can't make that mistake (print isn't saturated). Only fall back to plain darkness
     * when the seed has literally no colored ink at all — composed handwriting glyphs in a color
     * that doesn't stand out from print (graphite pencil, a black pen), the one case color can't
     * help with, which is the reason this fallback exists in the first place. */
    private fun paintInkMask(seed: Rect, gray: Mat, inkMask: Mat, labels: Mat, stats: Mat, numLabels: Int, mask: Mat) {
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

        val clippedSeed = Rect(
            max(0, seed.x),
            max(0, seed.y),
            min(gray.width() - max(0, seed.x), seed.width),
            min(gray.height() - max(0, seed.y), seed.height),
        )
        if (clippedSeed.width <= 0 || clippedSeed.height <= 0) return
        val coloredInkCrop = Mat(inkMask, clippedSeed)
        try {
            if (Core.countNonZero(coloredInkCrop) > 0) {
                val maskRoi = Mat(mask, clippedSeed)
                try {
                    Core.bitwise_or(maskRoi, coloredInkCrop, maskRoi)
                } finally {
                    maskRoi.release()
                }
                return
            }
        } finally {
            coloredInkCrop.release()
        }

        val seedCrop = Mat(gray, clippedSeed)
        val seedDark = Mat()
        try {
            Imgproc.threshold(seedCrop, seedDark, 0.0, 255.0, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)
            val maskRoi = Mat(mask, clippedSeed)
            try {
                Core.bitwise_or(maskRoi, seedDark, maskRoi)
            } finally {
                maskRoi.release()
            }
        } finally {
            seedDark.release()
            seedCrop.release()
        }
    }
}
