package com.example.beacon_smart_scan.imageprocessing

import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Colour-of-ink method: saturated (blue/red/green…) ink that is clearly darker than the paper.
 * Fast and offline, but cannot see black ink or pencil, and also catches coloured print. The
 * print layer is estimated (unsaturated ink, plus very dark pixels inside a coloured stroke).
 * Mirrors `inkColorMaskAtPath` in ios/Runner/ImageProcessingOpenCV.mm.
 */
object InkColorMasker {
    /** [minSaturation] is on OpenCV's 0-255 HSV saturation scale. */
    fun mask(inputPath: String, maskPath: String, minSaturation: Double): Map<String, Any> {
        val src = ImageIO.readOrThrow(inputPath)
        val gray = Mat()
        val hsv = Mat()
        val saturation = Mat()
        val value = Mat()
        val colored = Mat()
        val inkStrict = Mat()
        val inkLoose = Mat()
        val grown = Mat()
        val handwriting = Mat()
        val unsaturated = Mat()
        val print = Mat()
        val printUnder = Mat()
        return releasing(
            src, gray, hsv, saturation, value, colored, inkStrict, inkLoose, grown,
            handwriting, unsaturated, print, printUnder,
        ) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.cvtColor(src, hsv, Imgproc.COLOR_BGR2HSV)
            Core.extractChannel(hsv, saturation, 1)
            Core.extractChannel(hsv, value, 2)
            val contrast = HandwritingMask.inkContrast(gray)
            releasing(contrast) {
                // Seed: saturated pixels that are also real ink (not a tinted paper area).
                Imgproc.threshold(saturation, colored, minSaturation, 255.0, Imgproc.THRESH_BINARY)
                Imgproc.threshold(contrast, inkStrict, HandwritingMask.INK_CONTRAST_STRICT, 255.0, Imgproc.THRESH_BINARY)
                Imgproc.threshold(contrast, inkLoose, HandwritingMask.INK_CONTRAST_LOOSE, 255.0, Imgproc.THRESH_BINARY)
            }
            Core.bitwise_and(colored, inkStrict, colored)
            HandwritingMask.removeSmallComponents(colored, HandwritingMask.minSpeckleArea(src))

            // A stroke's anti-aliased rim is less saturated than its core, so grow the seed a
            // little — but only onto pixels that are ink, never onto paper.
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
            releasing(kernel) {
                Imgproc.dilate(colored, grown, kernel, Point(-1.0, -1.0), 2)
            }
            Core.bitwise_and(grown, inkLoose, grown)
            Core.bitwise_or(colored, grown, handwriting)

            // Print layer (no model here, so estimated): unsaturated ink OUTSIDE the strokes, plus
            // pixels inside a stroke dark enough to be that stroke crossing black print. A
            // stroke's own anti-aliased rim is unsaturated too, so unsaturated ink inside the
            // stroke must not count — it would be "restored" as a grey outline of the stroke.
            Imgproc.threshold(saturation, unsaturated, minSaturation, 255.0, Imgproc.THRESH_BINARY_INV)
            Core.bitwise_and(inkStrict, unsaturated, print)
            Core.bitwise_not(handwriting, grown)
            Core.bitwise_and(print, grown, print)
            Imgproc.threshold(value, printUnder, HandwritingMask.PRINT_UNDER_MAX_VALUE - 1, 255.0, Imgproc.THRESH_BINARY_INV)
            Core.bitwise_and(printUnder, handwriting, printUnder)
            Core.bitwise_or(print, printUnder, print)

            HandwritingMask.write(handwriting, print, maskPath)
            HandwritingMask.result(maskPath, handwriting)
        }
    }
}
