package com.example.beacon_smart_scan.imageprocessing

import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Removes the handwriting by rebuilding it, not inpainting: inside the erased area, print pixels
 * get the colour of the surrounding print (so a pen stroke crossing a word leaves the word's
 * strokes intact) and everything else gets the paper colour. Outside the area the page is
 * untouched. Mirrors `eraseWithMaskAtPath` in ios/Runner/ImageProcessingOpenCV.mm.
 */
object MaskEraser {
    fun erase(inputPath: String, maskPath: String, outputPath: String, dilatePx: Double): Map<String, Any> {
        val src = ImageIO.readOrThrow(inputPath)
        val layers = HandwritingMask.read(maskPath)
        val handwriting = layers.handwriting
        val print = layers.print
        val area = Mat()
        val grown = Mat()
        val notPrint = Mat()
        val restorePrint = Mat()
        val paperArea = Mat()
        val notArea = Mat()
        val visiblePrint = Mat()
        val dst = Mat()
        try {
            releasing(area, grown, notPrint, restorePrint, paperArea, notArea, visiblePrint, dst) {
                if (handwriting.size() != src.size()) {
                    Imgproc.resize(handwriting, handwriting, src.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
                    Imgproc.resize(print, print, src.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
                }

                // Grow over the stroke's faint rim — but only onto non-print pixels, so the growth
                // can never eat into print the stroke merely passes next to.
                handwriting.copyTo(area)
                val grow = dilatePx.roundToInt()
                if (grow > 0) {
                    val kernel = Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE, Size(2.0 * grow + 1, 2.0 * grow + 1),
                    )
                    releasing(kernel) { Imgproc.dilate(handwriting, grown, kernel) }
                    Core.bitwise_not(print, notPrint)
                    Core.bitwise_and(grown, notPrint, grown)
                    Core.bitwise_or(area, grown, area)
                }

                Core.bitwise_and(area, print, restorePrint)
                Core.bitwise_xor(area, restorePrint, paperArea)
                Core.bitwise_not(area, notArea)
                Core.bitwise_and(print, notArea, visiblePrint)

                src.copyTo(dst)
                val paperColor = estimatePaperColor(src)
                val printColor = estimatePrintColor(src, visiblePrint)
                releasing(paperColor, printColor) {
                    paperColor.copyTo(dst, paperArea)
                    printColor.copyTo(dst, restorePrint)
                }
                ImageIO.writeOrThrow(dst, outputPath)
            }
        } finally {
            src.release()
            layers.release()
        }
        return mapOf("outputPath" to outputPath)
    }

    /** Per-pixel paper colour: a large median over a 4x-downscaled copy wipes out ink strokes. */
    private fun estimatePaperColor(src: Mat): Mat {
        val small = Mat()
        return releasing(small) {
            Imgproc.resize(src, small, Size(), 0.25, 0.25, Imgproc.INTER_AREA)
            val kernel = min(21, (min(small.cols(), small.rows()) - 1) or 1)
            if (kernel >= 3) Imgproc.medianBlur(small, small, kernel)
            val paper = Mat()
            Imgproc.resize(small, paper, src.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            paper
        }
    }

    /**
     * Per-pixel colour of the NEARBY print (average over print pixels not covered by handwriting,
     * within roughly a text line's reach), falling back to the page-wide print average where there
     * is none — so a stroke crossing a blue heading restores blue, not black.
     */
    private fun estimatePrintColor(src: Mat, visiblePrint: Mat): Mat {
        val weight = Mat()
        val weight3 = Mat()
        val srcFloat = Mat()
        val weighted = Mat()
        val smallWeighted = Mat()
        val smallWeight = Mat()
        val gray = Mat()
        val core = Mat()
        return releasing(weight, weight3, srcFloat, weighted, smallWeighted, smallWeight, gray, core) {
            // Only the dark core of print strokes: their anti-aliased edges are much lighter and
            // would wash the restored colour out to grey.
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            val contrast = HandwritingMask.inkContrast(gray)
            releasing(contrast) {
                val c = ByteArray(contrast.total().toInt())
                val m = ByteArray(c.size)
                contrast.get(0, 0, c)
                visiblePrint.get(0, 0, m)
                val histogram = IntArray(256)
                var printCount = 0
                for (i in c.indices) {
                    if (m[i].toInt() != 0) {
                        histogram[c[i].toInt() and 0xFF]++
                        printCount++
                    }
                }
                var p95 = 0
                var seen = 0
                for (v in 0 until 256) {
                    seen += histogram[v]
                    if (seen >= printCount * 0.95) {
                        p95 = v
                        break
                    }
                }
                Imgproc.threshold(contrast, core, 0.85 * p95 - 0.5, 255.0, Imgproc.THRESH_BINARY)
            }
            Core.bitwise_and(core, visiblePrint, core)
            core.convertTo(weight, CvType.CV_32F, 1.0 / 255.0)
            src.convertTo(srcFloat, CvType.CV_32FC3)
            Core.merge(listOf(weight, weight, weight), weight3)
            Core.multiply(srcFloat, weight3, weighted)

            val f = 0.125
            Imgproc.resize(weighted, smallWeighted, Size(), f, f, Imgproc.INTER_AREA)
            Imgproc.resize(weight, smallWeight, Size(), f, f, Imgproc.INTER_AREA)
            val window = Size(15.0, 15.0)
            Imgproc.boxFilter(smallWeighted, smallWeighted, -1, window)
            Imgproc.boxFilter(smallWeight, smallWeight, -1, window)

            val total = Core.sumElems(weight).`val`[0]
            val sums = Core.sumElems(weighted).`val`
            val global = if (total > 0) doubleArrayOf(sums[0] / total, sums[1] / total, sums[2] / total)
            else doubleArrayOf(30.0, 30.0, 30.0)

            val w = FloatArray(smallWeight.total().toInt())
            smallWeight.get(0, 0, w)
            val c = FloatArray(w.size * 3)
            smallWeighted.get(0, 0, c)
            for (i in w.indices) {
                for (ch in 0 until 3) {
                    c[i * 3 + ch] = if (w[i] > 1e-3f) c[i * 3 + ch] / w[i] else global[ch].toFloat()
                }
            }
            val smallColor = Mat(smallWeight.size(), CvType.CV_32FC3)
            val color = Mat()
            releasing(smallColor) {
                smallColor.put(0, 0, c)
                Imgproc.resize(smallColor, color, src.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
                color.convertTo(color, CvType.CV_8UC3)
            }
            color
        }
    }
}
