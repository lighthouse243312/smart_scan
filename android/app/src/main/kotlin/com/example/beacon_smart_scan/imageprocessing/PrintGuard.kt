package com.example.beacon_smart_scan.imageprocessing

import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Multi-pass erase support. Later passes re-detect on the previous pass's result, where the pen
 * left behind is easy to see — but so is print the first pass damaged, which a re-detection can
 * call handwriting. The first pass's mask, measured on the untouched page's colours, is the
 * reference for what is print: later passes may not erase it, and print it says was there but
 * that came out as paper is copied back at the end. Mirrors the same methods in
 * ios/Runner/ImageProcessingOpenCV.mm.
 */
object PrintGuard {
    /** Clearance kept around any handwriting when restoring print, so a pen's rim never returns. */
    private const val RIM = 5

    /** How much lighter (8-bit grey) than the clean page a pixel must have become to count as erased. */
    private const val ERASED_DELTA = 40.0

    /** Print in [ref] with no handwriting on it (the mask file keeps overlap out of its print layer). */
    private fun purePrint(ref: HandwritingMask.Layers): Mat {
        val notHw = Mat(); Core.bitwise_not(ref.handwriting, notHw)
        return Mat().also { Core.bitwise_and(ref.print, notHw, it); notHw.release() }
    }

    /**
     * Print the reference mask remembered under the pen (overlap) that came out as paper is drawn
     * back in the page's own print colour — not copied from the clean page, where those pixels carry
     * the pen's colour. Returns the pixels drawn.
     */
    private fun redrawUnderPen(out: Mat, clean: Mat, ref: HandwritingMask.Layers): Int {
        val pure = purePrint(ref)
        val gray = Mat(); Imgproc.cvtColor(clean, gray, Imgproc.COLOR_BGR2GRAY)
        val gOut = Mat(); Imgproc.cvtColor(out, gOut, Imgproc.COLOR_BGR2GRAY)
        try {
            // the print's colour: the darker half of the pure print pixels (their cores, not the rims)
            val n = Core.countNonZero(pure)
            if (n < 50 || Core.countNonZero(ref.overlap) == 0) return 0
            val g = ByteArray(gray.total().toInt()); gray.get(0, 0, g)
            val p = ByteArray(pure.total().toInt()); pure.get(0, 0, p)
            val vals = ArrayList<Int>(n)
            for (i in g.indices) if (p[i].toInt() != 0) vals.add(g[i].toInt() and 0xff)
            vals.sort()
            val core = vals[vals.size / 4]
            val coreMask = Mat(); Core.compare(gray, org.opencv.core.Scalar(core.toDouble()), coreMask, Core.CMP_LE)
            Core.bitwise_and(coreMask, pure, coreMask)
            val colour = Core.mean(clean, coreMask)
            // overlap pixels the erase left lighter than the print: paper now, drawn back
            val paperNow = Mat(); Core.compare(gOut, org.opencv.core.Scalar(core + ERASED_DELTA), paperNow, Core.CMP_GT)
            Core.bitwise_and(paperNow, ref.overlap, paperNow)
            out.setTo(colour, paperNow)
            val drawn = Core.countNonZero(paperNow)
            coreMask.release(); paperNow.release()
            return drawn
        } finally {
            pure.release(); gray.release(); gOut.release()
        }
    }

    /** Smallest print piece (8-connected pixels) that is a letter, not a pen rim's stray print pixels. */
    private const val MIN_PIECE = 15

    /** The pieces of [print] (CV_8U) at least [MIN_PIECE] pixels big. */
    private fun letterPieces(print: Mat): Mat {
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(print, labels, stats, cents, 8, CvType.CV_32S)
        val keep = BooleanArray(n) { it > 0 && stats.get(it, Imgproc.CC_STAT_AREA)[0] >= MIN_PIECE }
        val out = Mat.zeros(print.size(), CvType.CV_8U)
        val w = print.cols(); val row = IntArray(w); val o = ByteArray(w)
        for (y in 0 until print.rows()) {
            labels.get(y, 0, row)
            for (x in 0 until w) o[x] = if (keep[row[x]]) 255.toByte() else 0
            out.put(y, 0, o)
        }
        listOf(labels, stats, cents).forEach { it.release() }
        return out
    }

    /** Handwriting of the reference and of every later mask, undilated, at [size]. */
    private fun hwTouch(ref: HandwritingMask.Layers, maskPaths: List<String>, size: Size): Mat {
        val all = ref.handwriting.clone()
        for (path in maskPaths) {
            val l = HandwritingMask.read(path)
            fit(l.handwriting, size)
            Core.bitwise_or(all, l.handwriting, all)
            l.release()
        }
        return all
    }

    private fun fit(m: Mat, size: Size) {
        if (m.size() != size) Imgproc.resize(m, m, size, 0.0, 0.0, Imgproc.INTER_NEAREST)
    }

    /**
     * Drops from [maskPath]'s handwriting every pixel [referencePath] calls pure print, adding them
     * to its print layer instead; written to [outputPath].
     */
    fun guardMask(maskPath: String, referencePath: String, outputPath: String): Map<String, Any> {
        val layers = HandwritingMask.read(maskPath)
        val ref = HandwritingMask.read(referencePath)
        try {
            val size = layers.handwriting.size()
            listOf(ref.handwriting, ref.print, ref.overlap).forEach { fit(it, size) }
            val protect = purePrint(ref)
            val notProtect = Mat()
            releasing(protect, notProtect) {
                Core.bitwise_not(protect, notProtect)
                Core.bitwise_and(layers.handwriting, notProtect, layers.handwriting)
                Core.bitwise_or(layers.print, protect, layers.print)
                HandwritingMask.write(layers.handwriting, layers.print, layers.overlap, outputPath)
            }
            return HandwritingMask.result(outputPath, layers.handwriting)
        } finally {
            layers.release(); ref.release()
        }
    }

    /**
     * Copies back from [cleanPath] (the page before any erase) the pure print of [referencePath]
     * that [inputPath] (the erased page) turned to paper, away from the handwriting of every mask in
     * [maskPaths]; written to [outputPath].
     */
    fun restorePrint(inputPath: String, cleanPath: String, referencePath: String, maskPaths: List<String>, outputPath: String): Map<String, Any> {
        val out = ImageIO.readOrThrow(inputPath)
        val clean = ImageIO.readOrThrow(cleanPath)
        val ref = HandwritingMask.read(referencePath)
        val hwAll = Mat.zeros(out.size(), CvType.CV_8U)
        try {
            if (clean.size() != out.size()) Imgproc.resize(clean, clean, out.size(), 0.0, 0.0, Imgproc.INTER_AREA)
            listOf(ref.handwriting, ref.print, ref.overlap).forEach { fit(it, out.size()) }
            for (path in maskPaths + referencePath) {
                val l = HandwritingMask.read(path)
                fit(l.handwriting, out.size())
                Core.bitwise_or(hwAll, l.handwriting, hwAll)
                l.release()
            }
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(2.0 * RIM + 1, 2.0 * RIM + 1))
            Imgproc.dilate(hwAll, hwAll, kernel); kernel.release()
            val restore = purePrint(ref)
            // print pieces the size of a letter come back even right beside the pen (a printed "A"
            // inside a pen circle): pixel by pixel, all of the piece but the 1 px touching the pen
            run {
                val pieces = letterPieces(restore)
                val touch = Mat(); Imgproc.dilate(hwTouch(ref, maskPaths, out.size()), touch, Mat.ones(3, 3, CvType.CV_8U))
                val notTouch = Mat(); Core.bitwise_not(touch, notTouch)
                Core.bitwise_and(pieces, notTouch, pieces)
                val notAll = Mat(); Core.bitwise_not(hwAll, notAll)
                Core.bitwise_and(restore, notAll, restore)
                Core.bitwise_or(restore, pieces, restore)
                listOf(pieces, touch, notTouch, notAll).forEach { it.release() }
                hwAll.setTo(org.opencv.core.Scalar(0.0))   // already applied above
            }
            val notHw = Mat(); val grayOut = Mat(); val grayClean = Mat(); val lighter = Mat()
            val restored = releasing(restore, notHw, grayOut, grayClean, lighter) {
                Core.bitwise_not(hwAll, notHw)
                Core.bitwise_and(restore, notHw, restore)
                Imgproc.cvtColor(out, grayOut, Imgproc.COLOR_BGR2GRAY)
                Imgproc.cvtColor(clean, grayClean, Imgproc.COLOR_BGR2GRAY)
                Core.subtract(grayOut, grayClean, lighter)          // saturates at 0 where not lighter
                Imgproc.threshold(lighter, lighter, ERASED_DELTA, 255.0, Imgproc.THRESH_BINARY)
                Core.bitwise_and(restore, lighter, restore)
                clean.copyTo(out, restore)
                Core.countNonZero(restore) + redrawUnderPen(out, clean, ref)
            }
            ImageIO.writeOrThrow(out, outputPath)
            return mapOf("outputPath" to outputPath, "restoredPixels" to restored)
        } finally {
            out.release(); clean.release(); ref.release(); hwAll.release()
        }
    }
}
