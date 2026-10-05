package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * Last say on the handwriting layer, per ink stroke, by the segmentation model. The model judges a
 * stroke by its shape in context and is right about whole strokes far more often than the colour
 * layers upstream, which split strokes pixel by pixel ("57,45" half pen, half print) and so leave
 * half a digit on the page after the erase. Per connected ink stroke:
 *  - the model sees handwriting in most of it ([WHOLE]): the whole stroke is handwriting; its
 *    pixels the model keeps as print become overlap (print under the pen, restored by the erase);
 *  - the model sees almost none ([NONE]): none of it is handwriting;
 *  - in between (pen crossing print, strokes fused with a rule): the model's own pixels added to
 *    what the colour layers found (the model never takes those away).
 * Mirrors StrokeVote in ios/Runner/InkAnalysis.hpp.
 */
object StrokeVote {
    private const val WHOLE = 0.75
    private const val NONE = 0.05

    /** Dev switch (InkHarness `-e strokevote 0`). */
    @JvmStatic var enabled = true

    /**
     * [handwriting] / [print] / [overlap]: CV_8UC1 255 = set, updated in place. [ink]: CV_8UC1 ink of
     * the page. [modelHw] / [modelPrint]: the model's probabilities (CV_32F). [threshold]: the
     * handwriting probability cut-off in use.
     */
    fun apply(handwriting: Mat, print: Mat, overlap: Mat, ink: Mat, modelHw: Mat, modelPrint: Mat, threshold: Double) {
        if (!enabled) return
        val w = ink.cols(); val h = ink.rows()
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(ink, labels, stats, cents, 8, CvType.CV_32S)
        stats.release(); cents.release()
        val area = IntArray(n); val hwCount = IntArray(n)
        val lab = IntArray(w); val ph = FloatArray(w)
        for (y in 0 until h) {
            labels.get(y, 0, lab); modelHw.get(y, 0, ph)
            for (x in 0 until w) { val l = lab[x]; if (l > 0) { area[l]++; if (ph[x] > threshold) hwCount[l]++ } }
        }
        val share = DoubleArray(n) { if (area[it] == 0) 0.0 else hwCount[it].toDouble() / area[it] }
        // rule-shaped ink (long thin straight runs, either way): grid and answer lines, which the
        // colour layers can mistake for pen (a faint coloured exercise-book grid)
        val ruleShaped = Mat()
        run {
            val k = InkAnalysis.strokeUnit(w, h)
            val hRun = Mat(); val vRun = Mat()
            val kh = Mat.ones(1, 6 * k + 1, CvType.CV_8U); val kv = Mat.ones(6 * k + 1, 1, CvType.CV_8U)
            Imgproc.morphologyEx(ink, hRun, Imgproc.MORPH_OPEN, kh)
            Imgproc.morphologyEx(ink, vRun, Imgproc.MORPH_OPEN, kv)
            org.opencv.core.Core.bitwise_or(hRun, vRun, ruleShaped)
            listOf(hRun, vRun, kh, kv).forEach { it.release() }
        }
        val rsRow = ByteArray(w)
        val hwRow = ByteArray(w); val prRow = ByteArray(w); val ovRow = ByteArray(w); val pp = FloatArray(w)
        val on = 255.toByte()
        for (y in 0 until h) {
            labels.get(y, 0, lab); modelHw.get(y, 0, ph); modelPrint.get(y, 0, pp)
            handwriting.get(y, 0, hwRow); print.get(y, 0, prRow); overlap.get(y, 0, ovRow)
            ruleShaped.get(y, 0, rsRow)
            for (x in 0 until w) {
                val l = lab[x]
                if (l == 0) continue
                val s = share[l]
                val modelSaysHw = ph[x] > threshold
                when {
                    s >= WHOLE -> {
                        hwRow[x] = on
                        if (!modelSaysHw && pp[x] > 0.5f) { ovRow[x] = on; prRow[x] = on }
                    }
                    s <= NONE -> { hwRow[x] = 0; ovRow[x] = 0; prRow[x] = on }
                    else -> {
                        // in between (pen crossing print, strokes fused with a rule): the model ADDS
                        // its own pixels but does not take away what the colour layers found — a pen
                        // "I" standing on an answer line looks like a printed stem to the model,
                        // while its ink colour says pen
                        // (except on rule-shaped ink: the colour layers' grid lines are dropped)
                        if (modelSaysHw) hwRow[x] = on
                        else if (rsRow[x].toInt() != 0) hwRow[x] = 0
                        if (hwRow[x].toInt() == 0) { ovRow[x] = 0; prRow[x] = on }
                    }
                }
            }
            handwriting.put(y, 0, hwRow); print.put(y, 0, prRow); overlap.put(y, 0, ovRow)
        }
        labels.release(); ruleShaped.release()
        growAlongStrokes(handwriting, print, overlap, ink, modelHw)
        rememberPrintUnderPen(handwriting, print, overlap, modelPrint)
    }

    /** The model's print probability above which a handwriting pixel has print under it. */
    private const val UNDER_PEN_PRINT = 0.5

    /**
     * Handwriting is erased whole; where the model also sees print under it (a printed letter a pen
     * circle crosses, a word a tick runs through) that print is remembered as overlap, which the
     * erase and the final restore draw back.
     */
    private fun rememberPrintUnderPen(handwriting: Mat, print: Mat, overlap: Mat, modelPrint: Mat) {
        val under = Mat()
        org.opencv.core.Core.compare(modelPrint, org.opencv.core.Scalar(UNDER_PEN_PRINT), under, org.opencv.core.Core.CMP_GT)
        org.opencv.core.Core.bitwise_and(under, handwriting, under)
        org.opencv.core.Core.bitwise_or(overlap, under, overlap)
        org.opencv.core.Core.bitwise_or(print, under, print)
        under.release()
    }

    /** Ink pixels the model gives at least this handwriting probability may join a touching stroke. */
    private const val GROW_HW = 0.2
    private const val GROW_STEPS = 60

    /**
     * Pixel by pixel from what is surely handwriting, along the ink: a pixel touching handwriting
     * whose handwriting probability is not low is the same stroke. A black pen circle round a printed
     * "B" is half-hearted for the model next to the letter (it looks like print there) and came out
     * broken; grown from its sure parts it closes, while the letter (no handwriting probability at
     * all) stays print.
     */
    private fun growAlongStrokes(handwriting: Mat, print: Mat, overlap: Mat, ink: Mat, modelHw: Mat) {
        val allowed = Mat(); val tmp = Mat(); val grown = handwriting.clone()
        val kernel = Mat.ones(3, 3, CvType.CV_8U)
        org.opencv.core.Core.compare(modelHw, org.opencv.core.Scalar(GROW_HW), tmp, org.opencv.core.Core.CMP_GT)
        org.opencv.core.Core.bitwise_and(tmp, ink, allowed)
        val next = Mat()
        for (i in 0 until GROW_STEPS) {
            Imgproc.dilate(grown, next, kernel)
            org.opencv.core.Core.bitwise_and(next, allowed, next)
            org.opencv.core.Core.bitwise_or(next, grown, next)
            org.opencv.core.Core.compare(next, grown, tmp, org.opencv.core.Core.CMP_NE)
            val changed = org.opencv.core.Core.countNonZero(tmp)
            next.copyTo(grown)
            if (changed == 0) break
        }
        // the new handwriting is no longer print / overlap
        org.opencv.core.Core.bitwise_xor(grown, handwriting, tmp)
        val notNew = Mat(); org.opencv.core.Core.bitwise_not(tmp, notNew)
        org.opencv.core.Core.bitwise_and(print, notNew, print)
        org.opencv.core.Core.bitwise_and(overlap, notNew, overlap)
        grown.copyTo(handwriting)
        listOf(allowed, tmp, grown, kernel, next, notNew).forEach { it.release() }
    }
}
