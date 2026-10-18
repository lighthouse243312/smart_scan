package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

/**
 * Second stage of "Ink colour + AI": the colour method has already decided, and what it calls
 * handwriting stays handwriting — except whole strokes the model is SURE are print (coloured
 * print: a red heading, a blue grid line), which colour alone cannot tell from pen. The model then
 * finishes the job on the ink colour left alone:
 *  - tails: ink touching the colour-found handwriting that the model sees as handwriting, grown
 *    pixel by pixel along the stroke (faint ends, the dark core of a coloured pen stroke);
 *  - pen the colour of the print (black ballpoint ticks, circles, dots, lines): whole strokes the
 *    model is sure are handwriting and not print. Strokes it is unsure about are left alone, so
 *    the model cannot smear into the print.
 * A stroke the model is less sure of is taken too when it does not stand like the page's print: its
 * stems lean another way than the print's, measured against the photo's own tilt (see [PrintSlant]).
 * Print the model sees under the new handwriting is remembered as overlap, to be drawn back.
 * Mirrors ModelFinish in ios/Runner/ImageProcessingOpenCV.mm.
 */
object ModelFinish {
    /** A colour-the-same-as-print stroke is taken only when the model is this sure on average… */
    private const val SURE_HW = 0.7
    /** …and does not see print in it… */
    @JvmStatic var MAX_PRINT = 1.0
    /**
     * …and sees handwriting this far above print: an outline of a printed drawing (a cloud's arcs)
     * can read as pen to the model, but never by such a margin over print.
     */
    @JvmStatic var SURE_MARGIN = 0.2
    /** Print probability above which a new handwriting pixel has print under it. */
    private const val UNDER_PEN_PRINT = 0.5
    /** Dev knob (InkHarness `-e grow N`). */
    @JvmStatic var GROW_STEPS = 40
    /** Rim growth round the strokes found: steps (px) and the model's handwriting probability it needs. */
    @JvmStatic var RIM_STEPS = 3
    @JvmStatic var RIM_THR = 0.1
    /** A colour-found stroke is print after all when the model sees this much print in it… */
    private const val VETO_PRINT = 0.6
    /** …and this little handwriting. */
    private const val VETO_HW = 0.2
    /** A stroke the model is less sure of is still taken when it is at least this likely… */
    private const val MAYBE_HW = 0.45
    /** …and its stems lean away from the page's print by this many of the print's spreads… */
    private const val SLANT_SPREADS = 2.5
    /** …and at least this many degrees. */
    private const val SLANT_MIN_DEG = 8.0

    /** In a colour-found stroke, a pixel is print when the model sees this much print in it… */
    private const val PIXEL_PRINT = 0.6
    /** …and this little handwriting… */
    private const val PIXEL_HW = 0.2
    /** …once such pixels are this share of the stroke (printed letters a pen line crossed). */
    @JvmStatic var PIXEL_SHARE = 0.02
    /** Dev switch (InkHarness `-e pixveto 0`). */
    @JvmStatic var pixelVeto = true

    /** Dev switch (InkHarness `-e slant 0`). */
    @JvmStatic var useSlant = false
    /** Dev switch (InkHarness `-e sure 0`): no whole confident strokes among the rest. */
    @JvmStatic var takeSure = true
    /** Dev counters of the last run (InkHarness report). */
    @JvmStatic var report = ""

    /**
     * [handwriting] / [print] / [overlap]: the colour method's layers (CV_8UC1, 255 = set), updated
     * in place. [ink]: the page's ink. [modelHw] / [modelPrint]: CV_32F probabilities.
     */
    fun apply(src: Mat, handwriting: Mat, print: Mat, overlap: Mat, ink: Mat, modelHw: Mat, modelPrint: Mat, threshold: Double, minArea: Int) {
        vetoColouredPrint(handwriting, print, overlap, modelHw, modelPrint)
        vetoPrintedRules(handwriting, print, overlap, modelHw, modelPrint)
        val modelSays = Mat(); val free = Mat(); val notHw = Mat()
        // the strokes are judged by the model above its threshold, or by the slant above MAYBE_HW
        Core.compare(modelHw, Scalar(if (useSlant) minOf(threshold, MAYBE_HW) else threshold), modelSays, Core.CMP_GT)
        Core.bitwise_not(handwriting, notHw)
        Core.bitwise_and(ink, notHw, free)              // ink the colour method left alone
        Core.bitwise_and(free, modelSays, modelSays)    // …that the model sees as handwriting

        // tails: grow the colour-found handwriting into model-handwriting ink it touches
        val grown = handwriting.clone()
        val next = Mat(); val changed = Mat()
        val k3 = Mat.ones(3, 3, CvType.CV_8U)
        for (i in 0 until GROW_STEPS) {
            Imgproc.dilate(grown, next, k3)
            Core.bitwise_and(next, modelSays, next)
            Core.bitwise_or(next, grown, next)
            Core.compare(next, grown, changed, Core.CMP_NE)
            val n = Core.countNonZero(changed)
            next.copyTo(grown)
            if (n == 0) break
        }

        // pen the colour of the print: whole confident strokes among the rest
        val rest = Mat(); val notGrown = Mat()
        Core.bitwise_not(grown, notGrown)
        Core.bitwise_and(modelSays, notGrown, rest)
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(rest, labels, stats, cents, 8, CvType.CV_32S)
        val area = IntArray(n); val sumHw = DoubleArray(n); val sumPr = DoubleArray(n)
        val w = rest.cols(); val lab = IntArray(w); val ph = FloatArray(w); val pp = FloatArray(w)
        for (y in 0 until rest.rows()) {
            labels.get(y, 0, lab); modelHw.get(y, 0, ph); modelPrint.get(y, 0, pp)
            for (x in 0 until w) {
                val l = lab[x]
                if (l == 0) continue
                area[l]++; sumHw[l] += ph[x]; sumPr[l] += pp[x]
            }
        }
        // the page's print: how its stems stand against the photo's tilt
        val edges = if (useSlant) PrintSlant.edges(src) else null
        val ref = edges?.let { PrintSlant.reference(it, print, minArea) }
        val slant = if (ref != null) PrintSlant.slants(edges!!, labels, n, ref.tilt) else null
        var bySlant = 0
        val take = BooleanArray(n) {
            if (it == 0 || area[it] < minArea || sumPr[it] / area[it] >= MAX_PRINT) return@BooleanArray false
            val hw = sumHw[it] / area[it]
            if (!takeSure) return@BooleanArray false
            if (hw >= SURE_HW && hw - sumPr[it] / area[it] >= SURE_MARGIN) return@BooleanArray true
            // less sure: a hand's lean, not the print's
            if (hw < MAYBE_HW || slant == null || slant[it].isNaN()) return@BooleanArray false
            val off = abs(slant[it] - ref!!.slant)
            (off >= SLANT_MIN_DEG && off >= SLANT_SPREADS * ref.spread).also { t -> if (t) bySlant++ }
        }
        edges?.release()
        report = if (ref == null) "slant: no print reference\n"
            else "slant: tilt %.1f print %.1f±%.1f, taken by slant %d\n".format(ref.tilt, ref.slant, ref.spread, bySlant)
        val sure = Mat.zeros(rest.size(), CvType.CV_8U)
        val row = ByteArray(w)
        for (y in 0 until rest.rows()) {
            labels.get(y, 0, lab)
            for (x in 0 until w) row[x] = if (take[lab[x]]) 255.toByte() else 0
            sure.put(y, 0, row)
        }
        Core.bitwise_or(grown, sure, grown)
        // rims and thin ends of the strokes found: the model sees them a little less sure of being pen
        // than the stroke's core; ink right next to a stroke that the model leans to as pen, and does
        // not see as print, goes with it
        if (RIM_STEPS > 0) {
            val rim = Mat(); val notPr = Mat()
            Core.compare(modelHw, Scalar(RIM_THR), rim, Core.CMP_GT)
            Core.compare(modelPrint, Scalar(0.5), notPr, Core.CMP_LT)
            Core.bitwise_and(rim, notPr, rim); Core.bitwise_and(rim, ink, rim)
            val nx = Mat()
            for (i in 0 until RIM_STEPS) {
                Imgproc.dilate(grown, nx, k3); Core.bitwise_and(nx, rim, nx); Core.bitwise_or(grown, nx, grown)
            }
            listOf(rim, notPr, nx).forEach { it.release() }
        }

        // what the model added: no longer print, except print it sees under the pen (redrawn)
        val added = Mat(); Core.bitwise_and(grown, notHw, added)
        val under = Mat(); Core.compare(modelPrint, Scalar(UNDER_PEN_PRINT), under, Core.CMP_GT)
        Core.bitwise_and(under, added, under)
        val notAdded = Mat(); Core.bitwise_not(added, notAdded)
        Core.bitwise_and(print, notAdded, print)
        Core.bitwise_or(print, under, print)
        Core.bitwise_or(overlap, under, overlap)
        grown.copyTo(handwriting)
        if (fillHoles) fillStrokeHoles(handwriting, print, ink, modelHw, modelPrint)

        listOf(modelSays, free, notHw, grown, next, changed, k3, rest, notGrown, labels, stats, cents, sure, added, under, notAdded)
            .forEach { it.release() }
    }

    /** Dev switch (InkHarness `-e holes 0`). */
    @JvmStatic var fillHoles = true

    /**
     * A pen stroke on a grainy photo comes out speckled: single ink pixels inside it fall just on
     * the print side of the colour test. Ink the stroke closes over (a 3×3 closing) is the stroke's
     * too — unless the model sees print there and no pen (a printed letter the stroke touches).
     */
    private fun fillStrokeHoles(handwriting: Mat, print: Mat, ink: Mat, modelHw: Mat, modelPrint: Mat) {
        val closed = Mat(); val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, org.opencv.core.Size(3.0, 3.0))
        Imgproc.morphologyEx(handwriting, closed, Imgproc.MORPH_CLOSE, k)
        Core.bitwise_and(closed, ink, closed)
        val isPrint = Mat(); val noPen = Mat()
        Core.compare(modelPrint, Scalar(VETO_PRINT), isPrint, Core.CMP_GT)
        Core.compare(modelHw, Scalar(VETO_HW), noPen, Core.CMP_LT)
        Core.bitwise_and(isPrint, noPen, isPrint)
        Core.bitwise_not(isPrint, isPrint)
        Core.bitwise_and(closed, isPrint, closed)
        val notClosed = Mat(); Core.bitwise_not(closed, notClosed)
        Core.bitwise_or(handwriting, closed, handwriting)
        Core.bitwise_and(print, notClosed, print)
        listOf(closed, k, isPrint, noPen, notClosed).forEach { it.release() }
    }

    /** Drops from [handwriting] whole strokes the model is sure are print; they become print. */
    private fun vetoColouredPrint(handwriting: Mat, print: Mat, overlap: Mat, modelHw: Mat, modelPrint: Mat) {
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(handwriting, labels, stats, cents, 8, CvType.CV_32S)
        if (n <= 1) { listOf(labels, stats, cents).forEach { it.release() }; return }
        val area = IntArray(n); val sumHw = DoubleArray(n); val sumPr = DoubleArray(n)
        val w = handwriting.cols(); val lab = IntArray(w); val ph = FloatArray(w); val pp = FloatArray(w)
        for (y in 0 until handwriting.rows()) {
            labels.get(y, 0, lab); modelHw.get(y, 0, ph); modelPrint.get(y, 0, pp)
            for (x in 0 until w) {
                val l = lab[x]
                if (l == 0) continue
                area[l]++; sumHw[l] += ph[x]; sumPr[l] += pp[x]
            }
        }
        val veto = BooleanArray(n) { it > 0 && sumPr[it] / area[it] >= VETO_PRINT && sumHw[it] / area[it] <= VETO_HW }
        // printed letters a black pen line crossed join the line's stroke, and the colour of the
        // print and the pen alike puts the whole word in the handwriting: where the model sees
        // plain print in much of such a stroke, those pixels are print — the pen line itself stays
        val sure = IntArray(n)
        if (pixelVeto) for (y in 0 until handwriting.rows()) {
            labels.get(y, 0, lab); modelHw.get(y, 0, ph); modelPrint.get(y, 0, pp)
            for (x in 0 until w) { val l = lab[x]; if (l != 0 && pp[x] >= PIXEL_PRINT && ph[x] <= PIXEL_HW) sure[l]++ }
        }
        val split = BooleanArray(n) { it > 0 && !veto[it] && pixelVeto && sure[it] >= PIXEL_SHARE * area[it] }
        val hwRow = ByteArray(w); val prRow = ByteArray(w); val ovRow = ByteArray(w)
        for (y in 0 until handwriting.rows()) {
            labels.get(y, 0, lab); handwriting.get(y, 0, hwRow); print.get(y, 0, prRow); overlap.get(y, 0, ovRow)
            if (pixelVeto) { modelHw.get(y, 0, ph); modelPrint.get(y, 0, pp) }
            for (x in 0 until w) {
                val l = lab[x]
                if (veto[l] || (split[l] && pp[x] >= PIXEL_PRINT && ph[x] <= PIXEL_HW)) { hwRow[x] = 0; prRow[x] = 255.toByte(); ovRow[x] = 0 }
            }
            handwriting.put(y, 0, hwRow); print.put(y, 0, prRow); overlap.put(y, 0, ovRow)
        }
        listOf(labels, stats, cents).forEach { it.release() }
    }

    /** On a ruled line, print with this much model print… */
    private const val RULE_PRINT = 0.5
    /** …and this little handwriting is print, even fused with a pen stroke. */
    private const val RULE_HW = 0.3

    /**
     * Coloured rules (a purple grid, a printed answer line) touching the writing join its stroke, so
     * the whole-stroke veto cannot take them out. Long straight runs of the handwriting layer are
     * judged pixel by pixel instead: where the model sees print and no pen, they are print — the
     * pixels where the pen crosses the rule keep the model's handwriting and stay erased.
     */
    private fun vetoPrintedRules(handwriting: Mat, print: Mat, overlap: Mat, modelHw: Mat, modelPrint: Mat) {
        val len = maxOf(15, maxOf(handwriting.cols(), handwriting.rows()) / 80)
        val h = Mat(); val v = Mat(); val rules = Mat()
        Imgproc.morphologyEx(handwriting, h, Imgproc.MORPH_OPEN, Mat.ones(1, len, CvType.CV_8U))
        Imgproc.morphologyEx(handwriting, v, Imgproc.MORPH_OPEN, Mat.ones(len, 1, CvType.CV_8U))
        Core.bitwise_or(h, v, rules)
        val isPrint = Mat(); val noPen = Mat()
        Core.compare(modelPrint, Scalar(RULE_PRINT), isPrint, Core.CMP_GT)
        Core.compare(modelHw, Scalar(RULE_HW), noPen, Core.CMP_LT)
        Core.bitwise_and(rules, isPrint, rules)
        Core.bitwise_and(rules, noPen, rules)
        val notRules = Mat(); Core.bitwise_not(rules, notRules)
        Core.bitwise_and(handwriting, notRules, handwriting)
        Core.bitwise_and(overlap, notRules, overlap)
        Core.bitwise_or(print, rules, print)
        listOf(h, v, rules, isPrint, noPen, notRules).forEach { it.release() }
    }
}
