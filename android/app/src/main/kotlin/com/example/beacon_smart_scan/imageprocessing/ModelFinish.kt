package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

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
 * Print the model sees under the new handwriting is remembered as overlap, to be drawn back.
 * Mirrors ModelFinish in ios/Runner/ImageProcessingOpenCV.mm.
 */
object ModelFinish {
    /** A colour-the-same-as-print stroke is taken only when the model is this sure on average… */
    private const val SURE_HW = 0.7
    /** …and does not see print in it. */
    private const val MAX_PRINT = 0.4
    /** Print probability above which a new handwriting pixel has print under it. */
    private const val UNDER_PEN_PRINT = 0.5
    private const val GROW_STEPS = 40
    /** A colour-found stroke is print after all when the model sees this much print in it… */
    private const val VETO_PRINT = 0.6
    /** …and this little handwriting. */
    private const val VETO_HW = 0.2

    /**
     * [handwriting] / [print] / [overlap]: the colour method's layers (CV_8UC1, 255 = set), updated
     * in place. [ink]: the page's ink. [modelHw] / [modelPrint]: CV_32F probabilities.
     */
    fun apply(handwriting: Mat, print: Mat, overlap: Mat, ink: Mat, modelHw: Mat, modelPrint: Mat, threshold: Double, minArea: Int) {
        vetoColouredPrint(handwriting, print, overlap, modelHw, modelPrint)
        vetoPrintedRules(handwriting, print, overlap, modelHw, modelPrint)
        val modelSays = Mat(); val free = Mat(); val notHw = Mat()
        Core.compare(modelHw, Scalar(threshold), modelSays, Core.CMP_GT)
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
        val take = BooleanArray(n) { it > 0 && area[it] >= minArea && sumHw[it] / area[it] >= SURE_HW && sumPr[it] / area[it] < MAX_PRINT }
        val sure = Mat.zeros(rest.size(), CvType.CV_8U)
        val row = ByteArray(w)
        for (y in 0 until rest.rows()) {
            labels.get(y, 0, lab)
            for (x in 0 until w) row[x] = if (take[lab[x]]) 255.toByte() else 0
            sure.put(y, 0, row)
        }
        Core.bitwise_or(grown, sure, grown)

        // what the model added: no longer print, except print it sees under the pen (redrawn)
        val added = Mat(); Core.bitwise_and(grown, notHw, added)
        val under = Mat(); Core.compare(modelPrint, Scalar(UNDER_PEN_PRINT), under, Core.CMP_GT)
        Core.bitwise_and(under, added, under)
        val notAdded = Mat(); Core.bitwise_not(added, notAdded)
        Core.bitwise_and(print, notAdded, print)
        Core.bitwise_or(print, under, print)
        Core.bitwise_or(overlap, under, overlap)
        grown.copyTo(handwriting)

        listOf(modelSays, free, notHw, grown, next, changed, k3, rest, notGrown, labels, stats, cents, sure, added, under, notAdded)
            .forEach { it.release() }
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
        val hwRow = ByteArray(w); val prRow = ByteArray(w); val ovRow = ByteArray(w)
        for (y in 0 until handwriting.rows()) {
            labels.get(y, 0, lab); handwriting.get(y, 0, hwRow); print.get(y, 0, prRow); overlap.get(y, 0, ovRow)
            for (x in 0 until w) if (veto[lab[x]]) { hwRow[x] = 0; prRow[x] = 255.toByte(); ovRow[x] = 0 }
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
