package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Second look at what the first pass left beside the handwriting it found: only pieces of "print"
 * touching the found strokes' surroundings are judged. Printed text never stands alone — a letter
 * has its word beside it, a word its line, a digit its label. Such a letter-sized piece with no
 * other print within [REACH] text heights is a pen leftover the detectors missed
 * (a black-pen dot, a short tick, a scribble, the tail of a stroke cut off from it) and becomes
 * handwriting — unless it is a letter or digit after all: a typeset page repeats its glyphs, so a
 * lone printed "5" or "a" has a twin of the same shape and size among the page's other print, and a
 * scribble has none; a glyph the page has only once (the one "8") the model still sees as print.
 * Specks (a full stop, a fringe pixel) are not judged, and neither are pieces larger than a few
 * letters (pictures, frames, rules). Rules are not text: a pen piece standing on an answer line is
 * judged without the line, and only the piece (never the line) becomes handwriting.
 * Mirrors IsolatedPrint in ios/Runner/ImageProcessingOpenCV.mm.
 */
object IsolatedPrint {
    /** A piece is alone when no other print is within this many text heights of it. */
    @JvmStatic var REACH = 1.0
    /** Only pieces up to this many text heights tall and wide are judged. */
    private const val MAX_SIZE = 1.5

    /** Dev switch (InkHarness `-e isolated 0`). */
    @JvmStatic var enabled = true
    /** A twin covers the piece by this much (intersection over union, scaled to the piece's box). */
    private const val TWIN_IOU = 0.6
    /** A piece is a stroke, not a speck, when it is at least this many text heights long. */
    private const val MIN_SIZE = 0.4
    /** The model sees a glyph in a piece with this much print on average. */
    @JvmStatic var SURE_PRINT = 0.6

    /** Dev switch (InkHarness `-e isoglyph 0`): no twin test. */
    @JvmStatic var glyphs = true
    /** Dev switch (InkHarness `-e isonear 0`): judge every piece on the page, not only the pen's leftovers. */
    @JvmStatic var nearPen = true
    /** A leftover lies within this many stroke units of the found handwriting. */
    private const val LEFTOVER_UNITS = 2
    /** A group of up to this many pieces, all at the pen, is the pen's leftovers too. */
    @JvmStatic var MAX_LEFTOVER_GROUP = 1
    /** Dev counters of the last run (InkHarness report). */
    @JvmStatic var report = ""

    /** [handwriting] / [print] / [overlap]: CV_8UC1 255 = set, updated in place. [modelPrint]: CV_32F. */
    fun apply(handwriting: Mat, print: Mat, overlap: Mat, modelPrint: Mat, minArea: Int) {
        report = ""
        if (!enabled) return
        val t = ArrayList<Mat>()
        fun own(m: Mat): Mat { t.add(m); return m }
        try {
            // print as the mask will show it: none of the pen's pixels; print under the pen counts
            val notHw = own(Mat()); Core.bitwise_not(handwriting, notHw)
            val pure = own(Mat()); Core.bitwise_and(print, notHw, pure)
            val allPrint = own(Mat()); Core.bitwise_or(pure, overlap, allPrint)
            HandwritingMask.removeSmallComponents(allPrint, max(3, minArea / 4))   // paper specks are not print
            // rules (answer lines, table grid): long straight runs, taken out of the print first
            val len = max(15, max(allPrint.cols(), allPrint.rows()) / 60)
            val hR = own(Mat()); val vR = own(Mat()); val rules = own(Mat())
            Imgproc.morphologyEx(allPrint, hR, Imgproc.MORPH_OPEN, own(Mat.ones(1, len, CvType.CV_8U)))
            Imgproc.morphologyEx(allPrint, vR, Imgproc.MORPH_OPEN, own(Mat.ones(len, 1, CvType.CV_8U)))
            Core.bitwise_or(hR, vR, rules)
            Imgproc.dilate(rules, rules, own(Mat.ones(3, 3, CvType.CV_8U)))
            val notRules = own(Mat()); Core.bitwise_not(rules, notRules)
            Core.bitwise_and(allPrint, notRules, allPrint)
            HandwritingMask.removeSmallComponents(allPrint, max(3, minArea / 4))

            val labels = own(Mat()); val stats = own(Mat()); val cents = own(Mat())
            val n = Imgproc.connectedComponentsWithStats(allPrint, labels, stats, cents, 8, CvType.CV_32S)
            if (n < 50) return   // too little print to know how it stands
            val st = IntArray(n * 5); stats.get(0, 0, st)
            fun h(i: Int) = st[i * 5 + Imgproc.CC_STAT_HEIGHT]
            fun w(i: Int) = st[i * 5 + Imgproc.CC_STAT_WIDTH]
            fun area(i: Int) = st[i * 5 + Imgproc.CC_STAT_AREA]
            // text height: the median letter (pieces of a letter's size, not specks or grid dots)
            val unit = InkAnalysis.strokeUnit(allPrint.cols(), allPrint.rows())
            val hs = (1 until n).filter { area(it) >= max(minArea, 4 * unit * unit) }.map { h(it) }.sorted()
            if (hs.isEmpty()) return
            val cap = hs[hs.size / 2].toDouble()

            // pieces closer than the reach join one group; a group of one is a piece alone
            val r = max(1, (REACH * cap / 2).roundToInt())
            val near = own(Mat())
            Imgproc.dilate(allPrint, near, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(2.0 * r + 1, 2.0 * r + 1)))
            val groups = own(Mat())
            val g = Imgproc.connectedComponents(near, groups, 8, CvType.CV_32S)
            val W = allPrint.cols(); val H = allPrint.rows()
            val lab = IntArray(W * H); labels.get(0, 0, lab)
            val grp = IntArray(W * H); groups.get(0, 0, grp)
            val members = IntArray(g); val groupOf = IntArray(n) { -1 }
            for (p in lab.indices) {
                val l = lab[p]; if (l == 0 || groupOf[l] >= 0) continue
                groupOf[l] = grp[p]; members[grp[p]]++
            }
            var pen: ByteArray? = null
            if (nearPen) {
                val rim = own(Mat())
                val d = 2.0 * LEFTOVER_UNITS * InkAnalysis.strokeUnit(W, H) + 1
                Imgproc.dilate(handwriting, rim, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(d, d)))
                pen = ByteArray(W * H); rim.get(0, 0, pen)
            }
            val ob = ByteArray(W * H); overlap.get(0, 0, ob)
            val pp = FloatArray(W * H); modelPrint.get(0, 0, pp)
            val touchesPen = BooleanArray(n); val underPen = BooleanArray(n); val sumPr = DoubleArray(n)
            for (p in lab.indices) {
                val l = lab[p]; if (l == 0) continue
                if (pen != null && pen[p].toInt() != 0) touchesPen[l] = true
                if (ob[p].toInt() != 0) underPen[l] = true
                sumPr[l] += pp[p]
            }
            // a small group whose every piece lies at the found handwriting: the pen's leftovers of one
            // word (a "5" and the stub of a "Y" beside it), judged piece by piece like a piece alone
            val allAtPen = BooleanArray(g) { true }
            for (l in 1 until n) if (!touchesPen[l]) allAtPen[groupOf[l]] = false
            var seenPrint = 0
            val alone = BooleanArray(n) {
                it > 0 && (members[groupOf[it]] == 1 || (pen != null && members[groupOf[it]] <= MAX_LEFTOVER_GROUP && allAtPen[groupOf[it]])) &&
                    !underPen[it] && (pen == null || touchesPen[it]) &&
                    h(it) <= MAX_SIZE * cap && w(it) <= MAX_SIZE * cap &&
                    max(h(it), w(it)) >= MIN_SIZE * cap &&
                    (sumPr[it] / area(it) < SURE_PRINT).also { p -> if (!p) seenPrint++ }
            }
            // a letter or digit after all: a twin among the print that stands with other print
            var glyph = 0
            if (glyphs) {
                val ref = (1 until n).filter { !alone[it] && members[groupOf[it]] > 1 && area(it) >= minArea &&
                    h(it) <= MAX_SIZE * cap && w(it) <= MAX_SIZE * cap }.sortedBy { h(it) }
                val refH = ref.map { h(it) }
                fun x(i: Int) = st[i * 5 + Imgproc.CC_STAT_LEFT]
                fun y(i: Int) = st[i * 5 + Imgproc.CC_STAT_TOP]
                fun iou(i: Int, j: Int): Double {
                    var inter = 0; var uni = 0
                    for (yy in 0 until h(i)) for (xx in 0 until w(i)) {
                        val a = lab[(y(i) + yy) * W + x(i) + xx] == i
                        val b = lab[(y(j) + yy * h(j) / h(i)) * W + x(j) + xx * w(j) / w(i)] == j
                        if (a && b) inter++; if (a || b) uni++
                    }
                    return if (uni == 0) 0.0 else inter.toDouble() / uni
                }
                fun hasTwin(i: Int): Boolean {
                    val hLo = Math.floor(h(i) * 0.85).toInt(); val hHi = Math.ceil(h(i) * 1.18).toInt()
                    var k = refH.binarySearch(hLo).let { if (it < 0) -it - 1 else it }
                    while (k > 0 && refH[k - 1] >= hLo) k--
                    while (k < ref.size && refH[k] <= hHi) {
                        val j = ref[k]; k++
                        if (w(j) < 0.82 * w(i) || w(j) > 1.22 * w(i)) continue
                        if (iou(i, j) >= TWIN_IOU) return true
                    }
                    return false
                }
                for (i in 1 until n) if (alone[i] && hasTwin(i)) { alone[i] = false; glyph++ }
            }
            var pieces = 0; var pixels = 0
            for (i in 1 until n) if (alone[i]) { pieces++; pixels += area(i) }
            report = "isolated: text %.0f px, reach %d, pieces %d (%d px), kept as glyphs %d, as model print %d\n".format(cap, r, pieces, pixels, glyph, seenPrint)
            if (pieces == 0) return

            val hb = ByteArray(W * H); handwriting.get(0, 0, hb)
            val pb = ByteArray(W * H); print.get(0, 0, pb)
            for (p in lab.indices) if (alone[lab[p]]) { hb[p] = 255.toByte(); pb[p] = 0 }
            handwriting.put(0, 0, hb); print.put(0, 0, pb)
        } finally { t.forEach { it.release() } }
    }
}
