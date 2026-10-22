package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.math.abs
import kotlin.math.min
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

    /** Dev switch (InkHarness `-e amid 0`): pieces amid the writing (case 2). */
    @JvmStatic var amidPen = true
    /** …taken when the handwriting around them outweighs the other print this many times. */
    @JvmStatic var AMID_RATIO = 3.0
    /** …with at most this much print beside it (times its own area)… */
    @JvmStatic var AMID_PRINT = 0.5
    /** …standing clear of the pen by this many pixels (touching it is case 1: print the pen crossed)… */
    private const val AMID_CLEAR = 2
    /** …and not crossed: the pen in this share of its box's columns. */
    private const val AMID_CROSSED = 0.6
    /** …at least this many text heights tall (an answer line's dashes are not judged). */
    private const val AMID_MIN_H = 0.55
    /** Chroma by which the writing around a piece outshines it when the pen is coloured. */
    private const val AMID_COLOUR = 4.0
    /** A printed twin for a piece amid the writing: closer, since a thin pen stroke loosely
     *  matches many printed strokes. */
    private const val AMID_TWIN_IOU = 0.75
    /** …unless it is a sign the page repeats and the model reads this surely as print. */
    @JvmStatic var AMID_KEEP_PRINT = 0.45
    /** …and at most this many text heights wide (a letter and its comma). */
    private const val AMID_MAX_W = 2.5

    /** Dev switch (InkHarness `-e isoglyph 0`): no twin test. */
    @JvmStatic var glyphs = true
    /** Dev switch (InkHarness `-e isonear 0`): judge every piece on the page, not only the pen's leftovers. */
    @JvmStatic var nearPen = true
    /** A leftover lies within this many stroke units of the found handwriting. */
    private const val LEFTOVER_UNITS = 2
    /** A group of up to this many pieces, all at the pen, is the pen's leftovers too. */
    @JvmStatic var MAX_LEFTOVER_GROUP = 1
    /** Dev probe (InkHarness `-e isoprobe x,y`): the case-2 decision on the piece at that pixel. */
    @JvmStatic var probe: IntArray? = null
    @JvmStatic var probeLog = ""
    /** Dev counters of the last run (InkHarness report). */
    @JvmStatic var report = ""

    /** [handwriting] / [print] / [overlap]: CV_8UC1 255 = set, updated in place. [modelPrint]: CV_32F. */
    fun apply(handwriting: Mat, print: Mat, overlap: Mat, modelPrint: Mat, minArea: Int, src: Mat? = null) {
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
            var twinOf: (Int) -> Boolean = { false }
            var twinFar: (Int, (Int) -> Boolean) -> Boolean = { _, _ -> false }
            var glyph = 0
            run {
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
                fun hasTwinAt(i: Int, need: Double): Boolean {
                    val hLo = Math.floor(h(i) * 0.85).toInt(); val hHi = Math.ceil(h(i) * 1.18).toInt()
                    var k = refH.binarySearch(hLo).let { if (it < 0) -it - 1 else it }
                    while (k > 0 && refH[k - 1] >= hLo) k--
                    while (k < ref.size && refH[k] <= hHi) {
                        val j = ref[k]; k++
                        if (w(j) < 0.82 * w(i) || w(j) > 1.22 * w(i)) continue
                        if (abs(area(j) - area(i)) > 0.25 * area(i)) continue
                        if (iou(i, j) >= need) return true
                    }
                    return false
                }
                fun hasTwin(i: Int): Boolean = hasTwinAt(i, TWIN_IOU)
                twinOf = { i -> hasTwinAt(i, AMID_TWIN_IOU) }
                twinFar = { i, far -> 
                    val hLo = Math.floor(h(i) * 0.85).toInt(); val hHi = Math.ceil(h(i) * 1.18).toInt()
                    (1 until n).any { j -> j != i && far(j) && h(j) in hLo..hHi && w(j) >= 0.82 * w(i) && w(j) <= 1.22 * w(i) &&
                        abs(area(j) - area(i)) <= 0.25 * area(i) && iou(i, j) >= AMID_TWIN_IOU }
                }
                if (glyphs) for (i in 1 until n) if (alone[i] && hasTwin(i)) { alone[i] = false; glyph++ }
            }
            // case 2 — a piece amid the writing: on its row, within a text height, the ink around it is
            // handwriting with hardly any print (a black-pen "Y" stroke or "s" of "Yes, please" beside the
            // printed arrow). The pen's colour and the model both took it for print, and its shape may
            // even match a printed letter — the place decides. Not where the pen surrounds it (a circled
            // printed letter) and not print under the pen (case 1, handled upstream).
            var amid = 0
            val dbgAll = StringBuilder()
            if (amidPen && pen != null) {
                val hb0 = ByteArray(W * H); handwriting.get(0, 0, hb0)
                val ab = ByteArray(W * H); allPrint.get(0, 0, ab)
                fun integral(set: (Int) -> Boolean): IntArray {
                    val s2 = IntArray((W + 1) * (H + 1))
                    for (yy in 0 until H) { var row = 0
                        for (xx in 0 until W) { if (set(yy * W + xx)) row++; s2[(yy + 1) * (W + 1) + xx + 1] = s2[yy * (W + 1) + xx + 1] + row } }
                    return s2
                }
                val hwI = integral { hb0[it].toInt() != 0 }
                // print beside it that is text: letter-tall pieces (the dots and dashes of an answer line
                // under the writing are not a word)
                val prI = integral { val l = lab[it]; l != 0 && h(l) >= AMID_MIN_H * cap }
                fun count(s2: IntArray, x0: Int, y0: Int, x1: Int, y1: Int): Int {
                    val a0 = max(0, x0); val b0 = max(0, y0); val a1 = min(W, x1); val b1 = min(H, y1)
                    if (a1 <= a0 || b1 <= b0) return 0
                    return s2[b1 * (W + 1) + a1] - s2[b0 * (W + 1) + a1] - s2[b1 * (W + 1) + a0] + s2[b0 * (W + 1) + a0]
                }
                fun x(i: Int) = st[i * 5 + Imgproc.CC_STAT_LEFT]
                fun y(i: Int) = st[i * 5 + Imgproc.CC_STAT_TOP]
                val gx = cap.roundToInt(); val gy = (0.3 * cap).roundToInt(); val half = (0.6 * cap).roundToInt()
                val px = (0.25 * cap).roundToInt()   // a letter gap: the next letter of a printed word is closer
                // within a text height of the found writing (a stroke cut off from its word sits a gap away)
                val reachPen = own(Mat())
                Imgproc.dilate(handwriting, reachPen, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(2.0 * gx + 1, 2.0 * gx + 1)))
                val rp = ByteArray(W * H); reachPen.get(0, 0, rp)
                val near0 = BooleanArray(n)
                for (p in lab.indices) { val l = lab[p]; if (l != 0 && rp[p].toInt() != 0) near0[l] = true }
                fun nearWriting0(i: Int) = near0[i]
                // touching the pen: print the pen crossed or struck (case 1) — never judged here
                val touchPen = own(Mat())
                Imgproc.dilate(handwriting, touchPen, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(2.0 * AMID_CLEAR + 1, 2.0 * AMID_CLEAR + 1)))
                val tp = ByteArray(W * H); touchPen.get(0, 0, tp)
                // crossed by the pen: the pen runs through the piece's box across most of its width (a
                // strike through a printed word, a circle through a letter) — case 1. The pen letters
                // beside a pen letter's stray stroke only touch the box at its edges
                val hbX = ByteArray(W * H); handwriting.get(0, 0, hbX)
                val obX = ByteArray(W * H); overlap.get(0, 0, obX)   // the pen over print counts as the pen
                // connected to the writing: the pen within the clear distance (the rest of its letter)
                val connected = BooleanArray(n)
                for (p in lab.indices) { val l = lab[p]; if (l != 0 && tp[p].toInt() != 0) connected[l] = true }
                val touchesInk = BooleanArray(n) { i ->
                    if (i == 0) return@BooleanArray false
                    var cols = 0
                    for (xx in x(i) until x(i) + w(i)) {
                        var hit = false
                        for (yy in y(i) until y(i) + h(i)) { val q = yy * W + xx
                            if ((hbX[q].toInt() != 0 || obX[q].toInt() != 0) && lab[q] != i) { hit = true; break } }
                        if (hit) cols++
                    }
                    cols >= AMID_CROSSED * w(i)
                }
                // ink colour: chroma (distance from grey in Lab ab) per pixel, when the page is given
                var chromaOf: ((Int) -> Double)? = null
                var hwChroma: (Int, Int, Int, Int) -> Double? = { _, _, _, _ -> null }
                var printChroma = 0.0
                if (src != null && src.size() == handwriting.size()) {
                    val lab3 = own(Mat()); Imgproc.cvtColor(src, lab3, Imgproc.COLOR_BGR2Lab)
                    val lb = ByteArray(W * H * 3); lab3.get(0, 0, lb)
                    val ch = FloatArray(W * H) { val a = (lb[it * 3 + 1].toInt() and 0xff) - 128.0; val b = (lb[it * 3 + 2].toInt() and 0xff) - 128.0; sqrt(a * a + b * b).toFloat() }
                    val sumC = DoubleArray(n); val cntC = IntArray(n)
                    for (p in lab.indices) { val l = lab[p]; if (l != 0) { sumC[l] += ch[p]; cntC[l]++ } }
                    chromaOf = { i -> sumC[i] / max(1, cntC[i]) }
                    // the page's print colour: letter-tall print standing well clear of the pen
                    var sp = 0.0; var np = 0
                    for (i in 1 until n) if (!nearWriting0(i) && h(i) >= MIN_SIZE * cap) { sp += sumC[i]; np += cntC[i] }
                    printChroma = if (np > 0) sp / np else 0.0
                    val hbC = ByteArray(W * H); handwriting.get(0, 0, hbC)
                    hwChroma = { x0, y0, x1, y1 ->
                        var s0 = 0.0; var c0 = 0
                        for (yy in max(0, y0) until min(H, y1)) for (xx in max(0, x0) until min(W, x1)) {
                            val q = yy * W + xx; if (hbC[q].toInt() != 0) { s0 += ch[q]; c0++ } }
                        if (c0 >= 10) s0 / c0 else null
                    }
                }
                val nearWriting = BooleanArray(n)
                for (p in lab.indices) { val l = lab[p]; if (l != 0 && rp[p].toInt() != 0) nearWriting[l] = true }
                val dbg = StringBuilder()
                fun probeHit(i: Int) = probe?.let { (px0, py0) -> px0 >= x(i) && px0 < x(i) + w(i) && py0 >= y(i) && py0 < y(i) + h(i) } ?: false
                for (i in 1 until n) {
                    if (probeHit(i)) dbg.append("probe piece x${x(i)} y${y(i)} w${w(i)} h${h(i)} a${area(i)} alone=${alone[i]} under=${underPen[i]} near=${nearWriting[i]} touch=${touchesInk[i]} cap=$cap\n")
                    if (alone[i] || underPen[i] || !nearWriting[i] || !connected[i] || touchesInk[i]) continue
                    // letter-sized and letter-tall: a rule's dash between the words is not judged here
                    if (h(i) < AMID_MIN_H * cap || h(i) > MAX_SIZE * cap || w(i) > AMID_MAX_W * cap || w(i) > 3 * h(i)) continue
                    val y0 = y(i) - gy; val y1 = y(i) + h(i) + gy
                    val hwN = count(hwI, x(i) - gx, y0, x(i) + w(i) + gx, y1)
                    // hardly any print right beside it: a printed letter has its word next to it
                    val prN = count(prI, x(i) - px, y0, x(i) + w(i) + px, y1) - area(i)
                    if (probeHit(i)) dbg.append("  hwN=$hwN prN=$prN pr=${sumPr[i] / area(i)}\n")
                    if (hwN < 2 * area(i) || prN > AMID_PRINT * area(i)) continue
                    // surrounded by the pen (a circle round a printed letter): pen above AND below it, or
                    // on all four sides within a text height
                    val above = count(hwI, x(i), y(i) - half, x(i) + w(i), y(i)) > 0
                    val below = count(hwI, x(i), y(i) + h(i), x(i) + w(i), y(i) + h(i) + half) > 0
                    if (above && below) continue
                    val up = count(hwI, x(i), y(i) - half, x(i) + w(i), y(i)) > 0
                    val down = count(hwI, x(i), y(i) + h(i), x(i) + w(i), y(i) + h(i) + half) > 0
                    val left = count(hwI, x(i) - half, y(i), x(i), y(i) + h(i)) > 0
                    val right = count(hwI, x(i) + w(i), y(i), x(i) + w(i) + half, y(i) + h(i)) > 0
                    if (up && down && left && right) { if (probeHit(i)) dbg.append("  surrounded\n"); continue }
                    // not the writing's ink: the pen around it is coloured (a red circle, a purple tick)
                    // while the piece is grey like the print — a printed letter the coloured pen marked
                    val cp = chromaOf?.invoke(i)
                    val cw = hwChroma(x(i) - gx, y0, x(i) + w(i) + gx, y1)
                    if (cp != null && cw != null && cw > max(cp, printChroma) + AMID_COLOUR) {
                        if (probeHit(i)) dbg.append("  colour pen $cw vs piece $cp\n"); continue }
                    // the model sure it is print (a printed letter the pen ran under): kept
                    if (sumPr[i] / area(i) >= AMID_KEEP_PRINT) { if (probeHit(i)) dbg.append("  model print\n"); continue }
                    // a letter or sign the page prints elsewhere (its arrows, its slashes) stays print; a
                    // pen stroke has no printed twin (twins only among print clear of the writing: a pen
                    // stroke the detectors missed twice — the "Y" of two "yes" answers — is no printed letter)
                    if (twinFar(i) { j -> !near0[j] }) { if (probeHit(i)) dbg.append("  twin\n"); continue }
                    if (probeHit(i)) dbg.append("  taken\n")
                    alone[i] = true; amid++
                }
                dbgAll.append(dbg)
            }
            var pieces = 0; var pixels = 0
            for (i in 1 until n) if (alone[i]) { pieces++; pixels += area(i) }
            probeLog = dbgAll.toString()
            report = "isolated: text %.0f px, reach %d, pieces %d (%d px), kept as glyphs %d, as model print %d, amid pen %d\n".format(cap, r, pieces, pixels, glyph, seenPrint, amid)
            if (pieces == 0) return

            val hb = ByteArray(W * H); handwriting.get(0, 0, hb)
            val pb = ByteArray(W * H); print.get(0, 0, pb)
            for (p in lab.indices) if (alone[lab[p]]) { hb[p] = 255.toByte(); pb[p] = 0 }
            handwriting.put(0, 0, hb); print.put(0, 0, pb)
        } finally { t.forEach { it.release() } }
    }
}
