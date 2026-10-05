package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.stream.IntStream
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Re-typesets printed letters the pen crossed, from the page's OWN glyphs — no text recognition.
 * Kotlin port of RetypesetPrint in ios/Runner/InkAnalysis.hpp (see there for the reasoning and
 * measurements): the erased page's clean letters form a glyph bank; for each printed line segment
 * the handwriting touched, a dynamic programme places the bank glyphs that best explain the print
 * still visible in the original (pen-coloured ink is unknown). Two passes — the majority first
 * (strongly confirmed glyphs), then the exceptions with the first pass's glyphs as evidence. A
 * placed glyph fills only erased pixels, with its own pixels. All per-pixel work stays in OpenCV.
 */
object PrintRetypeset {
    private const val CLIPPED_OD = 1.6

    private class Member(val i: Int, val x: Int, val y: Int, val w: Int, val h: Int, val clean: Boolean)
    private class Line(val m: ArrayList<Member>, var h: Float, var cy: Float) { var base = 0; var cap = 0 }
    private class Glyph(val G: Mat, val B: Mat, val srcX: Int, val srcY: Int, val dy: Int, val cap: Int)
    private class Key(val gi: Int, val jit: Int) { var s: FloatArray? = null }

    private fun medianF(v: List<Float>): Float = v.sorted()[v.size / 2]

    private fun inkMap(bgr: Mat): Mat {
        val g = Mat(); Imgproc.cvtColor(bgr, g, Imgproc.COLOR_BGR2GRAY)
        val small = Mat(); Imgproc.resize(g, small, Size(), 0.25, 0.25, Imgproc.INTER_AREA)
        Imgproc.medianBlur(small, small, min(21, (min(small.cols(), small.rows()) - 1) or 1))
        val paper = Mat(); Imgproc.resize(small, paper, g.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val pf = Mat(); paper.convertTo(pf, CvType.CV_32F)
        val gf = Mat(); g.convertTo(gf, CvType.CV_32F)
        val out = Mat(); Core.subtract(pf, gf, out)
        Core.multiply(out, Scalar(1.0 / 140.0), out)
        Core.max(out, Scalar(0.0), out); Core.min(out, Scalar(1.0), out)
        listOf(g, small, paper, pf, gf).forEach { it.release() }
        return out
    }

    /** Median of [m] over [mask] on a half-resolution grid (memory: a 12 MP page stays native). */
    private fun medianOf(m: Mat, mask: Mat): Float {
        val ms = Mat(); val ks = Mat()
        Imgproc.resize(m, ms, Size(), 0.5, 0.5, Imgproc.INTER_NEAREST)
        Imgproc.resize(mask, ks, Size(), 0.5, 0.5, Imgproc.INTER_NEAREST)
        val v = FloatArray(ms.total().toInt()); ms.get(0, 0, v)
        val k = ByteArray(ks.total().toInt()); ks.get(0, 0, k)
        ms.release(); ks.release()
        val sel = ArrayList<Float>()
        for (i in v.indices) if (k[i].toInt() != 0) sel.add(v[i])
        return if (sel.isEmpty()) 1f / 3 else medianF(sel)
    }

    /** Dev switch for comparing with/without re-typesetting (InkHarness `-e retypeset 0`). */
    @JvmStatic var enabled = true

    fun retypeset(dst: Mat, analysis: Mat, handwriting: Mat, printMask: Mat, k: Int) {
        if (!enabled) return
        if (Core.countNonZero(handwriting) == 0) return
        val owned = ArrayList<Mat>()
        fun <T : Mat> own(m: T): T { owned.add(m); return m }
        fun and(a: Mat, b: Mat) = own(Mat()).also { Core.bitwise_and(a, b, it) }
        fun or(a: Mat, b: Mat) = own(Mat()).also { Core.bitwise_or(a, b, it) }
        fun not(a: Mat) = own(Mat()).also { Core.bitwise_not(a, it) }
        fun cmp(a: Mat, v: Double, op: Int) = own(Mat()).also { Core.compare(a, Scalar(v), it, op) }
        try {
            val W = dst.cols(); val H = dst.rows()
            val unit = k / 5.0f
            val inkO = own(inkMap(analysis)); val inkE = own(inkMap(dst))

            // --- per-pixel colour: pen-coloured ink is unknown
            val paperSmall = own(InkRefine.paperLowRes(analysis))
            val d = InkRefine.densityInRoi(analysis, paperSmall, Rect(0, 0, W, H))
            owned.addAll(listOf(d.odB, d.odG, d.odR, d.mean, d.ink))
            val clipped = and(d.ink, cmp(d.mean, CLIPPED_OD, Core.CMP_GT))
            val coloured = and(d.ink, not(clipped))
            val cw = own(Mat()); coloured.convertTo(cw, CvType.CV_32F, 1.0 / 255.0)
            val w3 = own(Mat()); Imgproc.boxFilter(cw, w3, -1, Size(3.0, 3.0), Point(-1.0, -1.0), false)
            val sums = listOf(d.odB, d.odG, d.odR).map { od ->
                val t = own(Mat()); Core.multiply(od, cw, t)
                own(Mat()).also { Imgproc.boxFilter(t, it, -1, Size(3.0, 3.0), Point(-1.0, -1.0), false) }
            }
            val tot = own(Mat()); Core.add(sums[0], sums[1], tot); Core.add(tot, sums[2], tot)
            Core.max(tot, Scalar(1e-6), tot)
            val chroma = sums.map { own(Mat()).also { c -> Core.divide(it, tot, c) } }
            val decided = and(cmp(w3, 2.0, Core.CMP_GE), d.ink)
            val notHw = not(handwriting)
            val refPr = FloatArray(3); val refPen = FloatArray(3); val dir = FloatArray(3)
            var norm = 0f
            for (c in 0 until 3) {
                refPr[c] = medianOf(chroma[c], and(decided, notHw))
                refPen[c] = medianOf(chroma[c], and(decided, handwriting))
                dir[c] = refPen[c] - refPr[c]; norm += dir[c] * dir[c]
            }
            norm = sqrt(max(norm, 1e-12f))
            for (c in 0 until 3) dir[c] /= norm
            val thr = 0.5f * norm
            if (thr < 0.004f) return   // no distinguishable pen colour on this page
            val proj = own(Mat.zeros(H, W, CvType.CV_32F))
            for (c in 0 until 3) {
                val t = own(Mat()); Core.subtract(chroma[c], Scalar(refPr[c].toDouble()), t)
                Core.scaleAdd(t, dir[c].toDouble(), proj, proj)
            }
            val penPx = and(decided, cmp(proj, thr.toDouble(), Core.CMP_GT))
            val unknown = own(Mat()); Imgproc.dilate(penPx, unknown, Mat.ones(3, 3, CvType.CV_8U))
            val penNear = own(Mat()); Imgproc.dilate(penPx, penNear, Mat.ones(5, 5, CvType.CV_8U))
            Core.bitwise_or(unknown, and(clipped, penNear), unknown)
            // handwriting that touches no printed TEXT (an answer in its own cell; rules do not
            // count) holds no print to re-typeset: all of it is unknown, whatever its colour
            run {
                val lr = max(9, 6 * k) or 1
                val hR = own(Mat()); val vR = own(Mat())
                Imgproc.morphologyEx(printMask, hR, Imgproc.MORPH_OPEN, Mat.ones(1, lr, CvType.CV_8U))
                Imgproc.morphologyEx(printMask, vR, Imgproc.MORPH_OPEN, Mat.ones(lr, 1, CvType.CV_8U))
                val rulesM = or(hR, vR); Imgproc.dilate(rulesM, rulesM, Mat.ones(3, 3, CvType.CV_8U))
                val textNear = own(Mat()); Imgproc.dilate(and(printMask, not(rulesM)), textNear, Mat.ones(5, 5, CvType.CV_8U))
                val lab = own(Mat()); val st = own(Mat()); val ce = own(Mat())
                val nh = Imgproc.connectedComponentsWithStats(handwriting, lab, st, ce, 8, CvType.CV_32S)
                val la = IntArray(lab.total().toInt()); lab.get(0, 0, la)
                val tn = ByteArray(textNear.total().toInt()); textNear.get(0, 0, tn)
                val touch = IntArray(nh)
                for (i in la.indices) if (la[i] > 0 && tn[i].toInt() != 0) touch[la[i]]++
                val out = ByteArray(la.size) { if (la[it] > 0 && touch[la[it]] < 2) 255.toByte() else 0 }
                val alone = own(Mat(lab.size(), CvType.CV_8U)); alone.put(0, 0, out)
                Imgproc.dilate(alone, alone, Mat.ones(3, 3, CvType.CV_8U))
                Core.bitwise_or(unknown, alone, unknown)
            }
            var known = not(unknown)
            val erasedArea = own(Mat()); Imgproc.dilate(handwriting, erasedArea, Mat.ones(5, 5, CvType.CV_8U))

            // --- glyph bank: clean components of the erased page, grouped into text lines
            val binE = cmp(inkE, 0.45, Core.CMP_GT)
            val labels = own(Mat()); val stats = own(Mat()); val centroids = own(Mat())
            val n = Imgproc.connectedComponentsWithStats(binE, labels, stats, centroids, 8, CvType.CV_32S)
            val nearErase = own(Mat()); Imgproc.dilate(erasedArea, nearErase, Mat.ones(5, 5, CvType.CV_8U))
            val comps = ArrayList<Member>()
            for (i in 1 until n) {
                val x = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt(); val y = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
                val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt(); val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
                val area = stats.get(i, Imgproc.CC_STAT_AREA)[0]
                if (h < 6 * unit || h > 60 * unit || w > 60 * unit || area < 8 * unit * unit) continue
                val sub = nearErase.submat(Rect(x, y, w, h))
                val clean = Core.countNonZero(sub) == 0
                sub.release()
                comps.add(Member(i, x, y, w, h, clean))
            }
            comps.sortBy { it.y + it.h / 2f }
            val lines = ArrayList<Line>()
            for (m in comps) {
                val cy = m.y + m.h / 2f
                val L = lines.firstOrNull { abs(cy - it.cy) < 0.5f * it.h && abs(m.h - it.h) < 0.8f * it.h }
                if (L != null) {
                    L.m.add(m)
                    L.h = L.m.map { it.h.toFloat() }.sorted()[L.m.size / 2]
                    L.cy = medianF(L.m.map { it.y + it.h / 2f })
                } else lines.add(Line(arrayListOf(m), m.h.toFloat(), cy))
            }
            lines.removeAll { it.m.size < 5 }
            for (L in lines) {
                val tall = L.m.filter { it.h >= 0.7f * L.h }
                L.base = medianF(tall.map { (it.y + it.h).toFloat() }).toInt()
                L.cap = medianF(tall.map { it.h.toFloat() }).toInt()
            }
            val bank = ArrayList<Glyph>()
            run {
                // near-duplicates cost time and add nothing: greedy clusters, two kept each
                val repFirst = ArrayList<Int>(); val repCount = ArrayList<Int>()
                for (L in lines) for (m in L.m) {
                    if (!m.clean) continue
                    val x0 = max(0, m.x - 1); val y0 = max(0, m.y - 1); val x1 = min(W, m.x + m.w + 1); val y1 = min(H, m.y + m.h + 1)
                    val r = Rect(x0, y0, x1 - x0, y1 - y0)
                    val lab = labels.submat(r)
                    val ownM = Mat(); Core.compare(lab, Scalar(m.i.toDouble()), ownM, Core.CMP_EQ); lab.release()
                    Imgproc.dilate(ownM, ownM, Mat.ones(3, 3, CvType.CV_8U))
                    val G = Mat.zeros(r.size(), CvType.CV_32F)
                    val src = inkE.submat(r); src.copyTo(G, ownM); src.release(); ownM.release()
                    val B = Mat(); Core.compare(G, Scalar(0.35), B, Core.CMP_GT)
                    val g = Glyph(G, B, x0, y0, y0 - L.base, L.cap)
                    var dup = false
                    for (ri in repFirst.indices) {
                        val q = bank[repFirst[ri]]
                        if (abs(q.B.rows() - B.rows()) > 1 || abs(q.B.cols() - B.cols()) > 1 || abs(q.dy - g.dy) > 1 || q.cap != g.cap) continue
                        val hh = min(q.B.rows(), B.rows()); val ww = min(q.B.cols(), B.cols())
                        val a = q.B.submat(Rect(0, 0, ww, hh)); val b = B.submat(Rect(0, 0, ww, hh))
                        val i2 = Mat(); Core.bitwise_and(a, b, i2); val u2 = Mat(); Core.bitwise_or(a, b, u2)
                        val inter = Core.countNonZero(i2); val uni = Core.countNonZero(u2)
                        listOf(a, b, i2, u2).forEach { it.release() }
                        if (uni > 0 && inter >= 0.8 * uni) {
                            if (repCount[ri] < 2) { repCount[ri]++; bank.add(g) } else { G.release(); B.release() }
                            dup = true
                            break
                        }
                    }
                    if (!dup) { repFirst.add(bank.size); repCount.add(1); bank.add(g) }
                }
            }
            if (bank.isEmpty()) return
            bank.forEach { owned.add(it.G); owned.add(it.B) }

            // --- two passes over the damaged line segments
            val source = own(dst.clone())
            val sourceGray = own(Mat()); Imgproc.cvtColor(source, sourceGray, Imgproc.COLOR_BGR2GRAY)
            var printM = own(printMask.clone())
            val painted = own(Mat.zeros(H, W, CvType.CV_8U)); val paintedBoxes = own(Mat.zeros(H, W, CvType.CV_8U))
            val firstPass = HashMap<Pair<Int, Int>, List<IntArray>>()
            for (pass in 1..2) {
                if (pass == 2) {
                    // the first pass's glyphs are now certain print and count as evidence
                    inkO.setTo(Scalar(1.0), painted)
                    known = or(known, painted)
                    printM = or(printM, painted)
                }
                // ink that could ever confirm a glyph (visible, print by detector or colour, not a
                // rule): segments without any are skipped before the costly fit
                var evidence = and(and(known, cmp(inkO, 0.35, Core.CMP_GT)),
                    or(printM, and(decided, cmp(proj, 0.2 * thr, Core.CMP_LE))))
                run {
                    val runs = own(Mat())
                    Imgproc.morphologyEx(evidence, runs, Imgproc.MORPH_OPEN, Mat.ones(1, max(9, 6 * k) or 1, CvType.CV_8U))
                    Imgproc.dilate(runs, runs, Mat.ones(3, 1, CvType.CV_8U))
                    evidence = and(evidence, not(runs))
                }
                val ev01 = own(Mat()); Core.divide(evidence, Scalar(255.0), ev01)
                val textEvidence = and(and(evidence, printM), not(handwriting))
                val evidenceInt = own(Mat()); Imgproc.integral(ev01, evidenceInt, CvType.CV_32S)
                fun evSum(x0: Int, y0: Int, x1: Int, y1: Int): Int =
                    (evidenceInt.get(y1, x1)[0] - evidenceInt.get(y0, x1)[0] - evidenceInt.get(y1, x0)[0] + evidenceInt.get(y0, x0)[0]).toInt()

                for (L in lines) {
                    val top = max(0, L.base - (1.5f * L.cap).toInt() - 2); val bot = min(H, L.base + (0.5f * L.cap).toInt() + 2)
                    if (bot - top < 4) continue
                    val cols = ByteArray(W)
                    run {
                        val band = handwriting.rowRange(top, bot); val red = Mat()
                        Core.reduce(band, red, 0, Core.REDUCE_MAX); red.get(0, 0, cols); band.release(); red.release()
                    }
                    val spans = ArrayList<IntArray>()
                    var s0 = -1; var prev = -1
                    for (x in 0 until W) {
                        if (cols[x].toInt() == 0) continue
                        if (s0 < 0) s0 = x else if (x - prev > L.cap) { spans.add(intArrayOf(s0, prev)); s0 = x }
                        prev = x
                    }
                    if (s0 >= 0) spans.add(intArrayOf(s0, prev))
                    val cand = bank.indices.filter { bank[it].cap >= 0.8f * L.cap && bank[it].cap <= 1.25f * L.cap }
                    if (cand.isEmpty()) continue
                    for (sp in spans) {
                        var a = max(0, sp[0] - L.cap); var b = min(W - 1, sp[1] + L.cap)
                        repeat(3) {
                            for (m in L.m) if (m.x < b + 0.5f * L.cap && m.x + m.w > a - 0.5f * L.cap) {
                                a = max(0, min(a, m.x - 2)); b = min(W - 1, max(b, m.x + m.w + 2))
                            }
                        }
                        val strip = Rect(a, top, b - a + 1, bot - top)
                        val ea = erasedArea.submat(strip); val anyErased = Core.countNonZero(ea) > 0; ea.release()
                        if (!anyErased) continue
                        if (evSum(a, top, b + 1, bot) < 12f * (L.cap / 15f) * (L.cap / 15f)) continue
                        val segKey = Pair(top, a)
                        val pst = painted.submat(strip); val paintedHere = Core.countNonZero(pst) > 0; pst.release()
                        val reuse = pass == 2 && firstPass.containsKey(segKey) && !paintedHere
                        val width = strip.width
                        val placements: List<IntArray>
                        if (reuse) {
                            placements = firstPass[segKey]!!
                        } else {
                            val T = inkO.submat(strip)
                            val K = Mat(); val kn = known.submat(strip); kn.convertTo(K, CvType.CV_32F, 1.0 / 255.0); kn.release()
                            val KT = Mat(); Core.multiply(K, T, KT)
                            val tMask = Mat(); val tm8 = Mat(); Core.compare(T, Scalar(0.35), tm8, Core.CMP_GT); tm8.convertTo(tMask, CvType.CV_32F, 1.0 / 255.0)
                            val oneMinusK = Mat(); Core.subtract(Mat.ones(K.size(), CvType.CV_32F), K, oneMinusK)
                            val UT = Mat(); Core.multiply(oneMinusK, tMask, UT)
                            val colPenM = Mat(); Core.reduce(KT, colPenM, 0, Core.REDUCE_SUM, CvType.CV_32F)
                            val colPen = FloatArray(width); colPenM.get(0, 0, colPen)
                            val colPenCum = FloatArray(width + 1)
                            for (x in 0 until width) colPenCum[x + 1] = colPenCum[x] + colPen[x]
                            // one correlation with 3 KT - K + 0.3 UT gives the score, a strip 2 px
                            // taller gives all three vertical jitters at once
                            val M = Mat(); Core.addWeighted(KT, 3.0, K, -1.0, 0.0, M); Core.scaleAdd(UT, 0.3, M, M)
                            val table = ArrayList<Key>()
                            val firstKey = IntArray(bank.size) { -1 }
                            for (gi in cand) {
                                if (bank[gi].G.cols() > width) continue
                                for (jit in -1..1) {
                                    val gy = L.base + bank[gi].dy + jit - top
                                    if (gy < 0 || gy + bank[gi].G.rows() > bot - top) continue
                                    if (firstKey[gi] < 0) firstKey[gi] = table.size
                                    table.add(Key(gi, jit))
                                }
                            }
                            val glyphs = cand.filter { firstKey[it] >= 0 }
                            IntStream.range(0, glyphs.size).parallel().forEach { gn ->
                                val gi = glyphs[gn]; val g = bank[gi]
                                val gh = g.G.rows(); val gw = g.G.cols()
                                val y0 = L.base + g.dy - 1 - top
                                val ya = max(0, y0); val yb = min(bot - top, y0 + gh + 2)
                                if (yb - ya >= gh) {
                                    val sub = M.submat(Rect(0, ya, width, yb - ya)); val res = Mat()
                                    Imgproc.matchTemplate(sub, g.G, res, Imgproc.TM_CCORR)
                                    val rowBuf = FloatArray(res.cols())
                                    var ki = firstKey[gi]
                                    while (ki < table.size && table[ki].gi == gi) {
                                        val row = y0 + (table[ki].jit + 1) - ya
                                        if (row in 0 until res.rows()) {
                                            res.get(row, 0, rowBuf)
                                            table[ki].s = FloatArray(rowBuf.size) { x -> rowBuf[x] - (colPenCum[x + gw] - colPenCum[x]) }
                                        }
                                        ki++
                                    }
                                    sub.release(); res.release()
                                }
                            }
                            // free fit: gap (1 px, pays for uncovered visible ink) or a glyph with a positive score
                            val NEG = -1e30f
                            val score = FloatArray(width + 1) { NEG }; val backX = IntArray(width + 1) { -1 }; val backKey = IntArray(width + 1) { -1 }
                            score[0] = 0f
                            for (x in 0 until width) {
                                if (score[x] == NEG) continue
                                val v = score[x] - colPen[x]
                                if (v > score[x + 1]) { score[x + 1] = v; backX[x + 1] = x; backKey[x + 1] = -1 }
                                for (ki in table.indices) {
                                    val s = table[ki].s ?: continue
                                    if (x >= s.size || s[x] <= 0f) continue
                                    val gw = bank[table[ki].gi].G.cols()
                                    val v2 = score[x] + s[x]
                                    if (v2 > score[x + gw]) { score[x + gw] = v2; backX[x + gw] = x; backKey[x + gw] = ki }
                                }
                            }
                            val out = ArrayList<IntArray>()
                            var x = width
                            while (x > 0 && backX[x] >= 0) {
                                if (backKey[x] >= 0) out.add(intArrayOf(backX[x], table[backKey[x]].gi, table[backKey[x]].jit))
                                x = backX[x]
                            }
                            placements = out
                            if (pass == 1) firstPass[segKey] = out
                            listOf(T, K, KT, tMask, tm8, oneMinusK, UT, colPenM, M).forEach { it.release() }
                        }
                        for (pl in placements) accept(pass, pl, a, L, bank, W, H, erasedArea, painted, paintedBoxes, known, inkO,
                            printM, decided, proj, thr, dst, source, sourceGray, textEvidence)
                    }
                }
            }
        } finally {
            owned.forEach { it.release() }
        }
    }

    /** Pass-specific acceptance of one placed glyph, and painting it (see ios RetypesetPrint). */
    private fun accept(
        pass: Int, pl: IntArray, a: Int, L: Line, bank: List<Glyph>, W: Int, H: Int, erasedArea: Mat, painted: Mat, paintedBoxes: Mat,
        known: Mat, inkO: Mat, printM: Mat, decided: Mat, proj: Mat, thr: Float, dst: Mat, source: Mat, sourceGray: Mat,
        textEvidence: Mat,
    ) {
        val g = bank[pl[1]]
        val gh = g.G.rows(); val gw = g.G.cols()
        val gx = a + pl[0]; val gy = L.base + g.dy + pl[2]
        if (gy < 0 || gy + gh > H || gx + gw > W) return
        val r = Rect(gx, gy, gw, gh)
        val tmp = ArrayList<Mat>()
        fun sub(m: Mat) = m.submat(r).also { tmp.add(it) }
        fun op(f: (Mat) -> Unit) = Mat().also { f(it); tmp.add(it) }
        try {
            var need = op { Core.bitwise_and(sub(erasedArea), g.B, it) }
            if (pass == 2) {
                if (Core.mean(sub(paintedBoxes)).`val`[0] > 0.3 * 255) return
                val np = op { Core.bitwise_not(sub(painted), it) }
                need = op { Core.bitwise_and(need, np, it) }
            }
            if (Core.countNonZero(need) == 0) return
            val kb = op { Core.bitwise_and(sub(known), g.B, it) }
            val inkHi = op { Core.compare(sub(inkO), Scalar(0.35), it, Core.CMP_GT) }
            val mt = op { Core.bitwise_and(kb, inkHi, it) }
            val visInk = Core.countNonZero(kb)
            val nm = Core.countNonZero(mt)
            if (nm == 0) return
            // the confirmed ink must span the glyph's height (a rule confirms only a bottom bar)
            val rowsHit = op { Core.reduce(mt, it, 1, Core.REDUCE_MAX) }
            val rh = ByteArray(gh); rowsHit.get(0, 0, rh)
            var r0 = -1; var r1 = -1
            for (y in 0 until gh) if (rh[y].toInt() != 0) { if (r0 < 0) r0 = y; r1 = y }
            if (r1 - r0 + 1 < 0.4 * gh) return
            val ratio = nm.toFloat() / max(1, Core.countNonZero(g.B))
            val inPrint = Core.countNonZero(op { Core.bitwise_and(mt, sub(printM), it) }).toFloat() / nm
            val md = op { Core.bitwise_and(mt, sub(decided), it) }
            val mp = if (Core.countNonZero(md) > 0) (Core.mean(sub(proj), md).`val`[0] / thr).toFloat() else 0f
            val area = (L.cap / 15f) * (L.cap / 15f)
            val ok = if (pass == 1) ratio >= 0.5f && inPrint >= 0.7f && mp <= 0.35f && nm >= 0.8f * visInk
            else {
                // (a) needs visible print text right beside the glyph on its rows (see ios)
                var besidePrint = false
                val gap = max(3, L.cap / 2)
                for (side in listOf(Rect(max(0, gx - gap), gy, min(gap, gx), gh), Rect(gx + gw, gy, min(gap, W - gx - gw), gh))) {
                    if (side.width <= 0) continue
                    val t = textEvidence.submat(side); if (Core.countNonZero(t) >= 3) besidePrint = true; t.release()
                }
                nm >= 0.6f * visInk && ((ratio >= 0.5f && mp <= 0.2f && nm >= 20 * area && besidePrint) || (inPrint >= 0.8f && nm >= 12 * area && mp <= 0.3f))
            }
            if (!ok) return
            // the glyph must leave no visible print in its box unexplained (an "e" pasted where an
            // "a" lost its bowl under the pen: the a's own stem, visible beside it, is not the e's)
            run {
                val notB = op { Core.bitwise_not(op { Imgproc.dilate(g.B, it, Mat.ones(3, 3, CvType.CV_8U)) }, it) }
                val un = op { Core.bitwise_and(op { Core.bitwise_and(kb.let { sub(known) }, inkHi, it) }, sub(printM), it) }
                Core.bitwise_and(un, notB, un)
                if (Core.countNonZero(un) > 0.12f * nm) return
            }
            // the glyph's own pixels, only where darker than what the erase left there
            val rs = Rect(g.srcX, g.srcY, gw, gh)
            val dview = sub(dst)
            val dstGray = op { Imgproc.cvtColor(dview, it, Imgproc.COLOR_BGR2GRAY) }
            val srcGray = sourceGray.submat(rs).also { tmp.add(it) }
            val darker = op { Core.compare(srcGray, dstGray, it, Core.CMP_LT) }
            val m = op { Core.bitwise_and(need, darker, it) }
            val srcView = source.submat(rs).also { tmp.add(it) }
            srcView.copyTo(dview, m)
            val pv = sub(painted); Core.bitwise_or(pv, g.B, pv)
            sub(paintedBoxes).setTo(Scalar(255.0))
        } finally {
            tmp.forEach { it.release() }
        }
    }
}
