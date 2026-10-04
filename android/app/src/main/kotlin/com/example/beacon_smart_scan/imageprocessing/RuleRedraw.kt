package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Last erase step: printed rules (answer lines, table grid) that handwriting was written on are
 * cleared and drawn again. Pen strokes fused with a rule cannot be told from it pixel by pixel, so
 * the erase leaves bits of the writing stuck to the line; instead, where the handwriting mask
 * touches a rule, the band around the rule is cleared of pen leftovers and the rule is redrawn as a
 * straight line of its own measured thickness and colour. Mirrors RuleRedraw in
 * ios/Runner/ImageProcessingOpenCV.mm.
 *
 * A rule is only taken when it is (on its parts without handwriting on them) long, thin, straight,
 * solid (a dotted line is left alone), isolated (paper right above and below — not the top or
 * bottom edge of a text line) and print in the mask (not a pen underline). Leftover pieces cleared
 * must lie in the band near the rule AND in the handwriting mask — printed letters never qualify —
 * and another rule crossing the band is never touched.
 */
object RuleRedraw {
    /**
     * A rule in "rule space": along = x (page column, or page row when [transposed]), across = y.
     * [cleanX]/[cleanY]: its points without handwriting, for its colour.
     */
    private class Rule(
        val transposed: Boolean, val a: Double, val b: Double, val th: Double, val x0: Int, val x1: Int,
        val cleanX: IntArray, val cleanY: IntArray,
    ) {
        fun y(x: Int) = a * x + b
    }

    /** Dev switch (InkHarness `-e rules 0`). */
    @JvmStatic var enabled = true

    /**
     * [original]: the unprocessed page the mask was detected on; [handwriting] / [print]: the
     * mask's layers (CV_8UC1, 255 = set); [dst]: the erased page (CV_8UC3), updated in place.
     */
    fun apply(original: Mat, handwriting: Mat, print: Mat, dst: Mat) {
        if (!enabled) return
        val w = original.cols(); val h = original.rows(); val n = w * h
        val k = InkAnalysis.strokeUnit(w, h)
        val hw = bytes(handwriting); for (i in hw.indices) hw[i] = if (hw[i].toInt() != 0) 1 else 0
        val pr = bytes(print)
        val pure = ByteArray(n) { if ((pr[it].toInt() and 0xff) > 127 && hw[it].toInt() == 0) 1 else 0 }
        val ink = originalInk(original)
        val hwOn = dilate(hw, w, h, 3)
        val rules = detect(ink, pure, hwOn, w, h, k, false) +
            detect(transpose(ink, w, h), transpose(pure, w, h), transpose(hwOn, w, h), h, w, k, true)
        if (rules.isEmpty()) return
        val hwTight = dilate(hw, w, h, 5)
        val pap = paperOf(dst)
        val out = ByteArray(n * 3); dst.get(0, 0, out)
        val org = ByteArray(n * 3); original.get(0, 0, org)

        // page index of rule-space (x, y)
        fun idx(r: Rule, x: Int, y: Int) = if (r.transposed) x * w + y else y * w + x
        fun across(r: Rule) = if (r.transposed) w else h

        // every rule's own line (protected from the other rules' clearing)
        val lineMask = ByteArray(n)
        for (r in rules) for (x in r.x0 until r.x1) {
            val yc = r.y(x); var d = -r.th / 2 - 1
            while (d <= r.th / 2 + 1.01) {
                val y = (yc + d).roundToInt()
                if (y in 0 until across(r)) lineMask[idx(r, x, y)] = 1
                d += 1.0
            }
        }
        // affected columns: handwriting in the band near the rule
        class Col(val r: Rule, val x: Int, val yc: Double, val colour: FloatArray)
        val clearZone = ByteArray(n)
        val redraw = ArrayList<Col>()
        for (r in rules) {
            val len = r.x1 - r.x0; val band = r.th / 2 + 2 * k
            val aff = BooleanArray(len)
            for (j in 0 until len) {
                val x = r.x0 + j; val yc = r.y(x)
                val y1 = max(0, (yc - band).toInt()); val y2 = min(across(r) - 1, (yc + band).toInt())
                for (y in y1..y2) if (hw[idx(r, x, y)].toInt() != 0) { aff[j] = true; break }
            }
            if (aff.none { it }) continue
            val grown = BooleanArray(len) { j -> (max(0, j - k)..min(len - 1, j + k)).any { aff[it] } }
            val colour = ruleColour(r, org, w)
            for (j in 0 until len) {
                if (!grown[j]) continue
                val x = r.x0 + j; val yc = r.y(x)
                for (d in -band.toInt()..band.toInt()) {
                    val y = (yc + d).roundToInt()
                    if (y in 0 until across(r)) clearZone[idx(r, x, y)] = 1
                }
                redraw.add(Col(r, x, yc, colour))
            }
        }
        if (redraw.isEmpty()) return

        // pen leftovers near the rules: erased-page ink pieces (rule lines aside) lying in the zone
        // and inside the handwriting mask; printed letters never qualify
        val pieces = ByteArray(n)
        for (i in 0 until n) {
            if (lineMask[i].toInt() != 0) continue
            val g = grey(out, i); val pg = grey(pap, i)
            if (pg - g > 25) pieces[i] = 1
        }
        val pm = Mat(h, w, CvType.CV_8U); pm.put(0, 0, pieces)
        val labels = Mat()
        val count = Imgproc.connectedComponents(pm, labels, 8, CvType.CV_32S)
        pm.release()
        val lab = IntArray(n); labels.get(0, 0, lab); labels.release()
        val cnt = IntArray(count); val inZone = IntArray(count); val inHw = IntArray(count); val isPure = IntArray(count)
        for (i in 0 until n) {
            val l = lab[i]; if (l == 0) continue
            cnt[l]++
            if (clearZone[i].toInt() != 0) inZone[l]++
            if (hwTight[i].toInt() != 0) inHw[l]++
            if (pure[i].toInt() != 0) isPure[l]++
        }
        val go = BooleanArray(count) { l ->
            l > 0 && isPure[l] <= 0.2 * cnt[l] &&
                ((inZone[l] >= 0.85 * cnt[l] && inHw[l] >= 0.6 * cnt[l]) || (inZone[l] >= 0.4 * cnt[l] && inHw[l] >= 0.9 * cnt[l]))
        }
        for (i in 0 until n) if (go[lab[i]]) for (c in 0..2) out[i * 3 + c] = pap[i * 3 + c]

        // redraw each affected column: the rule's own band anti-aliased, its rim cleared to paper
        // (another rule crossing here is left alone)
        for (col in redraw) {
            val r = col.r; val th = r.th
            for (y in (col.yc - th / 2 - 2).toInt()..(col.yc + th / 2 + 2).toInt()) {
                if (y !in 0 until across(r)) continue
                val i = idx(r, col.x, y)
                val cov = (min(y + 0.5, col.yc + th / 2) - max(y - 0.5, col.yc - th / 2)).coerceIn(0.0, 1.0)
                if (cov > 0) {
                    for (c in 0..2) {
                        val p = (pap[i * 3 + c].toInt() and 0xff)
                        out[i * 3 + c] = (cov * col.colour[c] + (1 - cov) * p).roundToInt().coerceIn(0, 255).toByte()
                    }
                } else if (lineMask[i].toInt() == 0 || abs(y - r.y(col.x)) <= th / 2 + 1.01) {
                    for (c in 0..2) out[i * 3 + c] = pap[i * 3 + c]
                }
            }
        }
        dst.put(0, 0, out)
    }

    private fun detect(ink: ByteArray, pure: ByteArray, hwOn: ByteArray, w: Int, h: Int, k: Int, transposed: Boolean): List<Rule> {
        val minLen = max(8 * k, w / 8)
        val run = Mat(h, w, CvType.CV_8U); run.put(0, 0, ink)
        val kc = Mat.ones(1, k, CvType.CV_8U); val ko = Mat.ones(1, minLen, CvType.CV_8U)
        Imgproc.morphologyEx(run, run, Imgproc.MORPH_CLOSE, kc)
        Imgproc.morphologyEx(run, run, Imgproc.MORPH_OPEN, ko)
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val count = Imgproc.connectedComponentsWithStats(run, labels, stats, cents, 8, CvType.CV_32S)
        val lab = IntArray(w * h); labels.get(0, 0, lab)
        val st = IntArray(count * 5); stats.get(0, 0, st)
        listOf(run, kc, ko, labels, stats, cents).forEach { it.release() }
        fun at(m: ByteArray, x: Int, y: Int) = m[y.coerceIn(0, h - 1) * w + x].toInt() != 0
        val rules = ArrayList<Rule>()
        for (l in 1 until count) {
            val x0 = st[l * 5 + Imgproc.CC_STAT_LEFT]; val y0 = st[l * 5 + Imgproc.CC_STAT_TOP]
            val bw = st[l * 5 + Imgproc.CC_STAT_WIDTH]; val bh = st[l * 5 + Imgproc.CC_STAT_HEIGHT]
            if (bw < minLen) continue
            val top = IntArray(bw) { Int.MAX_VALUE }; val bot = IntArray(bw) { -1 }
            for (y in y0 until y0 + bh) for (x in x0 until x0 + bw) if (lab[y * w + x] == l) {
                val c = x - x0; if (y < top[c]) top[c] = y; if (y > bot[c]) bot[c] = y
            }
            // clean column: no handwriting ON the line (pen letters only touch a rule here and there)
            val clean = BooleanArray(bw) { c -> bot[c] >= 0 && !at(hwOn, x0 + c, (top[c] + bot[c]) / 2) }
            val nClean = clean.count { it }
            if (nClean < 0.3 * bw) continue
            val ths = DoubleArray(nClean); var j = 0
            for (c in 0 until bw) if (clean[c]) ths[j++] = (bot[c] - top[c] + 1).toDouble()
            ths.sort(); val th = ths[nClean / 2]
            if (th > 0.6 * k + 2) continue                                        // thin
            var sx = 0.0; var sy = 0.0; var sxx = 0.0; var sxy = 0.0
            for (c in 0 until bw) if (clean[c]) {
                val x = (x0 + c).toDouble(); val cy = (top[c] + bot[c]) / 2.0
                sx += x; sy += cy; sxx += x * x; sxy += x * cy
            }
            val den = nClean * sxx - sx * sx
            val a = if (den != 0.0) (nClean * sxy - sx * sy) / den else 0.0
            val b = (sy - a * sx) / nClean
            val res = DoubleArray(nClean); j = 0
            for (c in 0 until bw) if (clean[c]) res[j++] = abs((top[c] + bot[c]) / 2.0 - (a * (x0 + c) + b))
            res.sort()
            if (res[(0.8 * (nClean - 1)).toInt()] > 1.5) continue                 // straight
            val cx = IntArray(nClean); val cyArr = IntArray(nClean); j = 0
            for (c in 0 until bw) if (clean[c]) { cx[j] = x0 + c; cyArr[j] = (a * (x0 + c) + b).roundToInt().coerceIn(0, h - 1); j++ }
            val gap = (th / 2).toInt() + 2
            var solid = 0; var above = 0; var below = 0; var pureHit = 0
            for (i in 0 until nClean) {
                val x = cx[i]; val y = cyArr[i]
                if (at(ink, x, y) || at(ink, x, y - 1) || at(ink, x, y + 1)) solid++
                if (at(ink, x, y - gap - 1) || at(ink, x, y - gap - 3)) above++
                if (at(ink, x, y + gap + 1) || at(ink, x, y + gap + 3)) below++
                if (at(pure, x, y) || at(pure, x, y - 1) || at(pure, x, y + 1)) pureHit++
            }
            if (solid < 0.9 * nClean) continue                                    // solid, not dotted
            if (above > 0.2 * nClean || below > 0.2 * nClean) continue            // isolated
            if (pureHit < 0.5 * nClean) continue                                  // print, not pen
            rules.add(Rule(transposed, a, b, th, x0, x0 + bw, cx, cyArr))
        }
        return rules
    }

    private fun ruleColour(r: Rule, org: ByteArray, w: Int): FloatArray {
        val m = r.cleanX.size
        return FloatArray(3) { c ->
            val v = IntArray(m) { i ->
                val p = if (r.transposed) r.cleanX[i] * w + r.cleanY[i] else r.cleanY[i] * w + r.cleanX[i]
                org[p * 3 + c].toInt() and 0xff
            }
            v.sort(); v[m / 2].toFloat()
        }
    }

    private fun originalInk(original: Mat): ByteArray {
        val g = Mat(); val bg = Mat(); val d = Mat(); val kernel = Mat.ones(25, 25, CvType.CV_8U)
        Imgproc.cvtColor(original, g, Imgproc.COLOR_BGR2GRAY)
        Imgproc.morphologyEx(g, bg, Imgproc.MORPH_CLOSE, kernel)
        Core.subtract(bg, g, d)
        val out = bytes(d); for (i in out.indices) out[i] = if ((out[i].toInt() and 0xff) > 35) 1 else 0
        listOf(g, bg, d, kernel).forEach { it.release() }
        return out
    }

    private fun paperOf(page: Mat): ByteArray {
        val small = Mat(); val paper = Mat(); val k5 = Mat.ones(5, 5, CvType.CV_8U)
        Imgproc.resize(page, small, Size(), 0.125, 0.125, Imgproc.INTER_AREA)
        Imgproc.dilate(small, small, k5); Imgproc.medianBlur(small, small, 5)
        Imgproc.resize(small, paper, page.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        Imgproc.GaussianBlur(paper, paper, Size(0.0, 0.0), 4.0)
        val out = ByteArray((page.total() * 3).toInt()); paper.get(0, 0, out)
        listOf(small, paper, k5).forEach { it.release() }
        return out
    }

    private fun grey(bgr: ByteArray, i: Int): Int =
        (0.114 * (bgr[i * 3].toInt() and 0xff) + 0.587 * (bgr[i * 3 + 1].toInt() and 0xff) + 0.299 * (bgr[i * 3 + 2].toInt() and 0xff)).toInt()

    private fun bytes(m: Mat): ByteArray = ByteArray(m.total().toInt()).also { m.get(0, 0, it) }

    private fun dilate(m: ByteArray, w: Int, h: Int, size: Int): ByteArray {
        val src = Mat(h, w, CvType.CV_8U); src.put(0, 0, m)
        val kernel = Mat.ones(size, size, CvType.CV_8U)
        Imgproc.dilate(src, src, kernel)
        return bytes(src).also { src.release(); kernel.release() }
    }

    private fun transpose(m: ByteArray, w: Int, h: Int): ByteArray {
        val out = ByteArray(m.size)
        for (y in 0 until h) for (x in 0 until w) out[x * h + y] = m[y * w + x]
        return out
    }
}
