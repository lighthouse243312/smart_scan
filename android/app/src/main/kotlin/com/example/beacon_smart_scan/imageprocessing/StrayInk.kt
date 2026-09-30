package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Last review layer — stray ink around what was erased (Kotlin port of RemoveStrayInk in
 * ios/Runner/InkAnalysis.hpp). Every piece of ink left where the pen was must "mean something" on
 * this page: a typeset page repeats its glyphs (letters, digits, dots, accents), so a piece with no
 * twin among the page's untouched glyphs is a pen leftover. Rules and table lines — and rule pieces
 * the erase re-joined, on the rule's own rows — are taken out first and never touched, so a rule
 * keeps its exact thickness and length. Small twins (dots, accents) must sit where a mark belongs:
 * centred over or under a letter, close to it.
 */
object StrayInk {
    private fun ellipse(d: Int): Mat = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size((d or 1).toDouble(), (d or 1).toDouble()))

    fun remove(dst: Mat, erasedAll: Mat, handwriting: Mat, protectedInk: Mat, paperBright: Mat, k: Int) {
        if (Core.countNonZero(erasedAll) == 0) return
        val t = ArrayList<Mat>()
        fun <T : Mat> own(m: T): T { t.add(m); return m }
        try {
            val W = dst.cols(); val H = dst.rows()
            val g = own(Mat()); Imgproc.cvtColor(dst, g, Imgproc.COLOR_BGR2GRAY)
            val pg = own(Mat()); Imgproc.cvtColor(paperBright, pg, Imgproc.COLOR_BGR2GRAY)
            val diff = own(Mat()); Core.subtract(pg, g, diff, Mat(), CvType.CV_16S)
            val ink = own(Mat()); Core.compare(diff, Scalar(50.0), ink, Core.CMP_GT)
            val L = max(15, 8 * k) or 1
            val hR = own(Mat()); val vR = own(Mat())
            Imgproc.morphologyEx(ink, hR, Imgproc.MORPH_OPEN, Mat.ones(1, L, CvType.CV_8U))
            Imgproc.morphologyEx(ink, vR, Imgproc.MORPH_OPEN, Mat.ones(L, 1, CvType.CV_8U))
            val hExt = own(Mat()); val vExt = own(Mat())
            Imgproc.dilate(hR, hExt, Mat.ones(1, 6 * L, CvType.CV_8U))
            Imgproc.dilate(vR, vExt, Mat.ones(6 * L, 1, CvType.CV_8U))
            val rules = own(Mat()); Core.bitwise_or(hExt, vExt, rules); Core.bitwise_and(rules, ink, rules)
            Core.bitwise_or(rules, hR, rules); Core.bitwise_or(rules, vR, rules)
            val notRules = own(Mat()); Core.bitwise_not(rules, notRules)
            val glyphInk = own(Mat()); Core.bitwise_and(ink, notRules, glyphInk)

            val labels = own(Mat()); val stats = own(Mat()); val cent = own(Mat())
            val n = Imgproc.connectedComponentsWithStats(glyphInk, labels, stats, cent, 8, CvType.CV_32S)
            val lab = IntArray(W * H); labels.get(0, 0, lab)
            val st = IntArray(n * 5); stats.get(0, 0, st)
            fun x(i: Int) = st[i * 5 + Imgproc.CC_STAT_LEFT]
            fun y(i: Int) = st[i * 5 + Imgproc.CC_STAT_TOP]
            fun w(i: Int) = st[i * 5 + Imgproc.CC_STAT_WIDTH]
            fun h(i: Int) = st[i * 5 + Imgproc.CC_STAT_HEIGHT]
            fun area(i: Int) = st[i * 5 + Imgproc.CC_STAT_AREA]

            val zoneSrc = own(Mat()); Core.bitwise_or(erasedAll, protectedInk, zoneSrc)
            val zone = own(Mat()); Imgproc.dilate(zoneSrc, zone, ellipse(2 * k + 1))
            val hwRim = own(Mat()); Imgproc.dilate(handwriting, hwRim, Mat.ones(5, 5, CvType.CV_8U))
            val zb = ByteArray(W * H); zone.get(0, 0, zb)
            val pb = ByteArray(W * H); protectedInk.get(0, 0, pb)
            val hb = ByteArray(W * H); hwRim.get(0, 0, hb)
            val nz = IntArray(n); val np = IntArray(n); val nh = IntArray(n)
            for (p in lab.indices) { val a = lab[p]; if (a == 0) continue
                if (zb[p].toInt() != 0) nz[a]++; if (pb[p].toInt() != 0) np[a]++; if (hb[p].toInt() != 0) nh[a]++ }

            val ref = ArrayList<Int>(); val hs = ArrayList<Float>()
            for (i in 1 until n) if (nz[i] == 0 && area(i) >= 3) { ref.add(i); if (area(i) > 4 * k * k) hs.add(h(i).toFloat()) }
            if (ref.size < 50 || hs.isEmpty()) return
            hs.sort(); val cap = hs[hs.size / 2]
            ref.sortBy { h(it) }
            val refH = ref.map { h(it) }
            fun iou(i: Int, j: Int): Float {
                var inter = 0; var uni = 0
                for (yy in 0 until h(i)) for (xx in 0 until w(i)) {
                    val a = lab[(y(i) + yy) * W + x(i) + xx] == i
                    val b = lab[(y(j) + yy * h(j) / h(i)) * W + x(j) + xx * w(j) / w(i)] == j
                    if (a && b) inter++; if (a || b) uni++
                }
                return if (uni == 0) 0f else inter.toFloat() / uni
            }
            fun hasTwin(i: Int): Boolean {
                val tiny = h(i) <= 4 || w(i) <= 4
                val need = if (tiny) 0.5f else 0.6f
                val hLo = if (tiny) h(i) - 1 else floor(h(i) * 0.85).toInt()
                val hHi = if (tiny) h(i) + 1 else ceil(h(i) * 1.18).toInt()
                var idx = refH.binarySearch(hLo).let { if (it < 0) -it - 1 else it }
                while (idx > 0 && refH[idx - 1] >= hLo) idx--
                while (idx < ref.size && refH[idx] <= hHi) {
                    val j = ref[idx]; idx++
                    if (if (tiny) abs(w(j) - w(i)) > 1 else (w(j) < 0.82f * w(i) || w(j) > 1.22f * w(i))) continue
                    if (iou(i, j) >= need) return true
                }
                return false
            }
            // a small piece the print layer kept, but among erased strokes only: no untouched print and
            // no letter-sized ink left (a letter the pen crossed still counts) within a letter's reach
            // on its line, erased pen on two sides of it and touching the pen's rim — the bar of a pen
            // "=" written across a table rule, the tip of a stroke. Printed marks always have their
            // text beside them.
            val isRef = BooleanArray(n); for (i in ref) isRef[i] = true
            for (i in 1 until n) if (area(i) > k * k && h(i) >= 0.5f * cap) isRef[i] = true
            val eb = ByteArray(W * H); erasedAll.get(0, 0, eb)
            fun integral(set: (Int) -> Boolean): IntArray {
                val s = IntArray((W + 1) * (H + 1))
                for (yy in 0 until H) {
                    var row = 0
                    for (xx in 0 until W) {
                        if (set(yy * W + xx)) row++
                        s[(yy + 1) * (W + 1) + xx + 1] = s[yy * (W + 1) + xx + 1] + row
                    }
                }
                return s
            }
            val refInt = integral { isRef[lab[it]] }
            val erasedInt = integral { eb[it].toInt() != 0 }
            fun countIn(s: IntArray, rx: Int, ry: Int, rw: Int, rh: Int): Int {
                val x0 = max(0, rx); val y0 = max(0, ry); val x1 = min(W, rx + rw); val y1 = min(H, ry + rh)
                if (x1 <= x0 || y1 <= y0) return 0
                return s[y1 * (W + 1) + x1] - s[y0 * (W + 1) + x1] - s[y1 * (W + 1) + x0] + s[y0 * (W + 1) + x0]
            }
            fun amidPen(i: Int): Boolean {
                if (nh[i] == 0 || h(i) >= 0.5f * cap || area(i) > 4 * k * k) return false
                val gx = ceil(cap).toInt(); val gy = ceil(0.5f * cap).toInt()
                if (countIn(refInt, x(i) - gx, y(i) - gy, w(i) + 2 * gx, h(i) + 2 * gy) > 0) return false
                var sides = 0
                if (countIn(erasedInt, x(i) - gx, y(i), gx, h(i)) > 0) sides++
                if (countIn(erasedInt, x(i) + w(i), y(i), gx, h(i)) > 0) sides++
                if (countIn(erasedInt, x(i), y(i) - gy, w(i), gy) > 0) sides++
                if (countIn(erasedInt, x(i), y(i) + h(i), w(i), gy) > 0) sides++
                return sides >= 2
            }
            val stray = BooleanArray(n)
            for (i in 1 until n) {
                val a = area(i)
                // (restored print is kept — unless it is a meaningless crumb: a speck, or under half a
                // text height among erased strokes only — a pen "76" the overlap test took for print
                // under the pen; what is left of a printed letter has its word beside it)
                val crumb = a <= k * k && ((h(i) < 0.4f * cap && w(i) < 0.4f * cap) ||
                                           (h(i) < 0.5f * cap && amidPen(i)))
                if (nz[i] < 0.3f * a || (np[i] > 0.5f * a && !crumb) || (nh[i] < 0.6f * a && !amidPen(i))) continue
                if (h(i) > 2.5f * cap || w(i) > 4 * cap) continue
                if (!hasTwin(i)) { stray[i] = true; continue }
                if (h(i) < 0.5f * cap && a <= 4 * k * k) {
                    // a mark: centred on a letter right above or below it
                    val gap = ceil(0.5f * cap).toInt()
                    val y0 = max(0, y(i) - gap); val y1 = min(H, y(i) + h(i) + gap)
                    val cx = x(i) + w(i) / 2f
                    var centred = false
                    val seen = HashSet<Int>()
                    loop@ for (yy in y0 until y1) for (xx in x(i) until x(i) + w(i)) {
                        val j = lab[yy * W + xx]
                        if (j == 0 || j == i || h(j) < 0.5f * cap || area(j) <= k * k || !seen.add(j)) continue
                        val below = y(i) >= y(j) + h(j)
                        val gapV = if (below) y(i) - (y(j) + h(j)) else y(j) - (y(i) + h(i))
                        val close = gapV <= (if (below) 0.35f else 0.45f) * cap
                        if (close && cx >= x(j) + 0.2f * w(j) && cx <= x(j) + 0.8f * w(j)) { centred = true; break@loop }
                    }
                    if (!centred) stray[i] = true
                }
            }
            val out = ByteArray(W * H) { if (stray[lab[it]]) 255.toByte() else 0 }
            val strayMask = own(Mat(H, W, CvType.CV_8U)); strayMask.put(0, 0, out)
            val grown = own(Mat()); Imgproc.dilate(strayMask, grown, Mat.ones(3, 3, CvType.CV_8U))
            val notProt = own(Mat()); Core.bitwise_not(protectedInk, notProt); Core.bitwise_or(notProt, strayMask, notProt)
            Core.bitwise_and(grown, notRules, grown); Core.bitwise_and(grown, notProt, grown)
            val faint = own(Mat()); Core.compare(diff, Scalar(6.0), faint, Core.CMP_GT)
            Core.bitwise_and(grown, faint, grown); Core.bitwise_or(grown, strayMask, grown)
            paperBright.copyTo(dst, grown)
        } finally { t.forEach { it.release() } }
    }
}
