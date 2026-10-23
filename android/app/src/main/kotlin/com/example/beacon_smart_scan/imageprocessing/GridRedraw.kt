package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Grid paper (Vietnamese "ô ly" exercise books) drawn again where the erase took it with the
 * writing. The grid's faint lines are found where the page was never written on, and each one is
 * taken as straight, end to end: from the grid's frame to the frame, not where the writing hid it
 * (lines missing between found ones are added at the grid's pitch). Through the erased area each
 * line is painted with its own look — its colour and width measured beside each painted point on
 * the original — and only on paper: ink the erase kept is never painted over.
 *
 * Only on grid paper: thin vertical lines repeating at one pitch (see [verticalRhythm]). Measured on the 4 grid pages of the labelled
 * set: printed (grid) pixels kept +0.6 to +1.2 points; the handwriting pixels it paints over all lie
 * on the grid lines the pen crossed.
 * Mirrors GridRedraw in ios/Runner/ImageProcessingOpenCV.mm.
 */
object GridRedraw {
    @JvmStatic var enabled = true
    @JvmStatic var report = ""

    /** The lines are found at this long side at most (the grid is the same, found faster). */
    private const val WORK_LONG = 1536
    /** At least this many lines each way: a grid, not a table or a few answer lines. */
    private const val MIN_LINES = 8
    private const val BAND = 64
    /** Prominence of the column profile's periodic peak that makes the page grid paper. */
    private const val MIN_RHYTHM = 0.25
    /** A grid's pitch at the working scale at most (px); wider repeats are table columns. */
    private const val MAX_PITCH = 40

    /** A straight line: across = a·t + b for t in [t0, t1] (t runs along it). */
    private class Line(val a: Double, val b: Double, var t0: Double, var t1: Double)

    /**
     * [original]: the page before the erase (BGR); [dst]: the erased page (BGR), painted in place;
     * [handwriting] / [print]: the mask layers (CV_8UC1 255 = set), [dst]'s size.
     */
    fun apply(original: Mat, dst: Mat, handwriting: Mat, print: Mat) {
        report = ""
        if (!enabled || Core.countNonZero(handwriting) == 0) return
        val fw = dst.cols(); val fh = dst.rows()
        val s = min(1.0, WORK_LONG.toDouble() / max(fw, fh))
        val w = max(1, (fw * s).roundToInt()); val h = max(1, (fh * s).roundToInt())
        val k = max(5, max(w, h) / 450) or 1

        // ---- find the lines at the working scale, where the page was never written on
        val dark: FloatArray; val clean: BooleanArray
        run {
            val o = Mat(); Imgproc.resize(original, o, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            val g = Mat(); Imgproc.cvtColor(o, g, Imgproc.COLOR_BGR2GRAY)
            val p = paper(g)
            val gf = Mat(); g.convertTo(gf, CvType.CV_32F)
            val d = Mat(); Core.subtract(p, gf, d); Core.max(d, org.opencv.core.Scalar(0.0), d)
            dark = FloatArray(w * h); d.get(0, 0, dark)
            val hs = Mat(); Imgproc.resize(handwriting, hs, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
            val zone = Mat(); Imgproc.dilate(hs, zone, Mat.ones(2 * k + 1, 2 * k + 1, CvType.CV_8U))
            val zb = ByteArray(w * h); zone.get(0, 0, zb)
            clean = BooleanArray(w * h) { zb[it].toInt() == 0 }
            listOf(o, g, p, gf, d, hs, zone).forEach { it.release() }
        }
        // grid paper: the faint vertical lines repeat at one pitch across the clean page — a
        // clear periodic peak in the column profile of thin dark lines. Text, tables and answer
        // lines have no such vertical rhythm (measured on 30 pages: grid pages 0.31-0.72, the
        // others at most 0.20)
        val (prom, pitch) = verticalRhythm(dark, clean, w, h)
        report = String.format(java.util.Locale.US, "grid: rhythm %.2f at %d px", prom, pitch)
        if (prom < MIN_RHYTHM || pitch > MAX_PITCH) { report += " (not grid paper)\n"; return }
        var vl = findLines(dark, clean, w, h, true)
        var hl = findLines(dark, clean, w, h, false)
        vl = complete(vl, h); hl = complete(hl, w)
        span(vl, hl); span(hl, vl)
        vl = dedupe(vl, h); hl = dedupe(hl, w)
        report += ", ${vl.size} vertical, ${hl.size} horizontal lines"
        if (vl.size < MIN_LINES || hl.size < MIN_LINES) { report += " (not grid paper)\n"; return }

        // ---- paint at the working scale too (the app's Java heap holds no full-size float
        // planes); the pixels it changes are carried back to the page
        val sz = Size(w.toDouble(), h.toDouble())
        val oS = Mat(); val dS = Mat(); val hS = Mat(); val pS = Mat()
        if (s < 1.0) {
            Imgproc.resize(original, oS, sz, 0.0, 0.0, Imgproc.INTER_AREA); Imgproc.resize(dst, dS, sz, 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.resize(handwriting, hS, sz, 0.0, 0.0, Imgproc.INTER_NEAREST); Imgproc.resize(print, pS, sz, 0.0, 0.0, Imgproc.INTER_NEAREST)
        } else { original.copyTo(oS); dst.copyTo(dS); handwriting.copyTo(hS); print.copyTo(pS) }
        val painted = paint(oS, dS, hS, pS, vl, hl, k)
        report += ", painted $painted px\n"
        if (painted == 0) { listOf(oS, dS, hS, pS).forEach { it.release() }; return }
        if (s < 1.0) {
            // changed pixels only, carried up; elsewhere the page keeps its full resolution
            val before = Mat(); Imgproc.resize(dst, before, sz, 0.0, 0.0, Imgproc.INTER_AREA)
            val diff = Mat(); Core.absdiff(dS, before, diff); Imgproc.cvtColor(diff, diff, Imgproc.COLOR_BGR2GRAY)
            val changed = Mat(); Imgproc.threshold(diff, changed, 0.0, 255.0, Imgproc.THRESH_BINARY)
            val up = Mat(); Imgproc.resize(dS, up, Size(fw.toDouble(), fh.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            val chUp = Mat(); Imgproc.resize(changed, chUp, Size(fw.toDouble(), fh.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
            up.copyTo(dst, chUp)
            listOf(before, diff, changed, up, chUp).forEach { it.release() }
        } else dS.copyTo(dst)
        listOf(oS, dS, hS, pS).forEach { it.release() }
    }

    /** Paints [vl] / [hl] through the erased area of [dst] (all at one scale); returns the pixels painted. */
    private fun paint(original: Mat, dst: Mat, handwriting: Mat, print: Mat, vl: List<Line>, hl: List<Line>, k: Int): Int {
        val fw = dst.cols(); val fh = dst.rows(); val s = 1.0
        val n = fw * fh
        val zoneF = Mat(); Imgproc.dilate(handwriting, zoneF, Mat.ones(2 * k + 1, 2 * k + 1, CvType.CV_8U))
        val zone = ByteArray(n); zoneF.get(0, 0, zone)
        val farM = Mat(); Imgproc.dilate(zoneF, farM, Mat.ones(5, 5, CvType.CV_8U)); zoneF.release()
        val nearZone = ByteArray(n); farM.get(0, 0, nearZone); farM.release()
        // the erased area cleaned to the paper's own colour first (no grey eraser haze under the
        // lines); print and the ink the erase kept stay as they are
        val pap = Array(3) { FloatArray(n) }
        val outF = Array(3) { FloatArray(n) }
        run {
            val ch = ArrayList<Mat>(); Core.split(dst, ch)
            for (c in 0 until 3) {
                val p = paper(ch[c]); p.get(0, 0, pap[c]); p.release()
                val f = Mat(); ch[c].convertTo(f, CvType.CV_32F); f.get(0, 0, outF[c]); f.release()
            }
            ch.forEach { it.release() }
        }
        val prN = Mat(); Imgproc.dilate(print, prN, Mat.ones(3, 3, CvType.CV_8U))
        val prn = ByteArray(n); prN.get(0, 0, prn); prN.release()
        fun keptInk(): BooleanArray {
            val g = FloatArray(n) { 0.114f * outF[0][it] + 0.587f * outF[1][it] + 0.299f * outF[2][it] }
            val gm = Mat(fh, fw, CvType.CV_32F); gm.put(0, 0, g)
            val g8 = Mat(); gm.convertTo(g8, CvType.CV_8U)
            val p = paper(g8); val pa = FloatArray(n); p.get(0, 0, pa)
            listOf(gm, g8, p).forEach { it.release() }
            return BooleanArray(n) { pa[it] - g[it] > 25f }
        }
        var inkd = keptInk()
        for (i in 0 until n) if (zone[i].toInt() != 0 && prn[i].toInt() == 0 && !inkd[i]) for (c in 0 until 3) outF[c][i] = pap[c][i]
        inkd = keptInk()
        // the original's look relative to its paper: a line's cross-section
        val ratio = Array(3) { FloatArray(n) }
        run {
            val ch = ArrayList<Mat>(); Core.split(original, ch)
            for (c in 0 until 3) {
                val p = paper(ch[c]); val pa = FloatArray(n); p.get(0, 0, pa); p.release()
                val f = FloatArray(n); val fm = Mat(); ch[c].convertTo(fm, CvType.CV_32F); fm.get(0, 0, f); fm.release()
                for (i in 0 until n) ratio[c][i] = f[i] / max(pa[i], 1f)
            }
            ch.forEach { it.release() }
        }
        val R = max(2, ceil(2.0 / s).toInt())                 // half the cross-section, page pixels
        val near = (120 / s).toInt()
        var painted = 0
        for ((vert, lines) in listOf(Pair(true, vl), Pair(false, hl))) for (ln in lines) {
            val along = if (vert) fh else fw; val across = if (vert) fw else fh
            val a = ln.a; val b = ln.b / s
            val ta = max(0, (ln.t0 / s).toInt()); val tb = min(along, (ln.t1 / s).toInt())
            if (tb - ta < 10) continue
            fun idx(t: Int, c: Int) = if (vert) t * fw + c else c * fw + t
            val cleanT = ArrayList<Int>(); val zoneT = ArrayList<Int>()
            for (t in ta until tb) {
                val c = (a * t + b).roundToInt()
                if (c < R + 1 || c >= across - R - 1) continue
                val i = idx(t, c)
                if (zone[i].toInt() != 0) zoneT.add(t) else if (nearZone[i].toInt() == 0) cleanT.add(t)
            }
            if (cleanT.size < 20 || zoneT.isEmpty()) continue
            // per clean sample, the cross-section (offsets -R..R, three channels)
            val W = 2 * R + 1
            val samp = Array(cleanT.size) { j ->
                val t = cleanT[j]; val c0 = (a * t + b).roundToInt()
                FloatArray(W * 3) { q -> ratio[q % 3][idx(t, c0 + q / 3 - R)] }
            }
            fun profileNear(t: Int): FloatArray? {
                var lo = lowerBound(cleanT, t - near); val hi = lowerBound(cleanT, t + near + 1)
                if (hi - lo < 8) { lo = 0 }
                val sel = if (hi - lo >= 8) (lo until hi) else (0 until cleanT.size)
                return FloatArray(W * 3) { q -> median(sel.map { samp[it][q] }) }
            }
            val whole = profileNear(-1_000_000) ?: continue
            if ((0 until W * 3).all { whole[it] > 0.985f }) continue      // no visible line here
            var prof = whole; var lastT = -1_000_000
            for (t in zoneT) {
                if (t - lastT >= max(4, (8 / s).toInt())) { prof = profileNear(t) ?: whole; lastT = t }
                // the line's own width: offsets clearly darker than the paper, around the centre
                val dk = BooleanArray(W) { o -> min(prof[o * 3], min(prof[o * 3 + 1], prof[o * 3 + 2])) < 0.985f }
                val centre = (0 until W).minByOrNull { abs(it - R) + if (dk[it]) 0 else 1000 } ?: continue
                if (!dk[centre] || abs(centre - R) > max(1, R / 2)) continue
                var lo = centre; var hi = centre
                val maxW = max(3, ceil(3.0 / s).toInt())
                while (lo - 1 >= 0 && dk[lo - 1] && hi - lo + 1 < maxW) lo--
                while (hi + 1 < W && dk[hi + 1] && hi - lo + 1 < maxW) hi++
                val c0 = (a * t + b).roundToInt()
                for (o in lo..hi) {
                    val c = c0 + o - R
                    if (c < 0 || c >= across) continue
                    val i = idx(t, c)
                    if (zone[i].toInt() == 0 || inkd[i]) continue
                    for (ch in 0 until 3) outF[ch][i] = min(outF[ch][i], pap[ch][i] * prof[o * 3 + ch])
                    painted++
                }
            }
        }
        val chs = (0 until 3).map { c ->
            val m = Mat(fh, fw, CvType.CV_32F); m.put(0, 0, outF[c])
            val u = Mat(); m.convertTo(u, CvType.CV_8U); m.release(); u
        }
        Core.merge(chs, dst); chs.forEach { it.release() }
        return painted
    }

    private fun lowerBound(v: List<Int>, x: Int): Int {
        var lo = 0; var hi = v.size
        while (lo < hi) { val m = (lo + hi) / 2; if (v[m] < x) lo = m + 1 else hi = m }
        return lo
    }

    private fun median(v: List<Float>): Float { if (v.isEmpty()) return 1f; val s = v.sorted(); return s[s.size / 2] }

    /**
     * The best periodic peak of the share of clean pixels per column that are darker than 3 px to
     * either side (a thin vertical line): its prominence over the lowest autocorrelation before
     * it, and its lag (the pitch). Returns (0, 0) when there is none.
     */
    private fun verticalRhythm(dark: FloatArray, clean: BooleanArray, w: Int, h: Int): Pair<Double, Int> {
        val prof = DoubleArray(w)
        for (c in 3 until w - 3) {
            var cov = 0; var hit = 0
            for (y in 0 until h) {
                val i = y * w + c
                if (!clean[i]) continue
                cov++
                if (dark[i] - max(dark[i - 3], dark[i + 3]) > 2.5f) hit++
            }
            prof[c] = if (cov > 0) hit.toDouble() / cov else 0.0
        }
        val mean = prof.average()
        for (c in prof.indices) prof[c] -= mean
        val maxLag = 61
        val ac = DoubleArray(maxLag + 2)
        for (l in 0..maxLag + 1) { var sum = 0.0; for (c in 0 until w - l) sum += prof[c] * prof[c + l]; ac[l] = sum }
        if (ac[0] <= 1e-12) return Pair(0.0, 0)
        for (l in ac.indices) ac[l] /= ac[0]
        var best = 0.0; var lag = 0
        for (l in 6 until maxLag) {
            if (ac[l] < ac[l - 1] || ac[l] < ac[l + 1]) continue
            var lo = Double.MAX_VALUE; for (q in 1 until l) lo = min(lo, ac[q])
            if (ac[l] - lo > best) { best = ac[l] - lo; lag = l }
        }
        return Pair(best, lag)
    }

    /** The paper's brightness: a wide median over the page with its strokes closed away. */
    private fun paper(g: Mat): Mat {
        val d = Mat(); Imgproc.dilate(g, d, Mat.ones(7, 7, CvType.CV_8U))
        Imgproc.medianBlur(d, d, 31)
        val f = Mat(); d.convertTo(f, CvType.CV_32F); d.release()
        return f
    }

    /**
     * Lines running along one direction ([vertical]: columns): per band across them, the share of
     * clean pixels where the page is darker than 3 px to either side, across the line; peaks of
     * that share are line points, linked into straight lines.
     */
    private fun findLines(dark: FloatArray, clean: BooleanArray, w: Int, h: Int, vertical: Boolean): MutableList<Line> {
        val L = if (vertical) h else w      // along
        val A = if (vertical) w else h      // across
        fun at(t: Int, c: Int) = if (vertical) t * w + c else c * w + t
        val pts = ArrayList<DoubleArray>()  // (t, across)
        var t0 = 0
        while (t0 + BAND <= L) {
            val on = FloatArray(A)
            for (c in 3 until A - 3) {
                var cov = 0; var hit = 0
                for (t in t0 until t0 + BAND) {
                    val i = at(t, c)
                    if (!clean[i]) continue
                    cov++
                    val side = max(dark[at(t, c - 3)], dark[at(t, c + 3)])
                    if (dark[i] - side > 2.5f) hit++
                }
                on[c] = if (cov < BAND * 0.25) 0f else hit.toFloat() / cov
            }
            for (c in 3 until A - 3) if (on[c] >= 0.45f && on[c] >= on[c - 1] && on[c] > on[c + 1])
                pts.add(doubleArrayOf(t0 + BAND / 2.0, c.toDouble()))
            t0 += BAND / 2
        }
        // link points into straight lines, allowing a drift of 2.5 px
        pts.sortWith(compareBy({ it[1] }, { it[0] }))
        val groups = ArrayList<Pair<ArrayList<Double>, ArrayList<Double>>>()
        for (p in pts) {
            var best: Pair<ArrayList<Double>, ArrayList<Double>>? = null
            for (g in groups) {
                val ts = g.first; val cs = g.second
                if (abs(ts.last() - p[0]) < 1e-6) continue
                val pc = if (ts.size >= 2) { val (fa, fb) = fit(ts, cs); fa * p[0] + fb } else cs.last()
                if (abs(pc - p[1]) <= 2.5 && abs(ts.last() - p[0]) <= BAND * 8) { best = g; break }
            }
            if (best == null) groups.add(Pair(arrayListOf(p[0]), arrayListOf(p[1])))
            else { best.first.add(p[0]); best.second.add(p[1]) }
        }
        val out = ArrayList<Line>()
        for ((ts, cs) in groups) {
            if (ts.size < 4) continue
            val (fa, fb) = fit(ts, cs)
            if (ts.indices.any { abs(cs[it] - (fa * ts[it] + fb)) > 1.5 }) continue
            out.add(Line(fa, fb, ts.minOrNull()!! - BAND / 2.0, ts.maxOrNull()!! + BAND / 2.0))
        }
        return out
    }

    private fun fit(ts: List<Double>, cs: List<Double>): Pair<Double, Double> {
        val n = ts.size; var sx = 0.0; var sy = 0.0; var sxx = 0.0; var sxy = 0.0
        for (i in 0 until n) { sx += ts[i]; sy += cs[i]; sxx += ts[i] * ts[i]; sxy += ts[i] * cs[i] }
        val den = n * sxx - sx * sx
        val a = if (abs(den) > 1e-9) (n * sxy - sx * sy) / den else 0.0
        return Pair(a, (sy - a * sx) / n)
    }

    /** Grid lines repeat at one pitch: lines missing between found ones (hidden by the writing) are added. */
    private fun complete(lines0: MutableList<Line>, along: Int): MutableList<Line> {
        if (lines0.size < 6) return lines0
        val mid = along / 2.0
        val lines = ArrayList<Line>()
        for (l in lines0.sortedBy { it.a * mid + it.b }) {
            val last = lines.lastOrNull()
            if (last != null && abs((l.a * mid + l.b) - (last.a * mid + last.b)) < 3) {
                if (l.t1 - l.t0 > last.t1 - last.t0) lines[lines.size - 1] = l
            } else lines.add(l)
        }
        val off = lines.map { it.a * mid + it.b }
        val gaps = (0 until off.size - 1).map { off[it + 1] - off[it] }
        val out = ArrayList(lines)
        for (g in gaps.indices) {
            val win = gaps.subList(max(0, g - 6), min(gaps.size, g + 7)).filter { it >= 6 }
            if (win.isEmpty()) continue
            val mn = win.minOrNull()!!
            val singles = win.filter { it <= 1.4 * mn }
            if (singles.size < 3) continue
            val p = singles.sorted()[singles.size / 2]
            val q = (gaps[g] / p).roundToInt()
            if (q < 2 || q > 6 || abs(gaps[g] / q - p) > 0.15 * p) continue
            val l0 = lines[g]; val l1 = lines[g + 1]
            for (r in 1 until q) {
                val f = r.toDouble() / q
                out.add(Line(l0.a + (l1.a - l0.a) * f, l0.b + (l1.b - l0.b) * f, min(l0.t0, l1.t0), max(l0.t1, l1.t1)))
            }
        }
        return out
    }

    /** A straight line runs whole, between the outermost lines across it that reach it (the frame). */
    private fun span(lines: List<Line>, cross: List<Line>) {
        for (l in lines) {
            val pos = ArrayList<Double>()
            for (c in cross) {
                val den = 1 - l.a * c.a
                if (abs(den) < 1e-6) continue
                val t = (c.a * l.b + c.b) / den; val acr = l.a * t + l.b
                if (acr >= c.t0 - 6 && acr <= c.t1 + 6 && t >= l.t0 - 400 && t <= l.t1 + 400) pos.add(t)
            }
            if (pos.size >= 2) { l.t0 = min(pos.minOrNull()!!, l.t0); l.t1 = max(pos.maxOrNull()!!, l.t1) }
        }
    }

    /** One printed line found twice (its two edges): within 4 px along its run, the longer one. */
    private fun dedupe(lines: List<Line>, along: Int): MutableList<Line> {
        val mid = along / 2.0
        val out = ArrayList<Line>()
        for (l in lines.sortedBy { it.a * mid + it.b }) {
            val last = out.lastOrNull()
            if (last != null && abs((l.a * mid + l.b) - (last.a * mid + last.b)) < 4) {
                if (l.t1 - l.t0 > last.t1 - last.t0) out[out.size - 1] = l
            } else out.add(l)
        }
        return out
    }
}
