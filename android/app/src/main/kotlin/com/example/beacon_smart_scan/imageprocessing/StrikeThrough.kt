package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Pen strike-throughs across printed words ("Fruit / are / your" crossed out). A black pen line has
 * the print's colour, so the colour layers call it print and it was never erased. It is told from
 * print by shape — measured on 30 labelled pages, every feature below separates the two:
 *
 *  - it is one straight thin line of ink, unbroken for 1.5 text heights: an opening with a 1-px line
 *    kernel at small tilts, on the ink itself (no growth — growing the ink first merged the tops of
 *    bold letters into "lines"). Printed text at mid height always has paper between letters.
 *  - printed letters cross it (ink runs on through it, up and down) at 2+ places, and the letters
 *    are text sized (a drawing or a table border is not).
 *  - its thickness is a pen stroke's (0.5–1.25 stroke units): a bold letter's edge or a picture's
 *    line is thicker, a grid line or wood grain thinner.
 *  - it crosses the gaps between letters: in a share of its columns nothing touches it right above
 *    or below (a run along the tops of touching bold letters has a letter under every column).
 *  - it is hand drawn: it wobbles off a straight fit, or overshoots the word onto the paper.
 *
 * The line becomes handwriting, but it is erased only where it stands free (paper right above and
 * below: between letters, inside a counter, the overshoot). Where a letter crosses it, or its pixels
 * lie right against a letter's arch or bar, or taking them would open a letter's closed counter (the
 * eye of an "e"), they are print under the pen (overlap), drawn back. A line along the letters' tops
 * or feet loses only its word gaps and overshoot. Its own pixels go to the mask file (the strike
 * layer), so the erase does not re-join them as a printed rule nor re-spell the words it crossed.
 * Mirrors StrikeThrough in ios/Runner/ImageProcessingOpenCV.mm.
 */
object StrikeThrough {
    @JvmStatic var enabled = true
    /** Dev (InkHarness `-e strikeprobe 1`): one line per candidate with its features. */
    @JvmStatic var probe = false
    @JvmStatic var report = ""

    /** Found at this long side at most (the line opening costs pixels × kernel length). */
    private const val WORK_LONG = 1600
    private val ANGLES = doubleArrayOf(-8.0, -5.0, -2.5, 0.0, 2.5, 5.0, 8.0)
    /** An unbroken line at least this many text heights long (and 8 stroke units). */
    private const val MIN_LEN = 1.5
    /** No longer than this: a longer straight line is a rule or an answer line. */
    private const val MAX_LEN = 12.0
    /** Ink running on through the line this far (text heights) up and down is a letter crossing. */
    private const val CROSS_REACH = 0.15
    /** A column is over paper when no ink is this near (text heights) above and below. */
    private const val CLEAR_REACH = 0.4

    /**
     * [src]: the page (BGR); [handwriting] / [print] / [overlap]: CV_8UC1 255 = set, updated in place.
     * Returns the strike lines' own pixels (not where letters cross them), for the mask file — or null.
     */
    fun apply(src: Mat, handwriting: Mat, print: Mat, overlap: Mat): Mat? {
        report = ""
        if (!enabled) return null
        val fw = src.cols(); val fh = src.rows()
        val s = min(1.0, WORK_LONG.toDouble() / max(fw, fh))
        val w = max(1, (fw * s).roundToInt()); val h = max(1, (fh * s).roundToInt())
        // dark ink (well clear of the paper) at the working scale
        val inkM = Mat()
        run {
            val small = Mat(); val g = Mat(); val bg = Mat(); val kernel = Mat.ones(25, 25, CvType.CV_8U)
            if (s < 1.0) Imgproc.resize(src, small, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA) else src.copyTo(small)
            Imgproc.cvtColor(small, g, Imgproc.COLOR_BGR2GRAY)
            Imgproc.morphologyEx(g, bg, Imgproc.MORPH_CLOSE, kernel)
            Core.subtract(bg, g, g)
            Imgproc.threshold(g, inkM, 40.0, 1.0, Imgproc.THRESH_BINARY)
            listOf(small, g, bg, kernel).forEach { it.release() }
        }
        val ink = ByteArray(w * h); inkM.get(0, 0, ink)
        fun scaled(m: Mat): ByteArray {
            val t = Mat()
            if (s < 1.0) Imgproc.resize(m, t, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST) else m.copyTo(t)
            val a = ByteArray(w * h); t.get(0, 0, a); t.release(); return a
        }
        val hwS = scaled(handwriting); val prS = scaled(print)
        // pure print (print and not handwriting): what the line must cross
        val prm = BooleanArray(w * h) { prS[it].toInt() != 0 && hwS[it].toInt() == 0 }

        // stroke unit: twice the median ridge of the ink's distance transform
        val k: Int
        run {
            val dt = Mat(); Imgproc.distanceTransform(inkM, dt, org.opencv.geometry.Geometry.DIST_L2, 3)
            val dil = Mat(); Imgproc.dilate(dt, dil, Mat.ones(3, 3, CvType.CV_8U))
            val d = FloatArray(w * h); dt.get(0, 0, d); val dm = FloatArray(w * h); dil.get(0, 0, dm)
            dt.release(); dil.release()
            val ridge = ArrayList<Float>()
            for (i in d.indices) if (d[i] > 0f && d[i] >= dm[i]) ridge.add(d[i])
            if (ridge.isEmpty()) { inkM.release(); return null }
            ridge.sort(); k = max(2, (2 * ridge[ridge.size / 2]).roundToInt())
        }
        // text height: median height of letter-shaped print pieces (not dots, accents, rules)
        val textH: Double
        run {
            val pm = Mat(h, w, CvType.CV_8U); val pb = ByteArray(w * h) { if (prm[it]) 1 else 0 }; pm.put(0, 0, pb)
            val lb = Mat(); val st = Mat(); val ce = Mat()
            val n = Imgproc.connectedComponentsWithStats(pm, lb, st, ce, 8, CvType.CV_32S)
            val sa = IntArray(n * 5); st.get(0, 0, sa); listOf(pm, lb, st, ce).forEach { it.release() }
            val hs = ArrayList<Int>()
            for (l in 1 until n) {
                val cw = sa[l * 5 + 2]; val ch = sa[l * 5 + 3]; val a = sa[l * 5 + 4]
                if (a >= 6 * k * k && ch < h / 20 && cw >= 0.4 * ch) hs.add(ch)
            }
            hs.sort(); textH = if (hs.isEmpty()) 8.0 * k else hs[hs.size / 2].toDouble()
        }
        // straight unbroken lines: opening with a thin line kernel at small tilts
        val len = max(MIN_LEN * textH, 8.0 * k).roundToInt() or 1
        val runM = Mat.zeros(h, w, CvType.CV_8U)
        run {
            val t = Mat()
            for (deg in ANGLES) {
                val kern = lineKernel(len, deg)
                Imgproc.morphologyEx(inkM, t, Imgproc.MORPH_OPEN, kern)
                Core.bitwise_or(runM, t, runM); kern.release()
            }
            t.release()
        }
        val labM = Mat(); val stM = Mat(); val ceM = Mat()
        val n = Imgproc.connectedComponentsWithStats(runM, labM, stM, ceM, 8, CvType.CV_32S)
        runM.release(); ceM.release()
        val st = IntArray(n * 5); stM.get(0, 0, st); stM.release()
        val lab = IntArray(w * h); labM.get(0, 0, lab); labM.release()
        // ink blobs: a struck word stays about a text line tall, a drawing does not
        val inkLab = IntArray(w * h); val inkSt: IntArray
        run {
            val lb = Mat(); val sb = Mat(); val cb = Mat()
            val ni = Imgproc.connectedComponentsWithStats(inkM, lb, sb, cb, 8, CvType.CV_32S)
            lb.get(0, 0, inkLab); inkSt = IntArray(ni * 5); sb.get(0, 0, inkSt)
            listOf(lb, sb, cb).forEach { it.release() }
        }
        inkM.release()
        fun inkAt(x: Int, y: Int) = y in 0 until h && ink[y * w + x].toInt() != 0
        val reach = max(2, (CROSS_REACH * textH).toInt()); val clearR = max(4, (CLEAR_REACH * textH).toInt())
        val out = BooleanArray(w * h); val keep = BooleanArray(w * h)
        val rep = StringBuilder()
        var found = 0
        for (l in 1 until n) {
            val x0 = st[l * 5]; val y0 = st[l * 5 + 1]; val bw = st[l * 5 + 2]; val bh = st[l * 5 + 3]
            if (bw > MAX_LEN * textH) continue
            val top = IntArray(bw) { -1 }; val bot = IntArray(bw) { -1 }
            for (y in y0 until y0 + bh) for (c in 0 until bw) if (lab[y * w + x0 + c] == l) {
                if (top[c] < 0) top[c] = y; bot[c] = y
            }
            val cols = (0 until bw).filter { top[it] >= 0 }
            if (cols.size < 2) continue
            val cross = BooleanArray(cols.size); val clear = BooleanArray(cols.size)
            val vext = ArrayList<Int>(); var cpN = 0; var touchN = 0; val touchC = BooleanArray(cols.size)
            for ((j, c) in cols.withIndex()) {
                val x = x0 + c; val t = top[c]; val b = bot[c]
                var u = 0; while (inkAt(x, t - u - 1)) u++
                var d = 0; while (inkAt(x, b + d + 1)) d++
                var any = false
                for (y in max(0, t - clearR) until max(0, t - 1)) if (ink[y * w + x].toInt() != 0) { any = true; break }
                if (!any) for (y in b + 2 until min(h, b + 1 + clearR)) if (ink[y * w + x].toInt() != 0) { any = true; break }
                clear[j] = !any
                if (inkAt(x, t - 1) || inkAt(x, b + 1)) { touchN++; touchC[j] = true }
                if (u >= reach && d >= reach) {
                    cross[j] = true; vext.add(u + d + b - t + 1)
                    val up = prm[(t - 1) * w + x] || prm[(t - u) * w + x]
                    val dn = prm[(b + 1) * w + x] || prm[(b + d) * w + x]
                    if (up && dn) cpN++
                }
            }
            var groups = 0
            for (j in cross.indices) if (cross[j] && (j == 0 || !cross[j - 1])) groups++
            if (groups < 2) continue
            // features (units: text height / stroke unit)
            val ext = cols.map { bot[it] - top[it] + 1 }.sorted()
            val thick = ext[ext.size / 2].toDouble() / k
            vext.sort(); val vmed = vext[vext.size / 2] / textH
            val cprint = cpN.toDouble() / vext.size
            val touch = touchN.toDouble() / cols.size
            val bridge = clear.count { it }.toDouble() / cols.size
            val ovL = clear.indexOfFirst { !it }.let { if (it < 0) cols.size else it }
            val ovR = clear.reversed().indexOfFirst { !it }.let { if (it < 0) cols.size else it }
            val ovmax = max(ovL, ovR) / textH
            val dev = fitDeviation(cols, top, bot)
            val lenH = bw / textH
            var blob = 0
            run {
                val cnt = HashMap<Int, Int>()
                for (c in cols) for (y in top[c]..bot[c]) { val il = inkLab[y * w + x0 + c]; if (il > 0) cnt[il] = (cnt[il] ?: 0) + 1 }
                blob = cnt.maxByOrNull { it.value }?.key ?: 0
            }
            val blobH = if (blob > 0) inkSt[blob * 5 + 3] / textH else 0.0
            val ok = cprint >= 0.5 && blobH <= 2.0 && thick in 0.5..1.25 && vmed <= 1.0 && touch <= 0.9 &&
                (dev >= 0.3 || ovmax >= 0.2) && (lenH >= 2.0 || bridge >= 0.15)
            if (probe) rep.append(String.format(java.util.Locale.US,
                "  strike %s (%d,%d) len %.2f thick %.2f dev %.2f touch %.2f vmed %.2f groups %d bridge %.2f cprint %.2f ov %.2f blob %.2f\n",
                if (ok) "A" else "-", (x0 / s).roundToInt(), (y0 / s).roundToInt(), lenH, thick, dev, touch, vmed, groups, bridge, cprint, ovmax, blobH))
            if (!ok) continue
            found++
            // through the letters' middle, or along their tops / feet? Over the word, letters lie
            // close on both sides of a line through the middle; one along the tops has letter
            // bodies under it and only ascenders above
            val firstLetter = cols.indices.firstOrNull { cross[it] || touchC[it] } ?: 0
            val lastLetter = cols.indices.lastOrNull { cross[it] || touchC[it] } ?: cols.size - 1
            val near = max(2, (0.35 * textH).toInt())
            var nAbove = 0; var nBelow = 0
            for (j in firstLetter..lastLetter) {
                val x = x0 + cols[j]; val t = top[cols[j]]; val b = bot[cols[j]]
                if ((max(0, t - near) until t).any { ink[it * w + x].toInt() != 0 }) nAbove++
                if ((b + 1..min(h - 1, b + near)).any { ink[it * w + x].toInt() != 0 }) nBelow++
            }
            val span = max(1, lastLetter - firstLetter + 1)
            val edge = min(nAbove, nBelow) < 0.35 * span
            if (probe) rep.append(String.format(java.util.Locale.US, "    above %.2f below %.2f%s\n", nAbove.toDouble() / span, nBelow.toDouble() / span, if (edge) " edge" else ""))
            // along the letters' edge the line lies on their arches and bars: only its free runs
            // as wide as a word space (and the overshoot) go
            val wideFree = BooleanArray(cols.size)
            run {
                var j = 0
                while (j < cols.size) {
                    if (cross[j] || touchC[j]) { j++; continue }
                    var e = j; while (e + 1 < cols.size && !cross[e + 1] && !touchC[e + 1]) e++
                    if (e - j + 1 >= 0.3 * textH) for (q in j..e) wideFree[q] = true
                    j = e + 1
                }
            }
            // erased only where it stands free (paper right above and below: a gap between letters,
            // a letter's counter, the overshoot); where it lies along a letter's own stroke (crossing
            // a stem, merged with an arch or a bar's top) those pixels may be the letter's: kept
            for ((j, c) in cols.withIndex()) for (y in top[c]..bot[c]) {
                val i = y * w + x0 + c
                if (lab[i] != l) continue
                out[i] = true
                val over = j < firstLetter || j > lastLetter
                if (edge) { if (!over && !wideFree[j]) keep[i] = true }
                else if (cross[j]) keep[i] = true
                // merged with a letter's arch or bar: the pixels right against the letter's own ink
                else if (touchC[j] && besideLetter(ink, lab, l, w, h, x0 + c, y)) keep[i] = true
            }
            keepClosedCounters(ink, out, keep, w, h, max(0, x0 - 2), max(0, y0 - clearR), min(w, x0 + bw + 2), min(h, y0 + bh + clearR))
        }
        report = String.format(java.util.Locale.US, "strike: text %.0f px, unit %d, lines %d\n", textH / s, k, found) + rep
        if (found == 0) return null

        // back to the page's scale: the found line (grown a pixel) on the page's own ink
        val outM = Mat(h, w, CvType.CV_8U); outM.put(0, 0, ByteArray(w * h) { if (out[it]) 255.toByte() else 0 })
        val keepM = Mat(h, w, CvType.CV_8U); keepM.put(0, 0, ByteArray(w * h) { if (keep[it]) 255.toByte() else 0 })
        if (s < 1.0) {
            val sz = Size(fw.toDouble(), fh.toDouble())
            Imgproc.resize(outM, outM, sz, 0.0, 0.0, Imgproc.INTER_NEAREST)
            Imgproc.resize(keepM, keepM, sz, 0.0, 0.0, Imgproc.INTER_NEAREST)
            val k3 = Mat.ones(3, 3, CvType.CV_8U)
            Imgproc.dilate(outM, outM, k3); Imgproc.dilate(keepM, keepM, k3); k3.release()
        }
        val o = ByteArray(fw * fh); outM.get(0, 0, o); val kp = ByteArray(fw * fh); keepM.get(0, 0, kp)
        outM.release(); keepM.release()
        val hw = ByteArray(fw * fh); handwriting.get(0, 0, hw)
        val pr = ByteArray(fw * fh); print.get(0, 0, pr)
        val ov = ByteArray(fw * fh); overlap.get(0, 0, ov)
        val on = 255.toByte()
        val line = ByteArray(fw * fh)
        for (i in o.indices) {
            if (o[i].toInt() == 0) continue
            if (hw[i].toInt() == 0 && pr[i].toInt() == 0 && ov[i].toInt() == 0) continue    // paper
            if (kp[i].toInt() == 0) line[i] = on
            if (hw[i].toInt() != 0 && ov[i].toInt() == 0) continue                          // already pen
            hw[i] = on
            if (kp[i].toInt() != 0) { ov[i] = on; pr[i] = on } else { ov[i] = 0; pr[i] = 0 }
        }
        handwriting.put(0, 0, hw); print.put(0, 0, pr); overlap.put(0, 0, ov)
        val strikes = Mat(fh, fw, CvType.CV_8U); strikes.put(0, 0, line)
        return strikes
    }

    /** Ink of something other than line [l] right beside ([x], [y]) (8-neighbours). */
    private fun besideLetter(ink: ByteArray, lab: IntArray, l: Int, w: Int, h: Int, x: Int, y: Int): Boolean {
        for (dy in -1..1) for (dx in -1..1) {
            val px = x + dx; val py = y + dy
            if (px !in 0 until w || py !in 0 until h) continue
            val i = py * w + px
            if (ink[i].toInt() != 0 && lab[i] != l) return true
        }
        return false
    }

    /**
     * A letter's closed counter (the eye of an "e", whose bar the strike ran along) must stay closed:
     * erased line pixels that would open a hole of the ink to the paper outside are kept. Splitting
     * one hole in two by the strike and joining them back (an "o" struck through) opens nothing.
     */
    private fun keepClosedCounters(ink: ByteArray, out: BooleanArray, keep: BooleanArray, w: Int, h: Int, xa: Int, ya: Int, xb: Int, yb: Int) {
        val cw = xb - xa; val ch = yb - ya
        if (cw < 3 || ch < 3) return
        fun paperLabels(erased: Boolean): IntArray {
            val m = Mat(ch, cw, CvType.CV_8U)
            m.put(0, 0, ByteArray(cw * ch) { val x = xa + it % cw; val y = ya + it / cw; val i = y * w + x
                val inkHere = ink[i].toInt() != 0 && !(erased && out[i] && !keep[i]); if (inkHere) 0 else 1 })
            val lb = Mat(); Imgproc.connectedComponents(m, lb, 4, CvType.CV_32S)
            val a = IntArray(cw * ch); lb.get(0, 0, a); m.release(); lb.release(); return a
        }
        fun outside(lab: IntArray): Set<Int> {
            val o = HashSet<Int>()
            for (x in 0 until cw) { o.add(lab[x]); o.add(lab[(ch - 1) * cw + x]) }
            for (y in 0 until ch) { o.add(lab[y * cw]); o.add(lab[y * cw + cw - 1]) }
            o.remove(0); return o
        }
        val before = paperLabels(false); val outB = outside(before)
        val after = paperLabels(true); val outA = outside(after)
        // holes before whose paper is now joined to the outside
        val opened = HashSet<Int>()
        for (p in before.indices) if (before[p] > 0 && before[p] !in outB && after[p] in outA) opened.add(before[p])
        if (opened.isEmpty()) return
        for (p in before.indices) {
            val x = xa + p % cw; val y = ya + p / cw; val i = y * w + x
            if (!out[i] || keep[i]) continue
            // an erased pixel bordering an opened hole
            var near = false
            for (dy in -1..1) for (dx in -1..1) {
                val px = p % cw + dx; val py = p / cw + dy
                if (px in 0 until cw && py in 0 until ch && before[py * cw + px] in opened) near = true
            }
            if (near) keep[i] = true
        }
    }

    /** A 1-px line [len] long tilted by [deg] degrees, as a structuring element. */
    private fun lineKernel(len: Int, deg: Double): Mat {
        val half = len / 2
        val rise = tan(Math.toRadians(deg)) * half
        val dy = abs(rise).roundToInt()
        val kern = Mat.zeros(2 * dy + 1, 2 * half + 1, CvType.CV_8U)
        Imgproc.line(kern, Point(0.0, (dy - rise.roundToInt()).toDouble()),
            Point((2 * half).toDouble(), (dy + rise.roundToInt()).toDouble()), Scalar(1.0), 1)
        return kern
    }

    /** Standard deviation of the line's centre row around its least-squares straight fit. */
    private fun fitDeviation(cols: List<Int>, top: IntArray, bot: IntArray): Double {
        val m = cols.size; var sx = 0.0; var sy = 0.0; var sxx = 0.0; var sxy = 0.0
        for ((j, c) in cols.withIndex()) {
            val y = (top[c] + bot[c]) / 2.0
            sx += j; sy += y; sxx += j.toDouble() * j; sxy += j * y
        }
        val den = m * sxx - sx * sx
        val a = if (den != 0.0) (m * sxy - sx * sy) / den else 0.0
        val b = (sy - a * sx) / m
        var ss = 0.0
        for ((j, c) in cols.withIndex()) { val r = (top[c] + bot[c]) / 2.0 - (a * j + b); ss += r * r }
        return sqrt(ss / m)
    }
}
