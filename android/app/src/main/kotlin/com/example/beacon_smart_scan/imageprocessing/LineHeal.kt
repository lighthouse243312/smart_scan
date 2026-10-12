package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Very last erase step, on the erased page itself: a printed line (answer line, underline, table
 * rule — at whatever angle the photo put it) the handwriting sat on often comes out broken, gaps and
 * dashes where the pen strokes crossed it. A line is found at its own angle: a run of ink whose
 * centre stays on one straight line from start to end, at one thickness. Pieces of the same straight
 * line, with a gap between them that the handwriting covered, are joined again at the line's own
 * cross-section — the same pixels, as dark, as the line itself — anything else left in the thin band around the line near the handwriting
 * (pixels not on the straight line) becomes paper.
 * Mirrors LineHeal in ios/Runner/ImageProcessingOpenCV.mm.
 */
object LineHeal {
    /** Dev switch (InkHarness `-e lineheal 0`). */
    @JvmStatic var enabled = true
    /** Dev counters of the last run (InkHarness report). */
    @JvmStatic var stats = ""
    private val reject = IntArray(6)
    private var log = ""
    private var bridged = 0; private var refused = IntArray(2)

    private const val INK = 1
    private const val PURE = 2
    private const val HW = 4
    private const val ZONE = 8

    /**
     * A straight piece of a line: centre line from p to q, [th] px thick. [profile]: its measured
     * cross-section, darkness under the paper (B, G, R) at offsets -reach..reach from the centre.
     */
    private class Piece(val px: Double, val py: Double, val qx: Double, val qy: Double, val th: Double, val profile: Array<DoubleArray>) {
        val len = hypot(qx - px, qy - py)
        val ux = (qx - px) / len
        val uy = (qy - py) / len
        /** Position along the piece's direction. */
        fun along(x: Double, y: Double) = (x - px) * ux + (y - py) * uy
        /** Signed distance from the piece's line. */
        fun off(x: Double, y: Double) = -(x - px) * uy + (y - py) * ux
    }

    /**
     * [dst]: the erased page (CV_8UC3), updated in place; [handwriting] / [print]: the mask's layers
     * (CV_8UC1). Print is never painted over: a gap with print beside the line is not joined, and only
     * pixels in the handwriting's neighbourhood are cleared.
     */
    fun apply(dst: Mat, handwriting: Mat, print: Mat) {
        if (!enabled) return
        val w = dst.cols(); val h = dst.rows()
        val k = InkAnalysis.strokeUnit(w, h)
        val gray = Mat(); Imgproc.cvtColor(dst, gray, Imgproc.COLOR_BGR2GRAY)
        val paper = Mat()
        Imgproc.dilate(gray, paper, Mat.ones(2 * k + 1, 2 * k + 1, CvType.CV_8U))
        Imgproc.medianBlur(paper, paper, 21)
        val ink = Mat(); Core.subtract(paper, gray, ink)
        Imgproc.threshold(ink, ink, 25.0, 255.0, Imgproc.THRESH_BINARY)
        // the paper's own colour, to paint with (the dilated grey above is its brightest, for finding ink)
        Imgproc.medianBlur(dst, paper, 21)

        // one byte per pixel: ink / pure print / handwriting (3×3 grown) / clearing zone
        val notHw = Mat(); Core.bitwise_not(handwriting, notHw)
        val pure = Mat(); Core.bitwise_and(print, notHw, pure)
        val hwNear = Mat(); Imgproc.dilate(handwriting, hwNear, Mat.ones(3, 3, CvType.CV_8U))
        val zone = Mat(); Imgproc.dilate(handwriting, zone, Mat.ones(2 * k + 1, 2 * k + 1, CvType.CV_8U))
        val notPure = Mat(); Core.bitwise_not(pure, notPure); Core.bitwise_and(zone, notPure, zone)
        val flagsMat = Mat.zeros(h, w, CvType.CV_8U)
        val bit = Mat()
        for ((m, b) in listOf(ink to INK, pure to PURE, hwNear to HW, zone to ZONE)) {
            Core.min(m, Scalar(b.toDouble()), bit); Core.bitwise_or(flagsMat, bit, flagsMat)
        }
        val flags = ByteArray(w * h); flagsMat.get(0, 0, flags)
        listOf(notHw, pure, hwNear, zone, notPure, flagsMat, bit).forEach { it.release() }
        fun at(x: Int, y: Int, f: Int) = x in 0 until w && y in 0 until h && (flags[y * w + x].toInt() and f) != 0

        // candidates at any angle, then each measured along its own direction
        val minLen = max(4 * k, max(w, h) / 40)
        val lines = Mat()
        // on the ink grown by a pixel: a thin tilted line steps from row to row, which the Hough walk
        // loses; the pieces are then measured on the ink itself
        val grown = Mat(); Imgproc.dilate(ink, grown, Mat.ones(3, 3, CvType.CV_8U))
        Imgproc.HoughLinesP(grown, lines, 1.0, PI / 360, minLen / 2, minLen.toDouble(), 3.0)
        grown.release()
        reject.fill(0); bridged = 0; log = ""; refused.fill(0)
        val houghCount = lines.total().toInt()
        val all = IntArray(houghCount * 4); if (houghCount > 0) lines.get(0, 0, all)
        val pieces = ArrayList<Piece>()
        for (i in 0 until houghCount) {
            measure(all.copyOfRange(4 * i, 4 * i + 4), k, dst, paper, ::at)?.let { pieces.add(it) }
        }
        lines.release(); gray.release(); ink.release()

        // the page's rules run one way (and square to it): the angle most of the pieces' length has,
        // modulo 90 degrees — a straight leftover of a pen stroke at another angle is not a rule
        val ang = pieces.map { Math.toDegrees(kotlin.math.atan2(it.uy, it.ux)).let { a -> ((a % 90) + 135) % 90 - 45 } }
        val order = pieces.indices.sortedBy { ang[it] }
        val half = pieces.sumOf { it.len } / 2
        var acc = 0.0; var main = 0.0
        for (i in order) { acc += pieces[i].len; if (acc >= half) { main = ang[i]; break } }
        val kept = pieces.indices.filter { i -> abs(((ang[i] - main) + 135) % 90 - 45) <= 2.0 }.map { pieces[it] }
        reject[4] += pieces.size - kept.size
        pieces.clear(); pieces.addAll(kept)
        // pieces of one straight line: same angle, each one's ends on the other's line
        val n = pieces.size
        val parent = IntArray(n) { it }
        fun root(i: Int): Int { var r = i; while (parent[r] != r) r = parent[r]; parent[i] = r; return r }
        for (i in 0 until n) for (j in i + 1 until n) {
            val a = pieces[i]; val b = pieces[j]
            if (abs(a.ux * b.uy - a.uy * b.ux) > 0.026) continue           // within 1.5 degrees
            val tol = max(2.0, (a.th + b.th) / 4 + 1)
            if (abs(a.off(b.px, b.py)) > tol || abs(a.off(b.qx, b.qy)) > tol) continue
            parent[root(i)] = root(j)
        }
        val groups = (0 until n).groupBy { root(it) }.values

        val maxGap = max(w, h) / 5
        for (members in groups) {
            val ps = members.map { pieces[it] }
            // the line: the longest piece's, over the whole stretch its pieces cover
            val ref = ps.maxByOrNull { it.len }!!
            val ts = ps.flatMap { listOf(ref.along(it.px, it.py), ref.along(it.qx, it.qy)) }
            val tMin = ts.minOrNull()!!; val tMax = ts.maxOrNull()!!
            val ax = ref.px + tMin * ref.ux; val ay = ref.py + tMin * ref.uy
            val bx = ref.px + tMax * ref.ux; val by = ref.py + tMax * ref.uy
            clearOffLine(dst, paper, ax, ay, bx, by, ref.th, k, ::at)
            fill(dst, paper, ref, tMin, tMax, maxGap, k, ::at)
            // past its ends: dashes of it left between the pen strokes
            extend(dst, paper, ax, ay, -ref.ux, -ref.uy, ref, maxGap, k, ::at)
            extend(dst, paper, bx, by, ref.ux, ref.uy, ref, maxGap, k, ::at)
        }
        paper.release()
        stats += log
        stats += "hough $houghCount pieces $n groups ${groups.size} reject ${reject.joinToString()} bridged $bridged refused ${refused.joinToString()} k $k\n"
    }

    /**
     * The Hough segment [seg] measured across, sample by sample along its direction: the run of ink
     * at the centre gives the line's offset and thickness there. Accepted only as a printed line:
     * ink nearly all the way, mostly thin, and the centre on one straight line from start to end
     * (a hand-drawn stroke or a row of letters wanders off it). Returns the fitted piece.
     */
    private fun measure(seg: IntArray, k: Int, dst: Mat, paper: Mat, at: (Int, Int, Int) -> Boolean): Piece? {
        val x1 = seg[0].toDouble(); val y1 = seg[1].toDouble(); val x2 = seg[2].toDouble(); val y2 = seg[3].toDouble()
        val len = hypot(x2 - x1, y2 - y1)
        if (len < 1) return null
        val ux = (x2 - x1) / len; val uy = (y2 - y1) / len
        val nx = -uy; val ny = ux
        val reach = k + 3
        val ts = ArrayList<Double>(); val cs = ArrayList<Double>(); val ths = ArrayList<Double>()
        var hit = 0; var total = 0
        var t = 0.0
        while (t <= len) {
            total++
            val bx = x1 + t * ux; val by = y1 + t * uy
            fun on(d: Int) = at((bx + d * nx).roundToInt(), (by + d * ny).roundToInt(), INK)
            // the ink run nearest the centre
            val start = (0..2).flatMap { listOf(it, -it) }.firstOrNull { on(it) }
            if (start != null) {
                hit++
                var lo = start; while (lo > -reach && on(lo - 1)) lo--
                var hi = start; while (hi < reach && on(hi + 1)) hi++
                val thick = (hi - lo + 1).toDouble()
                if (lo > -reach && hi < reach && thick <= k + 2) { ts.add(t); cs.add((lo + hi) / 2.0); ths.add(thick) }
            }
            t += 1.0
        }
        if (hit < 0.9 * total) { reject[0]++; return null }
        if (ts.size < 0.8 * total || ts.size < 8) { reject[1]++; return null }
        // centre offset c = α·t + β, least squares
        val m = ts.size.toDouble()
        var st = 0.0; var sc = 0.0; var stt = 0.0; var stc = 0.0
        for (i in ts.indices) { st += ts[i]; sc += cs[i]; stt += ts[i] * ts[i]; stc += ts[i] * cs[i] }
        val den = m * stt - st * st
        val alpha = if (den != 0.0) (m * stc - st * sc) / den else 0.0
        val beta = (sc - alpha * st) / m
        val th = ths.sorted()[ths.size / 2]
        if (th > k + 2) { reject[2]++; return null }
        var res = 0.0
        for (i in ts.indices) { val e = cs[i] - alpha * ts[i] - beta; res += e * e }
        if (sqrt(res / m) > th / 2 + 0.75) { reject[3]++; return null }                    // not one straight line
        val px = x1 + beta * nx; val py = y1 + beta * ny
        val qx = x2 + (alpha * len + beta) * nx; val qy = y2 + (alpha * len + beta) * ny
        // the line's cross-section: per offset from the fitted centre, the median darkness under the
        // paper over the clean samples — copied as is into a gap, so the join has the line's own
        // pixel thickness and shade
        val step = max(1, ts.size / 96)
        val picks = ts.indices.filter { it % step == 0 }
        val vals = Array(2 * reach + 1) { Array(3) { DoubleArray(picks.size) } }
        val pix = ByteArray(3); val pap = ByteArray(3)
        var cn = 0
        for (i in picks) {
            val c = alpha * ts[i] + beta
            val cx = x1 + ts[i] * ux + c * nx; val cy = y1 + ts[i] * uy + c * ny
            var ok = true
            for (d in -reach..reach) {
                val sx = (cx + d * nx).roundToInt(); val sy = (cy + d * ny).roundToInt()
                if (sx !in 0 until dst.cols() || sy !in 0 until dst.rows()) { ok = false; break }
            }
            if (!ok) continue
            for (d in -reach..reach) {
                val sx = (cx + d * nx).roundToInt(); val sy = (cy + d * ny).roundToInt()
                dst.get(sy, sx, pix); paper.get(sy, sx, pap)
                for (ch in 0..2) vals[d + reach][ch][cn] = ((pap[ch].toInt() and 0xff) - (pix[ch].toInt() and 0xff)).toDouble()
            }
            cn++
        }
        if (cn < 4) { reject[5]++; return null }
        val profile = Array(2 * reach + 1) { j ->
            DoubleArray(3) { ch ->
                val v = vals[j][ch].copyOf(cn).sorted()[cn / 2]
                if (abs(j - reach) > (th + 1) / 2 || v < 8) 0.0 else v     // only the line itself
            }
        }
        if (profile.maxOf { it.average() } < 20) { reject[5]++; return null }    // no line there after all
        return Piece(px, py, qx, qy, th, profile)
    }

    /**
     * Along the line [p] from [t0] to [t1], step by step at its own angle: where the pen was, the line
     * is drawn again with its own cross-section (its own pixel rows and shade). A missing stretch with print
     * beside it on both sides (a text line between two pieces of a rule) is left as it is.
     */
    private fun fill(dst: Mat, paper: Mat, p: Piece, t0: Double, t1: Double, maxGap: Int, k: Int,
                     at: (Int, Int, Int) -> Boolean) {
        val nx = -p.uy; val ny = p.ux
        val half = p.th / 2 + 0.5
        val band = p.th / 2 + k
        fun hit(t: Double, off: Double, span: Double, f: Int): Boolean {
            val cx = p.px + t * p.ux + off * nx; val cy = p.py + t * p.uy + off * ny
            var d = -span
            while (d <= span) { if (at((cx + d * nx).roundToInt(), (cy + d * ny).roundToInt(), f)) return true; d += 1.0 }
            return false
        }
        val side = p.th / 2 + (k + 2) / 2.0
        var t = t0
        var runStart = Double.NaN; var printed = 0; var len = 0
        fun close(end: Double) {
            if (runStart.isNaN()) return
            val n = end - runStart
            if (n <= maxGap && n <= p.len && printed <= 0.3 * len) {
                bridged++
                log += "fill (%.0f,%.0f) +%.0f u (%.2f,%.2f) th %.1f prof %s\n".format(p.px + runStart * p.ux, p.py + runStart * p.uy, n, p.ux, p.uy, p.th, p.profile.joinToString(",") { it[1].toInt().toString() })
                paint(dst, paper, p.px + (runStart - 0.5) * p.ux, p.py + (runStart - 0.5) * p.uy,
                    p.px + (end + 0.5) * p.ux, p.py + (end + 0.5) * p.uy, p.th, p.profile, k, at, ext = 0.0)
            } else refused[1]++
            runStart = Double.NaN
        }
        var lastMissing = Double.NaN
        while (t <= t1) {
            // where the pen was, the line is redrawn whole: its ink there may be gone, dashes or
            // thickened by what is left of the pen
            val missing = hit(t, 0.0, band, ZONE)
            if (missing) {
                if (runStart.isNaN() || t - lastMissing > 2) { close(lastMissing); runStart = t; printed = 0; len = 0 }
                len++
                if (hit(t, -side, k / 2.0, PURE) && hit(t, side, k / 2.0, PURE)) printed++
                lastMissing = t
            }
            t += 1.0
        }
        close(lastMissing)
    }

    /**
     * From a line's end (ex, ey), outwards along (dx, dy): across the pen's zone, dashes of the line
     * (thin ink on the straight line) the erase left are collected; the walk stops at paper with no
     * pen, or between print. The line is drawn on to the farthest dash.
     */
    private fun extend(dst: Mat, paper: Mat, ex: Double, ey: Double, dx: Double, dy: Double, p: Piece,
                       maxGap: Int, k: Int, at: (Int, Int, Int) -> Boolean) {
        val nx = -dy; val ny = dx
        val half = p.th / 2 + 0.5
        val reach = (p.th + 2).toInt() + 1
        var lastInk = 0; var misses = 0; var pen = 0
        var t = 1
        while (t <= maxGap) {
            val cx = ex + t * dx; val cy = ey + t * dy
            fun f(d: Double, flag: Int) = at((cx + d * nx).roundToInt(), (cy + d * ny).roundToInt(), flag)
            // print beside the line on both sides: a text line, the rule does not go on there
            val side = p.th / 2 + (k + 2) / 2.0
            fun near(off: Double, flag: Int): Boolean {
                var e = -k / 2.0
                while (e <= k / 2.0) { if (f(off + e, flag)) return true; e += 1.0 }
                return false
            }
            if (near(-side, PURE) && near(side, PURE)) break
            // thin ink on the line: inside the line's thickness, paper just beyond it
            var onLine = false; var d = -half
            while (d <= half) { if (f(d, INK)) onLine = true; d += 1.0 }
            // ink running on sideways: another line crossing (a box's corner) — the line ends there
            if (onLine && (f(-reach.toDouble(), INK) || f(reach.toDouble(), INK))) break
            val thin = onLine
            val inZone = f(0.0, ZONE)
            if (inZone) pen++
            if (thin) { lastInk = t; misses = 0 }
            else if (inZone) misses = 0
            else if (++misses > 2) break
            t++
        }
        if (lastInk <= 1 || pen == 0) return
        log += "extend (%.0f,%.0f) by %d u (%.2f,%.2f)\n".format(ex, ey, lastInk, dx, dy)
        paint(dst, paper, ex, ey, ex + lastInk * dx, ey + lastInk * dy, p.th, p.profile, k, at)
    }

    /** Pixels in the band around a piece, off its straight line and near the handwriting, become paper. */
    private fun clearOffLine(dst: Mat, paper: Mat, ax: Double, ay: Double, bx: Double, by: Double, th: Double,
                             k: Int, at: (Int, Int, Int) -> Boolean) =
        paint(dst, paper, ax, ay, bx, by, th, null, k, at)

    /**
     * Over the band of the straight line from (ax, ay) to (bx, by) (and [k] px past its ends): pixels
     * off the line, in the clearing zone, become paper; with a [profile], the line itself is drawn
     * with that cross-section (darkness under the paper at each offset from the centre).
     */
    private fun paint(dst: Mat, paper: Mat, ax: Double, ay: Double, bx: Double, by: Double, th: Double,
                      profile: Array<DoubleArray>?, k: Int, at: (Int, Int, Int) -> Boolean, ext: Double = k.toDouble()) {
        val len = hypot(bx - ax, by - ay)
        if (len < 1) return
        val ux = (bx - ax) / len; val uy = (by - ay) / len
        val band = th / 2 + k
        val pad = band + k
        val x0 = max(0, floor(min(ax, bx) - pad).toInt()); val x1 = min(dst.cols() - 1, ceil(max(ax, bx) + pad).toInt())
        val y0 = max(0, floor(min(ay, by) - pad).toInt()); val y1 = min(dst.rows() - 1, ceil(max(ay, by) + pad).toInt())
        val pix = ByteArray(3); val pap = ByteArray(3)
        val dark = DoubleArray(3)
        for (y in y0..y1) for (x in x0..x1) {
            val t = (x - ax) * ux + (y - ay) * uy
            if (t < -ext || t > len + ext) continue
            val d = -(x - ax) * uy + (y - ay) * ux
            if (abs(d) > band) continue
            var on = false
            if (profile != null) {
                // the cross-section's pixel row at d: the line's own rows, as many as it has
                val reach = profile.size / 2
                val i = d.roundToInt() + reach
                for (ch in 0..2) dark[ch] = if (i in profile.indices) profile[i][ch] else 0.0
                on = dark.any { it > 0.0 }
            } else if (abs(d) <= th / 2 + 0.5) continue                   // the line itself: keep
            val inZone = at(x, y, ZONE)
            if (!on && (!inZone || (profile == null && !at(x, y, INK)))) continue
            paper.get(y, x, pap)
            for (ch in 0..2) {
                val p = (pap[ch].toInt() and 0xff).toDouble()
                pix[ch] = (if (on) p - dark[ch] else p).roundToInt().coerceIn(0, 255).toByte()
            }
            dst.put(y, x, pix)
        }
    }
}
