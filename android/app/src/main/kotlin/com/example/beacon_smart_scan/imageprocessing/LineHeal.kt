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
    /**
     * Off: measured on the erased output of 21 labelled pages (3 passes, as the app), healing took
     * print away — grid paper's lines beside the handwriting, table edges — print kept 97.8% with
     * it, 99.3% without, for 0.2 points of handwriting. Dev switch: InkHarness `-e lineheal 1`.
     */
    @JvmStatic var enabled = false
    /** Dev switch (InkHarness `-e healtext 0`): also draw lines that run through text. */
    @JvmStatic var notThroughText = true
    /** Dev counters of the last run (InkHarness report). */
    @JvmStatic var stats = ""
    private val reject = IntArray(6)
    private var log = ""
    private var bridged = 0; private var refused = IntArray(2)

    private const val INK = 1
    private const val PURE = 2
    private const val HW = 4
    private const val ZONE = 8
    private const val LINE_H = 16
    private const val LINE_V = 32

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
            val tol = max(3.0, (a.th + b.th) / 2 + 1)
            if (abs(a.off(b.px, b.py)) > tol || abs(a.off(b.qx, b.qy)) > tol) continue
            parent[root(i)] = root(j)
        }
        val groups = (0 until n).groupBy { root(it) }.values

        val maxGap = max(w, h) / 5
        // each line: its pieces chained along it while the break between two is one the pen made;
        // a line is then known by its two ends only, and drawn again as one straight stroke from
        // the first to the last — not patched piece by piece
        class Line(val ax: Double, val ay: Double, val bx: Double, val by: Double, val th: Double,
                   val profile: Array<DoubleArray>, val broken: Boolean) {
            val horizontal = abs(bx - ax) >= abs(by - ay)
        }
        val found = ArrayList<Line>()
        for (members in groups) {
            val ps = members.map { pieces[it] }
            val ref = ps.maxByOrNull { it.len }!!
            class Span(val t0: Double, val t1: Double, val p: Piece)
            val spans = ps.map { p ->
                val a = ref.along(p.px, p.py); val b = ref.along(p.qx, p.qy)
                Span(min(a, b), max(a, b), p)
            }.sortedBy { it.t0 }
            var chain = arrayListOf(spans[0]); var end = spans[0].t1; var broken = false
            fun close() {
                // the straight line through the chain: offsets of its pieces' ends from ref's line,
                // least squares weighted by length
                var sw = 0.0; var st = 0.0; var so = 0.0; var stt = 0.0; var sto = 0.0
                for (c in chain) for ((x, y) in listOf(c.p.px to c.p.py, c.p.qx to c.p.qy)) {
                    val wgt = c.p.len; val t = ref.along(x, y); val o = ref.off(x, y)
                    sw += wgt; st += wgt * t; so += wgt * o; stt += wgt * t * t; sto += wgt * t * o
                }
                val den = sw * stt - st * st
                val sl = if (abs(den) > 1e-9) (sw * sto - st * so) / den else 0.0
                val ic = (so - sl * st) / sw
                val t0 = chain.minOf { it.t0 }; val t1 = end
                fun px(t: Double) = ref.px + t * ref.ux - (sl * t + ic) * ref.uy
                fun py(t: Double) = ref.py + t * ref.uy + (sl * t + ic) * ref.ux
                val best = chain.maxByOrNull { it.p.len }!!.p
                val th = chain.map { it.p.th }.sorted()[chain.size / 2]
                found.add(Line(px(t0), py(t0), px(t1), py(t1), th, best.profile, broken))
            }
            for (sp in spans.drop(1)) {
                if (sp.t0 <= end + 1) { chain.add(sp); end = max(end, sp.t1); continue }
                if (joinable(ref, end, sp.t0, maxGap, k, ::at)) { chain.add(sp); end = max(end, sp.t1); broken = true; continue }
                close()
                chain = arrayListOf(sp); end = sp.t1; broken = false
            }
            close()
        }
        // the lines' own pixels are never wiped as stray ink around a line square to them
        for (l in found) {
            val len = hypot(l.bx - l.ax, l.by - l.ay); if (len < 1) continue
            val ux = (l.bx - l.ax) / len; val uy = (l.by - l.ay) / len
            val half = l.th / 2 + 0.5
            val bit = if (l.horizontal) LINE_H else LINE_V
            var t = 0.0
            while (t <= len) {
                var d = -half
                while (d <= half) {
                    val x = (l.ax + t * ux - d * uy).roundToInt(); val y = (l.ay + t * uy + d * ux).roundToInt()
                    if (x in 0 until w && y in 0 until h) flags[y * w + x] = (flags[y * w + x].toInt() or bit).toByte()
                    d += 0.5
                }
                t += 0.5
            }
        }
        for (l in found) {
            val len = hypot(l.bx - l.ax, l.by - l.ay); if (len < 1) continue
            val ux = (l.bx - l.ax) / len; val uy = (l.by - l.ay) / len
            // its true ends: past the pieces, dashes of it the pen strokes left
            val ea = reachEnd(l.ax, l.ay, -ux, -uy, l.th, maxGap, k, ::at)
            val eb = reachEnd(l.bx, l.by, ux, uy, l.th, maxGap, k, ::at)
            val ax = l.ax - ea * ux; val ay = l.ay - ea * uy
            val bx = l.bx + eb * ux; val by = l.by + eb * uy
            // only a line the pen touched is drawn again
            if (!l.broken && ea == 0 && eb == 0 && !touched(ax, ay, bx, by, l.th, k, ::at)) continue
            // a rule runs under or between the text, never through it: ink right above AND right
            // below it along much of its length is a stroke across letters (a pen strike-through, the
            // tops of a row of letters), not a printed line
            if (notThroughText && throughText(ax, ay, bx, by, l.th, k, ::at)) { refused[1]++; continue }
            bridged++
            log += "line (%.0f,%.0f)-(%.0f,%.0f) th %.1f ends +%d +%d\n".format(ax, ay, bx, by, l.th, ea, eb)
            paint(dst, paper, ax, ay, bx, by, l.th, l.profile, k, at = ::at, other = if (l.horizontal) LINE_V else LINE_H)
        }
        paper.release()
        stats += log
        stats += "hough $houghCount pieces $n groups ${groups.size} lines ${found.size} reject ${reject.joinToString()} redrawn $bridged refused ${refused.joinToString()} k $k\n"
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
        var firstHit = -1.0; var lastHit = 0.0
        var t = 0.0
        while (t <= len) {
            total++
            val bx = x1 + t * ux; val by = y1 + t * uy
            fun on(d: Int) = at((bx + d * nx).roundToInt(), (by + d * ny).roundToInt(), INK)
            // the ink run nearest the centre
            val start = (0..2).flatMap { listOf(it, -it) }.firstOrNull { on(it) }
            if (start != null) {
                hit++
                if (firstHit < 0) firstHit = t; lastHit = t
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
        // a rule is thin, and as thick all the way: the foot of a row of small letters is neither
        if (th > max(3.0, 0.6 * k + 1)) { reject[2]++; return null }
        if (ths.count { abs(it - th) <= 1 } < 0.8 * ths.size) { reject[2]++; return null }
        var res = 0.0
        for (i in ts.indices) { val e = cs[i] - alpha * ts[i] - beta; res += e * e }
        if (sqrt(res / m) > th / 2 + 0.75) { reject[3]++; return null }                    // not one straight line
        // its ends where its ink is: the Hough walk ran on the grown ink, a pixel or more past them
        val px = x1 + firstHit * ux + (alpha * firstHit + beta) * nx; val py = y1 + firstHit * uy + (alpha * firstHit + beta) * ny
        val qx = x1 + lastHit * ux + (alpha * lastHit + beta) * nx; val qy = y1 + lastHit * uy + (alpha * lastHit + beta) * ny
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
     * Whether the break in the line [p] from [t0] to [t1] (positions along it) is one the pen made:
     * not too long, the pen beside it (or so short a printed rule could not have it), no print on
     * both sides (a text line between two rules), and no other line meeting it at either side (a
     * box's corner: its edge ends there, the next box's edge is another line).
     */
    private fun joinable(p: Piece, t0: Double, t1: Double, maxGap: Int, k: Int, at: (Int, Int, Int) -> Boolean): Boolean {
        val gap = t1 - t0
        if (gap > maxGap || gap > p.len) { refused[0]++; return false }
        val nx = -p.uy; val ny = p.ux
        val half = p.th / 2 + 0.5
        val band = p.th / 2 + k
        fun hit(t: Double, off: Double, span: Double, f: Int): Boolean {
            val cx = p.px + t * p.ux + off * nx; val cy = p.py + t * p.uy + off * ny
            var d = -span
            while (d <= span) { if (at((cx + d * nx).roundToInt(), (cy + d * ny).roundToInt(), f)) return true; d += 1.0 }
            return false
        }
        val reach = p.th + 3
        fun corner(t: Double): Boolean {
            if (!hit(t, 0.0, half + 1, INK)) return false
            val a = hit(t, -reach, 0.0, INK) && !hit(t, -reach, 0.0, ZONE)
            val b = hit(t, reach, 0.0, INK) && !hit(t, reach, 0.0, ZONE)
            return a || b
        }
        var c = t0 - 4
        while (c <= t1 + 4) { if (corner(c)) { refused[1]++; return false }; c += 1.0 }
        val side = p.th / 2 + (k + 2) / 2.0
        var pen = 0; var printed = 0; var n = 0
        var t = t0
        while (t <= t1) {
            n++
            if (hit(t, 0.0, band, ZONE)) pen++
            if (hit(t, -side, k / 2.0, PURE) && hit(t, side, k / 2.0, PURE)) printed++
            t += 1.0
        }
        if (printed > 0.3 * n) { refused[1]++; return false }
        // the pen across most of it (a box's edge next to an answer is not crossed by it)
        return pen >= 0.6 * n || gap <= 2 * k
    }

    /**
     * From a line's end (ex, ey), outwards along (dx, dy): across the pen's zone, dashes of the line
     * (ink on the straight line) the erase left; the walk stops at paper with no pen, between print,
     * or at another line crossing. Returns how far the line truly goes on (0: it ends here).
     */
    private fun reachEnd(ex: Double, ey: Double, dx: Double, dy: Double, th: Double, maxGap: Int, k: Int,
                         at: (Int, Int, Int) -> Boolean): Int {
        val nx = -dy; val ny = dx
        val half = th / 2 + 1.5
        val reach = (th + 2).toInt() + 1
        val side = th / 2 + (k + 2) / 2.0
        var lastInk = 0; var misses = 0; var pen = 0; var penAtLast = 0
        var t = 1
        while (t <= maxGap) {
            val cx = ex + t * dx; val cy = ey + t * dy
            fun f(d: Double, flag: Int) = at((cx + d * nx).roundToInt(), (cy + d * ny).roundToInt(), flag)
            fun near(off: Double, flag: Int): Boolean {
                var e = -k / 2.0
                while (e <= k / 2.0) { if (f(off + e, flag)) return true; e += 1.0 }
                return false
            }
            if (near(-side, PURE) && near(side, PURE)) break
            var onLine = false; var d = -half
            while (d <= half) { if (f(d, INK)) onLine = true; d += 1.0 }
            // another line across the way ahead (the next box's edge): the line ends before it
            var crossing = false
            var a = -half
            while (a <= half + 1 && !crossing) {
                val sx = cx + a * dx; val sy = cy + a * dy
                fun g(d: Double) = at((sx + d * nx).roundToInt(), (sy + d * ny).roundToInt(), INK) &&
                    !at((sx + d * nx).roundToInt(), (sy + d * ny).roundToInt(), ZONE)
                if (g(-reach.toDouble()) || g(reach.toDouble())) crossing = true
                a += 1.0
            }
            if (crossing) break
            val inZone = f(0.0, ZONE)
            if (inZone) pen++
            if (onLine) { lastInk = t; penAtLast = pen; misses = 0 }
            else if (inZone) misses = 0
            else if (++misses > 2) break
            t++
        }
        return if (lastInk <= 1 || penAtLast < 0.6 * lastInk) 0 else lastInk
    }

    /** Share of a line's length with print or ink right above and right below it at which it runs through text. */
    private const val THROUGH_TEXT = 0.25

    /** Whether the line from (ax, ay) to (bx, by) has ink or print right beside it on both sides along much of it. */
    private fun throughText(ax: Double, ay: Double, bx: Double, by: Double, th: Double, k: Int, at: (Int, Int, Int) -> Boolean): Boolean {
        val len = hypot(bx - ax, by - ay); if (len < 1) return false
        val ux = (bx - ax) / len; val uy = (by - ay) / len
        val lo = th / 2 + 1.5; val hi = th / 2 + k + 1
        var both = 0; var n = 0
        var t = 0.0
        while (t <= len) {
            n++
            val cx = ax + t * ux; val cy = ay + t * uy
            fun side(sign: Int): Boolean {
                var d = lo
                while (d <= hi) {
                    val x = (cx - sign * d * uy).roundToInt(); val y = (cy + sign * d * ux).roundToInt()
                    if (at(x, y, INK or PURE) && !at(x, y, ZONE)) return true
                    d += 1.0
                }
                return false
            }
            if (side(1) && side(-1)) both++
            t += 1.0
        }
        return both > THROUGH_TEXT * n
    }

    /** Whether the pen was anywhere along the line from (ax, ay) to (bx, by). */
    private fun touched(ax: Double, ay: Double, bx: Double, by: Double, th: Double, k: Int, at: (Int, Int, Int) -> Boolean): Boolean {
        val len = hypot(bx - ax, by - ay)
        val ux = (bx - ax) / len; val uy = (by - ay) / len
        val band = th / 2 + k
        var t = 0.0
        while (t <= len) {
            var d = -band
            while (d <= band) {
                if (at((ax + t * ux - d * uy).roundToInt(), (ay + t * uy + d * ux).roundToInt(), ZONE)) return true
                d += 1.0
            }
            t += 2.0
        }
        return false
    }

    /**
     * The line drawn as one straight stroke from (ax, ay) to (bx, by) with [profile] (darkness under
     * the paper at each offset from the centre: the line's own pixel rows and shade). Around it, in
     * its band: what is left of the old line beside the new one (ink a pixel or two off it) and, near
     * the pen, any stray ink become paper — but never the pixels of a line square to it ([other]).
     */
    private fun paint(dst: Mat, paper: Mat, ax: Double, ay: Double, bx: Double, by: Double, th: Double,
                      profile: Array<DoubleArray>, k: Int, at: (Int, Int, Int) -> Boolean, other: Int) {
        val len = hypot(bx - ax, by - ay)
        if (len < 1) return
        val ux = (bx - ax) / len; val uy = (by - ay) / len
        val band = th / 2 + k
        val pad = band + k
        val x0 = max(0, floor(min(ax, bx) - pad).toInt()); val x1 = min(dst.cols() - 1, ceil(max(ax, bx) + pad).toInt())
        val y0 = max(0, floor(min(ay, by) - pad).toInt()); val y1 = min(dst.rows() - 1, ceil(max(ay, by) + pad).toInt())
        val pix = ByteArray(3); val pap = ByteArray(3)
        val dark = DoubleArray(3)
        val reach = profile.size / 2
        for (y in y0..y1) for (x in x0..x1) {
            val t = (x - ax) * ux + (y - ay) * uy
            if (t < -k || t > len + k) continue
            val d = -(x - ax) * uy + (y - ay) * ux
            if (abs(d) > band) continue
            var on = false
            if (t >= 0 && t <= len) {
                val i = d.roundToInt() + reach
                for (ch in 0..2) dark[ch] = if (i in profile.indices) profile[i][ch] else 0.0
                on = dark.any { it > 0.0 }
            }
            if (!on) {
                if (at(x, y, other)) continue                                  // a line square to it
                val ink = at(x, y, INK)
                val oldLine = ink && abs(d) <= th / 2 + 2.5 && t >= 0 && t <= len && (at(x, y, ZONE) || !at(x, y, PURE) || at(x, y, LINE_H or LINE_V))
                val stray = at(x, y, ZONE)
                if (!oldLine && !stray) continue
            }
            paper.get(y, x, pap)
            for (ch in 0..2) {
                val p = (pap[ch].toInt() and 0xff).toDouble()
                pix[ch] = (if (on) p - dark[ch] else p).roundToInt().coerceIn(0, 255).toByte()
            }
            dst.put(y, x, pix)
        }
    }

}
