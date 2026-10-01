package com.example.beacon_smart_scan.imageprocessing

import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/**
 * Final per-ink-component vote over the detectors' layers, by the ink's own colour against the
 * local print colour — no pen hue assumed, so a page with two pens (a purple answer pen and a blue
 * marking pen) is handled like one with a single pen.
 *
 * Measured on a real worksheet (full resolution, 3x3 chroma): printed glyphs deviate ≤ 0.016 from
 * the print colour with 0% of their pixels past [PEN_DEV]; pen strokes deviate 0.09-0.14 with
 * 85-100% past it. The layout heuristics upstream can get whole lines wrong (a printed word on a
 * line of handwriting marked pen, pen letters on a printed line marked print); this settles them:
 *  - pen-coloured component: all of it is handwriting (a stroke that does not touch print is
 *    erased whole), except printed underscores/rules fused into it;
 *  - neutral component: print when it is a picture, a thin rule, or the model does not see
 *    handwriting in it (neutral handwriting — black ballpoint — is left to the detectors);
 *  - mixed (pen touching print): each pixel goes with the nearer confident colour, printed rules
 *    stay print.
 * Heavy per-pixel work stays in OpenCV (native) memory; per-component counts stream row by row,
 * so the Java heap only holds a few rows and per-component tables.
 */
object PenComponentVote {
    private const val INK_OD = 0.30
    private const val CLIPPED_OD = 1.6
    /** Chroma distance from the local print colour past which a pixel is pen-coloured. */
    private const val PEN_DEV = 0.03
    /** ...and below which it is confidently print-coloured. */
    private const val NEUTRAL_DEV = 0.015
    /** A thin horizontal run is a printed rule when enough of its measurable pixels are neutral. */
    private const val RULE_MIN_NEUTRAL = 10
    private const val RULE_NEUTRAL_SHARE = 0.08
    /** ...or when it is shaped like one (see [ruleShaped]). */
    private const val RULE_MAX_WOBBLE = 0.7
    private const val RULE_MAX_THICK = 0.45
    private const val HIST_BINS = 1000
    /** A neutral component with less than this share of measurable (unclipped) ink is toner print. */
    private const val CLIPPED_SHARE = 0.25f

    /**
     * Updates [handwriting], [print] and [overlap] (CV_8UC1, 255 = set, [src]'s size) in place.
     * [modelHw]: the segmentation model's handwriting probability (CV_32F, [src]'s size) or null.
     */
    /** Dev switch for comparing with/without the vote (InkHarness `-e vote 0`). */
    @JvmStatic var enabled = true

    fun apply(src: Mat, handwriting: Mat, print: Mat, overlap: Mat, modelHw: Mat?) {
        if (!enabled) return
        val w = src.cols(); val h = src.rows()
        val k = InkAnalysis.strokeUnit(w, h)
        val (chroma, mean) = chromaAndDensity(src)
        val ink = Mat(); val valid = Mat(); val mag = Mat()
        try {
            Imgproc.threshold(mean, ink, INK_OD, 255.0, Imgproc.THRESH_BINARY)
            ink.convertTo(ink, CvType.CV_8U)
            val sw = chroma[3]
            Imgproc.threshold(sw, valid, 0.2, 255.0, Imgproc.THRESH_BINARY)
            valid.convertTo(valid, CvType.CV_8U)
            Core.bitwise_and(valid, ink, valid)
            val ref = printReference(chroma, valid, print, handwriting, mean, w, h)
            releasing(*ref) {
                val d = Mat(); val sq = Mat()
                mag.create(h, w, CvType.CV_32F); mag.setTo(Scalar(0.0))
                releasing(d, sq) {
                    for (c in 0 until 3) {
                        Core.subtract(chroma[c], ref[c], d)
                        Core.multiply(d, d, sq)
                        Core.add(mag, sq, mag)
                    }
                    Core.sqrt(mag, mag)
                }
            }
            chroma.forEach { it.release() }
            vote(ink, valid, mag, handwriting, print, overlap, modelHw, k)
        } finally {
            chroma.forEach { it.release() }
            releasing(ink, valid, mag, mean) {}
        }
    }

    /** Local ink chroma (OD_B, OD_G, OD_R)/sum over a 3x3 window of unclipped ink, plus that window's ink fill (index 3); and the mean optical density. */
    private fun chromaAndDensity(bgr: Mat): Pair<Array<Mat>, Mat> {
        val small = Mat(); val paper = Mat(); val od = Mat()
        val kernel = Mat.ones(9, 9, CvType.CV_8U)
        releasing(small, paper, kernel) {
            // paper colour, as InkAnalysis.opticalDensity estimates it
            Imgproc.resize(bgr, small, Size(), 0.125, 0.125, Imgproc.INTER_AREA)
            Imgproc.medianBlur(small, small, 15)
            Imgproc.dilate(small, small, kernel)
            Imgproc.resize(small, paper, bgr.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            paper.convertTo(paper, CvType.CV_32FC3)
            Imgproc.GaussianBlur(paper, paper, Size(0.0, 0.0), 8.0)
            Core.add(paper, Scalar(1.0, 1.0, 1.0), paper)
            bgr.convertTo(od, CvType.CV_32FC3)
            Core.add(od, Scalar(1.0, 1.0, 1.0), od)
            Core.divide(od, paper, od)
        }
        Core.min(od, Scalar(1.0, 1.0, 1.0), od)
        Core.max(od, Scalar(1e-3, 1e-3, 1e-3), od)
        Core.log(od, od)
        Core.multiply(od, Scalar(-1.0, -1.0, -1.0), od)
        val planes = ArrayList<Mat>(3)
        Core.split(od, planes)
        od.release()
        val mean = Mat()
        Core.add(planes[0], planes[1], mean)
        Core.add(mean, planes[2], mean)
        Core.multiply(mean, Scalar(1.0 / 3), mean)
        // weight: unclipped ink (clipped pixels carry no colour)
        val wk = Mat(); val lo = Mat(); val hi = Mat()
        Imgproc.threshold(mean, lo, INK_OD, 1.0, Imgproc.THRESH_BINARY)
        Imgproc.threshold(mean, hi, CLIPPED_OD, 1.0, Imgproc.THRESH_BINARY_INV)
        Core.multiply(lo, hi, wk)
        releasing(lo, hi) {}
        val win = Size(3.0, 3.0)
        val sums = Array(3) { c ->
            val t = Mat(); Core.multiply(planes[c], wk, t)
            Imgproc.boxFilter(t, t, -1, win)
            planes[c].release()
            t
        }
        val sw = Mat(); Imgproc.boxFilter(wk, sw, -1, win); wk.release()
        val total = Mat()
        Core.add(sums[0], sums[1], total); Core.add(total, sums[2], total)
        Core.add(total, Scalar(1e-6), total)
        for (s in sums) Core.divide(s, total, s)
        total.release()
        return Pair(arrayOf(sums[0], sums[1], sums[2], sw), mean)
    }

    /** Print colour per page region: per-cell median chroma of confident, dark print ink. */
    private fun printReference(chroma: Array<Mat>, valid: Mat, print: Mat, hw: Mat, mean: Mat, w: Int, h: Int): Array<Mat> {
        val sel = Mat(); val notHw = Mat(); val dark = Mat()
        Core.bitwise_not(hw, notHw)
        Core.bitwise_and(valid, print, sel)
        Core.bitwise_and(sel, notHw, sel)
        Imgproc.threshold(mean, dark, 0.5, 255.0, Imgproc.THRESH_BINARY)
        dark.convertTo(dark, CvType.CV_8U)
        Core.bitwise_and(sel, dark, sel)
        releasing(notHw, dark) {}
        val cell = max(64, max(w, h) / 12)
        val gh = (h + cell - 1) / cell; val gw = (w + cell - 1) / cell
        val hist = Array(gh * gw) { Array(3) { IntArray(HIST_BINS) } }
        val global = Array(3) { IntArray(HIST_BINS) }
        val count = IntArray(gh * gw)
        val selRow = ByteArray(w)
        val rows = Array(3) { FloatArray(w) }
        for (y in 0 until h) {
            sel.get(y, 0, selRow)
            for (c in 0 until 3) chroma[c].get(y, 0, rows[c])
            val base = (y / cell) * gw
            for (x in 0 until w) {
                if (selRow[x].toInt() == 0) continue
                val cellIdx = base + x / cell
                count[cellIdx]++
                for (c in 0 until 3) {
                    val b = (rows[c][x] * HIST_BINS).toInt().coerceIn(0, HIST_BINS - 1)
                    hist[cellIdx][c][b]++; global[c][b]++
                }
            }
        }
        sel.release()
        fun median(hs: IntArray): Float {
            val total = hs.sum(); if (total == 0) return 1f / 3
            var acc = 0
            for (i in hs.indices) { acc += hs[i]; if (acc * 2 >= total) return (i + 0.5f) / HIST_BINS }
            return 1f / 3
        }
        val g = FloatArray(3) { median(global[it]) }
        return Array(3) { c ->
            val grid = FloatArray(gh * gw) { if (count[it] >= 300) median(hist[it][c]) else g[c] }
            val small = Mat(gh, gw, CvType.CV_32F); small.put(0, 0, grid)
            Imgproc.GaussianBlur(small, small, Size(0.0, 0.0), 1.0)
            val full = Mat()
            Imgproc.resize(small, full, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            small.release()
            full
        }
    }

    /**
     * A printed rule by shape: long, straight (its centre line within [RULE_MAX_WOBBLE] px RMS of a
     * fitted line) and thin. Measured: printed rules 0.23-0.55 px RMS and ≈ k/3 thick; a pen
     * strike-through or hand-drawn underline 0.7-1.1 px and ≈ k/1.7 thick.
     */
    private fun ruleShaped(top: IntArray?, bot: IntArray?, minLen: Int, k: Int): Boolean {
        if (top == null || bot == null) return false
        var n = 0; var sx = 0.0; var sy = 0.0; var sxx = 0.0; var sxy = 0.0; var thick = 0.0
        for (c in top.indices) {
            if (bot[c] < 0) continue
            val cy = (top[c] + bot[c]) / 2.0
            n++; sx += c; sy += cy; sxx += c.toDouble() * c; sxy += c * cy; thick += bot[c] - top[c] + 1
        }
        if (n < minLen) return false
        val den = n * sxx - sx * sx
        val a = if (den != 0.0) (n * sxy - sx * sy) / den else 0.0
        val b = (sy - a * sx) / n
        var sq = 0.0
        for (c in top.indices) {
            if (bot[c] < 0) continue
            val r = (top[c] + bot[c]) / 2.0 - (a * c + b); sq += r * r
        }
        return kotlin.math.sqrt(sq / n) <= RULE_MAX_WOBBLE && thick / n <= RULE_MAX_THICK * k
    }

    private fun vote(ink: Mat, valid: Mat, mag: Mat, hw: Mat, print: Mat, overlap: Mat, modelHw: Mat?, k: Int) {
        val w = ink.cols(); val h = ink.rows()
        val penSeed = Mat(); val neutSeed = Mat(); val tmp = Mat()
        Imgproc.threshold(mag, penSeed, PEN_DEV, 255.0, Imgproc.THRESH_BINARY)
        penSeed.convertTo(penSeed, CvType.CV_8U); Core.bitwise_and(penSeed, valid, penSeed)
        Imgproc.threshold(mag, neutSeed, NEUTRAL_DEV, 255.0, Imgproc.THRESH_BINARY_INV)
        neutSeed.convertTo(neutSeed, CvType.CV_8U); Core.bitwise_and(neutSeed, valid, neutSeed)

        // each mixed pixel goes with the nearer confident colour
        val penSide = Mat()
        run {
            val dPen = Mat(); val dNeu = Mat()
            Core.bitwise_not(penSeed, tmp); Imgproc.distanceTransform(tmp, dPen, Geometry.DIST_L2, 3)
            Core.bitwise_not(neutSeed, tmp); Imgproc.distanceTransform(tmp, dNeu, Geometry.DIST_L2, 3)
            Core.compare(dPen, dNeu, penSide, Core.CMP_LT)
            releasing(dPen, dNeu) {}
        }
        // printed rules / blank underscores fused with pen strokes: thin horizontal runs, decided per
        // run — a printed rule shows neutral rims somewhere along it (its core is usually clipped,
        // and where letters sit on it the rims take their colour), a pen stroke along the line (a
        // strike-through, an underline) is pen-coloured wherever its colour can be measured
        val rule = Mat(); val ruleUnder = Mat()
        run {
            val runs = Mat(); val thick = Mat(); val segLabels = Mat(); val segStats = Mat(); val segCent = Mat()
            Imgproc.morphologyEx(ink, runs, Imgproc.MORPH_OPEN, Mat.ones(1, 2 * k + 1, CvType.CV_8U))
            Imgproc.morphologyEx(runs, thick, Imgproc.MORPH_OPEN, Mat.ones(k or 1, 1, CvType.CV_8U))
            Imgproc.dilate(thick, thick, Mat.ones(3, 3, CvType.CV_8U))
            Core.bitwise_not(thick, thick); Core.bitwise_and(runs, thick, runs)
            val segs = Imgproc.connectedComponentsWithStats(runs, segLabels, segStats, segCent, 8, CvType.CV_32S)
            val ss = IntArray(segs * 5); segStats.get(0, 0, ss)
            val measured = IntArray(segs); val neutral = IntArray(segs)
            // per long run, its top and bottom edge in each column (for the shape test)
            val minLen = 6 * k
            val top = arrayOfNulls<IntArray>(segs); val bot = arrayOfNulls<IntArray>(segs)
            for (l in 1 until segs) if (ss[l * 5 + Imgproc.CC_STAT_WIDTH] >= minLen) {
                val sw = ss[l * 5 + Imgproc.CC_STAT_WIDTH]
                top[l] = IntArray(sw) { Int.MAX_VALUE }; bot[l] = IntArray(sw) { -1 }
            }
            val lab = IntArray(w); val vRow = ByteArray(w); val nRow = ByteArray(w)
            for (y in 0 until h) {
                segLabels.get(y, 0, lab); valid.get(y, 0, vRow); neutSeed.get(y, 0, nRow)
                for (x in 0 until w) {
                    val l = lab[x]; if (l == 0) continue
                    if (vRow[x].toInt() != 0) measured[l]++
                    if (nRow[x].toInt() != 0) neutral[l]++
                    val t = top[l] ?: continue
                    val c = x - ss[l * 5 + Imgproc.CC_STAT_LEFT]
                    if (y < t[c]) t[c] = y
                    if (y > bot[l]!![c]) bot[l]!![c] = y
                }
            }
            val printed = BooleanArray(segs) {
                it > 0 && ((neutral[it] >= RULE_MIN_NEUTRAL && neutral[it] >= RULE_NEUTRAL_SHARE * measured[it]) ||
                    ruleShaped(top[it], bot[it], minLen, k))
            }
            val out = ByteArray(w)
            rule.create(h, w, CvType.CV_8U)
            for (y in 0 until h) {
                segLabels.get(y, 0, lab)
                for (x in 0 until w) out[x] = if (printed[lab[x]]) 255.toByte() else 0
                rule.put(y, 0, out)
            }
            // the rule runs on under a pen stroke crossing it: close its gaps along the row (where
            // there is ink); the part under the pen is erased with it and restored as print
            Imgproc.morphologyEx(rule, rule, Imgproc.MORPH_CLOSE, Mat.ones(1, 2 * k + 1, CvType.CV_8U))
            Imgproc.dilate(rule, rule, Mat.ones(3, 1, CvType.CV_8U))
            Core.bitwise_and(rule, ink, rule)
            Core.bitwise_and(rule, penSeed, ruleUnder)
            Core.bitwise_not(penSeed, tmp); Core.bitwise_and(rule, tmp, rule)
            Core.bitwise_not(rule, tmp); Core.bitwise_and(penSide, tmp, penSide)
            releasing(runs, thick, segLabels, segStats, segCent) {}
        }
        tmp.release()

        val labels = Mat(); val stats = Mat(); val centroids = Mat(); val pieces = Mat()
        val n = Imgproc.connectedComponentsWithStats(ink, labels, stats, centroids, 8, CvType.CV_32S)
        centroids.release()
        try {
            val st = IntArray(n * 5); stats.get(0, 0, st)
            val cntV = IntArray(n); val cntP = IntArray(n); val cntN = IntArray(n); val cntNear = IntArray(n); val sumMh = DoubleArray(n)
            val lab = IntArray(w); val vRow = ByteArray(w); val pRow = ByteArray(w); val nRow = ByteArray(w); val mRow = FloatArray(w)
            val sRow0 = ByteArray(w)
            for (y in 0 until h) {
                labels.get(y, 0, lab); valid.get(y, 0, vRow); penSeed.get(y, 0, pRow); neutSeed.get(y, 0, nRow)
                penSide.get(y, 0, sRow0)
                modelHw?.get(y, 0, mRow)
                for (x in 0 until w) {
                    val l = lab[x]; if (l == 0) continue
                    if (sRow0[x].toInt() != 0) cntNear[l]++
                    if (vRow[x].toInt() != 0) cntV[l]++
                    if (pRow[x].toInt() != 0) cntP[l]++
                    if (nRow[x].toInt() != 0) cntN[l]++
                    if (modelHw != null) sumMh[l] += mRow[x]
                }
            }
            // 0 keep, 1 pen, 2 print, 3 mixed, 4 big (only its pen pieces are handwriting)
            val cat = ByteArray(n)
            val fills = FloatArray(n); val ps = FloatArray(n)
            for (l in 1 until n) {
                if (cntV[l] < 10) {
                    // too small to vote on (a printed colon, comma, dot): print when every
                    // measurable pixel is print-coloured
                    if (cntV[l] >= 3 && cntN[l] >= cntV[l] && st[l * 5 + Imgproc.CC_STAT_AREA] <= 4 * k * k) cat[l] = 2
                    continue
                }
                val cw = st[l * 5 + Imgproc.CC_STAT_WIDTH]; val ch = st[l * 5 + Imgproc.CC_STAT_HEIGHT]
                val area = st[l * 5 + Imgproc.CC_STAT_AREA]
                val p = cntP[l].toFloat() / cntV[l]
                val big = cw > 12 * k && ch > 12 * k
                val fill = area.toFloat() / max(1, cw * ch)
                val thinRule = ch <= 2 * k + 1 && cw >= 6 * ch
                val mh = sumMh[l] / max(1, area)
                val neutShare = cntN[l].toFloat() / cntV[l]
                val near = cntNear[l].toFloat() / max(1, area)
                fills[l] = fill; ps[l] = p
                cat[l] = when {
                    // large and neutral, or large and solid (handwriting strokes are sparse): a picture
                    // or a printed frame — kept, except pen pieces written into it (a signature
                    // touching the header box)
                    big && (p < 0.5f || fill > 0.3f) -> 4
                    // whole-component pen only when nothing in it is confidently print-coloured: a
                    // bold printed hint letter fused with the pen letters beside it is clipped
                    // (colourless) but for its rims, which are print-coloured
                    p >= 0.9f && cntN[l] <= max(3f, 0.02f * cntV[l]) -> 1
                    // ...or its few neutral pixels are rims scattered along the pen (low resolution
                    // and JPEG soften a thin stroke's colour: "Sai: 10" in red, 6% neutral, 88% near)
                    p > 0.05f && neutShare <= 0.1f && near >= 0.85f -> 1
                    // ...or the colour is too weak to decide but the model is sure: a dark pen at low
                    // resolution (tick 0.90, underline 0.90; bold print fused with pen ≤ 0.56)
                    p > 0.05f && mh >= 0.8 && neutShare <= 0.4f -> 1
                    // neutral: print when a rule, when the model sees no handwriting in it, or when it
                    // is toner-dark — clipped almost everywhere with every measurable rim pixel
                    // print-coloured (a bold printed letter: 2% measurable, all neutral; a dark pen
                    // stroke clipped as hard still shows some hue at its rims)
                    p <= 0.05f && (thinRule || mh < 0.35 ||
                        (cntV[l] < CLIPPED_SHARE * area && cntN[l] >= 0.9f * cntV[l])) -> 2
                    p <= 0.05f -> 0
                    else -> 3
                }
            }
            // inside a picture (big and dense — not a sparse table or box frame): its small parts that
            // are not pen-coloured (eyes, cheeks) are the picture, whatever the model thinks
            val pictures = (1 until n).filter { cat[it].toInt() == 4 && fills[it] >= 0.25f }
            for (b in pictures) {
                val bx = st[b * 5 + Imgproc.CC_STAT_LEFT]; val by = st[b * 5 + Imgproc.CC_STAT_TOP]
                val bw = st[b * 5 + Imgproc.CC_STAT_WIDTH]; val bh = st[b * 5 + Imgproc.CC_STAT_HEIGHT]
                for (l in 1 until n) {
                    if (l == b) continue
                    val c = cat[l].toInt(); if (c != 0 && c != 3) continue
                    val lx = st[l * 5 + Imgproc.CC_STAT_LEFT]; val ly = st[l * 5 + Imgproc.CC_STAT_TOP]
                    if (lx < bx || ly < by || lx + st[l * 5 + Imgproc.CC_STAT_WIDTH] > bx + bw ||
                        ly + st[l * 5 + Imgproc.CC_STAT_HEIGHT] > by + bh) continue
                    if (cntV[l] == 0 || ps[l] < 0.5f) cat[l] = 2
                }
            }
            // pieces: a pen-touched component with its printed rules taken out falls apart into the
            // strokes written on the rule (letters, a tick); each piece is judged by its own colour,
            // whole — per-pixel nearest colour is noisy on a dark pen clipped all but its rims, and
            // the upstream "print under the pen" guess does not apply to a piece that is all pen
            val nPieces: Int
            val penPiece: BooleanArray; val neutralPiece: BooleanArray; val halfPenPiece: BooleanArray
            run {
                val inPen = Mat(h, w, CvType.CV_8U); val row = ByteArray(w)
                for (y in 0 until h) {
                    labels.get(y, 0, lab)
                    for (x in 0 until w) { val c = cat[lab[x]].toInt(); row[x] = if (c == 1 || c == 3 || c == 4) 255.toByte() else 0 }
                    inPen.put(y, 0, row)
                }
                val t = Mat()
                Core.bitwise_and(inPen, ink, inPen)
                Core.bitwise_not(rule, t); Core.bitwise_and(inPen, t, inPen); t.release()
                nPieces = Imgproc.connectedComponents(inPen, pieces, 8, CvType.CV_32S)
                inPen.release()
                val pv = IntArray(nPieces); val pp = IntArray(nPieces); val pn = IntArray(nPieces)
                val area = IntArray(nPieces); val near = IntArray(nPieces); val sRow = ByteArray(w)
                for (y in 0 until h) {
                    pieces.get(y, 0, lab); valid.get(y, 0, vRow); penSeed.get(y, 0, pRow); neutSeed.get(y, 0, nRow)
                    penSide.get(y, 0, sRow)
                    for (x in 0 until w) {
                        val l = lab[x]; if (l == 0) continue
                        area[l]++
                        if (sRow[x].toInt() != 0) near[l]++
                        if (vRow[x].toInt() != 0) pv[l]++
                        if (pRow[x].toInt() != 0) pp[l]++
                        if (nRow[x].toInt() != 0) pn[l]++
                    }
                }
                // pen: mostly pen-coloured, and either (almost) no print-coloured pixel at all, or its
                // few neutral pixels are rims scattered along the pen (≥ 90% of the piece is nearer
                // pen colour) — a printed word fused with a pen underline has its neutral pixels
                // in whole letters (measured 71%), a dark pen's letters 96-100%
                penPiece = BooleanArray(nPieces) {
                    it > 0 && pv[it] >= 5 && pp[it] >= 0.6f * pv[it] &&
                        (pn[it] <= 3 || (near[it] >= 0.9f * area[it] && pn[it] <= 0.1f * pv[it]))
                }
                neutralPiece = BooleanArray(nPieces) { it > 0 && pv[it] >= 5 && pn[it] >= 0.6f * pv[it] && pp[it] <= 0.1f * pv[it] }
                halfPenPiece = BooleanArray(nPieces) { it > 0 && pv[it] >= 5 && pp[it] >= 0.5f * pv[it] }
            }
            val hRow = ByteArray(w); val prRow = ByteArray(w); val oRow = ByteArray(w)
            val sideRow = ByteArray(w); val ruleRow = ByteArray(w); val pieceRow = IntArray(w); val underRow = ByteArray(w)
            val on = 255.toByte(); val off: Byte = 0
            for (y in 0 until h) {
                labels.get(y, 0, lab); pieces.get(y, 0, pieceRow)
                hw.get(y, 0, hRow); print.get(y, 0, prRow); overlap.get(y, 0, oRow)
                penSide.get(y, 0, sideRow); rule.get(y, 0, ruleRow); ruleUnder.get(y, 0, underRow)
                var changed = false
                for (x in 0 until w) {
                    val c = cat[lab[x]].toInt(); if (c == 0) continue
                    changed = true
                    val isRule = ruleRow[x].toInt() != 0
                    val piece = pieceRow[x]
                    when {
                        // a printed rule under the pen: erased with it, restored as print
                        c != 2 && underRow[x].toInt() != 0 -> { hRow[x] = on; prRow[x] = on; oRow[x] = on }
                        penPiece[piece] -> { hRow[x] = on; prRow[x] = off; oRow[x] = off }
                        // big: a mostly-pen piece still fused with the frame goes pixel by pixel,
                        // the rest of the picture/frame is print
                        c == 4 && halfPenPiece[piece] && sideRow[x].toInt() != 0 -> { hRow[x] = on; prRow[x] = off; oRow[x] = off }
                        c == 4 -> { hRow[x] = off; prRow[x] = on; oRow[x] = off }
                        c == 3 && neutralPiece[piece] -> { hRow[x] = off; prRow[x] = on }
                        c == 2 || (c == 1 && isRule) -> { hRow[x] = off; prRow[x] = on; oRow[x] = off }
                        c == 1 -> { hRow[x] = on; prRow[x] = off; oRow[x] = off }
                        sideRow[x].toInt() != 0 -> { hRow[x] = on; if (oRow[x].toInt() == 0) prRow[x] = off }
                        else -> { hRow[x] = off; prRow[x] = on }
                    }
                }
                if (changed) { hw.put(y, 0, hRow); print.put(y, 0, prRow); overlap.put(y, 0, oRow) }
            }
            // faint marks inside a picture (pink cheeks below the ink threshold) are the picture too
            for (b in pictures) {
                val bx = st[b * 5 + Imgproc.CC_STAT_LEFT]; val by = st[b * 5 + Imgproc.CC_STAT_TOP]
                val bw = st[b * 5 + Imgproc.CC_STAT_WIDTH]; val bh = st[b * 5 + Imgproc.CC_STAT_HEIGHT]
                val bl = IntArray(bw); val bhRow = ByteArray(bw)
                for (y in by until by + bh) {
                    labels.get(y, bx, bl); hw.get(y, bx, bhRow)
                    var changed = false
                    for (x in 0 until bw) if (bl[x] == 0 && bhRow[x].toInt() != 0) { bhRow[x] = off; changed = true }
                    if (changed) hw.put(y, bx, bhRow)
                }
            }
        } finally {
            releasing(labels, stats, pieces, penSeed, neutSeed, penSide, rule, ruleUnder) {}
        }
    }
}
