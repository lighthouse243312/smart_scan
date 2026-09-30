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
    private const val HIST_BINS = 1000

    /**
     * Updates [handwriting], [print] and [overlap] (CV_8UC1, 255 = set, [src]'s size) in place.
     * [modelHw]: the segmentation model's handwriting probability (CV_32F, [src]'s size) or null.
     */
    fun apply(src: Mat, handwriting: Mat, print: Mat, overlap: Mat, modelHw: Mat?) {
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
        val rule = Mat()
        run {
            val runs = Mat(); val thick = Mat(); val segLabels = Mat()
            Imgproc.morphologyEx(ink, runs, Imgproc.MORPH_OPEN, Mat.ones(1, 2 * k + 1, CvType.CV_8U))
            Imgproc.morphologyEx(runs, thick, Imgproc.MORPH_OPEN, Mat.ones(k or 1, 1, CvType.CV_8U))
            Imgproc.dilate(thick, thick, Mat.ones(3, 3, CvType.CV_8U))
            Core.bitwise_not(thick, thick); Core.bitwise_and(runs, thick, runs)
            val segs = Imgproc.connectedComponents(runs, segLabels, 8, CvType.CV_32S)
            val measured = IntArray(segs); val neutral = IntArray(segs)
            val lab = IntArray(w); val vRow = ByteArray(w); val nRow = ByteArray(w)
            for (y in 0 until h) {
                segLabels.get(y, 0, lab); valid.get(y, 0, vRow); neutSeed.get(y, 0, nRow)
                for (x in 0 until w) {
                    val l = lab[x]; if (l == 0) continue
                    if (vRow[x].toInt() != 0) measured[l]++
                    if (nRow[x].toInt() != 0) neutral[l]++
                }
            }
            val printed = BooleanArray(segs) { it > 0 && neutral[it] >= RULE_MIN_NEUTRAL && neutral[it] >= RULE_NEUTRAL_SHARE * measured[it] }
            val out = ByteArray(w)
            rule.create(h, w, CvType.CV_8U)
            for (y in 0 until h) {
                segLabels.get(y, 0, lab)
                for (x in 0 until w) out[x] = if (printed[lab[x]]) 255.toByte() else 0
                rule.put(y, 0, out)
            }
            Imgproc.dilate(rule, rule, Mat.ones(3, 1, CvType.CV_8U))
            Core.bitwise_and(rule, ink, rule)
            Core.bitwise_not(penSeed, tmp); Core.bitwise_and(rule, tmp, rule)
            Core.bitwise_not(rule, tmp); Core.bitwise_and(penSide, tmp, penSide)
            releasing(runs, thick, segLabels, neutSeed) {}
        }
        tmp.release()

        val labels = Mat(); val stats = Mat(); val centroids = Mat()
        val n = Imgproc.connectedComponentsWithStats(ink, labels, stats, centroids, 8, CvType.CV_32S)
        centroids.release()
        try {
            val st = IntArray(n * 5); stats.get(0, 0, st)
            val cntV = IntArray(n); val cntP = IntArray(n); val sumMh = DoubleArray(n)
            val lab = IntArray(w); val vRow = ByteArray(w); val pRow = ByteArray(w); val mRow = FloatArray(w)
            for (y in 0 until h) {
                labels.get(y, 0, lab); valid.get(y, 0, vRow); penSeed.get(y, 0, pRow)
                modelHw?.get(y, 0, mRow)
                for (x in 0 until w) {
                    val l = lab[x]; if (l == 0) continue
                    if (vRow[x].toInt() != 0) cntV[l]++
                    if (pRow[x].toInt() != 0) cntP[l]++
                    if (modelHw != null) sumMh[l] += mRow[x]
                }
            }
            // 0 keep, 1 pen, 2 print, 3 mixed
            val cat = ByteArray(n)
            for (l in 1 until n) {
                if (cntV[l] < 10) continue
                val cw = st[l * 5 + Imgproc.CC_STAT_WIDTH]; val ch = st[l * 5 + Imgproc.CC_STAT_HEIGHT]
                val area = st[l * 5 + Imgproc.CC_STAT_AREA]
                val p = cntP[l].toFloat() / cntV[l]
                val big = cw > 12 * k && ch > 12 * k
                val fill = area.toFloat() / max(1, cw * ch)
                val thinRule = ch <= 2 * k + 1 && cw >= 6 * ch
                val mh = sumMh[l] / max(1, area)
                cat[l] = when {
                    // pictures: large and neutral, or large and solid (handwriting strokes are sparse)
                    big && (p < 0.5f || fill > 0.3f) -> 2
                    p >= 0.9f -> 1
                    p <= 0.05f && (thinRule || mh < 0.35) -> 2
                    p <= 0.05f -> 0
                    else -> 3
                }
            }
            val hRow = ByteArray(w); val prRow = ByteArray(w); val oRow = ByteArray(w)
            val sideRow = ByteArray(w); val ruleRow = ByteArray(w)
            val on = 255.toByte(); val off: Byte = 0
            for (y in 0 until h) {
                labels.get(y, 0, lab)
                hw.get(y, 0, hRow); print.get(y, 0, prRow); overlap.get(y, 0, oRow)
                penSide.get(y, 0, sideRow); rule.get(y, 0, ruleRow)
                var changed = false
                for (x in 0 until w) {
                    val c = cat[lab[x]].toInt(); if (c == 0) continue
                    changed = true
                    val isRule = ruleRow[x].toInt() != 0
                    when {
                        c == 2 || (c == 1 && isRule) -> { hRow[x] = off; prRow[x] = on; oRow[x] = off }
                        c == 1 -> { hRow[x] = on; prRow[x] = off; oRow[x] = off }
                        sideRow[x].toInt() != 0 -> { hRow[x] = on; if (oRow[x].toInt() == 0) prRow[x] = off }
                        else -> { hRow[x] = off; prRow[x] = on }
                    }
                }
                if (changed) { hw.put(y, 0, hRow); print.put(y, 0, prRow); overlap.put(y, 0, oRow) }
            }
        } finally {
            releasing(labels, stats, penSeed, penSide, rule) {}
        }
    }
}
