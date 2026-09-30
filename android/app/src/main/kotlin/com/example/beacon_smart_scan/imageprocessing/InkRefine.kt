package com.example.beacon_smart_scan.imageprocessing

import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Full-resolution pixel refinement and erase — Kotlin port of the "Pixel-level refinement" and
 * "Erase" sections of ios/Runner/InkAnalysis.hpp (see there for the measurements behind them).
 * Everything here works on OpenCV Mats (native memory), region by region, so full-resolution
 * photos do not touch the Java heap.
 */
object InkRefine {
    private const val INK_OD = 0.30
    private const val CLIPPED_OD = 1.6
    private const val RIM_OD = 0.04
    private const val DIR_BAND = 3
    private val DIR_LENGTHS = intArrayOf(2, 4, 6)
    private const val OVERLAP_DARKER = 1.35
    private val PERPENDICULAR = intArrayOf(1, 0, 3, 2)

    private fun odd(v: Double): Int = max(1, v.roundToInt()) or 1

    /**
     * The Mat helpers below return fresh Mats for readability; their native memory is freed by
     * the Mat finalizer, which the JVM only runs under Java-heap pressure. Full-resolution
     * temporaries are native-heavy but Java-light, so collect explicitly after each region.
     */
    @Suppress("DEPRECATION")
    private fun collectNative() {
        System.gc()
        System.runFinalization()
    }

    private fun ellipse(d: Double): Mat {
        val s = odd(d).toDouble()
        return Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(s, s))
    }

    /** Mat ops with an owned result, so call sites read like the C++ expressions. */
    private fun and(a: Mat, b: Mat): Mat = Mat().also { Core.bitwise_and(a, b, it) }
    private fun or(a: Mat, b: Mat): Mat = Mat().also { Core.bitwise_or(a, b, it) }
    private fun not(a: Mat): Mat = Mat().also { Core.bitwise_not(a, it) }
    private fun cmp(a: Mat, v: Double, op: Int): Mat = Mat().also { Core.compare(a, Scalar(v), it, op) }
    private fun cmp(a: Mat, b: Mat, op: Int): Mat = Mat().also { Core.compare(a, b, it, op) }
    private fun dilate(a: Mat, kernel: Mat): Mat = Mat().also { Imgproc.dilate(a, it, kernel); kernel.release() }
    /** Connected components of [mask]: flat labels, component count, stats (8-connected). */
    private class Comps(val lab: IntArray, val n: Int, val stats: Mat) {
        fun area(i: Int) = stats.get(i, Imgproc.CC_STAT_AREA)[0]
        fun width(i: Int) = stats.get(i, Imgproc.CC_STAT_WIDTH)[0]
        fun release() = stats.release()
    }
    private fun components(mask: Mat): Comps {
        val labels = Mat(); val stats = Mat(); val centroids = Mat()
        val n = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids, 8, CvType.CV_32S)
        val lab = IntArray(labels.total().toInt()); labels.get(0, 0, lab)
        labels.release(); centroids.release()
        return Comps(lab, n, stats)
    }
    /** Share of each component's pixels that are set in [other]. */
    private fun fractionPer(c: Comps, other: Mat): FloatArray {
        val o = ByteArray(other.total().toInt()); other.get(0, 0, o)
        val hit = IntArray(c.n); val all = IntArray(c.n)
        for (i in c.lab.indices) { val l = c.lab[i]; if (l > 0) { all[l]++; if (o[i].toInt() != 0) hit[l]++ } }
        return FloatArray(c.n) { if (all[it] == 0) 0f else hit[it].toFloat() / all[it] }
    }
    private fun paint(c: Comps, keep: BooleanArray, like: Mat): Mat {
        val out = ByteArray(c.lab.size) { if (keep[c.lab[it]]) 255.toByte() else 0 }
        return Mat(like.size(), CvType.CV_8U).also { it.put(0, 0, out) }
    }

    private fun toFloat(mask: Mat): Mat = Mat().also { mask.convertTo(it, CvType.CV_32F, 1.0 / 255.0) }
    private fun filter(src: Mat, kernel: Mat): Mat =
        Mat().also { Imgproc.filter2D(src, it, CvType.CV_32F, kernel, Point(-1.0, -1.0), 0.0, Core.BORDER_CONSTANT) }

    /** 4 directional band kernels; `side` 0 = both sides of the centre, -1/+1 = one side only. */
    private fun directionKernels(length: Int, band: Int, side: Int = 0): List<Mat> {
        val c = length / 2
        val hb = band / 2
        return (0 until 4).map { dir ->
            val values = FloatArray(length * length)
            for (i in 0 until length) {
                for (o in -hb..hb) {
                    val (y, x) = when (dir) {
                        0 -> (c + o) to i
                        1 -> i to (c + o)
                        2 -> i to (i + o)
                        else -> i to (length - 1 - i + o)
                    }
                    if (x < 0 || y < 0 || x >= length || y >= length) continue
                    val along = if (dir == 0) x else y
                    if (abs(y - c) <= hb && abs(x - c) <= hb) continue
                    if (side < 0 && along >= c) continue
                    if (side > 0 && along <= c) continue
                    values[y * length + x] = 1f
                }
            }
            Mat(length, length, CvType.CV_32F).also { it.put(0, 0, values) }
        }
    }

    class Density(val odB: Mat, val odG: Mat, val odR: Mat, val mean: Mat, val ink: Mat) {
        fun release() { odB.release(); odG.release(); odR.release(); mean.release(); ink.release() }
    }

    /** Paper colour at 1/8 scale (brightest nearby — ink only darkens it). */
    fun paperLowRes(bgr: Mat): Mat {
        val small = Mat()
        Imgproc.resize(bgr, small, Size(), 0.125, 0.125, Imgproc.INTER_AREA)
        Imgproc.medianBlur(small, small, 15)
        val k = Mat.ones(9, 9, CvType.CV_8U)
        releasing(k) { Imgproc.dilate(small, small, k) }
        return small
    }

    private fun scaledRect(r: Rect, fx: Double, fy: Double, maxW: Int, maxH: Int): Rect {
        val x = floor(r.x * fx).toInt().coerceIn(0, maxW - 1)
        val y = floor(r.y * fy).toInt().coerceIn(0, maxH - 1)
        val w = max(1, min(maxW, ceil((r.x + r.width) * fx).toInt()) - x)
        val h = max(1, min(maxH, ceil((r.y + r.height) * fy).toInt()) - y)
        return Rect(x, y, w, h)
    }

    fun densityInRoi(bgr: Mat, paperSmall: Mat, roi: Rect): Density {
        val fx = paperSmall.cols().toDouble() / bgr.cols()
        val fy = paperSmall.rows().toDouble() / bgr.rows()
        val m = 64
        val bx = max(0, roi.x - m); val by = max(0, roi.y - m)
        val big = Rect(bx, by, min(bgr.cols(), roi.x + roi.width + m) - bx, min(bgr.rows(), roi.y + roi.height + m) - by)
        val sr = scaledRect(big, fx, fy, paperSmall.cols(), paperSmall.rows())
        val paperBig = Mat()
        val paperSub = paperSmall.submat(sr)
        releasing(paperSub) { Imgproc.resize(paperSub, paperBig, big.size(), 0.0, 0.0, Imgproc.INTER_LINEAR) }
        paperBig.convertTo(paperBig, CvType.CV_32FC3)
        Imgproc.GaussianBlur(paperBig, paperBig, Size(0.0, 0.0), 8.0)
        val paperView = paperBig.submat(Rect(roi.x - big.x, roi.y - big.y, roi.width, roi.height))
        val paper = paperView.clone()
        paperView.release(); paperBig.release()

        val image = Mat()
        val sub = bgr.submat(roi)
        releasing(sub) { sub.convertTo(image, CvType.CV_32FC3) }
        Core.add(image, Scalar(1.0, 1.0, 1.0), image)
        Core.add(paper, Scalar(1.0, 1.0, 1.0), paper)
        val t = Mat()
        Core.divide(image, paper, t)
        image.release(); paper.release()
        Core.min(t, Scalar(1.0, 1.0, 1.0), t)
        Core.max(t, Scalar(1e-3, 1e-3, 1e-3), t)
        Core.log(t, t)
        Core.multiply(t, Scalar(-1.0, -1.0, -1.0), t)
        val ch = ArrayList<Mat>()
        Core.split(t, ch)
        t.release()
        val mean = Mat()
        Core.add(ch[0], ch[1], mean)
        Core.add(mean, ch[2], mean)
        Core.multiply(mean, Scalar(1.0 / 3.0), mean)
        val ink = cmp(mean, INK_OD, Core.CMP_GT)
        return Density(ch[0], ch[1], ch[2], mean, ink)
    }

    /** Padded bounding boxes of `mask`'s regions, found on a 1/8 copy and merged where they overlap. */
    fun regionsOf(mask: Mat, pad: Int): List<Rect> {
        val f = 0.125
        val small = Mat()
        Imgproc.resize(mask, small, Size(), f, f, Imgproc.INTER_AREA)
        Imgproc.threshold(small, small, 0.0, 255.0, Imgproc.THRESH_BINARY)
        val p = max(1, ceil(pad * f).toInt())
        val k = Mat.ones(2 * p + 1, 2 * p + 1, CvType.CV_8U)
        val labels = Mat(); val stats = Mat(); val centroids = Mat()
        val boxes = ArrayList<Rect>()
        releasing(small, k, labels, stats, centroids) {
            Imgproc.dilate(small, small, k)
            val n = Imgproc.connectedComponentsWithStats(small, labels, stats, centroids, 8, CvType.CV_32S)
            for (i in 1 until n) {
                val x = (stats.get(i, Imgproc.CC_STAT_LEFT)[0] / f).toInt() - 8
                val y = (stats.get(i, Imgproc.CC_STAT_TOP)[0] / f).toInt() - 8
                val w = (stats.get(i, Imgproc.CC_STAT_WIDTH)[0] / f).toInt() + 16
                val h = (stats.get(i, Imgproc.CC_STAT_HEIGHT)[0] / f).toInt() + 16
                val x0 = max(0, x); val y0 = max(0, y)
                val x1 = min(mask.cols(), x + w); val y1 = min(mask.rows(), y + h)
                if (x1 > x0 && y1 > y0) boxes.add(Rect(x0, y0, x1 - x0, y1 - y0))
            }
        }
        var merged = true
        while (merged) {
            merged = false
            loop@ for (a in boxes.indices) for (b in a + 1 until boxes.size) {
                val ra = boxes[a]; val rb = boxes[b]
                if (ra.x < rb.x + rb.width && rb.x < ra.x + ra.width && ra.y < rb.y + rb.height && rb.y < ra.y + ra.height) {
                    val x0 = min(ra.x, rb.x); val y0 = min(ra.y, rb.y)
                    boxes[a] = Rect(x0, y0, max(ra.x + ra.width, rb.x + rb.width) - x0, max(ra.y + ra.height, rb.y + rb.height) - y0)
                    boxes.removeAt(b)
                    merged = true
                    break@loop
                }
            }
        }
        return boxes
    }

    private fun median(mat: Mat, mask: Mat, fallback: Float): Float {
        val v = FloatArray(mat.total().toInt()); mat.get(0, 0, v)
        val m = ByteArray(mask.total().toInt()); mask.get(0, 0, m)
        val picked = ArrayList<Float>()
        for (i in v.indices) if (m[i].toInt() != 0 && !v[i].isNaN()) picked.add(v[i])
        if (picked.size <= 50) return fallback
        picked.sort()
        return picked[picked.size / 2]
    }

    /** Refines one region: returns (handwriting, overlap) masks at full resolution. */
    private fun refineRoi(d: Density, coarse: Mat, fragments: Mat, mixed: Mat, printBand: Mat, printRefChroma: Mat, penDir: FloatArray, k: Int): Triple<Mat, Mat, Mat> {
        val zone = dilate(coarse, ellipse(4.0 * k + 1))

        // own colour: 3x3 window chroma over ink in (INK_OD, CLIPPED_OD], scored along the page's
        // pen hue against the local print chroma (see InkAnalysis.penScore)
        val inRange = and(d.ink, cmp(d.mean, CLIPPED_OD, Core.CMP_LE))
        val w = toFloat(inRange)
        val sw = Mat(); Imgproc.boxFilter(w, sw, -1, Size(3.0, 3.0))
        val sums = listOf(d.odB, d.odG, d.odR).map { od ->
            val t = Mat(); Core.multiply(od, w, t)
            val b = Mat(); Imgproc.boxFilter(t, b, -1, Size(3.0, 3.0)); t.release(); b
        }
        val total = Mat(); Core.add(sums[0], sums[1], total); Core.add(total, sums[2], total)
        val decided = and(cmp(sw, 0.3, Core.CMP_GT), cmp(total, 1e-3, Core.CMP_GT))
        Core.max(total, Scalar(1e-6), total)
        val refPlanes = ArrayList<Mat>(); Core.split(printRefChroma, refPlanes)
        val proj = Mat.zeros(total.size(), CvType.CV_32F)
        for (ch in 0 until 3) {
            val cc = Mat(); Core.divide(sums[ch], total, cc)
            Core.subtract(cc, refPlanes[ch], cc)
            Core.scaleAdd(cc, penDir[ch].toDouble(), proj, proj)
            cc.release()
        }
        val ratio = Mat()
        Core.multiply(proj, Scalar(-InkAnalysis.PEN_SCALE.toDouble()), ratio)
        Core.add(ratio, Scalar(1.0), ratio)
        val printRef = Mat(ratio.size(), CvType.CV_32F, Scalar(1.0))
        (sums + refPlanes).forEach { it.release() }
        releasing(inRange, w, sw, total, proj) {}

        val core = Mat()
        Imgproc.erode(coarse, core, Mat.ones(3, 3, CvType.CV_8U))
        val coreInk = and(and(core, d.ink), decided)
        val refMedian = Core.mean(printRef).`val`[0].toFloat()
        val penRef = median(ratio, coreInk, refMedian - 0.1f)
        core.release(); coreInk.release()

        // threshold = midpoint between the pen colour and the local print colour
        val thr = Mat()
        Core.add(printRef, Scalar(penRef.toDouble()), thr)
        Core.multiply(thr, Scalar(0.5), thr)
        val inkDecided = and(d.ink, decided)
        val ownPen = and(inkDecided, cmp(ratio, thr, Core.CMP_LT))
        val ownPrint = and(inkDecided, cmp(ratio, thr, Core.CMP_GE))
        val undecided = and(d.ink, not(decided))
        // confidently print-coloured: as close to the local print colour as print itself is
        val strongThr = Mat(); Core.subtract(printRef, Scalar(0.02), strongThr)
        var printStrong = and(inkDecided, cmp(ratio, strongThr, Core.CMP_GE))
        // (the print colour measured on the pixel itself, before the clipped blobs join it: the
        // wide-context vote below never takes a letter that shows it)
        val printByColour = printStrong.clone()
        // sensor-clipped cores (no colour of their own) are judged as whole blobs: a blob reaching
        // well outside the coarse handwriting is a print stroke's body the pen merely touches
        run {
            val labels = Mat(); val stats = Mat(); val centroids = Mat()
            val n = Imgproc.connectedComponentsWithStats(undecided, labels, stats, centroids, 8, CvType.CV_32S)
            val lab = IntArray(labels.total().toInt()); labels.get(0, 0, lab)
            val co = ByteArray(coarse.total().toInt()); coarse.get(0, 0, co)
            val inside = IntArray(n)
            for (i in lab.indices) if (lab[i] > 0 && co[i].toInt() != 0) inside[lab[i]]++
            val blobPrint = BooleanArray(n) { it > 0 && (1.0 - inside[it].toDouble() / max(1.0, stats.get(it, Imgproc.CC_STAT_AREA)[0])) >= 0.3 }
            val out = ByteArray(lab.size) { if (blobPrint[lab[it]]) 255.toByte() else 0 }
            val blobMat = Mat(labels.size(), CvType.CV_8U); blobMat.put(0, 0, out)
            printStrong = or(printStrong, blobMat)
            releasing(labels, stats, centroids, blobMat) {}
        }
        releasing(thr, strongThr, inkDecided, decided, ratio) {}

        // 4-direction arbitration on 3 px bands at several lengths. "Print runs through" counts
        // CONFIDENT print only — pen pixels that merely look slightly print-coloured on a small scan
        // must not turn a hand-drawn underline into an overlap
        val inkF = toFloat(d.ink); val penF = toFloat(ownPen); val printF = toFloat(printStrong)
        var printThrough = Mat.zeros(d.ink.size(), CvType.CV_8U)
        var penThrough = Mat.zeros(d.ink.size(), CvType.CV_8U)
        val penVotes = Mat.zeros(d.ink.size(), CvType.CV_8U)
        for ((li, m) in DIR_LENGTHS.withIndex()) {
            val length = (m * k) or 1
            for (kernel in directionKernels(length, DIR_BAND)) {
                val full = Core.sumElems(kernel).`val`[0]
                val n = filter(inkF, kernel)
                val nn = Mat(); Core.max(n, Scalar(1.0), nn)
                val pen = filter(penF, kernel); Core.divide(pen, nn, pen)
                val pr = filter(printF, kernel); Core.divide(pr, nn, pr)
                val runs = cmp(n, 0.6 * full / DIR_BAND, Core.CMP_GE)
                printThrough = or(printThrough, and(cmp(pr, 0.6, Core.CMP_GE), runs))
                penThrough = or(penThrough, and(cmp(pen, 0.6, Core.CMP_GE), runs))
                if (li == 0) {
                    val vote = cmp(pen, 0.5, Core.CMP_GE)
                    Core.divide(vote, Scalar(255.0), vote)
                    Core.add(penVotes, vote, penVotes)
                    vote.release()
                }
                releasing(kernel, n, nn, pen, pr, runs) {}
            }
        }
        val penMost = cmp(penVotes, 3.0, Core.CMP_GE)
        // the working-size (coarse) result is the base; pixels leave it only where a confident
        // print stroke runs through and pen does not dominate; carved pixels are never re-added
        var carve = and(and(printThrough, not(penMost)), or(printStrong, and(undecided, not(penThrough))))
        // pieces placed by context (fragments) and small detached pieces of the coarse handwriting
        // are never carved back out on their own colour
        run {
            val labels = Mat(); val stats = Mat(); val centroids = Mat()
            val n = Imgproc.connectedComponentsWithStats(coarse, labels, stats, centroids, 8, CvType.CV_32S)
            val lab = IntArray(labels.total().toInt()); labels.get(0, 0, lab)
            val small = BooleanArray(n) { it > 0 && stats.get(it, Imgproc.CC_STAT_AREA)[0] <= (4.0 * k) * (4.0 * k) }
            val out = ByteArray(lab.size) { if (small[lab[it]]) 255.toByte() else 0 }
            val smallMat = Mat(labels.size(), CvType.CV_8U); smallMat.put(0, 0, out)
            carve = and(carve, not(or(smallMat, fragments)))
            releasing(labels, stats, centroids, smallMat) {}
            // a context-placed piece that is mostly print-coloured itself is a printed letter the
            // pen touched (e.g. "Th" next to a circled number): its non-pen pixels are print
            val fragInk = and(fragments, d.ink)
            val cf = components(fragInk)
            val pc = fractionPer(cf, printStrong)
            val piece = paint(cf, BooleanArray(cf.n) { it > 0 && pc[it] >= 0.5f }, fragInk)
            carve = or(carve, and(piece, not(ownPen)))
            releasing(fragInk, piece) {}; cf.release()
        }
        val added = or(and(or(ownPen, undecided), penThrough), and(and(ownPrint, penMost), not(printThrough)))
        var hw = and(and(and(zone, d.ink), or(and(coarse, d.ink), added)), not(carve))
        // wide-context vote (≈6 stroke widths) — see ios InkAnalysis.hpp
        run {
            val w = (6 * k + 1).toDouble()
            val coarseNear = dilate(coarse, ellipse(2.0 * k + 1))
            val printFar = and(printStrong, not(coarseNear))
            val inkF2 = toFloat(d.ink); val farF = toFloat(printFar)
            val inkSum = Mat(); val farSum = Mat()
            Imgproc.boxFilter(inkF2, inkSum, -1, Size(w, w), Point(-1.0, -1.0), false)
            Imgproc.boxFilter(farF, farSum, -1, Size(w, w), Point(-1.0, -1.0), false)
            Core.max(inkSum, Scalar(1.0), inkSum)
            val farShare = Mat(); Core.divide(farSum, inkSum, farShare)
            // (never a letter that shows the print's own colour itself: a printed word the pen passes
            // just below — "của" over "1600:2" — is surrounded by pen, yet its measured colour is
            // print's. Only groups of such pixels, a letter's part, not a lone pixel on a pen stroke)
            val byColour = and(printByColour, not(coarse))
            val cp = components(byColour)
            val printOwn = paint(cp, BooleanArray(cp.n) { it > 0 && cp.area(it) >= k }, byColour)
            cp.release()
            repeat(2) {
                val hwF = toFloat(and(hw, d.ink))
                val hwSum = Mat(); Imgproc.boxFilter(hwF, hwSum, -1, Size(w, w), Point(-1.0, -1.0), false)
                val share = Mat(); Core.divide(hwSum, inkSum, share)
                val vote = and(and(and(and(and(zone, d.ink), cmp(share, 0.55, Core.CMP_GE)), cmp(farShare, 0.15, Core.CMP_LT)),
                    cmp(hwSum, 6.0 * k * k, Core.CMP_GE)), not(printOwn))
                hw = or(hw, vote)
                releasing(hwF, hwSum, share, vote) {}
            }
            printStrong = and(printStrong, not(hw))
            releasing(coarseNear, printFar, inkF2, farF, inkSum, farSum, farShare, byColour, printOwn, printByColour) {}
        }
        // pen stroke merged into printed letters: pen-coloured pixels that run along a pen stroke
        run {
            val mixedZone = dilate(mixed, ellipse(2.0 * k + 1))
            hw = or(hw, and(and(and(mixedZone, ownPen), penThrough), not(printThrough)))
            mixedZone.release()
        }

        // overlap: print runs through AND much darker than the pen around it
        val win = (4 * k + 1).toDouble()
        val odPen = Mat(); Core.multiply(d.mean, penF, odPen)
        val sumOD = Mat(); val sumW = Mat()
        Imgproc.boxFilter(odPen, sumOD, -1, Size(win, win), Point(-1.0, -1.0), false)
        Imgproc.boxFilter(penF, sumW, -1, Size(win, win), Point(-1.0, -1.0), false)
        Core.max(sumW, Scalar(1e-6), sumW)
        val penLocal = Mat(); Core.divide(sumOD, sumW, penLocal)
        Core.multiply(penLocal, Scalar(OVERLAP_DARKER), penLocal)
        // overlap: much darker than the pen around it AND print continues on both sides beyond
        // the pen in some direction (a hand-drawn underline has pen, not print, at both ends)
        val printVisibleF = toFloat(and(printStrong, not(hw)))
        var both = Mat.zeros(hw.size(), CvType.CV_8U)
        val bothA = directionKernels(4 * k + 1, DIR_BAND, -1)
        val bothB = directionKernels(4 * k + 1, DIR_BAND, +1)
        for (dir in 0 until 4) {
            both = or(both, and(cmp(filter(printVisibleF, bothA[dir]), 1.5, Core.CMP_GE), cmp(filter(printVisibleF, bothB[dir]), 1.5, Core.CMP_GE)))
        }
        var overlap = and(and(hw, both), cmp(d.mean, penLocal, Core.CMP_GT))
        (bothA + bothB).forEach { it.release() }
        releasing(odPen, sumOD, sumW, penLocal, printVisibleF, both) {}

        // print hidden under a pen stroke that crosses it: print on BOTH sides, perpendicular to
        // the local pen direction
        run {
            val hwF = toFloat(hw)
            val printVis = and(printStrong, not(hw))
            val printVisF = toFloat(printVis)
            val length = (2 * k) or 1
            val penScore = directionKernels(length, DIR_BAND).map { kernel -> filter(hwF, kernel).also { kernel.release() } }
            val reach = max(3, k)
            val sideA = directionKernels(2 * reach + 1, DIR_BAND, -1)
            val sideB = directionKernels(2 * reach + 1, DIR_BAND, +1)
            var bridge = Mat.zeros(hw.size(), CvType.CV_8U)
            for (dir in 0 until 4) {
                val both = and(cmp(filter(printVisF, sideA[dir]), 1.5, Core.CMP_GE), cmp(filter(printVisF, sideB[dir]), 1.5, Core.CMP_GE))
                val penDir = PERPENDICULAR[dir]
                var isPenDir = Mat(hw.size(), CvType.CV_8U, Scalar(255.0))
                for (o in 0 until 4) if (o != penDir) isPenDir = and(isPenDir, cmp(penScore[penDir], penScore[o], Core.CMP_GE))
                bridge = or(bridge, and(both, isPenDir))
                both.release(); isPenDir.release()
            }
            overlap = or(overlap, and(bridge, hw))
            // pen crossing PRINTED TEXT: in a print line's band, a sensor-clipped pixel has no colour
            // of its own, so it takes the colour of the NEAREST coloured rim pixel: a pen's core is
            // as dark as toner but its rims carry the pen's hue, a printed letter's rims are
            // print-coloured. Judged per pixel (pen strokes crossing a printed word fuse its letters
            // into one blob with mixed rims), guarded by the wider neighbourhood and a speck filter
            // — see ios InkAnalysis.hpp
            run {
                // clipped = ALL channels dark: a saturated pen (deep red) is dark on average but has
                // colour of its own and must not borrow a print rim's
                val minOD = Mat(); Core.min(d.odB, d.odG, minOD); Core.min(minOD, d.odR, minOD)
                val clipped = and(and(d.ink, cmp(d.mean, CLIPPED_OD, Core.CMP_GT)), cmp(minOD, CLIPPED_OD, Core.CMP_GT))
                minOD.release()
                val nearClipped = dilate(clipped, Mat.ones(7, 7, CvType.CV_8U))
                val notClipped = not(clipped)
                val rimPen = and(and(nearClipped, notClipped), ownPen)
                var rimPrint = and(and(and(nearClipped, notClipped), or(ownPrint, printStrong)), not(ownPen))
                // a printed RULE the pen writes on says nothing about the pen's core beside it
                run {
                    val inkOnly = and(d.ink, notClipped)
                    val ruleLike = Mat()
                    Imgproc.morphologyEx(inkOnly, ruleLike, Imgproc.MORPH_OPEN, Mat.ones(1, max(5, 4 * k) or 1, CvType.CV_8U))
                    val ruleNear = dilate(ruleLike, Mat.ones(3, 1, CvType.CV_8U))
                    rimPrint = and(rimPrint, not(ruleNear))
                    releasing(inkOnly, ruleLike, ruleNear) {}
                }
                val rims = or(rimPen, rimPrint)
                val dist = Mat(); val nearest = Mat()
                Imgproc.distanceTransformWithLabels(not(rims), dist, nearest, Geometry.DIST_L2, 3, Imgproc.DIST_LABEL_PIXEL)
                val size = clipped.total().toInt()
                val rb = ByteArray(size); rims.get(0, 0, rb)
                val rpb = ByteArray(size); rimPrint.get(0, 0, rpb)
                // DIST_LABEL_PIXEL numbers the zero pixels (the rims) in scan order from 1
                var count = 0
                for (i in 0 until size) if (rb[i].toInt() != 0) count++
                val idxIsPrint = BooleanArray(count + 1)
                var idx = 0
                for (i in 0 until size) if (rb[i].toInt() != 0) { idx++; idxIsPrint[idx] = rpb[i].toInt() != 0 }
                val cl = ByteArray(size); clipped.get(0, 0, cl)
                val ni = IntArray(size); nearest.get(0, 0, ni)
                val dd = FloatArray(size); dist.get(0, 0, dd)
                val vp = FloatArray(size); val vn = FloatArray(size)
                val reach = 2f * k
                for (i in 0 until size) {
                    if (cl[i].toInt() == 0 || dd[i] > reach || ni[i] <= 0 || ni[i] > count) continue
                    if (idxIsPrint[ni[i]]) vp[i] = 1f else vn[i] = 1f
                }
                val votePrint = Mat(clipped.size(), CvType.CV_32F); votePrint.put(0, 0, vp)
                val votePen = Mat(clipped.size(), CvType.CV_32F); votePen.put(0, 0, vn)
                // light majority smoothing, so a lone rim pixel of the wrong colour does not decide
                Imgproc.boxFilter(votePrint, votePrint, -1, Size(3.0, 3.0), Point(-1.0, -1.0), false)
                Imgproc.boxFilter(votePen, votePen, -1, Size(3.0, 3.0), Point(-1.0, -1.0), false)
                // print rims must hold their own in the wider neighbourhood too
                val wideOk = run {
                    val rp = toFloat(rimPrint); val rn = toFloat(rimPen)
                    val wide = (6 * k + 1).toDouble()
                    Imgproc.boxFilter(rp, rp, -1, Size(wide, wide), Point(-1.0, -1.0), false)
                    Imgproc.boxFilter(rn, rn, -1, Size(wide, wide), Point(-1.0, -1.0), false)
                    Core.multiply(rn, Scalar(0.5), rn)
                    cmp(rp, rn, Core.CMP_GT).also { releasing(rp, rn) {} }
                }
                val voteSum = Mat(); Core.add(votePrint, votePen, voteSum)
                Core.multiply(voteSum, Scalar(0.6), voteSum)
                var tonerCore = and(and(and(clipped, cmp(votePrint, voteSum, Core.CMP_GT)), cmp(votePrint, 0.0, Core.CMP_GT)), wideOk)
                val penCore = and(clipped, cmp(votePen, votePrint, Core.CMP_GT))
                // ...and per blob: a clipped blob whose votes are mostly print is print as a whole
                run {
                    vp.fill(0f); vn.fill(0f)
                    votePrint.get(0, 0, vp); votePen.get(0, 0, vn)
                    val cc = components(clipped)
                    val pv = IntArray(cc.n); val nv = IntArray(cc.n)
                    for (i in 0 until size) { val l = cc.lab[i]; if (l > 0) { if (vp[i] > vn[i]) pv[l]++ else if (vn[i] > vp[i]) nv[l]++ } }
                    val blob = paint(cc, BooleanArray(cc.n) { it > 0 && pv[it] + nv[it] >= 3 && pv[it] > 0.7f * (pv[it] + nv[it]) }, clipped)
                    tonerCore = or(tonerCore, blob)
                    blob.release(); cc.release()
                }
                // grown into the dark, non-pen pixels around it (a letter's body between clipped parts)
                // (only as dark as the print cores themselves: a nearly neutral second pen, a shade
                // lighter than toner, would otherwise be grown into and restored as "print")
                val coreOD = run {
                    val n = Core.countNonZero(tonerCore)
                    if (n == 0) 1.0 else {
                        val mv = FloatArray(d.mean.total().toInt()); d.mean.get(0, 0, mv)
                        val tv = ByteArray(tonerCore.total().toInt()); tonerCore.get(0, 0, tv)
                        val sel = ArrayList<Float>(n)
                        for (i in tv.indices) if (tv[i].toInt() != 0) sel.add(mv[i])
                        sel.sort()
                        max(1.0, 0.85 * sel[sel.size / 2])
                    }
                }
                val growInto = and(and(and(d.ink, cmp(d.mean, coreOD, Core.CMP_GT)), not(ownPen)), not(penCore))
                repeat(2) { tonerCore = or(tonerCore, and(dilate(tonerCore, Mat.ones(3, 3, CvType.CV_8U)), growInto)) }
                // ...plus its own anti-aliased rim, or restored letters come back thin
                tonerCore = or(tonerCore, and(and(and(dilate(tonerCore, Mat.ones(3, 3, CvType.CV_8U)), d.ink), not(ownPen)), not(penCore)))
                // specks are not letters
                run {
                    val ct = components(tonerCore)
                    val big = paint(ct, BooleanArray(ct.n) { it > 0 && ct.area(it) >= k.toDouble() * k }, tonerCore)
                    tonerCore = big; ct.release()
                }
                overlap = or(overlap, and(and(and(hw, printBand), not(ownPen)), tonerCore))
                releasing(clipped, nearClipped, notClipped, rimPen, rimPrint, rims, dist, nearest, votePrint, votePen, wideOk, voteSum, tonerCore, penCore, growInto) {}
            }
            (sideA + sideB + penScore).forEach { it.release() }
            releasing(hwF, printVis, printVisF, bridge) {}
        }
        releasing(zone, ownPen, ownPrint, undecided, inkF, penF, printF, printThrough, penThrough, penVotes, penMost, carve, added) {}
        return Triple(hw, overlap, printStrong)
    }

    /**
     * Full-resolution refinement of the working-size result: returns (handwriting, print, overlap).
     * [coarse], [coarsePrint] are at full size; [refWork] is the working-size print-colour map.
     */
    /** [enclosedPrint]: print circled by a pen ring; [ringPen]: that ring (full size, from the working size). */
    fun refine(full: Mat, coarse: Mat, coarsePrint: Mat, refWork: Mat, penDir: FloatArray, fragments: Mat, mixed: Mat, printBand: Mat,
               enclosedPrint: Mat, ringPen: Mat, colourless: Mat, modelPrint: Mat?, modelHw: Mat? = null): Triple<Mat, Mat, Mat> {
        val k = InkAnalysis.strokeUnit(full.cols(), full.rows())
        val hw = Mat.zeros(full.size(), CvType.CV_8U)
        val overlap = Mat.zeros(full.size(), CvType.CV_8U)
        val print = coarsePrint.clone()
        val paperSmall = paperLowRes(full)
        // the ring's own rim is pen too, but not the rim of the print it encloses
        val ring = run {
            val enclosedNear = dilate(enclosedPrint, Mat.ones(3, 3, CvType.CV_8U))
            val grown = dilate(ringPen, Mat.ones(3, 3, CvType.CV_8U))
            and(grown, not(enclosedNear)).also { releasing(enclosedNear, grown) {} }
        }
        // print may be restored under a pen stroke only where the model also sees print (see ios
        // DetectByInkColor: 5x5 mean, or 5x5 max while the mean is not near zero)
        val printish: Mat? = modelPrint?.let { mpIn ->
            val mp = Mat(); val mx = Mat()
            Imgproc.boxFilter(mpIn, mp, -1, Size(5.0, 5.0))
            Imgproc.dilate(mpIn, mx, Mat.ones(5, 5, CvType.CV_8U))
            val out = or(cmp(mp, 0.5, Core.CMP_GE), and(cmp(mx, 0.5, Core.CMP_GE), cmp(mp, 0.2, Core.CMP_GE)))
            releasing(mp, mx) {}
            out
        }
        try {
            val zoneSeed = or(coarse, mixed)
            val rois = regionsOf(zoneSeed, 8 * k)
            zoneSeed.release()
            for (roi in rois) {
                val d = densityInRoi(full, paperSmall, roi)
                val wr = scaledRect(roi, refWork.cols().toDouble() / full.cols(), refWork.rows().toDouble() / full.rows(), refWork.cols(), refWork.rows())
                val ref = Mat()
                val refSub = refWork.submat(wr)
                releasing(refSub) { Imgproc.resize(refSub, ref, roi.size(), 0.0, 0.0, Imgproc.INTER_LINEAR) }
                val coarseRoi = coarse.submat(roi)
                val fragRoi = fragments.submat(roi)
                val mixedRoi = mixed.submat(roi)
                val bandRoi = printBand.submat(roi)
                var (roiHw, roiOverlap, roiPrintStrong) = refineRoi(d, coarseRoi, fragRoi, mixedRoi, bandRoi, ref, penDir, k)
                if (printish != null) { val pv = printish.submat(roi); roiOverlap = and(roiOverlap, pv); pv.release() }
                run {   // circled print stays print, its pen ring is handwriting and never restored
                    val encRoi = enclosedPrint.submat(roi); val ringRoi = ring.submat(roi)
                    roiHw = and(or(roiHw, and(ringRoi, d.ink)), not(encRoi))
                    roiOverlap = and(roiOverlap, not(ringRoi))
                    roiPrintStrong = or(roiPrintStrong, and(encRoi, d.ink))
                    releasing(encRoi, ringRoi) {}
                }
                bandRoi.release()
                fragRoi.release()
                val hwView = hw.submat(roi); roiHw.copyTo(hwView); hwView.release()
                val ovView = overlap.submat(roi); roiOverlap.copyTo(ovView); ovView.release()
                // inside the refined zone only CONFIDENT print is protected: pen pixels merely left
                // out of the handwriting must stay erasable (else they become permanent ghosts)
                val coarsePrintRoi = coarsePrint.submat(roi)
                // right next to the pen the dilated working-size print is the pen's own rim
                val hwNear = dilate(roiHw, ellipse(2.0 * k + 1))
                val regionPrint = or(and(or(and(coarsePrintRoi, not(hwNear)), roiPrintStrong), not(roiHw)), roiOverlap)
                coarsePrintRoi.release(); roiPrintStrong.release(); hwNear.release()
                val zone = dilate(or(coarseRoi, mixedRoi), ellipse(4.0 * k + 1))
                val prView = print.submat(roi); regionPrint.copyTo(prView, zone); prView.release()
                mixedRoi.release()
                releasing(ref, coarseRoi, roiHw, roiOverlap, regionPrint, zone) {}
                d.release()
                collectNative()
            }
            // stray pen specks sitting on printed text — see ios InkAnalysis.hpp
            run {
                val dAll = densityInRoi(full, paperSmall, Rect(0, 0, full.cols(), full.rows()))
                val w = (6 * k + 1).toDouble()
                val nF = toFloat(and(dAll.ink, not(hw))); val iF = toFloat(dAll.ink)
                val nS = Mat(); val iS = Mat()
                Imgproc.boxFilter(nF, nS, -1, Size(w, w), Point(-1.0, -1.0), false)
                Imgproc.boxFilter(iF, iS, -1, Size(w, w), Point(-1.0, -1.0), false)
                Core.max(iS, Scalar(1.0), iS)
                val ratio = Mat(); Core.divide(nS, iS, ratio)
                val printAround = cmp(ratio, 0.5, Core.CMP_GE)
                val labels = Mat(); val stats = Mat(); val centroids = Mat()
                val n = Imgproc.connectedComponentsWithStats(hw, labels, stats, centroids, 8, CvType.CV_32S)
                val lab = IntArray(labels.total().toInt()); labels.get(0, 0, lab)
                val pa = ByteArray(printAround.total().toInt()); printAround.get(0, 0, pa)
                val fg = ByteArray(fragments.total().toInt()); fragments.get(0, 0, fg)
                val on = IntArray(n); val conf = IntArray(n)
                for (i in lab.indices) if (lab[i] > 0) {
                    if (pa[i].toInt() != 0) on[lab[i]]++
                    if (fg[i].toInt() != 0) conf[lab[i]]++
                }
                // pieces confirmed by context (fragments, fused pen underline) stay
                val drop = BooleanArray(n) {
                    val a = stats.get(it, Imgproc.CC_STAT_AREA)[0]
                    it > 0 && a <= (4.0 * k) * (4.0 * k) && on[it] >= 0.5 * a && conf[it] < 0.3 * a
                }
                val out = ByteArray(lab.size) { if (drop[lab[it]]) 255.toByte() else 0 }
                val dropMask = Mat(labels.size(), CvType.CV_8U); dropMask.put(0, 0, out)
                val keep = not(dropMask)
                Core.bitwise_and(hw, keep, hw); Core.bitwise_and(overlap, keep, overlap)
                releasing(nF, iF, nS, iS, ratio, printAround, labels, stats, centroids, dropMask, keep) {}
                // colourless handwriting found by layout + shape, verified by the model; the newly
                // found pieces are handwriting through and through, rim included
                if (Core.countNonZero(colourless) > 0) {
                    val cl = dilate(colourless, Mat.ones(3, 3, CvType.CV_8U))
                    Core.bitwise_and(cl, dAll.ink, cl)
                    Core.bitwise_or(hw, cl, hw)
                    val fresh = and(cl, not(coarse))
                    Core.bitwise_and(overlap, not(fresh), overlap)
                    val freshNear = dilate(fresh, Mat.ones(5, 5, CvType.CV_8U))
                    Core.bitwise_and(print, not(freshNear), print)
                    releasing(cl, fresh, freshNear) {}
                }
                // review layer — straight pen strokes: a short, thin, SLOPED ink run the model sees as
                // handwriting is a strike-through or underline whatever colour classes it got
                if (modelHw != null) {
                    val smear = dilate(dAll.ink, Mat.ones(3, 1, CvType.CV_8U))
                    val runs = Mat()
                    Imgproc.morphologyEx(smear, runs, Imgproc.MORPH_OPEN, Mat.ones(1, max(9, 12 * k) or 1, CvType.CV_8U))
                    Core.bitwise_and(runs, dAll.ink, runs)
                    val lg = max(15, 30 * k) or 1
                    val lh = Mat(); val lv = Mat()
                    Imgproc.morphologyEx(dAll.ink, lh, Imgproc.MORPH_OPEN, Mat.ones(1, lg, CvType.CV_8U))
                    Imgproc.morphologyEx(dAll.ink, lv, Imgproc.MORPH_OPEN, Mat.ones(lg, 1, CvType.CV_8U))
                    val grid = dilate(or(lh, lv), Mat.ones(3, 3, CvType.CV_8U))
                    val noGrid = and(dAll.ink, not(grid))
                    val c1 = Mat(); val s1 = Mat(); val ce1 = Mat(); val c2 = Mat(); val s2 = Mat(); val ce2 = Mat()
                    val n1 = Imgproc.connectedComponentsWithStats(runs, c1, s1, ce1, 8, CvType.CV_32S)
                    Imgproc.connectedComponentsWithStats(noGrid, c2, s2, ce2, 8, CvType.CV_32S)
                    val l1 = IntArray(c1.total().toInt()); c1.get(0, 0, l1)
                    val l2 = IntArray(c2.total().toInt()); c2.get(0, 0, l2)
                    val mh = FloatArray(modelHw.total().toInt()); modelHw.get(0, 0, mh)
                    val sum = DoubleArray(n1); val cnt = IntArray(n1); val strokeW = IntArray(n1)
                    for (p in l1.indices) {
                        val a = l1[p]; if (a == 0) continue
                        sum[a] += mh[p]; cnt[a]++
                        if (l2[p] > 0) strokeW[a] = max(strokeW[a], s2.get(l2[p], Imgproc.CC_STAT_WIDTH)[0].toInt())
                    }
                    val strike = BooleanArray(n1) {
                        if (it == 0 || cnt[it] == 0) false else {
                            val w = s1.get(it, Imgproc.CC_STAT_WIDTH)[0]; val h = s1.get(it, Imgproc.CC_STAT_HEIGHT)[0]
                            val t = max(1.0, s1.get(it, Imgproc.CC_STAT_AREA)[0] / max(1.0, w))
                            h - t > 0.015 * w + 2 && strokeW[it] < 0.25 * runs.cols() && sum[it] / cnt[it] >= 0.4
                        }
                    }
                    val out = ByteArray(l1.size) { if (strike[l1[it]]) 255.toByte() else 0 }
                    val sm0 = Mat(runs.size(), CvType.CV_8U); sm0.put(0, 0, out)
                    val sm = and(dilate(sm0, Mat.ones(3, 3, CvType.CV_8U)), dAll.ink)
                    Core.bitwise_or(hw, sm, hw)
                    val nsm = not(sm)
                    Core.bitwise_and(overlap, nsm, overlap); Core.bitwise_and(print, nsm, print)
                    releasing(smear, runs, lh, lv, grid, noGrid, c1, s1, ce1, c2, s2, ce2, sm0, sm, nsm) {}
                }
                // leftover pieces of a pen stroke: ink neither handwriting nor print, touching the
                // handwriting, is the same stroke
                run {
                    val loose = and(and(dAll.ink, not(print)), not(hw))
                    val near = dilate(hw, Mat.ones(3, 3, CvType.CV_8U))
                    val lb = Mat(); val st = Mat(); val ce = Mat()
                    val nl = Imgproc.connectedComponentsWithStats(loose, lb, st, ce, 8, CvType.CV_32S)
                    val la = IntArray(lb.total().toInt()); lb.get(0, 0, la)
                    val nb = ByteArray(near.total().toInt()); near.get(0, 0, nb)
                    val touch = BooleanArray(nl)
                    for (p in la.indices) if (la[p] > 0 && nb[p].toInt() != 0) touch[la[p]] = true
                    val take = BooleanArray(nl) { it > 0 && touch[it] && st.get(it, Imgproc.CC_STAT_AREA)[0] <= (8.0 * k) * (8.0 * k) }
                    val out = ByteArray(la.size) { if (take[la[it]]) 255.toByte() else 0 }
                    val tm = Mat(lb.size(), CvType.CV_8U); tm.put(0, 0, out)
                    Core.bitwise_or(hw, tm, hw)
                    releasing(loose, near, lb, st, ce, tm) {}
                }
                dAll.release()
            }
        } finally {
            paperSmall.release(); ring.release(); printish?.release()
        }
        return Triple(hw, print, overlap)
    }

    /**
     * Removes [handwriting] from [target] by rebuilding: overlap pixels get the nearby print colour
     * back, the rest of the stroke, its rim and the camera halo is inpainted from the PAPER only.
     * [analysis] is the unprocessed page the masks were computed on (same geometry).
     */
    /** Final review layer — leftovers around what was erased (see ios EraseHandwriting). */
    private fun cleanupLeftovers(dst: Mat, erasedAll: Mat, restoredAll: Mat, handwriting: Mat, print: Mat, overlap: Mat, brightSmall: Mat, k: Int) {
        val t = ArrayList<Mat>()
        fun <T : Mat> own(m: T): T { t.add(m); return m }
        try {
            val zone = own(dilate(erasedAll, ellipse(3.0 * k + 1)))
            val g0 = own(Mat()); Imgproc.cvtColor(dst, g0, Imgproc.COLOR_BGR2GRAY)
            val levelRuns = own(Mat()); Imgproc.morphologyEx(own(cmp(g0, 200.0, Core.CMP_LT)), levelRuns, Imgproc.MORPH_OPEN, Mat.ones(1, max(9, 6 * k) or 1, CvType.CV_8U))
            val hwNear = own(dilate(handwriting, ellipse(2.0 * k + 1)))
            val protect = own(or(or(or(own(and(print, own(not(hwNear)))), overlap), restoredAll), levelRuns))
            val printNear = own(dilate(protect, Mat.ones(3, 3, CvType.CV_8U)))
            val bright = own(Mat()); Imgproc.resize(brightSmall, bright, dst.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            Imgproc.GaussianBlur(bright, bright, Size(0.0, 0.0), 8.0)
            val pg = own(Mat()); Imgproc.cvtColor(bright, pg, Imgproc.COLOR_BGR2GRAY)
            val diff = own(Mat()); Core.subtract(pg, g0, diff, Mat(), CvType.CV_16S)
            val darkish = own(cmp(diff, 6.0, Core.CMP_GT)); val strong = own(cmp(diff, 60.0, Core.CMP_GT))
            val cand = own(and(and(zone, darkish), own(not(printNear))))
            // letter-sized ink staying on the page; a speck within a stroke width may be its accent
            val letters = run {
                val m = own(and(strong, own(not(erasedAll))))
                val lb = own(Mat()); val st = own(Mat()); val ce = own(Mat())
                val n = Imgproc.connectedComponentsWithStats(m, lb, st, ce, 8, CvType.CV_32S)
                val la = IntArray(lb.total().toInt()); lb.get(0, 0, la)
                val out = ByteArray(la.size) { if (la[it] > 0 && st.get(la[it], Imgproc.CC_STAT_AREA)[0] > (2.0 * k) * (2.0 * k)) 255.toByte() else 0 }
                val big = own(Mat(lb.size(), CvType.CV_8U)); big.put(0, 0, out)
                own(dilate(big, ellipse(2.0 * k + 1)))
            }
            val lb = own(Mat()); val st = own(Mat()); val ce = own(Mat())
            val n = Imgproc.connectedComponentsWithStats(cand, lb, st, ce, 8, CvType.CV_32S)
            val la = IntArray(lb.total().toInt()); lb.get(0, 0, la)
            val sb = ByteArray(la.size); strong.get(0, 0, sb)
            val pb = ByteArray(la.size); print.get(0, 0, pb)
            val lt = ByteArray(la.size); letters.get(0, 0, lt)
            val nStrong = IntArray(n); val nPrint = IntArray(n); val nLetter = IntArray(n)
            for (p in la.indices) { val a = la[p]; if (a == 0) continue
                if (sb[p].toInt() != 0) nStrong[a]++; if (pb[p].toInt() != 0) nPrint[a]++; if (lt[p].toInt() != 0) nLetter[a]++ }
            val clean = BooleanArray(n) {
                if (it == 0) false else {
                    val area = st.get(it, Imgproc.CC_STAT_AREA)[0]
                    val speck = area <= (2.0 * k) * (2.0 * k)
                    nStrong[it] < 0.3 * area || (speck && nPrint[it] < 0.5 * area && nLetter[it] == 0)
                }
            }
            val out = ByteArray(la.size) { if (clean[la[it]]) 255.toByte() else 0 }
            val cm = own(Mat(lb.size(), CvType.CV_8U)); cm.put(0, 0, out)
            bright.copyTo(dst, cm)
        } finally { t.forEach { it.release() } }
    }

    /**
     * Level rules (`rule` flags over the components `lab`) as the rule ITSELF, not the ink the run
     * holds: a pen stroke written along a rule fuses with it, and putting that fused ink back as
     * print left the pen's pieces on the rule as a ragged, dashed edge. Where the pen crosses it, a
     * column between clean columns on either side gets the rule's own band, its edges interpolated
     * between them, limited to ink and to the run's own columns and rows (see RuleBands in ios
     * InkAnalysis.hpp).
     */
    private fun ruleBands(lab: IntArray, cols: Int, rows: Int, stats: Mat, n: Int, rule: BooleanArray, pen: ByteArray, inkSmear: ByteArray, k: Int): ByteArray {
        val out = ByteArray(lab.size)
        for (i in 1 until n) {
            if (!rule[i]) continue
            val x0 = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt(); val y0 = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt(); val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val top = IntArray(w) { -1 }; val bot = IntArray(w) { -1 }; val dirty = BooleanArray(w)
            for (y in y0 until y0 + h) for (x in 0 until w) {
                val p = y * cols + x0 + x
                if (lab[p] != i) continue
                if (top[x] < 0) top[x] = y
                bot[x] = y
                if (pen[p].toInt() != 0) dirty[x] = true
            }
            fun ownInk(x: Int) { for (y in top[x]..bot[x]) { val p = y * cols + x0 + x; if (lab[p] == i) out[p] = 255.toByte() } }
            val clean = (0 until w).filter { top[it] >= 0 && !dirty[it] }
            // (and only a thin one: a dark band of a photo is no rule to rebuild)
            val thick = clean.map { bot[it] - top[it] + 1 }.sorted()
            val thin = thick.isNotEmpty() && thick[thick.size / 2] <= 1.5f * k
            if (clean.size < max(5, w / 10) || !thin) { for (x in 0 until w) if (top[x] >= 0) ownInk(x); continue }
            var next = 0   // first clean column at or right of x
            for (x in 0 until w) {
                while (next < clean.size && clean[next] < x) next++
                if (top[x] < 0) continue
                val l = if (next > 0) clean[next - 1] else -1; val r = if (next < clean.size) clean[next] else -1
                // a clean column, or the pen at the run's end: its ink as it is
                if (r == x || l < 0 || r < 0) { ownInk(x); continue }
                val f = (x - l) / (r - l).toFloat()
                val t0 = top[l] + (top[r] - top[l]) * f; val t1 = bot[l] + (bot[r] - bot[l]) * f
                val ya = maxOf(0, top[x] - 1, t0.roundToInt()); val yb = minOf(rows - 1, bot[x] + 1, t1.roundToInt())
                for (y in ya..yb) { val p = y * cols + x0 + x; if (inkSmear[p].toInt() != 0) out[p] = 255.toByte() }
            }
        }
        return out
    }

    /**
     * Rules running along the rows of [ink] — the pen written ALONG them fused with them; called on
     * the page and on its transpose, so table borders the pen crossed are found as well as fill-in
     * blanks. [thinOnly]: only runs no thicker than a printed rule (for the columns: a photo's dark
     * band or a coloured page edge is long and straight too). See LevelRules in ios InkAnalysis.hpp.
     */
    private fun levelRules(ink: Mat, handwriting: Mat, k: Int, thinOnly: Boolean = false): Mat {
        val smear = dilate(ink, Mat.ones(3, 1, CvType.CV_8U))
        val runs = Mat()
        // (lengths capped by the page width: k has a floor, so on a small page 20k/40k would be
        // longer than a whole blank)
        val l2 = max(5, min(20 * k, ink.cols() / 30)) or 1
        Imgproc.morphologyEx(smear, runs, Imgproc.MORPH_OPEN, Mat.ones(1, l2, CvType.CV_8U))
        Core.bitwise_and(runs, ink, runs)
        val hwNear = dilate(handwriting, ellipse(2.0 * k + 1))
        val labels = Mat(); val stats = Mat(); val centroids = Mat()
        val n = Imgproc.connectedComponentsWithStats(runs, labels, stats, centroids, 8, CvType.CV_32S)
        val lab = IntArray(labels.total().toInt()); labels.get(0, 0, lab)
        val hn = ByteArray(hwNear.total().toInt()); hwNear.get(0, 0, hn)
        val inside = IntArray(n)
        for (i in lab.indices) if (lab[i] > 0 && hn[i].toInt() != 0) inside[lab[i]]++
        // ...or so long and straight that no hand drew it (a blank the answer fills end to end) —
        // but it must still show some of itself OUTSIDE the pen strokes: a ruler-straight pen
        // underline is handwriting end to end, a rule peeks out past the answer written on it
        val hwAll = ByteArray(handwriting.total().toInt()); handwriting.get(0, 0, hwAll)
        val inHw = IntArray(n)
        for (i in lab.indices) if (lab[i] > 0 && hwAll[i].toInt() != 0) inHw[lab[i]]++
        val ruleLen = min(40 * k, ink.cols() / 18)
        val rule = BooleanArray(n) {
            val area = stats.get(it, Imgproc.CC_STAT_AREA)[0]
            it > 0 && (inside[it] <= 0.7 * area || (stats.get(it, Imgproc.CC_STAT_WIDTH)[0] >= ruleLen && inHw[it] <= 0.9 * area))
        }
        // a printed rule is level with the page; a pen strike-through drawn with a ruler slopes
        run {
            val thick = (1 until n).filter { rule[it] }.map { max(1.0, stats.get(it, Imgproc.CC_STAT_AREA)[0] / max(1.0, stats.get(it, Imgproc.CC_STAT_WIDTH)[0])) }.sorted()
            val t = if (thick.isEmpty()) k.toDouble() else thick[thick.size / 2]
            for (i in 1 until n) {
                val area = stats.get(i, Imgproc.CC_STAT_AREA)[0]
                if (rule[i] && inside[i] > 0.7 * area &&
                    stats.get(i, Imgproc.CC_STAT_HEIGHT)[0] - t > 0.015 * stats.get(i, Imgproc.CC_STAT_WIDTH)[0] + 2) rule[i] = false
            }
        }
        if (thinOnly)
            for (i in 1 until n)
                if (rule[i] && stats.get(i, Imgproc.CC_STAT_AREA)[0] > 1.5 * k * stats.get(i, Imgproc.CC_STAT_WIDTH)[0]) rule[i] = false
        val sm = ByteArray(smear.total().toInt()); smear.get(0, 0, sm)
        val out = ruleBands(lab, labels.cols(), labels.rows(), stats, n, rule, hn, sm, k)
        val rules = Mat(labels.size(), CvType.CV_8U); rules.put(0, 0, out)
        releasing(smear, runs, hwNear, labels, stats, centroids) {}
        return rules
    }

    /** [recognizer]: reads the page's text, to restore pen-hidden print (optional). */
    fun erase(target: Mat, analysis: Mat, handwriting: Mat, print: Mat, overlap: Mat, recognizer: TextRecognizer? = null): Mat {
        val k = InkAnalysis.strokeUnit(target.cols(), target.rows())
        val s = k / 9.0
        val dst = target.clone()
        val paperSmall = paperLowRes(analysis)
        val targetPaperSmall = paperLowRes(target)
        // the brightest paper around (~15 cells): under dense handwriting the local estimate is grey
        val targetPaperSmallBright = Mat(); Imgproc.dilate(targetPaperSmall, targetPaperSmallBright, Mat.ones(15, 15, CvType.CV_8U))
        val targetSmall = Mat()
        Imgproc.resize(target, targetSmall, Size(), 0.125, 0.125, Imgproc.INTER_AREA)
        Imgproc.medianBlur(targetSmall, targetSmall, 21)
        val margin = odd(61 * s) / 2 + odd(13 * s) + odd(11 * s) + 4
        val erasedAll = Mat.zeros(target.size(), CvType.CV_8U)
        val restoredAll = Mat.zeros(target.size(), CvType.CV_8U)
        // printed rules the pen wrote ALONG (fill-in blanks), found page-wide as long, thin,
        // straight ink runs; a rule runs well beyond the handwriting on it, or is so long no hand
        // drew it (see ios InkAnalysis.hpp)
        val pageRules: Mat
        run {
            val dAll = densityInRoi(analysis, paperSmall, Rect(0, 0, analysis.cols(), analysis.rows()))
            pageRules = levelRules(dAll.ink, handwriting, k)
            // (and vertical ones — a table's column borders an answer was written across — on the
            // transposed page)
            val inkT = Mat(); Core.transpose(dAll.ink, inkT)
            val hwT = Mat(); Core.transpose(handwriting, hwT)
            val verticalT = levelRules(inkT, hwT, k, thinOnly = true)
            val vertical = Mat(); Core.transpose(verticalT, vertical)
            Core.bitwise_or(pageRules, vertical, pageRules)
            releasing(inkT, hwT, verticalT, vertical) {}
            dAll.release()
        }
        try {
            for (roi in regionsOf(handwriting, margin)) {
                val d = densityInRoi(analysis, paperSmall, roi)
                // the target is sharpened: rims faint in the analysis page are dark there
                val dt = densityInRoi(target, targetPaperSmall, roi)
                val hw = handwriting.submat(roi); val pr = print.submat(roi); val ov = overlap.submat(roi)
                val printNear = dilate(pr, Mat.ones(3, 3, CvType.CV_8U))
                val grown = dilate(hw, ellipse(13 * s))
                val rim = or(cmp(d.mean, RIM_OD, Core.CMP_GT), cmp(dt.mean, RIM_OD, Core.CMP_GT))
                var area = or(hw, and(and(grown, rim), not(printNear)))
                dt.release()
                val halo = dilate(area, ellipse(11 * s))
                area = or(area, and(halo, not(printNear)))
                var restoreMask = ov.clone()
                // printed rules (long, thin, straight print runs) crossing the erased area are
                // re-joined: close short gaps along the line, keep runs long AND thin
                run {
                    // seed from print well away from the pen (the print layer hugs pen rims)
                    val hwFar = dilate(hw, ellipse(3.0 * k + 1))
                    val prInk = and(and(pr, not(hwFar)), d.ink)
                    hwFar.release()
                    val closed = Mat(); val runs = Mat(); val thick = Mat()
                    val l1 = max(3, 6 * k) or 1; val l2 = max(5, min(20 * k, analysis.cols() / 30)) or 1
                    Imgproc.morphologyEx(prInk, closed, Imgproc.MORPH_CLOSE, Mat.ones(1, l1, CvType.CV_8U))
                    Imgproc.morphologyEx(closed, runs, Imgproc.MORPH_OPEN, Mat.ones(1, l2, CvType.CV_8U))
                    Imgproc.morphologyEx(runs, thick, Imgproc.MORPH_OPEN, Mat.ones(max(3, k) or 1, 1, CvType.CV_8U))
                    val thickNear = dilate(thick, Mat.ones(3, 3, CvType.CV_8U))
                    restoreMask = or(restoreMask, and(and(runs, not(thickNear)), area))
                    releasing(prInk, closed, runs, thick, thickNear) {}
                }
                run {
                    val pr2 = pageRules.submat(roi)
                    restoreMask = or(restoreMask, and(pr2, area))
                    pr2.release()
                }
                // leftover pen specks: small ink bits right beside the erased strokes that are not
                // confident print belong to the handwriting too
                run {
                    val nearHw = dilate(hw, ellipse(4.0 * k + 1))
                    val leftover = and(and(and(cmp(d.mean, 0.2, Core.CMP_GT), nearHw), not(pr)), not(hw))
                    val labels = Mat(); val stats = Mat(); val centroids = Mat()
                    val n = Imgproc.connectedComponentsWithStats(leftover, labels, stats, centroids, 8, CvType.CV_32S)
                    val lab = IntArray(labels.total().toInt()); labels.get(0, 0, lab)
                    val small = BooleanArray(n) { it > 0 && stats.get(it, Imgproc.CC_STAT_AREA)[0] <= (2.0 * k) * (2.0 * k) }
                    val out = ByteArray(lab.size) { if (small[lab[it]]) 255.toByte() else 0 }
                    val specks = Mat(labels.size(), CvType.CV_8U); specks.put(0, 0, out)
                    area = or(area, specks)
                    releasing(nearHw, leftover, labels, stats, centroids, specks) {}
                }
                val restore = and(area, restoreMask)
                val paperArea = and(area, not(restore))

                // local print colour: average of nearby visible print
                val strong = and(and(pr, not(hw)), d.ink)
                val weight = toFloat(strong)
                val tf = Mat()
                val tsub = target.submat(roi)
                tsub.convertTo(tf, CvType.CV_32FC3)
                val weight3 = Mat(); Core.merge(listOf(weight, weight, weight), weight3)
                val weighted = Mat(); Core.multiply(tf, weight3, weighted)
                val win = odd(61 * s).toDouble()
                val num = Mat(); val den = Mat()
                Imgproc.boxFilter(weighted, num, -1, Size(win, win))
                Imgproc.boxFilter(weight, den, -1, Size(win, win))
                val empty = cmp(den, 1e-3, Core.CMP_LE)
                Core.max(den, Scalar(1e-3), den)
                val den3 = Mat(); Core.merge(listOf(den, den, den), den3)
                val local = Mat(); Core.divide(num, den3, local)
                val fallback = if (Core.countNonZero(strong) > 0) Core.mean(tsub, strong) else Scalar(40.0, 40.0, 40.0)
                local.setTo(fallback, empty)
                local.convertTo(local, CvType.CV_8UC3)

                val out = dst.submat(roi).clone()
                local.copyTo(out, restore)
                // inpaint from paper only: swap nearby print for paper colour meanwhile
                // (the brightest-nearby paper estimate: a median is pulled grey by dense handwriting)
                val sr = scaledRect(roi, 0.125, 0.125, targetPaperSmall.cols(), targetPaperSmall.rows())
                val paperColor = Mat()
                val ps = targetPaperSmall.submat(sr)
                Imgproc.resize(ps, paperColor, roi.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
                Imgproc.GaussianBlur(paperColor, paperColor, Size(0.0, 0.0), 8.0)
                val nearArea = dilate(area, ellipse(9 * s))
                val keep = or(and(and(printNear, nearArea), not(area)), restore)
                val tmp = out.clone()
                paperColor.copyTo(tmp, keep)
                val filled = Mat()
                Photo.inpaint(tmp, paperArea, filled, 3.0, Photo.INPAINT_TELEA)
                // erased pixels are never darker than the local paper (inpainting pulls the stroke's
                // pale edge inwards: a light ghost)
                run {
                    val fg = Mat(); val pg = Mat()
                    Imgproc.cvtColor(filled, fg, Imgproc.COLOR_BGR2GRAY)
                    val bsub = targetPaperSmallBright.submat(sr); val bright = Mat()
                    Imgproc.resize(bsub, bright, roi.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
                    Imgproc.GaussianBlur(bright, bright, Size(0.0, 0.0), 8.0)
                    Imgproc.cvtColor(bright, pg, Imgproc.COLOR_BGR2GRAY)
                    Core.subtract(pg, Scalar(6.0), pg)
                    val ghost = and(paperArea, cmp(fg, pg, Core.CMP_LT))
                    bright.copyTo(filled, ghost)
                    releasing(fg, pg, ghost, bsub, bright) {}
                }
                out.copyTo(filled, keep)
                val dview = dst.submat(roi); filled.copyTo(dview); dview.release()
                val ev = erasedAll.submat(roi); Core.bitwise_or(ev, paperArea, ev); ev.release()
                val rv = restoredAll.submat(roi); Core.bitwise_or(rv, restore, rv); rv.release()
                releasing(hw, pr, ov, printNear, grown, area, halo, restore, paperArea, strong, weight, tf, tsub, weight3, weighted, num, den,
                    empty, den3, local, out, paperColor, ps, nearArea, keep, tmp, filled) {}
                d.release()
                collectNative()
            }
            cleanupLeftovers(dst, erasedAll, restoredAll, handwriting, print, overlap, targetPaperSmallBright, k)
            val beforeRepair = dst.clone()
            PrintRepair.repairByExample(dst, erasedAll, target, k)
            // printed letters the pen crossed, re-typeset from the page's own glyphs
            val printOrOverlap = Mat(); Core.bitwise_or(print, overlap, printOrOverlap)
            PrintRetypeset.retypeset(dst, analysis, handwriting, printOrOverlap, k)
            printOrOverlap.release()
            // printed letters the pen hid entirely, restored by reading them
            PrintRecognition.restore(dst, analysis, erasedAll, k, recognizer)
            // stray ink review; print put back by the repair steps is protected like restored print
            run {
                val changed = Mat(); Core.absdiff(dst, beforeRepair, changed)
                Imgproc.cvtColor(changed, changed, Imgproc.COLOR_BGR2GRAY)
                val protect = cmp(changed, 30.0, Core.CMP_GT)
                Core.bitwise_or(protect, restoredAll, protect)
                Imgproc.dilate(protect, protect, Mat.ones(3, 3, CvType.CV_8U))
                val bright = Mat(); Imgproc.resize(targetPaperSmallBright, bright, dst.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
                Imgproc.GaussianBlur(bright, bright, Size(0.0, 0.0), 8.0)
                StrayInk.remove(dst, erasedAll, handwriting, protect, bright, k)
                releasing(changed, protect, bright, beforeRepair) {}
            }
        } finally {
            paperSmall.release(); targetSmall.release(); targetPaperSmall.release(); pageRules.release(); erasedAll.release()
            restoredAll.release(); targetPaperSmallBright.release()
        }
        return dst
    }
}
