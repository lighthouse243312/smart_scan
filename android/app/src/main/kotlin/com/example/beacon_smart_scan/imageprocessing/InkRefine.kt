package com.example.beacon_smart_scan.imageprocessing

import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
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

    class Density(val odB: Mat, val odR: Mat, val mean: Mat, val ink: Mat) {
        fun release() { odB.release(); odR.release(); mean.release(); ink.release() }
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
        ch[1].release()
        val ink = cmp(mean, INK_OD, Core.CMP_GT)
        return Density(ch[0], ch[2], mean, ink)
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
    private fun refineRoi(d: Density, coarse: Mat, fragments: Mat, mixed: Mat, printRef: Mat, k: Int): Triple<Mat, Mat, Mat> {
        val zone = dilate(coarse, ellipse(4.0 * k + 1))

        // own colour: 3x3 window over ink in (INK_OD, CLIPPED_OD]
        val inRange = and(d.ink, cmp(d.mean, CLIPPED_OD, Core.CMP_LE))
        val w = toFloat(inRange)
        val sb = Mat(); val sr = Mat(); val sw = Mat()
        val tb = Mat(); val tr = Mat()
        Core.multiply(d.odB, w, tb); Core.multiply(d.odR, w, tr)
        Imgproc.boxFilter(tb, sb, -1, Size(3.0, 3.0)); Imgproc.boxFilter(tr, sr, -1, Size(3.0, 3.0)); Imgproc.boxFilter(w, sw, -1, Size(3.0, 3.0))
        val ratio = Mat()
        Core.add(sr, Scalar(1e-3), sr)
        Core.divide(sb, sr, ratio)
        val decided = cmp(sw, 0.3, Core.CMP_GT)
        releasing(inRange, w, sb, sr, sw, tb, tr) {}

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
            repeat(2) {
                val hwF = toFloat(and(hw, d.ink))
                val hwSum = Mat(); Imgproc.boxFilter(hwF, hwSum, -1, Size(w, w), Point(-1.0, -1.0), false)
                val share = Mat(); Core.divide(hwSum, inkSum, share)
                val vote = and(and(and(and(zone, d.ink), cmp(share, 0.55, Core.CMP_GE)), cmp(farShare, 0.15, Core.CMP_LT)),
                    cmp(hwSum, 6.0 * k * k, Core.CMP_GE))
                hw = or(hw, vote)
                releasing(hwF, hwSum, share, vote) {}
            }
            printStrong = and(printStrong, not(hw))
            releasing(coarseNear, printFar, inkF2, farF, inkSum, farSum, farShare) {}
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
    fun refine(full: Mat, coarse: Mat, coarsePrint: Mat, refWork: Mat, fragments: Mat, mixed: Mat): Triple<Mat, Mat, Mat> {
        val k = InkAnalysis.strokeUnit(full.cols(), full.rows())
        val hw = Mat.zeros(full.size(), CvType.CV_8U)
        val overlap = Mat.zeros(full.size(), CvType.CV_8U)
        val print = coarsePrint.clone()
        val paperSmall = paperLowRes(full)
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
                val (roiHw, roiOverlap, roiPrintStrong) = refineRoi(d, coarseRoi, fragRoi, mixedRoi, ref, k)
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
                dAll.release()
            }
        } finally {
            paperSmall.release()
        }
        return Triple(hw, print, overlap)
    }

    /**
     * Removes [handwriting] from [target] by rebuilding: overlap pixels get the nearby print colour
     * back, the rest of the stroke, its rim and the camera halo is inpainted from the PAPER only.
     * [analysis] is the unprocessed page the masks were computed on (same geometry).
     */
    fun erase(target: Mat, analysis: Mat, handwriting: Mat, print: Mat, overlap: Mat): Mat {
        val k = InkAnalysis.strokeUnit(target.cols(), target.rows())
        val s = k / 9.0
        val dst = target.clone()
        val paperSmall = paperLowRes(analysis)
        val targetPaperSmall = paperLowRes(target)
        val targetSmall = Mat()
        Imgproc.resize(target, targetSmall, Size(), 0.125, 0.125, Imgproc.INTER_AREA)
        Imgproc.medianBlur(targetSmall, targetSmall, 21)
        val margin = odd(61 * s) / 2 + odd(13 * s) + odd(11 * s) + 4
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
                    val l1 = max(3, 6 * k) or 1; val l2 = max(5, 20 * k) or 1
                    Imgproc.morphologyEx(prInk, closed, Imgproc.MORPH_CLOSE, Mat.ones(1, l1, CvType.CV_8U))
                    Imgproc.morphologyEx(closed, runs, Imgproc.MORPH_OPEN, Mat.ones(1, l2, CvType.CV_8U))
                    Imgproc.morphologyEx(runs, thick, Imgproc.MORPH_OPEN, Mat.ones(max(3, k) or 1, 1, CvType.CV_8U))
                    val thickNear = dilate(thick, Mat.ones(3, 3, CvType.CV_8U))
                    restoreMask = or(restoreMask, and(and(runs, not(thickNear)), area))
                    releasing(prInk, closed, runs, thick, thickNear) {}
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
                val sr = scaledRect(roi, 0.125, 0.125, targetSmall.cols(), targetSmall.rows())
                val paperColor = Mat()
                val ps = targetSmall.submat(sr)
                Imgproc.resize(ps, paperColor, roi.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
                Imgproc.GaussianBlur(paperColor, paperColor, Size(0.0, 0.0), 8.0)
                val nearArea = dilate(area, ellipse(9 * s))
                val keep = or(and(and(printNear, nearArea), not(area)), restore)
                val tmp = out.clone()
                paperColor.copyTo(tmp, keep)
                val filled = Mat()
                Photo.inpaint(tmp, paperArea, filled, 3.0, Photo.INPAINT_TELEA)
                out.copyTo(filled, keep)
                val dview = dst.submat(roi); filled.copyTo(dview); dview.release()
                releasing(hw, pr, ov, printNear, grown, area, halo, restore, paperArea, strong, weight, tf, tsub, weight3, weighted, num, den,
                    empty, den3, local, out, paperColor, ps, nearArea, keep, tmp, filled) {}
                d.release()
                collectNative()
            }
        } finally {
            paperSmall.release(); targetSmall.release(); targetPaperSmall.release()
        }
        return dst
    }
}
