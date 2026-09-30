package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.stream.IntStream
import kotlin.math.max
import kotlin.math.min

/**
 * Repairs printed strokes the erase took away where the pen crossed print, by EXAMPLE — no text
 * recognition (OCR misreads exactly the letters a pen crosses). Kotlin port of
 * RepairPrintByExample in ios/Runner/InkAnalysis.hpp: for each erased spot next to visible print,
 * the visible surroundings are matched against the whole page; where two distinct places match
 * closely AND agree on what lies under the pen, the best one's ink fills the erased pixels.
 * All per-pixel work stays in OpenCV (native memory); windows are searched in parallel and applied
 * in window order afterwards, so the result does not depend on the thread count.
 */
object PrintRepair {
    private fun ellipse(size: Int): Mat =
        Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size((size or 1).toDouble(), (size or 1).toDouble()))

    /** The twin to copy from and whether only the rule it continues may be copied. */
    private class Twin(val at: Point, val alongRule: Boolean)

    /** [original]: the page before the erase (same geometry) — a fill only ever goes where it had ink. */
    fun repairByExample(dst: Mat, erased: Mat, original: Mat, k: Int) {
        if (Core.countNonZero(erased) == 0) return
        val owned = ArrayList<Mat>()
        fun <T : Mat> own(m: T): T { owned.add(m); return m }
        try {
            val gray = own(Mat()); Imgproc.cvtColor(dst, gray, Imgproc.COLOR_BGR2GRAY)
            val small = own(Mat()); Imgproc.resize(gray, small, Size(), 0.25, 0.25, Imgproc.INTER_AREA)
            Imgproc.medianBlur(small, small, min(21, (min(small.cols(), small.rows()) - 1) or 1))
            val paper = own(Mat()); Imgproc.resize(small, paper, gray.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            val inkLevel = own(Mat()); Core.subtract(paper, gray, inkLevel)
            val ink = own(Mat()); Core.compare(inkLevel, Scalar(60.0), ink, Core.CMP_GT)
            val notErased = own(Mat()); Core.bitwise_not(erased, notErased)
            val visiblePrint = own(Mat()); Core.bitwise_and(ink, notErased, visiblePrint)
            // ink in the page before the erase (same measure), and within 1 px of it (a twin sits on the
            // page's pixel grid, the damaged letter may be half a pixel off it — on 1-2 px strokes
            // that alone halved the overlap)
            val wasInk = own(Mat()); val wasNear = own(Mat())
            run {
                val og = own(Mat()); Imgproc.cvtColor(original, og, Imgproc.COLOR_BGR2GRAY)
                val os = own(Mat()); Imgproc.resize(og, os, Size(), 0.25, 0.25, Imgproc.INTER_AREA)
                Imgproc.medianBlur(os, os, min(21, (min(os.cols(), os.rows()) - 1) or 1))
                val op = own(Mat()); Imgproc.resize(os, op, og.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
                val ol = own(Mat()); Core.subtract(op, og, ol)
                Core.compare(ol, Scalar(60.0), wasInk, Core.CMP_GT)
                Imgproc.dilate(wasInk, wasNear, own(Mat.ones(3, 3, CvType.CV_8U)))
            }

            // erased pixels right next to visible print = where print was cut
            val nearPrint = own(Mat()); Imgproc.dilate(visiblePrint, nearPrint, own(ellipse(2 * k + 1)))
            val seeds = own(Mat()); Core.bitwise_and(erased, nearPrint, seeds)
            if (Core.countNonZero(seeds) == 0) return

            val t = max(15, 6 * k) or 1   // patch size
            // visible print that is not a rule, nor a flat line piece (what the erase left of a
            // slightly sloped pen underline matched other sloped lines and was "repaired" longer);
            // and the lines the visible rules run along
            val visibleText = own(Mat()); val ruleLines = own(Mat())
            run {
                val hR = own(Mat()); val vR = own(Mat())
                Imgproc.morphologyEx(visiblePrint, hR, Imgproc.MORPH_OPEN, own(Mat.ones(1, t, CvType.CV_8U)))
                Imgproc.morphologyEx(visiblePrint, vR, Imgproc.MORPH_OPEN, own(Mat.ones(t, 1, CvType.CV_8U)))
                val hL = own(Mat()); val vL = own(Mat())
                Imgproc.dilate(hR, hL, own(Mat.ones(3, 2 * t + 1, CvType.CV_8U)))
                Imgproc.dilate(vR, vL, own(Mat.ones(2 * t + 1, 3, CvType.CV_8U)))
                Core.bitwise_or(hL, vL, ruleLines)
                val near = own(Mat()); Core.bitwise_or(hR, vR, near)
                Imgproc.dilate(near, near, own(Mat.ones(3, 3, CvType.CV_8U)))
                Core.bitwise_not(near, near)
                Core.bitwise_and(visiblePrint, near, visibleText)
                val labels = own(Mat()); val stats = own(Mat()); val centroids = own(Mat())
                val n = Imgproc.connectedComponentsWithStats(visibleText, labels, stats, centroids, 8, CvType.CV_32S)
                val flat = BooleanArray(n) {
                    val w = stats.get(it, Imgproc.CC_STAT_WIDTH)[0]
                    it > 0 && w >= t / 2 && w >= 4 * stats.get(it, Imgproc.CC_STAT_HEIGHT)[0]
                }
                if (flat.any { it }) {
                    val lab = IntArray(labels.total().toInt()); labels.get(0, 0, lab)
                    val vt = ByteArray(lab.size); visibleText.get(0, 0, vt)
                    for (i in lab.indices) if (flat[lab[i]]) vt[i] = 0
                    visibleText.put(0, 0, vt)
                }
            }
            // matched on SHAPE: ink amount 0..1, lightly blurred
            val inkF = own(Mat())
            inkLevel.convertTo(inkF, CvType.CV_32F, 1.0 / 120.0)
            Core.min(inkF, Scalar(1.0), inkF)
            Core.max(inkF, Scalar(0.0), inkF)
            Imgproc.GaussianBlur(inkF, inkF, Size(0.0, 0.0), 1.0)
            // coarse search at half resolution, refined at full resolution around the best hit
            val sc = 2
            val inkH = own(Mat()); Imgproc.resize(inkF, inkH, Size(), 1.0 / sc, 1.0 / sc, Imgproc.INTER_AREA)
            val erasedH = own(Mat()); Imgproc.resize(erased, erasedH, inkH.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
            val th = t / sc
            val resW = inkH.cols() - th + 1; val resH = inkH.rows() - th + 1
            if (resW <= 0 || resH <= 0) return
            // spots a twin may come from: not (more than 2%) erased themselves
            val validBase = run {
                val e01 = own(Mat()); erasedH.convertTo(e01, CvType.CV_32F, 1.0 / 255.0)
                val sum = own(Mat())
                Imgproc.boxFilter(e01, sum, -1, Size(th.toDouble(), th.toDouble()), Point(0.0, 0.0), false)
                val v = own(Mat())
                Core.compare(sum.submat(Rect(0, 0, resW, resH)), Scalar(0.02 * th * th), v, Core.CMP_LE)
                v
            }

            // windows on a half-patch grid wherever print was cut
            val windows = ArrayList<Point>()
            val step = max(1, t / 2)
            var cy = 0
            while (cy < seeds.rows()) {
                var cx = 0
                while (cx < seeds.cols()) {
                    val cell = seeds.submat(Rect(cx, cy, min(step, seeds.cols() - cx), min(step, seeds.rows() - cy)))
                    if (Core.countNonZero(cell) >= k) {
                        windows.add(Point(
                            max(0, min(dst.cols() - t, cx + step / 2 - t / 2)).toDouble(),
                            max(0, min(dst.rows() - t, cy + step / 2 - t / 2)).toDouble()))
                    }
                    cell.release()
                    cx += step
                }
                cy += step
            }

            val twinOf = arrayOfNulls<Twin>(windows.size)
            IntStream.range(0, windows.size).parallel().forEach { w ->
                twinOf[w] = findTwin(windows[w].x.toInt(), windows[w].y.toInt(), t, th, sc, k,
                    erased, ink, visiblePrint, visibleText, wasNear, inkF, inkH, erasedH, validBase, dst.cols(), dst.rows())
            }

            // fill only the erased pixels that were ink in the original, and only with the twin's INK
            // (paper is already paper)
            val source = own(dst.clone())
            val done = own(Mat.zeros(dst.size(), CvType.CV_8U))
            for (w in windows.indices) {
                val tw = twinOf[w] ?: continue
                val r = Rect(windows[w].x.toInt(), windows[w].y.toInt(), t, t)
                val rs = Rect(tw.at.x.toInt(), tw.at.y.toInt(), t, t)
                val e = erased.submat(r); val srcInk = ink.submat(rs); val dn = done.submat(r)
                val was = wasInk.submat(r); val line = ruleLines.submat(r)
                val notDone = Mat(); Core.bitwise_not(dn, notDone)
                val mask = Mat(); Core.bitwise_and(e, srcInk, mask); Core.bitwise_and(mask, notDone, mask)
                Core.bitwise_and(mask, was, mask)
                if (tw.alongRule) Core.bitwise_and(mask, line, mask)
                val src = source.submat(rs); val out = dst.submat(r)
                src.copyTo(out, mask)
                Core.bitwise_or(dn, mask, dn)
                listOf(e, srcInk, dn, was, line, notDone, mask, src, out).forEach { it.release() }
            }
        } finally {
            owned.forEach { it.release() }
        }
    }

    /** The twin (full-resolution top-left) for the window at ([x0], [y0]), or null if none convinces. */
    private fun findTwin(
        x0: Int, y0: Int, t: Int, th: Int, sc: Int, k: Int,
        erased: Mat, ink: Mat, visiblePrint: Mat, visibleText: Mat, wasNear: Mat, inkF: Mat, inkH: Mat, erasedH: Mat, validBase: Mat,
        cols: Int, rows: Int,
    ): Twin? {
        val owned = ArrayList<Mat>()
        fun <T : Mat> own(m: T): T { owned.add(m); return m }
        try {
            val r = Rect(x0, y0, t, t)
            val known = own(Mat()); Core.bitwise_not(own(erased.submat(r)), known)
            val knownCount = Core.countNonZero(known)
            val vis = own(visiblePrint.submat(r))
            val knownInk = Core.countNonZero(vis)
            if (knownCount < 0.5 * t * t || knownInk < 2 * k * k) return null   // too little evidence
            run {   // only a printed rule in view: rules are re-joined separately
                val rowsWithInk = own(Mat())
                Core.reduce(vis, rowsWithInk, 1, Core.REDUCE_MAX)
                val b = ByteArray(rowsWithInk.rows()); rowsWithInk.get(0, 0, b)
                var top = -1; var bottom = -1
                for (y in b.indices) if (b[y].toInt() != 0) { if (top < 0) top = y; bottom = y }
                if (top < 0 || bottom - top + 1 <= k) return null
            }
            // otherwise only printed rules in view (a vertical table line beside a pen answer in its cell):
            // all a twin can tell is how the rule goes on — what lies beside a twin rule (a question
            // number next to the table's border) says nothing about this cell
            val ruleOnly = Core.countNonZero(own(visibleText.submat(r))) < 2 * k * k

            // coarse search
            val rh = Rect(x0 / sc, y0 / sc, th, th)
            if (rh.x + th > inkH.cols() || rh.y + th > inkH.rows()) return null
            val maskH = own(Mat())
            val knownH = own(Mat()); Core.bitwise_not(own(erasedH.submat(rh)), knownH)
            knownH.convertTo(maskH, CvType.CV_32F, 1.0 / 255.0)
            val res = own(Mat())
            Imgproc.matchTemplate(inkH, own(inkH.submat(rh)), res, Imgproc.TM_SQDIFF, maskH)
            // the best valid spot, then the best valid spot clear of the first (not itself)
            val valid = own(validBase.clone())
            fun exclude(p: Point) {
                val ex = Rect(max(0, p.x.toInt() - th + 1), max(0, p.y.toInt() - th + 1), 0, 0)
                ex.width = min(valid.cols(), p.x.toInt() + th) - ex.x
                ex.height = min(valid.rows(), p.y.toInt() + th) - ex.y
                if (ex.width > 0 && ex.height > 0) { val v = valid.submat(ex); v.setTo(Scalar(0.0)); v.release() }
            }
            exclude(Point(rh.x.toDouble(), rh.y.toDouble()))
            val twinsH = ArrayList<Point>()
            for (pass in 0 until 2) {
                if (Core.countNonZero(valid) == 0) break
                val mm = Core.minMaxLoc(res, valid)
                twinsH.add(mm.minLoc)
                exclude(mm.minLoc)
            }
            if (twinsH.size < 2) return null

            // full-resolution refinement ±sc px of each
            val maskF = own(Mat()); known.convertTo(maskF, CvType.CV_32F, 1.0 / 255.0)
            val self = own(inkF.submat(r))
            val diff = own(Mat()); val sq = own(Mat())
            val twins = ArrayList<Point>(); val scores = ArrayList<Double>()
            for (h in twinsH) {
                var bp: Point? = null
                var bf = 1e30
                for (dy in -sc..sc + 1) for (dx in -sc..sc + 1) {
                    val sx = h.x.toInt() * sc + dx; val sy = h.y.toInt() * sc + dy
                    if (sx < 0 || sy < 0 || sx + t > cols || sy + t > rows) continue
                    val cand = inkF.submat(Rect(sx, sy, t, t))
                    Core.subtract(cand, self, diff)
                    Core.multiply(diff, diff, sq)
                    Core.multiply(sq, maskF, sq)
                    val ssd = Core.sumElems(sq).`val`[0]
                    cand.release()
                    if (ssd < bf) { bf = ssd; bp = Point(sx.toDouble(), sy.toDouble()) }
                }
                if (bp == null) return null
                twins.add(bp)
                // normalised by the visible INK, so a mostly-blank patch cannot win on paper alone
                scores.add(bf / max(1, knownInk))
            }
            // both twins must be convincing AND agree on what lies under the pen
            val e = own(erased.submat(r))
            val f0 = own(Mat()); Core.bitwise_and(own(ink.submat(Rect(twins[0].x.toInt(), twins[0].y.toInt(), t, t))), e, f0)
            val f1 = own(Mat()); Core.bitwise_and(own(ink.submat(Rect(twins[1].x.toInt(), twins[1].y.toInt(), t, t))), e, f1)
            val both = own(Mat()); Core.bitwise_and(f0, f1, both)
            val either = own(Mat()); Core.bitwise_or(f0, f1, either)
            val inter = Core.countNonZero(both); val uni = Core.countNonZero(either)
            val agree = if (uni == 0) 1.0 else inter.toDouble() / uni
            if (scores[0] > 0.06 || scores[1] > 0.10 || agree < 0.6) return null
            // ...and on what was really there: print under the pen was ink in the original, so a twin
            // that puts its ink where the original shows paper is a lookalike context ("đầu." beside
            // a pen "76" matched "762. Tính" and pasted a "T" into the blank: 70% of it on ink, a
            // pen-crossed "thứ" 100%)
            val under = own(Mat()); Core.bitwise_and(f0, own(wasNear.submat(r)), under)
            if (Core.countNonZero(under) < 0.9 * Core.countNonZero(f0)) return null
            return Twin(twins[0], ruleOnly)
        } finally {
            owned.forEach { it.release() }
        }
    }
}
