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
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** One word read by the text recogniser: its graphemes ("ổ" is one) and its box in the page. */
class OcrWord(val graphemes: List<String>, val box: Rect)

/** One text line: its alternative readings (best first), each split into words. */
class OcrLine(val readings: List<List<OcrWord>>, val box: Rect)

/** Reads the text of a BGR page (ML Kit on Android, see [MlKitTextRecognizer]). */
fun interface TextRecognizer {
    fun recognize(bgr: Mat): List<OcrLine>
}

/**
 * Restores printed letters the pen hid, by READING them — Kotlin port of RestorePrintByRecognition
 * in ios/Runner/InkAnalysis.hpp (see there for the reasoning and measurements). The recogniser
 * reads the erased page and the original; words the erase touched are re-spelled from those
 * readings, each spelling typeset from the page's own letters and checked against what is visible
 * before any pixel is put back; only erased pixels that were ink in the original are filled.
 */
object PrintRecognition {
    private class Comps(val count: Int, val labels: IntArray, val x: IntArray, val y: IntArray, val w: IntArray, val h: IntArray, val area: IntArray)

    private fun components(mask: Mat): Comps {
        val labels = Mat(); val stats = Mat(); val cent = Mat()
        val n = Imgproc.connectedComponentsWithStats(mask, labels, stats, cent, 8, CvType.CV_32S)
        val lab = IntArray(labels.total().toInt()); labels.get(0, 0, lab)
        val st = IntArray(n * 5); stats.get(0, 0, st)
        labels.release(); stats.release(); cent.release()
        return Comps(n, lab, IntArray(n) { st[it * 5 + Imgproc.CC_STAT_LEFT] }, IntArray(n) { st[it * 5 + Imgproc.CC_STAT_TOP] },
            IntArray(n) { st[it * 5 + Imgproc.CC_STAT_WIDTH] }, IntArray(n) { st[it * 5 + Imgproc.CC_STAT_HEIGHT] },
            IntArray(n) { st[it * 5 + Imgproc.CC_STAT_AREA] })
    }

    private fun union(a: Rect, b: Rect): Rect {
        if (a.width <= 0 || a.height <= 0) return b.clone()
        val x0 = min(a.x, b.x); val y0 = min(a.y, b.y)
        return Rect(x0, y0, max(a.x + a.width, b.x + b.width) - x0, max(a.y + a.height, b.y + b.height) - y0)
    }

    private fun clip(r: Rect, w: Int, h: Int): Rect {
        val x0 = max(0, r.x); val y0 = max(0, r.y)
        val x1 = min(w, r.x + r.width); val y1 = min(h, r.y + r.height)
        return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))
    }

    private fun median(v: MutableList<Float>): Float { v.sort(); return v[min(v.size - 1, ((v.size - 1) * 0.5).roundToInt())] }

    fun editDistance(a: List<String>, b: List<String>): Int {
        var prev = IntArray(b.size + 1) { it }
        var cur = IntArray(b.size + 1)
        for (i in 1..a.size) {
            cur[0] = i
            for (j in 1..b.size) cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1)
            val t = prev; prev = cur; cur = t
        }
        return prev[b.size]
    }

    /** Candidate spellings from two readings: each alone and every mix where they disagree. */
    fun mixedSpellings(a: List<String>, b: List<String>): List<List<String>> {
        val n = a.size; val m = b.size
        val d = Array(n + 1) { IntArray(m + 1) }
        for (i in 0..n) d[i][0] = i
        for (j in 0..m) d[0][j] = j
        for (i in 1..n) for (j in 1..m) d[i][j] = minOf(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1)
        val out = ArrayList<List<String>>()
        // (both ways of breaking ties in the alignment: only one of them gives "Tìm" from "Tìnt"/"Tim")
        for (order in 0 until 2) {
            val pairs = ArrayList<Pair<String, String>>()
            var i = n; var j = m
            while (i > 0 || j > 0) {
                val diag = i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1
                val up = i > 0 && d[i][j] == d[i - 1][j] + 1
                val left = j > 0 && d[i][j] == d[i][j - 1] + 1
                if (diag && (order == 0 || (!up && !left))) { pairs.add(a[i - 1] to b[j - 1]); i--; j-- }
                else if (up) { pairs.add(a[i - 1] to ""); i-- }
                else if (left) { pairs.add("" to b[j - 1]); j-- }
                else { pairs.add(a[i - 1] to b[j - 1]); i--; j-- }
            }
            pairs.reverse()
            val diff = pairs.indices.filter { pairs[it].first != pairs[it].second }
            if (diff.size > 6) continue   // unrelated readings
            for (mask in 0 until (1 shl diff.size)) {
                val w = ArrayList<String>()
                var dIdx = 0
                for (q in pairs.indices) {
                    val isDiff = dIdx < diff.size && diff[dIdx] == q
                    val g = if (isDiff && (mask shr dIdx and 1) == 1) pairs[q].second else pairs[q].first
                    if (isDiff) dIdx++
                    if (g.isNotEmpty()) w.add(g)
                }
                if (w.isNotEmpty() && w !in out) out.add(w)
            }
        }
        for (w in listOf(a, b)) if (w !in out) out.add(w)
        return out
    }

    /** Ink amount 0..1 against the local paper (the same measure as the re-typesetting uses). */
    private fun inkMap(bgr: Mat): Mat {
        val g = Mat(); val small = Mat(); val paper = Mat(); val out = Mat()
        Imgproc.cvtColor(bgr, g, Imgproc.COLOR_BGR2GRAY)
        Imgproc.resize(g, small, Size(), 0.25, 0.25, Imgproc.INTER_AREA)
        Imgproc.medianBlur(small, small, min(21, (min(small.cols(), small.rows()) - 1) or 1))
        Imgproc.resize(small, paper, g.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        paper.convertTo(paper, CvType.CV_32F); g.convertTo(g, CvType.CV_32F)
        Core.subtract(paper, g, out)
        Core.multiply(out, Scalar(1.0 / 140.0), out)
        Core.max(out, Scalar(0.0), out); Core.min(out, Scalar(1.0), out)
        g.release(); small.release(); paper.release()
        return out
    }

    private class Glyph(val mask: Mat, val src: Rect, val dy: Int, val lineH: Float)
    private class Placement(val x: Int, val y: Int, val g: Glyph)
    private class Fit(val spelling: List<String>, val pl: List<Placement>, val total: Float, val contra: Float, val paint: Mat, val leftover: Mat)

    fun restore(dst: Mat, analysis: Mat, erasedAll: Mat, k: Int, recognizer: TextRecognizer?) {
        if (recognizer == null || Core.countNonZero(erasedAll) == 0) return
        val t = ArrayList<Mat>()
        fun <T : Mat> own(m: T): T { t.add(m); return m }
        fun cmp(a: Mat, v: Double, op: Int) = own(Mat()).also { Core.compare(a, Scalar(v), it, op) }
        fun and(a: Mat, b: Mat) = own(Mat()).also { Core.bitwise_and(a, b, it) }
        fun or(a: Mat, b: Mat) = own(Mat()).also { Core.bitwise_or(a, b, it) }
        fun not(a: Mat) = own(Mat()).also { Core.bitwise_not(a, it) }
        fun dil(a: Mat, kernel: Mat) = own(Mat()).also { Imgproc.dilate(a, it, kernel); kernel.release() }
        fun box3() = Mat.ones(3, 3, CvType.CV_8U)
        try {
            val W = dst.cols(); val H = dst.rows()
            val inkE = own(inkMap(dst)); val inkO = own(inkMap(analysis))
            val coreE = cmp(inkE, 0.45, Core.CMP_GT); val anyE = cmp(inkE, 0.2, Core.CMP_GT)
            val inkOrig = cmp(inkO, 0.35, Core.CMP_GT); val anyO = cmp(inkO, 0.2, Core.CMP_GT)
            // erased = pixels the erase rebuilt, and beside them ink of the original that is paper now
            val erasedNear = dil(erasedAll, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size((2.0 * k + 1), (2.0 * k + 1))))
            val erased = dil(or(erasedAll, and(and(inkOrig, not(anyE)), erasedNear)), box3())
            // evidence: visible print, visible paper, erased-but-ink-in-the-original, erased paper
            val anyENear = dil(anyE, box3()); val anyONear = dil(anyO, box3())
            // where a letter puts ink back, the original's ink within 1 px counts (a letter of the
            // page's own set lands a pixel off the grid of the one it replaces)
            val inkOrigNear = dil(inkOrig, box3())
            val vis = and(coreE, not(erased)); val visPaper = and(not(anyENear), not(erased))
            val hiddenInk = and(erased, inkOrig); val hiddenPaper = and(erased, not(anyONear))
            // rules and table lines: never a leftover of the pen
            val rulesNear = run {
                val lr = max(9, 6 * k) or 1
                val hR = own(Mat()); val vR = own(Mat())
                Imgproc.morphologyEx(anyE, hR, Imgproc.MORPH_OPEN, Mat.ones(1, lr, CvType.CV_8U))
                Imgproc.morphologyEx(anyE, vR, Imgproc.MORPH_OPEN, Mat.ones(lr, 1, CvType.CV_8U))
                dil(or(hR, vR), box3())
            }

            val linesE = recognizer.recognize(dst)
            if (linesE.isEmpty()) return
            val linesO = recognizer.recognize(analysis)

            // --- the page's letters: ink columns (a letter with its accents), labelled by clean words
            val c = components(coreE)
            class Column(var box: Rect, val comps: ArrayList<Int>)
            fun columnsIn(band: Rect, lineH: Float): List<Column> {
                val ids = (1 until c.count).filter {
                    val cx = c.x[it] + c.w[it] / 2; val cy = c.y[it] + c.h[it] / 2
                    cx >= band.x && cx < band.x + band.width && cy >= band.y && cy < band.y + band.height &&
                        c.h[it] <= 1.6f * lineH && c.w[it] <= 3 * lineH
                }.sortedBy { c.x[it] }
                val cols = ArrayList<Column>()
                for (i in ids) {
                    var merged = false
                    for (back in 0 until min(3, cols.size)) {
                        val col = cols[cols.size - 1 - back]
                        val ov = min(col.box.x + col.box.width, c.x[i] + c.w[i]) - max(col.box.x, c.x[i])
                        if (ov >= 0.5f * min(col.box.width, c.w[i])) {
                            col.box = union(col.box, Rect(c.x[i], c.y[i], c.w[i], c.h[i])); col.comps.add(i); merged = true; break
                        }
                    }
                    if (!merged) cols.add(Column(Rect(c.x[i], c.y[i], c.w[i], c.h[i]), arrayListOf(i)))
                }
                return cols.sortedBy { it.box.x }
            }
            fun columnMask(col: Column): Mat {
                val m = Mat.zeros(col.box.height, col.box.width, CvType.CV_8U)
                val b = ByteArray(col.box.width * col.box.height)
                val set = col.comps.toHashSet()
                for (yy in 0 until col.box.height) for (xx in 0 until col.box.width)
                    if (c.labels[(col.box.y + yy) * W + col.box.x + xx] in set) b[yy * col.box.width + xx] = 255.toByte()
                m.put(0, 0, b)
                return m
            }
            val base = FloatArray(linesE.size); val lineH = FloatArray(linesE.size)
            val bank = HashMap<String, ArrayList<Glyph>>()
            for ((li, line) in linesE.withIndex()) {
                val band = clip(line.box, W, H)
                if (band.width <= 0 || band.height <= 0) continue
                val cols = columnsIn(band, band.height.toFloat())
                if (cols.isEmpty()) continue
                base[li] = median(cols.map { (it.box.y + it.box.height).toFloat() }.toMutableList())
                lineH[li] = median(cols.map { it.box.height.toFloat() }.toMutableList())
                val words = line.readings.firstOrNull() ?: continue
                for (w in words) {
                    val wc = cols.filter { val cx = it.box.x + it.box.width / 2; cx >= w.box.x && cx < w.box.x + w.box.width }
                    if (wc.size != w.graphemes.size || wc.isEmpty()) continue
                    var all = Rect()
                    for (col in wc) all = union(all, col.box)
                    val sub = erasedNear.submat(clip(all, W, H)); val dirty = Core.countNonZero(sub); sub.release()
                    if (dirty > 0) continue   // clean words only
                    for ((q, col) in wc.withIndex()) {
                        val v = bank.getOrPut(w.graphemes[q]) { ArrayList() }
                        if (v.size >= 6) continue
                        v.add(Glyph(own(columnMask(col)), col.box, col.box.y - base[li].toInt(), lineH[li]))
                    }
                }
            }
            if (bank.isEmpty()) return

            val source = own(dst.clone())
            val sourceGray = own(Mat()); Imgproc.cvtColor(source, sourceGray, Imgproc.COLOR_BGR2GRAY)
            for ((li, line) in linesE.withIndex()) {
                val words = line.readings.firstOrNull() ?: continue
                if (lineH[li] <= 0f) continue
                val lh = lineH[li]; val lb = base[li]
                for (w in words) {
                    val wb = clip(w.box, W, H)
                    if (wb.width <= 0 || wb.height <= 0) continue
                    run { val s = erased.submat(wb); val n = Core.countNonZero(s); s.release(); if (n < k) return@run null; n }
                        ?: continue   // not touched by the erase
                    val spellings = arrayListOf(w.graphemes)
                    val readings = arrayListOf(w.graphemes)
                    var region = wb.clone()
                    fun consider(o: OcrWord) {
                        val ov = min(o.box.x + o.box.width, wb.x + wb.width) - max(o.box.x, wb.x)
                        val oy = min(o.box.y + o.box.height, wb.y + wb.height) - max(o.box.y, wb.y)
                        if (ov < 0.5f * min(o.box.width, wb.width) || oy < 0.5f * min(o.box.height, wb.height)) return
                        // only a reading of the SAME word — about as long, and mostly over it
                        val dLen = abs(o.graphemes.size - w.graphemes.size)
                        if (dLen > max(1, (0.34 * w.graphemes.size).roundToInt()) || ov < 0.7f * o.box.width) return
                        region = union(region, o.box)
                        readings.add(o.graphemes)
                        for (sp in mixedSpellings(w.graphemes, o.graphemes)) if (sp !in spellings) spellings.add(sp)
                    }
                    for (r in 1 until line.readings.size) for (o in line.readings[r]) consider(o)
                    for (lo in linesO) for (rd in lo.readings) for (o in rd) consider(o)
                    val pad = ceil(0.5f * lh).toInt()
                    val strip = clip(Rect(region.x - pad, (lb - 2.2f * lh).toInt(), region.width + 2 * pad, (3.2f * lh).toInt()), W, H)
                    if (strip.width < 4 || strip.height < 4) continue
                    val sw = strip.width; val sh = strip.height
                    fun bytes(m: Mat): ByteArray { val s = m.submat(strip); val b = ByteArray(sw * sh); s.get(0, 0, b); s.release(); return b }
                    // visible print of the word as the erased page reads it, in its line's band
                    val wordVis = ByteArray(sw * sh)
                    run {
                        val v = bytes(vis)
                        val bx0 = max(wb.x, strip.x) - strip.x; val bx1 = min(wb.x + wb.width, strip.x + sw) - strip.x
                        val by0 = max((lb - 1.8f * lh).toInt(), strip.y) - strip.y; val by1 = min((lb + 0.8f * lh).toInt(), strip.y + sh) - strip.y
                        for (yy in max(0, by0) until max(0, by1)) for (xx in max(0, bx0) until max(0, bx1)) wordVis[yy * sw + xx] = v[yy * sw + xx]
                    }
                    val visCount = wordVis.count { it.toInt() != 0 }
                    val bVisPaper = bytes(visPaper); val bHiddenInk = bytes(hiddenInk); val bHiddenPaper = bytes(hiddenPaper)
                    val bErased = bytes(erased); val bInkOrig = bytes(inkOrigNear); val bAnyE = bytes(anyE); val bRules = bytes(rulesNear)
                    val mArr = FloatArray(sw * sh) {
                        when {
                            bHiddenPaper[it].toInt() != 0 -> -4f
                            bHiddenInk[it].toInt() != 0 -> 0.3f
                            bVisPaper[it].toInt() != 0 -> -4f
                            wordVis[it].toInt() != 0 -> 2f
                            else -> 0f
                        }
                    }
                    val mMat = own(Mat(sh, sw, CvType.CV_32F)); mMat.put(0, 0, mArr)
                    val maxGap = ceil(0.6f * lh).toInt()
                    val accepted = ArrayList<Fit>()
                    for (sp in spellings) {
                        val options = sp.map { g -> bank[g]?.filter { abs(it.lineH - lh) <= 0.15f * lh } ?: emptyList() }
                        if (options.any { it.isEmpty() }) continue
                        // letters in order: letter q starts at most maxGap after letter q-1 ends, never before
                        var prev = FloatArray(sw + 1)
                        val back = Array(sp.size) { Array(sw + 1) { intArrayOf(-1, -1, -1, -1) } }
                        for (q in sp.indices) {
                            val reach = FloatArray(sw + 1) { -Float.MAX_VALUE }; val reachFrom = IntArray(sw + 1) { -1 }
                            for (x in 0..sw) {
                                if (q == 0) { reach[x] = 0f; continue }
                                for (e in max(0, x - maxGap)..x) if (prev[e] > reach[x]) { reach[x] = prev[e]; reachFrom[x] = e }
                            }
                            val cur = FloatArray(sw + 1) { -Float.MAX_VALUE }
                            for ((oi, g) in options[q].withIndex()) {
                                val gm = Mat(); g.mask.convertTo(gm, CvType.CV_32F, 1.0 / 255.0)
                                for (jit in -2..2) {
                                    val gy = lb.toInt() + g.dy + jit - strip.y
                                    if (gy < 0 || gy + gm.rows() > sh || gm.cols() > sw) continue
                                    val rowStrip = mMat.submat(Rect(0, gy, sw, gm.rows()))
                                    val res = Mat(); Imgproc.matchTemplate(rowStrip, gm, res, Imgproc.TM_CCORR)
                                    val rp = FloatArray(res.cols()); res.get(0, 0, rp)
                                    for (x in 0..sw - gm.cols()) {
                                        if (reach[x] == -Float.MAX_VALUE) continue
                                        val v = reach[x] + rp[x]
                                        if (v > cur[x + gm.cols()]) { cur[x + gm.cols()] = v; back[q][x + gm.cols()] = intArrayOf(x, gy, oi, reachFrom[x]) }
                                    }
                                    rowStrip.release(); res.release()
                                }
                                gm.release()
                            }
                            prev = cur
                        }
                        var end = -1
                        for (x in 0..sw) if (prev[x] > -Float.MAX_VALUE && (end < 0 || prev[x] > prev[end])) end = x
                        if (end < 0) continue
                        val pl = arrayOfNulls<Placement>(sp.size)
                        var e = end
                        for (q in sp.indices.reversed()) {
                            val b = back[q][e]
                            pl[q] = Placement(b[0], b[1], options[q][b[2]])
                            e = b[3]
                        }
                        val placements = pl.map { it!! }
                        // what the spelling explains and contradicts
                        val covered = ByteArray(sw * sh); var glyphPx = 0
                        val glyphBytes = placements.map { p -> ByteArray(p.g.mask.cols() * p.g.mask.rows()).also { p.g.mask.get(0, 0, it) } }
                        for ((pi, p) in placements.withIndex()) {
                            val gb = glyphBytes[pi]; val gw = p.g.mask.cols()
                            for (yy in 0 until p.g.mask.rows()) for (xx in 0 until gw)
                                if (gb[yy * gw + xx].toInt() != 0) { covered[(p.y + yy) * sw + p.x + xx] = 255.toByte(); glyphPx++ }
                        }
                        val coveredMat = own(Mat(sh, sw, CvType.CV_8U)); coveredMat.put(0, 0, covered)
                        val near = Mat(); Imgproc.dilate(coveredMat, near, Mat.ones(3, 3, CvType.CV_8U))
                        val coveredNear = ByteArray(sw * sh); near.get(0, 0, coveredNear); near.release()
                        var expl = 0; var contraPx = 0
                        for (i in 0 until sw * sh) {
                            if (wordVis[i].toInt() != 0 && coveredNear[i].toInt() != 0) expl++
                            if (covered[i].toInt() != 0 && (bVisPaper[i].toInt() != 0 || bHiddenPaper[i].toInt() != 0)) contraPx++
                        }
                        val explained = if (visCount > 0) expl.toFloat() / visCount else 0f
                        val contra = contraPx.toFloat() / max(1, glyphPx)
                        val total = prev[end] - 2f * visCount
                        if (explained < 0.9f || contra > 0.06f) continue
                        // the pixels it would put back: erased ones that were ink in the original — of
                        // letters only (a mark wholly under the pen fits any pen stroke)
                        val paint = ByteArray(sw * sh)
                        for ((pi, p) in placements.withIndex()) {
                            val gb = glyphBytes[pi]; val gw = p.g.mask.cols()
                            var seen = false
                            for (yy in 0 until p.g.mask.rows()) for (xx in 0 until gw)
                                if (gb[yy * gw + xx].toInt() != 0 && wordVis[(p.y + yy) * sw + p.x + xx].toInt() != 0) seen = true
                            if (!seen && p.g.mask.rows() < 0.6f * lh) continue
                            for (yy in 0 until p.g.mask.rows()) for (xx in 0 until gw) {
                                val i = (p.y + yy) * sw + p.x + xx
                                if (gb[yy * gw + xx].toInt() != 0 && bErased[i].toInt() != 0 && bInkOrig[i].toInt() != 0) paint[i] = 255.toByte()
                            }
                        }
                        // the ink it leaves unexplained that touches the erased pen: the pen's leftover
                        val leftover = ByteArray(sw * sh)
                        run {
                            val rest = ByteArray(sw * sh) { if (wordVis[it].toInt() != 0 && coveredNear[it].toInt() == 0) 255.toByte() else 0 }
                            val restMat = own(Mat(sh, sw, CvType.CV_8U)); restMat.put(0, 0, rest)
                            val cr = components(restMat)
                            val erasedMat = own(Mat(sh, sw, CvType.CV_8U)); erasedMat.put(0, 0, bErased)
                            val touchMat = Mat(); Imgproc.dilate(erasedMat, touchMat, Mat.ones(3, 3, CvType.CV_8U))
                            val touch = ByteArray(sw * sh); touchMat.get(0, 0, touch); touchMat.release()
                            val touches = BooleanArray(cr.count)
                            for (i in 0 until sw * sh) if (cr.labels[i] > 0 && touch[i].toInt() != 0) touches[cr.labels[i]] = true
                            val take = ByteArray(sw * sh) { val l = cr.labels[it]; if (l > 0 && touches[l] && cr.area[l] <= lh * lh) 255.toByte() else 0 }
                            val takeMat = own(Mat(sh, sw, CvType.CV_8U)); takeMat.put(0, 0, take)
                            val grown = Mat(); Imgproc.dilate(takeMat, grown, Mat.ones(3, 3, CvType.CV_8U))
                            val gr = ByteArray(sw * sh); grown.get(0, 0, gr); grown.release()
                            for (i in 0 until sw * sh)
                                if (gr[i].toInt() != 0 && bAnyE[i].toInt() != 0 && covered[i].toInt() == 0 && bRules[i].toInt() == 0) leftover[i] = 255.toByte()
                        }
                        val paintMat = own(Mat(sh, sw, CvType.CV_8U)); paintMat.put(0, 0, paint)
                        val leftMat = own(Mat(sh, sw, CvType.CV_8U)); leftMat.put(0, 0, leftover)
                        accepted.add(Fit(sp, placements, total, contra, paintMat, leftMat))
                    }
                    // spellings that would put back the same pixels are one answer (best fit kept)
                    accepted.sortByDescending { it.total }
                    val answers = ArrayList<Fit>()
                    for (f in accepted) {
                        val same = answers.any { a -> val d = Mat(); Core.bitwise_xor(a.paint, f.paint, d); val n = Core.countNonZero(d); d.release(); n <= k }
                        if (!same) answers.add(f)
                    }
                    if (answers.isEmpty()) continue
                    // ...among those that fit about as well as the best: a spelling that explains the
                    // page clearly worse (half a line height of score) is no rival — "nhít" / "nhất",
                    // where the pen hid the circumflex's bowl — and must not take away the strokes
                    // the others agree on
                    val bestTotal = answers[0].total
                    answers.removeAll { it !== answers[0] && it.total < bestTotal - 0.5f * lh }
                    // several answers: only what ALL of them put back is certain — unless one, as READ
                    // by the recogniser, restores everything the others do and more
                    var chosenPaint = answers[0].paint.clone().also { own(it) }
                    var chosenLeft = answers[0].leftover.clone().also { own(it) }
                    var chosenPl = answers[0].pl
                    for (q in 1 until answers.size) {
                        Core.bitwise_and(chosenPaint, answers[q].paint, chosenPaint)
                        Core.bitwise_and(chosenLeft, answers[q].leftover, chosenLeft)
                    }
                    if (answers.size > 1) for (f in answers) {
                        if (f.spelling !in readings) continue
                        // (and only one the page does not contradict, fitting about as well as the
                        // best: the covering reading may be a wrong letter — "thế" over "thứ", a "7"
                        // read in the remnant of a pen-crossed "ủa" — that puts back MORE only
                        // because it is wrong)
                        if (f.contra > 0.02f || f.total < bestTotal - 0.25f * lh) continue
                        val covers = answers.all { g ->
                            g === f || run { val d = Mat(); Core.bitwise_and(g.paint, not(f.paint), d); val n = Core.countNonZero(d); d.release(); n <= k }
                        }
                        if (covers) { chosenPaint = f.paint; chosenLeft = f.leftover; chosenPl = f.pl; break }
                    }
                    if (Core.countNonZero(chosenPaint) == 0 && Core.countNonZero(chosenLeft) == 0) continue
                    // leftover to the strip's paper colour, then the letters' own pixels
                    val paperMask = and(not(anyENear), not(erased))
                    val stripDst = dst.submat(strip); val stripPaper = paperMask.submat(strip)
                    val paperColour = Core.mean(stripDst, stripPaper)
                    stripDst.setTo(paperColour, chosenLeft)
                    stripPaper.release()
                    for (p in chosenPl) {
                        val r = Rect(p.x, p.y, p.g.mask.cols(), p.g.mask.rows())
                        val target = stripDst.submat(r)
                        val tg = Mat(); Imgproc.cvtColor(target, tg, Imgproc.COLOR_BGR2GRAY)
                        val srcG = sourceGray.submat(p.g.src)
                        val darker = Mat(); Core.compare(srcG, tg, darker, Core.CMP_LT)
                        val pm = chosenPaint.submat(r)
                        val m = Mat(); Core.bitwise_and(pm, darker, m)
                        val src = source.submat(p.g.src)
                        src.copyTo(target, m)
                        listOf(target, tg, srcG, darker, pm, m, src).forEach { it.release() }
                    }
                    stripDst.release()
                }
            }
        } finally { t.forEach { it.release() } }
    }
}
