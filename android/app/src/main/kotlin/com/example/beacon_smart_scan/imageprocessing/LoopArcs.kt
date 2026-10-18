package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * The rest of a pen loop round a printed word. Where a circle passes the letters' baseline the
 * detectors often keep its low arc as print (it lies in the text line and is as dark as the print),
 * and the erase leaves a crescent under the word. Such a piece is told from print by shape: it
 * touches the handwriting, it is flat (well under a letter's height, much wider than tall), thin,
 * curved (a printed rule is straight) and short (a few letters at most). It becomes handwriting.
 * Mirrors LoopArcs in ios/Runner/ImageProcessingOpenCV.mm.
 */
object LoopArcs {
    /**
     * Off: on 26 labelled pages it took printed brackets, dashes and small frames touching pen marks
     * as loops (precision p11 93 -> 80, p06 88 -> 77) for +0.2 erased. Dev switch: `-e arcs 1`.
     */
    @JvmStatic var enabled = false
    @JvmStatic var report = ""

    /** [ink] CV_8UC1; [handwriting] / [print] / [overlap] CV_8UC1 255 = set, updated in place. */
    fun apply(ink: Mat, handwriting: Mat, print: Mat, overlap: Mat, minArea: Int) {
        report = ""
        if (!enabled) return
        val w = ink.cols(); val h = ink.rows()
        val gh = textHeight(print, handwriting, minArea)
        if (gh <= 0) { report = "arcs: no text height\n"; return }
        val k = InkAnalysis.strokeUnit(w, h)
        val rest = Mat(); val notHw = Mat()
        Core.bitwise_not(handwriting, notHw); Core.bitwise_and(ink, notHw, rest); notHw.release()
        val near = Mat(); Imgproc.dilate(handwriting, near, Mat.ones(3, 3, CvType.CV_8U))
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(rest, labels, stats, cents, 8, CvType.CV_32S)
        rest.release(); cents.release()
        val st = IntArray(n * 5); stats.get(0, 0, st); stats.release()
        val la = IntArray(w * h); labels.get(0, 0, la); labels.release()
        val nr = ByteArray(w * h); near.get(0, 0, nr); near.release()
        val touch = BooleanArray(n)
        // per component: least-squares line of its pixels (curvature) and touch with the pen
        val sx = DoubleArray(n); val sy = DoubleArray(n); val sxx = DoubleArray(n); val sxy = DoubleArray(n); val syy = DoubleArray(n); val cnt = IntArray(n)
        for (i in 0 until w * h) {
            val l = la[i]; if (l == 0) continue
            if (nr[i].toInt() != 0) touch[l] = true
            val x = (i % w).toDouble(); val y = (i / w).toDouble()
            sx[l] += x; sy[l] += y; sxx[l] += x * x; sxy[l] += x * y; syy[l] += y * y; cnt[l]++
        }
        val take = BooleanArray(n)
        var taken = 0; var px = 0
        for (l in 1 until n) {
            if (!touch[l]) continue
            val bw = st[l * 5 + 2]; val bh = st[l * 5 + 3]; val area = st[l * 5 + 4]
            if (area < minArea || bh > 0.5 * gh || bw < 1.5 * bh || bw > 4 * gh) continue
            if (area.toDouble() / bw > 0.45 * k + 1) continue                        // thin
            val m = cnt[l].toDouble()
            val den = m * sxx[l] - sx[l] * sx[l]
            val a = if (den != 0.0) (m * sxy[l] - sx[l] * sy[l]) / den else 0.0
            val b = (sy[l] - a * sx[l]) / m
            val res = sqrt(max(0.0, (syy[l] - a * sxy[l] - b * sy[l]) / m))
            if (res < 1.2 || abs(a) > 1.0) continue                                    // straight: a rule
            take[l] = true; taken++; px += area
        }
        if (taken > 0) {
            val hw = ByteArray(w * h); handwriting.get(0, 0, hw)
            val pr = ByteArray(w * h); print.get(0, 0, pr)
            val ov = ByteArray(w * h); overlap.get(0, 0, ov)
            for (i in 0 until w * h) if (take[la[i]]) { hw[i] = 255.toByte(); pr[i] = 0; ov[i] = 0 }
            handwriting.put(0, 0, hw); print.put(0, 0, pr); overlap.put(0, 0, ov)
        }
        val loops = loopsRoundWords(ink, handwriting, print, overlap, gh, k)
        report = "arcs: text %d px, taken %d (%d px), loops round words %d\n".format(gh, taken, px, loops)
    }

    /**
     * Black-pen loops round printed words: the loop and the letters it touches are one dark piece
     * the detectors keep as print. A print piece much taller than the text line, wide and hollow is
     * such a loop: its pixels outside the text line (the rows the page's letters stand in, from
     * the free-standing glyphs beside it) are pen; inside the line, where the word is bold, the
     * loop's thin stroke is pen and the bold letters stay print. Returns the loops taken.
     */
    private fun loopsRoundWords(ink: Mat, handwriting: Mat, print: Mat, overlap: Mat, gh: Int, k: Int): Int {
        val w = ink.cols(); val h = ink.rows()
        val pure = Mat(); val notHw = Mat()
        Core.bitwise_not(handwriting, notHw); Core.bitwise_and(print, notHw, pure); Core.bitwise_and(pure, ink, pure); notHw.release()
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(pure, labels, stats, cents, 8, CvType.CV_32S)
        cents.release()
        val st = IntArray(n * 5); stats.get(0, 0, st); stats.release()
        // the text lines: rows of free-standing glyphs, grown along the line
        val glyph = BooleanArray(n) { it > 0 && st[it * 5 + 3] in (gh / 2)..(3 * gh / 2) && st[it * 5 + 2] <= 3 * gh }
        val loop = BooleanArray(n) {
            if (it == 0) return@BooleanArray false
            val bw = st[it * 5 + 2]; val bh = st[it * 5 + 3]; val a = st[it * 5 + 4]
            bh >= 1.4 * gh && bh <= 4 * gh && bw >= 1.8 * gh && bw <= 14 * gh && a < 0.35 * bw * bh
        }
        val la = IntArray(w * h); labels.get(0, 0, la); labels.release()
        val nearHw = Mat(); Imgproc.dilate(handwriting, nearHw, Mat.ones(3, 3, CvType.CV_8U))
        val nh = ByteArray(w * h); nearHw.get(0, 0, nh); nearHw.release()
        val touch = BooleanArray(n)
        for (i in 0 until w * h) if (nh[i].toInt() != 0) touch[la[i]] = true
        val gm = Mat.zeros(h, w, CvType.CV_8U); val row = ByteArray(w)
        for (y in 0 until h) { for (x in 0 until w) row[x] = if (glyph[la[y * w + x]]) 255.toByte() else 0; gm.put(y, 0, row) }
        val band = Mat(); Imgproc.dilate(gm, band, Mat.ones(max(3, gh / 4) or 1, 4 * gh + 1, CvType.CV_8U))
        // bold letters survive an opening the thin loop stroke does not
        val core = Mat(); Imgproc.morphologyEx(pure, core, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, org.opencv.core.Size((k / 2 + 2).toDouble(), (k / 2 + 2).toDouble())))
        Imgproc.dilate(core, core, Mat.ones(3, 3, CvType.CV_8U))
        val bd = ByteArray(w * h); band.get(0, 0, bd); val cr = ByteArray(w * h); core.get(0, 0, cr)
        listOf(gm, band, core, pure).forEach { it.release() }
        // pieces of a loop the pen ticks cut apart: small thin print pieces touching the pen, lying
        // mostly outside the text line but close to it (a picture's lines are far from any line)
        val bandNear = Mat(); run {
            val bm = Mat(h, w, CvType.CV_8U); bm.put(0, 0, bd)
            Imgproc.dilate(bm, bandNear, Mat.ones((gh or 1), 1, CvType.CV_8U)); bm.release()
        }
        val bn = ByteArray(w * h); bandNear.get(0, 0, bn); bandNear.release()
        val outCnt = IntArray(n); val nearCnt = IntArray(n)
        for (i in 0 until w * h) { val l = la[i]; if (l == 0 || !touch[l] || loop[l] || glyph[l]) continue
            if (bd[i].toInt() == 0) outCnt[l]++; if (bn[i].toInt() != 0) nearCnt[l]++ }
        val arc = BooleanArray(n) {
            if (it == 0 || !touch[it] || loop[it] || glyph[it]) return@BooleanArray false
            val bw = st[it * 5 + 2]; val bh = st[it * 5 + 3]; val a = st[it * 5 + 4]
            a >= 8 && bw <= 6 * gh && bh <= 0.8 * gh && bw >= bh && outCnt[it] >= 0.6 * a && nearCnt[it] >= 0.9 * a &&
                a.toDouble() / max(bw, bh) <= 0.45 * k + 1.5
        }
        // a loop is round a word ON a text line: its centre is in the line (a picture's is above its caption)
        for (l in 1 until n) if (loop[l]) {
            val cx = st[l * 5] + st[l * 5 + 2] / 2; val cy = st[l * 5 + 1] + st[l * 5 + 3] / 2
            if (bd[cy * w + cx].toInt() == 0) loop[l] = false
        }
        // per loop: is the word inside bold (enough core)?
        val coreCnt = IntArray(n); val inBand = IntArray(n)
        for (i in 0 until w * h) { val l = la[i]; if (!loop[l]) continue; if (bd[i].toInt() != 0) { inBand[l]++; if (cr[i].toInt() != 0) coreCnt[l]++ } }
        val bold = BooleanArray(n) { loop[it] && coreCnt[it] >= 0.4 * inBand[it] }
        val hw = ByteArray(w * h); handwriting.get(0, 0, hw)
        val pr = ByteArray(w * h); print.get(0, 0, pr)
        val ov = ByteArray(w * h); overlap.get(0, 0, ov)
        for (i in 0 until w * h) {
            val l = la[i]
            if (arc[l]) { hw[i] = 255.toByte(); pr[i] = 0; ov[i] = 0; continue }
            if (!loop[l]) continue
            val outside = bd[i].toInt() == 0
            if (outside || (bold[l] && cr[i].toInt() == 0)) { hw[i] = 255.toByte(); pr[i] = 0; ov[i] = 0 }
        }
        handwriting.put(0, 0, hw); print.put(0, 0, pr); overlap.put(0, 0, ov)
        return loop.count { it } + arc.count { it }
    }

    /** Median height of letter-sized print components off the handwriting, or 0. */
    private fun textHeight(print: Mat, handwriting: Mat, minArea: Int): Int {
        val pure = Mat(); val notHw = Mat()
        Core.bitwise_not(handwriting, notHw); Core.bitwise_and(print, notHw, pure); notHw.release()
        val l = Mat(); val s = Mat(); val c = Mat()
        val n = Imgproc.connectedComponentsWithStats(pure, l, s, c, 8, CvType.CV_32S)
        val st = IntArray(n * 5); if (n > 0) s.get(0, 0, st)
        listOf(pure, l, s, c).forEach { it.release() }
        val hs = (1 until n).filter { st[it * 5 + 4] >= minArea && st[it * 5 + 3] >= 5 && st[it * 5 + 2] <= 3 * st[it * 5 + 3] }
            .map { st[it * 5 + 3] }.sorted()
        return if (hs.size < 10) 0 else hs[hs.size / 2]
    }
}
