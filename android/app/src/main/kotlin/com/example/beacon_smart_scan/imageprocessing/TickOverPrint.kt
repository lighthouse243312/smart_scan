package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * A coloured mark (a red tick, a cross) drawn across print fuses with it into one stroke, and the
 * detectors take the whole stroke — the black "?(x)" under the tick too — as handwriting, so the
 * erase eats the print. Here such a stroke is split by each pixel's own colour: its black,
 * colourless pieces that stand on a printed text line — print of about the text's height right
 * beside them, in the same rows — are print after all and stay; only the coloured mark is erased.
 * Colour is read pixel by pixel (no window: a window beside a red tick turns the print red).
 * Mirrors TickOverPrint in ios/Runner/ImageProcessingOpenCV.mm.
 */
object TickOverPrint {
    /**
     * Off: on 21 labelled pages it kept black-pen writing standing beside print as print (recall
     * 95.6 → 92.4, p22 96.0 → 73.6) — a black pen has the print's colour. Dev switch: `-e tick 1`.
     */
    @JvmStatic var enabled = false
    /** Lab chroma past which a pixel is pen-coloured (a red tick: 40-70)... */
    private const val PEN_CHROMA = 25.0
    /** ...and below which it is colourless (black print, a pen photographed black). */
    private const val GREY_CHROMA = 12.0
    /** Darker than this (grey level) is ink for this test. */
    private const val DARK = 140
    /** A stroke needs this many pen-coloured pixels to be a coloured mark over print. */
    private const val MIN_PEN = 20
    /** A piece is text-sized up to this many text heights tall. */
    private const val MAX_HEIGHT = 1.6
    /** Print beside a piece is looked for within this many text heights left and right. */
    private const val REACH = 2.0

    /** Dev counters of the last run (InkHarness report). */
    @JvmStatic var report = ""

    /** [src] CV_8UC3; [handwriting] / [print] / [overlap] CV_8UC1 255 = set, updated in place. */
    fun apply(src: Mat, handwriting: Mat, print: Mat, overlap: Mat, minArea: Int) {
        report = ""
        if (!enabled) return
        val w = src.cols(); val h = src.rows()
        val lab = Mat(); Imgproc.cvtColor(src, lab, Imgproc.COLOR_BGR2Lab)
        val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
        val px = ByteArray(w * 3); val gr = ByteArray(w)
        // per pixel: 1 pen-coloured, 2 colourless dark
        val kind = ByteArray(w * h)
        for (y in 0 until h) {
            lab.get(y, 0, px); gray.get(y, 0, gr)
            for (x in 0 until w) {
                val a = (px[x * 3 + 1].toInt() and 0xff) - 128.0; val b = (px[x * 3 + 2].toInt() and 0xff) - 128.0
                val c = sqrt(a * a + b * b)
                val dark = (gr[x].toInt() and 0xff) < DARK
                kind[y * w + x] = when { c >= PEN_CHROMA -> 1; dark && c <= GREY_CHROMA -> 2; else -> 0 }
            }
        }
        lab.release(); gray.release()

        // the page's text height: printed glyphs off the handwriting
        val pureMat = Mat(); val notHw = Mat()
        Core.bitwise_not(handwriting, notHw); Core.bitwise_and(print, notHw, pureMat); notHw.release()
        val textH = textHeight(pureMat, minArea)
        if (textH <= 0) { pureMat.release(); report = "tick: no text height\n"; return }
        val pure = ByteArray(w * h); pureMat.get(0, 0, pure); pureMat.release()

        // handwriting strokes holding a coloured mark
        val hwArr = ByteArray(w * h); handwriting.get(0, 0, hwArr)
        val labels = Mat(); val n = Imgproc.connectedComponents(handwriting, labels, 8, CvType.CV_32S)
        val la = IntArray(w * h); labels.get(0, 0, la); labels.release()
        val pen = IntArray(n)
        for (i in 0 until w * h) if (la[i] > 0 && kind[i].toInt() == 1) pen[la[i]]++
        // their colourless pieces
        val greyMat = Mat(h, w, CvType.CV_8U)
        val grey = ByteArray(w * h) { i -> if (la[i] > 0 && pen[la[i]] >= MIN_PEN && kind[i].toInt() == 2) 255.toByte() else 0 }
        greyMat.put(0, 0, grey)
        val pl = Mat(); val ps = Mat(); val pc = Mat()
        val np = Imgproc.connectedComponentsWithStats(greyMat, pl, ps, pc, 8, CvType.CV_32S)
        greyMat.release(); pc.release()
        val st = IntArray(np * 5); ps.get(0, 0, st); ps.release()
        val reach = (REACH * textH).toInt()
        val keep = BooleanArray(np)
        var kept = 0
        for (l in 1 until np) {
            val x0 = st[l * 5]; val y0 = st[l * 5 + 1]; val bw = st[l * 5 + 2]; val bh = st[l * 5 + 3]; val area = st[l * 5 + 4]
            if (area < minArea || bh > MAX_HEIGHT * textH || bw > 4 * textH) continue
            // printed text in the same rows, right beside it (left or right)
            var beside = 0
            for (y in y0 until y0 + bh) {
                for (x in max(0, x0 - reach) until x0) if (pure[y * w + x].toInt() != 0) beside++
                for (x in x0 + bw until min(w, x0 + bw + reach)) if (pure[y * w + x].toInt() != 0) beside++
            }
            if (beside >= textH * 2) { keep[l] = true; kept++ }
        }
        val plA = IntArray(w * h); pl.get(0, 0, plA); pl.release()
        val prArr = ByteArray(w * h); print.get(0, 0, prArr)
        val ovArr = ByteArray(w * h); overlap.get(0, 0, ovArr)
        var pixels = 0
        for (i in 0 until w * h) if (keep[plA[i]]) {
            hwArr[i] = 0; prArr[i] = 255.toByte(); ovArr[i] = 0; pixels++
        }
        handwriting.put(0, 0, hwArr); print.put(0, 0, prArr); overlap.put(0, 0, ovArr)
        report = "tick: text %d px, strokes with a mark %d, pieces kept as print %d (%d px)\n"
            .format(textH, (1 until n).count { pen[it] >= MIN_PEN }, kept, pixels)
    }

    /** Median height of the printed glyphs in [pure] (letter-sized components), or 0. */
    private fun textHeight(pure: Mat, minArea: Int): Int {
        val l = Mat(); val s = Mat(); val c = Mat()
        val n = Imgproc.connectedComponentsWithStats(pure, l, s, c, 8, CvType.CV_32S)
        val st = IntArray(n * 5); if (n > 0) s.get(0, 0, st)
        listOf(l, s, c).forEach { it.release() }
        val hs = (1 until n).filter { st[it * 5 + 4] >= minArea && st[it * 5 + 3] >= 5 && st[it * 5 + 2] <= 3 * st[it * 5 + 3] }
            .map { st[it * 5 + 3] }.sorted()
        return if (hs.size < 10) 0 else hs[hs.size / 2]
    }
}
