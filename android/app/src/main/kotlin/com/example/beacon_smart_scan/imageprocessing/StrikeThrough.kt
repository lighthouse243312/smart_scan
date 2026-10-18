package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

/**
 * Pen strike-throughs across printed words ("Fruit / are / your" crossed out). A black pen line has
 * the print's colour, so the colour layers call it print and it was never erased. It is found by
 * shape and place instead: a long (6 stroke units), thin, horizontal run of ink with print right
 * above AND right below it along most of its length — it runs through the middle of the letters.
 * Printed text has no such line (an underline has print only above it, a table rule none close).
 *
 * The line becomes handwriting, to be erased whole; where a printed letter crosses it (print right
 * at its top and bottom edge in that column) the pixels are remembered as print under the pen
 * (overlap), which the erase draws back — so the word comes back without the line through it.
 * Mirrors StrikeThrough in ios/Runner/ImageProcessingOpenCV.mm.
 */
object StrikeThrough {
    /**
     * Off: measured on 21 labelled pages the shape test also takes table rules squeezed between
     * text rows and the horizontal bars of large print and drawings (print lost on 3 pages, more
     * handwriting left on 2). Dev switch: InkHarness `-e strike 1`.
     */
    @JvmStatic var enabled = false
    /** Dev switch (InkHarness `-e strikeslant 0`): straight runs only. */
    @JvmStatic var slanted = true
    /** A strike is at least this many stroke units long. */
    @JvmStatic var MIN_LEN = 3

    /** [src]: the page (BGR); [handwriting] / [print] / [overlap]: CV_8UC1 255 = set, updated in place. */
    fun apply(src: Mat, handwriting: Mat, print: Mat, overlap: Mat) {
        if (!enabled) return
        val w = src.cols(); val h = src.rows()
        val k = InkAnalysis.strokeUnit(w, h)
        // dark ink (well clear of the paper): the line's core, not its soft rim
        val ink = Mat()
        run {
            val g = Mat(); val bg = Mat(); val kernel = Mat.ones(25, 25, CvType.CV_8U)
            Imgproc.cvtColor(src, g, Imgproc.COLOR_BGR2GRAY)
            Imgproc.morphologyEx(g, bg, Imgproc.MORPH_CLOSE, kernel)
            Core.subtract(bg, g, g)
            Imgproc.threshold(g, ink, 40.0, 255.0, Imgproc.THRESH_BINARY)
            listOf(g, bg, kernel).forEach { it.release() }
        }
        // a hand-drawn strike leans and steps a pixel from row to row: the run is taken on the ink
        // grown a pixel up and down, then cut back to the ink itself
        val run = Mat(); val hk = Mat.ones(1, 3 * k, CvType.CV_8U)
        if (slanted) {
            val grown = Mat(); Imgproc.dilate(ink, grown, Mat.ones(3, 1, CvType.CV_8U))
            Imgproc.morphologyEx(grown, run, Imgproc.MORPH_OPEN, hk); grown.release()
            Core.bitwise_and(run, ink, run)
        } else Imgproc.morphologyEx(ink, run, Imgproc.MORPH_OPEN, hk)
        hk.release(); ink.release()
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(run, labels, stats, cents, 8, CvType.CV_32S)
        run.release(); cents.release()
        val st = IntArray(n * 5); stats.get(0, 0, st); stats.release()
        val pureRows = ByteArray(w * h)
        run {
            val pr = ByteArray(w * h); print.get(0, 0, pr)
            val hr = ByteArray(w * h); handwriting.get(0, 0, hr)
            for (i in pr.indices) pureRows[i] = if ((pr[i].toInt() and 0xff) > 127 && hr[i].toInt() == 0) 1 else 0
        }
        val lab = IntArray(w * h); labels.get(0, 0, lab); labels.release()
        val reach = (1.5 * k).toInt()
        fun pureIn(c: Int, y0: Int, y1: Int): Boolean {
            for (y in max(0, y0)..min(h - 1, y1)) if (pureRows[y * w + c].toInt() != 0) return true
            return false
        }
        val hw = ByteArray(w * h); handwriting.get(0, 0, hw)
        val pr = ByteArray(w * h); print.get(0, 0, pr)
        val ov = ByteArray(w * h); overlap.get(0, 0, ov)
        val on = 255.toByte()
        var changed = false
        for (l in 1 until n) {
            val x0 = st[l * 5 + Imgproc.CC_STAT_LEFT]; val y0 = st[l * 5 + Imgproc.CC_STAT_TOP]
            val bw = st[l * 5 + Imgproc.CC_STAT_WIDTH]; val bh = st[l * 5 + Imgproc.CC_STAT_HEIGHT]
            if (bw < MIN_LEN * k || bh > 2 * k) continue
            val top = IntArray(bw) { Int.MAX_VALUE }; val bot = IntArray(bw) { -1 }
            for (y in y0 until y0 + bh) for (x in x0 until x0 + bw) if (lab[y * w + x] == l) {
                val c = x - x0; top[c] = min(top[c], y); bot[c] = max(bot[c], y)
            }
            val ext = ArrayList<Int>(); var up = 0; var dn = 0; var cols = 0
            for (c in 0 until bw) {
                if (bot[c] < 0) continue
                cols++; ext.add(bot[c] - top[c] + 1)
                if (pureIn(x0 + c, top[c] - reach, top[c] - 2)) up++
                if (pureIn(x0 + c, bot[c] + 2, bot[c] + reach)) dn++
            }
            if (cols == 0) continue
            ext.sort()
            if (ext[ext.size / 2] > 0.6 * k) continue                 // thin, column by column
            if (up < 0.4 * cols || dn < 0.4 * cols) continue          // through the letters
            changed = true
            for (c in 0 until bw) {
                if (bot[c] < 0) continue
                val x = x0 + c
                // a printed letter crosses here: print right at the line's top and bottom edge
                val crossed = pureIn(x, top[c] - 2, top[c] - 1) && pureIn(x, bot[c] + 1, bot[c] + 2)
                for (y in top[c]..bot[c]) {
                    val i = y * w + x
                    if (lab[i] != l) continue
                    hw[i] = on
                    if (crossed) { ov[i] = on; pr[i] = on } else { ov[i] = 0; pr[i] = 0 }
                }
            }
        }
        if (changed) { handwriting.put(0, 0, hw); print.put(0, 0, pr); overlap.put(0, 0, ov) }
    }
}
