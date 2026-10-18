package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

/**
 * Very last erase step: specks the erase left where the handwriting was — a few dark pixels of a
 * stroke's rim, JPEG ringing round it — become paper. Only tiny pieces (up to [MAX_AREA_UNITS]
 * stroke units square), only inside the handwriting's zone, and never one touching print.
 * Mirrors SpeckClean in ios/Runner/ImageProcessingOpenCV.mm.
 */
object SpeckClean {
    /** Dev switch (InkHarness `-e speck 0`). */
    @JvmStatic var enabled = true
    private const val MAX_AREA_UNITS = 1.0
    @JvmStatic var report = ""

    /** [dst] the erased page (CV_8UC3), updated in place; [handwriting] / [print] the mask's layers. */
    fun apply(dst: Mat, handwriting: Mat, print: Mat) {
        report = ""
        if (!enabled) return
        val w = dst.cols(); val h = dst.rows()
        val k = InkAnalysis.strokeUnit(w, h)
        val gray = Mat(); Imgproc.cvtColor(dst, gray, Imgproc.COLOR_BGR2GRAY)
        val paperG = Mat(); Imgproc.dilate(gray, paperG, Mat.ones(2 * k + 1, 2 * k + 1, CvType.CV_8U))
        Imgproc.medianBlur(paperG, paperG, 21)
        val ink = Mat(); Core.subtract(paperG, gray, ink)
        Imgproc.threshold(ink, ink, 30.0, 255.0, Imgproc.THRESH_BINARY)
        val zone = Mat(); Imgproc.dilate(handwriting, zone, Mat.ones(2 * k + 1, 2 * k + 1, CvType.CV_8U))
        val nearPrint = Mat(); Imgproc.dilate(print, nearPrint, Mat.ones(3, 3, CvType.CV_8U))
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(ink, labels, stats, cents, 8, CvType.CV_32S)
        val st = IntArray(n * 5); stats.get(0, 0, st)
        val la = IntArray(w * h); labels.get(0, 0, la)
        val zo = ByteArray(w * h); zone.get(0, 0, zo)
        val np = ByteArray(w * h); nearPrint.get(0, 0, np)
        val maxArea = (MAX_AREA_UNITS * k * k).toInt()
        val inZone = IntArray(n); val touchPrint = BooleanArray(n)
        for (i in 0 until w * h) { val l = la[i]; if (l == 0) continue
            if (zo[i].toInt() != 0) inZone[l]++; if (np[i].toInt() != 0) touchPrint[l] = true }
        val clear = BooleanArray(n) { it > 0 && st[it * 5 + 4] <= maxArea && inZone[it] == st[it * 5 + 4] && !touchPrint[it] }
        val paper = Mat(); Imgproc.medianBlur(dst, paper, 21)
        val px = ByteArray(w * h * 3); dst.get(0, 0, px); val pp = ByteArray(w * h * 3); paper.get(0, 0, pp)
        var cleared = 0
        // a speck's own pixels and the 1 px halo round it
        for (i in 0 until w * h) {
            var hit = clear[la[i]]
            if (!hit) { val x = i % w; val y = i / w
                for (dy in -1..1) { for (dx in -1..1) { val xx = x + dx; val yy = y + dy
                    if (xx in 0 until w && yy in 0 until h && clear[la[yy * w + xx]]) hit = true } } }
            if (hit) { for (c in 0..2) px[i * 3 + c] = pp[i * 3 + c]; cleared++ }
        }
        dst.put(0, 0, px)
        listOf(gray, paperG, ink, zone, nearPrint, labels, stats, cents, paper).forEach { it.release() }
        report = "speck: cleared %d specks (%d px)\n".format(clear.count { it }, cleared)
    }
}
