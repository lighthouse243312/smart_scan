package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * The slant of the page's print, to tell pen strokes the colour of the print by. Print on one page
 * stands one way: its stems are all at one angle to its lines (upright, or one italic). A hand
 * leans its own way, and wanders. Angles are taken against the photo's own tilt — the direction the
 * page's print runs in the image, from the edges of all its print — never against the image frame,
 * so a page photographed askew is measured the same as a straight one.
 * Mirrors PrintSlant in ios/Runner/ImageProcessingOpenCV.mm.
 */
object PrintSlant {
    /** Gradient strength (grey levels per pixel, Sobel 3×3) of a stroke's edge. */
    private const val EDGE = 40.0
    /** Edges within this many degrees of a stem's are a stem's (a vertical-ish stroke's sides). */
    private const val STEM = 35.0
    /** Fewest stem-edge pixels to measure a stroke's slant by. */
    private const val MIN_STEM = 8

    /**
     * The page's print: [tilt] the direction its lines run in the image (degrees, its print's edges
     * modulo 90), [slant] its stems' lean against that, [spread] how much its strokes differ.
     */
    class Reference(val tilt: Double, val slant: Double, val spread: Double)

    /** Per-pixel edges of [src] (CV_8UC3): direction (degrees, 0..360) and strength. */
    class Edges(val angle: Mat, val mag: Mat) {
        fun release() { angle.release(); mag.release() }
    }

    fun edges(src: Mat): Edges {
        val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
        val gx = Mat(); val gy = Mat()
        Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0)
        Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1)
        val angle = Mat(); val mag = Mat()
        Core.phase(gx, gy, angle, true)
        Core.magnitude(gx, gy, mag)
        listOf(gray, gx, gy).forEach { it.release() }
        return Edges(angle, mag)
    }

    /** The page's print reference, from its [print] layer (CV_8UC1); null when there is too little print. */
    fun reference(e: Edges, print: Mat, minArea: Int): Reference? {
        val w = print.cols(); val h = print.rows()
        // the photo's tilt: the dominant direction of the print's edges, modulo 90 (text has its
        // baselines and its stems square to each other)
        val near = Mat(); Imgproc.dilate(print, near, Mat.ones(3, 3, CvType.CV_8U))
        var c4 = 0.0; var s4 = 0.0
        val ang = FloatArray(w); val mg = FloatArray(w); val pr = ByteArray(w)
        for (y in 0 until h) {
            e.angle.get(y, 0, ang); e.mag.get(y, 0, mg); near.get(y, 0, pr)
            for (x in 0 until w) {
                if (pr[x].toInt() == 0 || mg[x] < EDGE) continue
                val a = 4 * ang[x] * PI / 180
                c4 += mg[x] * cos(a); s4 += mg[x] * sin(a)
            }
        }
        if (c4 == 0.0 && s4 == 0.0) { near.release(); return null }
        val tilt = atan2(s4, c4) * 180 / PI / 4
        // each printed glyph's stem slant against that tilt
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(near, labels, stats, cents, 8, CvType.CV_32S)
        val area = IntArray(n) { stats.get(it, Imgproc.CC_STAT_AREA)[0].toInt() }
        val slants = slants(e, labels, n, tilt)
        listOf(near, labels, stats, cents).forEach { it.release() }
        val glyphs = (1 until n).filter { area[it] >= minArea && !slants[it].isNaN() }.map { slants[it] }.sorted()
        if (glyphs.size < 20) return null
        val med = glyphs[glyphs.size / 2]
        val mad = glyphs.map { abs(it - med) }.sorted()[glyphs.size / 2]
        return Reference(tilt, med, max(3.0, 1.4826 * mad))
    }

    /**
     * Per label of [labels] (CV_32S, [n] labels): the lean of its stems against [tilt], degrees
     * (positive: leaning right), from the edges of its near-vertical strokes; NaN when it has too
     * few (a round "o", a dot, a dash).
     */
    fun slants(e: Edges, labels: Mat, n: Int, tilt: Double): DoubleArray {
        val w = labels.cols(); val h = labels.rows()
        val sw = DoubleArray(n); val sa = DoubleArray(n); val cnt = IntArray(n)
        val lab = IntArray(w); val ang = FloatArray(w); val mg = FloatArray(w)
        for (y in 0 until h) {
            labels.get(y, 0, lab); e.angle.get(y, 0, ang); e.mag.get(y, 0, mg)
            for (x in 0 until w) {
                val l = lab[x]
                if (l == 0 || mg[x] < EDGE) continue
                // a stem's side has its gradient across the line direction: angle against the tilt,
                // folded to (-90, 90]
                var r = (ang[x] - tilt) % 180.0
                if (r > 90) r -= 180.0
                if (r <= -90) r += 180.0
                if (abs(r) > STEM) continue
                // gradient turned by +r from across-the-line: the stem leans by r (image y runs down,
                // so a positive r leans the top to the right)
                sw[l] += mg[x].toDouble(); sa[l] += mg[x] * r; cnt[l]++
            }
        }
        return DoubleArray(n) { if (cnt[it] >= MIN_STEM && sw[it] > 0) sa[it] / sw[it] else Double.NaN }
    }
}
