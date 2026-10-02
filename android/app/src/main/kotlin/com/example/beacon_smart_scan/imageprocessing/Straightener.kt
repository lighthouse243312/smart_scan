package com.example.beacon_smart_scan.imageprocessing

import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Straightens a photographed page before handwriting analysis. Mirrors ios/Runner/Straighten.hpp
 * and ml/prototype/straighten.py — keep the three in step.
 *
 * 1. rules: fit the page's printed rules (table borders, writing grids) — horizontal rules' angle
 *    as a function of height, vertical rule pieces' angle as a function of x, since perspective
 *    makes both drift across the page — intersect the outermost into a quad and warp it square.
 *    Fixes rotation AND keystone even with the sheet's edges out of frame.
 * 2. no usable rules: deskew by the angle that levels the print (neutral ink only, so a coloured
 *    pen can't drag it), applied only when clearly better than leaving it.
 * 3. an already-level page (scanner output) is returned unchanged.
 */
object Straightener {
    private const val WORK = 1200
    private const val MAX_DEG = 8.0
    private const val LEVEL_DEG = 0.6

    private data class Line(val px: Double, val py: Double, val dx: Double, val dy: Double, val length: Double, val angle: Double)
    private data class Piece(val x: Double, val angle: Double, val length: Double)

    fun straighten(inputPath: String, outputPath: String): Map<String, Any> {
        val src = ImageIO.readOrThrow(inputPath)
        return releasing(src) {
            val (dst, how) = straighten(src)
            releasing(dst) { ImageIO.writeOrThrow(dst, outputPath) }
            mapOf("outputPath" to outputPath, "method" to how)
        }
    }

    /** The straightened page (a new Mat) and how: "level", "rules" or "rotate". */
    fun straighten(img: Mat): Pair<Mat, String> {
        val s = WORK.toDouble() / max(img.cols(), img.rows())
        val small = Mat()
        Imgproc.resize(img, small, Size(), s, s, Imgproc.INTER_AREA)
        return releasing(small) {
            val quad = rulesQuad(small)
            if (quad != null) {
                val maxSide = maxOf(
                    abs(sideAngle(quad[0], quad[1], true)), abs(sideAngle(quad[3], quad[2], true)),
                    abs(sideAngle(quad[0], quad[3], false)), abs(sideAngle(quad[1], quad[2], false)),
                )
                if (maxSide < LEVEL_DEG) return@releasing Pair(img.clone(), "level")
                val warped = warpKeepPage(img, quad.map { Point(it.x / s, it.y / s) })
                if (warped != null) return@releasing Pair(warped, "rules")
            }
            val ink = inkMap(small, neutralOnly = true)
            releasing(ink) {
                var best = -1.0; var bestA = 0.0
                var a = -MAX_DEG
                while (a <= MAX_DEG + 1e-6) { val sc = projectionScore(ink, a); if (sc > best) { best = sc; bestA = a }; a += 0.25 }
                val coarse = bestA; best = -1.0
                a = coarse - 0.25
                while (a <= coarse + 0.25 + 1e-6) { val sc = projectionScore(ink, a); if (sc > best) { best = sc; bestA = a }; a += 0.05 }
                val conf = best / max(projectionScore(ink, 0.0), 1e-6)
                if (abs(bestA) >= 0.3 && conf >= 1.15) Pair(rotate(img, bestA), "rotate") else Pair(img.clone(), "level")
            }
        }
    }

    private fun inkMap(small: Mat, neutralOnly: Boolean): Mat {
        val g = Mat(); val bg = Mat(); val diff = Mat(); val ink = Mat()
        Imgproc.cvtColor(small, g, Imgproc.COLOR_BGR2GRAY)
        Imgproc.dilate(g, bg, Mat.ones(7, 7, CvType.CV_8U))
        Imgproc.medianBlur(bg, bg, 31)
        Core.subtract(bg, g, diff, Mat(), CvType.CV_16S)
        Core.compare(diff, Scalar(25.0), ink, Core.CMP_GT)
        if (neutralOnly) {
            val lab = Mat(); Imgproc.cvtColor(small, lab, Imgproc.COLOR_BGR2Lab)
            val ch = ArrayList<Mat>(); Core.split(lab, ch)
            val a = Mat(); val b = Mat(); val chroma = Mat(); val neutral = Mat()
            ch[1].convertTo(a, CvType.CV_32F, 1.0, -128.0); ch[2].convertTo(b, CvType.CV_32F, 1.0, -128.0)
            Core.magnitude(a, b, chroma)
            Core.compare(chroma, Scalar(12.0), neutral, Core.CMP_LT)
            Core.bitwise_and(ink, neutral, ink)
            releasing(lab, a, b, chroma, neutral, *ch.toTypedArray()) {}
        }
        releasing(g, bg, diff) {}
        return ink
    }

    /** Least-squares line through component [id] inside its box: (point, direction). */
    private fun fitLine(lab: IntArray, cols: Int, id: Int, x0: Int, y0: Int, w: Int, h: Int): DoubleArray {
        val pts = ArrayList<Point>()
        for (y in y0 until y0 + h) for (x in x0 until x0 + w) if (lab[y * cols + x] == id) pts.add(Point(x.toDouble(), y.toDouble()))
        val m = MatOfPoint2f(*pts.toTypedArray()); val l = Mat()
        Geometry.fitLine(m, l, Geometry.DIST_HUBER, 0.0, 0.01, 0.01)
        val r = DoubleArray(4) { l.get(it, 0)[0] }        // vx, vy, x0, y0
        releasing(m, l) {}
        return r
    }

    /** Long (≥ 35 % of the width), thin, nearly level printed rules. */
    private fun longHorizontals(ink: Mat): List<Line> {
        val w0 = ink.cols()
        val lines = Mat()
        Imgproc.morphologyEx(ink, lines, Imgproc.MORPH_OPEN, Mat.ones(1, max(15, w0 / 25), CvType.CV_8U))
        // re-join a rule the photo's slope or a pen crossing broke into pieces
        Imgproc.morphologyEx(lines, lines, Imgproc.MORPH_CLOSE, Mat.ones(3, 3, CvType.CV_8U))
        Imgproc.dilate(lines, lines, Mat.ones(3, 1, CvType.CV_8U))
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(lines, labels, stats, cents, 8, CvType.CV_32S)
        val lab = IntArray(labels.total().toInt()); labels.get(0, 0, lab)
        val out = ArrayList<Line>()
        for (i in 1 until n) {
            val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt(); val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            if (w < 0.35 * w0 || h > 0.08 * w0) continue
            val f = fitLine(lab, labels.cols(), i, stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt(), stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt(), w, h)
            var angle = Math.toDegrees(atan2(f[1], f[0]))
            angle = ((angle + 90 + 360) % 180.0) - 90
            if (abs(angle) > MAX_DEG) continue
            val norm = hypot(f[0], f[1])
            out.add(Line(f[2], f[3], f[0] / norm, f[1] / norm, w.toDouble(), angle))
        }
        releasing(lines, labels, stats, cents) {}
        return out
    }

    /** Short vertical rule pieces (grid / table columns). */
    private fun shortVerticals(ink: Mat): List<Piece> {
        val h0 = ink.rows(); val w0 = ink.cols()
        val v = Mat()
        Imgproc.morphologyEx(ink, v, Imgproc.MORPH_OPEN, Mat.ones(max(12, h0 / 60), 1, CvType.CV_8U))
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(v, labels, stats, cents, 8, CvType.CV_32S)
        val lab = IntArray(labels.total().toInt()); labels.get(0, 0, lab)
        val out = ArrayList<Piece>()
        for (i in 1 until n) {
            val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt(); val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            if (h < 0.04 * h0 || w > 0.02 * w0) continue
            val f = fitLine(lab, labels.cols(), i, stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt(), stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt(), w, h)
            var dx = f[0]; var dy = f[1]
            if (abs(dy) < abs(dx)) continue
            if (dy < 0) { dx = -dx; dy = -dy }
            val angle = Math.toDegrees(atan2(-dx, dy))          // 0 = vertical
            if (abs(angle) <= MAX_DEG) out.add(Piece(cents.get(i, 0)[0], angle, h.toDouble()))
        }
        releasing(v, labels, stats, cents) {}
        return out
    }

    /**
     * angle = c0 * t + c1 by weighted least squares (weights squared, as numpy.polyfit's `w`);
     * with [linear] false, the length-weighted mean angle.
     */
    private fun fitAngle(t: DoubleArray, a: DoubleArray, w: DoubleArray, linear: Boolean): DoubleArray {
        val sw = w.sum(); val sa = a.indices.sumOf { w[it] * a[it] }
        if (!linear) return doubleArrayOf(0.0, sa / sw)
        val w2 = DoubleArray(w.size) { w[it] * w[it] }
        val s2 = w2.sum()
        val mt = t.indices.sumOf { w2[it] * t[it] } / s2; val ma = a.indices.sumOf { w2[it] * a[it] } / s2
        val num = t.indices.sumOf { w2[it] * (t[it] - mt) * (a[it] - ma) }
        val den = t.indices.sumOf { w2[it] * (t[it] - mt) * (t[it] - mt) }
        val c0 = if (den > 1e-9) num / den else 0.0
        return doubleArrayOf(c0, ma - c0 * mt)
    }

    private fun intersect(p1: Point, d1: Point, p2: Point, d2: Point): Point? {
        val det = d1.x * -d2.y - (-d2.x) * d1.y
        if (abs(det) < 1e-6) return null
        val rx = p2.x - p1.x; val ry = p2.y - p1.y
        val t = (rx * -d2.y - (-d2.x) * ry) / det
        return Point(p1.x + t * d1.x, p1.y + t * d1.y)
    }

    private fun percentile(v: DoubleArray, q: Double): Double {
        val s = v.sorted()
        val idx = q / 100.0 * (s.size - 1)
        val lo = floor(idx).toInt(); val hi = min(lo + 1, s.size - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)
    }

    /** Quad (in [small] coordinates) squaring the page's rule field, or null. */
    private fun rulesQuad(small: Mat): List<Point>? {
        val ink = inkMap(small, neutralOnly = false)
        return releasing(ink) {
            val h0 = ink.rows().toDouble(); val w0 = ink.cols().toDouble()
            val hs = longHorizontals(ink)
            val vs = shortVerticals(ink)
            if (hs.size < 2) return@releasing null
            val hy = DoubleArray(hs.size) { hs[it].py + (w0 / 2 - hs[it].px) * hs[it].dy / hs[it].dx }
            val ha = DoubleArray(hs.size) { hs[it].angle }; val hl = DoubleArray(hs.size) { hs[it].length }
            val yTop = hy.min(); val yBot = hy.max()
            if (yBot - yTop < 0.25 * h0) return@releasing null
            val fh = fitAngle(hy, ha, hl, hs.size >= 3)
            val fv: DoubleArray
            var xL = 0.1 * w0; var xR = 0.9 * w0
            if (vs.size >= 6) {
                val vx = DoubleArray(vs.size) { vs[it].x }; val va = DoubleArray(vs.size) { vs[it].angle }; val vl = DoubleArray(vs.size) { vs[it].length }
                fv = fitAngle(vx, va, vl, vx.max() - vx.min() > 0.3 * w0)
                xL = percentile(vx, 5.0); xR = percentile(vx, 95.0)
            } else {
                // no verticals to read: keep them square to the mean horizontal (rotation only)
                fv = fitAngle(hy, ha, hl, false)
            }
            if (xR - xL < 0.3 * w0) { xL = 0.1 * w0; xR = 0.9 * w0 }
            val corners = listOf(yTop to xL, yTop to xR, yBot to xR, yBot to xL)
            corners.map { (y, x) ->
                val ah = Math.toRadians(fh[0] * y + fh[1]); val av = Math.toRadians(fv[0] * x + fv[1])
                intersect(Point(w0 / 2, y), Point(cos(ah), sin(ah)), Point(x, h0 / 2), Point(-sin(av), cos(av)))
                    ?: return@releasing null
            }
        }
    }

    private fun sideAngle(a: Point, b: Point, horizontal: Boolean): Double {
        val vx = b.x - a.x; val vy = b.y - a.y
        return if (horizontal) Math.toDegrees(atan2(vy, vx)) else Math.toDegrees(atan2(-vx, vy))
    }

    /** Squares [q] with a homography applied to the WHOLE image (nothing outside it cropped); null when degenerate. */
    private fun warpKeepPage(img: Mat, q: List<Point>): Mat? {
        fun dist(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)
        val w = (dist(q[1], q[0]) + dist(q[2], q[3])) / 2
        val h = (dist(q[3], q[0]) + dist(q[2], q[1])) / 2
        val srcQ = MatOfPoint2f(*q.toTypedArray())
        val dstQ = MatOfPoint2f(q[0], Point(q[0].x + w, q[0].y), Point(q[0].x + w, q[0].y + h), Point(q[0].x, q[0].y + h))
        val m = Geometry.getPerspectiveTransform(srcQ, dstQ)
        val frame = MatOfPoint2f(Point(0.0, 0.0), Point(img.cols().toDouble(), 0.0), Point(img.cols().toDouble(), img.rows().toDouble()), Point(0.0, img.rows().toDouble()))
        val warped = MatOfPoint2f()
        Core.perspectiveTransform(frame, warped, m)
        val pts = warped.toArray()
        val x0 = pts.minOf { it.x }; val y0 = pts.minOf { it.y }; val x1 = pts.maxOf { it.x }; val y1 = pts.maxOf { it.y }
        val size = Size(ceil(x1 - x0), ceil(y1 - y0))
        if (size.width > 1.6 * img.cols() || size.height > 1.6 * img.rows()) {
            releasing(srcQ, dstQ, m, frame, warped) {}
            return null
        }
        val t = Mat(3, 3, CvType.CV_64F); t.put(0, 0, 1.0, 0.0, -x0, 0.0, 1.0, -y0, 0.0, 0.0, 1.0)
        val tm = Mat(); Core.gemm(t, m, 1.0, Mat(), 0.0, tm)
        val out = Mat()
        Imgproc.warpPerspective(img, out, tm, size, Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, Scalar(255.0, 255.0, 255.0))
        releasing(srcQ, dstQ, m, frame, warped, t, tm) {}
        return out
    }

    private fun projectionScore(ink: Mat, deg: Double): Double {
        val m = Geometry.getRotationMatrix2D(Point(ink.cols() / 2.0, ink.rows() / 2.0), deg, 1.0)
        val r = Mat(); val prof = Mat()
        Imgproc.warpAffine(ink, r, m, ink.size())
        Core.reduce(r, prof, 1, Core.REDUCE_SUM, CvType.CV_64F)
        val p = DoubleArray(prof.rows()); prof.get(0, 0, p)
        var s = 0.0
        for (i in 1 until p.size) { val d = p[i] - p[i - 1]; s += d * d }
        releasing(m, r, prof) {}
        return s
    }

    private fun rotate(img: Mat, deg: Double): Mat {
        val m = Geometry.getRotationMatrix2D(Point(img.cols() / 2.0, img.rows() / 2.0), deg, 1.0)
        val c = abs(m.get(0, 0)[0]); val s = abs(m.get(0, 1)[0])
        val nW = (img.rows() * s + img.cols() * c).toInt(); val nH = (img.rows() * c + img.cols() * s).toInt()
        m.put(0, 2, m.get(0, 2)[0] + nW / 2.0 - img.cols() / 2.0)
        m.put(1, 2, m.get(1, 2)[0] + nH / 2.0 - img.rows() / 2.0)
        val out = Mat()
        Imgproc.warpAffine(img, out, m, Size(nW.toDouble(), nH.toDouble()), Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, Scalar(255.0, 255.0, 255.0))
        m.release()
        return out
    }
}
