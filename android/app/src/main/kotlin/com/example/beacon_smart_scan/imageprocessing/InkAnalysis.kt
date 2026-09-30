package com.example.beacon_smart_scan.imageprocessing

import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Handwriting vs print separation from ink colour + page layout, and the rebuild-erase that uses
 * it. Kotlin port of ios/Runner/InkAnalysis.hpp — both must implement the same steps with the
 * same constants; see that file for why these signals were chosen.
 *
 * Masks are handled as flat BooleanArrays / FloatArrays of width*height, which keeps the many
 * per-pixel steps simple on the JVM; OpenCV is used for filters and components. Those arrays live
 * on the Java heap, so analysis runs at a working resolution of at most [WORK_LONG_SIDE] px (a 12
 * MP photo would need hundreds of MB); masks are scaled back to the page, and the erase's pixel
 * work (print colour, inpaint) still happens at full resolution in native OpenCV memory.
 */
object InkAnalysis {
    private const val INK_OD = 0.30f
    private const val CLIPPED_OD = 1.6f
    private const val FAINT_OD = 0.06f
    private const val RIM_OD = 0.04f
    private const val WORK_LONG_SIDE = 2400
    /** Chroma distance -> score units (a unit conversion, not tied to any ink colour). */
    const val PEN_SCALE = 4f

    class Page(val width: Int, val height: Int) {
        val size = width * height
    }

    private class OpticalDensity(val odB: FloatArray, val odG: FloatArray, val odR: FloatArray, val mean: FloatArray, val ink: BooleanArray)

    /** Stroke-scale unit (≈ a pen stroke's width) for this resolution. */
    fun strokeUnit(width: Int, height: Int): Int = max(5, max(width, height) / 450) or 1

    private fun odd(v: Double): Int = max(1, v.roundToInt()) or 1

    // ---------------------------------------------------------------- Mat <-> array helpers

    private fun toMat(mask: BooleanArray, page: Page): Mat {
        val bytes = ByteArray(page.size) { if (mask[it]) 255.toByte() else 0 }
        val mat = Mat(page.height, page.width, CvType.CV_8UC1)
        mat.put(0, 0, bytes)
        return mat
    }

    private fun toMask(mat: Mat): BooleanArray {
        val bytes = ByteArray(mat.total().toInt())
        mat.get(0, 0, bytes)
        return BooleanArray(bytes.size) { bytes[it].toInt() != 0 }
    }

    private fun toFloatMat(values: FloatArray, page: Page): Mat {
        val mat = Mat(page.height, page.width, CvType.CV_32F)
        mat.put(0, 0, values)
        return mat
    }

    private fun toFloats(mat: Mat): FloatArray {
        val out = FloatArray(mat.total().toInt())
        mat.get(0, 0, out)
        return out
    }

    private fun boxFilter(values: FloatArray, page: Page, window: Int): FloatArray {
        val src = toFloatMat(values, page)
        val dst = Mat()
        return releasing(src, dst) {
            Imgproc.boxFilter(src, dst, -1, Size(window.toDouble(), window.toDouble()))
            toFloats(dst)
        }
    }

    private fun dilate(mask: BooleanArray, page: Page, diameter: Int, ellipse: Boolean): BooleanArray {
        val src = toMat(mask, page)
        val dst = Mat()
        val d = odd(diameter.toDouble()).toDouble()
        val kernel = Imgproc.getStructuringElement(if (ellipse) Imgproc.MORPH_ELLIPSE else Imgproc.MORPH_RECT, Size(d, d))
        return releasing(src, dst, kernel) {
            Imgproc.dilate(src, dst, kernel)
            toMask(dst)
        }
    }

    private class Components(
        val count: Int,
        val labels: IntArray,
        val x: IntArray, val y: IntArray, val w: IntArray, val h: IntArray, val area: IntArray,
    ) {
        val cx = FloatArray(count) { x[it] + w[it] / 2f }
        val bottom = FloatArray(count) { (y[it] + h[it]).toFloat() }
    }

    private fun components(mask: BooleanArray, page: Page): Components {
        val src = toMat(mask, page)
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        return releasing(src, labels, stats, centroids) {
            val count = Imgproc.connectedComponentsWithStats(src, labels, stats, centroids, 8, CvType.CV_32S)
            val lab = IntArray(page.size)
            labels.get(0, 0, lab)
            val st = IntArray(count * 5)
            stats.get(0, 0, st)
            Components(
                count, lab,
                IntArray(count) { st[it * 5 + Imgproc.CC_STAT_LEFT] },
                IntArray(count) { st[it * 5 + Imgproc.CC_STAT_TOP] },
                IntArray(count) { st[it * 5 + Imgproc.CC_STAT_WIDTH] },
                IntArray(count) { st[it * 5 + Imgproc.CC_STAT_HEIGHT] },
                IntArray(count) { st[it * 5 + Imgproc.CC_STAT_AREA] },
            )
        }
    }

    private fun paint(labels: IntArray, flags: BooleanArray): BooleanArray = BooleanArray(labels.size) { flags[labels[it]] }

    private fun fractionPerComponent(c: Components, mask: BooleanArray): FloatArray {
        val sum = FloatArray(c.count)
        for (i in mask.indices) if (mask[i]) sum[c.labels[i]] += 1f
        for (i in 0 until c.count) sum[i] /= max(1, c.area[i]).toFloat()
        return sum
    }

    private class UnionFind(n: Int) {
        private val parent = IntArray(n) { it }
        fun find(a0: Int): Int {
            var a = a0
            while (parent[a] != a) {
                parent[a] = parent[parent[a]]
                a = parent[a]
            }
            return a
        }
        fun join(a: Int, b: Int) {
            parent[find(a)] = find(b)
        }
    }

    private fun percentile(values: List<Float>, p: Float): Float {
        if (values.isEmpty()) return 0f
        val sorted = values.sorted()
        val i = min(sorted.size - 1, (p / 100.0 * (sorted.size - 1)).roundToInt())
        return sorted[i]
    }

    private fun percentile(values: FloatArray, count: Int, p: Float): Float {
        if (count == 0) return 0f
        val sorted = values.copyOf(count).also { it.sort() }
        return sorted[min(count - 1, (p / 100.0 * (count - 1)).roundToInt())]
    }

    // ---------------------------------------------------------------- analysis steps

    private fun opticalDensity(bgr: Mat, page: Page): OpticalDensity {
        val small = Mat()
        val paper = Mat()
        val image = Mat()
        val t = Mat()
        val kernel = Mat.ones(9, 9, CvType.CV_8U)
        return releasing(small, paper, image, t, kernel) {
            Imgproc.resize(bgr, small, Size(), 0.125, 0.125, Imgproc.INTER_AREA)
            Imgproc.medianBlur(small, small, 15)
            Imgproc.dilate(small, small, kernel)
            Imgproc.resize(small, paper, bgr.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            paper.convertTo(paper, CvType.CV_32FC3)
            Imgproc.GaussianBlur(paper, paper, Size(0.0, 0.0), 8.0)
            bgr.convertTo(image, CvType.CV_32FC3)
            Core.add(image, Scalar(1.0, 1.0, 1.0), image)
            Core.add(paper, Scalar(1.0, 1.0, 1.0), paper)
            Core.divide(image, paper, t)
            val v = FloatArray(page.size * 3)
            t.get(0, 0, v)
            val odB = FloatArray(page.size)
            val odG = FloatArray(page.size)
            val odR = FloatArray(page.size)
            val mean = FloatArray(page.size)
            val ink = BooleanArray(page.size)
            for (i in 0 until page.size) {
                val b = -kotlin.math.ln(v[i * 3].coerceIn(1e-3f, 1f))
                val g = -kotlin.math.ln(v[i * 3 + 1].coerceIn(1e-3f, 1f))
                val r = -kotlin.math.ln(v[i * 3 + 2].coerceIn(1e-3f, 1f))
                odB[i] = b; odG[i] = g; odR[i] = r
                mean[i] = (b + g + r) / 3f
                ink[i] = mean[i] > INK_OD
            }
            OpticalDensity(odB, odG, odR, mean, ink)
        }
    }

    /**
     * Local ink CHROMA (OD_B, OD_G, OD_R) / sum over ink in (minOD, maxOD] — sees every hue (a
     * purple pen has the same B/R as black toner). NaN where there is too little ink.
     */
    private fun inkChroma(d: OpticalDensity, page: Page, minOD: Float, maxOD: Float, window: Int, minFill: Float): Array<FloatArray> {
        val w = FloatArray(page.size) { if (d.mean[it] > minOD && d.mean[it] <= maxOD) 1f else 0f }
        val sb = boxFilter(FloatArray(page.size) { d.odB[it] * w[it] }, page, window)
        val sg = boxFilter(FloatArray(page.size) { d.odG[it] * w[it] }, page, window)
        val sr = boxFilter(FloatArray(page.size) { d.odR[it] * w[it] }, page, window)
        val sw = boxFilter(w, page, window)
        val cb = FloatArray(page.size); val cg = FloatArray(page.size); val cr = FloatArray(page.size)
        for (i in 0 until page.size) {
            val sum = sb[i] + sg[i] + sr[i]
            if (sw[i] > minFill && sum > 1e-3f) { cb[i] = sb[i] / sum; cg[i] = sg[i] / sum; cr[i] = sr[i] / sum }
            else { cb[i] = Float.NaN; cg[i] = Float.NaN; cr[i] = Float.NaN }
        }
        return arrayOf(cb, cg, cr)
    }

    /** Pen-likeness in the pipeline's score units: 1 = print colour, lower = towards the pen hue. */
    private fun penScore(chroma: Array<FloatArray>, ref: Array<FloatArray>, dir: FloatArray): FloatArray =
        FloatArray(chroma[0].size) {
            if (chroma[0][it].isNaN()) Float.NaN
            else 1f - PEN_SCALE * ((chroma[0][it] - ref[0][it]) * dir[0] + (chroma[1][it] - ref[1][it]) * dir[1] + (chroma[2][it] - ref[2][it]) * dir[2])
        }

    /**
     * The page's pen hue: direction of other ink's chroma deviation from the local print, judged
     * per ink stroke (lens fringes cancel over a stroke; the ink's own hue does not) and taken as
     * the dominant direction of a hue histogram. No ink colour is assumed. See ios InkAnalysis.hpp.
     */
    private fun penDirection(c: Components, regular: BooleanArray, d: OpticalDensity, chroma: Array<FloatArray>, ref: Array<FloatArray>, page: Page, k: Int): FloatArray {
        val bins = 72
        val hist = DoubleArray(bins)
        val e1 = doubleArrayOf(0.7071, 0.0, -0.7071)
        val e2 = doubleArrayOf(-0.4082, 0.8165, -0.4082)
        val sum = Array(3) { DoubleArray(c.count) }
        val cnt = IntArray(c.count)
        for (i in 0 until page.size) {
            val l = c.labels[i]
            if (!d.ink[i] || l == 0 || regular[l] || chroma[0][i].isNaN()) continue
            for (ch in 0 until 3) sum[ch][l] += (chroma[ch][i] - ref[ch][i]).toDouble()
            cnt[l]++
        }
        val maxArea = (0.002 * page.size).toInt()
        for (l in 1 until c.count) {
            if (cnt[l] < 2 * k || c.area[l] > maxArea) continue
            val dv = DoubleArray(3) { sum[it][l] / cnt[l] }
            val u = dv[0] * e1[0] + dv[1] * e1[1] + dv[2] * e1[2]
            val v = dv[0] * e2[0] + dv[1] * e2[1] + dv[2] * e2[2]
            val m = kotlin.math.sqrt(u * u + v * v)
            if (m < 0.006) continue
            val bin = (kotlin.math.floor((kotlin.math.atan2(v, u) + Math.PI) / (2 * Math.PI) * bins).toInt()) % bins
            hist[bin] += cnt[l] * min(m, 0.05)
        }
        var best = -1; var bestVal = 0.0
        for (i in 0 until bins) {
            var v = 0.0
            for (o in -2..2) v += hist[(i + o + bins) % bins]
            if (v > bestVal) { bestVal = v; best = i }
        }
        if (best < 0 || bestVal <= 0.5) return floatArrayOf(-0.7071f, 0f, 0.7071f)   // fallback only
        val ang = (best + 0.5) / bins * 2 * Math.PI - Math.PI
        return FloatArray(3) { (e1[it] * kotlin.math.cos(ang) + e2[it] * kotlin.math.sin(ang)).toFloat() }
    }

    private class TextLine(val members: IntArray, val medianHeight: Float)

    /**
     * Uniformity review (see ios DetectAtWorkingSize): a typeset page repeats its glyphs almost
     * exactly, a hand never does — a short "regular line" whose glyphs have no twin in any OTHER line
     * is neat handwriting that happened to be even and straight, not print.
     */
    private fun untwinnedLinesRemoved(lines: List<TextLine>, c: Components, page: Page): List<TextLine> {
        val lineOf = IntArray(c.count) { -1 }
        for ((li, l) in lines.withIndex()) for (i in l.members) lineOf[i] = li
        val all = (1 until c.count).filter { lineOf[it] >= 0 }.sortedBy { c.h[it] }
        val heights = all.map { c.h[it] }
        fun overlapIoU(i: Int, j: Int): Float {
            val w = c.w[i]; val h = c.h[i]
            var inter = 0; var uni = 0
            for (yy in 0 until h) for (xx in 0 until w) {
                val a = c.labels[(c.y[i] + yy) * page.width + c.x[i] + xx] == i
                val gx = c.x[j] + xx * c.w[j] / w; val gy = c.y[j] + yy * c.h[j] / h
                val b = c.labels[gy * page.width + gx] == j
                if (a && b) inter++
                if (a || b) uni++
            }
            return if (uni == 0) 0f else inter.toFloat() / uni
        }
        return lines.filterIndexed { li, l ->
            if (l.members.size > 40) return@filterIndexed true
            var withTwin = 0; var tested = 0
            for (i in l.members) {
                if (c.h[i] < 4) continue
                tested++
                val lo = heights.binarySearch(floor(c.h[i] * 0.88).toInt()).let { if (it < 0) -it - 1 else it }
                var k = lo
                while (k > 0 && heights[k - 1] >= floor(c.h[i] * 0.88).toInt()) k--
                var twin = false
                while (k < all.size && c.h[all[k]] <= c.h[i] * 1.14f && !twin) {
                    val j = all[k]; k++
                    if (lineOf[j] == li) continue
                    val rw = c.w[j].toFloat() / c.w[i]
                    if (rw < 0.85f || rw > 1.18f) continue
                    twin = overlapIoU(i, j) >= 0.7f
                }
                if (twin) withTwin++
            }
            !(tested >= 3 && withTwin < 0.3f * tested)
        }
    }

    /**
     * A printed line the pen wrote across breaks into pieces: its letters fused with the pen no
     * longer sit on the baseline, and the clean glyphs between them are too few to chain into a line
     * of their own. Two pieces on ONE baseline with the same text height are that line: the glyphs
     * between them on the baseline join it — only those with a twin among the printed glyphs, so an
     * answer written in a blank of the line stays out (see ios DetectAtWorkingSize).
     */
    private fun linesBridged(lines: List<TextLine>, c: Components, unit: Int, page: Page): List<TextLine> {
        class Fit(val a: Double, val b: Double, val x0: Int, val x1: Int, val h: Float)
        val inLine = BooleanArray(c.count)
        val fits = lines.map { l ->
            var sx = 0.0; var sy = 0.0; var sxx = 0.0; var sxy = 0.0
            var x0 = Int.MAX_VALUE; var x1 = Int.MIN_VALUE
            for (i in l.members) {
                sx += c.cx[i]; sy += c.bottom[i]; sxx += c.cx[i].toDouble() * c.cx[i]; sxy += c.cx[i].toDouble() * c.bottom[i]
                x0 = min(x0, c.x[i]); x1 = max(x1, c.x[i] + c.w[i])
                inLine[i] = true
            }
            val n = l.members.size.toDouble(); val den = n * sxx - sx * sx
            val a = if (den != 0.0) (n * sxy - sx * sy) / den else 0.0
            Fit(a, (sy - a * sx) / n, x0, x1, l.medianHeight)
        }
        val pool = (1 until c.count).filter { inLine[it] }.sortedBy { c.h[it] }
        val heights = pool.map { c.h[it] }
        fun overlapIoU(i: Int, j: Int): Float {
            val w = c.w[i]; val h = c.h[i]
            var inter = 0; var uni = 0
            for (yy in 0 until h) for (xx in 0 until w) {
                val a = c.labels[(c.y[i] + yy) * page.width + c.x[i] + xx] == i
                val b = c.labels[(c.y[j] + yy * c.h[j] / h) * page.width + c.x[j] + xx * c.w[j] / w] == j
                if (a && b) inter++
                if (a || b) uni++
            }
            return if (uni == 0) 0f else inter.toFloat() / uni
        }
        fun hasTwin(i: Int): Boolean {
            val lo = floor(c.h[i] * 0.88).toInt()
            var k = heights.binarySearch(lo).let { if (it < 0) -it - 1 else it }
            while (k > 0 && heights[k - 1] >= lo) k--
            while (k < pool.size && c.h[pool[k]] <= c.h[i] * 1.14f) {
                val j = pool[k]; k++
                val rw = c.w[j].toFloat() / c.w[i]
                if (rw < 0.85f || rw > 1.18f) continue
                if (overlapIoU(i, j) >= 0.7f) return true
            }
            return false
        }
        val loose = (1 until c.count).filter { !inLine[it] && isGlyph(c, it, unit, page.height) }
        val added = List(lines.size) { ArrayList<Int>() }
        for ((p, P) in fits.withIndex()) {
            // the nearest piece to its right on the same baseline, of the same text height
            // (compared where the pieces end, not extrapolated: a fitted slope is noisy)
            var best = -1
            for ((q, Q) in fits.withIndex()) {
                if (q == p || Q.x0 <= P.x1 || abs(Q.h - P.h) > 0.2f * max(Q.h, P.h)) continue
                val hm = max(P.h, Q.h).toDouble()
                if (abs((P.a * P.x1 + P.b) - (Q.a * Q.x0 + Q.b)) >= 0.18 * hm) continue
                if (best < 0 || Q.x0 < fits[best].x0) best = q
            }
            if (best < 0) continue
            val Q = fits[best]
            val hm = max(P.h, Q.h)
            for (i in loose) {
                if (inLine[i] || c.cx[i] <= P.x1 || c.cx[i] >= Q.x0) continue
                // (the baseline straight from one piece's end to the other's)
                val f = (c.cx[i] - P.x1) / (Q.x0 - P.x1).toDouble()
                val base = (1 - f) * (P.a * P.x1 + P.b) + f * (Q.a * Q.x0 + Q.b)
                val ratio = c.h[i] / hm
                if (abs(c.bottom[i] - base) >= 0.18 * hm || ratio <= 0.6f || ratio >= 1.67f || !hasTwin(i)) continue
                added[p].add(i)
                inLine[i] = true
            }
        }
        return lines.mapIndexed { i, l -> if (added[i].isEmpty()) l else TextLine(l.members + added[i].toIntArray(), l.medianHeight) }
    }

    private fun isGlyph(c: Components, i: Int, unit: Int, pageHeight: Int): Boolean =
        c.area[i] >= unit * 2 && c.h[i] >= unit && c.h[i] <= pageHeight * 0.03 && c.w[i] <= c.h[i] * 4

    private fun regularLines(c: Components, unit: Int, pageHeight: Int): List<TextLine> {
        val glyphs = (1 until c.count).filter { isGlyph(c, it, unit, pageHeight) }.sortedBy { c.cx[it] }
        val uf = UnionFind(c.count)
        for (a in glyphs.indices) {
            val i = glyphs[a]
            var b = a + 1
            while (b < glyphs.size && c.cx[glyphs[b]] - c.cx[i] < 3f * c.h[i]) {
                val o = glyphs[b]
                val hmax = max(c.h[i], c.h[o]).toFloat()
                val ratio = c.h[o].toFloat() / c.h[i]
                if (abs(c.bottom[o] - c.bottom[i]) < 0.18f * hmax && ratio > 0.6f && ratio < 1.67f) uf.join(o, i)
                b++
            }
        }
        val lines = ArrayList<TextLine>()
        for (g in glyphs.groupBy { uf.find(it) }.values) {
            if (g.size < 5) continue
            var sx = 0.0; var sy = 0.0; var sxx = 0.0; var sxy = 0.0
            for (i in g) { sx += c.cx[i]; sy += c.bottom[i]; sxx += c.cx[i].toDouble() * c.cx[i]; sxy += c.cx[i].toDouble() * c.bottom[i] }
            val n = g.size.toDouble(); val den = n * sxx - sx * sx
            val a = if (den != 0.0) (n * sxy - sx * sy) / den else 0.0
            val b = (sy - a * sx) / n
            val resid = g.map { abs(a * c.cx[it] + b - c.bottom[it]).toFloat() }
            val heights = g.map { c.h[it].toFloat() }
            val hm = percentile(heights, 50f)
            if (percentile(resid, 70f) < 0.08f * hm && percentile(heights.map { abs(it - hm) }, 50f) < 0.2f * hm) {
                lines.add(TextLine(g.toIntArray(), hm))
            }
        }
        return lines
    }

    /** Print-colour reference per page region from the regular-line pixels' ink colour. */
    private fun localPrintReference(ratio: FloatArray, regularPixels: BooleanArray, page: Page): FloatArray {
        val cell = max(64, max(page.width, page.height) / 12)
        val gh = (page.height + cell - 1) / cell
        val gw = (page.width + cell - 1) / cell
        val samples = Array(gh * gw) { FloatArray(256) }
        val counts = IntArray(gh * gw)
        for (y in 0 until page.height) {
            val cellRow = (y / cell) * gw
            for (x in 0 until page.width) {
                val i = y * page.width + x
                val r = ratio[i]
                if (!regularPixels[i] || r.isNaN()) continue
                val c = cellRow + x / cell
                if (counts[c] == samples[c].size) samples[c] = samples[c].copyOf(samples[c].size * 2)
                samples[c][counts[c]++] = r
            }
        }
        val grid = FloatArray(gh * gw) { if (counts[it] >= 200) percentile(samples[it], counts[it], 50f) else Float.NaN }
        val known = grid.filter { !it.isNaN() }
        val global = if (known.isEmpty()) 1f else percentile(known, 50f)
        var filled = FloatArray(gh * gw) { if (grid[it].isNaN()) global else grid[it] }
        repeat(3) {
            val next = filled.copyOf()
            for (y in 0 until gh) for (x in 0 until gw) {
                if (!grid[y * gw + x].isNaN()) continue
                var sum = 0f; var cnt = 0
                for (dy in -1..1) for (dx in -1..1) {
                    val yy = y + dy; val xx = x + dx
                    if (yy < 0 || xx < 0 || yy >= gh || xx >= gw) continue
                    val v = grid[yy * gw + xx]
                    if (!v.isNaN()) { sum += v; cnt++ }
                }
                if (cnt > 0) next[y * gw + x] = sum / cnt
            }
            filled = next
        }
        val small = Mat(gh, gw, CvType.CV_32F)
        val full = Mat()
        return releasing(small, full) {
            small.put(0, 0, filled)
            Imgproc.GaussianBlur(small, small, Size(0.0, 0.0), 1.0)
            Imgproc.resize(small, full, Size(page.width.toDouble(), page.height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            toFloats(full)
        }
    }

    private fun growWithin(seeds: BooleanArray, allowed: BooleanArray, page: Page): BooleanArray {
        val c = components(allowed, page)
        val seeded = BooleanArray(c.count)
        for (i in seeds.indices) if (seeds[i] && c.labels[i] > 0) seeded[c.labels[i]] = true
        return paint(c.labels, seeded)
    }

    private fun removeSpecks(mask: BooleanArray, page: Page, minArea: Int): BooleanArray {
        val c = components(mask, page)
        val keep = BooleanArray(c.count) { it > 0 && c.area[it] >= minArea }
        return paint(c.labels, keep)
    }

    /**
     * Detects handwriting and the print layer the erase must protect/restore from colour +
     * layout. Returns (handwriting, print) as CV_8UC1 Mats (255 = set); caller releases them.
     * [colorDelta]: how much bluer than the local print a stroke must be (smaller = more sensitive).
     */
    /**
     * Returns (handwriting, print, overlap) at [full]'s size: layout/colour context at working
     * size, then per-pixel refinement at full resolution ([InkRefine.refine]) — downscaling blends
     * thin strokes' edges and destroys the colour signal (measured 96% → 74% per-pixel accuracy).
     */
    /**
     * [modelPrint] / [modelHw]: the segmentation model's probabilities at [full]'s size (CV_32F), or
     * null — a second opinion only where colour cannot decide (see ios DetectByInkColor).
     */
    fun detectByInkColor(full: Mat, colorDelta: Double, modelPrint: Mat? = null, modelHw: Mat? = null): Triple<Mat, Mat, Mat> {
        val src = toWorkingSize(full)
        val r = try {
            val hwWork = modelHw?.let { m ->
                val t = Mat(); Imgproc.resize(m, t, src.size(), 0.0, 0.0, Imgproc.INTER_AREA)
                FloatArray(t.total().toInt()).also { t.get(0, 0, it); t.release() }
            }
            detectAtWorkingSize(src, colorDelta, hwWork)
        } finally {
            if (src !== full) src.release()
        }
        val coarse = upscale(r.hw, full)
        val coarsePrint = upscale(r.print, full)
        val fragments = upscale(r.fragments, full)
        val mixed = upscale(r.mixed, full)
        val band = upscale(r.printBand, full)
        val enclosed = upscale(r.enclosedPrint, full)
        val ring = upscale(r.ringPen, full)
        val colourless = upscale(r.colourless, full)
        return releasing(coarse, coarsePrint, r.ref, fragments, mixed, band, enclosed, ring, colourless) {
            InkRefine.refine(full, coarse, coarsePrint, r.ref, r.penDir, fragments, mixed, band, enclosed, ring, colourless, modelPrint, modelHw)
        }
    }

    private fun toWorkingSize(full: Mat): Mat {
        val longSide = max(full.cols(), full.rows())
        if (longSide <= WORK_LONG_SIDE) return full
        val scale = WORK_LONG_SIDE.toDouble() / longSide
        val out = Mat()
        Imgproc.resize(full, out, Size(), scale, scale, Imgproc.INTER_AREA)
        return out
    }

    /** Scales a working-size mask to [like]'s size (nearest), releasing [mask] if it was scaled. */
    private fun upscale(mask: Mat, like: Mat): Mat {
        if (mask.size() == like.size()) return mask
        val out = Mat()
        Imgproc.resize(mask, out, like.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
        mask.release()
        return out
    }

    private class WorkResult(
        val hw: Mat, val print: Mat, val ref: Mat, val penDir: FloatArray, val fragments: Mat, val mixed: Mat, val printBand: Mat,
        val enclosedPrint: Mat, val ringPen: Mat, val colourless: Mat,
    )

    /**
     * Handwriting with NO colour of its own (black ballpoint, or a scan app that made the page
     * almost greyscale), found by layout and shape and verified by the segmentation model — see
     * ColourlessHandwriting in ios/Runner/InkAnalysis.hpp for the measurements behind each gate.
     * [modelHw]: the model's handwriting probability at working size, or null (then nothing).
     */
    private fun colourlessHandwriting(d: OpticalDensity, c: Components, regular: BooleanArray, hw: BooleanArray,
                                      modelHw: FloatArray?, page: Page): BooleanArray {
        val W = page.width; val H = page.height
        val found = BooleanArray(page.size)
        if (modelHw == null) return found
        val regIds = (1 until c.count).filter { regular[it] }
        if (regIds.size < 20) return found
        fun med(v: List<Float>) = v.sorted()[v.size / 2]
        val cap = med(regIds.map { c.h[it].toFloat() })

        class RLine(val m: ArrayList<Int>, var cy: Float) { var top = 0f; var bot = 0f; val runs = ArrayList<FloatArray>() }
        val lines = ArrayList<RLine>()
        for (i in regIds.sortedBy { c.y[it] + c.h[it] / 2f }) {
            val cy = c.y[i] + c.h[i] / 2f
            val L = lines.firstOrNull { abs(cy - it.cy) < 0.5f * cap }
            if (L != null) { L.m.add(i); L.cy = med(L.m.map { j -> c.y[j] + c.h[j] / 2f }) } else lines.add(RLine(arrayListOf(i), cy))
        }
        for (L in lines) {
            val xs = L.m.map { intArrayOf(c.x[it], c.x[it] + c.w[it]) }.sortedBy { it[0] }
            var a = xs[0][0].toFloat(); var b = xs[0][1].toFloat()
            for (q in 1 until xs.size) {
                if (xs[q][0] - b > 1.5f * cap) { L.runs.add(floatArrayOf(a, b)); a = xs[q][0].toFloat(); b = xs[q][1].toFloat() }
                else b = max(b, xs[q][1].toFloat())
            }
            L.runs.add(floatArrayOf(a, b))
            L.top = med(L.m.map { c.y[it].toFloat() }); L.bot = med(L.m.map { (c.y[it] + c.h[it]).toFloat() })
        }
        // rules / table grid: long straight runs
        val lr = max(9, (3 * cap).toInt()) or 1
        val inkMat = toMat(d.ink, page)
        val hR = Mat(); val vR = Mat(); val rulesMat = Mat()
        Imgproc.morphologyEx(inkMat, hR, Imgproc.MORPH_OPEN, Mat.ones(1, lr, CvType.CV_8U))
        Imgproc.morphologyEx(inkMat, vR, Imgproc.MORPH_OPEN, Mat.ones(lr, 1, CvType.CV_8U))
        Core.bitwise_or(hR, vR, rulesMat)
        Imgproc.dilate(rulesMat, rulesMat, Mat.ones(3, 3, CvType.CV_8U))
        val rules = toMask(rulesMat)
        releasing(inkMat, hR, vR, rulesMat) {}
        val cc = components(BooleanArray(page.size) { d.ink[it] && !rules[it] && !regular[c.labels[it]] }, page)

        val regDark = ArrayList<Float>()
        var idx = 0
        while (idx < page.size) { if (regular[c.labels[idx]]) regDark.add(d.mean[idx]); idx += 3 }
        if (regDark.isEmpty()) return found
        val sortedDark = regDark.sorted()
        val printDark = sortedDark[sortedDark.size / 2]
        val coreDark = sortedDark[min(sortedDark.size - 1, (0.8 * (sortedDark.size - 1)).roundToInt())]

        // per-component pixel lists of the candidates
        val pixels = HashMap<Int, ArrayList<Int>>()
        for (i in 0 until page.size) { val l = cc.labels[i]; if (l > 0) pixels.getOrPut(l) { ArrayList() }.add(i) }
        class Cand(val i: Int, val x: Int, val y: Int, val w: Int, val h: Int, val twin: Float, val dark: Float, val model: Float, val n: Int)
        val cands = ArrayList<Cand>()
        for (i in 1 until cc.count) {
            val x = cc.x[i]; val y = cc.y[i]; val w = cc.w[i]; val h = cc.h[i]
            if (cc.area[i] < 0.3f * cap || h < 0.5f * cap || h > 3.5f * cap || w > 8 * cap) continue
            val px = pixels[i] ?: continue
            val hwShare = px.count { hw[it] }.toFloat() / max(1, cc.area[i])
            // a piece only partly handwriting may be a pen stroke fused with a printed letter (colour
            // splits it): skipped when it sits beside printed text and the model sees its other part as
            // print (see ios ColourlessHandwriting)
            if (hwShare > 0.05f && hwShare < 0.9f) {
                val rest = px.filter { !hw[it] }
                val besideText = regIds.any { j ->
                    val gap = max(c.x[j], x) - min(c.x[j] + c.w[j], x + w)
                    val oy = min(c.y[j] + c.h[j], y + h) - max(c.y[j], y)
                    gap <= 0.7f * cap && oy > 0.3f * min(c.h[j], h)
                }
                if (besideText && rest.isNotEmpty() && rest.sumOf { modelHw[it].toDouble() } / rest.size < 0.4) continue
            }
            val cy = y + h / 2f
            val inLine = lines.any { L ->
                cy >= L.top - 0.35f * cap && cy <= L.bot + 0.35f * cap && h <= 1.7f * (L.bot - L.top + 1) &&
                    L.runs.any { rn -> rn[0] - 0.7f * cap <= x + w && x <= rn[1] + 0.7f * cap }
            }
            if (inLine) continue
            // best twin among the regular printed glyphs of about the same size
            val B = BooleanArray(w * h)
            for (p in px) B[(p / W - y) * w + (p % W - x)] = true
            var best = 0f
            for (j in regIds) {
                val rh = c.h[j].toFloat() / h; val rw = c.w[j].toFloat() / w
                if (rh < 0.85f || rh > 1.18f || rw < 0.8f || rw > 1.25f) continue
                var inter = 0; var uni = 0
                for (yy in 0 until h) for (xx in 0 until w) {
                    val gx = c.x[j] + xx * c.w[j] / w; val gy = c.y[j] + yy * c.h[j] / h
                    val g = c.labels[gy * W + gx] == j
                    val bb = B[yy * w + xx]
                    if (g && bb) inter++
                    if (g || bb) uni++
                }
                if (uni > 0) best = max(best, inter.toFloat() / uni)
                if (best >= 0.95f) break
            }
            var dark = 0f; var model = 0f
            for (p in px) { dark += d.mean[p]; model += modelHw[p] }
            cands.add(Cand(i, x, y, w, h, best, dark / px.size, model / px.size, px.size))
        }
        // groups: candidates side by side on a row decide together
        val uf = UnionFind(cands.size)
        for (a in cands.indices) for (b in a + 1 until cands.size) {
            val A = cands[a]; val Bc = cands[b]
            val gapx = max(A.x, Bc.x) - min(A.x + A.w, Bc.x + Bc.w)
            val oy = min(A.y + A.h, Bc.y + Bc.h) - max(A.y, Bc.y)
            if (gapx <= 0.8f * cap && oy > 0.3f * min(A.h, Bc.h)) uf.join(a, b)
        }
        val groups = cands.indices.groupBy { uf.find(it) }
        for (G in groups.values) {
            val twins = G.count { cands[it].twin >= 0.7f }
            val maxTwin = G.maxOf { cands[it].twin }
            val darks = G.map { cands[it].dark }
            val modelN = G.sumOf { cands[it].n }
            val modelMean = if (modelN == 0) 0.0 else G.sumOf { (cands[it].model * cands[it].n).toDouble() } / modelN
            if (twins >= 0.5f * G.size) continue
            if (med(darks) < 0.4f * printDark) continue
            if (modelMean < 0.03) {
                if (G.size < 2 || maxTwin >= 0.55f || darks.average() >= 0.8f * coreDark) continue
            }
            for (a in G) for (p in pixels[cands[a].i]!!) found[p] = true
        }
        return found
    }

    private fun detectAtWorkingSize(src: Mat, colorDelta: Double, modelHw: FloatArray? = null): WorkResult {
        val page = Page(src.cols(), src.rows())
        val k = strokeUnit(page.width, page.height)
        val d = opticalDensity(src, page)
        // colour window: wide enough to average out sensor noise (a stroke-width window was too
        // noisy at the 2400 px working size — measured: whole printed words turned pen-coloured)
        val colorWindow = max(k, max(page.width, page.height) / 270) or 1
        val chroma = inkChroma(d, page, INK_OD, CLIPPED_OD, colorWindow, 0.05f)

        // regular lines by geometry, then the local print-colour (chroma) reference they give, and
        // the hue direction of the page's other ink (the pen)
        val c = components(d.ink, page)
        val lines = linesBridged(untwinnedLinesRemoved(regularLines(c, k, page.height), c, page), c, k, page)
        val regular = BooleanArray(c.count)
        for (l in lines) for (i in l.members) regular[i] = true
        val regularMask = paint(c.labels, regular)
        val refChroma = Array(3) { localPrintReference(chroma[it], regularMask, page) }
        val penDir = penDirection(c, regular, d, chroma, refChroma, page, k)
        val ratio = penScore(chroma, refChroma, penDir)
        val ref = FloatArray(page.size) { 1f }

        // pen candidates: ink noticeably bluer than the print around it
        val delta = colorDelta.toFloat()
        val relative = FloatArray(page.size) { if (ratio[it].isNaN()) Float.NaN else ratio[it] - ref[it] }
        val cand = BooleanArray(page.size) { d.ink[it] && !relative[it].isNaN() && relative[it] < -delta }
        val fracC = fractionPerComponent(c, cand)

        // line vote: a regular line whose glyphs are mostly print-coloured is print, plus the
        // dots/accents/punctuation inside its band
        val printComp = BooleanArray(c.count)
        val small = BooleanArray(c.count) { it > 0 && c.area[it] < k * k * 4 }
        // line consensus per character: a glyph whose neighbours on its (loose) text line are all
        // print is print too, unless it is itself strongly pen-coloured — merged letters can
        // break a line's regularity, but not its colour context
        run {
            val ids = (1 until c.count).filter { isGlyph(c, it, k, page.height) }.sortedBy { c.cx[it] }
            val uf = UnionFind(c.count)
            for (a in ids.indices) {
                val i = ids[a]
                val reach = 2.5f * max(c.h[i], 3 * k)
                var b = a + 1
                while (b < ids.size && c.cx[ids[b]] - c.cx[i] < reach) {
                    val o = ids[b]
                    val ci = c.y[i] + c.h[i] / 2f
                    val co = c.y[o] + c.h[o] / 2f
                    val hr = c.h[o].toFloat() / c.h[i]
                    if (abs(co - ci) < 0.5f * max(c.h[i], c.h[o]) && hr > 0.5f && hr < 2f) uf.join(o, i)
                    b++
                }
            }
            for (g in ids.groupBy { uf.find(it) }.values) {
                if (g.size < 4) continue   // already sorted by cx (ids were)
                for (p in g.indices) {
                    var pen = 0.0; var total = 0.0
                    for (q in max(0, p - 3)..min(g.size - 1, p + 3)) {
                        if (q == p) continue
                        pen += fracC[g[q]] * c.area[g[q]]; total += c.area[g[q]]
                    }
                    if (total > 0 && pen / total < 0.2 && fracC[g[p]] < 0.85f) printComp[g[p]] = true
                }
            }
        }
        for (l in lines) {
            // regularity alone cannot tell (neat handwriting lines are just as straight —
            // measured); the line's colour decides: pen lines ≈0.98 pen-coloured, print ≤0.45
            if (percentile(l.members.map { fracC[it] }, 50f) >= 0.6f) continue
            var x0 = Float.MAX_VALUE; var y0 = Float.MAX_VALUE; var x1 = -Float.MAX_VALUE; var y1 = -Float.MAX_VALUE
            for (i in l.members) {
                printComp[i] = true
                x0 = min(x0, c.x[i].toFloat()); x1 = max(x1, (c.x[i] + c.w[i]).toFloat())
                y0 = min(y0, c.y[i].toFloat()); y1 = max(y1, c.bottom[i])
            }
            y0 -= 0.6f * l.medianHeight; y1 += 0.3f * l.medianHeight
            for (i in 1 until c.count) {
                if (small[i] && c.cx[i] >= x0 && c.cx[i] <= x1 && c.y[i] >= y0 && c.y[i] + c.h[i] <= y1) printComp[i] = true
            }
        }
        // print-line bands spanning the text column (see ios InkAnalysis.hpp): where pen crosses
        // printed words, non-pen-coloured pixels inside them are restored as print
        val printBand = BooleanArray(page.size)
        run {
            val rows = lines.map { l ->
                var x0 = Float.MAX_VALUE; var y0 = Float.MAX_VALUE; var x1 = -Float.MAX_VALUE; var y1 = -Float.MAX_VALUE
                for (i in l.members) {
                    x0 = min(x0, c.x[i].toFloat()); x1 = max(x1, (c.x[i] + c.w[i]).toFloat())
                    y0 = min(y0, c.y[i].toFloat()); y1 = max(y1, c.bottom[i])
                }
                floatArrayOf(y0 - 0.2f * l.medianHeight, y1 + 0.2f * l.medianHeight, x0, x1)
            }
            if (rows.isNotEmpty()) {
                val colL = percentile(rows.map { it[2] }, 5f)
                val colR = percentile(rows.map { it[3] }, 95f)
                for (r in rows) {
                    val ya = max(0, r[0].toInt()); val yb = min(page.height - 1, r[1].toInt())
                    val xa = max(0, min(r[2], colL).toInt()); val xb = min(page.width - 1, max(r[3], colR).toInt())
                    for (y in ya..yb) for (x in xa..xb) printBand[y * page.width + x] = true
                }
            }
        }
        val printByLayout = paint(c.labels, printComp)

        // stroke vote
        val compHw = BooleanArray(c.count) { it > 0 && fracC[it] >= 0.35f && !printComp[it] }
        val candFree = BooleanArray(page.size) { cand[it] && !printByLayout[it] }
        val cb = components(candFree, page)
        val big = BooleanArray(cb.count) { it > 0 && cb.area[it] >= (2 * k) * (2 * k) }
        var hw = BooleanArray(page.size) { candFree[it] && (compHw[c.labels[it]] || big[cb.labels[it]]) }

        // small pen marks (accents, dots, commas) right next to handwriting
        val near = dilate(hw, page, 4 * k + 1, ellipse = true)
        val smallPen = BooleanArray(c.count) { small[it] && !printComp[it] && fracC[it] >= 0.15f }
        for (i in 0 until page.size) if (smallPen[c.labels[i]] && near[i] && d.ink[i]) hw[i] = true

        // handwritten lines: in a loose line of non-print strokes that is mostly pen already,
        // strokes only slightly bluer than print are pen too (pen colour varies with pressure)
        run {
            val ids = (1 until c.count).filter { isGlyph(c, it, k, page.height) && !printComp[it] }.sortedBy { c.cx[it] }
            val uf = UnionFind(c.count)
            for (a in ids.indices) {
                val i = ids[a]
                val reach = 2.5f * max(c.h[i], 3 * k)
                var b = a + 1
                while (b < ids.size && c.cx[ids[b]] - c.cx[i] < reach) {
                    val o = ids[b]
                    val ci = c.y[i] + c.h[i] / 2f
                    val co = c.y[o] + c.h[o] / 2f
                    if (abs(co - ci) < 0.6f * max(c.h[i], c.h[o])) uf.join(o, i)
                    b++
                }
            }
            val hwFrac = fractionPerComponent(c, hw)
            val relSum = DoubleArray(c.count)
            val relCnt = IntArray(c.count)
            for (i in 0 until page.size) {
                val l = c.labels[i]
                if (l > 0 && !relative[i].isNaN()) { relSum[l] += relative[i].toDouble(); relCnt[l]++ }
            }
            val weakPen = BooleanArray(c.count)
            for (g in ids.groupBy { uf.find(it) }.values) {
                if (g.size < 2) continue
                var pen = 0.0; var total = 0.0
                for (i in g) { pen += hwFrac[i] * c.area[i]; total += c.area[i] }
                if (pen < 0.5 * total) continue
                for (i in g) if (relCnt[i] > 0 && relSum[i] / relCnt[i] < -0.015) weakPen[i] = true
            }
            for (i in 0 until page.size) if (weakPen[c.labels[i]] && d.ink[i]) hw[i] = true
        }

        // neighbourhood consensus: a pen stroke's darkness/colour is uneven, so single pixels
        // flip; an ink pixel takes the majority label of the ink around it (≈3 stroke widths),
        // both ways — holes inside handwriting close, isolated pen-coloured specks in print go
        // back to print. Layout print (regular lines) keeps its label.
        run {
            val cwin = 3 * k + 1
            val inkSum = boxFilter(FloatArray(page.size) { if (d.ink[it]) 1f else 0f }, page, cwin)
            repeat(2) {
                val hwSum = boxFilter(FloatArray(page.size) { if (hw[it] && d.ink[it]) 1f else 0f }, page, cwin)
                for (i in 0 until page.size) {
                    if (!d.ink[i]) continue
                    hw[i] = !printByLayout[i] && hwSum[i] >= 0.5f * max(inkSum[i], 1e-6f)
                }
            }
        }

        // pen strokes merged INTO printed letters (an underline touching the word): mark their
        // pen-coloured pixels for full-resolution refinement
        val mixedMask = BooleanArray(page.size)
        run {
            val mixed = BooleanArray(c.count) { it > 0 && printComp[it] && fracC[it] >= 0.06f }
            for (i in 0 until page.size) if (mixed[c.labels[i]] && cand[i]) mixedMask[i] = true
        }
        // strongly mixed AND shaped like a letter + underline (much wider than tall, a stroke's
        // worth of pen colour): its pen-coloured pixels are handwriting, protected like fragments
        val mixedPen = BooleanArray(page.size)
        val enclosedPrint = BooleanArray(page.size)
        val ringPen = BooleanArray(page.size)
        run {
            val strong = BooleanArray(c.count) {
                it > 0 && printComp[it] && fracC[it] >= 0.4f && c.w[it] >= 2.5f * c.h[it] && fracC[it] * c.area[it] >= 3f * k * k
            }
            // ...or a pen LOOP around printed text (a circled item number): its pen-coloured pixels
            // enclose the component's other ink on at least 3 of 4 sides (the ring is often open
            // where it crosses the print it circles) — see ios InkAnalysis.hpp
            for (i in 1 until c.count) {
                if (strong[i] || !printComp[i] || fracC[i] < 0.4f || fracC[i] * c.area[i] < 3f * k * k) continue
                val bx = c.x[i]; val by = c.y[i]; val bw = c.w[i]; val bh = c.h[i]
                val rowFirst = IntArray(bh) { Int.MAX_VALUE }; val rowLast = IntArray(bh) { -1 }
                val colFirst = IntArray(bw) { Int.MAX_VALUE }; val colLast = IntArray(bw) { -1 }
                for (y in 0 until bh) for (x in 0 until bw) {
                    val p = (by + y) * page.width + bx + x
                    if (c.labels[p] != i || !cand[p]) continue
                    rowFirst[y] = min(rowFirst[y], x); rowLast[y] = max(rowLast[y], x)
                    colFirst[x] = min(colFirst[x], y); colLast[x] = max(colLast[x], y)
                }
                var restCount = 0; var enclosed = 0
                val inside = ArrayList<Int>()
                for (y in 0 until bh) for (x in 0 until bw) {
                    val p = (by + y) * page.width + bx + x
                    if (c.labels[p] != i || !d.ink[p] || cand[p]) continue
                    restCount++
                    var sides = 0
                    if (rowFirst[y] < x) sides++
                    if (rowLast[y] > x) sides++
                    if (colFirst[x] < y) sides++
                    if (colLast[x] > y) sides++
                    if (sides >= 3) { enclosed++; inside.add(p) }
                }
                if (enclosed >= k * k && enclosed >= 0.25 * restCount) {
                    strong[i] = true
                    // the circled print stays print through refinement; the ring is pen
                    for (p in inside) enclosedPrint[p] = true
                    for (y in 0 until bh) for (x in 0 until bw) {
                        val p = (by + y) * page.width + bx + x
                        if (c.labels[p] == i && cand[p]) ringPen[p] = true
                    }
                }
            }
            for (i in 0 until page.size) if (strong[c.labels[i]] && cand[i]) { mixedPen[i] = true; hw[i] = true }
        }

        // small detached pen fragments (lead-in curl, colon, dash, accent): colour is unreliable
        // on bits this thin, so their NEAREST neighbour decides — see ios InkAnalysis.hpp
        val fragmentMask = BooleanArray(page.size)
        run {
            val tinyMax = (2 * k) * (2 * k)
            val smallMax = (4 * k) * (4 * k)
            val nonHwFrac = fractionPerComponent(c, BooleanArray(page.size) { d.ink[it] && !hw[it] })
            val refComp = BooleanArray(c.count) { it > 0 && (printComp[it] || (c.area[it] > tinyMax && nonHwFrac[it] >= 0.5f)) }
            val hwInkMat = toMat(BooleanArray(page.size) { hw[it] && d.ink[it] }, page)
            val prRefMat = toMat(BooleanArray(page.size) { refComp[c.labels[it]] && d.ink[it] && !hw[it] }, page)
            val notHw = Mat(); val notPr = Mat(); val distHw = Mat(); val distPr = Mat()
            Core.bitwise_not(hwInkMat, notHw); Core.bitwise_not(prRefMat, notPr)
            Imgproc.distanceTransform(notHw, distHw, Geometry.DIST_L2, 3)
            Imgproc.distanceTransform(notPr, distPr, Geometry.DIST_L2, 3)
            val dh = toFloats(distHw); val dp = toFloats(distPr)
            releasing(hwInkMat, prRefMat, notHw, notPr, distHw, distPr) {}
            val dHw = FloatArray(c.count) { 1e9f }; val dPr = FloatArray(c.count) { 1e9f }
            for (i in 0 until page.size) {
                val l = c.labels[i]
                if (l == 0 || !d.ink[i]) continue
                if (dh[i] < dHw[l]) dHw[l] = dh[i]
                if (dp[i] < dPr[l]) dPr[l] = dp[i]
            }
            val fragment = BooleanArray(c.count) {
                it > 0 && !printComp[it] && (
                    (c.area[it] <= tinyMax && dHw[it] < dPr[it] && dHw[it] <= 8 * k) ||
                        (c.area[it] <= smallMax && fracC[it] >= 0.25f && dHw[it] <= 2 * k)
                    )
            }
            for (i in 0 until page.size) {
                if ((fragment[c.labels[i]] && d.ink[i]) || mixedPen[i]) { fragmentMask[i] = true; hw[i] = true }
            }
        }

        // pictures (photos, logos): ink dense over a wide area; text strokes never are
        val win = odd(max(page.width, page.height) * 0.03)
        val dens = boxFilter(FloatArray(page.size) { if (d.ink[it]) 1f else 0f }, page, win)
        val cd = components(BooleanArray(page.size) { dens[it] > 0.5f }, page)
        val bigDense = BooleanArray(cd.count) { it > 0 && cd.area[it] >= (2 * win) * (2 * win) }
        val picture = dilate(paint(cd.labels, bigDense), page, win, ellipse = false)
        for (i in 0 until page.size) if (picture[i]) hw[i] = false

        // print to protect: print-coloured ink, layout print, pictures, faint print-coloured ink
        val faintRatio = penScore(inkChroma(d, page, FAINT_OD, INK_OD, 2 * k + 1, 0.1f), refChroma, penDir)
        var print = BooleanArray(page.size) {
            val m = d.mean[it]
            (d.ink[it] && !relative[it].isNaN() && relative[it] > -0.02f) || printByLayout[it] || picture[it] ||
                (m > FAINT_OD && m <= INK_OD && !hw[it] && !faintRatio[it].isNaN() && faintRatio[it] > ref[it] - 0.03f)
        }
        print = dilate(print, page, 3, ellipse = false)

        // grow along the pen strokes into their faint parts, stopping at print
        val allowed = BooleanArray(page.size) { d.mean[it] > FAINT_OD && !print[it] }
        val grown = growWithin(BooleanArray(page.size) { hw[it] && allowed[it] }, allowed, page)
        for (i in 0 until page.size) if (grown[i]) hw[i] = true
        hw = removeSpecks(hw, page, max(12, k * k / 4))
        // context-placed dots are not noise
        for (i in 0 until page.size) if (fragmentMask[i] && !picture[i]) hw[i] = true
        val fragmentsOut = BooleanArray(page.size) { fragmentMask[it] && !picture[it] }
        val mixedOut = BooleanArray(page.size) { mixedMask[it] && !picture[it] }
        val enclosedOut = BooleanArray(page.size) { enclosedPrint[it] && !picture[it] }
        val colourless = colourlessHandwriting(d, c, regular, hw, modelHw, page)
        for (i in 0 until page.size) if (picture[i]) colourless[i] = false
        for (i in 0 until page.size) if (colourless[i]) print[i] = false
        val ringOut = BooleanArray(page.size) { ringPen[it] && !picture[it] }

        val refPlanes = refChroma.map { toFloatMat(it, page) }
        val refMat = Mat()
        Core.merge(refPlanes, refMat)
        refPlanes.forEach { it.release() }
        return WorkResult(toMat(hw, page), toMat(print, page), refMat, penDir, toMat(fragmentsOut, page), toMat(mixedOut, page), toMat(printBand, page),
            toMat(enclosedOut, page), toMat(ringOut, page), toMat(colourless, page))
    }

}
