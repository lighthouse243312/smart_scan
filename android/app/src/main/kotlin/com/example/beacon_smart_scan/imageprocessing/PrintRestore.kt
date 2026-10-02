package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Post-erase restore — runs right after [InkRefine.erase] on the same page. Mirrors
 * ios/Runner/PrintRestore.hpp and ml/prototype/print_restore.py (keep the three in step):
 *
 * 1. print the eraser took: erased strokes inside a printed-text line band; inside the
 *    handwriting mask only with the model's print layer or print geometry (level with the print
 *    beside it at its height, a digit on a fraction bar, a diacritic on a glyph)
 * 2. rules (grid) the eraser took: long thin straight neutral runs of the original that continue
 *    a surviving rule, bridged across the pen strokes crossing them
 * 3. pen the eraser left: bits of mostly-erased writing, and chromatic ink, off the print lines
 *
 * Works on primitive arrays (one byte / float per pixel) — the per-component statistics are
 * single passes over each component's pixel list.
 */
object PrintRestore {

    private class Components(mask: ByteArray, val w: Int, val h: Int) {
        val n: Int
        val labels: IntArray
        val bx: IntArray; val by: IntArray; val bw: IntArray; val bh: IntArray; val area: IntArray
        val start: IntArray
        val pix: IntArray

        init {
            val m = Mat(h, w, CvType.CV_8U); m.put(0, 0, mask)
            val lab = Mat(); val stats = Mat(); val cents = Mat()
            n = Imgproc.connectedComponentsWithStats(m, lab, stats, cents, 8, CvType.CV_32S)
            labels = IntArray(w * h); lab.get(0, 0, labels)
            val st = IntArray(n * 5); stats.get(0, 0, st)
            bx = IntArray(n) { st[it * 5] }; by = IntArray(n) { st[it * 5 + 1] }
            bw = IntArray(n) { st[it * 5 + 2] }; bh = IntArray(n) { st[it * 5 + 3] }; area = IntArray(n) { st[it * 5 + 4] }
            start = IntArray(n + 1)
            for (l in labels) start[l + 1]++
            for (i in 0 until n) start[i + 1] += start[i]
            pix = IntArray(labels.size)
            val fill = start.copyOf(n)
            for (i in labels.indices) pix[fill[labels[i]]++] = i
            m.release(); lab.release(); stats.release(); cents.release()
        }

        fun mean(i: Int, m: ByteArray): Double {
            var c = 0
            for (j in start[i] until start[i + 1]) if (m[pix[j]].toInt() != 0) c++
            return c.toDouble() / max(1, start[i + 1] - start[i])
        }

        fun meanF(i: Int, f: FloatArray): Double {
            var s = 0.0
            for (j in start[i] until start[i + 1]) s += f[pix[j]]
            return s / max(1, start[i + 1] - start[i])
        }

        fun medianF(i: Int, f: FloatArray): Double {
            val v = FloatArray(start[i + 1] - start[i]) { f[pix[start[i] + it]] }
            return median(v)
        }

        fun paint(i: Int, m: ByteArray) {
            for (j in start[i] until start[i + 1]) m[pix[j]] = ON
        }
    }

    private data class Box(val x: Float, val y: Float, val w: Float, val h: Float)

    private const val ON: Byte = 255.toByte()

    private fun median(v: FloatArray): Double {
        if (v.isEmpty()) return 0.0
        v.sort()
        return v[v.size / 2].toDouble()
    }

    private fun percentile(v: FloatArray, q: Double): Double {
        if (v.isEmpty()) return 0.0
        v.sort()
        val idx = q / 100.0 * (v.size - 1)
        val lo = idx.toInt(); val hi = min(lo + 1, v.size - 1)
        return v[lo] + (v[hi] - v[lo]) * (idx - lo)
    }

    private fun values(f: FloatArray, m: ByteArray): FloatArray {
        var c = 0
        for (b in m) if (b.toInt() != 0) c++
        val out = FloatArray(c); var j = 0
        for (i in m.indices) if (m[i].toInt() != 0) out[j++] = f[i]
        return out
    }

    private fun and(vararg ms: ByteArray): ByteArray = ByteArray(ms[0].size) { i -> if (ms.all { it[i].toInt() != 0 }) ON else 0 }
    private fun or(a: ByteArray, b: ByteArray) = ByteArray(a.size) { if (a[it].toInt() != 0 || b[it].toInt() != 0) ON else 0 }
    private fun not(a: ByteArray) = ByteArray(a.size) { if (a[it].toInt() != 0) 0 else ON }
    private fun where(n: Int, pred: (Int) -> Boolean) = ByteArray(n) { if (pred(it)) ON else 0 }

    private fun morph(m: ByteArray, w: Int, h: Int, op: Int, kw: Int, kh: Int, ellipse: Boolean = false): ByteArray {
        val src = Mat(h, w, CvType.CV_8U); src.put(0, 0, m)
        val k = Imgproc.getStructuringElement(if (ellipse) Imgproc.MORPH_ELLIPSE else Imgproc.MORPH_RECT, Size(max(1, kw).toDouble(), max(1, kh).toDouble()))
        val dst = Mat()
        if (op == -1) Imgproc.dilate(src, dst, k) else Imgproc.morphologyEx(src, dst, op, k)
        val out = ByteArray(w * h); dst.get(0, 0, out)
        src.release(); k.release(); dst.release()
        return out
    }

    private fun dilate(m: ByteArray, w: Int, h: Int, kw: Int, kh: Int, ellipse: Boolean = false) = morph(m, w, h, -1, kw, kh, ellipse)

    private fun blur(f: FloatArray, w: Int, h: Int, k: Int): FloatArray {
        val src = Mat(h, w, CvType.CV_32F); src.put(0, 0, f)
        val dst = Mat(); Imgproc.blur(src, dst, Size(k.toDouble(), k.toDouble()))
        val out = FloatArray(w * h); dst.get(0, 0, out)
        src.release(); dst.release()
        return out
    }

    private fun paper(img: Mat, k: Int, blurK: Int): Mat {
        val ch = ArrayList<Mat>(); Core.split(img, ch)
        val kern = Mat.ones(k, k, CvType.CV_8U)
        for (c in ch) { Imgproc.dilate(c, c, kern); Imgproc.medianBlur(c, c, blurK) }
        val out = Mat(); Core.merge(ch, out)
        ch.forEach { it.release() }; kern.release()
        return out
    }

    private fun normalize(img: Mat): Mat {
        val bg = Mat(); paper(img, 7, 31).let { it.convertTo(bg, CvType.CV_32FC3); it.release() }
        Core.max(bg, Scalar(1.0, 1.0, 1.0), bg)
        val f = Mat(); img.convertTo(f, CvType.CV_32FC3)
        val out = Mat(); Core.divide(f, bg, out, 255.0)
        out.convertTo(out, CvType.CV_8UC3)
        bg.release(); f.release()
        return out
    }

    /** darkness (255 - L), a and b, each as one float per pixel. */
    private fun lab(bgr: Mat): Triple<FloatArray, FloatArray, FloatArray> {
        val l = Mat(); Imgproc.cvtColor(bgr, l, Imgproc.COLOR_BGR2Lab)
        val n = (l.total()).toInt()
        val raw = ByteArray(n * 3); l.get(0, 0, raw); l.release()
        val d = FloatArray(n) { 255f - (raw[it * 3].toInt() and 0xff) }
        val a = FloatArray(n) { (raw[it * 3 + 1].toInt() and 0xff) - 128f }
        val b = FloatArray(n) { (raw[it * 3 + 2].toInt() and 0xff) - 128f }
        return Triple(d, a, b)
    }

    /**
     * Restores print / rules the eraser took from [erased] and removes the pen it left, in place.
     * [original]: the unprocessed page (colour source); [target]: the page the eraser worked on;
     * [handwriting] / [print]: the mask's layers (CV_8U, 255 = set), all [erased]'s size.
     */
    fun restore(original: Mat, target: Mat, erased: Mat, handwriting: Mat, print: Mat) {
        val w0 = erased.cols(); val h0 = erased.rows(); val n0 = w0 * h0
        val k = max(1, (max(w0, h0) / 1280.0).roundToInt())
        val o = if (original.size() == erased.size()) original else Mat().also { Imgproc.resize(original, it, erased.size(), 0.0, 0.0, Imgproc.INTER_AREA) }
        val c = normalize(o)
        val (dN, _, _) = lab(target)
        val (_, aN, bN) = lab(c)
        val (dE, _, _) = lab(erased)
        c.release(); if (o !== original) o.release()
        val paperE = paper(erased, 9, 21)
        val hwRaw = ByteArray(n0); handwriting.get(0, 0, hwRaw)
        val prRaw = ByteArray(n0); print.get(0, 0, prRaw)
        val hwMask = where(n0) { hwRaw[it].toInt() != 0 }
        val printMask = where(n0) { (prRaw[it].toInt() and 0xff) > 127 }
        val purple = FloatArray(n0) { aN[it] - bN[it] }
        val inkN = where(n0) { dN[it] > 40 }
        val inkE = where(n0) { dE[it] > 40 }
        val erasedM = where(n0) { inkN[it].toInt() != 0 && dE[it] < 0.35f * dN[it] }
        val faded = where(n0) { inkN[it].toInt() != 0 && dE[it] < 0.6f * dN[it] }

        val linesE = or(morph(inkE, w0, h0, Imgproc.MORPH_OPEN, 15 * k, 1), morph(inkE, w0, h0, Imgproc.MORPH_OPEN, 1, 15 * k))
        val textE = and(inkE, not(dilate(linesE, w0, h0, 3, 3)))
        val t = Components(textE, w0, h0)
        val glyph = BooleanArray(t.n)
        val heights = ArrayList<Int>()
        for (i in 1 until t.n) {
            glyph[i] = t.area[i] >= 12 * k * k && t.bh[i] >= 6 * k && t.bh[i] <= 0.03 * h0 && t.bw[i] <= 0.05 * w0
            if (glyph[i]) heights.add(t.bh[i])
        }
        val gh = if (heights.isEmpty()) 12 * k else heights.sorted()[heights.size / 2]
        // share of the original ink around each point the eraser took (tight window: a printed
        // word right beside an erased one stays intact; a stray pen bit sits in a cleared patch)
        val win = (gh / 2) or 1
        val ef = blur(FloatArray(n0) { if (erasedM[it].toInt() != 0) 1f else 0f }, w0, h0, win)
        val inf = blur(FloatArray(n0) { if (inkN[it].toInt() != 0) 1f else 0f }, w0, h0, win)
        val erasedRatio = FloatArray(n0) { ef[it] / max(inf[it], 1e-3f) }
        val intact = BooleanArray(t.n)
        for (i in 1 until t.n) { intact[i] = t.meanF(i, erasedRatio) < 0.5; glyph[i] = glyph[i] && intact[i] }
        // a printed line = several glyphs side by side on one row (dashes, dots count as members)
        val member = ByteArray(n0); var core = ByteArray(n0)
        for (i in 1 until t.n) {
            if (intact[i] && t.area[i] >= 4 * k * k) t.paint(i, member)
            if (glyph[i]) t.paint(i, core)
        }
        val rw = Components(morph(member, w0, h0, Imgproc.MORPH_CLOSE, 2 * gh, 1), w0, h0)
        val lineOk = BooleanArray(rw.n) { it > 0 && rw.bw[it] >= 6 * gh && rw.mean(it, core) > 0 }
        val gBox = ArrayList<Box>()
        for (i in 1 until t.n) {
            if (!glyph[i]) continue
            var ok = false
            for (j in t.start[i] until t.start[i + 1]) if (lineOk[rw.labels[t.pix[j]]]) { ok = true; break }
            if (ok) gBox.add(Box(t.bx[i].toFloat(), t.by[i].toFloat(), t.bw[i].toFloat(), t.bh[i].toFloat())) else glyph[i] = false
        }
        core = ByteArray(n0)
        for (i in 1 until t.n) if (glyph[i]) t.paint(i, core)
        val bBox = ArrayList<Box>()       // short surviving bars a fraction's digits hang on
        for (i in 1 until t.n)
            if (t.bw[i] <= 1.5 * gh && t.bw[i] >= 0.4 * gh && t.bh[i] <= max(2.0, 0.3 * gh))
                bBox.add(Box(t.bx[i].toFloat(), t.by[i].toFloat(), t.bw[i].toFloat(), t.bh[i].toFloat()))

        fun onPrintLine(x: Int, y: Int, w: Int, h: Int): Boolean {
            val near = gBox.filter { it.x < x + w + 4 * gh && it.x + it.w > x - 4 * gh && abs(it.y + it.h / 2 - (y + h / 2.0)) < gh }
            if (near.size < 2) return false
            val cy = median(FloatArray(near.size) { near[it].y + near[it].h / 2 })
            val ghn = median(FloatArray(near.size) { near[it].h })
            val base = median(FloatArray(near.size) { near[it].y + near[it].h })
            val level = min(abs(y + h / 2.0 - cy), abs(y + h - base)) <= 0.4 * ghn
            return level && h >= 0.6 * ghn && h <= 1.6 * ghn
        }

        fun hangsOn(boxes: List<Box>, x: Int, y: Int, w: Int, h: Int, up: Double, down: Double): Boolean = boxes.any { g ->
            if (!(g.x < x + w && g.x + g.w > x)) return@any false
            val a = g.y - (y + h); val b = y - (g.y + g.h)
            (a >= -1 && a <= up * gh) || (b >= -1 && b <= down * gh)
        }

        var band = dilate(core, w0, h0, 4 * gh, gh or 1)
        val tallBand = dilate(core, w0, h0, 2 * gh, (3 * gh) or 1)

        // colour models — pen from what was erased, print from surviving glyphs
        val dark70 = where(n0) { dN[it] > 70 }
        val penC = median(values(purple, and(erasedM, dark70)))
        val printC = median(values(purple, and(core, dark70)))
        val split = (penC + printC) / 2; val margin = 0.25 * abs(penC - printC)
        val separable = abs(penC - printC) >= 4

        // ── rules the eraser took ──
        val neutralN = where(n0) { dN[it] > 18 && purple[it] < split }
        val footprint = dilate(erasedM, w0, h0, 7 * k, 7 * k, ellipse = true)
        val missing = where(n0) { dN[it] > 18 && dE[it] < 0.5f * dN[it] && footprint[it].toInt() != 0 }
        var ruleGap = ByteArray(n0)
        for (dir in 0..1) {
            var run = morph(neutralN, w0, h0, Imgproc.MORPH_CLOSE, if (dir == 0) 7 * k else 1, if (dir == 0) 1 else 7 * k)
            run = morph(run, w0, h0, Imgproc.MORPH_OPEN, if (dir == 0) 31 * k else 1, if (dir == 0) 1 else 31 * k)
            val r = Components(run, w0, h0)
            for (i in 1 until r.n) {
                val thick = r.area[i].toDouble() / max(r.bw[i], r.bh[i])
                // a real rule mostly survived the eraser and was only cut
                if (thick <= 3 * k && r.mean(i, linesE) >= 0.3)
                    for (j in r.start[i] until r.start[i + 1]) { val p = r.pix[j]; if (missing[p].toInt() != 0) ruleGap[p] = ON }
            }
        }
        ruleGap = and(ruleGap, not(linesE))

        // ── print the eraser took ──
        val a0 = median(values(aN, and(core, dark70))); val b0 = median(values(bN, and(core, dark70)))
        val cdPx = FloatArray(n0) { kotlin.math.hypot(aN[it] - a0, bN[it] - b0).toFloat() }
        val cdPrintT = percentile(values(cdPx, and(core, dark70)), 90.0)
        val f = Components(and(faded, not(ruleGap)), w0, h0)
        val restore = ByteArray(n0)
        val done = BooleanArray(f.n)
        repeat(3) {                              // a restored glyph extends the band
            for (i in 1 until f.n) {
                if (done[i]) continue
                val x = f.bx[i]; val y = f.by[i]; val w = f.bw[i]; val h = f.bh[i]
                val pm = f.medianF(i, purple)
                val printColoured = pm < printC + margin && f.medianF(i, cdPx) <= cdPrintT
                if (separable && !printColoured && pm >= split) continue                // pen-coloured
                val inBand = f.mean(i, band) > 0.6
                val glyphSized = h <= 1.5 * gh && w <= 1.5 * gh
                val stacked = glyphSized && f.mean(i, tallBand) > 0.8 &&
                    (hangsOn(bBox, x, y, w, h, 0.5, 0.5) || (w <= 0.6 * gh && h <= 0.6 * gh && hangsOn(gBox, x, y, w, h, 0.6, 0.4)))
                if (!(inBand || stacked)) continue
                // beyond the handwriting mask (the eraser's halo): the band suffices. Inside it the
                // model called them handwriting — and a dark pen can be print-coloured — so it
                // takes the model's print layer, or print geometry
                val inside = f.mean(i, hwMask) > 0.5
                if (!inside || f.mean(i, printMask) > 0.5 || (printColoured && (stacked || onPrintLine(x, y, w, h)))) {
                    f.paint(i, restore); done[i] = true
                }
            }
            band = or(band, dilate(restore, w0, h0, 4 * gh, gh or 1))
        }

        // ── pen the eraser left ──
        val cdS: FloatArray
        run {
            val num = blur(FloatArray(n0) { if (inkN[it].toInt() != 0) cdPx[it] else 0f }, w0, h0, 3)
            val den = blur(FloatArray(n0) { if (inkN[it].toInt() != 0) 1f else 0f }, w0, h0, 3)
            cdS = FloatArray(n0) { num[it] / max(den[it], 1e-3f) }
        }
        val penCd = median(values(cdS, and(erasedM, dark70)))
        val printCd = percentile(values(cdS, and(core, dark70)), 90.0)
        val chromaT = (penCd + printCd) / 2
        val keep = or(or(or(band, restore), ruleGap), and(printMask, not(hwMask)))
        val beside = dilate(erasedM, w0, h0, (2 * gh) or 1, (2 * gh) or 1, ellipse = true)
        val rulesAll = dilate(or(linesE, ruleGap), w0, h0, 3, 3)
        var residue = ByteArray(n0)
        for (i in 1 until t.n) if (t.meanF(i, erasedRatio) > 0.5) t.paint(i, residue)
        residue = and(residue, not(keep), beside)
        if (separable) {
            val chromaInk = where(n0) { inkE[it].toInt() != 0 && cdS[it] > chromaT && beside[it].toInt() != 0 && rulesAll[it].toInt() == 0 }
            residue = or(residue, and(dilate(chromaInk, w0, h0, 3, 3), inkE, not(keep), not(rulesAll)))
        }

        // ── compose ──
        val tgt = ByteArray(n0 * 3); target.get(0, 0, tgt)
        val out = ByteArray(n0 * 3); erased.get(0, 0, out)
        val pap = ByteArray(n0 * 3); paperE.get(0, 0, pap); paperE.release()
        val ruleCol = IntArray(3) { 160 }
        run {
            val src = where(n0) { linesE[it].toInt() != 0 && dN[it] > 25 }
            for (ch in 0..2) {
                val v = values(FloatArray(n0) { (tgt[it * 3 + ch].toInt() and 0xff).toFloat() }, src)
                if (v.isNotEmpty()) ruleCol[ch] = median(v).toInt()
            }
        }
        for (p in 0 until n0) {
            when {
                residue[p].toInt() != 0 -> for (ch in 0..2) out[p * 3 + ch] = pap[p * 3 + ch]
                restore[p].toInt() != 0 -> for (ch in 0..2) out[p * 3 + ch] = tgt[p * 3 + ch]
                ruleGap[p].toInt() != 0 -> for (ch in 0..2)
                    out[p * 3 + ch] = if (purple[p] < split) tgt[p * 3 + ch] else ruleCol[ch].toByte()
            }
        }
        erased.put(0, 0, out)
    }
}
