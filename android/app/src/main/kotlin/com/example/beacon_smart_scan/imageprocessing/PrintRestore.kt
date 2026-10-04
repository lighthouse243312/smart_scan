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
 * Works on primitive arrays (one byte per pixel) — the per-component statistics are single passes
 * over each component's pixel list. The app's Java heap is capped (512 MB with largeHeap) whatever
 * the device's RAM, so full-resolution arrays are kept to bytes, float images stay in native Mats,
 * and each stage's temporaries live in their own function so they are collectable once it returns.
 */
object PrintRestore {

    /** Connected components of [mask]; [start]/[pix] list each foreground component's pixels. */
    private class Components(mask: ByteArray, val w: Int, val h: Int) {
        val n: Int
        val bx: IntArray; val by: IntArray; val bw: IntArray; val bh: IntArray; val area: IntArray
        val start: IntArray
        val pix: IntArray

        init {
            val m = Mat(h, w, CvType.CV_8U); m.put(0, 0, mask)
            val lab = Mat(); val stats = Mat(); val cents = Mat()
            n = Imgproc.connectedComponentsWithStats(m, lab, stats, cents, 8, CvType.CV_32S)
            m.release(); cents.release()
            val st = IntArray(n * 5); stats.get(0, 0, st); stats.release()
            bx = IntArray(n) { st[it * 5] }; by = IntArray(n) { st[it * 5 + 1] }
            bw = IntArray(n) { st[it * 5 + 2] }; bh = IntArray(n) { st[it * 5 + 3] }; area = IntArray(n) { st[it * 5 + 4] }
            // labels read a row at a time; the background (label 0) is never queried, so not listed
            val row = IntArray(w)
            start = IntArray(n + 1)
            for (y in 0 until h) { lab.get(y, 0, row); for (l in row) if (l > 0) start[l + 1]++ }
            for (i in 0 until n) start[i + 1] += start[i]
            pix = IntArray(start[n])
            val fill = start.copyOf(n)
            for (y in 0 until h) {
                lab.get(y, 0, row)
                val base = y * w
                for (x in 0 until w) { val l = row[x]; if (l > 0) pix[fill[l]++] = base + x }
            }
            lab.release()
        }

        fun mean(i: Int, m: ByteArray): Double {
            var c = 0
            for (j in start[i] until start[i + 1]) if (m[pix[j]].toInt() != 0) c++
            return c.toDouble() / max(1, start[i + 1] - start[i])
        }

        inline fun meanOf(i: Int, f: (Int) -> Float): Double {
            var s = 0.0
            for (j in start[i] until start[i + 1]) s += f(pix[j])
            return s / max(1, start[i + 1] - start[i])
        }

        inline fun medianOf(i: Int, f: (Int) -> Float): Double {
            val v = FloatArray(start[i + 1] - start[i]) { f(pix[start[i] + it]) }
            return median(v)
        }

        fun paint(i: Int, m: ByteArray) {
            for (j in start[i] until start[i + 1]) m[pix[j]] = ON
        }
    }

    /**
     * The page's per-pixel Lab, one unsigned byte each — exactly what the 8-bit conversion gives:
     * darkness (255 - L) of the target [dN] and of the erased page [dE], and the normalized
     * original's a / b (stored + 128).
     */
    private class Page(val w: Int, val h: Int, val k: Int, val dN: ByteArray, val dE: ByteArray, val aN: ByteArray, val bN: ByteArray) {
        val n = w * h
        fun dN(p: Int): Int = dN[p].toInt() and 0xff
        fun dE(p: Int): Int = dE[p].toInt() and 0xff
        fun a(p: Int): Float = (aN[p].toInt() and 0xff) - 128f
        fun b(p: Int): Float = (bN[p].toInt() and 0xff) - 128f
        fun purple(p: Int): Float = a(p) - b(p)
    }

    private data class Box(val x: Float, val y: Float, val w: Float, val h: Float)

    /** The printed text the eraser left: glyph height, glyph pixels, glyph / bar boxes, and mostly-erased writing. */
    private class Text(val gh: Int, val core: ByteArray, val gBox: List<Box>, val bBox: List<Box>, val residue: ByteArray)

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

    private inline fun values(m: ByteArray, f: (Int) -> Float): FloatArray {
        var c = 0
        for (b in m) if (b.toInt() != 0) c++
        val out = FloatArray(c); var j = 0
        for (i in m.indices) if (m[i].toInt() != 0) out[j++] = f(i)
        return out
    }

    private fun and(vararg ms: ByteArray): ByteArray = ByteArray(ms[0].size) { i -> if (ms.all { it[i].toInt() != 0 }) ON else 0 }
    private fun or(a: ByteArray, b: ByteArray) = ByteArray(a.size) { if (a[it].toInt() != 0 || b[it].toInt() != 0) ON else 0 }
    private fun not(a: ByteArray) = ByteArray(a.size) { if (a[it].toInt() != 0) 0 else ON }
    private inline fun where(n: Int, pred: (Int) -> Boolean) = ByteArray(n) { if (pred(it)) ON else 0 }

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

    /** Box blur ([k] x [k]) of [value] per pixel — filled a row at a time, the image stays native. */
    private inline fun blurred(w: Int, h: Int, k: Int, value: (Int) -> Float): Mat {
        val src = Mat(h, w, CvType.CV_32F)
        val row = FloatArray(w)
        for (y in 0 until h) {
            val base = y * w
            for (x in 0 until w) row[x] = value(base + x)
            src.put(y, 0, row)
        }
        val dst = Mat(); Imgproc.blur(src, dst, Size(k.toDouble(), k.toDouble()))
        src.release()
        return dst
    }

    /** num / max(den, 1e-3) per pixel; releases both. */
    private fun ratio(num: Mat, den: Mat): FloatArray {
        val w = num.cols()
        val out = FloatArray(w * num.rows()); num.get(0, 0, out)
        val row = FloatArray(w)
        for (y in 0 until den.rows()) {
            den.get(y, 0, row)
            val base = y * w
            for (x in 0 until w) out[base + x] = out[base + x] / max(row[x], 1e-3f)
        }
        num.release(); den.release()
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

    /** The given Lab channels of [bgr], one unsigned byte per pixel. */
    private fun labChannels(bgr: Mat, vararg channels: Int): List<ByteArray> {
        val lab = Mat(); Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)
        val c = Mat()
        val out = channels.map { ch ->
            Core.extractChannel(lab, c, ch)
            ByteArray(c.total().toInt()).also { c.get(0, 0, it) }
        }
        lab.release(); c.release()
        return out
    }

    /** darkness (255 - L) per pixel. */
    private fun darkness(bgr: Mat): ByteArray {
        val l = labChannels(bgr, 0)[0]
        for (i in l.indices) l[i] = (255 - (l[i].toInt() and 0xff)).toByte()
        return l
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
        val (aN, bN) = labChannels(c, 1, 2)
        c.release(); if (o !== original) o.release()
        val pg = Page(w0, h0, k, darkness(target), darkness(erased), aN, bN)
        val paperE = paper(erased, 9, 21)
        val hwMask = ByteArray(n0).also { handwriting.get(0, 0, it); for (i in it.indices) if (it[i].toInt() != 0) it[i] = ON }
        val printMask = ByteArray(n0).also { print.get(0, 0, it); for (i in it.indices) it[i] = if ((it[i].toInt() and 0xff) > 127) ON else 0 }
        val inkN = where(n0) { pg.dN(it) > 40 }
        val inkE = where(n0) { pg.dE(it) > 40 }
        val erasedM = where(n0) { inkN[it].toInt() != 0 && pg.dE(it) < 0.35f * pg.dN(it) }

        val linesE = or(morph(inkE, w0, h0, Imgproc.MORPH_OPEN, 15 * k, 1), morph(inkE, w0, h0, Imgproc.MORPH_OPEN, 1, 15 * k))
        val text = printedText(pg, inkE, linesE, inkN, erasedM)
        val gh = text.gh

        var band = dilate(text.core, w0, h0, 4 * gh, gh or 1)

        // colour models — pen from what was erased, print from surviving glyphs
        val dark70 = where(n0) { pg.dN(it) > 70 }
        val penC = median(values(and(erasedM, dark70)) { pg.purple(it) })
        val printC = median(values(and(text.core, dark70)) { pg.purple(it) })
        val split = (penC + printC) / 2; val margin = 0.25 * abs(penC - printC)
        val separable = abs(penC - printC) >= 4

        val ruleGap = ruleGaps(pg, erasedM, linesE, split)

        // ── print the eraser took ──
        val a0 = median(values(and(text.core, dark70)) { pg.a(it) }); val b0 = median(values(and(text.core, dark70)) { pg.b(it) })
        val restore = ByteArray(n0)
        band = restorePrint(pg, text, inkN, ruleGap, hwMask, printMask, dark70, band, restore, a0, b0, printC, margin, split, separable)

        // ── pen the eraser left ──
        val residue = penLeft(pg, text, inkN, inkE, erasedM, dark70, linesE, ruleGap, band, restore, hwMask, printMask, a0, b0, separable)

        compose(pg, target, erased, paperE, residue, restore, ruleGap, linesE, split)
        paperE.release()
    }

    /** print colour distance of a pixel from the print's own (a0, b0). */
    private fun cd(pg: Page, p: Int, a0: Double, b0: Double): Float = kotlin.math.hypot(pg.a(p) - a0, pg.b(p) - b0).toFloat()

    private fun printedText(pg: Page, inkE: ByteArray, linesE: ByteArray, inkN: ByteArray, erasedM: ByteArray): Text {
        val w0 = pg.w; val h0 = pg.h; val n0 = pg.n; val k = pg.k
        val t = Components(and(inkE, not(dilate(linesE, w0, h0, 3, 3))), w0, h0)
        val glyph = BooleanArray(t.n)
        val heights = ArrayList<Int>()
        for (i in 1 until t.n) {
            glyph[i] = t.area[i] >= 12 * k * k && t.bh[i] >= 6 * k && t.bh[i] <= 0.03 * h0 && t.bw[i] <= 0.05 * w0
            if (glyph[i]) heights.add(t.bh[i])
        }
        val gh = if (heights.isEmpty()) 12 * k else heights.sorted()[heights.size / 2]
        // share of the original ink around each point the eraser took (tight window: a printed
        // word right beside an erased one stays intact; a stray pen bit sits in a cleared patch)
        val erasedShare = erasedShare(t, pg, inkN, erasedM, (gh / 2) or 1)
        val intact = BooleanArray(t.n)
        for (i in 1 until t.n) { intact[i] = erasedShare[i] < 0.5; glyph[i] = glyph[i] && intact[i] }
        // a printed line = several glyphs side by side on one row (dashes, dots count as members)
        val member = ByteArray(n0); var core = ByteArray(n0)
        for (i in 1 until t.n) {
            if (intact[i] && t.area[i] >= 4 * k * k) t.paint(i, member)
            if (glyph[i]) t.paint(i, core)
        }
        val onLine = printLines(member, core, w0, h0, gh)
        val gBox = ArrayList<Box>()
        for (i in 1 until t.n) {
            if (!glyph[i]) continue
            var ok = false
            for (j in t.start[i] until t.start[i + 1]) if (onLine[t.pix[j]].toInt() != 0) { ok = true; break }
            if (ok) gBox.add(Box(t.bx[i].toFloat(), t.by[i].toFloat(), t.bw[i].toFloat(), t.bh[i].toFloat())) else glyph[i] = false
        }
        core = ByteArray(n0)
        for (i in 1 until t.n) if (glyph[i]) t.paint(i, core)
        val bBox = ArrayList<Box>()       // short surviving bars a fraction's digits hang on
        for (i in 1 until t.n)
            if (t.bw[i] <= 1.5 * gh && t.bw[i] >= 0.4 * gh && t.bh[i] <= max(2.0, 0.3 * gh))
                bBox.add(Box(t.bx[i].toFloat(), t.by[i].toFloat(), t.bw[i].toFloat(), t.bh[i].toFloat()))
        val residue = ByteArray(n0)
        for (i in 1 until t.n) if (erasedShare[i] > 0.5) t.paint(i, residue)
        return Text(gh, core, gBox, bBox, residue)
    }

    /** Each component's mean share of erased ink in the [win] window around its pixels. */
    private fun erasedShare(t: Components, pg: Page, inkN: ByteArray, erasedM: ByteArray, win: Int): DoubleArray {
        val ef = blurred(pg.w, pg.h, win) { if (erasedM[it].toInt() != 0) 1f else 0f }
        val inf = blurred(pg.w, pg.h, win) { if (inkN[it].toInt() != 0) 1f else 0f }
        val erasedRatio = ratio(ef, inf)
        return DoubleArray(t.n) { if (it == 0) 0.0 else t.meanOf(it) { p -> erasedRatio[p] } }
    }

    /** Pixels of the printed lines: [member] runs closed along the row, with a glyph of [core] and wide enough. */
    private fun printLines(member: ByteArray, core: ByteArray, w0: Int, h0: Int, gh: Int): ByteArray {
        val rw = Components(morph(member, w0, h0, Imgproc.MORPH_CLOSE, 2 * gh, 1), w0, h0)
        val out = ByteArray(w0 * h0)
        for (i in 1 until rw.n) if (rw.bw[i] >= 6 * gh && rw.mean(i, core) > 0) rw.paint(i, out)
        return out
    }

    /** ── rules the eraser took ── */
    private fun ruleGaps(pg: Page, erasedM: ByteArray, linesE: ByteArray, split: Double): ByteArray {
        val w0 = pg.w; val h0 = pg.h; val n0 = pg.n; val k = pg.k
        val neutralN = where(n0) { pg.dN(it) > 18 && pg.purple(it) < split }
        val footprint = dilate(erasedM, w0, h0, 7 * k, 7 * k, ellipse = true)
        val missing = where(n0) { pg.dN(it) > 18 && pg.dE(it) < 0.5f * pg.dN(it) && footprint[it].toInt() != 0 }
        val ruleGap = ByteArray(n0)
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
        return and(ruleGap, not(linesE))
    }

    /** ── print the eraser took ── marks it in [restore]; returns the band grown by what was restored. */
    private fun restorePrint(
        pg: Page, text: Text, inkN: ByteArray, ruleGap: ByteArray, hwMask: ByteArray, printMask: ByteArray, dark70: ByteArray,
        band0: ByteArray, restore: ByteArray, a0: Double, b0: Double, printC: Double, margin: Double, split: Double, separable: Boolean,
    ): ByteArray {
        val w0 = pg.w; val h0 = pg.h; val n0 = pg.n
        val gh = text.gh; val gBox = text.gBox; val bBox = text.bBox

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

        var band = band0
        val tallBand = dilate(text.core, w0, h0, 2 * gh, (3 * gh) or 1)
        val cdPrintT = percentile(values(and(text.core, dark70)) { cd(pg, it, a0, b0) }, 90.0)
        val faded = where(n0) { inkN[it].toInt() != 0 && pg.dE(it) < 0.6f * pg.dN(it) }
        val f = Components(and(faded, not(ruleGap)), w0, h0)
        val done = BooleanArray(f.n)
        repeat(3) {                              // a restored glyph extends the band
            for (i in 1 until f.n) {
                if (done[i]) continue
                val x = f.bx[i]; val y = f.by[i]; val w = f.bw[i]; val h = f.bh[i]
                val pm = f.medianOf(i) { pg.purple(it) }
                val printColoured = pm < printC + margin && f.medianOf(i) { cd(pg, it, a0, b0) } <= cdPrintT
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
        return band
    }

    /** ── pen the eraser left ── the pixels to clear to paper. */
    private fun penLeft(
        pg: Page, text: Text, inkN: ByteArray, inkE: ByteArray, erasedM: ByteArray, dark70: ByteArray, linesE: ByteArray,
        ruleGap: ByteArray, band: ByteArray, restore: ByteArray, hwMask: ByteArray, printMask: ByteArray, a0: Double, b0: Double,
        separable: Boolean,
    ): ByteArray {
        val w0 = pg.w; val h0 = pg.h; val n0 = pg.n; val gh = text.gh
        val cdS = ratio(
            blurred(w0, h0, 3) { if (inkN[it].toInt() != 0) cd(pg, it, a0, b0) else 0f },
            blurred(w0, h0, 3) { if (inkN[it].toInt() != 0) 1f else 0f },
        )
        val penCd = median(values(and(erasedM, dark70)) { cdS[it] })
        val printCd = percentile(values(and(text.core, dark70)) { cdS[it] }, 90.0)
        val chromaT = (penCd + printCd) / 2
        val keep = or(or(or(band, restore), ruleGap), and(printMask, not(hwMask)))
        val beside = dilate(erasedM, w0, h0, (2 * gh) or 1, (2 * gh) or 1, ellipse = true)
        val rulesAll = dilate(or(linesE, ruleGap), w0, h0, 3, 3)
        var residue = and(text.residue, not(keep), beside)
        if (separable) {
            val chromaInk = where(n0) { inkE[it].toInt() != 0 && cdS[it] > chromaT && beside[it].toInt() != 0 && rulesAll[it].toInt() == 0 }
            residue = or(residue, and(dilate(chromaInk, w0, h0, 3, 3), inkE, not(keep), not(rulesAll)))
        }
        return residue
    }

    /** ── compose ── into [erased], a row at a time. */
    private fun compose(pg: Page, target: Mat, erased: Mat, paperE: Mat, residue: ByteArray, restore: ByteArray, ruleGap: ByteArray, linesE: ByteArray, split: Double) {
        val w0 = pg.w; val h0 = pg.h
        val tgt = ByteArray(w0 * 3); val out = ByteArray(w0 * 3); val pap = ByteArray(w0 * 3)
        val ruleCol = IntArray(3) { 160 }
        run {
            val src = where(pg.n) { linesE[it].toInt() != 0 && pg.dN(it) > 25 }
            var c = 0
            for (b in src) if (b.toInt() != 0) c++
            if (c > 0) {
                val v = Array(3) { FloatArray(c) }; var j = 0
                for (y in 0 until h0) {
                    target.get(y, 0, tgt)
                    for (x in 0 until w0) if (src[y * w0 + x].toInt() != 0) {
                        for (ch in 0..2) v[ch][j] = (tgt[x * 3 + ch].toInt() and 0xff).toFloat()
                        j++
                    }
                }
                for (ch in 0..2) ruleCol[ch] = median(v[ch]).toInt()
            }
        }
        for (y in 0 until h0) {
            target.get(y, 0, tgt); erased.get(y, 0, out); paperE.get(y, 0, pap)
            for (x in 0 until w0) {
                val p = y * w0 + x
                when {
                    residue[p].toInt() != 0 -> for (ch in 0..2) out[x * 3 + ch] = pap[x * 3 + ch]
                    restore[p].toInt() != 0 -> for (ch in 0..2) out[x * 3 + ch] = tgt[x * 3 + ch]
                    ruleGap[p].toInt() != 0 -> for (ch in 0..2)
                        out[x * 3 + ch] = if (pg.purple(p) < split) tgt[x * 3 + ch] else ruleCol[ch].toByte()
                }
            }
            erased.put(y, 0, out)
        }
    }
}
