package com.example.beacon_smart_scan.imageprocessing

import android.content.Context
import android.graphics.BitmapFactory
import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Handwriting detection and erase run on the page scaled down to [LONG_SIDE] px: their cost grows
 * much faster than the pixel count (an 8 MP camera scan took minutes to erase, a 1.5 MP one tens of
 * seconds), and the steps are tuned on pages this size anyway — the model itself sees the page at
 * 1536 px. The page keeps its full resolution: a mask is scaled back up to it, and an erase result
 * replaces only the pixels the erase changed (see [composite]), so untouched print stays sharp.
 * Mirrors the "Working size" section of ios/Runner/ImageProcessingOpenCV.mm.
 */
object WorkingSize {
    const val LONG_SIDE = 1600

    /** Gray-level change (0-255) from the input that marks a pixel as changed by the erase. */
    private const val CHANGED_DELTA = 6.0

    /** The factor (< 1) the page at [path] is scaled by to work on, or 1 when it is small enough. */
    fun scaleFor(path: String): Double {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        val longSide = max(bounds.outWidth, bounds.outHeight)
        return if (longSide <= LONG_SIDE) 1.0 else LONG_SIDE.toDouble() / longSide
    }

    fun newWorkDir(context: Context): File =
        File(context.cacheDir, "working_${System.nanoTime()}").apply { mkdirs() }

    /** Writes the page at [path] scaled by [scale] to [outPath]; returns [outPath]. */
    fun downscaleImage(path: String, outPath: String, scale: Double): String {
        val src = ImageIO.readOrThrow(path)
        val small = Mat()
        releasing(src, small) {
            val size = Size((src.cols() * scale).roundToInt().toDouble(), (src.rows() * scale).roundToInt().toDouble())
            Imgproc.resize(src, small, size, 0.0, 0.0, Imgproc.INTER_AREA)
            ImageIO.writeOrThrow(small, outPath)
        }
        return outPath
    }

    /**
     * Writes the mask file at [maskPath] scaled to [size] to [outPath]. Area-averaged on the raw file:
     * a working pixel is handwriting if any of it was (alpha > 0 — no pen is lost), print / print under
     * the pen if most of it was (> 127).
     */
    fun resizeMask(maskPath: String, outPath: String, size: Size): String {
        val raw = Imgcodecs.imread(maskPath, Imgcodecs.IMREAD_UNCHANGED)
        val out = Mat()
        releasing(raw, out) {
            if (raw.empty() || raw.channels() != 4) {
                throw ImageProcessingException("FILE_NOT_FOUND", "Không đọc được mask tại: $maskPath")
            }
            val interpolation = if (size.width < raw.cols()) Imgproc.INTER_AREA else Imgproc.INTER_NEAREST
            Imgproc.resize(raw, out, size, 0.0, 0.0, interpolation)
            ImageIO.writeOrThrow(out, outPath)
        }
        return outPath
    }

    fun sizeOf(path: String): Size {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        return Size(bounds.outWidth.toDouble(), bounds.outHeight.toDouble())
    }

    /**
     * Runs [detect] (input path, mask path -> mask result) at the working size and writes its mask,
     * scaled back to the page's size, to [maskPath].
     */
    fun detect(
        context: Context, inputPath: String, maskPath: String,
        detect: (String, String) -> Map<String, Any>,
    ): Map<String, Any> {
        val scale = scaleFor(inputPath)
        if (scale >= 1.0) return detect(inputPath, maskPath)
        val work = newWorkDir(context)
        try {
            val small = downscaleImage(inputPath, File(work, "page.png").path, scale)
            val smallMask = File(work, "mask.png").path
            val result = detect(small, smallMask)
            resizeMask(smallMask, maskPath, sizeOf(inputPath))
            return mapOf("maskPath" to maskPath, "coverage" to (result["coverage"] as Double))
        } finally {
            work.deleteRecursively()
        }
    }

    /**
     * The full-resolution page at [fullPath] with the erase result [smallOutPath] (made from
     * [smallInPath], its working-size copy) laid over it where the erase changed something, written
     * to [outputPath]. Everywhere else the page keeps its own, sharper pixels.
     */
    fun composite(fullPath: String, smallInPath: String, smallOutPath: String, outputPath: String) {
        val full = ImageIO.readOrThrow(fullPath)
        val smallIn = ImageIO.readOrThrow(smallInPath)
        val smallOut = ImageIO.readOrThrow(smallOutPath)
        val changed = Mat()
        val up = Mat()
        val changedFull = Mat()
        releasing(full, smallIn, smallOut, changed, up, changedFull) {
            Core.absdiff(smallOut, smallIn, changed)
            Imgproc.cvtColor(changed, changed, Imgproc.COLOR_BGR2GRAY)
            Imgproc.threshold(changed, changed, CHANGED_DELTA, 255.0, Imgproc.THRESH_BINARY)
            // one working pixel of margin: the seam falls on pixels the erase left alone
            val margin = Mat.ones(3, 3, CvType.CV_8U)
            releasing(margin) { Imgproc.dilate(changed, changed, margin) }
            if (Core.countNonZero(changed) == 0) {
                ImageIO.writeOrThrow(full, outputPath)
                return
            }
            Imgproc.resize(smallOut, up, full.size(), 0.0, 0.0, Imgproc.INTER_CUBIC)
            Imgproc.resize(changed, changedFull, full.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            Imgproc.threshold(changedFull, changedFull, 0.0, 255.0, Imgproc.THRESH_BINARY)
            up.copyTo(full, changedFull)
            ImageIO.writeOrThrow(full, outputPath)
        }
    }
}
