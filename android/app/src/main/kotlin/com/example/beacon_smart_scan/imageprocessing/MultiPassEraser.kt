package com.example.beacon_smart_scan.imageprocessing

import android.content.Context
import java.io.File

/**
 * Erases in passes: one pass can leave pen pieces behind (faint ends, strokes the first mask only
 * half covered) that stand out once the rest is gone, so each later pass re-detects on the previous
 * result. The first mask — the one the user reviewed, measured on the untouched page — guards the
 * print ([PrintGuard]): later passes may not erase what it calls print, and print the passes turned
 * to paper anyway is copied back from the clean page at the end. Mirrors `eraseHandwritingAtPath`
 * in ios/Runner/ImageProcessingOpenCV.mm.
 */
object MultiPassEraser {
    /**
     * Most erase passes per page (the first included). Two: measured on 30 labelled pages (release
     * build) a third pass changed nothing (handwriting 91.96% / print 97.66% either way) and cost ~5 s.
     */
    @JvmStatic var MAX_PASSES = 2

    /** A later pass runs only if its re-detection, once guarded, still covers this share of the page. */
    private const val MIN_PASS_COVERAGE = 0.0002

    /**
     * [inputPath]: the processed page to erase from; [analysisPath]: the unprocessed page [maskPath]
     * was detected on. Later passes re-detect with the model + colour method when [useModel], else
     * the colour method alone, at [threshold] / [colorDelta]. A page larger than the working size is
     * erased at that size and the result laid back over the full-resolution page (see [WorkingSize]).
     */
    fun erase(
        context: Context, inputPath: String, analysisPath: String, maskPath: String, outputPath: String,
        useModel: Boolean, threshold: Double, colorDelta: Double,
    ): Map<String, Any> {
        val scale = WorkingSize.scaleFor(inputPath)
        if (scale >= 1.0) return eraseAtSize(context, inputPath, analysisPath, maskPath, outputPath, useModel, threshold, colorDelta)
        val work = WorkingSize.newWorkDir(context)
        try {
            val smallIn = WorkingSize.downscaleImage(inputPath, File(work, "page.png").path, scale)
            val size = WorkingSize.sizeOf(smallIn)
            val smallAnalysis = File(work, "analysis.png").path
            run {
                val analysis = ImageIO.readOrThrow(analysisPath)
                val small = org.opencv.core.Mat()
                try {
                    org.opencv.imgproc.Imgproc.resize(analysis, small, size, 0.0, 0.0, org.opencv.imgproc.Imgproc.INTER_AREA)
                    ImageIO.writeOrThrow(small, smallAnalysis)
                } finally {
                    analysis.release(); small.release()
                }
            }
            val smallMask = WorkingSize.resizeMask(maskPath, File(work, "mask.png").path, size)
            val smallOut = File(work, "erased.png").path
            val result = eraseAtSize(context, smallIn, smallAnalysis, smallMask, smallOut, useModel, threshold, colorDelta)
            WorkingSize.composite(inputPath, smallIn, smallOut, outputPath)
            return mapOf("outputPath" to outputPath, "passes" to result.getValue("passes"))
        } finally {
            work.deleteRecursively()
        }
    }

    private fun eraseAtSize(
        context: Context, inputPath: String, analysisPath: String, maskPath: String, outputPath: String,
        useModel: Boolean, threshold: Double, colorDelta: Double,
    ): Map<String, Any> {
        val work = File(context.cacheDir, "multipass_${System.nanoTime()}").apply { mkdirs() }
        try {
            var erased = File(work, "erased_1.png").path
            var tt = System.currentTimeMillis(); val tb = StringBuilder("erase stages:")
            fun lap(name: String) { val now = System.currentTimeMillis(); tb.append(" $name ${now - tt}"); tt = now }
            MaskEraser.erase(inputPath, analysisPath, maskPath, erased, grid = false)
            lap("pass1")
            val laterMasks = ArrayList<String>()
            var passes = 1
            for (pass in 2..MAX_PASSES) {
                val found = File(work, "found_$pass.png").path
                val coverage = (if (useModel) InkSegmenter.mask(context, erased, found, threshold, colorDelta)
                    else InkColorMasker.mask(context, erased, found, colorDelta))["coverage"] as Double
                lap("detect$pass")
                if (coverage < MIN_PASS_COVERAGE) break
                val guarded = File(work, "mask_$pass.png").path
                if ((PrintGuard.guardMask(found, maskPath, guarded)["coverage"] as Double) < MIN_PASS_COVERAGE) break
                val next = File(work, "erased_$pass.png").path
                MaskEraser.erase(erased, erased, guarded, next, grid = false)
                lap("erase$pass")
                erased = next
                laterMasks.add(guarded)
                passes = pass
            }
            PrintGuard.restorePrint(erased, inputPath, maskPath, laterMasks, outputPath)
            lap("restore")
            // the restore draws print back from the clean page, pen bits stuck to the rules with it:
            // the rules the handwriting touched are cleared and drawn once more, last
            run {
                val dst = ImageIO.readOrThrow(outputPath)
                val analysis = ImageIO.readOrThrow(analysisPath)
                val ref = HandwritingMask.read(maskPath)
                try {
                    if (analysis.size() != dst.size()) org.opencv.imgproc.Imgproc.resize(analysis, analysis, dst.size(), 0.0, 0.0, org.opencv.imgproc.Imgproc.INTER_AREA)
                    if (ref.handwriting.size() != dst.size()) {
                        org.opencv.imgproc.Imgproc.resize(ref.handwriting, ref.handwriting, dst.size(), 0.0, 0.0, org.opencv.imgproc.Imgproc.INTER_NEAREST)
                        org.opencv.imgproc.Imgproc.resize(ref.print, ref.print, dst.size(), 0.0, 0.0, org.opencv.imgproc.Imgproc.INTER_NEAREST)
                    }
                    RuleRedraw.apply(analysis, ref.handwriting, ref.print, dst)
                    LineHeal.apply(dst, ref.handwriting, ref.print)
                    SpeckClean.apply(dst, ref.handwriting, ref.print)
                    // grid paper the writing was on, drawn again through the erased area, last
                    GridRedraw.apply(analysis, dst, ref.handwriting, ref.print)
                    ImageIO.writeOrThrow(dst, outputPath)
                } finally {
                    dst.release(); analysis.release(); ref.release()
                }
            }
            lap("rules")
            timing = tb.toString() + "\n"
            return mapOf("outputPath" to outputPath, "passes" to passes)
        } finally {
            work.deleteRecursively()
        }
    }

    /** Dev: time of each erase stage of the last run (InkHarness report). */
    @JvmStatic var timing = ""
}
