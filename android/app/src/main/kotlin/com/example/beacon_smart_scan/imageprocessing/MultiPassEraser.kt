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
    /** Most erase passes per page (the first included). */
    const val MAX_PASSES = 3

    /** A later pass runs only if its re-detection, once guarded, still covers this share of the page. */
    private const val MIN_PASS_COVERAGE = 0.0002

    /**
     * [inputPath]: the processed page to erase from; [analysisPath]: the unprocessed page [maskPath]
     * was detected on. Later passes re-detect with the model + colour method when [useModel], else
     * the colour method alone, at [threshold] / [colorDelta].
     */
    fun erase(
        context: Context, inputPath: String, analysisPath: String, maskPath: String, outputPath: String,
        useModel: Boolean, threshold: Double, colorDelta: Double,
    ): Map<String, Any> {
        val work = File(context.cacheDir, "multipass_${System.nanoTime()}").apply { mkdirs() }
        try {
            var erased = File(work, "erased_1.png").path
            MaskEraser.erase(inputPath, analysisPath, maskPath, erased)
            val laterMasks = ArrayList<String>()
            var passes = 1
            for (pass in 2..MAX_PASSES) {
                val found = File(work, "found_$pass.png").path
                val coverage = (if (useModel) InkSegmenter.mask(context, erased, found, threshold, colorDelta)
                    else InkColorMasker.mask(context, erased, found, colorDelta))["coverage"] as Double
                if (coverage < MIN_PASS_COVERAGE) break
                val guarded = File(work, "mask_$pass.png").path
                if ((PrintGuard.guardMask(found, maskPath, guarded)["coverage"] as Double) < MIN_PASS_COVERAGE) break
                val next = File(work, "erased_$pass.png").path
                MaskEraser.erase(erased, erased, guarded, next)
                erased = next
                laterMasks.add(guarded)
                passes = pass
            }
            PrintGuard.restorePrint(erased, inputPath, maskPath, laterMasks, outputPath)
            return mapOf("outputPath" to outputPath, "passes" to passes)
        } finally {
            work.deleteRecursively()
        }
    }
}
