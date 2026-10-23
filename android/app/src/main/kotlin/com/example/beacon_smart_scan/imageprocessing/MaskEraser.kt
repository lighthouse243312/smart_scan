package com.example.beacon_smart_scan.imageprocessing

import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.imgproc.Imgproc

/**
 * Removes the mask's handwriting by rebuilding, not blurring (see [InkRefine.erase]).
 * [analysisPath] is the unprocessed page the mask was computed on; [inputPath] (same geometry) is
 * the page to erase from. Mirrors `eraseWithMaskAtPath` in ios/Runner/ImageProcessingOpenCV.mm.
 */
object MaskEraser {
    /** [grid]: draw grid paper again here (the multi-pass erase does it once, last). */
    fun erase(inputPath: String, analysisPath: String, maskPath: String, outputPath: String, grid: Boolean = true): Map<String, Any> {
        val target = ImageIO.readOrThrow(inputPath)
        val analysis = ImageIO.readOrThrow(analysisPath)
        val layers = HandwritingMask.read(maskPath)
        try {
            if (analysis.size() != target.size()) {
                Imgproc.resize(analysis, analysis, target.size(), 0.0, 0.0, Imgproc.INTER_AREA)
            }
            if (layers.handwriting.size() != target.size()) {
                Imgproc.resize(layers.handwriting, layers.handwriting, target.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
                Imgproc.resize(layers.print, layers.print, target.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
                Imgproc.resize(layers.overlap, layers.overlap, target.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
                Imgproc.resize(layers.strike, layers.strike, target.size(), 0.0, 0.0, Imgproc.INTER_NEAREST)
            }
            val dst = InkRefine.erase(target, analysis, layers.handwriting, layers.print, layers.overlap, MlKitTextRecognizer(), layers.strike)
            // give back print / rules the erase took beyond the writing, clear the pen it left
            PrintRestore.restore(analysis, target, dst, layers.handwriting, layers.print)
            // rules the handwriting was written on: cleared of pen leftovers and drawn again
            RuleRedraw.apply(analysis, layers.handwriting, layers.print, dst)
            // grid paper the writing was on: its lines drawn again through the erased area
            if (grid) GridRedraw.apply(analysis, dst, layers.handwriting, layers.print)
            releasing(dst) { ImageIO.writeOrThrow(dst, outputPath) }
        } finally {
            target.release()
            analysis.release()
            layers.release()
        }
        return mapOf("outputPath" to outputPath)
    }
}
