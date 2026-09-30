package com.example.beacon_smart_scan.imageprocessing

import android.content.Context
import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing

/**
 * Ink-colour + layout method (see [InkAnalysis]): ink judged by its optical-density colour
 * against the LOCAL print colour, backed by page layout (print forms regular lines). Mirrors
 * `inkColorMaskAtPath` in ios/Runner/ImageProcessingOpenCV.mm.
 */
object InkColorMasker {
    /** [colorDelta]: how much bluer than print (OD_B/OD_R) a stroke must be; smaller = more sensitive. */
    fun mask(context: Context, inputPath: String, maskPath: String, colorDelta: Double): Map<String, Any> {
        val src = ImageIO.readOrThrow(inputPath)
        // the model as a second opinion only (colourless handwriting, print under a near-black pen)
        val probs = InkSegmenter.probabilities(context, src)
        val (handwriting, print, overlap) = releasing(src) {
            try {
                InkAnalysis.detectByInkColor(src, colorDelta, probs?.first, probs?.second).also { (hw, pr, ov) ->
                    PenComponentVote.apply(src, hw, pr, ov, probs?.second)
                }
            } finally {
                probs?.first?.release(); probs?.second?.release()
            }
        }
        return releasing(handwriting, print, overlap) {
            HandwritingMask.write(handwriting, print, overlap, maskPath)
            HandwritingMask.result(maskPath, handwriting)
        }
    }
}
