package com.example.beacon_smart_scan.imageprocessing

import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing

/**
 * Ink-colour + layout method (see [InkAnalysis]): ink judged by its optical-density colour
 * against the LOCAL print colour, backed by page layout (print forms regular lines). Mirrors
 * `inkColorMaskAtPath` in ios/Runner/ImageProcessingOpenCV.mm.
 */
object InkColorMasker {
    /** [colorDelta]: how much bluer than print (OD_B/OD_R) a stroke must be; smaller = more sensitive. */
    fun mask(inputPath: String, maskPath: String, colorDelta: Double): Map<String, Any> {
        val src = ImageIO.readOrThrow(inputPath)
        val (handwriting, print, overlap) = releasing(src) { InkAnalysis.detectByInkColor(src, colorDelta) }
        return releasing(handwriting, print, overlap) {
            HandwritingMask.write(handwriting, print, overlap, maskPath)
            HandwritingMask.result(maskPath, handwriting)
        }
    }
}
