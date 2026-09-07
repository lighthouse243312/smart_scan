package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core

/**
 * Manual 90°-step rotation. Document scanners (VNDocumentCameraViewController on iOS,
 * GmsDocumentScanner on Android) correctly crop the page rectangle but do not know which edge
 * is "up" for reading — a well-known limitation shared by every automatic scanner (Adobe Scan,
 * CamScanner, Apple Notes' scanner all ship a manual rotate button for exactly this reason).
 * When the scan comes out sideways/upside-down, ML Kit text recognition finds little or nothing
 * on it; this lets the user fix orientation before that step runs.
 */
object Rotator {
    fun rotate(inputPath: String, outputPath: String, quarterTurnsClockwise: Int): Map<String, Any> {
        val src = ImageIO.readOrThrow(inputPath)
        try {
            val normalizedTurns = ((quarterTurnsClockwise % 4) + 4) % 4
            if (normalizedTurns != 0) {
                val rotateCode = when (normalizedTurns) {
                    1 -> Core.ROTATE_90_CLOCKWISE
                    2 -> Core.ROTATE_180
                    else -> Core.ROTATE_90_COUNTERCLOCKWISE
                }
                Core.rotate(src, src, rotateCode)
            }
            ImageIO.writeOrThrow(src, outputPath)
        } finally {
            src.release()
        }
        return mapOf("outputPath" to outputPath)
    }
}
