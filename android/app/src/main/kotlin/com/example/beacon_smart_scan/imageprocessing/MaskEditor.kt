package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Manual correction: paints (or, with `erase`, clears) brush strokes into the handwriting layer;
 * the print layer is kept, so print under a stroke the user brushes in is still restored on
 * erase. Each stroke is `{ "points": [x0, y0, x1, y1, ...] (image px), "width": Double,
 * "erase": Bool }`. Mirrors `applyMaskStrokesAtPath` in ios/Runner/ImageProcessingOpenCV.mm.
 */
object MaskEditor {
    fun applyStrokes(maskPath: String, outputPath: String, strokes: List<Map<String, Any>>): Map<String, Any> {
        val layers = HandwritingMask.read(maskPath)
        try {
            val handwriting = layers.handwriting
            for (stroke in strokes) {
                val coords = (stroke["points"] as? List<*>)?.mapNotNull { (it as? Number)?.toDouble() } ?: continue
                if (coords.size < 2) continue
                val thickness = max(1, ((stroke["width"] as? Number)?.toDouble() ?: 1.0).roundToInt())
                val value = Scalar(if (stroke["erase"] == true) 0.0 else 255.0)
                val points = (0 until coords.size / 2).map { Point(coords[2 * it], coords[2 * it + 1]) }
                if (points.size == 1) {
                    Imgproc.circle(handwriting, points[0], thickness / 2, value, -1)
                } else {
                    val polyline = MatOfPoint(*points.toTypedArray())
                    HandwritingMask.releasing(polyline) {
                        Imgproc.polylines(handwriting, listOf(polyline), false, value, thickness, Imgproc.LINE_8)
                    }
                }
            }
            HandwritingMask.write(handwriting, layers.print, outputPath)
            return HandwritingMask.result(outputPath, handwriting)
        } finally {
            layers.release()
        }
    }
}
