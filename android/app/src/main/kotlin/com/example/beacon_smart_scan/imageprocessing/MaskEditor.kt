package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
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
                val erase = stroke["erase"] == true
                val points = (0 until coords.size / 2).map { Point(coords[2 * it], coords[2 * it + 1]) }
                val brush = Mat.zeros(handwriting.size(), CvType.CV_8U)
                if (points.size == 1) {
                    Imgproc.circle(brush, points[0], thickness / 2, Scalar(255.0), -1)
                } else {
                    val polyline = MatOfPoint(*points.toTypedArray())
                    HandwritingMask.releasing(polyline) {
                        Imgproc.polylines(brush, listOf(polyline), false, Scalar(255.0), thickness, Imgproc.LINE_8)
                    }
                }
                if (erase) {
                    handwriting.setTo(Scalar(0.0), brush)
                    layers.overlap.setTo(Scalar(0.0), brush)
                } else {
                    // "add to erase" never takes print: a wide brush over handwriting written on
                    // print erases the pen and keeps the print underneath
                    val notPrint = Mat(); Core.bitwise_not(layers.print, notPrint)
                    val add = Mat(); Core.bitwise_and(brush, notPrint, add)
                    Core.bitwise_or(handwriting, add, handwriting)
                    notPrint.release(); add.release()
                }
                brush.release()
            }
            HandwritingMask.write(handwriting, layers.print, layers.overlap, outputPath)
            return HandwritingMask.result(outputPath, handwriting)
        } finally {
            layers.release()
        }
    }
}
