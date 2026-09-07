package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo

/** Erases the given regions via OpenCV inpainting (Telea) — fills them in from surrounding pixels. */
object Inpainter {
    fun erase(
        inputPath: String,
        outputPath: String,
        rects: List<Map<String, Any>>,
        padding: Double,
        inpaintRadius: Double,
    ): Map<String, Any> {
        val src = ImageIO.readOrThrow(inputPath)
        val mask = Mat.zeros(src.size(), CvType.CV_8UC1)
        val dst = Mat()
        try {
            rects.forEach { rectMap ->
                val rect = ImageIO.mapToClippedRect(rectMap, src.width(), src.height(), padding)
                Imgproc.rectangle(mask, rect.tl(), rect.br(), Scalar(255.0), -1)
            }
            Photo.inpaint(src, mask, dst, inpaintRadius, Photo.INPAINT_TELEA)
            ImageIO.writeOrThrow(dst, outputPath)
        } finally {
            mask.release()
            dst.release()
            src.release()
        }
        return mapOf("outputPath" to outputPath)
    }
}
