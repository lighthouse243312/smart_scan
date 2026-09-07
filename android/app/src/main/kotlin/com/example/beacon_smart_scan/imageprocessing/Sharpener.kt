package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** Unsharp-mask sharpening: subtract a blurred copy from the original to boost edge contrast. */
object Sharpener {
    fun sharpen(inputPath: String, outputPath: String, amount: Double, radius: Double): Map<String, Any> {
        val src = ImageIO.readOrThrow(inputPath)
        val blurred = Mat()
        try {
            val kernelRadius = if (radius <= 0.0) 3.0 else radius
            Imgproc.GaussianBlur(src, blurred, Size(0.0, 0.0), kernelRadius)

            val sharpened = Mat()
            try {
                // dst = src*(1+amount) + blurred*(-amount): boosts the high-frequency
                // (edge) component that GaussianBlur removed, i.e. classic unsharp mask.
                Core.addWeighted(src, 1.0 + amount, blurred, -amount, 0.0, sharpened)
                ImageIO.writeOrThrow(sharpened, outputPath)
            } finally {
                sharpened.release()
            }
        } finally {
            blurred.release()
            src.release()
        }
        return mapOf("outputPath" to outputPath)
    }
}
