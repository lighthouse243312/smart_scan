package com.example.beacon_smart_scan.imageprocessing

import java.util.ArrayList
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

/**
 * Illumination normalization: estimate the page's shading via a heavily blurred copy of the
 * lightness channel, divide it out, then recover local contrast with CLAHE. Working in Lab
 * (rather than per BGR channel) keeps color from shifting — only lightness is touched.
 */
object ShadowRemover {
    fun removeShadow(inputPath: String, outputPath: String): Map<String, Any> {
        val src = ImageIO.readOrThrow(inputPath)
        val lab = Mat()
        val channels = ArrayList<Mat>()
        try {
            Imgproc.cvtColor(src, lab, Imgproc.COLOR_BGR2Lab)
            Core.split(lab, channels)
            val l = channels[0]

            val sigma = (max(src.width(), src.height()) * 0.05).coerceIn(15.0, 60.0)
            val background = Mat()
            val lFloat = Mat()
            val bgFloat = Mat()
            val normFloat = Mat()
            val lNormalized = Mat()
            val lFinal = Mat()
            try {
                Imgproc.GaussianBlur(l, background, Size(0.0, 0.0), sigma)

                l.convertTo(lFloat, CvType.CV_32F)
                background.convertTo(bgFloat, CvType.CV_32F)
                Core.add(bgFloat, Scalar(1.0), bgFloat) // avoid divide-by-zero on pure black

                // normalized = l * 255 / background: pixels at the local background level
                // land near 255 (white), ink stays dark relative to its neighborhood — the
                // page's overall shading (shadow) is what "background" captures and removes.
                Core.divide(lFloat, bgFloat, normFloat, 255.0)
                normFloat.convertTo(lNormalized, CvType.CV_8U)

                val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
                clahe.apply(lNormalized, lFinal)

                val mergedChannels = arrayListOf(lFinal, channels[1], channels[2])
                val merged = Mat()
                val bgr = Mat()
                try {
                    Core.merge(mergedChannels, merged)
                    Imgproc.cvtColor(merged, bgr, Imgproc.COLOR_Lab2BGR)
                    ImageIO.writeOrThrow(bgr, outputPath)
                } finally {
                    merged.release()
                    bgr.release()
                }
            } finally {
                background.release()
                lFloat.release()
                bgFloat.release()
                normFloat.release()
                lNormalized.release()
                lFinal.release()
            }
        } finally {
            channels.forEach { it.release() }
            lab.release()
            src.release()
        }
        return mapOf("outputPath" to outputPath)
    }
}
