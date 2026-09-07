package com.example.beacon_smart_scan.imageprocessing

import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgcodecs.Imgcodecs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Shared helpers for reading/writing images and translating Dart rect maps into clipped [Rect]s. */
object ImageIO {
    fun readOrThrow(path: String): Mat {
        val mat = Imgcodecs.imread(path, Imgcodecs.IMREAD_COLOR)
        if (mat.empty()) {
            throw ImageProcessingException("FILE_NOT_FOUND", "Không đọc được ảnh tại: $path")
        }
        return mat
    }

    fun writeOrThrow(mat: Mat, path: String) {
        if (!Imgcodecs.imwrite(path, mat)) {
            throw ImageProcessingException("PROCESSING_FAILED", "Không ghi được ảnh tại: $path")
        }
    }

    /** Converts a Dart `{left, top, right, bottom}` map into a [Rect] clipped to the image bounds. */
    fun mapToClippedRect(map: Map<String, Any>, imageWidth: Int, imageHeight: Int, paddingPx: Double = 0.0): Rect {
        val left = numberOf(map, "left") - paddingPx
        val top = numberOf(map, "top") - paddingPx
        val right = numberOf(map, "right") + paddingPx
        val bottom = numberOf(map, "bottom") + paddingPx

        val clippedLeft = max(0.0, left).roundToInt()
        val clippedTop = max(0.0, top).roundToInt()
        val clippedRight = min(imageWidth.toDouble(), right).roundToInt()
        val clippedBottom = min(imageHeight.toDouble(), bottom).roundToInt()

        val width = max(1, clippedRight - clippedLeft)
        val height = max(1, clippedBottom - clippedTop)
        return Rect(clippedLeft, clippedTop, width, height)
    }

    private fun numberOf(map: Map<String, Any>, key: String): Double =
        (map[key] as? Number)?.toDouble()
            ?: throw ImageProcessingException("INVALID_ARGUMENT", "Missing numeric field: $key")
}
