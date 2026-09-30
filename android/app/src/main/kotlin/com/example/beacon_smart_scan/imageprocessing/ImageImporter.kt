package com.example.beacon_smart_scan.imageprocessing

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Brings a picture from outside the app (gallery / files) into the pipeline's format. The document
 * scanner hands over upright JPEGs; an arbitrary picture may instead be HEIF, carry its rotation
 * only as an EXIF tag, be transparent (a PNG screenshot) or be huge. Decoded with the platform
 * decoder (HEIF on Android 9+), turned upright, flattened onto white, capped at [MAX_LONG_SIDE]
 * (processing time grows with the pixel count; this is well above what text needs) and written as
 * JPEG, which OpenCV reads everywhere.
 */
object ImageImporter {
    private const val MAX_LONG_SIDE = 4000

    fun import(inputPath: String, outputPath: String): Map<String, Any> {
        if (!File(inputPath).exists()) throw ImageProcessingException("FILE_NOT_FOUND", "Không tìm thấy ảnh: $inputPath")
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(inputPath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw ImageProcessingException("INVALID_ARGUMENT", "Không đọc được định dạng ảnh này")
        }
        // decode no larger than needed (power-of-two subsampling), then scale exactly below
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= MAX_LONG_SIDE) sample *= 2
        val decoded = BitmapFactory.decodeFile(inputPath, BitmapFactory.Options().apply { inSampleSize = sample })
            ?: throw ImageProcessingException("INVALID_ARGUMENT", "Không đọc được định dạng ảnh này")

        val matrix = Matrix()
        when (ExifInterface(inputPath).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.postRotate(90f); matrix.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.postRotate(270f); matrix.postScale(-1f, 1f) }
        }
        val longSide = max(decoded.width, decoded.height)
        if (longSide > MAX_LONG_SIDE) {
            val s = MAX_LONG_SIDE.toFloat() / longSide
            matrix.postScale(s, s)
        }
        val upright = if (matrix.isIdentity) decoded else Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
        // flattened onto white: transparent areas would otherwise come out black
        val out = Bitmap.createBitmap(upright.width, upright.height, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawColor(Color.WHITE)
            drawBitmap(upright, 0f, 0f, Paint(Paint.FILTER_BITMAP_FLAG))
        }
        try {
            FileOutputStream(outputPath).use { stream ->
                if (!out.compress(Bitmap.CompressFormat.JPEG, 95, stream)) {
                    throw ImageProcessingException("PROCESSING_FAILED", "Không ghi được ảnh")
                }
            }
        } finally {
            if (upright !== decoded) upright.recycle()
            decoded.recycle()
            out.recycle()
        }
        return mapOf("outputPath" to outputPath, "width" to (upright.width), "height" to (upright.height))
    }
}
