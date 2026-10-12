package com.example.beacon_smart_scan.imageprocessing

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.pdf.PdfRenderer
import android.os.ParcelFileDescriptor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Brings a PDF into the pipeline: every page rendered by the platform renderer at [DPI], capped
 * at [MAX_LONG_SIDE] like an imported picture (see [ImageImporter]), on white, as lossless PNG.
 * Mirrors `renderPdf` in ios/Runner/ImageProcessingChannel.swift.
 */
object PdfImporter {
    private const val DPI = 200
    private const val MAX_LONG_SIDE = 4000

    /** Renders the pages of [inputPath] into [outputDir] as `<prefix>_<n>.png`; returns their paths, in order. */
    fun render(inputPath: String, outputDir: String, prefix: String): Map<String, Any> {
        val file = File(inputPath)
        if (!file.exists()) throw ImageProcessingException("FILE_NOT_FOUND", "Không tìm thấy file: $inputPath")
        File(outputDir).mkdirs()
        val descriptor = ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
        val renderer = try {
            PdfRenderer(descriptor)
        } catch (e: SecurityException) {
            descriptor.close()
            throw ImageProcessingException("INVALID_ARGUMENT", "File PDF có mật khẩu, không mở được")
        } catch (e: IOException) {
            descriptor.close()
            throw ImageProcessingException("INVALID_ARGUMENT", "Không đọc được file PDF này")
        }
        val paths = ArrayList<String>()
        try {
            for (i in 0 until renderer.pageCount) {
                renderer.openPage(i).use { page ->
                    // page size is in points (1/72 inch)
                    val scale = min(DPI / 72.0, MAX_LONG_SIDE.toDouble() / max(page.width, page.height))
                    val w = max(1, (page.width * scale).roundToInt())
                    val h = max(1, (page.height * scale).roundToInt())
                    val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    try {
                        // the renderer leaves the page transparent where nothing is drawn
                        bitmap.eraseColor(Color.WHITE)
                        val matrix = Matrix().apply { setScale(w.toFloat() / page.width, h.toFloat() / page.height) }
                        page.render(bitmap, null, matrix, PdfRenderer.Page.RENDER_MODE_FOR_PRINT)
                        val out = File(outputDir, "${prefix}_${i + 1}.png").path
                        FileOutputStream(out).use { stream ->
                            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
                                throw ImageProcessingException("PROCESSING_FAILED", "Không ghi được ảnh")
                            }
                        }
                        paths.add(out)
                    } finally {
                        bitmap.recycle()
                    }
                }
            }
        } finally {
            renderer.close()
            descriptor.close()
        }
        if (paths.isEmpty()) throw ImageProcessingException("INVALID_ARGUMENT", "File PDF không có trang nào")
        return mapOf("pages" to paths)
    }
}
