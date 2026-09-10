package com.example.beacon_smart_scan.imageprocessing

import android.content.Context
import android.os.Handler
import android.os.Looper
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.util.concurrent.Executors
import org.opencv.android.OpenCVLoader

/** Errors thrown by the native image-processing pipeline, mapped 1:1 to a [MethodChannel.Result.error] code. */
class ImageProcessingException(val code: String, message: String) : Exception(message)

/**
 * Bridges Dart to the native OpenCV pipeline via a single MethodChannel.
 * Every call is dispatched onto [executor] (never the platform/UI thread) since these are
 * multi-hundred-ms to multi-second operations on full-resolution scan images.
 */
object ImageProcessingChannel {
    private const val CHANNEL_NAME = "beacon_smart_scan/image_processing"
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    @Volatile private var openCvLoaded = false
    private lateinit var appContext: Context

    fun register(flutterEngine: FlutterEngine, context: Context) {
        appContext = context.applicationContext
        val channel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL_NAME)
        channel.setMethodCallHandler { call, result ->
            executor.execute {
                try {
                    ensureOpenCvLoaded()
                    val response = handle(call)
                    mainHandler.post { result.success(response) }
                } catch (e: ImageProcessingException) {
                    mainHandler.post { result.error(e.code, e.message, null) }
                } catch (e: Exception) {
                    mainHandler.post { result.error("PROCESSING_FAILED", e.message, null) }
                }
            }
        }
    }

    private fun ensureOpenCvLoaded() {
        if (openCvLoaded) return
        if (!OpenCVLoader.initLocal()) {
            throw ImageProcessingException("PROCESSING_FAILED", "Không thể khởi tạo thư viện OpenCV")
        }
        openCvLoaded = true
    }

    private fun handle(call: MethodCall): Any {
        return when (call.method) {
            "sharpen" ->
                Sharpener.sharpen(
                    requirePath(call, "inputPath"),
                    requirePath(call, "outputPath"),
                    requireDouble(call, "amount", 1.5),
                    requireDouble(call, "radius", 3.0),
                )
            "removeShadow" ->
                ShadowRemover.removeShadow(
                    requirePath(call, "inputPath"),
                    requirePath(call, "outputPath"),
                )
            "rotate" ->
                Rotator.rotate(
                    requirePath(call, "inputPath"),
                    requirePath(call, "outputPath"),
                    (call.argument<Any>("quarterTurnsClockwise") as? Number)?.toInt() ?: 1,
                )
            "detectOrphanRegions" ->
                OrphanInkDetector.detect(
                    requirePath(call, "imagePath"),
                    requireMapList(call, "existingBlocks"),
                )
            "detectHandwritingRegions" ->
                HandwritingDetector.detect(
                    requirePath(call, "imagePath"),
                    requireMapList(call, "textBlocks"),
                )
            "classifyHandwriting" ->
                MlHandwritingClassifier.classify(
                    appContext,
                    requirePath(call, "imagePath"),
                    requireMapList(call, "textBlocks"),
                )
            "eraseRegions" ->
                Inpainter.erase(
                    requirePath(call, "inputPath"),
                    requirePath(call, "outputPath"),
                    requireMapList(call, "rects"),
                    requireMapList(call, "keepRects"),
                    requireDouble(call, "padding", 6.0),
                    requireDouble(call, "inpaintRadius", 5.0),
                )
            else -> throw ImageProcessingException("INVALID_ARGUMENT", "Unknown method: ${call.method}")
        }
    }

    private fun requirePath(call: MethodCall, key: String): String =
        call.argument<String>(key)
            ?: throw ImageProcessingException("INVALID_ARGUMENT", "Missing argument: $key")

    private fun requireDouble(call: MethodCall, key: String, default: Double): Double =
        (call.argument<Any>(key) as? Number)?.toDouble() ?: default

    @Suppress("UNCHECKED_CAST")
    private fun requireMapList(call: MethodCall, key: String): List<Map<String, Any>> =
        (call.argument<List<Any>>(key) ?: emptyList()) as List<Map<String, Any>>
}
