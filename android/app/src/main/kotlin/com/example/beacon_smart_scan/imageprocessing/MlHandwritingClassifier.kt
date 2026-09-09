package com.example.beacon_smart_scan.imageprocessing

import android.content.Context
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Runs the trained printed-vs-handwriting CNN (see beacon_smart_scan/ml/) on each word crop.
 * Input: 64(H)x128(W) grayscale, normalized to [0,1] — must match ml/train.py exactly.
 * Output: a single sigmoid in [0,1], 1.0 == handwriting.
 */
object MlHandwritingClassifier {
    private const val MODEL_ASSET = "handwriting_classifier.tflite"
    private const val INPUT_H = 64
    private const val INPUT_W = 128

    // Must match ml/generate_dataset.py's NATIVE_PADDING_PX exactly. A raw ML Kit word box
    // resized straight to INPUT_W x INPUT_H makes the glyph fill ~100% of the frame — verified
    // against a real photo, this alone (not stroke shape) made the model flag nearly every
    // word, print included, as handwriting. Training now crops the same way this does.
    private const val PADDING_PX = 5.0

    @Volatile private var interpreter: Interpreter? = null

    private fun ensureLoaded(context: Context): Interpreter {
        interpreter?.let { return it }
        synchronized(this) {
            interpreter?.let { return it }
            val buffer = context.assets.openFd(MODEL_ASSET).use { fd ->
                FileInputStream(fd.fileDescriptor).channel.use { channel ->
                    channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
            val loaded = Interpreter(buffer)
            interpreter = loaded
            return loaded
        }
    }

    /** [textBlocks] keys: id, left, top, right, bottom (image pixel coordinates). */
    fun classify(context: Context, imagePath: String, textBlocks: List<Map<String, Any>>): List<Map<String, Any>> {
        val src = ImageIO.readOrThrow(imagePath)
        val gray = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            val model = ensureLoaded(context)
            return textBlocks.map { block ->
                val id = block["id"] as? String ?: ""
                val rect = ImageIO.mapToClippedRect(block, gray.width(), gray.height(), PADDING_PX)
                mapOf("id" to id, "mlConfidence" to classifyRegion(model, gray, rect))
            }
        } finally {
            gray.release()
            src.release()
        }
    }

    private fun classifyRegion(model: Interpreter, gray: Mat, rect: Rect): Double {
        val crop = Mat(gray, rect)
        val resized = Mat()
        try {
            Imgproc.resize(crop, resized, Size(INPUT_W.toDouble(), INPUT_H.toDouble()))
            val pixels = ByteArray(INPUT_H * INPUT_W)
            resized.get(0, 0, pixels)

            val input = ByteBuffer.allocateDirect(4 * INPUT_H * INPUT_W).order(ByteOrder.nativeOrder())
            for (b in pixels) {
                input.putFloat((b.toInt() and 0xFF) / 255.0f)
            }
            input.rewind()

            val output = Array(1) { FloatArray(1) }
            model.run(input, output)
            return output[0][0].toDouble()
        } finally {
            resized.release()
            crop.release()
        }
    }
}
