package com.example.beacon_smart_scan.imageprocessing

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognizer as MlKitClient
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import java.text.BreakIterator

/**
 * [TextRecognizer] on ML Kit (on-device, bundled models). ML Kit recognises one script family per
 * model: Latin (which covers Vietnamese and the other Latin-script languages) is tried first; when
 * it reads the page poorly the other scripts are tried and the one that reads the most is kept for
 * the rest of this erase (the original is read with the same one). ML Kit gives one reading per
 * line, words boxed. Must be called off the main thread (it blocks on the recogniser).
 */
class MlKitTextRecognizer : TextRecognizer {
    private val latin by lazy { TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS) }
    private val others by lazy {
        listOf(
            TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build()),
            TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build()),
            TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()),
            TextRecognition.getClient(DevanagariTextRecognizerOptions.Builder().build()),
        )
    }
    private var chosen: MlKitClient? = null

    override fun recognize(bgr: Mat): List<OcrLine> {
        val rgba = Mat()
        Imgproc.cvtColor(bgr, rgba, Imgproc.COLOR_BGR2RGBA)
        val bitmap = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bitmap)
        rgba.release()
        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            val client = chosen
            if (client != null) return lines(Tasks.await(client.process(image)))
            var best = Tasks.await(latin.process(image))
            var bestClient: MlKitClient = latin
            // (a page Latin reads with low confidence may be in another script)
            if (confidence(best) < 0.7f) {
                for (other in others) {
                    val text = Tasks.await(other.process(image))
                    if (score(text) > score(best)) { best = text; bestClient = other }
                }
            }
            chosen = bestClient
            return lines(best)
        } finally {
            bitmap.recycle()
        }
    }

    private fun elements(text: Text) = text.textBlocks.flatMap { b -> b.lines.flatMap { it.elements } }

    private fun confidence(text: Text): Float {
        val e = elements(text)
        if (e.isEmpty()) return 0f
        return e.sumOf { it.confidence.toDouble() * it.text.length }.toFloat() / e.sumOf { it.text.length }
    }

    private fun score(text: Text): Float = elements(text).sumOf { it.confidence.toDouble() * it.text.length }.toFloat()

    private fun graphemes(s: String): List<String> {
        val it = BreakIterator.getCharacterInstance()
        it.setText(s)
        val out = ArrayList<String>()
        var start = it.first()
        var end = it.next()
        while (end != BreakIterator.DONE) {
            val g = s.substring(start, end)
            if (g.isNotBlank()) out.add(g)
            start = end
            end = it.next()
        }
        return out
    }

    private fun lines(text: Text): List<OcrLine> {
        val out = ArrayList<OcrLine>()
        for (block in text.textBlocks) for (line in block.lines) {
            val lb = line.boundingBox ?: continue
            val words = line.elements.mapNotNull { e ->
                val b = e.boundingBox ?: return@mapNotNull null
                val g = graphemes(e.text)
                if (g.isEmpty()) null else OcrWord(g, Rect(b.left, b.top, b.width(), b.height()))
            }
            out.add(OcrLine(listOf(words), Rect(lb.left, lb.top, lb.width(), lb.height())))
        }
        return out
    }
}
