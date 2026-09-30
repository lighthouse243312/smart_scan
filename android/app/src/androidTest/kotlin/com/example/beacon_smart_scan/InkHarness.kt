package com.example.beacon_smart_scan

import android.app.Instrumentation
import android.os.Bundle
import com.example.beacon_smart_scan.imageprocessing.ImageIO
import com.example.beacon_smart_scan.imageprocessing.InkAnalysis
import com.example.beacon_smart_scan.imageprocessing.InkColorMasker
import com.example.beacon_smart_scan.imageprocessing.InkSegmenter
import com.example.beacon_smart_scan.imageprocessing.MaskEraser
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

/**
 * Dev harness: runs the native mask + erase pipeline on a page pushed to the device, inside the
 * app's own process (same heap limits as the app), so detection changes can be checked on real
 * photos without driving the UI.
 *
 *   adb push page.jpg clean.png /sdcard/Android/data/com.example.beacon_smart_scan/files/harness/
 *   adb shell am instrument -w -e dir harness -e input page.jpg -e clean clean.png \
 *       com.example.beacon_smart_scan.test/com.example.beacon_smart_scan.InkHarness
 *
 * Optional: -e method color|segmentation (default segmentation), -e sensitivity 0.6.
 * Writes <input>_mask.png and <input>_erased.png next to the input.
 */
class InkHarness : Instrumentation() {
    private lateinit var args: Bundle

    override fun onCreate(arguments: Bundle) {
        super.onCreate(arguments)
        args = arguments
        start()
    }

    override fun onStart() {
        super.onStart()
        val out = Bundle()
        try {
            check(OpenCVLoader.initLocal()) { "OpenCV failed to load" }
            val context = targetContext
            val dir = File(context.getExternalFilesDir(null), args.getString("dir") ?: "harness")
            val input = File(dir, args.getString("input") ?: error("missing -e input")).path
            val clean = args.getString("clean")?.let { File(dir, it).path } ?: input
            val s = (args.getString("sensitivity") ?: "0.6").toDouble().coerceIn(0.0, 1.0)
            val base = input.substringBeforeLast('.')
            val mask = "${base}_mask.png"
            if (args.getString("layers") == "1") {
                // the two detectors separately: model probabilities and the colour method's layers
                val src = ImageIO.readOrThrow(input)
                val probs = InkSegmenter.probabilities(context, src)!!
                for ((m, name) in listOf(probs.first to "model_print", probs.second to "model_hw")) {
                    val u8 = Mat(); m.convertTo(u8, CvType.CV_8U, 255.0); Imgcodecs.imwrite("${base}_$name.png", u8); u8.release()
                }
                val (hw, print, overlap) = InkAnalysis.detectByInkColor(src, 0.08 - 0.06 * s, probs.first, probs.second)
                Imgcodecs.imwrite("${base}_color_hw.png", hw)
                Imgcodecs.imwrite("${base}_color_print.png", print)
                Imgcodecs.imwrite("${base}_color_overlap.png", overlap)
            }
            val t0 = System.currentTimeMillis()
            val result = when (args.getString("method") ?: "segmentation") {
                "color" -> InkColorMasker.mask(context, input, mask, 0.08 - 0.06 * s)
                else -> InkSegmenter.mask(context, input, mask, 0.8 - 0.6 * s, 0.08 - 0.06 * s)
            }
            val t1 = System.currentTimeMillis()
            if (args.getString("erase") != "0") MaskEraser.erase(clean, input, mask, "${base}_erased.png")
            val t2 = System.currentTimeMillis()
            out.putString(REPORT_KEY_STREAMRESULT, "mask ${t1 - t0} ms, erase ${t2 - t1} ms, coverage ${result["coverage"]}\n")
            finish(0, out)
        } catch (t: Throwable) {
            out.putString(REPORT_KEY_STREAMRESULT, "FAILED: ${t.stackTraceToString()}\n")
            finish(1, out)
        }
    }
}
