package com.example.beacon_smart_scan

import android.app.Instrumentation
import android.os.Bundle
import com.example.beacon_smart_scan.imageprocessing.ImageIO
import com.example.beacon_smart_scan.imageprocessing.InkAnalysis
import com.example.beacon_smart_scan.imageprocessing.InkColorMasker
import com.example.beacon_smart_scan.imageprocessing.InkSegmenter
import com.example.beacon_smart_scan.imageprocessing.MaskEraser
import com.example.beacon_smart_scan.imageprocessing.MultiPassEraser
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
            if (args.getString("bench") == "filter") {
                // -e bench filter: filter2D timings on a page-sized float plane, the refine step's hot spot
                val sb = StringBuilder("threads ${org.opencv.core.Core.getNumThreads()}\n")
                val src = Mat(3711, 2679, CvType.CV_32F)
                org.opencv.core.Core.randu(src, 0.0, 1.0)
                for (len in intArrayOf(9, 19, 37, 55)) {
                    val kernel = Mat.zeros(len, len, CvType.CV_32F)
                    for (o in -1..1) for (x in 0 until len) kernel.put(len / 2 + o, x, 1.0)
                    val dst = Mat()
                    for (rep in 0 until 2) {
                        val t = System.nanoTime()
                        org.opencv.imgproc.Imgproc.filter2D(src, dst, CvType.CV_32F, kernel, org.opencv.core.Point(-1.0, -1.0), 0.0, org.opencv.core.Core.BORDER_CONSTANT)
                        if (rep == 1) sb.append("filter2D ${len}x$len: ${(System.nanoTime() - t) / 1_000_000} ms\n")
                    }
                    kernel.release(); dst.release()
                }
                src.release()
                // the erase's twin search: masked SQDIFF of a 27 px patch over a half-resolution page
                val half = Mat(1856, 1340, CvType.CV_32F); org.opencv.core.Core.randu(half, 0.0, 1.0)
                val mask = Mat(27, 27, CvType.CV_32F); org.opencv.core.Core.randu(mask, 0.0, 1.0)
                org.opencv.imgproc.Imgproc.threshold(mask, mask, 0.2, 1.0, org.opencv.imgproc.Imgproc.THRESH_BINARY)
                val patch = half.submat(500, 527, 600, 627).clone()
                val res = Mat()
                for (masked in listOf(true, false)) for (rep in 0 until 3) {
                    val t = System.nanoTime()
                    if (masked) org.opencv.imgproc.Imgproc.matchTemplate(half, patch, res, org.opencv.imgproc.Imgproc.TM_SQDIFF, mask)
                    else org.opencv.imgproc.Imgproc.matchTemplate(half, patch, res, org.opencv.imgproc.Imgproc.TM_SQDIFF)
                    if (rep == 2) sb.append("matchTemplate ${if (masked) "masked" else "unmasked"} 27x27: ${(System.nanoTime() - t) / 1_000_000} ms\n")
                }
                listOf(half, mask, patch, res).forEach { it.release() }
                out.putString(REPORT_KEY_STREAMRESULT, sb.toString())
                finish(0, out)
                return
            }
            val context = targetContext
            val dir = File(context.getExternalFilesDir(null), args.getString("dir") ?: "harness")
            val input = File(dir, args.getString("input") ?: error("missing -e input")).path
            val clean = args.getString("clean")?.let { File(dir, it).path } ?: input
            val s = (args.getString("sensitivity") ?: "0.6").toDouble().coerceIn(0.0, 1.0)
            val base = input.substringBeforeLast('.') + (args.getString("tag") ?: "")
            args.getString("usemask")?.let { m ->
                // erase only, with a given mask (e.g. one pulled from the app's cache)
                MaskEraser.erase(clean, input, File(dir, m).path, "${base}_erased.png")
                out.putString(REPORT_KEY_STREAMRESULT, "erase-only done\n"); finish(0, out); return
            }
            com.example.beacon_smart_scan.imageprocessing.PenComponentVote.enabled = args.getString("vote") != "0"
            com.example.beacon_smart_scan.imageprocessing.PrintRetypeset.enabled = args.getString("retypeset") != "0"
            com.example.beacon_smart_scan.imageprocessing.PrintRecognition.enabled = args.getString("recognition") != "0"
            com.example.beacon_smart_scan.imageprocessing.StrokeVote.enabled = args.getString("strokevote") != "0"
            com.example.beacon_smart_scan.imageprocessing.HandwritingMask.consistent = args.getString("consistent") != "0"
            com.example.beacon_smart_scan.imageprocessing.RuleRedraw.enabled = args.getString("rules") != "0"
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
            val report = StringBuilder("mask ${t1 - t0} ms, erase ${t2 - t1} ms, coverage ${result["coverage"]}\n")
            // -e passes N: the app's multi-pass erase (lib/state/scan_session.dart eraseHandwriting)
            val passes = args.getString("passes")?.toInt() ?: 1
            if (passes > 1 && args.getString("erase") != "0") {
                val tp = System.currentTimeMillis()
                val r = MultiPassEraser.erase(context, clean, input, mask, "${base}_final.png",
                    args.getString("method") != "color", 0.8 - 0.6 * s, 0.08 - 0.06 * s)
                report.append("multi-pass: ${r["passes"]} passes, ${System.currentTimeMillis() - tp} ms\n")
            }
            out.putString(REPORT_KEY_STREAMRESULT, report.toString())
            finish(0, out)
        } catch (t: Throwable) {
            out.putString(REPORT_KEY_STREAMRESULT, "FAILED: ${t.stackTraceToString()}\n")
            finish(1, out)
        }
    }
}
