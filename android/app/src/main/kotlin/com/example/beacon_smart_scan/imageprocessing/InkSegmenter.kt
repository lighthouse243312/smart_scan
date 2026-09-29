package com.example.beacon_smart_scan.imageprocessing

import android.content.Context
import com.example.beacon_smart_scan.imageprocessing.HandwritingMask.releasing
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Segmentation method: the bundled ink_segmenter.tflite U-Net (see ml/train_ink_seg.py)
 * predicts two independent per-pixel layers, print and handwriting. Mirrors
 * `segmentationMaskAtPath` in ios/Runner/ImageProcessingOpenCV.mm.
 *
 * Model contract (must match ml/export_ink_seg.py): input float32 NHWC [1, 256, 256, 3], RGB,
 * raw 0-255; output float32 [1, 256, 256, 2] independent sigmoids — channel 0 print, channel 1
 * handwriting (both can be set where handwriting crosses print).
 * The page is resized so its longer side is [LONG_SIDE] px and processed in overlapping tiles.
 */
object InkSegmenter {
    private const val MODEL_ASSET = "ink_segmenter.tflite"
    private const val LONG_SIDE = 1536
    private const val TILE = 256
    private const val OVERLAP = 32
    private const val CHANNEL_COUNT = 2
    private const val PRINT_CHANNEL = 0
    private const val HANDWRITING_CHANNEL = 1
    private const val PRINT_THRESHOLD = 0.5

    @Volatile private var interpreter: Interpreter? = null

    private fun ensureLoaded(context: Context): Interpreter {
        interpreter?.let { return it }
        synchronized(this) {
            interpreter?.let { return it }
            val buffer = try {
                context.assets.openFd(MODEL_ASSET).use { fd ->
                    FileInputStream(fd.fileDescriptor).channel.use { channel ->
                        channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                    }
                }
            } catch (e: FileNotFoundException) {
                throw ImageProcessingException(
                    "PROCESSING_FAILED",
                    "Chưa có model $MODEL_ASSET trong app (chạy ml/export_ink_seg.py rồi build lại)",
                )
            }
            val loaded = Interpreter(buffer, Interpreter.Options().setNumThreads(4))
            interpreter = loaded
            return loaded
        }
    }

    fun mask(context: Context, inputPath: String, maskPath: String, threshold: Double, colorDelta: Double): Map<String, Any> {
        val model = ensureLoaded(context)
        val src = ImageIO.readOrThrow(inputPath)
        val printProb = Mat()
        val handwritingProb = Mat()
        val gray = Mat()
        val ink = Mat()
        val handwriting = Mat()
        val print = Mat()
        return releasing(src, printProb, handwritingProb, gray, ink, handwriting, print) {
            segmentPage(model, src, printProb, handwritingProb)

            // The model runs at a reduced scale, so its masks are soft at full resolution — snap
            // both to the page's actual ink so erase touches strokes, not blobs of paper.
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            val contrast = HandwritingMask.inkContrast(gray)
            releasing(contrast) {
                Imgproc.threshold(contrast, ink, HandwritingMask.INK_CONTRAST_LOOSE, 255.0, Imgproc.THRESH_BINARY)
            }
            Imgproc.threshold(handwritingProb, handwriting, threshold, 255.0, Imgproc.THRESH_BINARY)
            handwriting.convertTo(handwriting, CvType.CV_8U)
            Core.bitwise_and(handwriting, ink, handwriting)
            HandwritingMask.removeSmallComponents(handwriting, HandwritingMask.minSpeckleArea(src))
            Imgproc.threshold(printProb, print, PRINT_THRESHOLD, 255.0, Imgproc.THRESH_BINARY)
            print.convertTo(print, CvType.CV_8U)
            Core.bitwise_and(print, ink, print)

            // combined with the ink-colour + layout method (see ios segmentationMaskAtPath): colour
            // finds neat pen writing the model misses, the model finds pen ink the same colour as
            // the print. A model pixel is dropped only where BOTH agree it is print.
            val (colorHw, colorPrint, colorOverlap) = InkAnalysis.detectByInkColor(src, colorDelta)
            val tmp = Mat(); val hw = Mat(); val overlap = Mat(); val outPrint = Mat(); val notHw = Mat(); val notColorHw = Mat()
            releasing(colorHw, colorPrint, colorOverlap, tmp, hw, overlap, outPrint, notHw, notColorHw) {
                Core.bitwise_and(colorPrint, print, tmp)
                Core.bitwise_not(tmp, tmp)
                Core.bitwise_and(handwriting, tmp, tmp)
                Core.bitwise_or(colorHw, tmp, hw)
                // print under the pen: colour's overlap, plus where the model's print layer is set
                // under model-only handwriting
                Core.bitwise_not(colorHw, notColorHw)
                Core.bitwise_and(hw, print, tmp)
                Core.bitwise_and(tmp, notColorHw, tmp)
                Core.bitwise_or(colorOverlap, tmp, overlap)
                Core.bitwise_or(colorPrint, print, outPrint)
                Core.bitwise_not(hw, notHw)
                Core.bitwise_and(outPrint, notHw, outPrint)
                Core.bitwise_or(outPrint, overlap, outPrint)
                HandwritingMask.write(hw, outPrint, overlap, maskPath)
                HandwritingMask.result(maskPath, hw)
            }
        }
    }

    /**
     * Fills [outPrint] / [outHandwriting] (CV_32F, original size) with the per-pixel print and
     * handwriting probabilities.
     */
    private fun segmentPage(model: Interpreter, src: Mat, outPrint: Mat, outHandwriting: Mat) {
        val scale = LONG_SIDE.toDouble() / max(src.cols(), src.rows()).toDouble()
        val work = Mat()
        val padded = Mat()
        val paddedFloat = Mat()
        val printSum = Mat()
        val handwritingSum = Mat()
        val weightSum = Mat()
        val weights = tileWeights()
        val printPlane = Mat(TILE, TILE, CvType.CV_32F)
        val handwritingPlane = Mat(TILE, TILE, CvType.CV_32F)
        val weighted = Mat()
        val prob = Mat()
        releasing(
            work, padded, paddedFloat, printSum, handwritingSum, weightSum, weights,
            printPlane, handwritingPlane, weighted, prob,
        ) {
            Imgproc.resize(
                src, work, org.opencv.core.Size(), scale, scale,
                if (scale < 1.0) Imgproc.INTER_AREA else Imgproc.INTER_CUBIC,
            )
            Imgproc.cvtColor(work, work, Imgproc.COLOR_BGR2RGB)

            val step = TILE - OVERLAP
            val paddedW = max(TILE, ceil((work.cols() - TILE) / step.toDouble()).toInt() * step + TILE)
            val paddedH = max(TILE, ceil((work.rows() - TILE) / step.toDouble()).toInt() * step + TILE)
            Core.copyMakeBorder(
                work, padded, 0, paddedH - work.rows(), 0, paddedW - work.cols(),
                Core.BORDER_CONSTANT, Scalar(255.0, 255.0, 255.0),
            )
            padded.convertTo(paddedFloat, CvType.CV_32FC3)
            for (sum in listOf(printSum, handwritingSum, weightSum)) {
                sum.create(paddedH, paddedW, CvType.CV_32F)
                sum.setTo(Scalar(0.0))
            }

            val input = ByteBuffer.allocateDirect(4 * TILE * TILE * 3).order(ByteOrder.nativeOrder())
            val output = ByteBuffer.allocateDirect(4 * TILE * TILE * CHANNEL_COUNT).order(ByteOrder.nativeOrder())
            val tilePixels = FloatArray(TILE * TILE * 3)
            val outputValues = FloatArray(TILE * TILE * CHANNEL_COUNT)
            val printValues = FloatArray(TILE * TILE)
            val handwritingValues = FloatArray(TILE * TILE)

            var y = 0
            while (y + TILE <= paddedH) {
                var x = 0
                while (x + TILE <= paddedW) {
                    val roi = Rect(x, y, TILE, TILE)
                    val tile = paddedFloat.submat(roi).clone()
                    releasing(tile) { tile.get(0, 0, tilePixels) }
                    input.rewind()
                    input.asFloatBuffer().put(tilePixels)
                    output.rewind()
                    model.run(input, output)
                    output.rewind()
                    output.asFloatBuffer().get(outputValues)
                    for (i in printValues.indices) {
                        printValues[i] = outputValues[i * CHANNEL_COUNT + PRINT_CHANNEL]
                        handwritingValues[i] = outputValues[i * CHANNEL_COUNT + HANDWRITING_CHANNEL]
                    }
                    printPlane.put(0, 0, printValues)
                    handwritingPlane.put(0, 0, handwritingValues)

                    for ((plane, sum) in listOf(printPlane to printSum, handwritingPlane to handwritingSum)) {
                        Core.multiply(plane, weights, weighted)
                        val sumRoi = sum.submat(roi)
                        releasing(sumRoi) { Core.add(sumRoi, weighted, sumRoi) }
                    }
                    val weightRoi = weightSum.submat(roi)
                    releasing(weightRoi) { Core.add(weightRoi, weights, weightRoi) }
                    x += step
                }
                y += step
            }

            val valid = Rect(0, 0, work.cols(), work.rows())
            for ((sum, out) in listOf(printSum to outPrint, handwritingSum to outHandwriting)) {
                Core.divide(sum, weightSum, prob)
                val cropped = prob.submat(valid)
                releasing(cropped) {
                    Imgproc.resize(cropped, out, src.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
                }
            }
        }
    }

    /**
     * Tile blending weight: 1 in the tile's interior, tapering towards its edges across the overlap
     * band, so each pixel is decided mostly by the tile that sees it with the most context.
     */
    private fun tileWeights(): Mat {
        val values = FloatArray(TILE * TILE)
        for (y in 0 until TILE) {
            val wy = min(min(y + 1, TILE - y), OVERLAP) / OVERLAP.toFloat()
            for (x in 0 until TILE) {
                val wx = min(min(x + 1, TILE - x), OVERLAP) / OVERLAP.toFloat()
                values[y * TILE + x] = wy * wx
            }
        }
        val weights = Mat(TILE, TILE, CvType.CV_32F)
        weights.put(0, 0, values)
        return weights
    }
}
