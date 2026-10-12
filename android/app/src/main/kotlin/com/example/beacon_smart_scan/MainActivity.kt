package com.example.beacon_smart_scan

import android.app.Activity
import android.content.Intent
import android.os.Handler
import android.os.Looper
import com.example.beacon_smart_scan.imageprocessing.ImageProcessingChannel
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File

class MainActivity : FlutterActivity() {
    /** The Dart call waiting for the document picker (one at a time). */
    private var pendingPick: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        ImageProcessingChannel.register(flutterEngine, this)
        // the system document picker for a PDF (no storage permission needed); answers a path to a
        // copy in the app's cache, or null if the user cancelled. Mirrors AppDelegate.swift on iOS.
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, DOCUMENTS_CHANNEL).setMethodCallHandler { call, result ->
            if (call.method != "pickPdf") return@setMethodCallHandler result.notImplemented()
            if (pendingPick != null) return@setMethodCallHandler result.error("BUSY", "Đang mở trình chọn file", null)
            pendingPick = result
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "application/pdf"
            }
            startActivityForResult(intent, PICK_PDF_REQUEST)
        }
    }

    @Deprecated("FlutterActivity has no ActivityResult API")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != PICK_PDF_REQUEST) return
        val result = pendingPick ?: return
        pendingPick = null
        val uri = data?.data
        if (resultCode != Activity.RESULT_OK || uri == null) return result.success(null)
        // copied off the main thread: a file from a cloud provider may download first
        val main = Handler(Looper.getMainLooper())
        Thread {
            try {
                val target = File(cacheDir, "picked_${System.currentTimeMillis()}.pdf")
                val input = contentResolver.openInputStream(uri) ?: error("Không mở được file")
                input.use { target.outputStream().use { out -> it.copyTo(out) } }
                main.post { result.success(target.path) }
            } catch (e: Exception) {
                main.post { result.error("FILE_NOT_FOUND", "Không mở được file: ${e.message}", null) }
            }
        }.start()
    }

    private companion object {
        const val DOCUMENTS_CHANNEL = "beacon_smart_scan/documents"
        const val PICK_PDF_REQUEST = 4201
    }
}
