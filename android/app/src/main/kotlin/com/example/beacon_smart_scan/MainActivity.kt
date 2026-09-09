package com.example.beacon_smart_scan

import com.example.beacon_smart_scan.imageprocessing.ImageProcessingChannel
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        ImageProcessingChannel.register(flutterEngine, this)
    }
}
