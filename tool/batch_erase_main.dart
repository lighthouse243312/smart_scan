// Dev harness: runs the app's real native erase pipeline on every image in
// <Documents>/batch_in and copies each step's output to <Documents>/batch_out.
//
//   flutter run -d <simulator> -t tool/batch_erase_main.dart
//
// Same calls and parameters as ScanSession (import → sharpen → removeShadow →
// segmentation mask at the default sensitivity → erase), so the outputs match
// what a user gets from the app with no manual mask edits.
import 'dart:io';

import 'package:beacon_smart_scan/core/models/handwriting_method.dart';
import 'package:beacon_smart_scan/services/image_processing_service.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(home: Scaffold(body: Center(child: Text('batch erase…')))));

  final docs = await getApplicationDocumentsDirectory();
  final input = Directory('${docs.path}/batch_in');
  final output = Directory('${docs.path}/batch_out');
  final service = ImageProcessingService();
  // On a device the images are copied in after install (devicectl) — wait for the `.go` marker.
  input.createSync(recursive: true);
  final go = File('${input.path}/.go');
  debugPrint('BATCH_WAIT ${input.path}');
  while (!go.existsSync()) {
    await Future<void>.delayed(const Duration(seconds: 1));
  }
  final token = go.readAsStringSync().trim();       // echoed back in .done for the runner
  go.deleteSync();
  if (output.existsSync()) output.deleteSync(recursive: true);
  output.createSync(recursive: true);
  final files = input.listSync().whereType<File>().where((f) => !f.path.endsWith('/.go')).toList()
    ..sort((a, b) => a.path.compareTo(b.path));

  for (final file in files) {
    final name = file.uri.pathSegments.last.split('.').first;
    final sw = Stopwatch()..start();
    try {
      final original = await service.straighten(await service.importImage(file.path));
      final sharpened = await service.sharpen(original);
      final clean = await service.removeShadow(sharpened);
      final mask = await service.detectHandwritingMask(
        original,
        method: HandwritingMethod.segmentation,
        sensitivity: 0.6,
      );
      final erased = await service.eraseWithMask(clean, analysisPath: original, maskPath: mask.maskPath);
      File(original).copySync('${output.path}/${name}_original.png');
      File(clean).copySync('${output.path}/${name}_clean.png');
      File(mask.maskPath).copySync('${output.path}/${name}_mask.png');
      File(erased).copySync('${output.path}/${name}_erased.png');
      debugPrint('BATCH_OK $name ${sw.elapsedMilliseconds}ms coverage=${mask.coverage}');
      file.deleteSync();          // processed: the next run starts from an empty inbox
    } catch (e) {
      debugPrint('BATCH_FAIL $name $e');
    }
  }
  File('${output.path}/.done').writeAsStringSync(token);
  debugPrint('BATCH_DONE ${files.length}');
}
