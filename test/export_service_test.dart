import 'dart:io';
import 'dart:ui' as ui;

import 'package:beacon_smart_scan/services/export_service.dart';
import 'package:flutter_test/flutter_test.dart';

Future<String> _png(Directory dir, String name, int w, int h) async {
  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);
  canvas.drawRect(ui.Rect.fromLTWH(0, 0, w.toDouble(), h.toDouble()), ui.Paint()..color = const ui.Color(0xFFFFFFFF));
  canvas.drawRect(ui.Rect.fromLTWH(10, 10, w / 2, h / 4), ui.Paint()..color = const ui.Color(0xFF222222));
  final image = await recorder.endRecording().toImage(w, h);
  final data = await image.toByteData(format: ui.ImageByteFormat.png);
  final f = File('${dir.path}/$name.png');
  await f.writeAsBytes(data!.buffer.asUint8List());
  return f.path;
}

void main() {
  testWidgets('builds pdf, docx and png exports', (tester) async {
    await tester.runAsync(() async {
      final dir = Directory(Platform.environment['EXPORT_TEST_DIR'] ?? Directory.systemTemp.createTempSync('exp').path)..createSync(recursive: true);
      final pages = [await _png(dir, 'a', 300, 420), await _png(dir, 'b', 420, 300)];
      final svc = ExportService();
      for (final f in ExportFormat.values) {
        final files = await svc.build(pages, f, baseName: 'test', outDir: dir);
        expect(files, isNotEmpty);
        for (final file in files) {
          expect(file.lengthSync(), greaterThan(100));
        }
      }
    });
  });
}
