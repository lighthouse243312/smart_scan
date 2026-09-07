import 'dart:io';

import 'package:path_provider/path_provider.dart';

/// Generates fresh file paths in the app's temp dir for intermediate processing steps —
/// the native platform channel always works with file paths, never raw byte buffers.
class TempPaths {
  TempPaths._();

  static Future<String> next(String suffix) async {
    final dir = await getTemporaryDirectory();
    final scanDir = Directory('${dir.path}/smart_scan');
    if (!await scanDir.exists()) {
      await scanDir.create(recursive: true);
    }
    final name = '${DateTime.now().microsecondsSinceEpoch}_$suffix';
    return '${scanDir.path}/$name';
  }
}
