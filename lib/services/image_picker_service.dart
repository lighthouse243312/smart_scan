import 'package:image_picker/image_picker.dart';

/// Picks existing pictures from the photo library (the system picker on iOS 14+ / Android 13+,
/// which needs no permission). Returns file paths, empty if the user cancelled.
class ImagePickerService {
  ImagePickerService({ImagePicker? picker}) : _picker = picker ?? ImagePicker();

  final ImagePicker _picker;

  /// Original files, full quality: no resizing or re-compression here — the native import step
  /// decodes (HEIC too), turns upright and caps the size in one place for both platforms.
  Future<List<String>> pickImages({int limit = 10}) async {
    final files = await _picker.pickMultiImage(limit: limit, requestFullMetadata: false);
    return files.map((f) => f.path).toList();
  }
}
