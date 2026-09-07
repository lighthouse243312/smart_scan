import 'package:gal/gal.dart';

/// Saves a finished scan page to the device's photo gallery.
class ExportService {
  Future<void> saveToGallery(String imagePath) async {
    final hasAccess = await Gal.hasAccess();
    if (!hasAccess) {
      final granted = await Gal.requestAccess();
      if (!granted) {
        throw ExportException('Chưa cấp quyền truy cập thư viện ảnh');
      }
    }
    await Gal.putImage(imagePath, album: 'Beacon Smart Scan');
  }
}

class ExportException implements Exception {
  ExportException(this.message);
  final String message;

  @override
  String toString() => message;
}
