import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:archive/archive.dart';
import 'package:gal/gal.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:share_plus/share_plus.dart';

enum ExportFormat { pdf, word, png }

/// Builds shareable files from the finished pages (images on disk) and hands them to the system
/// share sheet, or saves them to the photo gallery.
class ExportService {
  /// Writes [imagePaths] as [format] into a fresh export folder; returns the files to share.
  Future<List<File>> build(List<String> imagePaths, ExportFormat format, {required String baseName, Directory? outDir}) async {
    final dir = outDir ?? await _exportDir();
    return switch (format) {
      ExportFormat.pdf => [await _pdf(imagePaths, File('${dir.path}/$baseName.pdf'))],
      ExportFormat.word => [await _docx(imagePaths, File('${dir.path}/$baseName.docx'))],
      ExportFormat.png => [
          for (var i = 0; i < imagePaths.length; i++)
            await _png(imagePaths[i], File('${dir.path}/${baseName}_${i + 1}.png')),
        ],
    };
  }

  Future<void> share(List<File> files, {required String subject}) async {
    await SharePlus.instance.share(ShareParams(
      files: [for (final f in files) XFile(f.path)],
      subject: subject,
    ));
  }

  /// Throws [GalleryAccessDenied] when the user refuses photo-library access.
  Future<void> saveToGallery(List<String> imagePaths) async {
    final hasAccess = await Gal.hasAccess();
    if (!hasAccess) {
      final granted = await Gal.requestAccess();
      if (!granted) throw GalleryAccessDenied();
    }
    for (final p in imagePaths) {
      await Gal.putImage(p, album: 'Smart Scan');
    }
  }

  Future<Directory> _exportDir() async {
    final base = await getTemporaryDirectory();
    final dir = Directory('${base.path}/exports/${DateTime.now().millisecondsSinceEpoch}');
    await dir.create(recursive: true);
    return dir;
  }

  /// One A4 page per image, the image fitted inside a small margin.
  Future<File> _pdf(List<String> imagePaths, File out) async {
    final doc = pw.Document(title: 'Smart Scan', creator: 'Smart Scan');
    for (final path in imagePaths) {
      final image = pw.MemoryImage(await File(path).readAsBytes());
      doc.addPage(pw.Page(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.all(18),
        build: (_) => pw.Center(child: pw.Image(image, fit: pw.BoxFit.contain)),
      ));
    }
    return out.writeAsBytes(await doc.save());
  }

  /// PNG copy (the pipeline already writes PNG; anything else is re-encoded).
  Future<File> _png(String path, File out) async {
    if (path.toLowerCase().endsWith('.png')) return File(path).copy(out.path);
    final codec = await ui.instantiateImageCodec(await File(path).readAsBytes());
    final frame = await codec.getNextFrame();
    final data = await frame.image.toByteData(format: ui.ImageByteFormat.png);
    return out.writeAsBytes(data!.buffer.asUint8List());
  }

  /// A minimal Office Open XML document: one page per image, each image scaled to fit an A4 page
  /// with 1 cm margins, separated by page breaks.
  Future<File> _docx(List<String> imagePaths, File out) async {
    const emuPerCm = 360000;
    const maxW = 19 * emuPerCm; // A4 21 cm - 2 x 1 cm
    const maxH = 27.7 * emuPerCm; // A4 29.7 cm - 2 x 1 cm
    final archive = Archive();
    final body = StringBuffer();
    final rels = StringBuffer();
    for (var i = 0; i < imagePaths.length; i++) {
      final png = await _pngBytes(imagePaths[i]);
      final size = await _imageSize(png);
      final scale = [maxW / size.$1, maxH / size.$2].reduce((a, b) => a < b ? a : b);
      final cx = (size.$1 * scale).round(), cy = (size.$2 * scale).round();
      final rid = 'rIdImg${i + 1}';
      archive.addFile(ArchiveFile('word/media/image${i + 1}.png', png.length, png));
      rels.write('<Relationship Id="$rid" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image${i + 1}.png"/>');
      if (i > 0) body.write('<w:p><w:r><w:br w:type="page"/></w:r></w:p>');
      body.write(_docxImageParagraph(rid, i + 1, cx, cy));
    }
    void add(String name, String content) {
      final bytes = utf8.encode(content);
      archive.addFile(ArchiveFile(name, bytes.length, bytes));
    }

    add('[Content_Types].xml',
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>');
    add('_rels/.rels',
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>');
    add('word/_rels/document.xml.rels',
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">$rels</Relationships>');
    add('word/document.xml',
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<w:body>$body'
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="567" w:right="567" w:bottom="567" w:left="567" w:header="0" w:footer="0" w:gutter="0"/></w:sectPr>'
        '</w:body></w:document>');
    final bytes = ZipEncoder().encode(archive);
    return out.writeAsBytes(bytes);
  }

  String _docxImageParagraph(String rid, int id, int cx, int cy) =>
      '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="0"/></w:pPr><w:r><w:drawing>'
      '<wp:inline distT="0" distB="0" distL="0" distR="0">'
      '<wp:extent cx="$cx" cy="$cy"/><wp:docPr id="$id" name="Page $id"/>'
      '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
      '<pic:pic><pic:nvPicPr><pic:cNvPr id="$id" name="page$id.png"/><pic:cNvPicPr/></pic:nvPicPr>'
      '<pic:blipFill><a:blip r:embed="$rid"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
      '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="$cx" cy="$cy"/></a:xfrm>'
      '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
      '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>';

  Future<List<int>> _pngBytes(String path) async {
    final bytes = await File(path).readAsBytes();
    if (path.toLowerCase().endsWith('.png')) return bytes;
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final data = await frame.image.toByteData(format: ui.ImageByteFormat.png);
    return data!.buffer.asUint8List();
  }

  Future<(int, int)> _imageSize(List<int> png) async {
    final buffer = await ui.ImmutableBuffer.fromUint8List(Uint8List.fromList(png));
    final descriptor = await ui.ImageDescriptor.encoded(buffer);
    final size = (descriptor.width, descriptor.height);
    descriptor.dispose();
    buffer.dispose();
    return size;
  }
}

class GalleryAccessDenied implements Exception {}
