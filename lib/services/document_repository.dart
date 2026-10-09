import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../core/models/saved_document.dart';

/// Where a user's documents live. The app talks only to this interface: today
/// [LocalDocumentRepository] keeps them on the device per owner; a signed-in, cloud-synced
/// implementation can replace it without touching the screens.
abstract class DocumentRepository {
  String get ownerId;

  Future<List<SavedDocument>> list();

  /// New document from capture originals (copied in untouched). Returns it with page ids.
  Future<SavedDocument> create(List<String> originalPaths, {required String title});

  /// Appends capture originals to [docId]; returns the new pages.
  Future<List<SavedPage>> addPages(String docId, List<String> originalPaths);

  /// Keeps [editedPath] (copied in) as page [pageId]'s processed version.
  Future<void> saveEdited(String docId, String pageId, String editedPath);

  Future<void> removePage(String docId, String pageId);
  Future<void> rename(String docId, String title);
  Future<void> delete(String docId);
}

/// Documents in the app's private storage: `users/<owner>/documents/<id>/` holding `meta.json`
/// and the page files.
class LocalDocumentRepository implements DocumentRepository {
  LocalDocumentRepository({required this.ownerId});

  @override
  final String ownerId;

  Future<Directory> _root() async {
    final base = await getApplicationDocumentsDirectory();
    final dir = Directory('${base.path}/users/$ownerId/documents');
    await dir.create(recursive: true);
    return dir;
  }

  Future<Directory> _docDir(String id) async => Directory('${(await _root()).path}/$id');

  String _newId() => DateTime.now().microsecondsSinceEpoch.toRadixString(36);

  String _ext(String path) {
    final dot = path.lastIndexOf('.');
    return dot < 0 ? 'jpg' : path.substring(dot + 1).toLowerCase();
  }

  @override
  Future<List<SavedDocument>> list() async {
    final root = await _root();
    final docs = <SavedDocument>[];
    await for (final e in root.list()) {
      if (e is! Directory) continue;
      final doc = await _read(e);
      if (doc != null) docs.add(doc);
    }
    docs.sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
    return docs;
  }

  @override
  Future<SavedDocument> create(List<String> originalPaths, {required String title}) async {
    final id = _newId();
    final dir = await _docDir(id);
    await dir.create(recursive: true);
    final now = DateTime.now();
    final pages = await _copyOriginals(dir, originalPaths);
    final doc = SavedDocument(id: id, ownerId: ownerId, title: title, createdAt: now, updatedAt: now, pages: pages);
    await _write(dir, doc);
    return doc;
  }

  @override
  Future<List<SavedPage>> addPages(String docId, List<String> originalPaths) async {
    final dir = await _docDir(docId);
    final doc = await _read(dir);
    if (doc == null) return const [];
    final added = await _copyOriginals(dir, originalPaths);
    await _write(dir, _with(doc, pages: [...doc.pages, ...added]));
    return added;
  }

  @override
  Future<void> saveEdited(String docId, String pageId, String editedPath) async {
    final dir = await _docDir(docId);
    final doc = await _read(dir);
    if (doc == null) return;
    final target = File('${dir.path}/${pageId}_edited.${_ext(editedPath)}');
    await File(editedPath).copy(target.path);
    final pages = [for (final p in doc.pages) p.id == pageId ? p.copyWith(editedPath: target.path) : p];
    await _write(dir, _with(doc, pages: pages));
  }

  @override
  Future<void> removePage(String docId, String pageId) async {
    final dir = await _docDir(docId);
    final doc = await _read(dir);
    if (doc == null) return;
    final page = doc.pages.where((p) => p.id == pageId).firstOrNull;
    if (page == null) return;
    for (final f in [page.originalPath, page.editedPath]) {
      if (f != null && await File(f).exists()) await File(f).delete();
    }
    final rest = doc.pages.where((p) => p.id != pageId).toList();
    if (rest.isEmpty) {
      await dir.delete(recursive: true);
    } else {
      await _write(dir, _with(doc, pages: rest));
    }
  }

  @override
  Future<void> rename(String docId, String title) async {
    final dir = await _docDir(docId);
    final doc = await _read(dir);
    if (doc != null) await _write(dir, _with(doc, title: title));
  }

  @override
  Future<void> delete(String docId) async {
    final dir = await _docDir(docId);
    if (await dir.exists()) await dir.delete(recursive: true);
  }

  Future<List<SavedPage>> _copyOriginals(Directory dir, List<String> paths) async {
    final pages = <SavedPage>[];
    for (final p in paths) {
      final id = _newId();
      final target = File('${dir.path}/${id}_original.${_ext(p)}');
      await File(p).copy(target.path);
      pages.add(SavedPage(id: id, originalPath: target.path));
    }
    return pages;
  }

  SavedDocument _with(SavedDocument d, {String? title, List<SavedPage>? pages}) => SavedDocument(
        id: d.id,
        ownerId: d.ownerId,
        title: title ?? d.title,
        createdAt: d.createdAt,
        updatedAt: DateTime.now(),
        pages: pages ?? d.pages,
      );

  // Files are stored by name only, so the folder can move (app container paths change on iOS).
  Future<void> _write(Directory dir, SavedDocument d) async {
    String name(String p) => p.substring(p.lastIndexOf('/') + 1);
    final json = {
      'id': d.id,
      'ownerId': d.ownerId,
      'title': d.title,
      'createdAt': d.createdAt.toIso8601String(),
      'updatedAt': d.updatedAt.toIso8601String(),
      'pages': [
        for (final p in d.pages)
          {'id': p.id, 'original': name(p.originalPath), if (p.editedPath != null) 'edited': name(p.editedPath!)},
      ],
    };
    await File('${dir.path}/meta.json').writeAsString(jsonEncode(json));
  }

  Future<SavedDocument?> _read(Directory dir) async {
    final meta = File('${dir.path}/meta.json');
    if (!await meta.exists()) return null;
    try {
      final j = jsonDecode(await meta.readAsString()) as Map<String, dynamic>;
      return SavedDocument(
        id: j['id'] as String,
        ownerId: j['ownerId'] as String,
        title: j['title'] as String,
        createdAt: DateTime.parse(j['createdAt'] as String),
        updatedAt: DateTime.parse(j['updatedAt'] as String),
        pages: [
          for (final p in (j['pages'] as List).cast<Map<String, dynamic>>())
            SavedPage(
              id: p['id'] as String,
              originalPath: '${dir.path}/${p['original']}',
              editedPath: p['edited'] == null ? null : '${dir.path}/${p['edited']}',
            ),
        ],
      );
    } catch (_) {
      return null;
    }
  }
}
