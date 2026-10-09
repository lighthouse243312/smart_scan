import 'package:flutter/foundation.dart';

import '../core/models/saved_document.dart';
import '../services/document_repository.dart';
import 'auth_controller.dart';

/// The current user's saved documents. Rebuilds its repository when the user changes, so the
/// library always shows the signed-in account's documents (the guest's until sign-in exists).
class DocumentLibrary extends ChangeNotifier {
  DocumentLibrary(this._auth, {DocumentRepository Function(String ownerId)? repositoryFor})
      : _repositoryFor = repositoryFor ?? ((id) => LocalDocumentRepository(ownerId: id)) {
    _repo = _repositoryFor(_auth.userId);
    _auth.addListener(_onUserChanged);
    refresh();
  }

  final AuthController _auth;
  final DocumentRepository Function(String ownerId) _repositoryFor;
  late DocumentRepository _repo;

  List<SavedDocument> _documents = const [];
  List<SavedDocument> get documents => _documents;
  bool loading = true;

  DocumentRepository get repository => _repo;

  void _onUserChanged() {
    if (_repo.ownerId == _auth.userId) return;
    _repo = _repositoryFor(_auth.userId);
    refresh();
  }

  Future<void> refresh() async {
    loading = true;
    notifyListeners();
    try {
      _documents = await _repo.list();
    } catch (_) {
      // storage unavailable: show what we had rather than failing the screen
    } finally {
      loading = false;
      notifyListeners();
    }
  }

  SavedDocument? byId(String id) => _documents.where((d) => d.id == id).firstOrNull;

  Future<SavedDocument> create(List<String> originalPaths, {required String title}) async {
    final doc = await _repo.create(originalPaths, title: title);
    await refresh();
    return doc;
  }

  Future<List<SavedPage>> addPages(String docId, List<String> originalPaths) async {
    final pages = await _repo.addPages(docId, originalPaths);
    await refresh();
    return pages;
  }

  /// Keeps each (pageId -> processed image) pair as that page's edited version.
  Future<void> saveEdited(String docId, Map<String, String> editedByPage) async {
    for (final e in editedByPage.entries) {
      await _repo.saveEdited(docId, e.key, e.value);
    }
    await refresh();
  }

  Future<void> removePage(String docId, String pageId) async {
    await _repo.removePage(docId, pageId);
    await refresh();
  }

  Future<void> rename(String docId, String title) async {
    await _repo.rename(docId, title);
    await refresh();
  }

  Future<void> delete(String docId) async {
    await _repo.delete(docId);
    await refresh();
  }

  @override
  void dispose() {
    _auth.removeListener(_onUserChanged);
    super.dispose();
  }
}
