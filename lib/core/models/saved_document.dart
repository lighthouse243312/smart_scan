/// One page of a saved document: the capture exactly as it came out of the scanner / library
/// (never modified), and optionally the processed version the user chose to keep.
class SavedPage {
  const SavedPage({required this.id, required this.originalPath, this.editedPath});

  final String id;
  final String originalPath;
  final String? editedPath;

  /// What a viewer shows by default: the kept edit, else the original.
  String get bestPath => editedPath ?? originalPath;

  SavedPage copyWith({String? editedPath}) =>
      SavedPage(id: id, originalPath: originalPath, editedPath: editedPath ?? this.editedPath);
}

/// A document in the user's library. [ownerId] is the account it belongs to ("guest" until
/// sign-in exists), so a synced, per-user store can replace the local one later.
class SavedDocument {
  const SavedDocument({
    required this.id,
    required this.ownerId,
    required this.title,
    required this.createdAt,
    required this.updatedAt,
    required this.pages,
  });

  final String id;
  final String ownerId;
  final String title;
  final DateTime createdAt;
  final DateTime updatedAt;
  final List<SavedPage> pages;

  bool get hasEdits => pages.any((p) => p.editedPath != null);
  String? get coverPath => pages.isEmpty ? null : pages.first.bestPath;
}
