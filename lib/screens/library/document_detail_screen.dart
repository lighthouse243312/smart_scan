import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/models/saved_document.dart';
import '../../core/theme/app_theme.dart';
import '../../l10n/app_localizations.dart';
import '../../state/document_library.dart';
import '../../state/scan_session.dart';
import '../export/export_screen.dart';
import '../processing/processing_preview_screen.dart';
import 'library_screen.dart';

/// One saved document: its pages as captured, or as processed, with share / edit again /
/// rename / delete.
class DocumentDetailScreen extends StatefulWidget {
  const DocumentDetailScreen({super.key, required this.documentId});

  final String documentId;

  @override
  State<DocumentDetailScreen> createState() => _DocumentDetailScreenState();
}

class _DocumentDetailScreenState extends State<DocumentDetailScreen> {
  bool _showEdited = true;
  int _page = 0;

  List<String> _paths(SavedDocument doc) =>
      [for (final p in doc.pages) _showEdited && p.editedPath != null ? p.editedPath! : p.originalPath];

  Future<void> _rename(SavedDocument doc) async {
    final l = AppLocalizations.of(context);
    final controller = TextEditingController(text: doc.title);
    final title = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(l.rename),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: InputDecoration(labelText: l.documentName),
          onSubmitted: (v) => Navigator.pop(context, v),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text(l.cancel)),
          TextButton(onPressed: () => Navigator.pop(context, controller.text), child: Text(l.save)),
        ],
      ),
    );
    if (title != null && title.trim().isNotEmpty && mounted) {
      await context.read<DocumentLibrary>().rename(doc.id, title.trim());
    }
  }

  Future<void> _delete(SavedDocument doc) async {
    final l = AppLocalizations.of(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(l.deleteDocumentTitle),
        content: Text(l.deleteDocumentBody),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: Text(l.cancel)),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: Text(l.delete, style: const TextStyle(color: AppColors.danger)),
          ),
        ],
      ),
    );
    if (ok != true || !mounted) return;
    final library = context.read<DocumentLibrary>();
    Navigator.of(context).pop();
    await library.delete(doc.id);
  }

  void _editAgain(SavedDocument doc) {
    context.read<ScanSession>().openDocument(doc);
    Navigator.of(context).push(MaterialPageRoute(builder: (_) => const ProcessingPreviewScreen()));
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final doc = context.watch<DocumentLibrary>().byId(widget.documentId);
    if (doc == null) return Scaffold(appBar: AppBar());
    final paths = _paths(doc);
    final page = _page.clamp(0, doc.pages.length - 1);
    final text = Theme.of(context).textTheme;

    return Scaffold(
      appBar: AppBar(
        title: Text(doc.title, maxLines: 1, overflow: TextOverflow.ellipsis),
        actions: [
          PopupMenuButton<String>(
            icon: const Icon(Icons.more_vert_rounded),
            onSelected: (v) => v == 'rename' ? _rename(doc) : _delete(doc),
            itemBuilder: (context) => [
              PopupMenuItem(value: 'rename', child: Text(l.rename)),
              PopupMenuItem(value: 'delete', child: Text(l.deleteDocument, style: const TextStyle(color: AppColors.danger))),
            ],
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: Text(
              '${l.pagesCount(doc.pages.length)} · ${formatDate(context, doc.createdAt)}',
              style: text.bodySmall,
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: SegmentedButton<bool>(
              showSelectedIcon: false,
              segments: [
                ButtonSegment(value: false, label: Text(l.original), icon: const Icon(Icons.photo_outlined)),
                ButtonSegment(
                  value: true,
                  label: Text(l.processed),
                  icon: const Icon(Icons.auto_fix_high_rounded),
                  enabled: doc.hasEdits,
                ),
              ],
              selected: {_showEdited && doc.hasEdits},
              onSelectionChanged: (s) => setState(() => _showEdited = s.first),
            ),
          ),
          Expanded(
            child: PageView.builder(
              controller: PageController(viewportFraction: 0.88, initialPage: page),
              itemCount: paths.length,
              onPageChanged: (i) => setState(() => _page = i),
              itemBuilder: (context, i) => Padding(
                padding: const EdgeInsets.fromLTRB(6, 12, 6, 8),
                child: Container(
                  decoration: BoxDecoration(
                    color: AppColors.surface,
                    borderRadius: BorderRadius.circular(18),
                    border: Border.all(color: AppColors.line),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: InteractiveViewer(maxScale: 5, child: Center(child: Image.file(File(paths[i]), gaplessPlayback: true))),
                ),
              ),
            ),
          ),
          Text(
            _showEdited && doc.hasEdits && doc.pages[page].editedPath == null
                ? '${l.pageOf(page + 1, paths.length)} · ${l.noEdits}'
                : l.pageOf(page + 1, paths.length),
            style: text.bodySmall,
          ),
          SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 10, 16, 12),
              child: Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _editAgain(doc),
                      icon: const Icon(Icons.edit_rounded),
                      label: Text(l.editAgain),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: () => Navigator.of(context).push(
                        MaterialPageRoute(builder: (_) => ExportScreen(paths: paths, title: doc.title)),
                      ),
                      icon: const Icon(Icons.ios_share_rounded),
                      label: Text(l.share),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
