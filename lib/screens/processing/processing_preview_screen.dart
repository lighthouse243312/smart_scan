import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/theme/app_theme.dart';
import '../../l10n/app_localizations.dart';
import '../../state/scan_session.dart';
import '../../widgets/page_strip.dart';
import '../export/export_screen.dart';
import '../review/handwriting_review_screen.dart';

/// Step 2: sharpen + shadow removal, page by page. From here the user either goes on to the
/// handwriting erase or exports straight away.
class ProcessingPreviewScreen extends StatefulWidget {
  const ProcessingPreviewScreen({super.key});

  @override
  State<ProcessingPreviewScreen> createState() => _ProcessingPreviewScreenState();
}

class _ProcessingPreviewScreenState extends State<ProcessingPreviewScreen> {
  bool _showOriginal = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _processIfNeeded());
  }

  void _processIfNeeded() {
    final session = context.read<ScanSession>();
    if (session.currentPage != null && session.currentPage!.shadowRemovedPath == null && !session.isProcessing) {
      session.processCurrentPage();
    }
  }

  Future<void> _addPages(ScanSession session) async {
    final l = AppLocalizations.of(context);
    final source = await showModalBottomSheet<Future<bool> Function()>(
      context: context,
      showDragHandle: true,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.photo_camera_rounded, color: AppColors.primary),
              title: Text(l.addScanPage),
              onTap: () => Navigator.pop(context, session.capture),
            ),
            ListTile(
              leading: const Icon(Icons.photo_library_rounded, color: AppColors.primary),
              title: Text(l.addLibraryPage),
              onTap: () => Navigator.pop(context, session.importImages),
            ),
            ListTile(
              leading: const Icon(Icons.picture_as_pdf_rounded, color: AppColors.primary),
              title: Text(l.addPdfPage),
              onTap: () => Navigator.pop(context, session.importPdf),
            ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
    if (source == null) return;
    final ok = await source();
    if (ok && mounted) _processIfNeeded();
  }

  Future<void> _confirmRemove(ScanSession session) async {
    final l = AppLocalizations.of(context);
    final count = session.pages.length;
    // With several pages, ask first whether to drop just this one or all of them.
    var all = false;
    if (count > 1) {
      final choice = await showModalBottomSheet<bool>(
        context: context,
        showDragHandle: true,
        builder: (context) => SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.delete_outline_rounded, color: AppColors.danger),
                title: Text(l.deleteThisPage),
                onTap: () => Navigator.pop(context, false),
              ),
              ListTile(
                leading: const Icon(Icons.delete_sweep_rounded, color: AppColors.danger),
                title: Text('${l.deleteAllPages} ($count)'),
                onTap: () => Navigator.pop(context, true),
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      );
      if (choice == null || !mounted) return;
      all = choice;
    }
    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(all ? l.deleteAllPagesTitle(count) : l.deletePageTitle),
        content: Text(all ? l.deleteAllPagesBody : l.deletePageBody(session.currentPageIndex + 1)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: Text(l.cancel)),
          TextButton(onPressed: () => Navigator.pop(context, true), child: Text(l.delete)),
        ],
      ),
    );
    if (ok != true) return;
    if (all) {
      session.removeAllPages();
    } else {
      session.removePage(session.currentPageIndex);
    }
    if (session.pages.isEmpty && mounted) {
      Navigator.of(context).popUntil((r) => r.isFirst);
    } else {
      _processIfNeeded();
    }
  }

  @override
  Widget build(BuildContext context) {
    final session = context.watch<ScanSession>();
    final page = session.currentPage;
    final l = AppLocalizations.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(session.pages.length > 1 ? l.pageOf(session.currentPageIndex + 1, session.pages.length) : l.processTitle),
        actions: [
          IconButton(
            tooltip: l.rotateLeft,
            icon: const Icon(Icons.rotate_left_rounded),
            onPressed: session.isProcessing || page == null ? null : () => session.rotateCurrentPage(clockwise: false),
          ),
          IconButton(
            tooltip: l.rotateRight,
            icon: const Icon(Icons.rotate_right_rounded),
            onPressed: session.isProcessing || page == null ? null : () => session.rotateCurrentPage(clockwise: true),
          ),
          IconButton(
            tooltip: l.deletePage,
            icon: const Icon(Icons.delete_outline_rounded),
            onPressed: session.isProcessing || page == null ? null : () => _confirmRemove(session),
          ),
        ],
      ),
      body: page == null
          ? const SizedBox.shrink()
          : Column(
              children: [
                const SizedBox(height: 8),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(16, 4, 16, 8),
                    child: Container(
                      decoration: BoxDecoration(
                        color: AppColors.surface,
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: AppColors.line),
                      ),
                      clipBehavior: Clip.antiAlias,
                      child: session.isProcessing
                          ? _Working(
                              label: switch (session.batchProgress) {
                                final b? => l.detectingPages(b.current, b.total),
                                null => l.processing,
                              },
                            )
                          : InteractiveViewer(
                              maxScale: 6,
                              child: Center(
                                child: Image.file(
                                  File(_showOriginal ? page.originalPath : page.displayPath),
                                  gaplessPlayback: true,
                                ),
                              ),
                            ),
                    ),
                  ),
                ),
                if (session.lastError != null)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                    child: Text(session.lastError!, style: const TextStyle(color: AppColors.danger)),
                  ),
                if (page.shadowRemovedPath != null)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                    child: SegmentedButton<bool>(
                      showSelectedIcon: false,
                      segments: [
                        ButtonSegment(value: true, label: Text(l.original)),
                        ButtonSegment(value: false, label: Text(l.processed)),
                      ],
                      selected: {_showOriginal},
                      onSelectionChanged: (s) => setState(() => _showOriginal = s.first),
                    ),
                  ),
                const SizedBox(height: 8),
                PageStrip(
                  pages: session.pages,
                  selected: session.currentPageIndex,
                  onSelect: (i) {
                    session.selectPage(i);
                    _processIfNeeded();
                  },
                  onAdd: session.isProcessing ? null : () => _addPages(session),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
                  child: Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          icon: const Icon(Icons.ios_share_rounded),
                          onPressed: session.isProcessing
                              ? null
                              : () async {
                                  await session.processRemainingPages();
                                  if (!context.mounted || session.lastError != null) return;
                                  session.goToExport();
                                  Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => const ExportScreen()));
                                },
                          label: Text(l.exportNow),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: FilledButton.icon(
                          icon: const Icon(Icons.auto_fix_high_rounded),
                          onPressed: session.isProcessing || page.shadowRemovedPath == null
                              ? null
                              : () async {
                                  await session.detectHandwriting();
                                  if (context.mounted && session.lastError == null) {
                                    Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => const HandwritingReviewScreen()));
                                  }
                                },
                          label: Text(l.eraseWritingShort),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }
}

class _Working extends StatelessWidget {
  const _Working({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(width: 28, height: 28, child: CircularProgressIndicator(strokeWidth: 2.6)),
          const SizedBox(height: 14),
          Text(label, style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
    );
  }
}
