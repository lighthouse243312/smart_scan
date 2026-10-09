import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/models/saved_document.dart';
import '../../core/theme/app_theme.dart';
import '../../l10n/app_localizations.dart';
import '../../state/document_library.dart';
import 'document_detail_screen.dart';

String formatDate(BuildContext context, DateTime d) => MaterialLocalizations.of(context).formatMediumDate(d);

/// The user's saved documents, newest first.
class LibraryScreen extends StatelessWidget {
  const LibraryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final library = context.watch<DocumentLibrary>();
    final l = AppLocalizations.of(context);
    final docs = library.documents;
    return SafeArea(
      child: RefreshIndicator(
        onRefresh: library.refresh,
        child: CustomScrollView(
          slivers: [
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(20, 24, 20, 16),
              sliver: SliverToBoxAdapter(
                child: Text(l.documentsTitle, style: Theme.of(context).textTheme.headlineSmall),
              ),
            ),
            if (docs.isEmpty && !library.loading)
              SliverFillRemaining(hasScrollBody: false, child: _Empty(l: l))
            else
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
                sliver: SliverGrid(
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    mainAxisSpacing: 14,
                    crossAxisSpacing: 14,
                    childAspectRatio: 0.72,
                  ),
                  delegate: SliverChildBuilderDelegate(
                    (context, i) => DocumentCard(doc: docs[i]),
                    childCount: docs.length,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class DocumentCard extends StatelessWidget {
  const DocumentCard({super.key, required this.doc});

  final SavedDocument doc;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final text = Theme.of(context).textTheme;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () => Navigator.of(context).push(
          MaterialPageRoute(builder: (_) => DocumentDetailScreen(documentId: doc.id)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: Container(
                color: AppColors.background,
                padding: const EdgeInsets.all(10),
                child: doc.coverPath == null
                    ? const Icon(Icons.description_outlined, color: AppColors.muted)
                    : Image.file(File(doc.coverPath!), fit: BoxFit.contain, cacheWidth: 400),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 10, 12, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(doc.title, maxLines: 1, overflow: TextOverflow.ellipsis, style: text.titleMedium?.copyWith(fontSize: 14)),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          '${l.pagesCount(doc.pages.length)} · ${formatDate(context, doc.updatedAt)}',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: text.bodySmall,
                        ),
                      ),
                      if (doc.hasEdits) const Icon(Icons.auto_fix_high_rounded, size: 14, color: AppColors.primary),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Empty extends StatelessWidget {
  const _Empty({required this.l});

  final AppLocalizations l;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(color: AppColors.primarySoft, borderRadius: BorderRadius.circular(22)),
            child: const Icon(Icons.folder_open_rounded, color: AppColors.primary, size: 34),
          ),
          const SizedBox(height: 16),
          Text(l.documentsEmpty, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 6),
          Text(l.documentsEmptyHint, textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
    );
  }
}
