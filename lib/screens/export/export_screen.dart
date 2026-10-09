import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/theme/app_theme.dart';
import '../../l10n/app_localizations.dart';
import '../../services/export_service.dart';
import '../../state/scan_session.dart';

/// Last step: preview every page, pick PDF / Word / PNG, then share or save.
class ExportScreen extends StatefulWidget {
  /// Without [paths] the screen exports the current scan session (and can keep it in the
  /// library); with them (a saved document) it only shares / saves those images.
  const ExportScreen({super.key, this.paths, this.title});

  final List<String>? paths;
  final String? title;

  @override
  State<ExportScreen> createState() => _ExportScreenState();
}

class _ExportScreenState extends State<ExportScreen> {
  final _exportService = ExportService();
  final _pageController = PageController(viewportFraction: 0.86);
  late final TextEditingController _name;
  ExportFormat _format = ExportFormat.pdf;
  bool _busy = false;
  int _visible = 0;

  @override
  void initState() {
    super.initState();
    final now = DateTime.now();
    String two(int v) => v.toString().padLeft(2, '0');
    _name = TextEditingController(
      text: widget.title ?? 'Scan_${now.year}${two(now.month)}${two(now.day)}_${two(now.hour)}${two(now.minute)}',
    );
  }

  @override
  void dispose() {
    _pageController.dispose();
    _name.dispose();
    super.dispose();
  }

  String get _baseName {
    final cleaned = _name.text.trim().replaceAll(RegExp(r'[\\/:*?"<>|]'), '_');
    return cleaned.isEmpty ? 'Scan' : cleaned;
  }

  Future<void> _run(Future<void> Function() action, String done) async {
    final l = AppLocalizations.of(context);
    setState(() => _busy = true);
    try {
      await action();
      if (mounted && done.isNotEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(done)));
      }
    } on GalleryAccessDenied {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(l.galleryPermission)));
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(l.actionFailed('$e'))));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final session = context.watch<ScanSession>();
    final fromSession = widget.paths == null;
    final paths = widget.paths ?? [for (final p in session.pages) p.displayPath];
    final text = Theme.of(context).textTheme;
    final l = AppLocalizations.of(context);

    // closing a later step ends the scan: a new capture starts a new document
    return PopScope(
      onPopInvokedWithResult: (didPop, _) {
        if (didPop && fromSession) session.reset();
      },
      child: Scaffold(
        appBar: AppBar(
          // in the scan flow steps only go forward: leaving here closes the flow, it does not reopen
          // the erase
          leading: fromSession ? const CloseButton() : null,
          title: Text(l.exportTitle),
        ),
        body: paths.isEmpty
            ? const SizedBox.shrink()
            : Column(
                children: [
                  Expanded(
                    child: PageView.builder(
                      controller: _pageController,
                      itemCount: paths.length,
                      onPageChanged: (i) => setState(() => _visible = i),
                      itemBuilder: (context, i) => Padding(
                        padding: const EdgeInsets.fromLTRB(6, 8, 6, 8),
                        child: Container(
                          decoration: BoxDecoration(
                            color: AppColors.surface,
                            borderRadius: BorderRadius.circular(18),
                            border: Border.all(color: AppColors.line),
                            boxShadow: [
                              BoxShadow(
                                color: AppColors.ink.withValues(alpha: 0.05),
                                blurRadius: 16,
                                offset: const Offset(0, 6),
                              ),
                            ],
                          ),
                          clipBehavior: Clip.antiAlias,
                          child: InteractiveViewer(maxScale: 5, child: Center(child: Image.file(File(paths[i])))),
                        ),
                      ),
                    ),
                  ),
                  if (paths.length > 1)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Text(l.pageOf(_visible + 1, paths.length), style: text.bodySmall),
                    ),
                  Container(
                    decoration: const BoxDecoration(
                      color: AppColors.surface,
                      borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
                      border: Border(top: BorderSide(color: AppColors.line)),
                    ),
                    padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
                    child: SafeArea(
                      top: false,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          TextField(
                            controller: _name,
                            decoration: InputDecoration(
                              labelText: l.fileName,
                              prefixIcon: const Icon(Icons.edit_note_rounded),
                              filled: true,
                              fillColor: AppColors.background,
                              border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(14),
                                borderSide: BorderSide.none,
                              ),
                            ),
                          ),
                          const SizedBox(height: 14),
                          Row(
                            children: [
                              for (final f in ExportFormat.values) ...[
                                Expanded(
                                  child: _FormatOption(
                                    format: f,
                                    selected: f == _format,
                                    onTap: () => setState(() => _format = f),
                                  ),
                                ),
                                if (f != ExportFormat.values.last) const SizedBox(width: 8),
                              ],
                            ],
                          ),
                          const SizedBox(height: 6),
                          Text(_description(l, _format), style: text.bodySmall),
                          const SizedBox(height: 14),
                          FilledButton.icon(
                            onPressed: _busy
                                ? null
                                : () => _run(() async {
                                    final files = await _exportService.build(paths, _format, baseName: _baseName);
                                    await _exportService.share(files, subject: _baseName);
                                  }, ''),
                            icon: _busy
                                ? const SizedBox(
                                    width: 18,
                                    height: 18,
                                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                                  )
                                : const Icon(Icons.ios_share_rounded),
                            label: Text(l.shareFormat(_label(l, _format))),
                          ),
                          const SizedBox(height: 10),
                          Row(
                            children: [
                              if (fromSession) ...[
                                Expanded(
                                  child: OutlinedButton.icon(
                                    onPressed: _busy
                                        ? null
                                        : () => _run(() async {
                                            await session.saveProcessedToLibrary();
                                          }, l.savedToLibrary),
                                    icon: const Icon(Icons.bookmark_add_outlined),
                                    label: Text(l.saveToLibrary, maxLines: 1, overflow: TextOverflow.ellipsis),
                                  ),
                                ),
                                const SizedBox(width: 10),
                              ],
                              Expanded(
                                child: OutlinedButton.icon(
                                  onPressed: _busy
                                      ? null
                                      : () => _run(
                                          () => _exportService.saveToGallery(paths),
                                          l.savedImages(paths.length),
                                        ),
                                  icon: const Icon(Icons.download_rounded),
                                  label: Text(l.saveImages, maxLines: 1, overflow: TextOverflow.ellipsis),
                                ),
                              ),
                            ],
                          ),
                          if (fromSession) ...[
                            const SizedBox(height: 4),
                            Center(
                              child: TextButton.icon(
                                onPressed: _busy
                                    ? null
                                    : () {
                                        session.reset();
                                        Navigator.of(context).popUntil((route) => route.isFirst);
                                      },
                                icon: const Icon(Icons.add_rounded, size: 18),
                                label: Text(l.newDocument),
                              ),
                            ),
                            Center(child: Text(l.originalsKept, style: text.bodySmall)),
                          ],
                        ],
                      ),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}

String _label(AppLocalizations l, ExportFormat f) => switch (f) {
  ExportFormat.pdf => l.formatPdf,
  ExportFormat.word => l.formatWord,
  ExportFormat.png => l.formatPng,
};

String _description(AppLocalizations l, ExportFormat f) => switch (f) {
  ExportFormat.pdf => l.formatPdfDesc,
  ExportFormat.word => l.formatWordDesc,
  ExportFormat.png => l.formatPngDesc,
};

class _FormatOption extends StatelessWidget {
  const _FormatOption({required this.format, required this.selected, required this.onTap});

  final ExportFormat format;
  final bool selected;
  final VoidCallback onTap;

  IconData get _icon => switch (format) {
    ExportFormat.pdf => Icons.picture_as_pdf_rounded,
    ExportFormat.word => Icons.description_rounded,
    ExportFormat.png => Icons.image_rounded,
  };

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          color: selected ? AppColors.primarySoft : AppColors.background,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: selected ? AppColors.primary : AppColors.line, width: selected ? 1.5 : 1),
        ),
        child: Column(
          children: [
            Icon(_icon, color: selected ? AppColors.primary : AppColors.muted),
            const SizedBox(height: 4),
            Text(
              _label(AppLocalizations.of(context), format),
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: selected ? AppColors.ink : AppColors.muted,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
