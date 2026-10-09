import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/theme/app_theme.dart';
import '../../l10n/app_localizations.dart';
import '../../state/document_library.dart';
import '../library/library_screen.dart';
import '../shell/home_shell.dart';
import '../../state/scan_session.dart';
import '../processing/processing_preview_screen.dart';
import '../../widgets/language_button.dart';

/// Home: scan with the camera or pick pictures from the library.
class CaptureScreen extends StatelessWidget {
  const CaptureScreen({super.key});

  Future<void> _start(BuildContext context, Future<bool> Function() action) async {
    final ok = await action();
    if (ok && context.mounted) {
      Navigator.of(context).push(MaterialPageRoute(builder: (_) => const ProcessingPreviewScreen()));
    }
  }

  @override
  Widget build(BuildContext context) {
    final session = context.watch<ScanSession>();
    final text = Theme.of(context).textTheme;
    final l = AppLocalizations.of(context);
    final recent = context.watch<DocumentLibrary>().documents.take(4).toList();
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 24, 20, 24),
          children: [
            Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(color: AppColors.primarySoft, borderRadius: BorderRadius.circular(14)),
                  child: const Icon(Icons.document_scanner_rounded, color: AppColors.primary),
                ),
                const SizedBox(width: 12),
                Text(l.appTitle, style: text.titleMedium?.copyWith(fontSize: 18)),
                const Spacer(),
                const LanguageButton(),
              ],
            ),
            const SizedBox(height: 28),
            Text(l.homeHeadline, style: text.headlineSmall),
            const SizedBox(height: 8),
            Text(
              l.homeSubtitle,
              style: text.bodySmall?.copyWith(fontSize: 13.5),
            ),
            const SizedBox(height: 28),
            _ActionCard(
              icon: Icons.photo_camera_rounded,
              title: l.scanTitle,
              subtitle: l.scanSubtitle,
              color: AppColors.primary,
              background: AppColors.primarySoft,
              busy: session.isProcessing,
              onTap: () => _start(context, session.capture),
            ),
            const SizedBox(height: 14),
            _ActionCard(
              icon: Icons.photo_library_rounded,
              title: l.libraryTitle,
              subtitle: l.librarySubtitle,
              color: const Color(0xFFB9775D),
              background: AppColors.accentSoft,
              busy: session.isProcessing,
              onTap: () => _start(context, session.importImages),
            ),
            const SizedBox(height: 14),
            _ActionCard(
              icon: Icons.picture_as_pdf_rounded,
              title: l.pdfTitle,
              subtitle: l.pdfSubtitle,
              color: AppColors.danger,
              background: AppColors.danger.withValues(alpha: 0.1),
              busy: session.isProcessing,
              onTap: () => _start(context, session.importPdf),
            ),
            if (session.lastError != null) ...[
              const SizedBox(height: 16),
              Text(session.lastError!, style: const TextStyle(color: AppColors.danger)),
            ],
            const SizedBox(height: 32),
            Text(l.exportTo, style: text.bodySmall?.copyWith(fontWeight: FontWeight.w600, letterSpacing: 0.4)),
            const SizedBox(height: 10),
            Row(
              children: [
                _FormatChip(icon: Icons.picture_as_pdf_rounded, label: l.formatPdf),
                const SizedBox(width: 8),
                _FormatChip(icon: Icons.description_rounded, label: l.formatWord),
                const SizedBox(width: 8),
                _FormatChip(icon: Icons.image_rounded, label: l.formatPng),
              ],
            ),
            if (recent.isNotEmpty) ...[
              const SizedBox(height: 32),
              Row(
                children: [
                  Expanded(
                    child: Text(l.recentDocuments, style: text.titleMedium),
                  ),
                  TextButton(
                    onPressed: () => HomeShell.selectTab(context, HomeShell.documentsTab),
                    child: Text(l.seeAll),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              SizedBox(
                height: 210,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: recent.length,
                  separatorBuilder: (_, _) => const SizedBox(width: 12),
                  itemBuilder: (context, i) => SizedBox(width: 150, child: DocumentCard(doc: recent[i])),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  const _ActionCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.color,
    required this.background,
    required this.busy,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final Color color;
  final Color background;
  final bool busy;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: busy ? null : onTap,
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Row(
            children: [
              Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(color: background, borderRadius: BorderRadius.circular(16)),
                child: busy
                    ? Padding(padding: const EdgeInsets.all(16), child: CircularProgressIndicator(strokeWidth: 2.4, color: color))
                    : Icon(icon, color: color, size: 28),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 4),
                    Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right_rounded, color: AppColors.muted),
            ],
          ),
        ),
      ),
    );
  }
}

class _FormatChip extends StatelessWidget {
  const _FormatChip({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.line),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: AppColors.muted),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(fontSize: 13, color: AppColors.ink)),
        ],
      ),
    );
  }
}
