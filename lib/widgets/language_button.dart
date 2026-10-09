import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../core/theme/app_theme.dart';
import '../l10n/app_localizations.dart';
import '../state/locale_controller.dart';

/// Chip showing the language in use ("Tiếng Việt", "日本語"…); tap to change it.
class LanguageButton extends StatelessWidget {
  const LanguageButton({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = context.watch<LocaleController>();
    // the language actually shown: the chosen one, else the one the device resolved to
    final code = controller.locale?.languageCode ?? Localizations.localeOf(context).languageCode;
    final name = LocaleController.names[code] ?? code;
    return Material(
      color: AppColors.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
        side: const BorderSide(color: AppColors.line),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: () => showLanguagePicker(context),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.translate_rounded, size: 16, color: AppColors.primary),
              const SizedBox(width: 6),
              Text(name, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.ink)),
              const Icon(Icons.expand_more_rounded, size: 18, color: AppColors.muted),
            ],
          ),
        ),
      ),
    );
  }
}

Future<void> showLanguagePicker(BuildContext context) async {
  final controller = context.read<LocaleController>();
  final l = AppLocalizations.of(context);
  Widget tile(BuildContext context, Locale? locale, String name) {
    final selected = controller.locale?.languageCode == locale?.languageCode;
    return ListTile(
      title: Text(name),
      trailing: selected ? const Icon(Icons.check_rounded, color: AppColors.primary) : null,
      onTap: () {
        controller.set(locale);
        Navigator.pop(context);
      },
    );
  }

  await showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    isScrollControlled: true,
    builder: (context) => SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Text(l.language, style: Theme.of(context).textTheme.titleMedium),
          ),
          tile(context, null, l.systemLanguage),
          for (final locale in LocaleController.supported) tile(context, locale, LocaleController.names[locale.languageCode]!),
          const SizedBox(height: 8),
        ],
      ),
    ),
  );
}
