import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:provider/provider.dart';

import 'core/theme/app_theme.dart';
import 'l10n/app_localizations.dart';
import 'screens/shell/home_shell.dart';
import 'state/auth_controller.dart';
import 'state/document_library.dart';
import 'state/locale_controller.dart';
import 'state/scan_session.dart';

class SmartScanApp extends StatelessWidget {
  const SmartScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthController()),
        ChangeNotifierProvider(create: (context) => DocumentLibrary(context.read<AuthController>())),
        ChangeNotifierProvider(create: (context) => ScanSession(library: context.read<DocumentLibrary>())),
        ChangeNotifierProvider(create: (_) => LocaleController()..load()),
      ],
      child: Consumer<LocaleController>(
        builder: (context, locale, _) => MaterialApp(
          onGenerateTitle: (context) => AppLocalizations.of(context).appTitle,
          debugShowCheckedModeBanner: false,
          theme: AppTheme.light,
          locale: locale.locale,
          supportedLocales: LocaleController.supported,
          localizationsDelegates: const [
            AppLocalizations.delegate,
            GlobalMaterialLocalizations.delegate,
            GlobalWidgetsLocalizations.delegate,
            GlobalCupertinoLocalizations.delegate,
          ],
          home: const HomeShell(),
        ),
      ),
    );
  }
}
