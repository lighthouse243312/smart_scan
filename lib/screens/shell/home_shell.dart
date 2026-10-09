import 'package:flutter/material.dart';

import '../../core/theme/app_theme.dart';
import '../../l10n/app_localizations.dart';
import '../capture/capture_screen.dart';
import '../library/library_screen.dart';

/// App frame: Scan / Documents tabs (an Account tab can join when sign-in exists). Each tab keeps its state when switching.
class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  /// Lets a tab switch to another (e.g. "See all" on Scan opens Documents).
  static void selectTab(BuildContext context, int index) =>
      context.findAncestorStateOfType<_HomeShellState>()?._select(index);

  static const scanTab = 0, documentsTab = 1;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;

  void _select(int i) => setState(() => _index = i);

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Scaffold(
      body: IndexedStack(
        index: _index,
        children: const [CaptureScreen(), LibraryScreen()],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: _select,
        backgroundColor: AppColors.surface,
        indicatorColor: AppColors.primarySoft,
        surfaceTintColor: Colors.transparent,
        height: 68,
        destinations: [
          NavigationDestination(
            icon: const Icon(Icons.document_scanner_outlined),
            selectedIcon: const Icon(Icons.document_scanner_rounded, color: AppColors.primary),
            label: l.tabScan,
          ),
          NavigationDestination(
            icon: const Icon(Icons.folder_outlined),
            selectedIcon: const Icon(Icons.folder_rounded, color: AppColors.primary),
            label: l.tabDocuments,
          ),
        ],
      ),
    );
  }
}
