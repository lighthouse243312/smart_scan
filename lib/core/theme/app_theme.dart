import 'package:flutter/material.dart';

/// Soft, calm palette: a muted sage-teal accent on warm ivory paper, with a dusty peach for
/// secondary accents. Shared by every screen through [AppTheme.light].
class AppColors {
  static const primary = Color(0xFF5B8A84); // sage teal
  static const primarySoft = Color(0xFFDCEBE7);
  static const accent = Color(0xFFE3A68C); // dusty peach
  static const accentSoft = Color(0xFFF8E6DD);
  static const background = Color(0xFFF8F6F2); // warm ivory
  static const surface = Color(0xFFFFFFFF);
  static const ink = Color(0xFF2F3B3A);
  static const muted = Color(0xFF7D8886);
  static const line = Color(0xFFE7E3DC);
  static const danger = Color(0xFFC96B5C);
}

class AppTheme {
  static ThemeData get light {
    final scheme = ColorScheme.fromSeed(
      seedColor: AppColors.primary,
      brightness: Brightness.light,
    ).copyWith(
      primary: AppColors.primary,
      onPrimary: Colors.white,
      primaryContainer: AppColors.primarySoft,
      onPrimaryContainer: AppColors.ink,
      secondary: AppColors.accent,
      secondaryContainer: AppColors.accentSoft,
      onSecondaryContainer: AppColors.ink,
      surface: AppColors.surface,
      onSurface: AppColors.ink,
      error: AppColors.danger,
      outline: AppColors.line,
      outlineVariant: AppColors.line,
    );
    final rounded = RoundedRectangleBorder(borderRadius: BorderRadius.circular(16));
    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: AppColors.background,
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.background,
        foregroundColor: AppColors.ink,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: true,
        titleTextStyle: TextStyle(fontSize: 18, fontWeight: FontWeight.w600, color: AppColors.ink, letterSpacing: 0.2),
      ),
      textTheme: const TextTheme(
        headlineSmall: TextStyle(fontSize: 24, fontWeight: FontWeight.w700, color: AppColors.ink, letterSpacing: -0.2),
        titleMedium: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.ink),
        bodyMedium: TextStyle(fontSize: 14, color: AppColors.ink, height: 1.4),
        bodySmall: TextStyle(fontSize: 12.5, color: AppColors.muted, height: 1.4),
      ),
      cardTheme: CardThemeData(
        color: AppColors.surface,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: const BorderSide(color: AppColors.line),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          minimumSize: const Size.fromHeight(52),
          shape: rounded,
          textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          minimumSize: const Size.fromHeight(52),
          shape: rounded,
          side: const BorderSide(color: AppColors.line),
          foregroundColor: AppColors.ink,
          textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
        ),
      ),
      segmentedButtonTheme: SegmentedButtonThemeData(
        style: SegmentedButton.styleFrom(
          selectedBackgroundColor: AppColors.primarySoft,
          selectedForegroundColor: AppColors.ink,
          side: const BorderSide(color: AppColors.line),
        ),
      ),
      sliderTheme: const SliderThemeData(
        activeTrackColor: AppColors.primary,
        inactiveTrackColor: AppColors.primarySoft,
        thumbColor: AppColors.primary,
      ),
      snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
        backgroundColor: AppColors.ink,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
      progressIndicatorTheme: const ProgressIndicatorThemeData(color: AppColors.primary),
      dividerTheme: const DividerThemeData(color: AppColors.line, space: 1),
    );
  }
}
