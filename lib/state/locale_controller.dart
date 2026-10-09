import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// The app language: null follows the device, otherwise one of [supported]. Remembered across
/// launches.
class LocaleController extends ChangeNotifier {
  static const supported = [
    Locale('vi'),
    Locale('en'),
    Locale('ja'),
    Locale('zh'),
    Locale('ko'),
    Locale('fr'),
  ];

  /// Each language's own name, shown in the picker whatever the current language is.
  static const names = {
    'vi': 'Tiếng Việt',
    'en': 'English',
    'ja': '日本語',
    'zh': '中文',
    'ko': '한국어',
    'fr': 'Français',
  };

  static const _key = 'app_locale';

  Locale? _locale;
  Locale? get locale => _locale;

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    final code = prefs.getString(_key);
    if (code != null && names.containsKey(code)) {
      _locale = Locale(code);
      notifyListeners();
    }
  }

  Future<void> set(Locale? locale) async {
    _locale = locale;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    if (locale == null) {
      await prefs.remove(_key);
    } else {
      await prefs.setString(_key, locale.languageCode);
    }
  }
}
