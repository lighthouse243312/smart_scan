import 'dart:async';

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';

import '../../firebase_options.dart';
import 'firebase_user_log.dart';

/// Messages that arrive while the app is in the background or terminated (runs in its own
/// isolate, so Firebase is initialised again here).
@pragma('vm:entry-point')
Future<void> firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  debugPrint('FCM background message: ${message.messageId}');
}

/// Firebase start-up: core, Crashlytics, Cloud Messaging and the per-device user log.
class FirebaseSetup {
  FirebaseSetup._();

  static bool _ready = false;
  static bool get isReady => _ready;

  /// Call before runApp. Never throws: the app works offline / without Firebase.
  static Future<void> initialize() async {
    try {
      await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);

      // Crashlytics collection is noisy/unwanted for local debug runs.
      await FirebaseCrashlytics.instance.setCrashlyticsCollectionEnabled(!kDebugMode);
      FlutterError.onError = FirebaseCrashlytics.instance.recordFlutterFatalError;
      // Errors thrown outside the Flutter framework (e.g. inside a bare `Future`) never reach
      // FlutterError.onError above — this is the catch-all for those.
      PlatformDispatcher.instance.onError = (error, stack) {
        FirebaseCrashlytics.instance.recordError(error, stack, fatal: true);
        return true;
      };

      FirebaseMessaging.onBackgroundMessage(firebaseMessagingBackgroundHandler);
      _ready = true;
    } catch (e) {
      debugPrint('Firebase init error: $e');
    }
  }

  /// After the first frame: notification permission, FCM token, user log, Crashlytics user id.
  static Future<void> start() async {
    if (!_ready) return;
    try {
      final log = FirebaseUserLog.instance;
      unawaited(log.deviceId().then(FirebaseCrashlytics.instance.setUserIdentifier));

      final messaging = FirebaseMessaging.instance;
      await messaging.requestPermission();
      // iOS: show notifications that arrive while the app is open, too
      await messaging.setForegroundNotificationPresentationOptions(alert: true, badge: true, sound: true);

      String? token;
      try {
        token = await messaging.getToken();
      } catch (e) {
        // iOS without an APNs token yet (simulator, push capability missing) lands here
        debugPrint('FCM getToken error: $e');
      }
      debugPrint('FCM token: $token');

      await log.logSession(fcmToken: token);
      if (token != null) await log.updateFcmToken(token);
      messaging.onTokenRefresh.listen(log.updateFcmToken);

      FirebaseMessaging.onMessage.listen((m) => debugPrint('FCM foreground message: ${m.notification?.title}'));
    } catch (e) {
      debugPrint('Firebase start error: $e');
    }
  }

  /// Keeps the log's app language in sync when the user changes it.
  static Future<void> onAppLanguageChanged(String? languageCode) async {
    if (!_ready) return;
    await FirebaseUserLog.instance.updateLoggedInfo(appLanguage: languageCode ?? 'system');
  }
}
