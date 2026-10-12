import 'dart:io';
import 'dart:ui' as ui;

import 'package:android_id/android_id.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// One Firestore document per device, the same layout as the pregnancy app:
/// `userLog/{Android|iOS}/devices/{uuid}` holding app / device / locale info and the FCM token.
///
/// [logSession] writes the document once, on the device's first launch (a no-op afterwards, so
/// it is cheap to call every launch); [updateLoggedInfo] merges fresh info into it when something
/// worth knowing changes (FCM token, app language).
class FirebaseUserLog {
  FirebaseUserLog._();

  static final instance = FirebaseUserLog._();

  static const _tokenKey = 'user_log_fcm_token';

  FirebaseFirestore get _firestore => FirebaseFirestore.instance;

  String? _uuid;

  /// The id the device's document is keyed by (also used as the Crashlytics user id).
  Future<String> deviceId() async => _uuid ??= (await _deviceInfo())['uuid'] as String;

  Future<void> logSession({String? fcmToken, String? appLanguage}) async {
    try {
      final info = await getFullDeviceInfo(fcmToken: fcmToken, appLanguage: appLanguage);
      final doc = _deviceDocRef(info['uuid'] as String);
      if ((await doc.get()).exists) return;
      await doc.set({...info, 'sessionTime': FieldValue.serverTimestamp()});
      if (fcmToken != null) await _rememberToken(fcmToken);
    } catch (e) {
      debugPrint('logSession error: $e');
    }
  }

  /// Merges the current info into the device's document (creating it if missing).
  Future<void> updateLoggedInfo({String? fcmToken, String? appLanguage}) async {
    try {
      final info = await getFullDeviceInfo(fcmToken: fcmToken, appLanguage: appLanguage);
      await _deviceDocRef(info['uuid'] as String).set(
        {...info, 'updatedTime': FieldValue.serverTimestamp()},
        SetOptions(merge: true),
      );
      if (fcmToken != null) await _rememberToken(fcmToken);
    } catch (e) {
      debugPrint('updateLoggedInfo error: $e');
    }
  }

  /// Writes [token] only when it differs from the last one written from this device.
  Future<void> updateFcmToken(String token) async {
    final prefs = await SharedPreferences.getInstance();
    if (prefs.getString(_tokenKey) == token) return;
    await updateLoggedInfo(fcmToken: token);
  }

  Future<void> _rememberToken(String token) async =>
      (await SharedPreferences.getInstance()).setString(_tokenKey, token);

  DocumentReference<Map<String, dynamic>> _deviceDocRef(String uuid) {
    final platform = Platform.isAndroid ? 'Android' : 'iOS';
    return _firestore.collection('userLog').doc(platform).collection('devices').doc(uuid);
  }

  Future<Map<String, dynamic>> getFullDeviceInfo({String? fcmToken, String? appLanguage}) async {
    final packageInfo = await PackageInfo.fromPlatform();
    final locale = ui.PlatformDispatcher.instance.locale;
    final prefs = await SharedPreferences.getInstance();
    return {
      // App
      'appVersion': '${packageInfo.version} (${packageInfo.buildNumber})',
      'appLanguage': appLanguage ?? prefs.getString('app_locale') ?? 'system',
      'fcmToken': ?fcmToken,

      // Device's own OS-level locale & timezone
      'language': Platform.localeName,
      'deviceLanguage': locale.languageCode,
      'deviceRegion': locale.countryCode ?? '',
      'timezone': DateTime.now().timeZoneName,

      ...await _deviceInfo(),
    };
  }

  Future<Map<String, dynamic>> _deviceInfo() async {
    final plugin = DeviceInfoPlugin();
    try {
      if (Platform.isAndroid) {
        final d = await plugin.androidInfo;
        return {
          'uuid': await _androidIdentifier(d),
          'platform': 'android',
          'osVersion': d.version.release,
          'modelName': '${d.manufacturer}_${d.model}',
          'isPhysicalDevice': d.isPhysicalDevice,
        };
      }
      if (Platform.isIOS) {
        final d = await plugin.iosInfo;
        return {
          'uuid': d.identifierForVendor ?? 'simulator',
          'platform': 'ios',
          'osVersion': d.systemVersion,
          'modelName': d.modelName,
          'isPhysicalDevice': d.isPhysicalDevice,
        };
      }
    } on PlatformException {
      debugPrint('Failed to get device info');
    }
    return {'uuid': 'unknown', 'platform': Platform.operatingSystem};
  }

  /// Settings.Secure.ANDROID_ID survives reinstalls (like iOS's identifierForVendor); Build.ID
  /// (androidInfo.id) is shared by every device on the same firmware, so only a fallback.
  Future<String> _androidIdentifier(AndroidDeviceInfo device) async {
    final id = await const AndroidId().getId();
    return (id != null && id.isNotEmpty) ? id : device.id;
  }
}
