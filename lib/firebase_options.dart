// Firebase options for the smart-scan-1c1b8 project, taken from android/app/google-services.json
// and ios/Runner/GoogleService-Info.plist (regenerate both together if the project changes).
import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart' show defaultTargetPlatform, TargetPlatform;

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform => switch (defaultTargetPlatform) {
        TargetPlatform.android => android,
        TargetPlatform.iOS => ios,
        _ => throw UnsupportedError('DefaultFirebaseOptions are not supported for this platform.'),
      };

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyDPm3T4DuHtv-MjEICdXVuYqKTJUnQ2tIg',
    appId: '1:63155180902:android:0a02bbe6c9b56dd9fda4a7',
    messagingSenderId: '63155180902',
    projectId: 'smart-scan-1c1b8',
    storageBucket: 'smart-scan-1c1b8.firebasestorage.app',
  );

  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'AIzaSyDEpG2nMyzaE7FQaIy27hfpo68czDY-QQs',
    appId: '1:63155180902:ios:e3aba9878848866cfda4a7',
    messagingSenderId: '63155180902',
    projectId: 'smart-scan-1c1b8',
    storageBucket: 'smart-scan-1c1b8.firebasestorage.app',
    iosBundleId: 'com.beacon.smartScan',
  );
}
