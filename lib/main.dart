import 'package:flutter/material.dart';

import 'app.dart';
import 'services/firebase/firebase_setup.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await FirebaseSetup.initialize();
  runApp(const SmartScanApp());
}
