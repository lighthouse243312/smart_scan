import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'screens/capture/capture_screen.dart';
import 'state/scan_session.dart';

class SmartScanApp extends StatelessWidget {
  const SmartScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => ScanSession(),
      child: MaterialApp(
        title: 'Beacon Smart Scan',
        theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple)),
        home: const CaptureScreen(),
      ),
    );
  }
}
