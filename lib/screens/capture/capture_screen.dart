import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../state/scan_session.dart';
import '../processing/processing_preview_screen.dart';

class CaptureScreen extends StatelessWidget {
  const CaptureScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final session = context.watch<ScanSession>();
    return Scaffold(
      appBar: AppBar(title: const Text('Quét tài liệu')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.document_scanner_outlined, size: 96, color: Colors.grey),
              const SizedBox(height: 16),
              const Text(
                'Chụp tài liệu để làm nét, xoá bóng và xoá chữ viết tay tự động',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 15, color: Colors.grey),
              ),
              const SizedBox(height: 32),
              if (session.lastError != null) ...[
                Text(session.lastError!, style: const TextStyle(color: Colors.red)),
                const SizedBox(height: 16),
              ],
              FilledButton.icon(
                onPressed: session.isProcessing
                    ? null
                    : () async {
                        final ok = await session.capture();
                        if (ok && context.mounted) {
                          Navigator.of(context).push(
                            MaterialPageRoute(builder: (_) => const ProcessingPreviewScreen()),
                          );
                        }
                      },
                icon: session.isProcessing
                    ? const SizedBox(
                        width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.camera_alt_outlined),
                label: const Text('Quét tài liệu'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
