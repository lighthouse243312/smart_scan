import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../services/export_service.dart';
import '../../state/scan_session.dart';

class ExportScreen extends StatefulWidget {
  const ExportScreen({super.key});

  @override
  State<ExportScreen> createState() => _ExportScreenState();
}

class _ExportScreenState extends State<ExportScreen> {
  final _exportService = ExportService();
  bool _saving = false;
  String? _message;

  @override
  Widget build(BuildContext context) {
    final session = context.watch<ScanSession>();
    final page = session.currentPage;

    return Scaffold(
      appBar: AppBar(title: const Text('Hoàn tất')),
      body: page == null
          ? const SizedBox.shrink()
          : Column(
              children: [
                Expanded(
                  child: Center(
                    child: InteractiveViewer(child: Image.file(File(page.displayPath))),
                  ),
                ),
                if (_message != null)
                  Padding(padding: const EdgeInsets.all(8), child: Text(_message!)),
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      FilledButton.icon(
                        onPressed: _saving
                            ? null
                            : () async {
                                setState(() {
                                  _saving = true;
                                  _message = null;
                                });
                                try {
                                  await _exportService.saveToGallery(page.displayPath);
                                  setState(() => _message = 'Đã lưu vào thư viện ảnh');
                                } catch (e) {
                                  setState(() => _message = 'Lưu thất bại: $e');
                                } finally {
                                  setState(() => _saving = false);
                                }
                              },
                        icon: _saving
                            ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                            : const Icon(Icons.save_alt),
                        label: const Text('Lưu vào thư viện ảnh'),
                      ),
                      const SizedBox(height: 8),
                      OutlinedButton.icon(
                        onPressed: () {
                          session.reset();
                          Navigator.of(context).popUntil((route) => route.isFirst);
                        },
                        icon: const Icon(Icons.done_all),
                        label: const Text('Xong'),
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }
}
