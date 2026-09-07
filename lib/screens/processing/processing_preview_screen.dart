import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../state/scan_session.dart';
import '../../widgets/step_progress_indicator.dart';
import '../review/handwriting_review_screen.dart';

class ProcessingPreviewScreen extends StatefulWidget {
  const ProcessingPreviewScreen({super.key});

  @override
  State<ProcessingPreviewScreen> createState() => _ProcessingPreviewScreenState();
}

class _ProcessingPreviewScreenState extends State<ProcessingPreviewScreen> {
  bool _showOriginal = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final session = context.read<ScanSession>();
      if (session.currentPage?.shadowRemovedPath == null) {
        session.processCurrentPage();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final session = context.watch<ScanSession>();
    final page = session.currentPage;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Làm nét & xoá bóng'),
        actions: [
          IconButton(
            tooltip: 'Xoay trái',
            icon: const Icon(Icons.rotate_left),
            onPressed: session.isProcessing || page == null
                ? null
                : () => session.rotateCurrentPage(clockwise: false),
          ),
          IconButton(
            tooltip: 'Xoay phải',
            icon: const Icon(Icons.rotate_right),
            onPressed: session.isProcessing || page == null
                ? null
                : () => session.rotateCurrentPage(clockwise: true),
          ),
        ],
      ),
      body: Column(
        children: [
          StepProgressIndicator(currentStep: session.step),
          Expanded(
            child: page == null
                ? const SizedBox.shrink()
                : Center(
                    child: session.isProcessing
                        ? const CircularProgressIndicator()
                        : InteractiveViewer(
                            child: Image.file(
                              File(_showOriginal ? page.originalPath : page.displayPath),
                            ),
                          ),
                  ),
          ),
          if (session.lastError != null)
            Padding(
              padding: const EdgeInsets.all(8),
              child: Text(session.lastError!, style: const TextStyle(color: Colors.red)),
            ),
          if (page?.shadowRemovedPath != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: SegmentedButton<bool>(
                segments: const [
                  ButtonSegment(value: true, label: Text('Ảnh gốc')),
                  ButtonSegment(value: false, label: Text('Đã xử lý')),
                ],
                selected: {_showOriginal},
                onSelectionChanged: (s) => setState(() => _showOriginal = s.first),
              ),
            ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: session.isProcessing ? null : () => session.processCurrentPage(),
                    child: const Text('Xử lý lại'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton(
                    onPressed: session.isProcessing || page?.shadowRemovedPath == null
                        ? null
                        : () async {
                            await session.detectHandwriting();
                            if (context.mounted) {
                              Navigator.of(context).push(
                                MaterialPageRoute(builder: (_) => const HandwritingReviewScreen()),
                              );
                            }
                          },
                    child: const Text('Tiếp tục'),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
