import 'package:flutter/material.dart';

import '../state/scan_session.dart';

const _stepLabels = ['Chụp', 'Xử lý', 'Chữ viết tay', 'Xuất'];

class StepProgressIndicator extends StatelessWidget {
  const StepProgressIndicator({super.key, required this.currentStep});

  final ScanStep currentStep;

  @override
  Widget build(BuildContext context) {
    final currentIndex = ScanStep.values.indexOf(currentStep);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: List.generate(_stepLabels.length, (i) {
          final isActive = i <= currentIndex;
          return Expanded(
            child: Row(
              children: [
                CircleAvatar(
                  radius: 12,
                  backgroundColor: isActive ? Theme.of(context).colorScheme.primary : Colors.grey.shade300,
                  child: Text(
                    '${i + 1}',
                    style: TextStyle(fontSize: 12, color: isActive ? Colors.white : Colors.grey.shade600),
                  ),
                ),
                if (i < _stepLabels.length - 1)
                  Expanded(
                    child: Container(
                      height: 2,
                      color: i < currentIndex ? Theme.of(context).colorScheme.primary : Colors.grey.shade300,
                    ),
                  ),
              ],
            ),
          );
        }),
      ),
    );
  }
}
