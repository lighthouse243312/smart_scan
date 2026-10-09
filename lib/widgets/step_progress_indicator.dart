import 'package:flutter/material.dart';

import '../core/theme/app_theme.dart';
import '../l10n/app_localizations.dart';
import '../state/scan_session.dart';

/// Four-step progress bar: dots joined by a line, labels underneath.
class StepProgressIndicator extends StatelessWidget {
  const StepProgressIndicator({super.key, required this.currentStep});

  final ScanStep currentStep;

  @override
  Widget build(BuildContext context) {
    final current = ScanStep.values.indexOf(currentStep);
    final l = AppLocalizations.of(context);
    final labels = [l.stepCapture, l.stepProcess, l.stepHandwriting, l.stepExport];
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 4, 24, 10),
      child: Row(
        children: List.generate(labels.length * 2 - 1, (k) {
          if (k.isOdd) {
            final i = k ~/ 2;
            return Expanded(
              child: Container(
                height: 2,
                margin: const EdgeInsets.only(bottom: 18),
                color: i < current ? AppColors.primary : AppColors.line,
              ),
            );
          }
          final i = k ~/ 2;
          final done = i < current, active = i == current;
          return Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: active ? 22 : 18,
                height: active ? 22 : 18,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: done || active ? AppColors.primary : AppColors.surface,
                  border: Border.all(color: done || active ? AppColors.primary : AppColors.line, width: 1.5),
                ),
                child: done ? const Icon(Icons.check_rounded, size: 12, color: Colors.white) : null,
              ),
              const SizedBox(height: 4),
              Text(
                labels[i],
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: active ? FontWeight.w600 : FontWeight.w400,
                  color: active ? AppColors.ink : AppColors.muted,
                ),
              ),
            ],
          );
        }),
      ),
    );
  }
}
