import 'dart:io';

import 'package:flutter/material.dart';

import '../core/models/scan_page.dart';
import '../core/theme/app_theme.dart';

/// Horizontal strip of page thumbnails: tap to select, optional trailing "add page" tile.
class PageStrip extends StatelessWidget {
  const PageStrip({
    super.key,
    required this.pages,
    required this.selected,
    required this.onSelect,
    this.onAdd,
  });

  final List<ScanPage> pages;
  final int selected;
  final ValueChanged<int> onSelect;
  final VoidCallback? onAdd;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 92,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        itemCount: pages.length + (onAdd == null ? 0 : 1),
        separatorBuilder: (_, _) => const SizedBox(width: 10),
        itemBuilder: (context, i) {
          if (i == pages.length) {
            return _Tile(
              selected: false,
              onTap: onAdd!,
              child: const Icon(Icons.add_rounded, color: AppColors.primary),
            );
          }
          return _Tile(
            selected: i == selected,
            onTap: () => onSelect(i),
            child: Stack(
              fit: StackFit.expand,
              children: [
                Image.file(File(pages[i].displayPath), fit: BoxFit.cover, cacheWidth: 160),
                Positioned(
                  left: 4,
                  bottom: 4,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                    decoration: BoxDecoration(color: AppColors.ink.withValues(alpha: 0.65), borderRadius: BorderRadius.circular(6)),
                    child: Text('${i + 1}', style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w600)),
                  ),
                ),
                if (pages[i].finalPath != null)
                  const Positioned(
                    right: 4,
                    top: 4,
                    child: Icon(Icons.check_circle_rounded, size: 16, color: AppColors.primary),
                  ),
              ],
            ),
          );
        },
      ),
    );
  }
}

class _Tile extends StatelessWidget {
  const _Tile({required this.selected, required this.onTap, required this.child});

  final bool selected;
  final VoidCallback onTap;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        width: 68,
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: selected ? AppColors.primary : AppColors.line, width: selected ? 2 : 1),
        ),
        child: ClipRRect(borderRadius: BorderRadius.circular(selected ? 10 : 11), child: Center(child: child)),
      ),
    );
  }
}
