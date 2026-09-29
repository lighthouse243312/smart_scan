# Handwriting separation — Python reference prototypes

Reference implementations the native code (`ios/Runner/InkAnalysis.hpp`,
`android/.../imageprocessing/InkAnalysis.kt` + `InkRefine.kt`) was ported from. Use them to check
the native output against a known-good result on the same image.

- `ink_color_coarse.py` — working-size (≤2400 px) pass: optical-density ink colour vs the local
  print colour, regular text lines by geometry, line/character/neighbourhood consensus, pictures.
  `python ink_color_coarse.py <outdir> <image> <tag>` → `<tag>_hw.npy`, `<tag>_ref.npy`, previews.
- `ink_refine.py` — full-resolution refinement: own 3x3 colour, 4-direction arbitration on 3 px
  bands at 2/4/6 stroke widths, overlap (print under pen) = print running through + darker than the
  pen, print bridging across the pen, and the paper-only erase.
- `make_overlap_test_page.py` — synthetic page with known print/pen/overlap pixels
  (low-saturation blue-black pen, multiply blend).
- `eval_overlap.py`, `sweep_directions.py` — outcome metrics after erase (pen erased, print intact,
  print under pen kept) and the parameter sweep behind the chosen band/lengths/darkness.

Scripts take the scratch/output directory as their first argument; the real test photo is
`ml/real/IMG_0101_reading.png`.
