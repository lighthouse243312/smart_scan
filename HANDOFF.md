# Smart Scan — ghi chú bàn giao (2026-10-17)

Tài liệu để tiếp tục công việc ở phiên / tài khoản khác. Người dùng giao tiếp bằng tiếng Việt.

## Dự án
- Flutter app quét tài liệu, làm nét, khử bóng, **xoá chữ viết tay** và giữ chữ in. Xuất PDF/Word/PNG, 6 ngôn ngữ (en, ja, zh, fr, ko, vi), thư viện tài liệu, Firebase (FCM, Firestore `userLog/{Android|iOS}/devices/{uuid}`, Crashlytics; project `smart-scan-1c1b8`).
- Nguyên tắc của người dùng: **Dart chỉ làm UI, mọi logic xử lý ảnh nằm ở native** (Android Kotlin + OpenCV 5 + TFLite; iOS C++/ObjC++ + OpenCV + CoreML trong `ios/Runner/ImageProcessingOpenCV.mm`, `InkAnalysis.hpp`, `RuleRedraw.hpp`).
- Mask: PNG BGRA. A = chữ viết tay, B = chữ in, G = chữ in nằm dưới nét bút (vẽ lại sau khi xoá).
- Model: U-Net `ink_segmenter.tflite` / `InkSegmenter.mlpackage`, bản v24. Việc train đang dừng theo yêu cầu người dùng.

## Pipeline hiện tại (Android, `android/app/src/main/kotlin/com/example/beacon_smart_scan/imageprocessing/`)
Nhận diện (`InkSegmenter.mask`):
1. `InkAnalysis.detectByInkColor`: màu mực quyết định trước.
2. `PenComponentVote`.
3. `ModelFinish`: model chỉ hoàn thiện phần màu bỏ lại (đuôi nét, nét bút cùng màu chữ in khi model chắc ≥0.7). Có hai veto: nét màu mà model chắc là chữ in, và đường kẻ in màu.
4. `StrikeThrough`: đang tắt.
5. `IsolatedPrint`: mảnh "chữ in" đứng một mình thì là chữ viết tay, trừ khi có glyph sinh đôi.

Xoá (`MultiPassEraser.eraseAtSize`): MaskEraser → các pass lặp → `PrintGuard.restorePrint` → `RuleRedraw` → **`LineHeal`**.

## Việc vừa làm (chưa commit, chưa port iOS)
- **LineHeal.kt** (viết lại theo yêu cầu "kéo thẳng 2 đầu"):
  - Tìm đường kẻ ở mọi góc bằng HoughLinesP trên mặt nạ mực đã nở 3×3. Mỗi đoạn được đo theo hướng của chính nó: phải thẳng, mảnh (≤ max(3, 0.6k+1)), dày đều, và cùng hướng chủ đạo của trang (±2°).
  - Gom các đoạn thẳng hàng thành chuỗi. Chỉ nối qua chỗ đứt khi bút phủ ≥60% chỗ đứt (hoặc khe ≤2k), không có đường khác cắt ngang, không có chữ in ở hai bên.
  - Dò tiếp ở hai đầu để tìm đầu mút thật.
  - Vẽ lại **một nét thẳng từ đầu đến cuối** bằng mặt cắt đo từ đường thật, cùng số pixel dày và cùng độ đậm. Màu giấy lấy bằng median theo từng kênh màu.
  - Đã thử tốt trên p10, p02, p03. Còn vài chỗ thò ra vài pixel ở góc ô của p02.
  - Vẫn còn log chẩn đoán (`stats`, `log`) dùng cho harness.
- **PrintSlant.kt** + `ModelFinish.useSlant`:
  - Đo độ nghiêng nét đứng so với độ nghiêng chung của ảnh.
  - Kết quả không cải thiện trên 21 trang (recall 95.9 / precision 86.3 / giữ chữ in 99.15, y như khi không dùng). Đề xuất tắt; **hiện vẫn đang bật**, chờ người dùng xác nhận.
- Phân tích độ đậm, độ dày nét, chiều cao (theo chuẩn chữ in của từng trang): trên từng mảnh nét **không tách được** chữ viết tay khỏi chữ in. Hướng đề xuất: gom nét thành cả chữ/cả con số rồi mới so.

## Vấn đề đang mở (tin nhắn cuối của người dùng)
Ảnh worksheet tiếng Anh (`2.jpg` gốc, `1.jpg` kết quả app; hiện ở scratchpad, có thể đã mất), bản model mới nhất còn lỗi nhiều:
- Chữ viết tay mực đen hoặc xanh đậm, gần màu chữ in, còn sót các mảnh "} \ 5 T | | f . e.".
- Dấu tích và nét đỏ đè lên chữ in làm hỏng chữ in ("cake-", "cow-", "light c", "birthdav c-ake", "part'").
- Ảnh 901×1280 trên harness: coverage 0.0595, 2 pass. IsolatedPrint chỉ bắt được 0 mảnh. Không có tham chiếu độ nghiêng.
- Chưa kịp chẩn đoán xong. Bước tiếp theo: xem từng tầng (model_hw, color_hw, mask cuối) bằng `-e layers 1` để tìm chỗ sót / chỗ hỏng.

## Cập nhật 2026-10-17 (phiên sửa lỗi xoá nhầm / xoá không hết trên worksheet tiếng Anh)

Ảnh thử là `u1.jpg` đã push lên harness trên máy test.

- **Đã sửa, đang bật — `PenComponentVote` "modelPiece":**
  - Bút xanh đậm chụp gần đen có màu giống chữ in. Các chữ cái của nó, khi tách khỏi đường kẻ, bị gạt về chữ in.
  - Quy tắc mới: một mảnh trong khối "lẫn" được coi là chữ viết tay nếu model chắc ≥ `PIECE_SURE` = 0.8. Nếu model chắc ≥ 0.8 cho cả khối chứa mảnh đó, thì mảnh chỉ cần đạt ≥ `PIECE_IN_SURE` = 0.6.
  - Trên 21 trang: không hại (giữ chữ in 99.75, độ chính xác 94.5, recall p15 tăng 94.9 → 95.3).
  - Trang u1: "How does this pillow feel", "What is the weather" giờ được xoá hết.
  - Cờ harness: `-e piecesure`, `-e piecein`. Probe (`-e probe x,y`) giờ in thêm thông số của mảnh.
- **Đã thử nhưng TẮT vì hại hoặc chưa có tác dụng:**
  - `TickOverPrint.kt` (giữ mảnh đen dính dấu tích đứng cạnh dòng chữ in): recall 95.6 → 92.4 (p22 96 → 73.6), vì bút đen cùng màu mực in. Bật bằng `-e tick 1`.
  - `StrikeThrough` bản nghiêng (`slanted`, `MIN_LEN` = 3): độ chính xác 57.5, giữ chữ in 95.97. Mặc định vẫn tắt (`-e strike 1`).
  - `PrintRestore.keepStrikesOut` (tách nét gạch khỏi chữ in khi khôi phục): chưa có tác dụng, vì các nét gạch đen trên u1 vốn không nằm trong mask. Bật bằng `-e strikeout 1`.
- **Còn mở trên u1 (bút đen hoặc mực đỏ sẫm đè lên chữ in):**
  - Nét gạch ngang qua các từ in ("is / a / There / room / music", "Fruit / are / ...") không được nhận là chữ viết tay: cùng màu, model không thấy.
  - "?(x)", chữ "s" bị khoanh trong "cakes" / "cows" bị xoá theo dấu tích hoặc vòng khoanh. Ảnh đã tăng tương phản nên mọi mực đều có L = 5, a = b = 0; màu không tách được.
  - Hướng đề xuất: tách bằng hình dạng hoặc "chữ sinh đôi" (so khớp glyph in trên trang, như PrintRepair, chỉ khôi phục pixel trùng glyph), hoặc train thêm model với ca bút đen đè chữ in.
- Bước PrintSlant (`useSlant`) đã tắt mặc định (harness: `-e slant 1` để bật).
- Script đánh giá trong scratchpad bị dọn mất. Đã tạo lại `runeval.sh <method> <tag> "<args>"` và `score.py <tags...>`. runeval đẩy `ml/real/batch2/p??.jpg` lên máy, chạy harness với `-e erase 0 -e tag _<tag>`, rồi kéo mask về `scratchpad/eval/<tag>/`. score đọc lớp A của mask, tính `RealAcc` theo từng trang và tổng.
- Baseline 21 trang hiện tại (cấu hình mặc định): recall 95.6, độ chính xác 94.5, giữ chữ in 99.75.

## Cập nhật 2026-10-17 (chiều): hướng 1 so khớp chữ sinh đôi, hướng 2 model

- **Hướng 1, chữ in sinh đôi** (prototype Python trong scratchpad, `twin*.py`): không đạt.
  - Đã nhận diện đúng các dòng chữ in (≥4 glyph sạch thẳng hàng) và khoanh đúng chỗ nghi bút đè.
  - Nhưng chữ bị che gần hết thì không còn đủ phần nhìn thấy để so. Nới điều kiện "chứa trọn" thì chữ viết tay lại bị coi là chữ in.
- **Hướng 2, model.** App đang ship v24 (bản sao lưu ở `ml/model_backup_v24_shipped/`). Kết quả trong pipeline thật, 21 trang, theo thứ tự recall / precision / kept:

  | Model | recall | precision | kept |
  |---|---|---|---|
  | v24 | 95.6 | 94.5 | 99.75 |
  | v27 | 93.5 | 95.8 | 99.83 |
  | v28 (từ v27, `--all`, pen 0.6, recall-w 3) | 93.6 | 94.6 | 99.76 |

  - v27/v28 tụt recall ở p22/p23: nét bút đen khoanh/tích **lên hình vẽ in** bị coi là một phần của hình.
  - Trên u1, v27 giữ được "?" và đỡ hỏng "comic books"/"(x)", nhưng sót thêm vài mẩu chữ viết tay. Nét gạch ngang đen qua từ in vẫn không được xoá với mọi bản.
  - **Đã trả app về v24.** Có thể dùng `ink_segmenter_v27.keras` / `v28.keras` để export thử: `venv/bin/python export_ink_seg.py --keras <file>` (ghi đè cả tflite Android lẫn mlpackage iOS).
- Hướng tiếp theo có thể làm:
  - Bổ sung nhãn thật cho trang worksheet tiếng Anh (u1) và các trang có bút đen gạch qua chữ in, dùng `label_tool`. Hiện 21 trang không có ca nào như vậy.
  - Thêm augment "bút đen trên hình vẽ = chữ viết tay" để v27 không mất recall ở hình.

## Cập nhật 2026-10-18: model v40 đã vào app

- **App giờ dùng v40** = trộn trọng số v39 50% + v38 30% + v24 20% (`ml/ink_segmenter_v40_torch.pt` / `.keras`).
  - Đã export ra cả tflite (Android) lẫn mlpackage (iOS).
  - v24 cũ vẫn ở `ml/model_backup_v24_shipped/`.
- **ModelFinish đổi ngưỡng mặc định** (Android + iOS): `MAX_PRINT` 0.4 → 1.0, `SURE_MARGIN` 0.5 → 0.2.
  - Model mới học đúng "chữ in dưới nét bút", nên ngưỡng cũ gạt bỏ cả nét bút thật.
  - iOS chỉ đổi hằng số, **chưa build thử trên iOS**.
- Kết quả pipeline thật trên 26 trang (recall / precision / kept):

  | Model | Cả 26 trang | Bỏ p29 |
  |---|---|---|
  | v24 | 95.80 / 92.62 / 99.516 | 95.89 / 94.92 / 99.712 |
  | v40 | **96.00 / 94.24 / 99.657** | **96.00 / 95.26 / 99.748** |

- **Dữ liệu:**
  - Thêm p24 (worksheet tiếng Anh), p25, p26, p27, p29. Lưu ý: p25 trùng phiếu p18, p27 trùng phiếu p15.
  - Ảnh WEEK 23.1 chụp thô nằm ở `real/batch2_excluded/`.
- **Nhãn** (`label_real.py`) chia trường hợp rõ ràng:
  - A — bút thuần: hw 1, print 0, chấm cả hai.
  - B — bút đè lên chữ in thật (≥30% thân chữ ngoài nét bút): hw chấm, print không chấm.
  - C — chữ in thuần.
  - D — bút trên đường kẻ: không chấm.
- **Train** (`finetune_v7.py`):
  - Thêm `--near-pen-print-w`: phạt hw trên vành chữ in cách nét bút 2–4 px.
  - Bài học: phải giữ `--pen 0.6` và `--hw-pos-w` ≥ 4.5 như v24. Hạ xuống thì model quên dấu tích cắt qua hình vẽ (p22 tụt mạnh ở v27–v36).
- **Công cụ phân tích lỗi:** scratchpad `why/why.py <tag>` phân loại lỗi theo trang:
  - FN: mờ / sát chữ in / trên đường kẻ / chấm nhỏ / cả nét.
  - FP: viền / chữ in / đường kẻ / hình vẽ / khác.
- **Còn yếu:**
  - p24: chữ in dưới gạch ngang bị xoá theo.
  - p21: chữ in trong vòng khoanh tím.
  - p19, p21: pixel lẻ trong nét bút nhạt. Phần lớn là do cách chấm, vì khi xoá app nới vùng rộng ra.
  - p01, p11: nền giấy ô li tạo chấm nhiễu.

## Cập nhật 2026-10-18 (tối): app dùng v45 + nới viền + dọn chấm

- **Model trong app: v45.**
  - Train từ v43, với `--focus p18,p20,p22,p25,p29,p24:3` (lấy mẫu gấp 3 các trang còn lỗi).
  - v43 = train từ v42 sau khi sửa nhãn p22, p23, p24.
  - v42 = trộn v40 và v41.
- **ModelFinish thêm bước nới viền nét:** `RIM_STEPS` = 3, `RIM_THR` = 0.1. Nét bút đã tìm được nới vào phần mực liền kề có xác suất bút > 0.1 và không phải chữ in. Đã port sang iOS (`kFinishRim*`).
- **SpeckClean.kt** (chỉ có ở Android, chưa port iOS): sau LineHeal, xoá các chấm ≤ k² px nằm trong vùng chữ viết tay, không chạm chữ in. Hiệu quả nhỏ (+0.02).
- **LoopArcs.kt:** nhận vòng khoanh quanh từ. **Tắt** vì làm hỏng ngoặc và khung in ở p05, p06, p11.
- **TickOverPrint, StrikeThrough nghiêng, PrintSlant:** đều đang tắt.
- **Nhãn đã sửa:**
  - p22: vùng hình vẽ gắn là chữ in, dải vòng khoanh để bỏ qua; vòng khoanh quanh từ tách theo dòng chữ.
  - p23: vùng trắc nghiệm.
  - p24: bổ sung chữ viết tay bị thiếu.
  - Script trong scratchpad: `lab3/circles.py`, `lab3/pics22.py`, `lab/brush24.py`.
- **Chấm trên ảnh kết quả cuối** (`scratchpad/runfinal.sh`, `scorefinal.py`), 26 trang:

  | | Xoá sạch chữ viết tay | Giữ chữ in |
  |---|---|---|
  | v42 | 96.96% | 99.140% |
  | v45 + nới viền + dọn chấm | **97.00%** | **99.520%** |

  Trên mask (recall / precision / kept): v42 95.86 / 91.13 / 99.435 → v45 **96.13 / 94.01 / 99.693**.
- **Còn yếu:**
  - p22: dấu tích đè lên hình vẽ, xoá sạch 77%.
  - p20: giấy xám, chấm sót.
  - p19, p21: nét bút mờ, chủ yếu thấp do cách chấm.
- **Build:**
  - Android OK.
  - iOS: `flutter build ios --debug --no-codesign` OK. Build cho simulator không được vì `opencv2.framework` chỉ có bản cho máy thật.
- Các model khác (v27–v44, các bản trộn) nằm trong `ml/`. v24 gốc ở `ml/model_backup_v24_shipped/`.

## Công cụ thử nghiệm
- Build harness: `cd android && ./gradlew -q -PharnessRelease :app:installRelease :app:installReleaseAndroidTest`
- Thiết bị: `R5CR60GQSZX`; package `com.beacon.smartScan`; thư mục `/sdcard/Android/data/com.beacon.smartScan/files/harness/` (nhớ `chmod 666` file đã push).
- Lệnh chạy:
  `adb shell am instrument -w -e dir harness -e input p10.jpg -e method segmentation -e passes 3 -e tag _x com.beacon.smartScan.test/com.example.beacon_smart_scan.InkHarness`
- Các cờ:
  - `-e lineheal 0`, `-e slant 0`, `-e layers 1` (xuất model_hw / model_print / color_*)
  - `-e erase 0`
  - `-e usemask m.png` (chỉ xoá)
  - `-e vote 0`, `-e isolated 0`
- Đánh giá 21 trang có nhãn trong `ml/real/batch2`, qua `ml/finetune_v7.py` (`load_page`, `RealAcc`). Python dùng `ml/venv/bin/python` (có cv2).
- Kết quả tốt nhất hiện tại trên 21 trang: tìm được 95.9% chữ viết tay, đúng 86.3%, giữ chữ in 99.15%.

## Việc còn chờ
- Port `LineHeal` (và `ModelFinish` / `IsolatedPrint` nếu chưa có) sang iOS: `ImageProcessingOpenCV.mm`, chạy ngay sau RuleRedraw.
- Tắt `useSlant` nếu người dùng đồng ý; có thể xoá `StrokeVote` không còn dùng.
- Phía người dùng tự làm:
  - Bật GitHub Pages (main, /docs).
  - Firestore rules cho `userLog`.
  - Upload APNs key.
  - Ảnh chụp màn hình iPad, hoặc đặt `TARGETED_DEVICE_FAMILY` = "1".

## Lưu ý an toàn
- Chỉ commit hoặc push khi người dùng yêu cầu. Lần duy nhất đã commit: docs/ (86cff78).
- URL git remote có chứa token GitHub: **không hiển thị, không dùng**. Người dùng đã được nhắc thu hồi token này.
- `pod install` cần `LANG=en_US.UTF-8`. Firebase iOS cài qua SPM, nên script dSYM của Crashlytics dùng đường dẫn SourcePackages.
