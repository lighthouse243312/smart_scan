// OpenCV headers must be imported before Foundation/the class's own header — OpenCV's C++
// headers conflict with Objective-C globals (e.g. `NO`) when included afterwards.
// Using OpenCV 4.10.0 here (not 5.0.0, matching Android) — the official 5.0.0 iOS framework
// ships a DNN module that references ONNX Runtime/MLAS symbols missing for the device arm64
// slice, so any app linking it fails with "Undefined symbol: MlasGemmBatch" etc. on a real
// device. 4.10.0 predates that DNN/ONNX backend and has no such issue.
#import <opencv2/opencv.hpp>

#import <algorithm>
#import <cmath>
#import <map>
#import <set>
#import <vector>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import "ImageProcessingOpenCV.h"
#import "GridRedraw.hpp"
#import "InkAnalysis.hpp"
#import "PrintRestore.hpp"
#import "RuleRedraw.hpp"
#import "Straighten.hpp"

NSString *const ImageProcessingErrorDomain = @"ImageProcessingOpenCV";

static NSError *MakeError(ImageProcessingErrorCode code, NSString *message) {
    return [NSError errorWithDomain:ImageProcessingErrorDomain
                                code:code
                            userInfo:@{NSLocalizedDescriptionKey : message}];
}

static bool ReadOrFail(NSString *path, cv::Mat *outMat, NSError **error) {
    *outMat = cv::imread([path UTF8String], cv::IMREAD_COLOR);
    if (outMat->empty()) {
        *error = MakeError(ImageProcessingErrorFileNotFound,
                            [NSString stringWithFormat:@"Không đọc được ảnh tại: %@", path]);
        return false;
    }
    return true;
}

static bool WriteOrFail(const cv::Mat &mat, NSString *path, NSError **error) {
    if (!cv::imwrite([path UTF8String], mat)) {
        *error = MakeError(ImageProcessingErrorProcessingFailed,
                            [NSString stringWithFormat:@"Không ghi được ảnh tại: %@", path]);
        return false;
    }
    return true;
}

// MARK: - Handwriting mask helpers (mirror HandwritingMask.kt on Android)
//
// A mask file is a BGRA PNG the same size as the page holding two INDEPENDENT layers:
//  - alpha > 0      → handwriting (drawn semi-transparent red, so the file doubles as the review
//                     overlay);
//  - blue  > 127    → printed ink, stored even where alpha is 0 (invisible in the overlay).
// A pixel can be both — handwriting written over print — and then shows magenta in the overlay.
// The erase step repaints handwriting pixels from these layers instead of inpainting, so print
// crossed by a pen stroke is restored rather than smeared.

static const int kOverlayAlpha = 160;
// Alpha of a pen strike-through's pixels (StrikeThrough): handwriting like any other, but the erase
// must not draw it back as a printed rule or print (it runs level through printed words).
static const int kStrikeAlpha = 159;
// "Ink" = noticeably darker than the local paper — used to snap the model's soft, reduced-scale
// masks onto the page's actual strokes.
static const int kInkContrastLoose = 8;

// Segmentation model contract — must match ml/train_ink_seg.py / export_ink_seg.py: two
// independent sigmoid channels.
static const int kSegLongSide = 1536;
static const int kSegTile = 256;
static const int kSegOverlap = 32;
static const int kSegPrintChannel = 0;
static const int kSegHandwritingChannel = 1;
static const int kSegChannelCount = 2;
static const double kSegPrintThreshold = 0.5;

/// Paper brightness per pixel: a large median over a 4x-downscaled copy wipes out text strokes
/// (thin relative to the kernel) and keeps shading, then is scaled back up.
static cv::Mat EstimatePaper(const cv::Mat &gray) {
    cv::Mat small;
    cv::resize(gray, small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    int kernel = std::min(21, (std::min(small.cols, small.rows) - 1) | 1);
    if (kernel >= 3) cv::medianBlur(small, small, kernel);
    cv::Mat paper;
    cv::resize(small, paper, gray.size(), 0, 0, cv::INTER_LINEAR);
    return paper;
}

/// How much darker than the paper each pixel is (0 where lighter).
static cv::Mat InkContrast(const cv::Mat &gray) {
    cv::Mat contrast;
    cv::subtract(EstimatePaper(gray), gray, contrast);
    return contrast;
}

/// Isolated specks (sensor noise, paper texture) are never worth erasing; scaled to the image
/// so a 12 MP photo and a small scan drop roughly the same physical size.
static int MinSpeckleArea(const cv::Mat &image) {
    return std::max(12, (int)(image.total() * 0.000004));
}

static void RemoveSmallComponents(cv::Mat &mask, int minArea) {
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    std::vector<uchar> keep(count, 0);
    for (int i = 1; i < count; i++) {
        keep[i] = stats.at<int32_t>(i, cv::CC_STAT_AREA) >= minArea ? 255 : 0;
    }
    for (int y = 0; y < mask.rows; y++) {
        const int32_t *labelRow = labels.ptr<int32_t>(y);
        uchar *maskRow = mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; x++) maskRow[x] = keep[labelRow[x]];
    }
}

/// `handwriting` / `print` / `overlap`: CV_8UC1, 255 = set. Layers: A = handwriting (overlay
/// alpha, drawn red), B = print, G = overlap (print hidden under handwriting — restored on erase;
/// red + green shows it yellow in the overlay).
// Overlap (print under the pen) kept only where the print runs THROUGH the pen: pure print within
// 2 stroke units on both sides along the row or along the column, pixel by pixel. A rule or a
// printed letter a pen crosses has print on either side; the body of a pen digit written on a rule
// does not — the model's print layer bleeds onto it, and restoring it left dashes and pieces of the
// digits behind. Mirrors HandwritingMask.printThroughPen on Android.
static const int kRimClear = 2;   // print this close to handwriting is its rim

// Handwriting strokes (connected pieces) with print near them that is not their own rim: pure
// print more than kRimClear px from any handwriting (specks aside), within 2 stroke units.
static cv::Mat StrokesTouchingPrint(const cv::Mat &purePrint, const cv::Mat &handwriting, int k) {
    cv::Mat rim;
    cv::dilate(handwriting, rim, cv::Mat::ones(2 * kRimClear + 1, 2 * kRimClear + 1, CV_8U));
    cv::Mat far = purePrint & ~rim;
    RemoveSmallComponents(far, k * k);   // specks of noise on the paper are not print
    cv::Mat reach;
    cv::dilate(far, reach, cv::Mat::ones(4 * k + 1, 4 * k + 1, CV_8U));
    cv::Mat labels;
    const int n = cv::connectedComponents(handwriting, labels, 8, CV_32S);
    std::vector<char> touches(n, 0);
    for (int y = 0; y < labels.rows; y++) {
        const int *lp = labels.ptr<int>(y); const uchar *rp = reach.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; x++) if (lp[x] > 0 && rp[x]) touches[lp[x]] = 1;
    }
    cv::Mat out = cv::Mat::zeros(labels.size(), CV_8U);
    for (int y = 0; y < labels.rows; y++) {
        const int *lp = labels.ptr<int>(y); uchar *op = out.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; x++) if (touches[lp[x]]) op[x] = 255;
    }
    return out;
}

static cv::Mat PrintThroughPen(const cv::Mat &overlap, const cv::Mat &purePrint, const cv::Mat &handwriting, int k) {
    const int d = 2 * k;
    cv::Mat left, right, up, down;
    cv::dilate(purePrint, left, cv::Mat::ones(1, d + 1, CV_8U), cv::Point(d, 0));
    cv::dilate(purePrint, right, cv::Mat::ones(1, d + 1, CV_8U), cv::Point(0, 0));
    cv::dilate(purePrint, up, cv::Mat::ones(d + 1, 1, CV_8U), cv::Point(0, d));
    cv::dilate(purePrint, down, cv::Mat::ones(d + 1, 1, CV_8U), cv::Point(0, 0));
    // ...or it lies on a rule: a long (6 stroke units), thin horizontal run of print and print under
    // the pen — a blank line written along its whole length has no print beside the pen close by,
    // but no digit or letter body is a run that long and thin
    cv::Mat rule, thick;
    cv::morphologyEx(overlap | purePrint, rule, cv::MORPH_OPEN, cv::Mat::ones(1, 6 * k + 1, CV_8U));
    cv::morphologyEx(rule, thick, cv::MORPH_OPEN, cv::Mat::ones(k, 1, CV_8U));
    rule &= ~thick;
    // a pen stroke that touches no print at all (a "1800" alone in an answer box) has nothing under
    // it: the print layer only carries its own soft rim, which would bracket the stroke and bring
    // it back
    return ((left & right) | (up & down) | rule) & overlap & StrokesTouchingPrint(purePrint, handwriting, k);
}

static bool WriteMaskFile(const cv::Mat &handwriting, const cv::Mat &print, const cv::Mat &overlap, NSString *path, NSError **error,
                          const cv::Mat &strike = cv::Mat()) {
    cv::Mat file(handwriting.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
    std::vector<cv::Mat> channels;
    cv::split(file, channels);
    // B: print with no handwriting on it, pixel by pixel — a pixel both layers claim is print only
    // when marked as print under the pen (G, overlap; kept apart so it shows yellow, not white).
    // Per-component votes upstream can leave a pen pixel flagged print too (pen digits fused with a
    // table's grid make one huge mostly-print component); the erase would keep and restore it.
    channels[0] = print & ~handwriting;
    channels[1] = PrintThroughPen(overlap, channels[0], handwriting, inkanalysis::StrokeUnit(handwriting));   // G: overlap
    channels[2].setTo(255, handwriting);                           // R: overlay colour
    channels[3].setTo(kOverlayAlpha, handwriting);                 // A: handwriting
    if (!strike.empty()) channels[3].setTo(kStrikeAlpha, strike & handwriting);   // A: strike-through
    cv::merge(channels, file);
    return WriteOrFail(file, path, error);
}

/// `outStrike` (optional): the strike-through layer (alpha == kStrikeAlpha), part of the handwriting.
static bool ReadMaskFileOrFail(NSString *path, cv::Mat *outHandwriting, cv::Mat *outPrint, cv::Mat *outOverlap, NSError **error,
                               cv::Mat *outStrike = nullptr) {
    cv::Mat raw = cv::imread([path UTF8String], cv::IMREAD_UNCHANGED);
    if (raw.empty() || raw.channels() != 4) {
        *error = MakeError(ImageProcessingErrorFileNotFound,
                            [NSString stringWithFormat:@"Không đọc được mask tại: %@", path]);
        return false;
    }
    cv::Mat alpha, blue, green;
    cv::extractChannel(raw, alpha, 3);
    cv::extractChannel(raw, blue, 0);
    cv::extractChannel(raw, green, 1);
    cv::threshold(alpha, *outHandwriting, 0, 255, cv::THRESH_BINARY);
    cv::threshold(blue, *outPrint, 127, 255, cv::THRESH_BINARY);
    cv::threshold(green, *outOverlap, 127, 255, cv::THRESH_BINARY);
    if (outStrike) cv::inRange(alpha, kStrikeAlpha, kStrikeAlpha, *outStrike);
    return true;
}

static NSNumber *Coverage(const cv::Mat &mask) {
    return @((double)cv::countNonZero(mask) / (double)std::max<size_t>(1, mask.total()));
}

// MARK: - Working size (mirror WorkingSize.kt on Android)
//
// Handwriting detection and erase run on the page scaled down to kWorkingLongSide px: their cost
// grows much faster than the pixel count (an 8 MP camera scan took minutes to erase), and the steps
// are tuned on pages this size anyway — the model itself sees the page at kSegLongSide px. The page
// keeps its full resolution: a mask is scaled back up to it, and an erase result replaces only the
// pixels the erase changed (CompositeErased), so untouched print stays sharp.

static const int kWorkingLongSide = 1600;
/// Gray-level change (0-255) from the input that marks a pixel as changed by the erase.
static const double kChangedDelta = 6;

/// The factor (< 1) a page of `size` is scaled by to work on, or 1 when it is small enough.
static double WorkingScale(const cv::Size &size) {
    const int longSide = std::max(size.width, size.height);
    return longSide <= kWorkingLongSide ? 1.0 : (double)kWorkingLongSide / longSide;
}

static NSString *NewWorkDir(void) {
    NSString *work = [NSTemporaryDirectory() stringByAppendingPathComponent:[[NSUUID UUID] UUIDString]];
    [[NSFileManager defaultManager] createDirectoryAtPath:work withIntermediateDirectories:YES attributes:nil error:nil];
    return work;
}

static bool WriteScaled(const cv::Mat &src, const cv::Size &size, NSString *path, NSError **error) {
    cv::Mat small;
    cv::resize(src, small, size, 0, 0, cv::INTER_AREA);
    return WriteOrFail(small, path, error);
}

/// The mask file at `maskPath` scaled to `size`, written to `outPath`. Area-averaged on the raw file
/// when shrinking: a working pixel is handwriting if any of it was (alpha > 0 — no pen is lost),
/// print / print under the pen if most of it was (> 127).
static bool ResizeMaskFile(NSString *maskPath, NSString *outPath, const cv::Size &size, NSError **error) {
    cv::Mat raw = cv::imread([maskPath UTF8String], cv::IMREAD_UNCHANGED);
    if (raw.empty() || raw.channels() != 4) {
        *error = MakeError(ImageProcessingErrorFileNotFound,
                            [NSString stringWithFormat:@"Không đọc được mask tại: %@", maskPath]);
        return false;
    }
    cv::Mat out;
    cv::resize(raw, out, size, 0, 0, size.width < raw.cols ? cv::INTER_AREA : cv::INTER_NEAREST);
    return WriteOrFail(out, outPath, error);
}

/// Runs `detect` (input path, mask path) on `src` scaled down to the working size and writes its
/// mask, scaled back to the page's size, to `maskPath`. Returns the coverage.
static NSNumber *DetectAtWorkingSize(const cv::Mat &src, double scale, NSString *maskPath, NSError **error,
                                     NSNumber *(^detect)(NSString *inputPath, NSString *maskPath, NSError **error)) {
    NSString *work = NewWorkDir();
    NSString *page = [work stringByAppendingPathComponent:@"page.png"];
    NSString *smallMask = [work stringByAppendingPathComponent:@"mask.png"];
    NSNumber *coverage = nil;
    const cv::Size size(cvRound(src.cols * scale), cvRound(src.rows * scale));
    if (WriteScaled(src, size, page, error)) {
        coverage = detect(page, smallMask, error);
        if (coverage != nil && !ResizeMaskFile(smallMask, maskPath, src.size(), error)) coverage = nil;
    }
    [[NSFileManager defaultManager] removeItemAtPath:work error:nil];
    return coverage;
}

/// The full-resolution page `full` with the erase result `smallOut` (made from `smallIn`, its
/// working-size copy) laid over it where the erase changed something, written to `outputPath`.
/// Everywhere else the page keeps its own, sharper pixels.
static bool CompositeErased(cv::Mat full, const cv::Mat &smallIn, const cv::Mat &smallOut, NSString *outputPath, NSError **error) {
    cv::Mat changed;
    cv::absdiff(smallOut, smallIn, changed);
    cv::cvtColor(changed, changed, cv::COLOR_BGR2GRAY);
    cv::threshold(changed, changed, kChangedDelta, 255, cv::THRESH_BINARY);
    // one working pixel of margin: the seam falls on pixels the erase left alone
    cv::dilate(changed, changed, cv::Mat::ones(3, 3, CV_8U));
    if (cv::countNonZero(changed) > 0) {
        cv::Mat up, changedFull;
        cv::resize(smallOut, up, full.size(), 0, 0, cv::INTER_CUBIC);
        cv::resize(changed, changedFull, full.size(), 0, 0, cv::INTER_LINEAR);
        up.copyTo(full, changedFull > 0);
    }
    return WriteOrFail(full, outputPath, error);
}

// MARK: - Segmentation model (Core ML)

static MLModel *LoadSegmentationModel(NSError **error) {
    static MLModel *model = nil;
    static NSError *loadError = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSURL *url = [[NSBundle mainBundle] URLForResource:@"InkSegmenter" withExtension:@"mlmodelc"];
        if (url == nil) {
            loadError = MakeError(ImageProcessingErrorProcessingFailed,
                                   @"Chưa có model InkSegmenter trong app (chạy ml/export_ink_seg.py rồi build lại)");
            return;
        }
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        NSError *err = nil;
        model = [MLModel modelWithContentsOfURL:url configuration:config error:&err];
        if (model == nil) {
            loadError = MakeError(ImageProcessingErrorProcessingFailed,
                                   [NSString stringWithFormat:@"Không tải được model InkSegmenter: %@",
                                                              err.localizedDescription]);
        }
    });
    if (model == nil && error) *error = loadError;
    return model;
}

/// Reads channel `channel` of an NHWC [1, H, W, C] output array into `out` (H x W, CV_32F),
/// honouring the array's own strides and element type (Core ML may hand back float16).
static bool ReadClassPlane(MLMultiArray *array, int channel, cv::Mat &out) {
    if (array.shape.count != 4) return false;
    NSInteger h = array.shape[1].integerValue, w = array.shape[2].integerValue;
    NSInteger sy = array.strides[1].integerValue, sx = array.strides[2].integerValue,
              sc = array.strides[3].integerValue;
    out.create((int)h, (int)w, CV_32F);
    if (array.dataType == MLMultiArrayDataTypeFloat32) {
        const float *base = (const float *)array.dataPointer;
        for (int y = 0; y < h; y++) {
            float *row = out.ptr<float>(y);
            for (int x = 0; x < w; x++) row[x] = base[y * sy + x * sx + channel * sc];
        }
        return true;
    }
    if (array.dataType == MLMultiArrayDataTypeFloat16) {
        const __fp16 *base = (const __fp16 *)array.dataPointer;
        for (int y = 0; y < h; y++) {
            float *row = out.ptr<float>(y);
            for (int x = 0; x < w; x++) row[x] = (float)base[y * sy + x * sx + channel * sc];
        }
        return true;
    }
    if (array.dataType == MLMultiArrayDataTypeDouble) {
        const double *base = (const double *)array.dataPointer;
        for (int y = 0; y < h; y++) {
            float *row = out.ptr<float>(y);
            for (int x = 0; x < w; x++) row[x] = (float)base[y * sy + x * sx + channel * sc];
        }
        return true;
    }
    return false;
}

/// Tile blending weight: 1 in the tile's interior, tapering towards its edges across the overlap
/// band, so each pixel is decided mostly by the tile that sees it with the most context.
static cv::Mat TileWeights() {
    cv::Mat weights(kSegTile, kSegTile, CV_32F);
    for (int y = 0; y < kSegTile; y++) {
        float wy = std::min({y + 1, kSegTile - y, kSegOverlap}) / (float)kSegOverlap;
        for (int x = 0; x < kSegTile; x++) {
            float wx = std::min({x + 1, kSegTile - x, kSegOverlap}) / (float)kSegOverlap;
            weights.at<float>(y, x) = wy * wx;
        }
    }
    return weights;
}

/// Runs the U-Net over the page in overlapping tiles at the model's working scale and returns the
/// print and handwriting probability maps at the ORIGINAL image size.
static bool SegmentPage(const cv::Mat &src, cv::Mat *outPrintProb, cv::Mat *outHandwritingProb, NSError **error) {
    MLModel *model = LoadSegmentationModel(error);
    if (model == nil) return false;

    double scale = (double)kSegLongSide / (double)std::max(src.cols, src.rows);
    cv::Mat work;
    cv::resize(src, work, cv::Size(), scale, scale, scale < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC);
    cv::cvtColor(work, work, cv::COLOR_BGR2RGB);

    int step = kSegTile - kSegOverlap;
    int paddedW = std::max(kSegTile, (int)std::ceil((work.cols - kSegTile) / (double)step) * step + kSegTile);
    int paddedH = std::max(kSegTile, (int)std::ceil((work.rows - kSegTile) / (double)step) * step + kSegTile);
    cv::Mat padded;
    cv::copyMakeBorder(work, padded, 0, paddedH - work.rows, 0, paddedW - work.cols,
                       cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    cv::Mat paddedFloat;
    padded.convertTo(paddedFloat, CV_32FC3);

    NSError *arrayError = nil;
    MLMultiArray *input = [[MLMultiArray alloc] initWithShape:@[ @1, @(kSegTile), @(kSegTile), @3 ]
                                                     dataType:MLMultiArrayDataTypeFloat32
                                                        error:&arrayError];
    if (input == nil) {
        if (error) *error = MakeError(ImageProcessingErrorProcessingFailed, arrayError.localizedDescription ?: @"MLMultiArray");
        return false;
    }

    cv::Mat weights = TileWeights();
    cv::Mat printSum = cv::Mat::zeros(paddedH, paddedW, CV_32F);
    cv::Mat handwritingSum = cv::Mat::zeros(paddedH, paddedW, CV_32F);
    cv::Mat weightSum = cv::Mat::zeros(paddedH, paddedW, CV_32F);
    cv::Mat printPlane, handwritingPlane;
    for (int y = 0; y + kSegTile <= paddedH; y += step) {
        for (int x = 0; x + kSegTile <= paddedW; x += step) {
            cv::Mat tile = paddedFloat(cv::Rect(x, y, kSegTile, kSegTile)).clone();
            memcpy(input.dataPointer, tile.data, sizeof(float) * kSegTile * kSegTile * 3);

            @autoreleasepool {
                NSError *predictionError = nil;
                MLDictionaryFeatureProvider *features =
                    [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"input" : input} error:&predictionError];
                id<MLFeatureProvider> output = features ? [model predictionFromFeatures:features error:&predictionError] : nil;
                MLMultiArray *probs = [output featureValueForName:@"probs"].multiArrayValue;
                if (probs == nil || probs.shape.count != 4 || probs.shape[3].integerValue != kSegChannelCount ||
                    !ReadClassPlane(probs, kSegPrintChannel, printPlane) ||
                    !ReadClassPlane(probs, kSegHandwritingChannel, handwritingPlane)) {
                    if (error) {
                        *error = MakeError(ImageProcessingErrorProcessingFailed,
                                            predictionError.localizedDescription ?: @"Model trả về output không hợp lệ");
                    }
                    return false;
                }
            }
            cv::Rect roi(x, y, kSegTile, kSegTile);
            cv::Mat printRoi = printSum(roi), handwritingRoi = handwritingSum(roi), weightRoi = weightSum(roi);
            printRoi += printPlane.mul(weights);
            handwritingRoi += handwritingPlane.mul(weights);
            weightRoi += weights;
        }
    }

    cv::Rect valid(0, 0, work.cols, work.rows);
    cv::Mat printProb, handwritingProb;
    cv::divide(printSum, weightSum, printProb);
    cv::divide(handwritingSum, weightSum, handwritingProb);
    cv::resize(printProb(valid), *outPrintProb, src.size(), 0, 0, cv::INTER_LINEAR);
    cv::resize(handwritingProb(valid), *outHandwritingProb, src.size(), 0, 0, cv::INTER_LINEAR);
    return true;
}

// MARK: - Text recognition (Vision)

/// Reads the printed text of `bgr` with Vision, any language it knows (detected per page): each
/// line's top readings, split into words at whitespace, each word's graphemes and box in pixels.
/// Vision boxes words only — the letters' own positions come from the page's ink (see
/// RestorePrintByRecognition).
static std::vector<inkanalysis::OcrLine> RecognizeText(const cv::Mat &bgr) {
    std::vector<inkanalysis::OcrLine> lines;
    cv::Mat rgba;
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    const int W = rgba.cols, H = rgba.rows;
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(rgba.data, W, H, 8, rgba.step, space,
                                             kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
    CGImageRef image = ctx ? CGBitmapContextCreateImage(ctx) : nil;
    if (ctx) CGContextRelease(ctx);
    CGColorSpaceRelease(space);
    if (image == nil) return lines;
    @autoreleasepool {
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.revision = VNRecognizeTextRequestRevision3;
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.automaticallyDetectsLanguage = YES;
        request.usesLanguageCorrection = YES;
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image options:@{}];
        NSError *error = nil;
        if (![handler performRequests:@[ request ] error:&error]) {
            CGImageRelease(image);
            return lines;
        }
        // Vision boxes are normalised with the origin bottom-left
        auto toRect = [&](CGRect b) {
            return cv::Rect((int)std::floor(b.origin.x * W), (int)std::floor((1.0 - b.origin.y - b.size.height) * H),
                            (int)std::ceil(b.size.width * W), (int)std::ceil(b.size.height * H));
        };
        for (VNRecognizedTextObservation *observation in request.results) {
            inkanalysis::OcrLine line;
            line.box = toRect(observation.boundingBox);
            for (VNRecognizedText *candidate in [observation topCandidates:3]) {
                NSString *text = candidate.string;
                std::vector<inkanalysis::OcrWord> words;
                inkanalysis::OcrWord word;
                NSRange wordRange = NSMakeRange(NSNotFound, 0);
                auto flush = [&] {
                    if (!word.graphemes.empty()) {
                        NSError *boxError = nil;
                        VNRectangleObservation *box = [candidate boundingBoxForRange:wordRange error:&boxError];
                        if (box != nil) {
                            word.box = toRect(box.boundingBox);
                            words.push_back(word);
                        }
                    }
                    word = inkanalysis::OcrWord();
                    wordRange = NSMakeRange(NSNotFound, 0);
                };
                for (NSUInteger i = 0; i < text.length;) {
                    const NSRange range = [text rangeOfComposedCharacterSequenceAtIndex:i];
                    NSString *grapheme = [text substringWithRange:range];
                    i = range.location + range.length;
                    if ([grapheme stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet].length == 0) {
                        flush();
                        continue;
                    }
                    word.graphemes.push_back(std::string(grapheme.UTF8String));
                    wordRange = wordRange.location == NSNotFound ? range : NSUnionRange(wordRange, range);
                }
                flush();
                line.readings.push_back(words);
            }
            lines.push_back(line);
        }
    }
    CGImageRelease(image);
    return lines;
}

// MARK: - Stroke vote (mirror StrokeVote.kt on Android)
//
// Last say on the handwriting layer, per ink stroke, by the segmentation model: the colour layers
// split strokes pixel by pixel ("57,45" half pen, half print) and leave half a digit after the
// erase. A stroke the model sees as mostly handwriting is handwriting whole; one it sees almost no
// handwriting in is print; in between (pen crossing print) the model's own pixels. Then handwriting
// grows pixel by pixel along the ink into neighbours the model does not rule out (a black pen
// circle round a printed letter, half-hearted next to the letter, closes), and print the model
// sees under the handwriting is remembered as overlap, to be drawn back after the erase.

static const double kStrokeWhole = 0.75, kStrokeNone = 0.05, kGrowHw = 0.2, kUnderPenPrint = 0.5;
static const int kGrowSteps = 60;
static const double kPieceHalf = 0.5;   // a piece of ink (rules aside) this much handwriting is handwriting whole

static void StrokeVote(cv::Mat &handwriting, cv::Mat &print, cv::Mat &overlap, const cv::Mat &ink,
                       const cv::Mat &modelHw, const cv::Mat &modelPrint, double threshold) {
    cv::Mat labels;
    const int n = cv::connectedComponents(ink, labels, 8, CV_32S);
    std::vector<int> area(n, 0), hwCount(n, 0);
    for (int y = 0; y < ink.rows; y++) {
        const int *lab = labels.ptr<int>(y);
        const float *ph = modelHw.ptr<float>(y);
        for (int x = 0; x < ink.cols; x++)
            if (lab[x] > 0) { area[lab[x]]++; if (ph[x] > threshold) hwCount[lab[x]]++; }
    }
    // rule-shaped ink (long thin straight runs, either way): grid and answer lines, which the colour
    // layers can mistake for pen (a faint coloured exercise-book grid)
    cv::Mat ruleShaped;
    {
        const int k = inkanalysis::StrokeUnit(ink);
        cv::Mat hRun, vRun;
        cv::morphologyEx(ink, hRun, cv::MORPH_OPEN, cv::Mat::ones(1, 6 * k + 1, CV_8U));
        cv::morphologyEx(ink, vRun, cv::MORPH_OPEN, cv::Mat::ones(6 * k + 1, 1, CV_8U));
        ruleShaped = hRun | vRun;
    }
    for (int y = 0; y < ink.rows; y++) {
        const int *lab = labels.ptr<int>(y);
        const uchar *rs = ruleShaped.ptr<uchar>(y);
        const float *ph = modelHw.ptr<float>(y), *pp = modelPrint.ptr<float>(y);
        uchar *hw = handwriting.ptr<uchar>(y), *pr = print.ptr<uchar>(y), *ov = overlap.ptr<uchar>(y);
        for (int x = 0; x < ink.cols; x++) {
            const int l = lab[x];
            if (l == 0) continue;
            const double share = area[l] ? (double)hwCount[l] / area[l] : 0.0;
            const bool modelSaysHw = ph[x] > threshold;
            if (share >= kStrokeWhole) {
                hw[x] = 255;
                if (!modelSaysHw && pp[x] > 0.5f) { ov[x] = 255; pr[x] = 255; }
            } else if (share <= kStrokeNone) {
                hw[x] = 0; ov[x] = 0; pr[x] = 255;
            } else {
                // in between (pen crossing print, strokes fused with a rule): the model ADDS its
                // own pixels but does not take away what the colour layers found — a pen "I"
                // standing on an answer line looks like a printed stem to the model, while its ink
                // colour says pen
                // (except on rule-shaped ink: the colour layers' grid lines are dropped)
                if (modelSaysHw) hw[x] = 255;
                else if (rs[x]) hw[x] = 0;
                if (!hw[x]) { ov[x] = 0; pr[x] = 255; }
            }
        }
    }
    // per piece of ink with the rules taken out: a pen letter written on an answer line touches the
    // line, so as one stroke with it, it was judged pixel by pixel — and a black pen "I" came out
    // half handwriting, half print (its colour is the print's), leaving half its stem after the
    // erase. Off the rules each letter is its own piece; one that is mostly handwriting is
    // handwriting whole (print under it is still remembered from the model's print layer below)
    {
        cv::Mat pieces = ink & ~ruleShaped, plab;
        const int np_ = cv::connectedComponents(pieces, plab, 8, CV_32S);
        // print clear of the pen (more than 2 px from handwriting): a printed letter a pen circle
        // runs into has its body there; the half of a pen stem the colour called print does not
        cv::Mat near; cv::dilate(handwriting, near, cv::Mat::ones(5, 5, CV_8U));
        const cv::Mat farPrint = print & ~near;
        // only pieces written on a rule (touching it): elsewhere a piece is a whole stroke already
        cv::Mat onRule; cv::dilate(ruleShaped, onRule, cv::Mat::ones(3, 3, CV_8U));
        std::vector<int> parea(np_, 0), phw(np_, 0), pfar(np_, 0);
        std::vector<char> touch(np_, 0);
        for (int y = 0; y < ink.rows; y++) {
            const int *lp = plab.ptr<int>(y);
            const uchar *hp = handwriting.ptr<uchar>(y), *fp = farPrint.ptr<uchar>(y), *rp = onRule.ptr<uchar>(y);
            for (int x = 0; x < ink.cols; x++) if (lp[x] > 0) {
                parea[lp[x]]++; if (hp[x]) phw[lp[x]]++; if (fp[x]) pfar[lp[x]]++; if (rp[x]) touch[lp[x]] = 1;
            }
        }
        std::vector<char> whole(np_, 0);
        for (int l = 1; l < np_; l++)
            whole[l] = touch[l] && phw[l] >= kPieceHalf * parea[l] && phw[l] < parea[l] && pfar[l] < std::max(5.0, 0.05 * parea[l]);
        for (int y = 0; y < ink.rows; y++) {
            const int *lp = plab.ptr<int>(y);
            uchar *hp = handwriting.ptr<uchar>(y), *pr = print.ptr<uchar>(y), *ov = overlap.ptr<uchar>(y);
            for (int x = 0; x < ink.cols; x++) if (whole[lp[x]] && !hp[x]) { hp[x] = 255; pr[x] = 0; ov[x] = 0; }
        }
    }
    // grow along the ink from what is surely handwriting
    cv::Mat allowed = (modelHw > kGrowHw) & ink, grown = handwriting.clone(), next;
    const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    for (int i = 0; i < kGrowSteps; i++) {
        cv::dilate(grown, next, kernel);
        next = (next & allowed) | grown;
        const bool changed = cv::countNonZero(next != grown) > 0;
        grown = next;
        if (!changed) break;
    }
    const cv::Mat fresh = grown & ~handwriting;
    print &= ~fresh;
    overlap &= ~fresh;
    handwriting = grown;
    // print under the pen, remembered for the redraw
    const cv::Mat under = (modelPrint > kUnderPenPrint) & handwriting;
    overlap |= under;
    print |= under;
}


// MARK: - Model finish (mirror ModelFinish.kt on Android)
//
// "Ink colour + AI" in two stages: the colour method decides first and what it calls handwriting
// stays handwriting, except whole strokes (and long ruled runs) the model is SURE are print —
// coloured print colour alone cannot tell from pen. The model then only finishes what colour left:
// tails grown from the colour-found strokes, and whole pen strokes the colour of the print that it
// is sure of. Print it sees under the new handwriting is remembered as overlap, to be redrawn.

static const double kFinishSureHw = 0.7, kFinishMaxPrint = 1.0, kFinishUnderPen = 0.5;
static const double kVetoPrint = 0.6, kVetoHw = 0.2, kRulePrint = 0.5, kRuleHw = 0.3;
// a colour stroke's pixels the model sees as plain print (this much print, this little pen) are
// print once they are this share of the stroke
static const double kPixelPrint = 0.6, kPixelHw = 0.2, kPixelShare = 0.02;
// a whole stroke among the rest is taken only when the model sees handwriting this far above print
static const double kFinishSureMargin = 0.2;
static const int kFinishRimSteps = 3;
static const double kFinishRimThr = 0.1;
static const int kFinishGrowSteps = 40;

/// Mean of [a] and [b] per connected component of [mask]; returns labels and the per-label means.
static int ComponentMeans(const cv::Mat &mask, const cv::Mat &a, const cv::Mat &b, cv::Mat &labels,
                          std::vector<int> &area, std::vector<double> &meanA, std::vector<double> &meanB) {
    const int n = cv::connectedComponents(mask, labels, 8, CV_32S);
    area.assign(n, 0); meanA.assign(n, 0); meanB.assign(n, 0);
    for (int y = 0; y < mask.rows; y++) {
        const int *l = labels.ptr<int>(y);
        const float *pa = a.ptr<float>(y), *pb = b.ptr<float>(y);
        for (int x = 0; x < mask.cols; x++)
            if (l[x] > 0) { area[l[x]]++; meanA[l[x]] += pa[x]; meanB[l[x]] += pb[x]; }
    }
    for (int i = 1; i < n; i++) if (area[i]) { meanA[i] /= area[i]; meanB[i] /= area[i]; }
    return n;
}

static void ModelFinish(cv::Mat &handwriting, cv::Mat &print, cv::Mat &overlap, const cv::Mat &ink,
                        const cv::Mat &modelHw, const cv::Mat &modelPrint, double threshold, int minArea) {
    // 1. whole colour strokes the model is sure are print
    {
        cv::Mat labels; std::vector<int> area; std::vector<double> mh, mp;
        const int n = ComponentMeans(handwriting, modelHw, modelPrint, labels, area, mh, mp);
        std::vector<char> veto(n, 0);
        for (int i = 1; i < n; i++) veto[i] = mp[i] >= kVetoPrint && mh[i] <= kVetoHw;
        // printed letters a pen line crossed or circled join the pen's stroke, and the colour test puts
        // the whole word in the handwriting: where the model sees plain print in such a stroke, those
        // pixels are print — the pen's own pixels stay handwriting
        auto plainPrint = [&](int y, int x) {
            return modelPrint.at<float>(y, x) >= kPixelPrint && modelHw.at<float>(y, x) <= kPixelHw;
        };
        std::vector<int> sure(n, 0);
        for (int y = 0; y < labels.rows; y++) {
            const int *l = labels.ptr<int>(y);
            for (int x = 0; x < labels.cols; x++) if (l[x] && plainPrint(y, x)) sure[l[x]]++;
        }
        std::vector<char> split(n, 0);
        for (int i = 1; i < n; i++) split[i] = !veto[i] && sure[i] >= kPixelShare * area[i];
        for (int y = 0; y < labels.rows; y++) {
            const int *l = labels.ptr<int>(y);
            uchar *hw = handwriting.ptr<uchar>(y), *pr = print.ptr<uchar>(y), *ov = overlap.ptr<uchar>(y);
            for (int x = 0; x < labels.cols; x++)
                if (veto[l[x]] || (split[l[x]] && plainPrint(y, x))) { hw[x] = 0; pr[x] = 255; ov[x] = 0; }
        }
    }
    // 2. long straight runs (a coloured grid / answer line fused with the writing), pixel by pixel
    {
        const int len = std::max(15, std::max(handwriting.cols, handwriting.rows) / 80);
        cv::Mat h, v;
        cv::morphologyEx(handwriting, h, cv::MORPH_OPEN, cv::Mat::ones(1, len, CV_8U));
        cv::morphologyEx(handwriting, v, cv::MORPH_OPEN, cv::Mat::ones(len, 1, CV_8U));
        const cv::Mat rules = (h | v) & (modelPrint > kRulePrint) & (modelHw < kRuleHw);
        handwriting &= ~rules;
        overlap &= ~rules;
        print |= rules;
    }
    const cv::Mat colourHw = handwriting.clone();
    const cv::Mat modelSays = (modelHw > threshold) & ink & ~colourHw;
    // 3. tails: grow the colour-found handwriting into model-handwriting ink it touches
    cv::Mat grown = colourHw.clone(), next;
    const cv::Mat k3 = cv::Mat::ones(3, 3, CV_8U);
    for (int i = 0; i < kFinishGrowSteps; i++) {
        cv::dilate(grown, next, k3);
        next = (next & modelSays) | grown;
        const bool changed = cv::countNonZero(next != grown) > 0;
        grown = next;
        if (!changed) break;
    }
    // 4. pen the colour of the print: whole strokes the model is sure of
    {
        const cv::Mat rest = modelSays & ~grown;
        cv::Mat labels; std::vector<int> area; std::vector<double> mh, mp;
        const int n = ComponentMeans(rest, modelHw, modelPrint, labels, area, mh, mp);
        std::vector<char> take(n, 0);
        // (an outline of a printed drawing — a cloud's arcs — can read as pen to the model, but never
        // by such a margin over print)
        for (int i = 1; i < n; i++)
            take[i] = area[i] >= minArea && mh[i] >= kFinishSureHw && mp[i] < kFinishMaxPrint && mh[i] - mp[i] >= kFinishSureMargin;
        for (int y = 0; y < labels.rows; y++) {
            const int *l = labels.ptr<int>(y);
            uchar *g = grown.ptr<uchar>(y);
            for (int x = 0; x < labels.cols; x++) if (take[l[x]]) g[x] = 255;
        }
    }
    // rims and thin ends of the strokes found: ink right next to a stroke that the model leans to
    // as pen (above kFinishRimThr) and does not see as print goes with it (kFinishRimSteps px)
    {
        const cv::Mat rim = (modelHw > kFinishRimThr) & (modelPrint < 0.5) & ink;
        cv::Mat nx;
        for (int i = 0; i < kFinishRimSteps; i++) { cv::dilate(grown, nx, k3); grown |= (nx & rim); }
    }
    // what the model added: no longer print, except print it sees under the pen (redrawn)
    const cv::Mat added = grown & ~colourHw;
    const cv::Mat under = (modelPrint > kFinishUnderPen) & added;
    print = (print & ~added) | under;
    overlap |= under;
    handwriting = grown;
    // a pen stroke on a grainy photo comes out speckled: ink the stroke closes over (a 3×3 closing) is
    // the stroke's too — unless the model sees print there and no pen (a printed letter it touches)
    cv::Mat closed;
    cv::morphologyEx(handwriting, closed, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    closed &= ink & ~((modelPrint > kVetoPrint) & (modelHw < kVetoHw));
    handwriting |= closed;
    print &= ~closed;
}

// MARK: - Isolated print (mirror IsolatedPrint.kt on Android)
//
// Second look at what the first pass left beside the handwriting it found: only pieces of "print"
// touching the found strokes' surroundings are judged. Printed text never stands alone — a letter
// has its word beside it, a word its line, a digit its label. Such a letter-sized piece with no
// other print within a text height of it is a pen leftover the detectors missed (a black-pen dot, a
// short tick, a scribble, the tail of a stroke cut off from it) and becomes handwriting — unless it
// is a letter or digit after all: a typeset page repeats its glyphs, so a lone printed "5" has a
// twin of the same shape and size among the page's other print, and a scribble has none; a glyph
// the page has only once the model still sees as print. Specks are not judged, and neither are
// pieces larger than a few letters. Rules are not text: a pen piece standing on an answer line is
// judged without the line, and only the piece (never the line) becomes handwriting.

static const double kIsoReach = 1.0, kIsoMaxSize = 1.5, kIsoMinSize = 0.4, kIsoTwinIou = 0.6, kIsoSurePrint = 0.6;
static const int kIsoLeftoverUnits = 2;

// case 2 — a piece amid the writing (see IsolatedPrint.kt): connected to the pen, letter-tall, with
// handwriting around it and no printed letter right beside it, not crossed or encircled by the pen,
// not set off by a coloured pen, the model not reading it as print, and no printed twin.
static const double kAmidMinH = 0.55, kAmidMaxW = 2.5, kAmidPrint = 0.5, kAmidCrossed = 0.6,
                    kAmidColour = 4.0, kAmidKeepPrint = 0.45, kAmidTwinIou = 0.75;
static const int kAmidClear = 2;

static void IsolatedPrint(cv::Mat &handwriting, cv::Mat &print, const cv::Mat &overlap, const cv::Mat &modelPrint, int minArea, const cv::Mat &src) {
    const int W = handwriting.cols, H = handwriting.rows;
    const int unit = inkanalysis::StrokeUnit(handwriting);
    // print as the mask will show it: none of the pen's pixels; print under the pen counts
    cv::Mat allPrint = (print & ~handwriting) | overlap;
    RemoveSmallComponents(allPrint, std::max(3, minArea / 4));   // paper specks are not print
    // rules (answer lines, table grid): long straight runs, taken out of the print first
    {
        const int len = std::max(15, std::max(W, H) / 60);
        cv::Mat h, v;
        cv::morphologyEx(allPrint, h, cv::MORPH_OPEN, cv::Mat::ones(1, len, CV_8U));
        cv::morphologyEx(allPrint, v, cv::MORPH_OPEN, cv::Mat::ones(len, 1, CV_8U));
        cv::Mat rules = h | v;
        cv::dilate(rules, rules, cv::Mat::ones(3, 3, CV_8U));
        allPrint &= ~rules;
        RemoveSmallComponents(allPrint, std::max(3, minArea / 4));
    }
    cv::Mat labels, stats, cents;
    const int n = cv::connectedComponentsWithStats(allPrint, labels, stats, cents, 8, CV_32S);
    if (n < 50) return;   // too little print to know how it stands
    auto S = [&](int i, int f) { return stats.at<int>(i, f); };
    // text height: the median letter (pieces of a letter's size, not specks or grid dots)
    std::vector<int> hs;
    for (int i = 1; i < n; i++) if (S(i, cv::CC_STAT_AREA) >= std::max(minArea, 4 * unit * unit)) hs.push_back(S(i, cv::CC_STAT_HEIGHT));
    if (hs.empty()) return;
    std::sort(hs.begin(), hs.end());
    const double cap = hs[hs.size() / 2];
    // pieces closer than the reach join one group; a group of one is a piece alone
    const int r = std::max(1, (int)std::lround(kIsoReach * cap / 2));
    cv::Mat near, groups;
    cv::dilate(allPrint, near, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r + 1, 2 * r + 1)));
    const int g = cv::connectedComponents(near, groups, 8, CV_32S);
    std::vector<int> members(g, 0), groupOf(n, -1);
    cv::Mat rim;
    const int d = 2 * kIsoLeftoverUnits * unit + 1;
    cv::dilate(handwriting, rim, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(d, d)));
    std::vector<char> touchesPen(n, 0), underPen(n, 0);
    std::vector<double> sumPr(n, 0);
    for (int y = 0; y < H; y++) {
        const int *lp = labels.ptr<int>(y), *gp = groups.ptr<int>(y);
        const uchar *rp = rim.ptr<uchar>(y), *op = overlap.ptr<uchar>(y);
        const float *pp = modelPrint.ptr<float>(y);
        for (int x = 0; x < W; x++) {
            const int l = lp[x]; if (!l) continue;
            if (groupOf[l] < 0) { groupOf[l] = gp[x]; members[gp[x]]++; }
            if (rp[x]) touchesPen[l] = 1;
            if (op[x]) underPen[l] = 1;
            sumPr[l] += pp[x];
        }
    }
    auto h = [&](int i) { return S(i, cv::CC_STAT_HEIGHT); };
    auto w = [&](int i) { return S(i, cv::CC_STAT_WIDTH); };
    std::vector<char> alone(n, 0);
    for (int i = 1; i < n; i++)
        alone[i] = members[groupOf[i]] == 1 && !underPen[i] && touchesPen[i] &&
                   h(i) <= kIsoMaxSize * cap && w(i) <= kIsoMaxSize * cap &&
                   std::max(h(i), w(i)) >= kIsoMinSize * cap &&
                   sumPr[i] / S(i, cv::CC_STAT_AREA) < kIsoSurePrint;
    // a letter or digit after all: a twin among the print that stands with other print
    std::vector<int> ref;
    for (int i = 1; i < n; i++)
        if (!alone[i] && members[groupOf[i]] > 1 && S(i, cv::CC_STAT_AREA) >= minArea && h(i) <= kIsoMaxSize * cap && w(i) <= kIsoMaxSize * cap)
            ref.push_back(i);
    std::sort(ref.begin(), ref.end(), [&](int a, int b) { return h(a) < h(b); });
    auto iou = [&](int i, int j) {
        const int xi = S(i, cv::CC_STAT_LEFT), yi = S(i, cv::CC_STAT_TOP), xj = S(j, cv::CC_STAT_LEFT), yj = S(j, cv::CC_STAT_TOP);
        int inter = 0, uni = 0;
        for (int yy = 0; yy < h(i); yy++) for (int xx = 0; xx < w(i); xx++) {
            const bool a = labels.at<int>(yi + yy, xi + xx) == i;
            const bool b = labels.at<int>(yj + yy * h(j) / h(i), xj + xx * w(j) / w(i)) == j;
            if (a && b) inter++;
            if (a || b) uni++;
        }
        return uni ? (double)inter / uni : 0.0;
    };
    for (int i = 1; i < n; i++) {
        if (!alone[i]) continue;
        const int hLo = (int)std::floor(h(i) * 0.85), hHi = (int)std::ceil(h(i) * 1.18);
        auto it = std::lower_bound(ref.begin(), ref.end(), hLo, [&](int a, int v) { return h(a) < v; });
        for (; it != ref.end() && h(*it) <= hHi; ++it) {
            const int j = *it;
            if (w(j) < 0.82 * w(i) || w(j) > 1.22 * w(i)) continue;
            if (iou(i, j) >= kIsoTwinIou) { alone[i] = 0; break; }
        }
    }
    // case 2: pieces amid the writing
    {
        auto X = [&](int i) { return S(i, cv::CC_STAT_LEFT); };
        auto Y = [&](int i) { return S(i, cv::CC_STAT_TOP); };
        auto A = [&](int i) { return S(i, cv::CC_STAT_AREA); };
        const int gx = (int)std::lround(cap), gy = (int)std::lround(0.3 * cap), half = (int)std::lround(0.6 * cap);
        const int px = (int)std::lround(0.25 * cap);
        cv::Mat hwI, prI, hw01 = handwriting > 0, tall(H, W, CV_8U, cv::Scalar(0));
        for (int y = 0; y < H; y++) { const int *lp = labels.ptr<int>(y); uchar *tp = tall.ptr<uchar>(y);
            for (int x = 0; x < W; x++) tp[x] = (lp[x] && h(lp[x]) >= kAmidMinH * cap) ? 1 : 0; }
        cv::integral(hw01 / 255, hwI, CV_32S); cv::integral(tall, prI, CV_32S);
        auto count = [&](const cv::Mat &I, int x0, int y0, int x1, int y1) {
            x0 = std::max(0, x0); y0 = std::max(0, y0); x1 = std::min(W, x1); y1 = std::min(H, y1);
            if (x1 <= x0 || y1 <= y0) return 0;
            return I.at<int>(y1, x1) - I.at<int>(y0, x1) - I.at<int>(y1, x0) + I.at<int>(y0, x0);
        };
        cv::Mat reach, touch;
        cv::dilate(handwriting, reach, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * gx + 1, 2 * gx + 1)));
        cv::dilate(handwriting, touch, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kAmidClear + 1, 2 * kAmidClear + 1)));
        cv::Mat lab3; cv::cvtColor(src, lab3, cv::COLOR_BGR2Lab);
        std::vector<char> nearW(n, 0), connected(n, 0);
        std::vector<double> sumC(n, 0); std::vector<int> cntC(n, 0);
        cv::Mat chroma(H, W, CV_32F);
        for (int y = 0; y < H; y++) {
            const int *lp = labels.ptr<int>(y); const uchar *rp = reach.ptr<uchar>(y), *tp = touch.ptr<uchar>(y);
            const cv::Vec3b *lb = lab3.ptr<cv::Vec3b>(y); float *cp = chroma.ptr<float>(y);
            for (int x = 0; x < W; x++) {
                const double a = lb[x][1] - 128.0, b = lb[x][2] - 128.0; cp[x] = (float)std::sqrt(a * a + b * b);
                const int l = lp[x]; if (!l) continue;
                if (rp[x]) nearW[l] = 1;
                if (tp[x]) connected[l] = 1;
                sumC[l] += cp[x]; cntC[l]++;
            }
        }
        double sp = 0; long np = 0;
        for (int i = 1; i < n; i++) if (!nearW[i] && h(i) >= kIsoMinSize * cap) { sp += sumC[i]; np += cntC[i]; }
        const double printChroma = np ? sp / np : 0.0;
        auto hwChroma = [&](int x0, int y0, int x1, int y1, double &out) {
            double s0 = 0; int c0 = 0;
            for (int yy = std::max(0, y0); yy < std::min(H, y1); yy++) for (int xx = std::max(0, x0); xx < std::min(W, x1); xx++)
                if (handwriting.at<uchar>(yy, xx)) { s0 += chroma.at<float>(yy, xx); c0++; }
            if (c0 < 10) return false;
            out = s0 / c0; return true;
        };
        for (int i = 1; i < n; i++) {
            if (alone[i] || underPen[i] || !nearW[i] || !connected[i]) continue;
            if (h(i) < kAmidMinH * cap || h(i) > kIsoMaxSize * cap || w(i) > kAmidMaxW * cap || w(i) > 3 * h(i)) continue;
            // crossed by the pen (case 1): pen or pen-over-print through most of its box's columns
            int cols = 0;
            for (int xx = X(i); xx < X(i) + w(i); xx++) for (int yy = Y(i); yy < Y(i) + h(i); yy++)
                if ((handwriting.at<uchar>(yy, xx) || overlap.at<uchar>(yy, xx)) && labels.at<int>(yy, xx) != i) { cols++; break; }
            if (cols >= kAmidCrossed * w(i)) continue;
            const int y0 = Y(i) - gy, y1 = Y(i) + h(i) + gy;
            const int hwN = count(hwI, X(i) - gx, y0, X(i) + w(i) + gx, y1);
            const int prN = count(prI, X(i) - px, y0, X(i) + w(i) + px, y1) - A(i);
            if (hwN < 2 * A(i) || prN > kAmidPrint * A(i)) continue;
            const bool up = count(hwI, X(i), Y(i) - half, X(i) + w(i), Y(i)) > 0;
            const bool down = count(hwI, X(i), Y(i) + h(i), X(i) + w(i), Y(i) + h(i) + half) > 0;
            if (up && down) continue;
            const bool left = count(hwI, X(i) - half, Y(i), X(i), Y(i) + h(i)) > 0;
            const bool right = count(hwI, X(i) + w(i), Y(i), X(i) + w(i) + half, Y(i) + h(i)) > 0;
            if (up && down && left && right) continue;
            double cw;
            const double cpc = sumC[i] / std::max(1, cntC[i]);
            if (hwChroma(X(i) - gx, y0, X(i) + w(i) + gx, y1, cw) && cw > std::max(cpc, printChroma) + kAmidColour) continue;
            if (sumPr[i] / A(i) >= kAmidKeepPrint) continue;
            // a printed twin among print standing clear of the writing
            bool twin = false;
            const int hLo = (int)std::floor(h(i) * 0.85), hHi = (int)std::ceil(h(i) * 1.18);
            for (int j = 1; j < n && !twin; j++) {
                if (j == i || nearW[j] || h(j) < hLo || h(j) > hHi || w(j) < 0.82 * w(i) || w(j) > 1.22 * w(i)) continue;
                if (std::abs(A(j) - A(i)) > 0.25 * A(i)) continue;
                twin = iou(i, j) >= kAmidTwinIou;
            }
            if (twin) continue;
            alone[i] = 1;
        }
    }
    for (int y = 0; y < H; y++) {
        const int *lp = labels.ptr<int>(y);
        uchar *hp = handwriting.ptr<uchar>(y), *pr = print.ptr<uchar>(y);
        for (int x = 0; x < W; x++) if (alone[lp[x]]) { hp[x] = 255; pr[x] = 0; }
    }
}

// MARK: - Strike-throughs (mirror StrikeThrough.kt on Android)
//
// Pen strike-throughs across printed words ("Fruit / are / your" crossed out). A black pen line has
// the print's colour, so the colour layers call it print and it was never erased. It is told from
// print by shape — measured on 30 labelled pages, every feature below separates the two:
//  - it is one straight thin line of ink, unbroken for 1.5 text heights: an opening with a 1-px line
//    kernel at small tilts, on the ink itself (no growth — growing the ink first merged the tops of
//    bold letters into "lines"). Printed text at mid height always has paper between letters.
//  - printed letters cross it (ink runs on through it, up and down) at 2+ places, and the letters
//    are text sized (a drawing or a table border is not).
//  - its thickness is a pen stroke's (0.5–1.25 stroke units): a bold letter's edge or a picture's
//    line is thicker, a grid line or wood grain thinner.
//  - it crosses the gaps between letters: in a share of its columns nothing touches it right above
//    or below (a run along the tops of touching bold letters has a letter under every column).
//  - it is hand drawn: it wobbles off a straight fit, or overshoots the word onto the paper.
// The line becomes handwriting, but it is erased only where it stands free (paper right above and
// below: between letters, inside a counter, the overshoot). Where a letter crosses it, or its pixels
// lie right against a letter's arch or bar, or taking them would open a letter's closed counter (the
// eye of an "e"), they are print under the pen (overlap), drawn back. A line along the letters' tops
// or feet loses only its word gaps and overshoot. Its own pixels go to the mask file (the strike
// layer, kStrikeAlpha), so the erase does not re-join them as a printed rule nor re-spell the words
// it crossed.

/// Found at this long side at most (the line opening costs pixels × kernel length).
static const int kStrikeWorkLong = 1600;
static const double kStrikeAngles[] = {-8.0, -5.0, -2.5, 0.0, 2.5, 5.0, 8.0};
/// An unbroken line at least this many text heights long (and 8 stroke units).
static const double kStrikeMinLen = 1.5;
/// No longer than this: a longer straight line is a rule or an answer line.
static const double kStrikeMaxLen = 12.0;
/// Ink running on through the line this far (text heights) up and down is a letter crossing.
static const double kStrikeCrossReach = 0.15;
/// A column is over paper when no ink is this near (text heights) above and below.
static const double kStrikeClearReach = 0.4;

/// Kotlin's roundToInt (halves round up), so both platforms build the same kernels and sizes.
static int RoundHalfUp(double v) { return (int)std::floor(v + 0.5); }

/// A 1-px line `len` long tilted by `deg` degrees, as a structuring element.
static cv::Mat StrikeLineKernel(int len, double deg) {
    const int half = len / 2;
    const double rise = std::tan(deg * CV_PI / 180.0) * half;
    const int dy = RoundHalfUp(std::fabs(rise));
    cv::Mat kern = cv::Mat::zeros(2 * dy + 1, 2 * half + 1, CV_8U);
    cv::line(kern, cv::Point(0, dy - RoundHalfUp(rise)), cv::Point(2 * half, dy + RoundHalfUp(rise)), cv::Scalar(1), 1);
    return kern;
}

/// Standard deviation of the line's centre row around its least-squares straight fit.
static double StrikeFitDeviation(const std::vector<int> &cols, const std::vector<int> &top, const std::vector<int> &bot) {
    const int m = (int)cols.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int j = 0; j < m; j++) {
        const double y = (top[cols[j]] + bot[cols[j]]) / 2.0;
        sx += j; sy += y; sxx += (double)j * j; sxy += j * y;
    }
    const double den = m * sxx - sx * sx;
    const double a = den != 0 ? (m * sxy - sx * sy) / den : 0;
    const double b = (sy - a * sx) / m;
    double ss = 0;
    for (int j = 0; j < m; j++) { const double r = (top[cols[j]] + bot[cols[j]]) / 2.0 - (a * j + b); ss += r * r; }
    return std::sqrt(ss / m);
}

/// Ink of something other than line `l` right beside (`x`, `y`) (8-neighbours).
static bool StrikeBesideLetter(const std::vector<uchar> &ink, const std::vector<int> &lab, int l, int w, int h, int x, int y) {
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        const int px = x + dx, py = y + dy;
        if (px < 0 || px >= w || py < 0 || py >= h) continue;
        const int i = py * w + px;
        if (ink[i] && lab[i] != l) return true;
    }
    return false;
}

/// A letter's closed counter (the eye of an "e", whose bar the strike ran along) must stay closed:
/// erased line pixels that would open a hole of the ink to the paper outside are kept. Splitting one
/// hole in two by the strike and joining them back (an "o" struck through) opens nothing.
static void StrikeKeepClosedCounters(const std::vector<uchar> &ink, const std::vector<char> &out, std::vector<char> &keep,
                                     int w, int xa, int ya, int xb, int yb) {
    const int cw = xb - xa, ch = yb - ya;
    if (cw < 3 || ch < 3) return;
    auto paperLabels = [&](bool erased) {
        cv::Mat m(ch, cw, CV_8U);
        for (int y = 0; y < ch; y++) {
            uchar *mp = m.ptr<uchar>(y);
            for (int x = 0; x < cw; x++) {
                const int i = (ya + y) * w + xa + x;
                const bool inkHere = ink[i] && !(erased && out[i] && !keep[i]);
                mp[x] = inkHere ? 0 : 1;
            }
        }
        cv::Mat lb;
        cv::connectedComponents(m, lb, 4, CV_32S);
        return lb;
    };
    auto outside = [&](const cv::Mat &lab) {
        std::set<int> o;
        for (int x = 0; x < cw; x++) { o.insert(lab.at<int>(0, x)); o.insert(lab.at<int>(ch - 1, x)); }
        for (int y = 0; y < ch; y++) { o.insert(lab.at<int>(y, 0)); o.insert(lab.at<int>(y, cw - 1)); }
        o.erase(0);
        return o;
    };
    const cv::Mat before = paperLabels(false);
    const std::set<int> outB = outside(before);
    const cv::Mat after = paperLabels(true);
    const std::set<int> outA = outside(after);
    // holes before whose paper is now joined to the outside
    std::set<int> opened;
    for (int y = 0; y < ch; y++) for (int x = 0; x < cw; x++) {
        const int b = before.at<int>(y, x);
        if (b > 0 && !outB.count(b) && outA.count(after.at<int>(y, x))) opened.insert(b);
    }
    if (opened.empty()) return;
    for (int y = 0; y < ch; y++) for (int x = 0; x < cw; x++) {
        const int i = (ya + y) * w + xa + x;
        if (!out[i] || keep[i]) continue;
        // an erased pixel bordering an opened hole
        bool near = false;
        for (int dy = -1; dy <= 1 && !near; dy++) for (int dx = -1; dx <= 1; dx++) {
            const int px = x + dx, py = y + dy;
            if (px >= 0 && px < cw && py >= 0 && py < ch && opened.count(before.at<int>(py, px))) { near = true; break; }
        }
        if (near) keep[i] = 1;
    }
}

/// `src`: the page (BGR); `handwriting` / `print` / `overlap`: CV_8UC1 255 = set, updated in place.
/// Returns the strike lines' own pixels (not where letters cross them), for the mask file — or an
/// empty Mat when there are none.
static cv::Mat StrikeThrough(const cv::Mat &src, cv::Mat &handwriting, cv::Mat &print, cv::Mat &overlap) {
    const int fw = src.cols, fh = src.rows;
    const double s = std::min(1.0, (double)kStrikeWorkLong / std::max(fw, fh));
    const int w = std::max(1, RoundHalfUp(fw * s)), h = std::max(1, RoundHalfUp(fh * s));
    // dark ink (well clear of the paper) at the working scale
    cv::Mat inkM;
    {
        cv::Mat small, g, bg;
        if (s < 1.0) cv::resize(src, small, cv::Size(w, h), 0, 0, cv::INTER_AREA); else small = src;
        cv::cvtColor(small, g, cv::COLOR_BGR2GRAY);
        cv::morphologyEx(g, bg, cv::MORPH_CLOSE, cv::Mat::ones(25, 25, CV_8U));
        cv::threshold(bg - g, inkM, 40, 1, cv::THRESH_BINARY);
    }
    std::vector<uchar> ink(inkM.begin<uchar>(), inkM.end<uchar>());
    auto scaled = [&](const cv::Mat &m) {
        cv::Mat t;
        if (s < 1.0) cv::resize(m, t, cv::Size(w, h), 0, 0, cv::INTER_NEAREST); else t = m;
        return t;
    };
    const cv::Mat hwS = scaled(handwriting), prS = scaled(print);
    // pure print (print and not handwriting): what the line must cross
    const cv::Mat prmM = (prS != 0) & (hwS == 0);
    std::vector<char> prm(w * h);
    for (int y = 0; y < h; y++) { const uchar *p = prmM.ptr<uchar>(y); for (int x = 0; x < w; x++) prm[y * w + x] = p[x] != 0; }

    // stroke unit: twice the median ridge of the ink's distance transform
    int k;
    {
        cv::Mat dt, dil;
        cv::distanceTransform(inkM, dt, cv::DIST_L2, 3);
        cv::dilate(dt, dil, cv::Mat::ones(3, 3, CV_8U));
        std::vector<float> ridge;
        for (int y = 0; y < h; y++) {
            const float *dp = dt.ptr<float>(y), *mp = dil.ptr<float>(y);
            for (int x = 0; x < w; x++) if (dp[x] > 0 && dp[x] >= mp[x]) ridge.push_back(dp[x]);
        }
        if (ridge.empty()) return cv::Mat();
        std::sort(ridge.begin(), ridge.end());
        k = std::max(2, RoundHalfUp(2 * ridge[ridge.size() / 2]));
    }
    // text height: median height of letter-shaped print pieces (not dots, accents, rules)
    double textH;
    {
        cv::Mat lb, st, ce;
        const int n = cv::connectedComponentsWithStats(prmM, lb, st, ce, 8, CV_32S);
        std::vector<int> hs;
        for (int l = 1; l < n; l++) {
            const int cw = st.at<int>(l, cv::CC_STAT_WIDTH), ch = st.at<int>(l, cv::CC_STAT_HEIGHT), a = st.at<int>(l, cv::CC_STAT_AREA);
            if (a >= 6 * k * k && ch < h / 20 && cw >= 0.4 * ch) hs.push_back(ch);
        }
        std::sort(hs.begin(), hs.end());
        textH = hs.empty() ? 8.0 * k : (double)hs[hs.size() / 2];
    }
    // straight unbroken lines: opening with a thin line kernel at small tilts
    const int len = RoundHalfUp(std::max(kStrikeMinLen * textH, 8.0 * k)) | 1;
    cv::Mat runM = cv::Mat::zeros(h, w, CV_8U);
    for (double deg : kStrikeAngles) {
        cv::Mat t;
        cv::morphologyEx(inkM, t, cv::MORPH_OPEN, StrikeLineKernel(len, deg));
        runM |= t;
    }
    cv::Mat labM, stats, cents;
    const int n = cv::connectedComponentsWithStats(runM, labM, stats, cents, 8, CV_32S);
    std::vector<int> lab(labM.begin<int>(), labM.end<int>());
    // ink blobs: a struck word stays about a text line tall, a drawing does not
    cv::Mat inkLabM, inkSt, inkCe;
    cv::connectedComponentsWithStats(inkM, inkLabM, inkSt, inkCe, 8, CV_32S);
    std::vector<int> inkLab(inkLabM.begin<int>(), inkLabM.end<int>());
    auto inkAt = [&](int x, int y) { return y >= 0 && y < h && ink[y * w + x] != 0; };
    const int reach = std::max(2, (int)(kStrikeCrossReach * textH)), clearR = std::max(4, (int)(kStrikeClearReach * textH));
    std::vector<char> out(w * h, 0), keep(w * h, 0);
    int found = 0;
    for (int l = 1; l < n; l++) {
        const int x0 = stats.at<int>(l, cv::CC_STAT_LEFT), y0 = stats.at<int>(l, cv::CC_STAT_TOP);
        const int bw = stats.at<int>(l, cv::CC_STAT_WIDTH), bh = stats.at<int>(l, cv::CC_STAT_HEIGHT);
        if (bw > kStrikeMaxLen * textH) continue;
        std::vector<int> top(bw, -1), bot(bw, -1);
        for (int y = y0; y < y0 + bh; y++) for (int c = 0; c < bw; c++) if (lab[y * w + x0 + c] == l) {
            if (top[c] < 0) top[c] = y;
            bot[c] = y;
        }
        std::vector<int> cols;
        for (int c = 0; c < bw; c++) if (top[c] >= 0) cols.push_back(c);
        if (cols.size() < 2) continue;
        const int m = (int)cols.size();
        std::vector<char> cross(m, 0), clear(m, 0), touchC(m, 0);
        std::vector<int> vext;
        int cpN = 0, touchN = 0;
        for (int j = 0; j < m; j++) {
            const int c = cols[j], x = x0 + c, t = top[c], b = bot[c];
            int u = 0; while (inkAt(x, t - u - 1)) u++;
            int d = 0; while (inkAt(x, b + d + 1)) d++;
            bool any = false;
            for (int y = std::max(0, t - clearR); y < std::max(0, t - 1); y++) if (ink[y * w + x]) { any = true; break; }
            if (!any) for (int y = b + 2; y < std::min(h, b + 1 + clearR); y++) if (ink[y * w + x]) { any = true; break; }
            clear[j] = !any;
            if (inkAt(x, t - 1) || inkAt(x, b + 1)) { touchN++; touchC[j] = 1; }
            if (u >= reach && d >= reach) {
                cross[j] = 1; vext.push_back(u + d + b - t + 1);
                const bool up = prm[(t - 1) * w + x] || prm[(t - u) * w + x];
                const bool dn = prm[(b + 1) * w + x] || prm[(b + d) * w + x];
                if (up && dn) cpN++;
            }
        }
        int groups = 0;
        for (int j = 0; j < m; j++) if (cross[j] && (j == 0 || !cross[j - 1])) groups++;
        if (groups < 2) continue;
        // features (units: text height / stroke unit)
        std::vector<int> ext;
        for (int c : cols) ext.push_back(bot[c] - top[c] + 1);
        std::sort(ext.begin(), ext.end());
        const double thick = (double)ext[ext.size() / 2] / k;
        std::sort(vext.begin(), vext.end());
        const double vmed = vext[vext.size() / 2] / textH;
        const double cprint = (double)cpN / vext.size();
        const double touch = (double)touchN / m;
        const double bridge = (double)std::count(clear.begin(), clear.end(), 1) / m;
        int ovL = m, ovR = m;
        for (int j = 0; j < m; j++) if (!clear[j]) { ovL = j; break; }
        for (int j = 0; j < m; j++) if (!clear[m - 1 - j]) { ovR = j; break; }
        const double ovmax = std::max(ovL, ovR) / textH;
        const double dev = StrikeFitDeviation(cols, top, bot);
        const double lenH = bw / textH;
        int blob = 0;
        {
            std::map<int, int> cnt;
            for (int c : cols) for (int y = top[c]; y <= bot[c]; y++) { const int il = inkLab[y * w + x0 + c]; if (il > 0) cnt[il]++; }
            int best = 0;
            for (const auto &e : cnt) if (e.second > best) { best = e.second; blob = e.first; }
        }
        const double blobH = blob > 0 ? inkSt.at<int>(blob, cv::CC_STAT_HEIGHT) / textH : 0.0;
        const bool ok = cprint >= 0.5 && blobH <= 2.0 && thick >= 0.5 && thick <= 1.25 && vmed <= 1.0 && touch <= 0.9 &&
                        (dev >= 0.3 || ovmax >= 0.2) && (lenH >= 2.0 || bridge >= 0.15);
#ifdef INK_DEBUG
        fprintf(stderr, "  strike %s (%d,%d) len %.2f thick %.2f dev %.2f touch %.2f vmed %.2f groups %d bridge %.2f cprint %.2f ov %.2f blob %.2f\n",
                ok ? "A" : "-", RoundHalfUp(x0 / s), RoundHalfUp(y0 / s), lenH, thick, dev, touch, vmed, groups, bridge, cprint, ovmax, blobH);
#endif
        if (!ok) continue;
        found++;
        // through the letters' middle, or along their tops / feet? Over the word, letters lie close
        // on both sides of a line through the middle; one along the tops has letter bodies under it
        // and only ascenders above
        int firstLetter = 0, lastLetter = m - 1;
        for (int j = 0; j < m; j++) if (cross[j] || touchC[j]) { firstLetter = j; break; }
        for (int j = m - 1; j >= 0; j--) if (cross[j] || touchC[j]) { lastLetter = j; break; }
        const int near = std::max(2, (int)(0.35 * textH));
        int nAbove = 0, nBelow = 0;
        for (int j = firstLetter; j <= lastLetter; j++) {
            const int x = x0 + cols[j], t = top[cols[j]], b = bot[cols[j]];
            for (int y = std::max(0, t - near); y < t; y++) if (ink[y * w + x]) { nAbove++; break; }
            for (int y = b + 1; y <= std::min(h - 1, b + near); y++) if (ink[y * w + x]) { nBelow++; break; }
        }
        const int span = std::max(1, lastLetter - firstLetter + 1);
        const bool edge = std::min(nAbove, nBelow) < 0.35 * span;
        // along the letters' edge the line lies on their arches and bars: only its free runs as wide
        // as a word space (and the overshoot) go
        std::vector<char> wideFree(m, 0);
        for (int j = 0; j < m;) {
            if (cross[j] || touchC[j]) { j++; continue; }
            int e = j;
            while (e + 1 < m && !cross[e + 1] && !touchC[e + 1]) e++;
            if (e - j + 1 >= 0.3 * textH) for (int q = j; q <= e; q++) wideFree[q] = 1;
            j = e + 1;
        }
        // erased only where it stands free (paper right above and below: a gap between letters, a
        // letter's counter, the overshoot); where it lies along a letter's own stroke (crossing a
        // stem, merged with an arch or a bar's top) those pixels may be the letter's: kept
        for (int j = 0; j < m; j++) {
            const int c = cols[j];
            for (int y = top[c]; y <= bot[c]; y++) {
                const int i = y * w + x0 + c;
                if (lab[i] != l) continue;
                out[i] = 1;
                const bool over = j < firstLetter || j > lastLetter;
                if (edge) { if (!over && !wideFree[j]) keep[i] = 1; }
                else if (cross[j]) keep[i] = 1;
                // merged with a letter's arch or bar: the pixels right against the letter's own ink
                else if (touchC[j] && StrikeBesideLetter(ink, lab, l, w, h, x0 + c, y)) keep[i] = 1;
            }
        }
        StrikeKeepClosedCounters(ink, out, keep, w, std::max(0, x0 - 2), std::max(0, y0 - clearR),
                                 std::min(w, x0 + bw + 2), std::min(h, y0 + bh + clearR));
    }
#ifdef INK_DEBUG
    fprintf(stderr, "strike: text %.0f px, unit %d, lines %d\n", textH / s, k, found);
#endif
    if (found == 0) return cv::Mat();

    // back to the page's scale: the found line (grown a pixel) on the page's own ink
    cv::Mat outM(h, w, CV_8U), keepM(h, w, CV_8U);
    for (int y = 0; y < h; y++) {
        uchar *op = outM.ptr<uchar>(y), *kp = keepM.ptr<uchar>(y);
        for (int x = 0; x < w; x++) { op[x] = out[y * w + x] ? 255 : 0; kp[x] = keep[y * w + x] ? 255 : 0; }
    }
    if (s < 1.0) {
        cv::resize(outM, outM, cv::Size(fw, fh), 0, 0, cv::INTER_NEAREST);
        cv::resize(keepM, keepM, cv::Size(fw, fh), 0, 0, cv::INTER_NEAREST);
        cv::dilate(outM, outM, cv::Mat::ones(3, 3, CV_8U));
        cv::dilate(keepM, keepM, cv::Mat::ones(3, 3, CV_8U));
    }
    cv::Mat line = cv::Mat::zeros(fh, fw, CV_8U);
    for (int y = 0; y < fh; y++) {
        const uchar *op = outM.ptr<uchar>(y), *kp = keepM.ptr<uchar>(y);
        uchar *hp = handwriting.ptr<uchar>(y), *pp = print.ptr<uchar>(y), *vp = overlap.ptr<uchar>(y), *lp = line.ptr<uchar>(y);
        for (int x = 0; x < fw; x++) {
            if (!op[x]) continue;
            if (!hp[x] && !pp[x] && !vp[x]) continue;                       // paper
            if (!kp[x]) lp[x] = 255;
            if (hp[x] && !vp[x]) continue;                                  // already pen
            hp[x] = 255;
            if (kp[x]) { vp[x] = 255; pp[x] = 255; } else { vp[x] = 0; pp[x] = 0; }
        }
    }
    return line;
}

@implementation ImageProcessingOpenCV

+ (BOOL)sharpenAtPath:(NSString *)inputPath
            outputPath:(NSString *)outputPath
                amount:(double)amount
                radius:(double)radius
                 error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(inputPath, &src, error)) return NO;

    cv::Mat blurred;
    double kernelRadius = radius <= 0.0 ? 3.0 : radius;
    cv::GaussianBlur(src, blurred, cv::Size(0, 0), kernelRadius);

    cv::Mat sharpened;
    cv::addWeighted(src, 1.0 + amount, blurred, -amount, 0.0, sharpened);

    return WriteOrFail(sharpened, outputPath, error);
}
                                                                                   
+ (BOOL)rotateAtPath:(NSString *)inputPath
           outputPath:(NSString *)outputPath
quarterTurnsClockwise:(NSInteger)quarterTurnsClockwise
                error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(inputPath, &src, error)) return NO;

    NSInteger normalizedTurns = ((quarterTurnsClockwise % 4) + 4) % 4;
    if (normalizedTurns != 0) {
        int rotateCode = normalizedTurns == 1 ? cv::ROTATE_90_CLOCKWISE
                        : normalizedTurns == 2 ? cv::ROTATE_180
                                                : cv::ROTATE_90_COUNTERCLOCKWISE;
        cv::rotate(src, src, rotateCode);
    }

    return WriteOrFail(src, outputPath, error);
}

+ (nullable NSString *)straightenAtPath:(NSString *)inputPath
                              outputPath:(NSString *)outputPath
                                   error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(inputPath, &src, error)) return nil;
    std::string how;
    cv::Mat dst = straighten::Straighten(src, &how);
    if (!WriteOrFail(dst, outputPath, error)) return nil;
    return [NSString stringWithUTF8String:how.c_str()];
}

+ (BOOL)removeShadowAtPath:(NSString *)inputPath
                 outputPath:(NSString *)outputPath
                      error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(inputPath, &src, error)) return NO;

    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    cv::Mat &l = channels[0];

    double sigma = std::min(60.0, std::max(15.0, std::max(src.cols, src.rows) * 0.05));
    cv::Mat background;
    cv::GaussianBlur(l, background, cv::Size(0, 0), sigma);

    cv::Mat lFloat, bgFloat, normFloat, lNormalized, lFinal;
    l.convertTo(lFloat, CV_32F);
    background.convertTo(bgFloat, CV_32F);
    bgFloat += 1.0; // avoid divide-by-zero on pure black

    // normalized = l * 255 / background: removes the page's overall shading (shadow),
    // ink stays dark relative to its local neighborhood.
    cv::divide(lFloat, bgFloat, normFloat, 255.0);
    normFloat.convertTo(lNormalized, CV_8U);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lNormalized, lFinal);

    std::vector<cv::Mat> mergedChannels = {lFinal, channels[1], channels[2]};
    cv::Mat merged, bgr;
    cv::merge(mergedChannels, merged);
    cv::cvtColor(merged, bgr, cv::COLOR_Lab2BGR);

    return WriteOrFail(bgr, outputPath, error);
}

+ (nullable NSNumber *)inkColorMaskAtPath:(NSString *)inputPath
                                  maskPath:(NSString *)maskPath
                                colorDelta:(double)colorDelta
                                     error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(inputPath, &src, error)) return nil;
    const double scale = WorkingScale(src.size());
    if (scale < 1.0) {
        return DetectAtWorkingSize(src, scale, maskPath, error, ^NSNumber *(NSString *input, NSString *mask, NSError **e) {
            return [self inkColorMaskAtPath:input maskPath:mask colorDelta:colorDelta error:e];
        });
    }

    // the model is only a second opinion here (colourless handwriting, print under a near-black
    // pen); without it the colour method still runs on its own
    cv::Mat modelPrint, modelHw;
    NSError *modelError = nil;
    if (!SegmentPage(src, &modelPrint, &modelHw, &modelError)) { modelPrint.release(); modelHw.release(); }
    cv::Mat handwriting, print, overlap;
    inkanalysis::DetectByInkColor(src, colorDelta, &handwriting, &print, &overlap, modelPrint, modelHw);
    inkanalysis::PenComponentVote(src, handwriting, print, overlap, modelHw);

    if (!WriteMaskFile(handwriting, print, overlap, maskPath, error)) return nil;
    return Coverage(handwriting);
}

+ (nullable NSNumber *)segmentationMaskAtPath:(NSString *)inputPath
                                      maskPath:(NSString *)maskPath
                                     threshold:(double)threshold
                                    colorDelta:(double)colorDelta
                                         error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(inputPath, &src, error)) return nil;
    const double scale = WorkingScale(src.size());
    if (scale < 1.0) {
        return DetectAtWorkingSize(src, scale, maskPath, error, ^NSNumber *(NSString *input, NSString *mask, NSError **e) {
            return [self segmentationMaskAtPath:input maskPath:mask threshold:threshold colorDelta:colorDelta error:e];
        });
    }

    cv::Mat printProb, handwritingProb;
    if (!SegmentPage(src, &printProb, &handwritingProb, error)) return nil;

    // The model runs at a reduced scale, so its masks are soft at full resolution — snap both to
    // the page's actual ink so erase touches strokes, not blobs of paper around them.
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat ink, handwriting, print;
    cv::compare(InkContrast(gray), kInkContrastLoose, ink, cv::CMP_GT);
    cv::compare(handwritingProb, threshold, handwriting, cv::CMP_GT);
    cv::bitwise_and(handwriting, ink, handwriting);
    RemoveSmallComponents(handwriting, MinSpeckleArea(src));
    cv::compare(printProb, kSegPrintThreshold, print, cv::CMP_GT);
    cv::bitwise_and(print, ink, print);
    // Two stages: the ink-colour method decides first (ink clearly unlike the print is handwriting,
    // exactly as "Ink colour" alone), then the model only finishes what colour left (ModelFinish).
    cv::Mat combinedHw, combinedPrint, overlap;
    inkanalysis::DetectByInkColor(src, colorDelta, &combinedHw, &combinedPrint, &overlap, printProb, handwritingProb);
    inkanalysis::PenComponentVote(src, combinedHw, combinedPrint, overlap, handwritingProb);
    ModelFinish(combinedHw, combinedPrint, overlap, ink, handwritingProb, printProb, threshold, MinSpeckleArea(src));
    const cv::Mat strikes = StrikeThrough(src, combinedHw, combinedPrint, overlap);
    IsolatedPrint(combinedHw, combinedPrint, overlap, printProb, MinSpeckleArea(src), src);

    if (!WriteMaskFile(combinedHw, combinedPrint, overlap, maskPath, error, strikes)) return nil;
    return Coverage(combinedHw);
}

+ (nullable NSNumber *)applyMaskStrokesAtPath:(NSString *)maskPath
                                    outputPath:(NSString *)outputPath
                                       strokes:(NSArray<NSDictionary<NSString *, id> *> *)strokes
                                         error:(NSError **)error {
    // Only the handwriting layer is edited — the print layer stays, so print under a stroke the
    // user brushes in is still restored on erase.
    cv::Mat handwriting, print, overlap, strike;
    if (!ReadMaskFileOrFail(maskPath, &handwriting, &print, &overlap, error, &strike)) return nil;

    for (NSDictionary<NSString *, id> *stroke in strokes) {
        NSArray<NSNumber *> *points = stroke[@"points"];
        if (![points isKindOfClass:[NSArray class]] || points.count < 2) continue;
        int thickness = std::max(1, (int)std::lround([stroke[@"width"] doubleValue]));
        cv::Scalar value([stroke[@"erase"] boolValue] ? 0 : 255);
        std::vector<cv::Point> polyline;
        for (NSUInteger i = 0; i + 1 < points.count; i += 2) {
            polyline.emplace_back((int)std::lround(points[i].doubleValue), (int)std::lround(points[i + 1].doubleValue));
        }
        bool erase = [stroke[@"erase"] boolValue];
        cv::Mat brush = cv::Mat::zeros(handwriting.size(), CV_8U);
        if (polyline.size() == 1) {
            cv::circle(brush, polyline[0], thickness / 2, cv::Scalar(255), -1);
        } else {
            cv::polylines(brush, polyline, false, cv::Scalar(255), thickness, cv::LINE_8);
        }
        if (erase) {
            handwriting.setTo(0, brush);
            overlap.setTo(0, brush);
        } else {
            // "add to erase" never takes print: a wide brush over handwriting written on print
            // erases the pen and keeps the print underneath
            cv::Mat add = brush & ~print;
            handwriting |= add;
        }
        (void)value;
    }

    if (!WriteMaskFile(handwriting, print, overlap, outputPath, error, strike)) return nil;
    return Coverage(handwriting);
}

/// One erase of `inputPath` with the mask at `maskPath`, written to `outputPath`. `grid`: draw grid
/// paper again here (the multi-pass erase does it once, last).
static bool EraseWithMask(NSString *inputPath, NSString *analysisPath, NSString *maskPath, NSString *outputPath, bool grid,
                          NSError **error) {
    cv::Mat target, analysis, handwriting, print, overlap, strike;
    if (!ReadOrFail(inputPath, &target, error)) return NO;
    if (!ReadOrFail(analysisPath, &analysis, error)) return NO;
    if (!ReadMaskFileOrFail(maskPath, &handwriting, &print, &overlap, error, &strike)) return NO;
    if (analysis.size() != target.size()) {
        cv::resize(analysis, analysis, target.size(), 0, 0, cv::INTER_AREA);
    }
    if (handwriting.size() != target.size()) {
        cv::resize(handwriting, handwriting, target.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(print, print, target.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(overlap, overlap, target.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(strike, strike, target.size(), 0, 0, cv::INTER_NEAREST);
    }
    cv::Mat dst = inkanalysis::EraseHandwriting(target, analysis, handwriting, print, overlap, RecognizeText, strike);
    // give back print / rules the erase took beyond the writing, clear the pen it left
    printrestore::Restore(analysis, target, dst, handwriting, print);
    // rules the handwriting was written on: cleared of pen leftovers and drawn again
    ruleredraw::Apply(analysis, handwriting, print, dst);
    // grid paper the writing was on: its lines drawn again through the erased area
    if (grid) gridredraw::Apply(analysis, dst, handwriting, print);
    return WriteOrFail(dst, outputPath, error);
}

+ (BOOL)eraseWithMaskAtPath:(NSString *)inputPath
                analysisPath:(NSString *)analysisPath
                    maskPath:(NSString *)maskPath
                  outputPath:(NSString *)outputPath
                       error:(NSError **)error {
    return EraseWithMask(inputPath, analysisPath, maskPath, outputPath, true, error);
}

// MARK: - Multi-pass erase (mirror PrintGuard.kt / MultiPassEraser.kt on Android)
//
// One pass can leave pen pieces behind (faint ends, strokes the first mask only half covered) that
// stand out once the rest is gone, so each later pass re-detects on the previous result. That can
// also call print the first pass damaged "handwriting", so the first mask — the one the user
// reviewed, measured on the untouched page's colours — is the reference for what is print.

/// Most erase passes per page (the first included); two — a third changed nothing on 30 labelled pages.
static const int kMaxErasePasses = 2;
/// A later pass runs only if its re-detection, once guarded, still covers this share of the page.
static const double kMinPassCoverage = 0.0002;
/// Clearance kept around any handwriting when restoring print, so a pen's rim never returns.
static const int kGuardRim = 5;
/// How much lighter (8-bit grey) than the clean page a pixel must have become to count as erased.
static const double kGuardErasedDelta = 40;
/// Smallest print piece (8-connected pixels) that is a letter, not a pen rim's stray print pixels.
static const int kGuardMinPiece = 15;

static void FitMask(cv::Mat &m, const cv::Size &size) {
    if (m.size() != size) cv::resize(m, m, size, 0, 0, cv::INTER_NEAREST);
}

/// Drops from the mask at `maskPath` every handwriting pixel `referenceMaskPath` calls pure print,
/// moving it to the print layer; written to `outputPath`. Returns the handwriting coverage.
static NSNumber *GuardMask(NSString *maskPath, NSString *referenceMaskPath, NSString *outputPath, NSError **error) {
    cv::Mat handwriting, print, overlap, strike, refHw, refPrint, refOverlap;
    if (!ReadMaskFileOrFail(maskPath, &handwriting, &print, &overlap, error, &strike)) return nil;
    if (!ReadMaskFileOrFail(referenceMaskPath, &refHw, &refPrint, &refOverlap, error)) return nil;
    FitMask(refHw, handwriting.size());
    FitMask(refPrint, handwriting.size());
    // the mask file keeps overlap out of its print layer: this is print with no pen on it
    cv::Mat protect = refPrint & ~refHw;
    handwriting &= ~protect;
    print |= protect;
    if (!WriteMaskFile(handwriting, print, overlap, outputPath, error, strike)) return nil;
    return Coverage(handwriting);
}

/// Copies back from `cleanPath` (the page before any erase) the pure print of `referenceMaskPath`
/// that the erased page `inputPath` turned to paper, away from the handwriting of every mask in
/// `maskPaths`; written to `outputPath`.
static bool RestorePrint(NSString *inputPath, NSString *cleanPath, NSString *referenceMaskPath, NSArray<NSString *> *maskPaths,
                         NSString *outputPath, NSError **error) {
    cv::Mat out, clean, refHw, refPrint, refOverlap;
    if (!ReadOrFail(inputPath, &out, error)) return false;
    if (!ReadOrFail(cleanPath, &clean, error)) return false;
    if (!ReadMaskFileOrFail(referenceMaskPath, &refHw, &refPrint, &refOverlap, error)) return false;
    if (clean.size() != out.size()) cv::resize(clean, clean, out.size(), 0, 0, cv::INTER_AREA);
    FitMask(refHw, out.size());
    FitMask(refPrint, out.size());
    cv::Mat hwAll = refHw.clone();
    for (NSString *path in maskPaths) {
        cv::Mat hw, pr, ov;
        if (!ReadMaskFileOrFail(path, &hw, &pr, &ov, error)) return false;
        FitMask(hw, out.size());
        hwAll |= hw;
    }
    const cv::Mat hwTouch = hwAll.clone();
    cv::dilate(hwAll, hwAll, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kGuardRim + 1, 2 * kGuardRim + 1)));
    cv::Mat grayOut, grayClean, lighter;
    cv::cvtColor(out, grayOut, cv::COLOR_BGR2GRAY);
    cv::cvtColor(clean, grayClean, cv::COLOR_BGR2GRAY);
    cv::subtract(grayOut, grayClean, lighter);   // saturates at 0 where not lighter
    cv::threshold(lighter, lighter, kGuardErasedDelta, 255, cv::THRESH_BINARY);
    const cv::Mat pure = refPrint & ~refHw;
    // print pieces the size of a letter come back even right beside the pen (a printed "A" inside a
    // pen circle): all of the piece but the 1 px touching the pen
    cv::Mat pieces = cv::Mat::zeros(pure.size(), CV_8U);
    {
        cv::Mat lab, stats, cents;
        const int n = cv::connectedComponentsWithStats(pure, lab, stats, cents, 8, CV_32S);
        for (int y = 0; y < lab.rows; y++) {
            const int *l = lab.ptr<int>(y);
            uchar *o = pieces.ptr<uchar>(y);
            for (int x = 0; x < lab.cols; x++) if (l[x] > 0 && stats.at<int>(l[x], cv::CC_STAT_AREA) >= kGuardMinPiece) o[x] = 255;
        }
        cv::Mat touch;
        cv::dilate(hwTouch, touch, cv::Mat::ones(3, 3, CV_8U));
        pieces &= ~touch;
    }
    cv::Mat restore = ((pure & ~hwAll) | pieces) & lighter;
    clean.copyTo(out, restore);
    // the page as restored so far (PrintGuard.redrawUnderPen reads it after the copy above)
    cv::cvtColor(out, grayOut, cv::COLOR_BGR2GRAY);
    // print remembered under the pen (overlap) that came out as paper: drawn back in the page's
    // print colour (the clean page carries the pen's colour there)
    FitMask(refOverlap, out.size());
    if (cv::countNonZero(pure) >= 50 && cv::countNonZero(refOverlap) > 0) {
        std::vector<uchar> vals;
        for (int y = 0; y < grayClean.rows; y++) {
            const uchar *g = grayClean.ptr<uchar>(y), *p = pure.ptr<uchar>(y);
            for (int x = 0; x < grayClean.cols; x++) if (p[x]) vals.push_back(g[x]);
        }
        std::nth_element(vals.begin(), vals.begin() + vals.size() / 4, vals.end());
        const int core = vals[vals.size() / 4];
        const cv::Scalar colour = cv::mean(clean, (grayClean <= core) & pure);
        const cv::Mat paperNow = (grayOut > core + kGuardErasedDelta) & refOverlap;
        out.setTo(colour, paperNow);
    }
    return WriteOrFail(out, outputPath, error);
}

// MARK: - Speck clean (mirror SpeckClean.kt on Android)
//
// Very last erase step: specks the erase left where the handwriting was — a few dark pixels of a
// stroke's rim, JPEG ringing round it — become paper. Only tiny pieces (up to kSpeckMaxAreaUnits
// stroke units square), only inside the handwriting's zone, and never one touching print.

/// Dev switch (Android: InkHarness `-e speck 0`).
static const bool kSpeckCleanEnabled = true;
static const double kSpeckMaxAreaUnits = 1.0;

/// `dst` the erased page (CV_8UC3), updated in place; `handwriting` / `print` the mask's layers.
static void SpeckClean(cv::Mat &dst, const cv::Mat &handwriting, const cv::Mat &print) {
    if (!kSpeckCleanEnabled) return;
    const int w = dst.cols, h = dst.rows;
    const int k = inkanalysis::StrokeUnit(dst);
    cv::Mat gray, paperG, ink;
    cv::cvtColor(dst, gray, cv::COLOR_BGR2GRAY);
    cv::dilate(gray, paperG, cv::Mat::ones(2 * k + 1, 2 * k + 1, CV_8U));
    cv::medianBlur(paperG, paperG, 21);
    cv::subtract(paperG, gray, ink);   // saturates at 0
    cv::threshold(ink, ink, 30, 255, cv::THRESH_BINARY);
    cv::Mat zone, nearPrint;
    cv::dilate(handwriting, zone, cv::Mat::ones(2 * k + 1, 2 * k + 1, CV_8U));
    cv::dilate(print, nearPrint, cv::Mat::ones(3, 3, CV_8U));
    cv::Mat labels, stats, cents;
    const int n = cv::connectedComponentsWithStats(ink, labels, stats, cents, 8, CV_32S);
    const int maxArea = (int)(kSpeckMaxAreaUnits * k * k);
    std::vector<int> inZone(n, 0);
    std::vector<char> touchPrint(n, 0);
    for (int y = 0; y < h; y++) {
        const int *la = labels.ptr<int>(y);
        const uchar *zo = zone.ptr<uchar>(y), *np = nearPrint.ptr<uchar>(y);
        for (int x = 0; x < w; x++) {
            const int l = la[x];
            if (l == 0) continue;
            if (zo[x]) inZone[l]++;
            if (np[x]) touchPrint[l] = 1;
        }
    }
    std::vector<char> clear(n, 0);
    for (int l = 1; l < n; l++) {
        const int area = stats.at<int>(l, cv::CC_STAT_AREA);
        clear[l] = area <= maxArea && inZone[l] == area && !touchPrint[l];
    }
    cv::Mat paper;
    cv::medianBlur(dst, paper, 21);
    // a speck's own pixels and the 1 px halo round it
    for (int y = 0; y < h; y++) {
        cv::Vec3b *o = dst.ptr<cv::Vec3b>(y);
        const cv::Vec3b *p = paper.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; x++) {
            bool hit = clear[labels.at<int>(y, x)];
            for (int dy = -1; dy <= 1 && !hit; dy++) {
                const int yy = y + dy;
                if (yy < 0 || yy >= h) continue;
                const int *la = labels.ptr<int>(yy);
                for (int dx = -1; dx <= 1; dx++) {
                    const int xx = x + dx;
                    if (xx >= 0 && xx < w && clear[la[xx]]) { hit = true; break; }
                }
            }
            if (hit) o[x] = p[x];
        }
    }
}

/// Last steps of the multi-pass erase on the restored page at `outputPath` (in place), in
/// MultiPassEraser.kt's order: the restore draws print back from the clean page, pen bits stuck to
/// the rules with it, so the rules the handwriting touched are cleared and drawn once more
/// (RuleRedraw); LineHeal (off on Android, `LineHeal.enabled = false`, not ported); specks the
/// erase left become paper (SpeckClean); grid paper the writing was on, drawn again through the
/// erased area once, after every pass (GridRedraw), last.
static bool FinishErase(NSString *outputPath, NSString *analysisPath, NSString *maskPath, NSError **error) {
    cv::Mat dst, analysis, refHw, refPrint, refOverlap;
    if (!ReadOrFail(outputPath, &dst, error)) return false;
    if (!ReadOrFail(analysisPath, &analysis, error)) return false;
    if (!ReadMaskFileOrFail(maskPath, &refHw, &refPrint, &refOverlap, error)) return false;
    if (analysis.size() != dst.size()) cv::resize(analysis, analysis, dst.size(), 0, 0, cv::INTER_AREA);
    FitMask(refHw, dst.size());
    FitMask(refPrint, dst.size());
    ruleredraw::Apply(analysis, refHw, refPrint, dst);
    // LineHeal.apply(dst, hw, print): disabled on Android (enabled = false) — a no-op there
    SpeckClean(dst, refHw, refPrint);
    gridredraw::Apply(analysis, dst, refHw, refPrint);
    return WriteOrFail(dst, outputPath, error);
}

+ (nullable NSNumber *)eraseHandwritingAtPath:(NSString *)inputPath
                                  analysisPath:(NSString *)analysisPath
                                      maskPath:(NSString *)maskPath
                                    outputPath:(NSString *)outputPath
                                      useModel:(BOOL)useModel
                                     threshold:(double)threshold
                                    colorDelta:(double)colorDelta
                                         error:(NSError **)error {
    // a page larger than the working size is erased at that size, the result laid back over it
    {
        cv::Mat full;
        if (!ReadOrFail(inputPath, &full, error)) return nil;
        const double scale = WorkingScale(full.size());
        if (scale < 1.0) {
            NSString *work = NewWorkDir();
            NSString *(^file)(NSString *) = ^NSString *(NSString *name) { return [work stringByAppendingPathComponent:name]; };
            NSNumber *result = nil;
            do {
                const cv::Size size(cvRound(full.cols * scale), cvRound(full.rows * scale));
                cv::Mat smallIn, analysis, smallOut;
                cv::resize(full, smallIn, size, 0, 0, cv::INTER_AREA);
                if (!WriteOrFail(smallIn, file(@"page.png"), error)) break;
                if (!ReadOrFail(analysisPath, &analysis, error)) break;
                if (!WriteScaled(analysis, size, file(@"analysis.png"), error)) break;
                if (!ResizeMaskFile(maskPath, file(@"mask.png"), size, error)) break;
                NSNumber *passes = [self eraseHandwritingAtPath:file(@"page.png") analysisPath:file(@"analysis.png")
                                                       maskPath:file(@"mask.png") outputPath:file(@"erased.png")
                                                       useModel:useModel threshold:threshold colorDelta:colorDelta error:error];
                if (passes == nil) break;
                if (!ReadOrFail(file(@"erased.png"), &smallOut, error)) break;
                if (!CompositeErased(full, smallIn, smallOut, outputPath, error)) break;
                result = passes;
            } while (false);
            [[NSFileManager defaultManager] removeItemAtPath:work error:nil];
            return result;
        }
    }
    NSString *work = NewWorkDir();
    NSString *(^file)(NSString *) = ^NSString *(NSString *name) { return [work stringByAppendingPathComponent:name]; };
    NSNumber *result = nil;
    do {
        NSString *erased = file(@"erased_1.png");
        if (!EraseWithMask(inputPath, analysisPath, maskPath, erased, false, error)) break;
        NSMutableArray<NSString *> *laterMasks = [NSMutableArray array];
        int passes = 1;
        bool failed = false;
        for (int pass = 2; pass <= kMaxErasePasses; pass++) {
            NSString *found = file([NSString stringWithFormat:@"found_%d.png", pass]);
            NSNumber *coverage = useModel
                ? [self segmentationMaskAtPath:erased maskPath:found threshold:threshold colorDelta:colorDelta error:error]
                : [self inkColorMaskAtPath:erased maskPath:found colorDelta:colorDelta error:error];
            if (coverage == nil) { failed = true; break; }
            if (coverage.doubleValue < kMinPassCoverage) break;
            NSString *guarded = file([NSString stringWithFormat:@"mask_%d.png", pass]);
            NSNumber *guardedCoverage = GuardMask(found, maskPath, guarded, error);
            if (guardedCoverage == nil) { failed = true; break; }
            if (guardedCoverage.doubleValue < kMinPassCoverage) break;
            NSString *next = file([NSString stringWithFormat:@"erased_%d.png", pass]);
            if (!EraseWithMask(erased, erased, guarded, next, false, error)) { failed = true; break; }
            erased = next;
            [laterMasks addObject:guarded];
            passes = pass;
        }
        if (failed) break;
        if (!RestorePrint(erased, inputPath, maskPath, laterMasks, outputPath, error)) break;
        if (!FinishErase(outputPath, analysisPath, maskPath, error)) break;
        result = @(passes);
    } while (false);
    [[NSFileManager defaultManager] removeItemAtPath:work error:nil];
    return result;
}

@end
