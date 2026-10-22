// OpenCV headers must be imported before Foundation/the class's own header — OpenCV's C++
// headers conflict with Objective-C globals (e.g. `NO`) when included afterwards.
// Using OpenCV 4.10.0 here (not 5.0.0, matching Android) — the official 5.0.0 iOS framework
// ships a DNN module that references ONNX Runtime/MLAS symbols missing for the device arm64
// slice, so any app linking it fails with "Undefined symbol: MlasGemmBatch" etc. on a real
// device. 4.10.0 predates that DNN/ONNX backend and has no such issue.
#import <opencv2/opencv.hpp>

#import <algorithm>
#import <cmath>
#import <vector>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import "ImageProcessingOpenCV.h"
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

static bool WriteMaskFile(const cv::Mat &handwriting, const cv::Mat &print, const cv::Mat &overlap, NSString *path, NSError **error) {
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
    cv::merge(channels, file);
    return WriteOrFail(file, path, error);
}

static bool ReadMaskFileOrFail(NSString *path, cv::Mat *outHandwriting, cv::Mat *outPrint, cv::Mat *outOverlap, NSError **error) {
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
// Pen strike-throughs across printed words. A black pen line has the print's colour, so the colour
// layers call it print and it was never erased. It is found by shape and place instead: a long
// (6 stroke units), thin, horizontal run of ink with print right above AND right below it along
// most of its length — it runs through the middle of the letters (an underline has print only above
// it, a table rule none close). The line becomes handwriting, erased whole; where a printed letter
// crosses it (print right at its top and bottom edge) the pixels are remembered as print under the
// pen. The letters it crossed are then redrawn from the page's own glyphs (RetypesetPrint).
static void StrikeThrough(const cv::Mat &src, cv::Mat &handwriting, cv::Mat &print, cv::Mat &overlap) {
    const int W = src.cols, H = src.rows;
    const int k = inkanalysis::StrokeUnit(src);
    cv::Mat ink;
    {
        cv::Mat g, bg;
        cv::cvtColor(src, g, cv::COLOR_BGR2GRAY);
        cv::morphologyEx(g, bg, cv::MORPH_CLOSE, cv::Mat::ones(25, 25, CV_8U));
        ink = (bg - g) > 40;
    }
    cv::Mat run;
    cv::morphologyEx(ink, run, cv::MORPH_OPEN, cv::Mat::ones(1, 3 * k, CV_8U));
    cv::Mat labels, stats, cents;
    const int n = cv::connectedComponentsWithStats(run, labels, stats, cents, 8, CV_32S);
    const cv::Mat pure = (print > 127) & (handwriting == 0);
    const int reach = (int)(1.5 * k);
    auto pureIn = [&](int c, int y0, int y1) {
        for (int y = std::max(0, y0); y <= std::min(H - 1, y1); y++) if (pure.at<uchar>(y, c)) return true;
        return false;
    };
    for (int l = 1; l < n; l++) {
        const int x0 = stats.at<int>(l, cv::CC_STAT_LEFT), y0 = stats.at<int>(l, cv::CC_STAT_TOP);
        const int bw = stats.at<int>(l, cv::CC_STAT_WIDTH), bh = stats.at<int>(l, cv::CC_STAT_HEIGHT);
        if (bw < 6 * k || bh > 2 * k) continue;
        std::vector<int> top(bw, INT_MAX), bot(bw, -1);
        for (int y = y0; y < y0 + bh; y++) {
            const int *lp = labels.ptr<int>(y);
            for (int x = x0; x < x0 + bw; x++) if (lp[x] == l) { const int c = x - x0; top[c] = std::min(top[c], y); bot[c] = std::max(bot[c], y); }
        }
        std::vector<int> ext; int up = 0, dn = 0, cols = 0;
        for (int c = 0; c < bw; c++) {
            if (bot[c] < 0) continue;
            cols++; ext.push_back(bot[c] - top[c] + 1);
            if (pureIn(x0 + c, top[c] - reach, top[c] - 2)) up++;
            if (pureIn(x0 + c, bot[c] + 2, bot[c] + reach)) dn++;
        }
        if (!cols) continue;
        std::nth_element(ext.begin(), ext.begin() + ext.size() / 2, ext.end());
        if (ext[ext.size() / 2] > 0.6 * k) continue;                 // thin, column by column
        if (up < 0.4 * cols || dn < 0.4 * cols) continue;            // through the letters
        for (int c = 0; c < bw; c++) {
            if (bot[c] < 0) continue;
            const int x = x0 + c;
            const bool crossed = pureIn(x, top[c] - 2, top[c] - 1) && pureIn(x, bot[c] + 1, bot[c] + 2);
            for (int y = top[c]; y <= bot[c]; y++) {
                if (labels.at<int>(y, x) != l) continue;
                handwriting.at<uchar>(y, x) = 255;
                overlap.at<uchar>(y, x) = crossed ? 255 : 0;
                print.at<uchar>(y, x) = crossed ? 255 : 0;
            }
        }
    }
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
    // StrikeThrough(src, combinedHw, combinedPrint, overlap);   // off, see StrikeThrough.kt
    IsolatedPrint(combinedHw, combinedPrint, overlap, printProb, MinSpeckleArea(src), src);

    if (!WriteMaskFile(combinedHw, combinedPrint, overlap, maskPath, error)) return nil;
    return Coverage(combinedHw);
}

+ (nullable NSNumber *)applyMaskStrokesAtPath:(NSString *)maskPath
                                    outputPath:(NSString *)outputPath
                                       strokes:(NSArray<NSDictionary<NSString *, id> *> *)strokes
                                         error:(NSError **)error {
    // Only the handwriting layer is edited — the print layer stays, so print under a stroke the
    // user brushes in is still restored on erase.
    cv::Mat handwriting, print, overlap;
    if (!ReadMaskFileOrFail(maskPath, &handwriting, &print, &overlap, error)) return nil;

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

    if (!WriteMaskFile(handwriting, print, overlap, outputPath, error)) return nil;
    return Coverage(handwriting);
}

+ (BOOL)eraseWithMaskAtPath:(NSString *)inputPath
                analysisPath:(NSString *)analysisPath
                    maskPath:(NSString *)maskPath
                  outputPath:(NSString *)outputPath
                       error:(NSError **)error {
    cv::Mat target, analysis, handwriting, print, overlap;
    if (!ReadOrFail(inputPath, &target, error)) return NO;
    if (!ReadOrFail(analysisPath, &analysis, error)) return NO;
    if (!ReadMaskFileOrFail(maskPath, &handwriting, &print, &overlap, error)) return NO;
    if (analysis.size() != target.size()) {
        cv::resize(analysis, analysis, target.size(), 0, 0, cv::INTER_AREA);
    }
    if (handwriting.size() != target.size()) {
        cv::resize(handwriting, handwriting, target.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(print, print, target.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(overlap, overlap, target.size(), 0, 0, cv::INTER_NEAREST);
    }
    cv::Mat dst = inkanalysis::EraseHandwriting(target, analysis, handwriting, print, overlap, RecognizeText);
    // give back print / rules the erase took beyond the writing, clear the pen it left
    printrestore::Restore(analysis, target, dst, handwriting, print);
    // rules the handwriting was written on: cleared of pen leftovers and drawn again
    ruleredraw::Apply(analysis, handwriting, print, dst);
    return WriteOrFail(dst, outputPath, error);
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
    cv::Mat handwriting, print, overlap, refHw, refPrint, refOverlap;
    if (!ReadMaskFileOrFail(maskPath, &handwriting, &print, &overlap, error)) return nil;
    if (!ReadMaskFileOrFail(referenceMaskPath, &refHw, &refPrint, &refOverlap, error)) return nil;
    FitMask(refHw, handwriting.size());
    FitMask(refPrint, handwriting.size());
    // the mask file keeps overlap out of its print layer: this is print with no pen on it
    cv::Mat protect = refPrint & ~refHw;
    handwriting &= ~protect;
    print |= protect;
    if (!WriteMaskFile(handwriting, print, overlap, outputPath, error)) return nil;
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
        if (![self eraseWithMaskAtPath:inputPath analysisPath:analysisPath maskPath:maskPath outputPath:erased error:error]) break;
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
            if (![self eraseWithMaskAtPath:erased analysisPath:erased maskPath:guarded outputPath:next error:error]) { failed = true; break; }
            erased = next;
            [laterMasks addObject:guarded];
            passes = pass;
        }
        if (failed) break;
        if (!RestorePrint(erased, inputPath, maskPath, laterMasks, outputPath, error)) break;
        result = @(passes);
    } while (false);
    [[NSFileManager defaultManager] removeItemAtPath:work error:nil];
    return result;
}

@end
