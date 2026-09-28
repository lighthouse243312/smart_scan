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
#import "ImageProcessingOpenCV.h"

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
// "Ink" = noticeably darker than the local paper. The strict value seeds the colour method; the
// loose one lets masks grow onto anti-aliased stroke edges and faint pencil without spilling
// onto bare paper.
static const int kInkContrastStrict = 18;
static const int kInkContrastLoose = 8;
// Colour method only: inside a coloured stroke, a pixel this dark is the stroke crossing black
// print (a multiply of both inks), so it is counted as print to restore.
static const int kPrintUnderMaxValue = 100;

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

/// `handwriting` / `print`: CV_8UC1, 255 = set.
static bool WriteMaskFile(const cv::Mat &handwriting, const cv::Mat &print, NSString *path, NSError **error) {
    cv::Mat file(handwriting.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
    std::vector<cv::Mat> channels;
    cv::split(file, channels);
    channels[0] = print.clone();                                   // B: print
    channels[2].setTo(255, handwriting);                           // R: overlay colour
    channels[3].setTo(kOverlayAlpha, handwriting);                 // A: handwriting
    cv::merge(channels, file);
    return WriteOrFail(file, path, error);
}

static bool ReadMaskFileOrFail(NSString *path, cv::Mat *outHandwriting, cv::Mat *outPrint, NSError **error) {
    cv::Mat raw = cv::imread([path UTF8String], cv::IMREAD_UNCHANGED);
    if (raw.empty() || raw.channels() != 4) {
        *error = MakeError(ImageProcessingErrorFileNotFound,
                            [NSString stringWithFormat:@"Không đọc được mask tại: %@", path]);
        return false;
    }
    cv::Mat alpha, blue;
    cv::extractChannel(raw, alpha, 3);
    cv::extractChannel(raw, blue, 0);
    cv::threshold(alpha, *outHandwriting, 0, 255, cv::THRESH_BINARY);
    cv::threshold(blue, *outPrint, 127, 255, cv::THRESH_BINARY);
    return true;
}

static NSNumber *Coverage(const cv::Mat &mask) {
    return @((double)cv::countNonZero(mask) / (double)std::max<size_t>(1, mask.total()));
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

/// Per-pixel paper colour: a large median over a 4x-downscaled copy wipes out ink strokes.
static cv::Mat EstimatePaperColor(const cv::Mat &src) {
    cv::Mat small;
    cv::resize(src, small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    int kernel = std::min(21, (std::min(small.cols, small.rows) - 1) | 1);
    if (kernel >= 3) cv::medianBlur(small, small, kernel);
    cv::Mat paper;
    cv::resize(small, paper, src.size(), 0, 0, cv::INTER_LINEAR);
    return paper;
}

/// Per-pixel colour of the NEARBY print (average over print pixels not covered by handwriting,
/// within roughly a text line's reach), falling back to the page-wide print average where there
/// is none — so a stroke crossing a blue heading restores blue, not black.
static cv::Mat EstimatePrintColor(const cv::Mat &src, const cv::Mat &visiblePrint) {
    // Only the dark core of print strokes: their anti-aliased edges are much lighter and would
    // wash the restored colour out to grey.
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat contrast = InkContrast(gray);
    int histogram[256] = {0};
    int printCount = 0;
    for (int y = 0; y < contrast.rows; y++) {
        const uchar *c = contrast.ptr<uchar>(y);
        const uchar *m = visiblePrint.ptr<uchar>(y);
        for (int x = 0; x < contrast.cols; x++) {
            if (m[x]) { histogram[c[x]]++; printCount++; }
        }
    }
    int p95 = 0;
    for (int v = 0, seen = 0; v < 256; v++) {
        seen += histogram[v];
        if (seen >= printCount * 0.95) { p95 = v; break; }
    }
    cv::Mat core;
    cv::compare(contrast, 0.85 * p95, core, cv::CMP_GE);
    cv::bitwise_and(core, visiblePrint, core);

    cv::Mat weight, srcFloat;
    core.convertTo(weight, CV_32F, 1.0 / 255.0);
    src.convertTo(srcFloat, CV_32FC3);
    cv::Mat weighted;
    cv::Mat weight3;
    cv::merge(std::vector<cv::Mat>{weight, weight, weight}, weight3);
    cv::multiply(srcFloat, weight3, weighted);

    const double f = 0.125;
    cv::Mat smallWeighted, smallWeight;
    cv::resize(weighted, smallWeighted, cv::Size(), f, f, cv::INTER_AREA);
    cv::resize(weight, smallWeight, cv::Size(), f, f, cv::INTER_AREA);
    cv::Size window(15, 15);
    cv::boxFilter(smallWeighted, smallWeighted, -1, window);
    cv::boxFilter(smallWeight, smallWeight, -1, window);

    double total = cv::sum(weight)[0];
    cv::Scalar globalColor = total > 0 ? cv::sum(weighted) / total : cv::Scalar(30, 30, 30);

    cv::Mat smallColor(smallWeight.size(), CV_32FC3);
    for (int y = 0; y < smallWeight.rows; y++) {
        const float *w = smallWeight.ptr<float>(y);
        const cv::Vec3f *c = smallWeighted.ptr<cv::Vec3f>(y);
        cv::Vec3f *out = smallColor.ptr<cv::Vec3f>(y);
        for (int x = 0; x < smallWeight.cols; x++) {
            out[x] = w[x] > 1e-3f ? c[x] / w[x]
                                  : cv::Vec3f((float)globalColor[0], (float)globalColor[1], (float)globalColor[2]);
        }
    }
    cv::Mat color;
    cv::resize(smallColor, color, src.size(), 0, 0, cv::INTER_LINEAR);
    color.convertTo(color, CV_8UC3);
    return color;
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
                             minSaturation:(double)minSaturation
                                     error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(inputPath, &src, error)) return nil;

    cv::Mat gray, hsv;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    cv::Mat saturation, value;
    cv::extractChannel(hsv, saturation, 1);
    cv::extractChannel(hsv, value, 2);
    cv::Mat contrast = InkContrast(gray);

    // Seed: saturated pixels that are also real ink (not a tinted paper area).
    cv::Mat colored, inkStrict, inkLoose;
    cv::compare(saturation, minSaturation, colored, cv::CMP_GT);
    cv::compare(contrast, kInkContrastStrict, inkStrict, cv::CMP_GT);
    cv::compare(contrast, kInkContrastLoose, inkLoose, cv::CMP_GT);
    cv::bitwise_and(colored, inkStrict, colored);
    RemoveSmallComponents(colored, MinSpeckleArea(src));

    // A stroke's anti-aliased rim is less saturated than its core, so grow the seed a little —
    // but only onto pixels that are ink, never onto paper.
    cv::Mat grown;
    cv::dilate(colored, grown, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)), cv::Point(-1, -1), 2);
    cv::bitwise_and(grown, inkLoose, grown);
    cv::Mat handwriting;
    cv::bitwise_or(colored, grown, handwriting);

    // Print layer (no model here, so estimated): unsaturated ink OUTSIDE the strokes, plus pixels
    // inside a stroke dark enough to be that stroke crossing black print. A stroke's own
    // anti-aliased rim is unsaturated too, so unsaturated ink inside the stroke must not count —
    // it would be "restored" as a grey outline of the erased stroke.
    cv::Mat unsaturated, print, printUnder, notHandwriting;
    cv::compare(saturation, minSaturation, unsaturated, cv::CMP_LE);
    cv::bitwise_not(handwriting, notHandwriting);
    cv::bitwise_and(inkStrict, unsaturated, print);
    cv::bitwise_and(print, notHandwriting, print);
    cv::compare(value, kPrintUnderMaxValue, printUnder, cv::CMP_LT);
    cv::bitwise_and(printUnder, handwriting, printUnder);
    cv::bitwise_or(print, printUnder, print);

    if (!WriteMaskFile(handwriting, print, maskPath, error)) return nil;
    return Coverage(handwriting);
}

+ (nullable NSNumber *)segmentationMaskAtPath:(NSString *)inputPath
                                      maskPath:(NSString *)maskPath
                                     threshold:(double)threshold
                                         error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(inputPath, &src, error)) return nil;

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

    if (!WriteMaskFile(handwriting, print, maskPath, error)) return nil;
    return Coverage(handwriting);
}

+ (nullable NSNumber *)applyMaskStrokesAtPath:(NSString *)maskPath
                                    outputPath:(NSString *)outputPath
                                       strokes:(NSArray<NSDictionary<NSString *, id> *> *)strokes
                                         error:(NSError **)error {
    // Only the handwriting layer is edited — the print layer stays, so print under a stroke the
    // user brushes in is still restored on erase.
    cv::Mat handwriting, print;
    if (!ReadMaskFileOrFail(maskPath, &handwriting, &print, error)) return nil;

    for (NSDictionary<NSString *, id> *stroke in strokes) {
        NSArray<NSNumber *> *points = stroke[@"points"];
        if (![points isKindOfClass:[NSArray class]] || points.count < 2) continue;
        int thickness = std::max(1, (int)std::lround([stroke[@"width"] doubleValue]));
        cv::Scalar value([stroke[@"erase"] boolValue] ? 0 : 255);
        std::vector<cv::Point> polyline;
        for (NSUInteger i = 0; i + 1 < points.count; i += 2) {
            polyline.emplace_back((int)std::lround(points[i].doubleValue), (int)std::lround(points[i + 1].doubleValue));
        }
        if (polyline.size() == 1) {
            cv::circle(handwriting, polyline[0], thickness / 2, value, -1);
        } else {
            cv::polylines(handwriting, polyline, false, value, thickness, cv::LINE_8);
        }
    }

    if (!WriteMaskFile(handwriting, print, outputPath, error)) return nil;
    return Coverage(handwriting);
}

+ (BOOL)eraseWithMaskAtPath:(NSString *)inputPath
                    maskPath:(NSString *)maskPath
                  outputPath:(NSString *)outputPath
                    dilatePx:(double)dilatePx
                       error:(NSError **)error {
    cv::Mat src, handwriting, print;
    if (!ReadOrFail(inputPath, &src, error)) return NO;
    if (!ReadMaskFileOrFail(maskPath, &handwriting, &print, error)) return NO;
    if (handwriting.size() != src.size()) {
        cv::resize(handwriting, handwriting, src.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(print, print, src.size(), 0, 0, cv::INTER_NEAREST);
    }

    // Grow over the stroke's faint rim — but only onto non-print pixels, so the growth can never
    // eat into print the stroke merely passes next to.
    cv::Mat area = handwriting.clone();
    int grow = (int)std::lround(dilatePx);
    if (grow > 0) {
        cv::Mat grown, notPrint;
        cv::dilate(handwriting, grown, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * grow + 1, 2 * grow + 1)));
        cv::bitwise_not(print, notPrint);
        cv::bitwise_and(grown, notPrint, grown);
        cv::bitwise_or(area, grown, area);
    }

    // Rebuild instead of inpainting: inside the erased area, print pixels get the colour of the
    // surrounding print (so a pen stroke crossing a word leaves the word's strokes intact) and
    // everything else gets the paper colour. Outside the area the page is untouched.
    cv::Mat restorePrint, paperArea, visiblePrint, notArea;
    cv::bitwise_and(area, print, restorePrint);
    cv::bitwise_xor(area, restorePrint, paperArea);
    cv::bitwise_not(area, notArea);
    cv::bitwise_and(print, notArea, visiblePrint);

    cv::Mat dst = src.clone();
    EstimatePaperColor(src).copyTo(dst, paperArea);
    EstimatePrintColor(src, visiblePrint).copyTo(dst, restorePrint);
    return WriteOrFail(dst, outputPath, error);
}

@end
