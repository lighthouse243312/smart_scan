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
static cv::Mat PrintThroughPen(const cv::Mat &overlap, const cv::Mat &purePrint, int k) {
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
    return ((left & right) | (up & down) | rule) & overlap;
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
    channels[1] = PrintThroughPen(overlap, channels[0], inkanalysis::StrokeUnit(handwriting));   // G: overlap
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
    // Combined with the ink-colour + layout method: each covers the other's blind spot — colour
    // finds neat pen writing the model (trained on synthetic data) misses; the model finds pen ink
    // the same colour as the print (black ballpoint), which colour cannot see.
    cv::Mat colorHw, colorPrint, colorOverlap;
    inkanalysis::DetectByInkColor(src, colorDelta, &colorHw, &colorPrint, &colorOverlap, printProb, handwritingProb);
    // a model pixel is dropped only where BOTH methods agree it is print
    cv::Mat combinedHw = colorHw | (handwriting & ~(colorPrint & print));
    // model's two layers are independent: where both are set, print lies under the pen
    cv::Mat overlap = colorOverlap | (combinedHw & print & ~colorHw);
    cv::Mat combinedPrint = ((colorPrint | print) & ~combinedHw) | overlap;
    inkanalysis::PenComponentVote(src, combinedHw, combinedPrint, overlap, handwritingProb);
    StrokeVote(combinedHw, combinedPrint, overlap, ink, handwritingProb, printProb, threshold);

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

/// Most erase passes per page (the first included).
static const int kMaxErasePasses = 3;
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
    NSString *work = [NSTemporaryDirectory() stringByAppendingPathComponent:[[NSUUID UUID] UUIDString]];
    [[NSFileManager defaultManager] createDirectoryAtPath:work withIntermediateDirectories:YES attributes:nil error:nil];
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
