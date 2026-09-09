// OpenCV headers must be imported before Foundation/the class's own header — OpenCV's C++
// headers conflict with Objective-C globals (e.g. `NO`) when included afterwards.
// Using OpenCV 4.10.0 here (not 5.0.0, matching Android) — the official 5.0.0 iOS framework
// ships a DNN module that references ONNX Runtime/MLAS symbols missing for the device arm64
// slice, so any app linking it fails with "Undefined symbol: MlasGemmBatch" etc. on a real
// device. 4.10.0 predates that DNN/ONNX backend and has no such issue. The algorithms used
// here (GaussianBlur, threshold, distanceTransform, findContours, contourArea, arcLength,
// inpaint, CLAHE...) are stable, unchanged APIs across 4.x/5.x.
#import <opencv2/opencv.hpp>

#import <set>

#import <Foundation/Foundation.h>
#import "ImageProcessingOpenCV.h"

NSString *const ImageProcessingErrorDomain = @"ImageProcessingOpenCV";

static NSError *MakeError(ImageProcessingErrorCode code, NSString *message) {
    return [NSError errorWithDomain:ImageProcessingErrorDomain
                                code:code
                            userInfo:@{NSLocalizedDescriptionKey : message}];
}

static double NumberOrThrow(NSDictionary<NSString *, id> *map, NSString *key, BOOL *ok, NSError **error) {
    id value = map[key];
    if (![value isKindOfClass:[NSNumber class]]) {
        *ok = NO;
        *error = MakeError(ImageProcessingErrorInvalidArgument,
                            [NSString stringWithFormat:@"Missing numeric field: %@", key]);
        return 0.0;
    }
    *ok = YES;
    return [(NSNumber *)value doubleValue];
}

/// Converts a Dart `{left, top, right, bottom}` map into a cv::Rect clipped to the image bounds.
static BOOL MapToClippedRect(NSDictionary<NSString *, id> *map, int imageWidth, int imageHeight,
                              double paddingPx, cv::Rect *outRect, NSError **error) {
    BOOL ok = NO;
    double left = NumberOrThrow(map, @"left", &ok, error) - paddingPx;
    if (!ok) return NO;
    double top = NumberOrThrow(map, @"top", &ok, error) - paddingPx;
    if (!ok) return NO;
    double right = NumberOrThrow(map, @"right", &ok, error) + paddingPx;
    if (!ok) return NO;
    double bottom = NumberOrThrow(map, @"bottom", &ok, error) + paddingPx;
    if (!ok) return NO;

    int clippedLeft = (int)MAX(0.0, left);
    int clippedTop = (int)MAX(0.0, top);
    int clippedRight = (int)MIN((double)imageWidth, right);
    int clippedBottom = (int)MIN((double)imageHeight, bottom);

    int width = MAX(1, clippedRight - clippedLeft);
    int height = MAX(1, clippedBottom - clippedTop);
    *outRect = cv::Rect(clippedLeft, clippedTop, width, height);
    return YES;
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

/// Raw per-word measurements — mirrors HandwritingDetector.kt's RegionStats on Android. This
/// does NOT decide handwriting vs print itself; Dart (ImageProcessingService) makes that call
/// by comparing every word's stats against the PAGE'S OWN most-common values (a self-calibrating
/// "reference style/color" instead of a fixed threshold).
struct RegionStats {
    double strokeVariationScore = 0.0;
    double componentRatioScore = 0.0;
    double angleVariationScore = 0.0;
    double avgStrokeWidth = 0.0;
    double inkColorB = 0.0;
    double inkColorG = 0.0;
    double inkColorR = 0.0;
    double inkIntensityStdDev = 0.0;
    bool hasWideUnderline = false;
    bool hasInk = false;
};

/**
 * A fill-in-the-blank answer is written ON TOP of a pre-printed blank line — the ink usually
 * touches or overlaps it, not sitting cleanly above it with a gap — and that line is wider than
 * the answer itself (the blank was sized for a guessed-longer answer). A printed word's own
 * underline (used for in-text emphasis) hugs the word tightly instead. So: search a band spanning
 * from partway UP INSIDE the word's own box down through a generous margin below it (covering
 * both "line touches the ink" and "line has a small gap"), and look for a long, near-solid dark
 * horizontal run spanning noticeably wider than the word's own box. A fixed darkness threshold is
 * used instead of a fresh Otsu computation — Otsu on a thin, almost-entirely-blank strip (a few
 * dark line pixels among mostly paper) is not a reliable split.
 */
static bool HasWideUnderlineBelow(const cv::Mat &gray, const cv::Rect &rect) {
    const int kDarkPixelThreshold = 150;

    int marginX = std::max(4, (int)(rect.width * 0.6));
    int bandLeft = std::max(0, rect.x - marginX);
    int bandRight = std::min(gray.cols, rect.x + rect.width + marginX);
    int bandWidth = bandRight - bandLeft;
    if (bandWidth <= 0) return false;

    int bandTop = std::min(gray.rows - 1, rect.y + (int)(rect.height * 0.6));
    int bandBottom = std::min(gray.rows, rect.y + (int)(rect.height * 1.6));
    if (bandBottom <= bandTop) return false;

    cv::Mat band = gray(cv::Rect(bandLeft, bandTop, bandWidth, bandBottom - bandTop));

    int longestRun = 0;
    for (int y = 0; y < band.rows; y++) {
        const uchar *row = band.ptr<uchar>(y);
        int currentRun = 0;
        for (int x = 0; x < band.cols; x++) {
            if (row[x] < kDarkPixelThreshold) {
                currentRun++;
                if (currentRun > longestRun) longestRun = currentRun;
            } else {
                currentRun = 0;
            }
        }
    }
    return longestRun > rect.width * 1.2;
}

/**
 * How much each letter's own tilt varies from the next, WITHIN this one word — a signal
 * intrinsic to the ink shape itself, independent of position/underline/color, so it still fires
 * on handwriting that isn't sitting on a fill-in-blank line. A printed font renders the exact
 * same glyph outline every time a letter repeats, so every stroke sits at the same angle across
 * the whole word; a human hand never repeats a stroke at a perfectly identical angle twice.
 *
 * Deliberately NOT filtered to "tall" strokes only: the fill-in-blank answers on a real
 * worksheet are mostly short 2-4 letter words ("he", "it", "us", "they"...), which often have
 * zero or one component tall enough to pass a height filter — that filter silently starved this
 * signal on exactly the majority case. Using every component with a minimum area instead means
 * even a two-letter word usually has enough data points.
 */
static double ComponentAngleVariationScore(const cv::Mat &binaryInk) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryInk, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<double> angleDeviations;
    for (auto &contour : contours) {
        if (contour.size() >= 5 && cv::contourArea(contour) >= 3) {
            double angle = cv::minAreaRect(contour).angle;
            // minAreaRect's angle is ambiguous mod 90° (which side is "width" flips it); fold
            // into "deviation from the nearest axis" in [0, 45] so that's comparable.
            double mod90 = std::fmod(std::fmod(angle, 90.0) + 90.0, 90.0);
            angleDeviations.push_back(std::min(mod90, 90.0 - mod90));
        }
    }
    if (angleDeviations.size() < 2) return 0.0;

    double sum = 0.0;
    for (double a : angleDeviations) sum += a;
    double mean = sum / angleDeviations.size();
    double variance = 0.0;
    for (double a : angleDeviations) variance += (a - mean) * (a - mean);
    variance /= angleDeviations.size();
    double stdDev = std::sqrt(variance);
    // Degrees; not tuned against a labeled dataset yet.
    return std::min(1.0, std::max(0.0, stdDev / 12.0));
}

/// Stroke Width Transform-style measurement (Epshtein et al.) — printed fonts render every
/// character with the same stroke width by design; handwriting's stroke width drifts with pen
/// pressure/speed. The key point: aggregate stroke width PER CONNECTED COMPONENT (per character)
/// first, then compare variance ACROSS components — pooling every ink pixel together (an earlier
/// version of this function did that) mixes in each letter's own thick joints/corners and drowns
/// out the real signal; verified empirically, that approach scored a clean printed line *higher*
/// than actual handwriting.
static RegionStats ScoreRegion(const cv::Mat &color, const cv::Mat &gray, const cv::Rect &rect, int charCount) {
    cv::Mat colorCrop = color(rect);
    cv::Mat crop = gray(rect);
    cv::Mat binary;
    cv::threshold(crop, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    RegionStats result;
    if (cv::countNonZero(binary) < 20) {
        return result; // too little ink in this box to say anything meaningful (hasInk stays false)
    }

    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    cv::Mat distance;
    cv::distanceTransform(binary, distance, cv::DIST_L2, 3);

    std::vector<double> sumByLabel(numLabels, 0.0);
    std::vector<int> countByLabel(numLabels, 0);
    for (int y = 0; y < binary.rows; y++) {
        const int32_t *labelRow = labels.ptr<int32_t>(y);
        const float *distRow = distance.ptr<float>(y);
        for (int x = 0; x < binary.cols; x++) {
            int label = labelRow[x];
            if (label != 0) { // 0 == background
                sumByLabel[label] += distRow[x];
                countByLabel[label]++;
            }
        }
    }

    const int kMinComponentArea = 3;
    std::vector<double> strokeWidthsPerComponent;
    int componentCount = 0;
    for (int label = 1; label < numLabels; label++) {
        if (countByLabel[label] >= kMinComponentArea) {
            componentCount++;
            strokeWidthsPerComponent.push_back(2.0 * sumByLabel[label] / countByLabel[label]);
        }
    }

    double strokeScore = 0.0;
    if (strokeWidthsPerComponent.size() >= 2) {
        double sum = 0.0;
        for (double w : strokeWidthsPerComponent) sum += w;
        double mean = sum / strokeWidthsPerComponent.size();
        double variance = 0.0;
        for (double w : strokeWidthsPerComponent) variance += (w - mean) * (w - mean);
        variance /= strokeWidthsPerComponent.size();
        double stdDev = std::sqrt(variance);
        double variationRatio = stdDev / (mean + 1e-6);
        strokeScore = std::min(1.0, std::max(0.0, variationRatio / 0.5));
    }

    // Fewer components than characters (letters visually joined) suggests cursive handwriting.
    double safeCharCount = std::max(1, charCount);
    double ratio = componentCount / safeCharCount;
    double ratioScore = std::min(1.0, std::max(0.0, 1.0 - ratio));

    double angleScore = ComponentAngleVariationScore(binary);

    double avgStrokeWidth = 0.0;
    if (!strokeWidthsPerComponent.empty()) {
        double sum = 0.0;
        for (double w : strokeWidthsPerComponent) sum += w;
        avgStrokeWidth = sum / strokeWidthsPerComponent.size();
    }

    // Mean BGR of the ink pixels (the actual glyph strokes, not the paper background).
    cv::Scalar inkColor = cv::mean(colorCrop, binary);

    // How much pixel darkness varies within the ink itself. Printed toner/ink lays down at a
    // near-uniform density, so its ink pixels cluster tightly around one dark value; pen ink
    // varies with pressure/speed/flow (skips, fades, presses darker), spreading that value out.
    cv::Scalar intensityMean, intensityStdDev;
    cv::meanStdDev(crop, intensityMean, intensityStdDev, binary);

    result.strokeVariationScore = strokeScore;
    result.componentRatioScore = ratioScore;
    result.angleVariationScore = angleScore;
    result.avgStrokeWidth = avgStrokeWidth;
    result.inkColorB = inkColor[0];
    result.inkColorG = inkColor[1];
    result.inkColorR = inkColor[2];
    result.inkIntensityStdDev = intensityStdDev[0];
    result.hasWideUnderline = HasWideUnderlineBelow(gray, rect);
    result.hasInk = true;
    return result;
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

+ (nullable NSArray<NSDictionary<NSString *, id> *> *)detectHandwritingRegionsAtPath:(NSString *)imagePath
                                                                           textBlocks:(NSArray<NSDictionary<NSString *, id> *> *)textBlocks
                                                                                error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(imagePath, &src, error)) return nil;

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    NSMutableArray<NSDictionary<NSString *, id> *> *results = [NSMutableArray arrayWithCapacity:textBlocks.count];
    for (NSDictionary<NSString *, id> *block in textBlocks) {
        NSString *blockId = block[@"id"] ?: @"";
        cv::Rect rect;
        if (!MapToClippedRect(block, gray.cols, gray.rows, 0.0, &rect, error)) {
            return nil;
        }
        NSNumber *charCountNumber = block[@"charCount"];
        int charCount = [charCountNumber isKindOfClass:[NSNumber class]] ? charCountNumber.intValue : 1;
        RegionStats stats = ScoreRegion(src, gray, rect, charCount);
        // Native-only fallback confidence (stroke shape + component count), used if the
        // page-relative signals in Dart have nothing to compare against.
        double confidence = 0.5 * stats.strokeVariationScore + 0.5 * stats.componentRatioScore;
        [results addObject:@{
            @"id" : blockId,
            @"confidence" : @(confidence),
            @"angleVariationScore" : @(stats.angleVariationScore),
            @"avgStrokeWidth" : @(stats.avgStrokeWidth),
            @"inkColorB" : @(stats.inkColorB),
            @"inkColorG" : @(stats.inkColorG),
            @"inkColorR" : @(stats.inkColorR),
            @"inkIntensityStdDev" : @(stats.inkIntensityStdDev),
            @"hasWideUnderline" : @(stats.hasWideUnderline),
            @"hasInk" : @(stats.hasInk),
        }];
    }
    return results;
}

+ (nullable NSArray<NSData *> *)handwritingCropsAtPath:(NSString *)imagePath
                                              textBlocks:(NSArray<NSDictionary<NSString *, id> *> *)textBlocks
                                                   error:(NSError **)error {
    static const int kInputW = 128;
    static const int kInputH = 64;
    // Must match ml/generate_dataset.py's NATIVE_PADDING_PX exactly. A raw ML Kit word box
    // resized straight to kInputW x kInputH makes the glyph fill ~100% of the frame — verified
    // against a real photo, this alone (not stroke shape) made the model flag nearly every
    // word, print included, as handwriting. Training now crops the same way this does.
    static const double kPaddingPx = 5.0;

    cv::Mat src;
    if (!ReadOrFail(imagePath, &src, error)) return nil;

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    NSMutableArray<NSData *> *results = [NSMutableArray arrayWithCapacity:textBlocks.count];
    for (NSDictionary<NSString *, id> *block in textBlocks) {
        cv::Rect rect;
        if (!MapToClippedRect(block, gray.cols, gray.rows, kPaddingPx, &rect, error)) {
            return nil;
        }
        cv::Mat crop(gray, rect);
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(kInputW, kInputH));
        if (!resized.isContinuous()) {
            resized = resized.clone();
        }
        [results addObject:[NSData dataWithBytes:resized.data length:(NSUInteger)(kInputW * kInputH)]];
    }
    return results;
}

// Grading ink is a distinct color (commonly red) from printed black/gray text. A mask built
// from color deviation ("redness"), not plain darkness, can never include a black-print pixel
// no matter how close or how it's grown — verified: growing on plain darkness bridged into
// unrelated words only ~7px away on a dense worksheet, but a color-gated mask left print
// untouched even where a stroke crosses directly over it.
static const double kRednessThreshold = 30.0;
static const int kInkDilatePx = 5;
// How close a colored-ink connected component must be to a flagged word's box to count as
// "its" mark. Since inclusion is gated by color, not distance, there is no risk of ever
// marking a black-print pixel this way — so the WHOLE component is taken once any part of it
// is this close, however far the component itself runs (a long diagonal strike-through can
// extend 100px+ from the word it crosses out; clipping the mask to a fixed-size window around
// the word left such strokes half-erased).
static const int kProximityPx = 20;

/// Computes the whole-page colored-ink mask and its connected components ONCE, reused for every
/// flagged word (cheaper than re-deriving a local mask per word, and is what lets a component's
/// full extent be found regardless of which word ends up near which part of it).
static int BuildInkComponents(const cv::Mat &colorSrc, const cv::Mat &graySrc, cv::Mat *labelsOut, cv::Mat *statsOut) {
    std::vector<cv::Mat> channels;
    cv::split(colorSrc, channels);
    cv::Mat bg, redness, rednessMask, darkMask, inkMask, dilated, centroids;
    cv::addWeighted(channels[0], 0.5, channels[1], 0.5, 0.0, bg);
    cv::subtract(channels[2], bg, redness);
    cv::compare(redness, kRednessThreshold, rednessMask, cv::CMP_GT);
    cv::compare(graySrc, 220, darkMask, cv::CMP_LT);
    cv::bitwise_and(rednessMask, darkMask, inkMask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kInkDilatePx, kInkDilatePx));
    cv::dilate(inkMask, dilated, kernel);
    return cv::connectedComponentsWithStats(dilated, *labelsOut, *statsOut, centroids, 8, CV_32S);
}

/// Paints every colored-ink component near `seed` into `mask` in full (not clipped to a local
/// window), plus the seed's own tight box (handles plain composed handwriting glyphs, e.g.
/// fill-in-blank answers, which aren't a distinct color from print).
static void PaintInkMask(const cv::Rect &seed, const cv::Mat &labels, const cv::Mat &stats, int numLabels, cv::Mat *mask) {
    int sx0 = MAX(0, seed.x - kProximityPx);
    int sy0 = MAX(0, seed.y - kProximityPx);
    int sx1 = MIN(labels.cols, seed.x + seed.width + kProximityPx);
    int sy1 = MIN(labels.rows, seed.y + seed.height + kProximityPx);
    if (sx1 > sx0 && sy1 > sy0) {
        cv::Mat window = labels(cv::Rect(sx0, sy0, sx1 - sx0, sy1 - sy0));
        std::set<int> nearbyLabels;
        for (int y = 0; y < window.rows; y++) {
            const int32_t *row = window.ptr<int32_t>(y);
            for (int x = 0; x < window.cols; x++) {
                if (row[x] != 0) nearbyLabels.insert(row[x]);
            }
        }
        for (int label : nearbyLabels) {
            if (label < 1 || label >= numLabels) continue;
            if (stats.at<int32_t>(label, 2) * stats.at<int32_t>(label, 3) < 4) continue;
            cv::Mat componentMask;
            cv::compare(labels, label, componentMask, cv::CMP_EQ);
            cv::bitwise_or(*mask, componentMask, *mask);
        }
    }
    cv::rectangle(*mask, seed, cv::Scalar(255), -1);
}

+ (BOOL)eraseRegionsAtPath:(NSString *)inputPath
                 outputPath:(NSString *)outputPath
                      rects:(NSArray<NSDictionary<NSString *, id> *> *)rects
                    padding:(double)padding
              inpaintRadius:(double)inpaintRadius
                      error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(inputPath, &src, error)) return NO;

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat labels, stats;
    int numLabels = BuildInkComponents(src, gray, &labels, &stats);

    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    for (NSDictionary<NSString *, id> *rectMap in rects) {
        cv::Rect rect;
        if (!MapToClippedRect(rectMap, src.cols, src.rows, padding, &rect, error)) {
            return NO;
        }
        PaintInkMask(rect, labels, stats, numLabels, &mask);
    }

    cv::Mat dst;
    cv::inpaint(src, mask, dst, inpaintRadius, cv::INPAINT_TELEA);

    return WriteOrFail(dst, outputPath, error);
}

@end
