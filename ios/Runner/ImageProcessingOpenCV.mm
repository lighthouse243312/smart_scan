// OpenCV headers must be imported before Foundation/the class's own header — OpenCV's C++
// headers conflict with Objective-C globals (e.g. `NO`) when included afterwards.
// Using OpenCV 4.10.0 here (not 5.0.0, matching Android) — the official 5.0.0 iOS framework
// ships a DNN module that references ONNX Runtime/MLAS symbols missing for the device arm64
// slice, so any app linking it fails with "Undefined symbol: MlasGemmBatch" etc. on a real
// device. 4.10.0 predates that DNN/ONNX backend and has no such issue. The algorithms used
// here (GaussianBlur, threshold, distanceTransform, findContours, contourArea, arcLength,
// inpaint, CLAHE...) are stable, unchanged APIs across 4.x/5.x.
#import <opencv2/opencv.hpp>

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
 * A fill-in-the-blank answer sits on top of a pre-printed blank line that's wider than the
 * answer itself (the blank was sized for a guessed-longer answer). A printed word's own
 * underline (used for in-text emphasis) hugs the word tightly instead. So: look for a long,
 * near-solid dark horizontal run in a thin strip just below the word, spanning noticeably wider
 * than the word's own bounding box — that combination is specific to "sits on a blank line",
 * not just "happens to be underlined".
 */
static bool HasWideUnderlineBelow(const cv::Mat &gray, const cv::Rect &rect) {
    int marginX = std::max(4, (int)(rect.width * 0.6));
    int bandLeft = std::max(0, rect.x - marginX);
    int bandRight = std::min(gray.cols, rect.x + rect.width + marginX);
    int bandWidth = bandRight - bandLeft;
    if (bandWidth <= 0) return false;

    int gapBelow = std::max(1, (int)(rect.height * 0.05));
    int bandHeight = std::max(2, (int)(rect.height * 0.25));
    int bandTop = std::min(gray.rows - 1, rect.y + rect.height + gapBelow);
    int bandBottom = std::min(gray.rows, bandTop + bandHeight);
    if (bandBottom <= bandTop) return false;

    cv::Mat band = gray(cv::Rect(bandLeft, bandTop, bandWidth, bandBottom - bandTop));
    cv::Mat binaryBand;
    cv::threshold(band, binaryBand, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    int longestRun = 0;
    for (int y = 0; y < binaryBand.rows; y++) {
        const uchar *row = binaryBand.ptr<uchar>(y);
        int currentRun = 0;
        for (int x = 0; x < binaryBand.cols; x++) {
            if (row[x] != 0) {
                currentRun++;
                if (currentRun > longestRun) longestRun = currentRun;
            } else {
                currentRun = 0;
            }
        }
    }
    return longestRun > rect.width * 1.25;
}

/**
 * How much each letter's own tilt varies from the next, WITHIN this one word — a signal
 * intrinsic to the ink shape itself, independent of position/underline/color, so it still fires
 * on handwriting that isn't sitting on a fill-in-blank line. A printed font renders the exact
 * same glyph outline every time a letter repeats, so every "tall" stroke (a letter's main
 * vertical, not a dot or serif fleck) sits at the same angle across the whole word; a human hand
 * never repeats a stroke at a perfectly identical angle twice.
 */
static double ComponentAngleVariationScore(const cv::Mat &binaryInk, int wordHeight) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryInk, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<double> angleDeviations;
    for (auto &contour : contours) {
        cv::Rect boundingRect = cv::boundingRect(contour);
        // Only "tall" strokes carry a meaningful dominant angle — skip dots/serifs/noise.
        if (boundingRect.height >= wordHeight * 0.4 && contour.size() >= 5) {
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

    double angleScore = ComponentAngleVariationScore(binary, rect.height);

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

+ (BOOL)eraseRegionsAtPath:(NSString *)inputPath
                 outputPath:(NSString *)outputPath
                      rects:(NSArray<NSDictionary<NSString *, id> *> *)rects
                    padding:(double)padding
              inpaintRadius:(double)inpaintRadius
                      error:(NSError **)error {
    cv::Mat src;
    if (!ReadOrFail(inputPath, &src, error)) return NO;

    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    for (NSDictionary<NSString *, id> *rectMap in rects) {
        cv::Rect rect;
        if (!MapToClippedRect(rectMap, src.cols, src.rows, padding, &rect, error)) {
            return NO;
        }
        cv::rectangle(mask, rect, cv::Scalar(255), -1);
    }

    cv::Mat dst;
    cv::inpaint(src, mask, dst, inpaintRadius, cv::INPAINT_TELEA);

    return WriteOrFail(dst, outputPath, error);
}

@end
