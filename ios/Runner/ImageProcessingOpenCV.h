#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Domain for errors produced by ImageProcessingOpenCV. `code` is one of ImageProcessingErrorCode.
extern NSString *const ImageProcessingErrorDomain;

typedef NS_ENUM(NSInteger, ImageProcessingErrorCode) {
    ImageProcessingErrorFileNotFound = 1,
    ImageProcessingErrorInvalidArgument = 2,
    ImageProcessingErrorProcessingFailed = 3,
};

/**
 * Plain Objective-C façade over the OpenCV (Objective-C++) implementation, so Swift can call it
 * without ever seeing a C++ type. Mirrors the Kotlin/OpenCV implementation on Android method for
 * method so both platforms produce the same result for a given input.
 */
@interface ImageProcessingOpenCV : NSObject

+ (BOOL)sharpenAtPath:(NSString *)inputPath
            outputPath:(NSString *)outputPath
                amount:(double)amount
                radius:(double)radius
                 error:(NSError **)error
    NS_SWIFT_NAME(sharpen(atPath:outputPath:amount:radius:));

+ (BOOL)removeShadowAtPath:(NSString *)inputPath
                 outputPath:(NSString *)outputPath
                      error:(NSError **)error
    NS_SWIFT_NAME(removeShadow(atPath:outputPath:));

/// Manual 90°-step rotation — document scanners crop the page rectangle correctly but don't
/// know which edge is "up" for reading, so this lets the user fix orientation by hand.
+ (BOOL)rotateAtPath:(NSString *)inputPath
           outputPath:(NSString *)outputPath
 quarterTurnsClockwise:(NSInteger)quarterTurnsClockwise
                error:(NSError **)error
    NS_SWIFT_NAME(rotate(atPath:outputPath:quarterTurnsClockwise:));

/// Straightens a photographed page (see Straighten.hpp): squares it to its printed rules
/// (rotation + keystone) or, without rules, levels its text rows; a level page is copied
/// unchanged. Returns how: "level", "rules" or "rotate".
+ (nullable NSString *)straightenAtPath:(NSString *)inputPath
                              outputPath:(NSString *)outputPath
                                   error:(NSError **)error
    NS_SWIFT_NAME(straighten(atPath:outputPath:));

/// Mask files are BGRA PNGs the size of the page with two independent layers: alpha = handwriting
/// (drawn semi-transparent red, so the file doubles as the review overlay), blue = printed ink
/// (kept even where alpha is 0). A pixel can be both — handwriting written over print. The
/// `...Mask` methods return the fraction of the page covered by handwriting.

/// Ink-colour + layout method (see InkAnalysis.hpp): ink judged by its optical-density colour
/// against the LOCAL print colour, backed by page layout (print forms regular lines). Works for
/// pen ink that differs from the print even slightly (blue-black ballpoint), not for pen ink that
/// is the same colour as the print. `colorDelta` = how much bluer than print (OD_B/OD_R) a stroke
/// must be; smaller = more sensitive.
+ (nullable NSNumber *)inkColorMaskAtPath:(NSString *)inputPath
                                  maskPath:(NSString *)maskPath
                                colorDelta:(double)colorDelta
                                     error:(NSError **)error
    NS_SWIFT_NAME(inkColorMask(atPath:maskPath:colorDelta:));

/// Model method, combined with the ink-colour method (union of both; print only where both agree):
/// the bundled InkSegmenter U-Net (see ml/train_ink_seg.py) predicts two
/// independent per-pixel layers, print and handwriting; pixels whose handwriting probability
/// exceeds `threshold` (and that are actual ink) form the handwriting layer.
+ (nullable NSNumber *)segmentationMaskAtPath:(NSString *)inputPath
                                      maskPath:(NSString *)maskPath
                                     threshold:(double)threshold
                                    colorDelta:(double)colorDelta
                                         error:(NSError **)error
    NS_SWIFT_NAME(segmentationMask(atPath:maskPath:threshold:colorDelta:));

/// Manual correction: paints (or, with `erase`, clears) brush strokes into the handwriting layer;
/// the print layer is kept. Each stroke is
/// `{ "points": [x0, y0, x1, y1, ...] (image px), "width": Double, "erase": Bool }`.
+ (nullable NSNumber *)applyMaskStrokesAtPath:(NSString *)maskPath
                                    outputPath:(NSString *)outputPath
                                       strokes:(NSArray<NSDictionary<NSString *, id> *> *)strokes
                                         error:(NSError **)error
    NS_SWIFT_NAME(applyMaskStrokes(atPath:outputPath:strokes:));

/// Removes the handwriting by rebuilding it, not blurring: pixels where a stroke crosses real print
/// get the nearby print colour back, the rest is inpainted from the surrounding paper.
/// `analysisPath` is the unprocessed page the mask was computed on; `inputPath` (same geometry)
/// is the page to erase from.
+ (BOOL)eraseWithMaskAtPath:(NSString *)inputPath
                analysisPath:(NSString *)analysisPath
                    maskPath:(NSString *)maskPath
                  outputPath:(NSString *)outputPath
                       error:(NSError **)error
    NS_SWIFT_NAME(eraseWithMask(atPath:analysisPath:maskPath:outputPath:));

/// Erases in passes (see ImageProcessingOpenCV.mm "Multi-pass erase"): the first pass with
/// `maskPath` as `eraseWithMask` does, then up to two more that re-detect on the previous result —
/// the model + colour method when `useModel`, else the colour method — kept off the print the first
/// mask found; print the passes turned to paper is copied back from `inputPath`. Returns the number
/// of passes run.
+ (nullable NSNumber *)eraseHandwritingAtPath:(NSString *)inputPath
                                  analysisPath:(NSString *)analysisPath
                                      maskPath:(NSString *)maskPath
                                    outputPath:(NSString *)outputPath
                                      useModel:(BOOL)useModel
                                     threshold:(double)threshold
                                    colorDelta:(double)colorDelta
                                         error:(NSError **)error
    NS_SWIFT_NAME(eraseHandwriting(atPath:analysisPath:maskPath:outputPath:useModel:threshold:colorDelta:));

@end

NS_ASSUME_NONNULL_END
