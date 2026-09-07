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

/// `textBlocks` / return entries are `{ "id": String, "left": Double, "top": Double, "right": Double, "bottom": Double }`.
/// Returns `{ "id": String, "confidence": Double, "isLikelyHandwriting": Bool }` per input block.
+ (nullable NSArray<NSDictionary<NSString *, id> *> *)detectHandwritingRegionsAtPath:(NSString *)imagePath
                                                                           textBlocks:(NSArray<NSDictionary<NSString *, id> *> *)textBlocks
                                                                                error:(NSError **)error
    NS_SWIFT_NAME(detectHandwritingRegions(atPath:textBlocks:));

+ (BOOL)eraseRegionsAtPath:(NSString *)inputPath
                 outputPath:(NSString *)outputPath
                      rects:(NSArray<NSDictionary<NSString *, id> *> *)rects
                    padding:(double)padding
              inpaintRadius:(double)inpaintRadius
                      error:(NSError **)error
    NS_SWIFT_NAME(eraseRegions(atPath:outputPath:rects:padding:inpaintRadius:));

@end

NS_ASSUME_NONNULL_END
