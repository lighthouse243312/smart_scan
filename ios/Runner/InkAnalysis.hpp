// Handwriting vs print separation from ink colour + page layout, and the rebuild-erase that uses
// it. Pure OpenCV/C++ (no Objective-C) so it is easy to keep in step with the Kotlin port in
// android/app/src/main/kotlin/.../imageprocessing/InkAnalysis.kt — both must implement the same
// steps with the same constants.
//
// Why these signals (measured on a real phone photo of a printed CV annotated in blue-black
// ballpoint — the case the earlier HSV-saturation method missed entirely):
//  - Ink colour must be judged independently of how dark a pixel is. Ink acts as a filter on the
//    paper, so each channel's optical density OD_c = -ln(pixel_c / paper_c) scales with ink
//    amount but the RATIO OD_B / OD_R does not: toner ≈ 0.98, that ballpoint ≈ 0.86-0.93, from
//    the faint stroke edge to the near-black core. HSV saturation of the same pixels was noise.
//  - The margin is small, so it is compared against the print colour measured LOCALLY (phone
//    lenses shift colour increasingly towards the frame edges) and backed by layout: print sits
//    in regular lines — shared baseline, equal heights, even spacing — handwriting does not.
#pragma once

#import <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace inkanalysis {

// Per-pixel ink test: mean optical density above this is "ink".
constexpr float kInkOD = 0.30f;
// Cores darker than this clip on the sensor, so they carry no colour information.
constexpr float kClippedOD = 1.6f;
// Faint ink (flicks, stroke ends, light pressure) that the colour test cannot judge on its own.
constexpr float kFaintOD = 0.06f;
constexpr float kRimOD = 0.04f;
// Analysis runs at most at this resolution (the Android port must, to fit the Java heap; iOS does
// the same so both platforms decide identically). Masks are scaled back to the page, and the
// erase's pixel work (print colour, inpaint) still happens at full resolution.
constexpr int kWorkLongSide = 2400;

inline cv::Mat ToWorkingSize(const cv::Mat &full) {
    int longSide = std::max(full.cols, full.rows);
    if (longSide <= kWorkLongSide) return full;
    double scale = (double)kWorkLongSide / longSide;
    cv::Mat out;
    cv::resize(full, out, cv::Size(), scale, scale, cv::INTER_AREA);
    return out;
}

inline cv::Mat ScaleMaskTo(const cv::Mat &mask, cv::Size size) {
    if (mask.size() == size) return mask;
    cv::Mat out;
    cv::resize(mask, out, size, 0, 0, cv::INTER_NEAREST);
    return out;
}

struct OpticalDensity {
    cv::Mat od[3];   // per channel (B, G, R), CV_32F
    cv::Mat mean;    // CV_32F
    cv::Mat ink;     // CV_8U 255 = ink
};

/// Stroke-scale unit in pixels (≈ a pen stroke's width) for this resolution; every window size
/// below is a multiple of it so behaviour does not depend on the photo's pixel count.
inline int StrokeUnit(const cv::Mat &image) {
    return std::max(5, (int)(std::max(image.cols, image.rows) / 450)) | 1;
}

inline int Odd(double v) { return std::max(1, (int)std::lround(v)) | 1; }

/// Optical density of every pixel against the local paper colour. Paper = the brightest value
/// nearby (ink only ever darkens it) from a heavily downscaled median, smoothed back up.
inline OpticalDensity ComputeOpticalDensity(const cv::Mat &bgr) {
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(), 0.125, 0.125, cv::INTER_AREA);
    cv::medianBlur(small, small, 15);
    cv::dilate(small, small, cv::Mat::ones(9, 9, CV_8U));
    cv::Mat paper;
    cv::resize(small, paper, bgr.size(), 0, 0, cv::INTER_LINEAR);
    paper.convertTo(paper, CV_32FC3);
    cv::GaussianBlur(paper, paper, cv::Size(0, 0), 8);

    cv::Mat image;
    bgr.convertTo(image, CV_32FC3);
    image += cv::Scalar(1, 1, 1);
    paper += cv::Scalar(1, 1, 1);
    cv::Mat transmittance;
    cv::divide(image, paper, transmittance);
    cv::min(transmittance, 1.0, transmittance);
    cv::max(transmittance, 1e-3, transmittance);
    cv::log(transmittance, transmittance);
    transmittance *= -1.0;

    OpticalDensity result;
    cv::split(transmittance, result.od);
    result.mean = (result.od[0] + result.od[1] + result.od[2]) / 3.0;
    cv::compare(result.mean, kInkOD, result.ink, cv::CMP_GT);
    return result;
}

/// Local ink CHROMA over a window of ink pixels in (minOD, maxOD]: each channel's share of the
/// optical density, (OD_B, OD_G, OD_R) / sum — density-independent like a ratio, but it sees every
/// hue (a purple pen absorbs mostly green and has the same B/R as black toner — measured 0.99).
/// NaN where there is too little ink to tell.
inline void InkChroma(const OpticalDensity &d, float minOD, float maxOD, int window, float minFill, cv::Mat out[3]) {
    cv::Mat inRange = (d.mean > minOD) & (d.mean <= maxOD);
    cv::Mat w;
    inRange.convertTo(w, CV_32F, 1.0 / 255.0);
    cv::Mat s[3], sw;
    for (int c = 0; c < 3; c++) cv::boxFilter(d.od[c].mul(w), s[c], -1, cv::Size(window, window));
    cv::boxFilter(w, sw, -1, cv::Size(window, window));
    for (int c = 0; c < 3; c++) out[c].create(d.mean.size(), CV_32F);
    for (int y = 0; y < sw.rows; y++) {
        const float *b = s[0].ptr<float>(y), *g = s[1].ptr<float>(y), *r = s[2].ptr<float>(y), *c = sw.ptr<float>(y);
        float *ob = out[0].ptr<float>(y), *og = out[1].ptr<float>(y), *orr = out[2].ptr<float>(y);
        for (int x = 0; x < sw.cols; x++) {
            float sum = b[x] + g[x] + r[x];
            if (c[x] > minFill && sum > 1e-3f) { ob[x] = b[x] / sum; og[x] = g[x] / sum; orr[x] = r[x] / sum; }
            else { ob[x] = og[x] = orr[x] = NAN; }
        }
    }
}

/// How the page's pen differs in colour from its print: `ref` = local print chroma (3 planes),
/// `dir` = unit direction of the pen's chroma deviation from it (found per page, so blue, purple,
/// red… pens all work).
struct PenColor {
    cv::Mat ref[3];
    cv::Vec3f dir{-0.7071f, 0.f, 0.7071f};   // fallback only (no clearly other ink on the page)
    // chroma distance -> score units (the units every threshold downstream was tuned in; a unit
    // conversion only, not tied to any ink colour)
    float scale = 4.0f;
};

/// Pen-likeness as a pseudo "ratio" in the units the rest of the pipeline was tuned in (measured
/// on blue ballpoint, where it equals the old OD_B/OD_R ratio relative to print): 1 = print colour,
/// lower = further towards the pen colour. `ref` planes must be at `chroma`'s size.
inline cv::Mat PenScore(const cv::Mat chroma[3], const cv::Mat ref[3], const cv::Vec3f &dir, float scale) {
    cv::Mat score(chroma[0].size(), CV_32F);
    for (int y = 0; y < score.rows; y++) {
        const float *c0 = chroma[0].ptr<float>(y), *c1 = chroma[1].ptr<float>(y), *c2 = chroma[2].ptr<float>(y);
        const float *r0 = ref[0].ptr<float>(y), *r1 = ref[1].ptr<float>(y), *r2 = ref[2].ptr<float>(y);
        float *o = score.ptr<float>(y);
        for (int x = 0; x < score.cols; x++) {
            if (std::isnan(c0[x])) { o[x] = NAN; continue; }
            float proj = (c0[x] - r0[x]) * dir[0] + (c1[x] - r1[x]) * dir[1] + (c2[x] - r2[x]) * dir[2];
            o[x] = 1.0f - scale * proj;
        }
    }
    return score;
}

struct Components {
    int count = 0;
    cv::Mat labels;             // CV_32S
    std::vector<int> x, y, w, h, area;
    std::vector<float> cx, bottom;
};

inline Components FindComponents(const cv::Mat &mask) {
    Components c;
    cv::Mat stats, centroids;
    c.count = cv::connectedComponentsWithStats(mask, c.labels, stats, centroids, 8, CV_32S);
    c.x.resize(c.count); c.y.resize(c.count); c.w.resize(c.count); c.h.resize(c.count);
    c.area.resize(c.count); c.cx.resize(c.count); c.bottom.resize(c.count);
    for (int i = 0; i < c.count; i++) {
        c.x[i] = stats.at<int>(i, cv::CC_STAT_LEFT);
        c.y[i] = stats.at<int>(i, cv::CC_STAT_TOP);
        c.w[i] = stats.at<int>(i, cv::CC_STAT_WIDTH);
        c.h[i] = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        c.area[i] = stats.at<int>(i, cv::CC_STAT_AREA);
        c.cx[i] = c.x[i] + c.w[i] / 2.0f;
        c.bottom[i] = (float)(c.y[i] + c.h[i]);
    }
    return c;
}

/// Mask of pixels whose component label is flagged in `flags`.
inline cv::Mat PaintComponents(const cv::Mat &labels, const std::vector<char> &flags) {
    cv::Mat out(labels.size(), CV_8U);
    for (int y = 0; y < labels.rows; y++) {
        const int *l = labels.ptr<int>(y);
        uchar *o = out.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; x++) o[x] = flags[l[x]] ? 255 : 0;
    }
    return out;
}

/// Fraction of each component's pixels that are set in `mask`.
inline std::vector<float> FractionPerComponent(const Components &c, const cv::Mat &mask) {
    std::vector<float> sum(c.count, 0.f);
    for (int y = 0; y < c.labels.rows; y++) {
        const int *l = c.labels.ptr<int>(y);
        const uchar *m = mask.ptr<uchar>(y);
        for (int x = 0; x < c.labels.cols; x++) if (m[x]) sum[l[x]] += 1.f;
    }
    for (int i = 0; i < c.count; i++) sum[i] /= std::max(1, c.area[i]);
    return sum;
}

struct UnionFind {
    std::vector<int> parent;
    explicit UnionFind(int n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    int Find(int a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    }
    void Join(int a, int b) { parent[Find(a)] = Find(b); }
};

inline float Percentile(std::vector<float> v, float p) {
    if (v.empty()) return 0.f;
    size_t i = std::min(v.size() - 1, (size_t)std::lround(p / 100.0 * (v.size() - 1)));
    std::nth_element(v.begin(), v.begin() + i, v.end());
    return v[i];
}

struct TextLine {
    std::vector<int> members;
    float medianHeight;
};

/// Regular text lines by GEOMETRY alone: glyph-sized components that follow each other on a
/// shared baseline with near-equal heights, fitted to a straight line. Printed text always forms
/// these; handwriting only occasionally does, which the colour vote in the caller resolves.
inline std::vector<TextLine> FindRegularLines(const Components &c, int unit, int imageHeight) {
    std::vector<int> glyphs;
    for (int i = 1; i < c.count; i++) {
        if (c.area[i] >= unit * 2 && c.h[i] >= unit && c.h[i] <= imageHeight * 0.03 && c.w[i] <= c.h[i] * 4) {
            glyphs.push_back(i);
        }
    }
    std::sort(glyphs.begin(), glyphs.end(), [&](int a, int b) { return c.cx[a] < c.cx[b]; });
    UnionFind uf(c.count);
    for (size_t a = 0; a < glyphs.size(); a++) {
        int i = glyphs[a];
        for (size_t b = a + 1; b < glyphs.size() && c.cx[glyphs[b]] - c.cx[i] < 3.0f * c.h[i]; b++) {
            int o = glyphs[b];
            float hmax = (float)std::max(c.h[i], c.h[o]);
            float ratio = (float)c.h[o] / (float)c.h[i];
            if (std::fabs(c.bottom[o] - c.bottom[i]) < 0.18f * hmax && ratio > 0.6f && ratio < 1.67f) uf.Join(o, i);
        }
    }
    std::vector<std::vector<int>> groups(c.count);
    for (int i : glyphs) groups[uf.Find(i)].push_back(i);

    std::vector<TextLine> lines;
    for (auto &g : groups) {
        if (g.size() < 5) continue;
        // least-squares baseline: bottom = a * cx + b
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i : g) { sx += c.cx[i]; sy += c.bottom[i]; sxx += c.cx[i] * c.cx[i]; sxy += c.cx[i] * c.bottom[i]; }
        double n = (double)g.size(), den = n * sxx - sx * sx;
        double a = den != 0 ? (n * sxy - sx * sy) / den : 0, b = (sy - a * sx) / n;
        std::vector<float> resid, heights;
        for (int i : g) { resid.push_back((float)std::fabs(a * c.cx[i] + b - c.bottom[i])); heights.push_back((float)c.h[i]); }
        float hm = Percentile(heights, 50);
        std::vector<float> dev;
        for (float h : heights) dev.push_back(std::fabs(h - hm));
        if (Percentile(resid, 70) < 0.08f * hm && Percentile(dev, 50) < 0.2f * hm) lines.push_back({g, hm});
    }
    return lines;
}

/// Print-colour reference per region of the page: median ink colour of the regular-line pixels
/// in each coarse cell, empty cells filled from their neighbours, smoothed and scaled to the page.
inline cv::Mat LocalPrintReference(const cv::Mat &ratio, const cv::Mat &regularPixels, float *outGlobal) {
    int H = ratio.rows, W = ratio.cols;
    int cell = std::max(64, std::max(H, W) / 12);
    int gh = (H + cell - 1) / cell, gw = (W + cell - 1) / cell;
    std::vector<std::vector<float>> samples(gh * gw);
    for (int y = 0; y < H; y++) {
        const float *r = ratio.ptr<float>(y);
        const uchar *m = regularPixels.ptr<uchar>(y);
        for (int x = 0; x < W; x++) if (m[x] && !std::isnan(r[x])) samples[(y / cell) * gw + (x / cell)].push_back(r[x]);
    }
    cv::Mat grid(gh, gw, CV_32F, cv::Scalar(NAN));
    std::vector<float> known;
    for (int i = 0; i < gh * gw; i++) {
        if (samples[i].size() >= 200) {
            float m = Percentile(samples[i], 50);
            grid.at<float>(i / gw, i % gw) = m;
            known.push_back(m);
        }
    }
    float global = known.empty() ? 1.0f : Percentile(known, 50);
    *outGlobal = global;
    cv::Mat filled(gh, gw, CV_32F, cv::Scalar(global));
    grid.copyTo(filled, grid == grid);   // NaN != NaN, so this copies only the known cells
    for (int pass = 0; pass < 3; pass++) {
        cv::Mat next = filled.clone();
        for (int y = 0; y < gh; y++) {
            for (int x = 0; x < gw; x++) {
                if (!std::isnan(grid.at<float>(y, x))) continue;
                float sum = 0; int cnt = 0;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                    int yy = y + dy, xx = x + dx;
                    if (yy < 0 || xx < 0 || yy >= gh || xx >= gw) continue;
                    float v = grid.at<float>(yy, xx);
                    if (!std::isnan(v)) { sum += v; cnt++; }
                }
                if (cnt > 0) next.at<float>(y, x) = sum / cnt;
            }
        }
        filled = next;
    }
    cv::GaussianBlur(filled, filled, cv::Size(0, 0), 1.0);
    cv::Mat ref;
    cv::resize(filled, ref, ratio.size(), 0, 0, cv::INTER_LINEAR);
    return ref;
}

inline cv::Mat Ellipse(int diameter) {
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(Odd(diameter), Odd(diameter)));
}

/// Keeps only the components of `seeds`' connected regions within `allowed` that contain a seed.
inline cv::Mat GrowWithin(const cv::Mat &seeds, const cv::Mat &allowed) {
    Components c = FindComponents(allowed);
    std::vector<char> seeded(c.count, 0);
    for (int y = 0; y < seeds.rows; y++) {
        const uchar *s = seeds.ptr<uchar>(y);
        const int *l = c.labels.ptr<int>(y);
        for (int x = 0; x < seeds.cols; x++) if (s[x] && l[x] > 0) seeded[l[x]] = 1;
    }
    return PaintComponents(c.labels, seeded);
}

inline void RemoveSpecks(cv::Mat &mask, int minArea) {
    Components c = FindComponents(mask);
    std::vector<char> keep(c.count, 0);
    for (int i = 1; i < c.count; i++) keep[i] = c.area[i] >= minArea;
    mask = PaintComponents(c.labels, keep);
}

/// Detects handwriting (`outHandwriting`) and the print layer the erase must protect/restore
/// (`outPrint`) from colour + layout. The print's own colour is measured from the page's regular
/// text lines and the pen's hue is found as the dominant colour deviation of the other ink — no ink
/// colour is assumed. `colorDelta`: how far towards that hue (in score units) an ink must be to
/// count as pen (smaller = more sensitive).
inline void DetectAtWorkingSize(const cv::Mat &src, double colorDelta, cv::Mat *outHandwriting, cv::Mat *outPrint, PenColor *outPen, cv::Mat *outFragments, cv::Mat *outMixed) {
    const int H = src.rows, W = src.cols, longSide = std::max(H, W);
    const int k = StrokeUnit(src);
    OpticalDensity d = ComputeOpticalDensity(src);
    // colour window: wide enough to average out sensor noise (a stroke-width window was too noisy
    // at the 2400 px working size — measured: whole printed words turned pen-coloured)
    const int colorWindow = std::max(k, longSide / 270) | 1;
    cv::Mat chroma[3];
    InkChroma(d, kInkOD, kClippedOD, colorWindow, 0.05f, chroma);

    // --- regular lines by geometry, then the local print-colour reference they give
    Components c = FindComponents(d.ink);
    std::vector<TextLine> lines = FindRegularLines(c, k, H);
    std::vector<char> regular(c.count, 0);
    for (auto &l : lines) for (int i : l.members) regular[i] = 1;
    cv::Mat regularMask = PaintComponents(c.labels, regular);
    PenColor pen;
    float unusedGlobal = 0.f;
    for (int ch = 0; ch < 3; ch++) pen.ref[ch] = LocalPrintReference(chroma[ch], regularMask, &unusedGlobal);
    // Other ink: how does ink OUTSIDE regular print lines differ in colour from the print? Its
    // deviations are binned by hue direction (angle in the chroma plane); the pen shows up as the
    // dominant direction, while lens fringes, photo and paper noise spread over all directions.
    // No ink colour is assumed.
    {
        const int bins = 72;
        std::vector<double> hist(bins, 0.0);
        const cv::Vec3d e1(0.7071, 0.0, -0.7071), e2(-0.4082, 0.8165, -0.4082);  // basis of the chroma plane
        // judged per ink STROKE (connected component), not per pixel: a lens shifts colour in
        // opposite directions on either edge of every stroke, which cancels over the stroke while the
        // ink's own hue does not (measured: per-pixel peaks landed ~40° off the real pen hue)
        std::vector<cv::Vec3d> devSum(c.count, cv::Vec3d(0, 0, 0));
        std::vector<int> devCnt(c.count, 0);
        for (int y = 0; y < H; y++) {
            const int *l = c.labels.ptr<int>(y);
            const uchar *ink = d.ink.ptr<uchar>(y);
            const float *c0 = chroma[0].ptr<float>(y), *c1 = chroma[1].ptr<float>(y), *c2 = chroma[2].ptr<float>(y);
            const float *r0 = pen.ref[0].ptr<float>(y), *r1 = pen.ref[1].ptr<float>(y), *r2 = pen.ref[2].ptr<float>(y);
            for (int x = 0; x < W; x++) {
                if (!ink[x] || l[x] == 0 || regular[l[x]] || std::isnan(c0[x])) continue;
                devSum[l[x]] += cv::Vec3d(c0[x] - r0[x], c1[x] - r1[x], c2[x] - r2[x]);
                devCnt[l[x]]++;
            }
        }
        const int maxArea = (int)(0.002 * H * W);   // photos, logos, page edges are not strokes
        for (int i = 1; i < c.count; i++) {
            if (devCnt[i] < 2 * k || c.area[i] > maxArea) continue;
            cv::Vec3d dv = devSum[i] / devCnt[i];
            double u = dv.dot(e1), v = dv.dot(e2), m = std::sqrt(u * u + v * v);
            if (m < 0.006) continue;
            int bin = ((int)std::floor((std::atan2(v, u) + CV_PI) / (2 * CV_PI) * bins)) % bins;
            hist[bin] += devCnt[i] * std::min(m, 0.05);
        }
        int best = -1;
        double bestVal = 0.0;
        for (int i = 0; i < bins; i++) {
            double v = 0;
            for (int o = -2; o <= 2; o++) v += hist[(i + o + bins) % bins];   // smooth over ±12°
            if (v > bestVal) { bestVal = v; best = i; }
        }
        if (best >= 0 && bestVal > 0.5) {
            double ang = (best + 0.5) / bins * 2 * CV_PI - CV_PI;
            cv::Vec3d dirv = e1 * std::cos(ang) + e2 * std::sin(ang);
            pen.dir = cv::Vec3f((float)dirv[0], (float)dirv[1], (float)dirv[2]);
        }
#ifdef INK_DEBUG
        fprintf(stderr, "pen dir B,G,R = %.3f %.3f %.3f scale %.2f (best bin %d val %.1f)\n", pen.dir[0], pen.dir[1], pen.dir[2], pen.scale, best, bestVal);
        for (int i = 0; i < bins; i += 6) fprintf(stderr, "%d:%.0f ", i, hist[i] + hist[(i+1)%bins] + hist[(i+2)%bins] + hist[(i+3)%bins] + hist[(i+4)%bins] + hist[(i+5)%bins]);
        fprintf(stderr, "\n");
#endif
    }
    cv::Mat ratio = PenScore(chroma, pen.ref, pen.dir, pen.scale);
    cv::Mat ref(src.size(), CV_32F, cv::Scalar(1.0f));

    // --- pen candidates: ink noticeably bluer than the print around it
    cv::Mat cand(src.size(), CV_8U, cv::Scalar(0)), relative(src.size(), CV_32F);
    for (int y = 0; y < H; y++) {
        const float *r = ratio.ptr<float>(y), *rf = ref.ptr<float>(y);
        const uchar *ink = d.ink.ptr<uchar>(y);
        uchar *o = cand.ptr<uchar>(y);
        float *rel = relative.ptr<float>(y);
        for (int x = 0; x < W; x++) {
            rel[x] = std::isnan(r[x]) ? NAN : r[x] - rf[x];
            o[x] = (ink[x] && !std::isnan(r[x]) && r[x] < rf[x] - colorDelta) ? 255 : 0;
        }
    }
    std::vector<float> fracC = FractionPerComponent(c, cand);

    // --- line vote: a regular line whose glyphs are mostly print-coloured is print, together
    // with the dots/accents/punctuation inside its band
    std::vector<char> printComp(c.count, 0);
    std::vector<char> small(c.count, 0);
    for (int i = 1; i < c.count; i++) small[i] = c.area[i] < k * k * 4;
    // line consensus per character: a glyph whose neighbours on its (loose) text line are all
    // print is print too, unless it is itself strongly pen-coloured — merged letters can break a
    // line's regularity, but not its colour context
    {
        std::vector<int> ids;
        for (int i = 1; i < c.count; i++) {
            if (c.area[i] >= k * 2 && c.h[i] >= k && c.h[i] <= H * 0.03 && c.w[i] <= c.h[i] * 4) ids.push_back(i);
        }
        std::sort(ids.begin(), ids.end(), [&](int a, int b) { return c.cx[a] < c.cx[b]; });
        UnionFind uf(c.count);
        for (size_t a = 0; a < ids.size(); a++) {
            int i = ids[a];
            float reach = 2.5f * std::max(c.h[i], 3 * k);
            for (size_t b = a + 1; b < ids.size() && c.cx[ids[b]] - c.cx[i] < reach; b++) {
                int o = ids[b];
                float ci = c.y[i] + c.h[i] / 2.0f, co = c.y[o] + c.h[o] / 2.0f;
                float hr = (float)c.h[o] / (float)c.h[i];
                if (std::fabs(co - ci) < 0.5f * std::max(c.h[i], c.h[o]) && hr > 0.5f && hr < 2.0f) uf.Join(o, i);
            }
        }
        std::vector<std::vector<int>> groups(c.count);
        for (int i : ids) groups[uf.Find(i)].push_back(i);
        for (auto &g : groups) {
            if (g.size() < 4) continue;   // already sorted by cx (ids were)
            for (size_t p = 0; p < g.size(); p++) {
                double pen = 0, total = 0;
                for (size_t q = p >= 3 ? p - 3 : 0; q < std::min(g.size(), p + 4); q++) {
                    if (q == p) continue;
                    pen += fracC[g[q]] * c.area[g[q]]; total += c.area[g[q]];
                }
                if (total > 0 && pen / total < 0.2 && fracC[g[p]] < 0.85f) printComp[g[p]] = 1;
            }
        }
    }
    for (auto &l : lines) {
        std::vector<float> f;
        for (int i : l.members) f.push_back(fracC[i]);
        // regularity alone cannot tell (neat handwriting lines are just as straight — measured);
        // the line's colour decides: pen lines measured ≈0.98 pen-coloured, print lines ≤0.45
        // even at the lens-shifted page edge
        if (Percentile(f, 50) >= 0.6f) continue;
        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
        for (int i : l.members) {
            printComp[i] = 1;
            x0 = std::min(x0, (float)c.x[i]); x1 = std::max(x1, (float)(c.x[i] + c.w[i]));
            y0 = std::min(y0, (float)c.y[i]); y1 = std::max(y1, c.bottom[i]);
        }
        y0 -= 0.6f * l.medianHeight; y1 += 0.3f * l.medianHeight;
        for (int i = 1; i < c.count; i++) {
            if (small[i] && c.cx[i] >= x0 && c.cx[i] <= x1 && c.y[i] >= y0 && c.y[i] + c.h[i] <= y1) printComp[i] = 1;
        }
    }
    cv::Mat printByLayout = PaintComponents(c.labels, printComp);

    // --- stroke vote: a stroke is pen if enough of it is pen-coloured, or its pen-coloured part
    // is itself a sizeable blob (handwriting crossing print)
    std::vector<char> compHw(c.count, 0);
    for (int i = 1; i < c.count; i++) compHw[i] = fracC[i] >= 0.35f && !printComp[i];
    cv::Mat candFree = cand & ~printByLayout;
    Components cb = FindComponents(candFree);
    std::vector<char> big(cb.count, 0);
    for (int i = 1; i < cb.count; i++) big[i] = cb.area[i] >= (2 * k) * (2 * k);
    cv::Mat hw = candFree & (PaintComponents(c.labels, compHw) | PaintComponents(cb.labels, big));

    // small pen marks (accents, dots, commas) right next to handwriting
    cv::Mat near;
    cv::dilate(hw, near, Ellipse(4 * k + 1));
    std::vector<char> smallPen(c.count, 0);
    for (int i = 1; i < c.count; i++) smallPen[i] = small[i] && !printComp[i] && fracC[i] >= 0.15f;
    hw |= PaintComponents(c.labels, smallPen) & near & d.ink;

    // --- handwritten lines: in a loose line of non-print strokes that is mostly pen already,
    // strokes only slightly bluer than print are pen too (pen colour varies with pressure)
    {
        std::vector<int> ids;
        for (int i = 1; i < c.count; i++) {
            bool glyph = c.area[i] >= k * 2 && c.h[i] >= k && c.h[i] <= H * 0.03 && c.w[i] <= c.h[i] * 4;
            if (glyph && !printComp[i]) ids.push_back(i);
        }
        std::sort(ids.begin(), ids.end(), [&](int a, int b) { return c.cx[a] < c.cx[b]; });
        UnionFind uf(c.count);
        for (size_t a = 0; a < ids.size(); a++) {
            int i = ids[a];
            float reach = 2.5f * std::max(c.h[i], 3 * k);
            for (size_t b = a + 1; b < ids.size() && c.cx[ids[b]] - c.cx[i] < reach; b++) {
                int o = ids[b];
                float ci = c.y[i] + c.h[i] / 2.0f, co = c.y[o] + c.h[o] / 2.0f;
                if (std::fabs(co - ci) < 0.6f * std::max(c.h[i], c.h[o])) uf.Join(o, i);
            }
        }
        std::vector<float> hwFrac = FractionPerComponent(c, hw);
        std::vector<double> relSum(c.count, 0.0);
        std::vector<int> relCnt(c.count, 0);
        for (int y = 0; y < H; y++) {
            const int *l = c.labels.ptr<int>(y);
            const float *rel = relative.ptr<float>(y);
            for (int x = 0; x < W; x++) if (l[x] > 0 && !std::isnan(rel[x])) { relSum[l[x]] += rel[x]; relCnt[l[x]]++; }
        }
        std::vector<std::vector<int>> groups(c.count);
        for (int i : ids) groups[uf.Find(i)].push_back(i);
        std::vector<char> weakPen(c.count, 0);
        for (auto &g : groups) {
            if (g.size() < 2) continue;
            double pen = 0, total = 0;
            for (int i : g) { pen += hwFrac[i] * c.area[i]; total += c.area[i]; }
            if (pen < 0.5 * total) continue;
            for (int i : g) if (relCnt[i] > 0 && relSum[i] / relCnt[i] < -0.015) weakPen[i] = 1;
        }
        hw |= PaintComponents(c.labels, weakPen) & d.ink;
    }

    // --- neighbourhood consensus: a pen stroke's darkness/colour is uneven, so single pixels
    // flip; an ink pixel takes the majority label of the ink around it (≈3 stroke widths), both
    // ways — holes inside handwriting close, isolated pen-coloured specks in print go back to
    // print. Layout print (regular lines) keeps its label.
    {
        int cwin = 3 * k + 1;
        cv::Mat inkF, inkSum;
        d.ink.convertTo(inkF, CV_32F, 1.0 / 255.0);
        cv::boxFilter(inkF, inkSum, -1, cv::Size(cwin, cwin));
        for (int pass = 0; pass < 2; pass++) {
            cv::Mat hwInk, hwF, hwSum;
            hwInk = hw & d.ink;
            hwInk.convertTo(hwF, CV_32F, 1.0 / 255.0);
            cv::boxFilter(hwF, hwSum, -1, cv::Size(cwin, cwin));
            for (int y = 0; y < H; y++) {
                const float *hs = hwSum.ptr<float>(y), *is = inkSum.ptr<float>(y);
                const uchar *ink = d.ink.ptr<uchar>(y), *lp = printByLayout.ptr<uchar>(y);
                uchar *h = hw.ptr<uchar>(y);
                for (int x = 0; x < W; x++) {
                    if (!ink[x]) continue;
                    h[x] = (!lp[x] && hs[x] >= 0.5f * std::max(is[x], 1e-6f)) ? 255 : 0;
                }
            }
        }
    }

    // --- pen strokes merged INTO printed letters (an underline touching the word): the whole
    // component was voted print by layout, so mark its pen-coloured pixels for full-resolution
    // refinement, where the pixel colour + direction decides which pixels are the pen stroke
    std::vector<char> mixed(c.count, 0);
    for (int i = 1; i < c.count; i++) mixed[i] = printComp[i] && fracC[i] >= 0.06f;
    cv::Mat mixedMask = PaintComponents(c.labels, mixed) & cand;
    // strongly mixed (a pen underline fused to a printed word, measured 65% pen-coloured): its
    // pen-coloured pixels are handwriting outright, protected from carving like fragments
    std::vector<char> mixedStrong(c.count, 0);
    // ...and shaped like it: much wider than tall (a letter + its underline, measured 3:1, vs
    // ~1:1 for letters) with a real stroke's worth of pen-coloured pixels — colour noise alone
    // makes plain printed letters 40% "pen" near the frame edge
    for (int i = 1; i < c.count; i++) {
        mixedStrong[i] = printComp[i] && fracC[i] >= 0.4f && c.w[i] >= 2.5f * c.h[i] &&
                         fracC[i] * c.area[i] >= (float)(3 * k * k);
    }
    cv::Mat mixedPen = PaintComponents(c.labels, mixedStrong) & cand;
    hw |= mixedPen;

    // --- small detached pen fragments (a letter's lead-in curl, a colon, a dash, an accent):
    // colour is unreliable on bits this thin, so their NEAREST neighbour decides. Tiny pieces
    // outside any regular print line that come closer to handwriting than to any letter-sized
    // non-handwriting ink (within ~8 stroke widths) are handwriting; mid-sized pieces need some
    // pen colour and to lie within 2 stroke widths of handwriting (a curl hugging a printed rule).
    cv::Mat fragmentMask = cv::Mat::zeros(src.size(), CV_8U);
    {
        const int tinyMax = (2 * k) * (2 * k), smallMax = (4 * k) * (4 * k);
        std::vector<float> nonHwFrac = FractionPerComponent(c, d.ink & ~hw);
        std::vector<char> refComp(c.count, 0);
        for (int i = 1; i < c.count; i++) refComp[i] = printComp[i] || (c.area[i] > tinyMax && nonHwFrac[i] >= 0.5f);
        cv::Mat printRefMask = PaintComponents(c.labels, refComp) & d.ink & ~hw;
        cv::Mat hwInk = hw & d.ink, distHw, distPr;
        cv::distanceTransform(~hwInk, distHw, cv::DIST_L2, 3);
        cv::distanceTransform(~printRefMask, distPr, cv::DIST_L2, 3);
        std::vector<float> dHw(c.count, 1e9f), dPr(c.count, 1e9f);
        for (int y = 0; y < H; y++) {
            const int *l = c.labels.ptr<int>(y);
            const float *dh = distHw.ptr<float>(y), *dp = distPr.ptr<float>(y);
            const uchar *ink = d.ink.ptr<uchar>(y);
            for (int x = 0; x < W; x++) {
                if (!ink[x] || l[x] == 0) continue;
                dHw[l[x]] = std::min(dHw[l[x]], dh[x]);
                dPr[l[x]] = std::min(dPr[l[x]], dp[x]);
            }
        }
        std::vector<char> fragment(c.count, 0);
        for (int i = 1; i < c.count; i++) {
            if (printComp[i]) continue;
            bool tinyRule = c.area[i] <= tinyMax && dHw[i] < dPr[i] && dHw[i] <= 8 * k;
            bool midRule = c.area[i] <= smallMax && fracC[i] >= 0.25f && dHw[i] <= 2 * k;
            fragment[i] = tinyRule || midRule;
        }
        fragmentMask = (PaintComponents(c.labels, fragment) & d.ink) | mixedPen;
        hw |= fragmentMask;
    }

    // --- pictures (photos, logos): ink dense over a wide area; text strokes never are
    int win = Odd(longSide * 0.03);
    cv::Mat inkF, dens;
    d.ink.convertTo(inkF, CV_32F, 1.0 / 255.0);
    cv::boxFilter(inkF, dens, -1, cv::Size(win, win));
    cv::Mat dense = dens > 0.5f;
    Components cd = FindComponents(dense);
    std::vector<char> bigDense(cd.count, 0);
    for (int i = 1; i < cd.count; i++) bigDense[i] = cd.area[i] >= (2 * win) * (2 * win);
    cv::Mat picture;
    cv::dilate(PaintComponents(cd.labels, bigDense), picture, cv::Mat::ones(win, win, CV_8U));
    hw &= ~picture;

    // --- print to protect: print-coloured ink, layout print, pictures, and faint ink that is
    // print-coloured (light grey rules, light print)
    cv::Mat printColored(src.size(), CV_8U);
    for (int y = 0; y < H; y++) {
        const float *rel = relative.ptr<float>(y);
        const uchar *ink = d.ink.ptr<uchar>(y);
        uchar *o = printColored.ptr<uchar>(y);
        for (int x = 0; x < W; x++) o[x] = (ink[x] && !std::isnan(rel[x]) && rel[x] > -0.02f) ? 255 : 0;
    }
    cv::Mat print = printColored | printByLayout | picture;
    cv::Mat faintChroma[3];
    InkChroma(d, kFaintOD, kInkOD, 2 * k + 1, 0.1f, faintChroma);
    cv::Mat faintRatio = PenScore(faintChroma, pen.ref, pen.dir, pen.scale);
    cv::Mat faintPrint(src.size(), CV_8U);
    for (int y = 0; y < H; y++) {
        const float *fr = faintRatio.ptr<float>(y), *rf = ref.ptr<float>(y), *m = d.mean.ptr<float>(y);
        const uchar *h = hw.ptr<uchar>(y);
        uchar *o = faintPrint.ptr<uchar>(y);
        for (int x = 0; x < W; x++) {
            bool faint = m[x] > kFaintOD && m[x] <= kInkOD;
            o[x] = (faint && !h[x] && !std::isnan(fr[x]) && fr[x] > rf[x] - 0.03f) ? 255 : 0;
        }
    }
    print |= faintPrint;
    cv::dilate(print, print, cv::Mat::ones(3, 3, CV_8U));

    // --- grow along the pen strokes into their faint parts, stopping at print
    cv::Mat allowed = (d.mean > kFaintOD) & ~print;
    hw |= GrowWithin(hw & allowed, allowed);
    RemoveSpecks(hw, std::max(12, k * k / 4));
    hw |= fragmentMask & ~picture;   // context-placed dots are not noise

    *outHandwriting = hw;
    *outPrint = print;
    *outPen = pen;
    *outFragments = fragmentMask & ~picture;
    *outMixed = mixedMask & ~picture;
}

// MARK: - Pixel-level refinement at full resolution
//
// Downscaling to the working size blends each thin stroke's edge pixels with the paper and the
// neighbouring ink, and that destroys the colour signal (measured on the real photo: per-pixel
// pen-vs-print accuracy 96% at 4032 px, 74% at 2400 px). So the working-size pass only says
// WHERE handwriting is; inside that zone every pixel is decided again at full resolution from its
// own colour, then arbitrated along 4 directions:
//  - pen along (almost) every direction, no print running through → pen (closes holes left by
//    uneven pen darkness);
//  - a print stroke running straight through a pen pixel that is much darker than the pen around
//    it (ink densities add where two inks stack) → OVERLAP: erase the pen, restore the print;
//  - print continuing on both sides across a pen stroke (perpendicular to it) → print hidden under
//    the pen, also restored.
// Directions are judged on 3 px wide bands at several lengths (2, 4 and 6 stroke widths) — a
// single-pixel line misjudges slightly slanted or curved strokes (measured on a ground-truth page:
// print-under-pen kept 70% → 90%).

constexpr int kDirBand = 3;
constexpr int kDirLengths[] = {2, 4, 6};
constexpr float kOverlapDarker = 1.35f;
constexpr int kPerpendicular[4] = {1, 0, 3, 2};   // horizontal<->vertical, diagonal<->anti-diagonal

/// 4 directional band kernels (horizontal, vertical, diagonal, anti-diagonal), `band` px wide and
/// L long, with the centre block zeroed so a pixel is judged only by its neighbours on the line.
/// `side` 0 = both sides, -1 / +1 = only one side of the centre.
inline std::vector<cv::Mat> DirectionKernels(int L, int band, int side = 0) {
    std::vector<cv::Mat> ks;
    int c = L / 2, hb = band / 2;
    for (int dir = 0; dir < 4; dir++) {
        cv::Mat k = cv::Mat::zeros(L, L, CV_32F);
        for (int i = 0; i < L; i++) {
            for (int o = -hb; o <= hb; o++) {
                int y, x;
                if (dir == 0) { y = c + o; x = i; }
                else if (dir == 1) { y = i; x = c + o; }
                else if (dir == 2) { y = i; x = i + o; }
                else { y = i; x = L - 1 - i + o; }
                if (x < 0 || y < 0 || x >= L || y >= L) continue;
                int along = (dir == 1 || dir >= 2) ? y : x;   // position along the line
                if (std::abs(y - c) <= hb && std::abs(x - c) <= hb) continue;   // centre block
                if (side < 0 && along >= c) continue;
                if (side > 0 && along <= c) continue;
                k.at<float>(y, x) = 1.f;
            }
        }
        ks.push_back(k);
    }
    return ks;
}

inline cv::Mat Filter(const cv::Mat &src, const cv::Mat &kernel) {
    cv::Mat out;
    cv::filter2D(src, out, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    return out;
}

/// Paper colour at 1/8 scale (brightest nearby — ink only darkens it), for ROI-wise density.
inline cv::Mat PaperLowRes(const cv::Mat &bgr) {
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(), 0.125, 0.125, cv::INTER_AREA);
    cv::medianBlur(small, small, 15);
    cv::dilate(small, small, cv::Mat::ones(9, 9, CV_8U));
    return small;
}

/// Optical density of one region of the page against the global paper estimate.
inline OpticalDensity DensityInRoi(const cv::Mat &bgr, const cv::Mat &paperSmall, const cv::Rect &roi) {
    double fx = (double)paperSmall.cols / bgr.cols, fy = (double)paperSmall.rows / bgr.rows;
    // paper crop with a margin so the blur below has context, then trimmed back to the ROI
    int m = 64;
    cv::Rect big(std::max(0, roi.x - m), std::max(0, roi.y - m), 0, 0);
    big.width = std::min(bgr.cols, roi.x + roi.width + m) - big.x;
    big.height = std::min(bgr.rows, roi.y + roi.height + m) - big.y;
    cv::Rect smallRect((int)std::floor(big.x * fx), (int)std::floor(big.y * fy), 0, 0);
    smallRect.width = std::max(1, std::min(paperSmall.cols, (int)std::ceil((big.x + big.width) * fx)) - smallRect.x);
    smallRect.height = std::max(1, std::min(paperSmall.rows, (int)std::ceil((big.y + big.height) * fy)) - smallRect.y);
    cv::Mat paper;
    cv::resize(paperSmall(smallRect), paper, big.size(), 0, 0, cv::INTER_LINEAR);
    paper.convertTo(paper, CV_32FC3);
    cv::GaussianBlur(paper, paper, cv::Size(0, 0), 8);
    paper = paper(cv::Rect(roi.x - big.x, roi.y - big.y, roi.width, roi.height)).clone();

    cv::Mat image;
    bgr(roi).convertTo(image, CV_32FC3);
    image += cv::Scalar(1, 1, 1);
    paper += cv::Scalar(1, 1, 1);
    cv::Mat t;
    cv::divide(image, paper, t);
    cv::min(t, 1.0, t);
    cv::max(t, 1e-3, t);
    cv::log(t, t);
    t *= -1.0;
    OpticalDensity d;
    cv::split(t, d.od);
    d.mean = (d.od[0] + d.od[1] + d.od[2]) / 3.0;
    cv::compare(d.mean, kInkOD, d.ink, cv::CMP_GT);
    return d;
}

/// Bounding boxes (full-resolution, padded by `pad`) of the regions covered by `mask`, merged
/// where they overlap — the pixel-level passes only run inside these. Found on a 1/8 copy so a
/// page of scattered strokes gives a handful of boxes, not thousands.
inline std::vector<cv::Rect> RegionsOf(const cv::Mat &mask, int pad) {
    const double f = 0.125;
    cv::Mat small;
    cv::resize(mask, small, cv::Size(), f, f, cv::INTER_AREA);
    small = small > 0;
    int p = std::max(1, (int)std::ceil(pad * f));
    cv::dilate(small, small, cv::Mat::ones(2 * p + 1, 2 * p + 1, CV_8U));
    Components c = FindComponents(small);
    cv::Rect page(0, 0, mask.cols, mask.rows);
    std::vector<cv::Rect> boxes;
    for (int i = 1; i < c.count; i++) {
        cv::Rect r((int)(c.x[i] / f) - 8, (int)(c.y[i] / f) - 8, (int)(c.w[i] / f) + 16, (int)(c.h[i] / f) + 16);
        boxes.push_back(r & page);
    }
    bool merged = true;
    while (merged) {
        merged = false;
        for (size_t a = 0; a < boxes.size() && !merged; a++) {
            for (size_t b = a + 1; b < boxes.size(); b++) {
                if ((boxes[a] & boxes[b]).area() > 0) {
                    boxes[a] |= boxes[b];
                    boxes.erase(boxes.begin() + b);
                    merged = true;
                    break;
                }
            }
        }
    }
    return boxes;
}

/// Refines one region: returns handwriting and overlap (print under pen) at full resolution.
inline void RefineRoi(const OpticalDensity &d, const cv::Mat &coarse, const cv::Mat &fragments, const cv::Mat &mixed, const PenColor &pen, int k,
                      cv::Mat *outHw, cv::Mat *outOverlap, cv::Mat *outPrintStrong) {
    cv::Mat zone;
    cv::dilate(coarse, zone, Ellipse(4 * k + 1));

    // own colour: 3x3 window only (larger windows smear pen colour onto adjacent print)
    cv::Mat roiChroma[3];
    InkChroma(d, kInkOD, kClippedOD, 3, 0.3f, roiChroma);
    cv::Mat ratio = PenScore(roiChroma, pen.ref, pen.dir, pen.scale);
    cv::Mat printRef(ratio.size(), CV_32F, cv::Scalar(1.0f));
    cv::Mat core;
    cv::erode(coarse, core, cv::Mat::ones(3, 3, CV_8U));
    std::vector<float> penSamples;
    std::vector<float> refSamples;
    for (int y = 0; y < ratio.rows; y++) {
        const float *r = ratio.ptr<float>(y), *rf = printRef.ptr<float>(y);
        const uchar *cr = core.ptr<uchar>(y), *ink = d.ink.ptr<uchar>(y);
        for (int x = 0; x < ratio.cols; x++) {
            if (cr[x] && ink[x] && !std::isnan(r[x])) penSamples.push_back(r[x]);
            if ((x & 7) == 0 && (y & 7) == 0) refSamples.push_back(rf[x]);
        }
    }
    float penRef = penSamples.size() > 50 ? Percentile(penSamples, 50) : Percentile(refSamples, 50) - 0.1f;

    cv::Mat ownPen(ratio.size(), CV_8U, cv::Scalar(0)), ownPrint(ratio.size(), CV_8U, cv::Scalar(0)), undecided(ratio.size(), CV_8U, cv::Scalar(0));
    // confidently print-coloured: as close to the local print colour as print itself is
    cv::Mat printStrong(ratio.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < ratio.rows; y++) {
        const float *r = ratio.ptr<float>(y), *rf = printRef.ptr<float>(y);
        const uchar *ink = d.ink.ptr<uchar>(y);
        uchar *op = ownPen.ptr<uchar>(y), *opr = ownPrint.ptr<uchar>(y), *un = undecided.ptr<uchar>(y), *ps = printStrong.ptr<uchar>(y);
        for (int x = 0; x < ratio.cols; x++) {
            if (!ink[x]) continue;
            if (std::isnan(r[x])) { un[x] = 255; continue; }
            float thr = (penRef + rf[x]) / 2;
            (r[x] < thr ? op[x] : opr[x]) = 255;
            if (r[x] >= rf[x] - 0.02f) ps[x] = 255;
        }
    }
    // sensor-clipped cores (no colour of their own) are judged as whole blobs: a blob reaching
    // well outside the coarse handwriting is a print stroke's body the pen merely touches/crosses
    {
        Components cc = FindComponents(undecided);
        std::vector<float> inside = FractionPerComponent(cc, coarse);
        std::vector<char> blobPrint(cc.count, 0);
        for (int i = 1; i < cc.count; i++) blobPrint[i] = (1.0f - inside[i]) >= 0.3f;
        printStrong |= PaintComponents(cc.labels, blobPrint);
    }

    // 4-direction arbitration on 3 px bands at several lengths
    // "print runs through" counts CONFIDENT print only — pen pixels that merely look slightly
    // print-coloured on a small scan must not turn a hand-drawn underline into an overlap
    cv::Mat inkF, penF, printF;
    d.ink.convertTo(inkF, CV_32F, 1.0 / 255.0);
    ownPen.convertTo(penF, CV_32F, 1.0 / 255.0);
    printStrong.convertTo(printF, CV_32F, 1.0 / 255.0);
    cv::Mat printThrough = cv::Mat::zeros(ratio.size(), CV_8U), penThrough = cv::Mat::zeros(ratio.size(), CV_8U);
    cv::Mat penVotes = cv::Mat::zeros(ratio.size(), CV_8U);
    bool first = true;
    for (int m : kDirLengths) {
        int L = (m * k) | 1;
        for (const cv::Mat &kernel : DirectionKernels(L, kDirBand)) {
            float full = (float)cv::sum(kernel)[0];
            cv::Mat n = Filter(inkF, kernel), nn;
            cv::max(n, 1.0, nn);
            cv::Mat pen = Filter(penF, kernel) / nn, pr = Filter(printF, kernel) / nn;
            cv::Mat runs = n >= 0.6f * full / kDirBand;
            printThrough |= (pr >= 0.6f) & runs;
            penThrough |= (pen >= 0.6f) & runs;
            if (first) {
                cv::Mat vote = pen >= 0.5f;
                cv::add(penVotes, vote / 255, penVotes);
            }
        }
        first = false;
    }
    cv::Mat penMost = penVotes >= 3;
    // the working-size (coarse) result is the base — per-pixel colour alone is unreliable on
    // small scans (measured: a 1312 px scan lost most pen pixels when colour alone decided).
    // Pixels leave it only where a confident print stroke runs through and pen does not dominate
    // (including clipped pixels, which only the directions can judge); carved pixels are never
    // re-added.
    cv::Mat carve = printThrough & ~penMost & (printStrong | (undecided & ~penThrough));
    // pieces the coarse pass placed by context (fragments) and small detached pieces of the coarse
    // handwriting are never carved back out on their own colour
    {
        Components cc = FindComponents(coarse);
        std::vector<char> smallPiece(cc.count, 0);
        for (int i = 1; i < cc.count; i++) smallPiece[i] = cc.area[i] <= (4 * k) * (4 * k);
        carve &= ~(PaintComponents(cc.labels, smallPiece) | fragments);
    }
    cv::Mat added = ((ownPen | undecided) & penThrough) | (ownPrint & penMost & ~printThrough);
    cv::Mat hw = zone & d.ink & ((coarse & d.ink) | added) & ~carve;
    // wide-context vote (≈6 stroke widths): a pen stroke's darkest/clipped parts look like print
    // up close, so an ink pixel whose wide surroundings are mostly handwriting — and with no REAL
    // print nearby (print well away from the coarse pen strokes) — is handwriting
    {
        int W = 6 * k + 1;
        cv::Mat coarseNear;
        cv::dilate(coarse, coarseNear, Ellipse(2 * k + 1));
        cv::Mat printFar = printStrong & ~coarseNear;
        cv::Mat inkF2, farF, inkSum, farSum;
        d.ink.convertTo(inkF2, CV_32F, 1.0 / 255.0);
        printFar.convertTo(farF, CV_32F, 1.0 / 255.0);
        cv::boxFilter(inkF2, inkSum, -1, cv::Size(W, W), cv::Point(-1, -1), false);
        cv::boxFilter(farF, farSum, -1, cv::Size(W, W), cv::Point(-1, -1), false);
        cv::max(inkSum, 1.0, inkSum);
        cv::Mat farShare = farSum / inkSum;
        for (int pass = 0; pass < 2; pass++) {
            cv::Mat hwF, hwSum;
            cv::Mat hi = hw & d.ink;
            hi.convertTo(hwF, CV_32F, 1.0 / 255.0);
            cv::boxFilter(hwF, hwSum, -1, cv::Size(W, W), cv::Point(-1, -1), false);
            cv::Mat share = hwSum / inkSum;
            // and enough pen mass around (a stray speck on a printed letter must not spread)
            cv::Mat enoughPen = hwSum >= (float)(6 * k * k);
            hw |= zone & d.ink & (share >= 0.55f) & (farShare < 0.15f) & enoughPen;
        }
        printStrong &= ~hw;
    }
    // pen stroke merged into printed letters: pen-coloured pixels that run along a pen stroke
    {
        cv::Mat mixedZone;
        cv::dilate(mixed, mixedZone, Ellipse(2 * k + 1));
        hw |= mixedZone & ownPen & penThrough & ~printThrough;
    }

    // overlap: print runs through AND much darker than the pen around it
    cv::Mat penW, sumOD, sumW, penLocal;
    ownPen.convertTo(penW, CV_32F, 1.0 / 255.0);
    int win = 4 * k + 1;
    cv::boxFilter(d.mean.mul(penW), sumOD, -1, cv::Size(win, win), cv::Point(-1, -1), false);
    cv::boxFilter(penW, sumW, -1, cv::Size(win, win), cv::Point(-1, -1), false);
    cv::max(sumW, 1e-6, sumW);
    cv::divide(sumOD, sumW, penLocal);
    // overlap: much darker than the pen around it AND print continues on both sides beyond the
    // pen in some direction (a hand-drawn underline has pen, not print, at both ends)
    cv::Mat overlap;
    {
        cv::Mat pvF;
        cv::Mat printVisible = printStrong & ~hw;
        printVisible.convertTo(pvF, CV_32F, 1.0 / 255.0);
        std::vector<cv::Mat> a = DirectionKernels(4 * k + 1, kDirBand, -1), b = DirectionKernels(4 * k + 1, kDirBand, +1);
        cv::Mat both = cv::Mat::zeros(hw.size(), CV_8U);
        for (int dir = 0; dir < 4; dir++) both |= (Filter(pvF, a[dir]) >= 1.5f) & (Filter(pvF, b[dir]) >= 1.5f);
        overlap = hw & both & (d.mean > penLocal * kOverlapDarker);
    }

    // print hidden under a pen stroke that crosses it: print on BOTH sides of the pixel,
    // perpendicular to the local pen direction
    {
        cv::Mat hwF, printVisF;
        hw.convertTo(hwF, CV_32F, 1.0 / 255.0);
        cv::Mat printVis = printStrong & ~hw;
        printVis.convertTo(printVisF, CV_32F, 1.0 / 255.0);
        std::vector<cv::Mat> penScore;
        int L = (2 * k) | 1;
        for (const cv::Mat &kernel : DirectionKernels(L, kDirBand)) penScore.push_back(Filter(hwF, kernel));
        int reach = std::max(3, k);
        std::vector<cv::Mat> sideA = DirectionKernels(2 * reach + 1, kDirBand, -1), sideB = DirectionKernels(2 * reach + 1, kDirBand, +1);
        cv::Mat bridge = cv::Mat::zeros(hw.size(), CV_8U);
        for (int dir = 0; dir < 4; dir++) {
            cv::Mat both = (Filter(printVisF, sideA[dir]) >= 1.5f) & (Filter(printVisF, sideB[dir]) >= 1.5f);
            // pen direction here = the direction with the most pen; bridge only across it
            int penDir = kPerpendicular[dir];
            cv::Mat isPenDir = cv::Mat::ones(hw.size(), CV_8U) * 255;
            for (int o = 0; o < 4; o++) if (o != penDir) isPenDir &= penScore[penDir] >= penScore[o];
            bridge |= both & isPenDir;
        }
        overlap |= bridge & hw;
    }
    *outHw = hw;
    *outOverlap = overlap;
    *outPrintStrong = printStrong;
}

/// Detects handwriting, the print layer to protect and the overlap (print under pen) to restore,
/// all at full resolution: layout/colour context at working size, then per-pixel refinement.
inline void DetectByInkColor(const cv::Mat &full, double colorDelta, cv::Mat *outHandwriting, cv::Mat *outPrint, cv::Mat *outOverlap) {
    cv::Mat work = ToWorkingSize(full);
    cv::Mat coarseHw, coarsePrint, fragWork, mixedWork;
    PenColor penWork;
    DetectAtWorkingSize(work, colorDelta, &coarseHw, &coarsePrint, &penWork, &fragWork, &mixedWork);
    cv::Mat coarse = ScaleMaskTo(coarseHw, full.size());
    cv::Mat fragments = ScaleMaskTo(fragWork, full.size());
    cv::Mat mixed = ScaleMaskTo(mixedWork, full.size());
    cv::Mat print = ScaleMaskTo(coarsePrint, full.size());

    const int k = StrokeUnit(full);
    cv::Mat hw = cv::Mat::zeros(full.size(), CV_8U), overlap = cv::Mat::zeros(full.size(), CV_8U);
    cv::Mat paperSmall = PaperLowRes(full);
    for (const cv::Rect &roi : RegionsOf(coarse | mixed, 8 * k)) {
        OpticalDensity d = DensityInRoi(full, paperSmall, roi);
        // print-colour reference for this region, from the working-size maps
        cv::Rect workRoi((int)(roi.x * (double)work.cols / full.cols), (int)(roi.y * (double)work.rows / full.rows), 0, 0);
        workRoi.width = std::max(1, std::min(work.cols, (int)std::ceil((roi.x + roi.width) * (double)work.cols / full.cols)) - workRoi.x);
        workRoi.height = std::max(1, std::min(work.rows, (int)std::ceil((roi.y + roi.height) * (double)work.rows / full.rows)) - workRoi.y);
        PenColor penRoi;
        penRoi.dir = penWork.dir;
        penRoi.scale = penWork.scale;
        for (int ch = 0; ch < 3; ch++) cv::resize(penWork.ref[ch](workRoi), penRoi.ref[ch], roi.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat roiHw, roiOverlap, roiPrintStrong;
        RefineRoi(d, coarse(roi), fragments(roi), mixed(roi), penRoi, k, &roiHw, &roiOverlap, &roiPrintStrong);
        roiHw.copyTo(hw(roi));
        roiOverlap.copyTo(overlap(roi));
        // inside the refined region only CONFIDENT print is protected: pen pixels merely left out
        // of the handwriting must stay erasable (protecting "all other ink" turned every missed pen
        // pixel into permanent print — the ghost strokes seen on a real scan)
        // the working-size print layer is dilated; right next to the pen that dilation is the pen's
        // own rim, so there only pixel-level confident print counts
        cv::Mat hwNear;
        cv::dilate(roiHw, hwNear, Ellipse(2 * k + 1));
        cv::Mat regionPrint = (((print(roi) & ~hwNear) | roiPrintStrong) & ~roiHw) | roiOverlap;
        cv::Mat zone;
        cv::dilate(coarse(roi) | mixed(roi), zone, Ellipse(4 * k + 1));
        regionPrint.copyTo(print(roi), zone);
    }
    // stray pen specks sitting on printed text: a tiny handwriting blob whose surroundings are
    // mostly non-handwriting ink is a colour-noise false positive — and the erase's rim growth
    // around it would wipe the printed letter — so drop it (real accents/colons sit next to
    // handwriting or on bare paper, never inside a block of print)
    {
        cv::Mat paperSmallAll = PaperLowRes(full);
        OpticalDensity dAll = DensityInRoi(full, paperSmallAll, cv::Rect(0, 0, full.cols, full.rows));
        int W = 6 * k + 1;
        cv::Mat nonHw = dAll.ink & ~hw, nF, iF, nS, iS;
        nonHw.convertTo(nF, CV_32F, 1.0 / 255.0);
        dAll.ink.convertTo(iF, CV_32F, 1.0 / 255.0);
        cv::boxFilter(nF, nS, -1, cv::Size(W, W), cv::Point(-1, -1), false);
        cv::boxFilter(iF, iS, -1, cv::Size(W, W), cv::Point(-1, -1), false);
        cv::max(iS, 1.0, iS);
        cv::Mat printAround = (nS / iS) >= 0.5f;
        Components ch = FindComponents(hw);
        std::vector<float> onPrint = FractionPerComponent(ch, printAround);
        // pieces confirmed by context (fragments, a pen underline fused to a printed word) stay
        std::vector<float> confirmed = FractionPerComponent(ch, fragments);
        std::vector<char> drop(ch.count, 0);
        for (int i = 1; i < ch.count; i++) drop[i] = ch.area[i] <= (4 * k) * (4 * k) && onPrint[i] >= 0.5f && confirmed[i] < 0.3f;
        cv::Mat dropMask = PaintComponents(ch.labels, drop);
        hw &= ~dropMask;
        overlap &= ~dropMask;
    }
    *outHandwriting = hw;
    *outPrint = print;
    *outOverlap = overlap;
}

// MARK: - Erase

/// Removes `handwriting` from `target` by rebuilding, not blurring: overlap pixels (print under the
/// pen) get the nearby print colour back, everything else the stroke, its faint rim and the camera
/// sharpening halo cover is inpainted from the PAPER only (print next to the stroke is swapped for
/// paper colour during inpainting so it cannot bleed in, then put back). `analysis` is the
/// unprocessed page the masks were computed on (same geometry).
inline cv::Mat EraseHandwriting(const cv::Mat &target, const cv::Mat &analysis, const cv::Mat &handwriting, const cv::Mat &print, const cv::Mat &overlap) {
    const int k = StrokeUnit(target);
    const double s = k / 9.0;   // the window sizes below were measured at k = 9 (a 4032 px photo)
    cv::Mat dst = target.clone();
    cv::Mat paperSmall = PaperLowRes(analysis);
    cv::Mat targetPaperSmall = PaperLowRes(target);
    cv::Mat targetSmall;
    cv::resize(target, targetSmall, cv::Size(), 0.125, 0.125, cv::INTER_AREA);
    cv::medianBlur(targetSmall, targetSmall, 21);
    int margin = Odd(61 * s) / 2 + Odd(13 * s) + Odd(11 * s) + 4;

    for (const cv::Rect &roi : RegionsOf(handwriting, margin)) {
        OpticalDensity d = DensityInRoi(analysis, paperSmall, roi);
        // the target is sharpened: stroke rims that were faint in the analysis page are dark
        // there, so the rim is judged on BOTH pages
        OpticalDensity dt = DensityInRoi(target, targetPaperSmall, roi);
        cv::Mat hw = handwriting(roi), pr = print(roi), ov = overlap(roi);
        cv::Mat printNear, grown, halo;
        cv::dilate(pr, printNear, cv::Mat::ones(3, 3, CV_8U));
        cv::dilate(hw, grown, Ellipse(13 * s));
        cv::Mat rim = (d.mean > kRimOD) | (dt.mean > kRimOD);
        cv::Mat area = hw | (grown & rim & ~printNear);
        cv::dilate(area, halo, Ellipse(11 * s));
        area |= halo & ~printNear;
        cv::Mat restoreMask = ov.clone();
        // printed rules (long, thin, straight print runs) crossing the erased area are re-joined:
        // close short gaps along the line, keep runs that are long AND thin (text lines are tall)
        {
            // seed from print well away from the pen: the print layer is dilated and hugs pen rims,
            // which a long closing would otherwise turn into fake dashed "rules" through the text
            cv::Mat hwFar;
            cv::dilate(hw, hwFar, Ellipse(3 * k + 1));
            cv::Mat prInk = pr & ~hwFar & d.ink, closed, runs, thick, thickNear;
            int l1 = std::max(3, 6 * k) | 1, l2 = std::max(5, 20 * k) | 1;
            cv::morphologyEx(prInk, closed, cv::MORPH_CLOSE, cv::Mat::ones(1, l1, CV_8U));
            cv::morphologyEx(closed, runs, cv::MORPH_OPEN, cv::Mat::ones(1, l2, CV_8U));
            cv::morphologyEx(runs, thick, cv::MORPH_OPEN, cv::Mat::ones(std::max(3, k) | 1, 1, CV_8U));
            cv::dilate(thick, thickNear, cv::Mat::ones(3, 3, CV_8U));
            restoreMask |= runs & ~thickNear & area;
        }
        // leftover pen specks: small ink bits right beside the erased strokes that are not
        // confident print belong to the handwriting too
        {
            cv::Mat nearHw;
            cv::dilate(hw, nearHw, Ellipse(4 * k + 1));
            cv::Mat leftover = (d.mean > 0.2f) & nearHw & ~pr & ~hw;
            Components cl = FindComponents(leftover);
            std::vector<char> small(cl.count, 0);
            for (int i = 1; i < cl.count; i++) small[i] = cl.area[i] <= (2 * k) * (2 * k);
            area |= PaintComponents(cl.labels, small);
        }
        cv::Mat restore = area & restoreMask;
        cv::Mat paperArea = area & ~restore;

        // local print colour: average of nearby visible print
        cv::Mat strong = pr & ~hw & d.ink;
        cv::Mat weight, tf, weighted, num, den;
        strong.convertTo(weight, CV_32F, 1.0 / 255.0);
        target(roi).convertTo(tf, CV_32FC3);
        cv::Mat weight3;
        cv::merge(std::vector<cv::Mat>{weight, weight, weight}, weight3);
        cv::multiply(tf, weight3, weighted);
        int win = Odd(61 * s);
        cv::boxFilter(weighted, num, -1, cv::Size(win, win));
        cv::boxFilter(weight, den, -1, cv::Size(win, win));
        cv::Mat empty = den <= 1e-3f;
        cv::max(den, 1e-3, den);
        cv::Mat den3, local;
        cv::merge(std::vector<cv::Mat>{den, den, den}, den3);
        cv::divide(num, den3, local);
        cv::Scalar fallback = cv::countNonZero(strong) > 0 ? cv::mean(target(roi), strong) : cv::Scalar(40, 40, 40);
        local.setTo(fallback, empty);
        local.convertTo(local, CV_8UC3);

        cv::Mat out = dst(roi).clone();
        local.copyTo(out, restore);
        // inpaint from paper only
        cv::Mat paperColor;
        cv::Rect sr((int)(roi.x / 8.0), (int)(roi.y / 8.0), 0, 0);
        sr.width = std::max(1, std::min(targetSmall.cols, (int)std::ceil((roi.x + roi.width) / 8.0)) - sr.x);
        sr.height = std::max(1, std::min(targetSmall.rows, (int)std::ceil((roi.y + roi.height) / 8.0)) - sr.y);
        cv::resize(targetSmall(sr), paperColor, roi.size(), 0, 0, cv::INTER_LINEAR);
        cv::GaussianBlur(paperColor, paperColor, cv::Size(0, 0), 8);
        cv::Mat nearArea;
        cv::dilate(area, nearArea, Ellipse(9 * s));
        cv::Mat keep = (printNear & nearArea & ~area) | restore;
        cv::Mat tmp = out.clone();
        paperColor.copyTo(tmp, keep);
        cv::Mat filled;
        cv::inpaint(tmp, paperArea, filled, 3, cv::INPAINT_TELEA);
        out.copyTo(filled, keep);
        filled.copyTo(dst(roi));
    }
    return dst;
}

}  // namespace inkanalysis
