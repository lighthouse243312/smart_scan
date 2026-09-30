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
#include <cfloat>
#include <climits>
#include <cmath>
#include <array>
#include <map>
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

/// Handwriting with NO colour of its own (black ballpoint, or a scan app that made the page almost
/// greyscale), where the pen colour says nothing. Found by LAYOUT and SHAPE, verified by the model:
///  - not part of a regular printed line (and not beside one, within a letter gap, in its band —
///    descenders, list numbers);
///  - not a rule (long straight runs are removed first) and glyph-sized;
///  - its word/number group mostly has no twin among the page's regular printed glyphs (a
///    typeset page repeats its glyphs almost exactly; a hand never does — measured: handwritten
///    digits <= 0.56 IoU to their best twin, printed letters mostly >= 0.74);
///  - not faint (a light grey watermark or footer), and
///  - the segmentation model sees at least a trace of handwriting there (it gives printed text
///    exactly 0.00 — headers, bold words, list numbers included — and colourless handwriting
///    0.06..0.6 even where it is not sure enough to mark it itself).
inline cv::Mat ColourlessHandwriting(const OpticalDensity &d, const Components &c, const std::vector<char> &regular,
                                     const cv::Mat &hw, const cv::Mat &modelHw) {
    const int H = d.ink.rows, W = d.ink.cols;
    cv::Mat found = cv::Mat::zeros(H, W, CV_8U);
    if (modelHw.empty()) return found;
    std::vector<int> regIds;
    std::vector<float> hs;
    for (int i = 1; i < c.count; i++) if (regular[i]) { regIds.push_back(i); hs.push_back((float)c.h[i]); }
    if (regIds.size() < 20) return found;
    std::nth_element(hs.begin(), hs.begin() + hs.size() / 2, hs.end());
    const float cap = hs[hs.size() / 2];

    // regular printed lines, split into runs at gaps wider than a word gap (table cells)
    struct Run { float x0, x1; };
    struct RLine { std::vector<int> m; float cy, top, bot; std::vector<Run> runs; };
    std::vector<int> byCy = regIds;
    std::sort(byCy.begin(), byCy.end(), [&](int a, int b) { return c.y[a] + c.h[a] / 2.0f < c.y[b] + c.h[b] / 2.0f; });
    std::vector<RLine> lines;
    auto med = [](std::vector<float> v) { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
    for (int i : byCy) {
        float cy = c.y[i] + c.h[i] / 2.0f;
        bool joined = false;
        for (RLine &L : lines) {
            if (std::fabs(cy - L.cy) < 0.5f * cap) {
                L.m.push_back(i);
                std::vector<float> cys;
                for (int j : L.m) cys.push_back(c.y[j] + c.h[j] / 2.0f);
                L.cy = med(cys);
                joined = true;
                break;
            }
        }
        if (!joined) { RLine L; L.m.push_back(i); L.cy = cy; lines.push_back(L); }
    }
    for (RLine &L : lines) {
        std::vector<std::pair<int, int>> xs;
        std::vector<float> tops, bots;
        for (int j : L.m) { xs.push_back({c.x[j], c.x[j] + c.w[j]}); tops.push_back((float)c.y[j]); bots.push_back((float)(c.y[j] + c.h[j])); }
        std::sort(xs.begin(), xs.end());
        float a = (float)xs[0].first, b = (float)xs[0].second;
        for (size_t q = 1; q < xs.size(); q++) {
            if (xs[q].first - b > 1.5f * cap) { L.runs.push_back({a, b}); a = (float)xs[q].first; b = (float)xs[q].second; }
            else b = std::max(b, (float)xs[q].second);
        }
        L.runs.push_back({a, b});
        L.top = med(tops); L.bot = med(bots);
    }

    // rules / table grid: long straight runs
    const int Lr = std::max(9, (int)(3 * cap)) | 1;
    cv::Mat hRuns, vRuns, rules;
    cv::morphologyEx(d.ink, hRuns, cv::MORPH_OPEN, cv::Mat::ones(1, Lr, CV_8U));
    cv::morphologyEx(d.ink, vRuns, cv::MORPH_OPEN, cv::Mat::ones(Lr, 1, CV_8U));
    cv::dilate(hRuns | vRuns, rules, cv::Mat::ones(3, 3, CV_8U));
    cv::Mat regularMask = PaintComponents(c.labels, regular);
    cv::Mat candMask = d.ink & ~rules & ~regularMask;

    Components cc = FindComponents(candMask);

    // printed darkness (a faint watermark is lighter than print and pen alike)
    float printDark, coreDark;
    {
        std::vector<float> v;
        for (int y = 0; y < H; y += 2) {
            const uchar *r = regularMask.ptr<uchar>(y);
            const float *m = d.mean.ptr<float>(y);
            for (int x = 0; x < W; x += 2) if (r[x]) v.push_back(m[x]);
        }
        printDark = v.empty() ? 1.0f : med(v);
        // the toner's core (80th percentile): printed strokes reach it, most pens do not
        coreDark = v.empty() ? 2.0f : Percentile(v, 80);
    }

    struct Cand { int i, x, y, w, h; float twin, dark, model; int modelN; };
    std::vector<Cand> cands;
    for (int i = 1; i < cc.count; i++) {
        const int x = cc.x[i], y = cc.y[i], w = cc.w[i], h = cc.h[i];
        if (cc.area[i] < 0.3f * cap || h < 0.5f * cap || h > 3.5f * cap || w > 8 * cap) continue;
        cv::Rect r(x, y, w, h);
        cv::Mat B = cc.labels(r) == i;
        // (pieces the coarse pass already calls handwriting as a whole are candidates too: the
        // full-resolution refinement may carve a colourless pen's piece back out — the "1" of a
        // handwritten 1800; a piece only PARTLY handwriting is a pen stroke fused with a printed
        // letter, which the colour passes split pixel by pixel — taking it whole erased the "T")
        bool partlyHw;
        {
            const float hwShare = (float)cv::countNonZero(B & hw(r)) / std::max(1, cc.area[i]);
            partlyHw = hwShare > 0.05f && hwShare < 0.9f;
        }
        const float cy = y + h / 2.0f;
        bool inLine = false;
        for (const RLine &L : lines) {
            if (cy < L.top - 0.35f * cap || cy > L.bot + 0.35f * cap || h > 1.7f * (L.bot - L.top + 1)) continue;
            for (const Run &rn : L.runs)
                if (rn.x0 - 0.7f * cap <= x + w && x <= rn.x1 + 0.7f * cap) { inLine = true; break; }
            if (inLine) break;
        }
        if (inLine) continue;
        // best twin among the regular printed glyphs of about the same size
        auto twinOf = [&](const cv::Mat &S) {
            cv::Mat rowsM, colsM;
            cv::reduce(S, rowsM, 1, cv::REDUCE_MAX);
            cv::reduce(S, colsM, 0, cv::REDUCE_MAX);
            int y0 = -1, y1 = -1, x0 = -1, x1 = -1;
            for (int y = 0; y < rowsM.rows; y++) if (rowsM.at<uchar>(y)) { if (y0 < 0) y0 = y; y1 = y; }
            for (int x = 0; x < colsM.cols; x++) if (colsM.at<uchar>(0, x)) { if (x0 < 0) x0 = x; x1 = x; }
            cv::Rect bb = y0 < 0 ? cv::Rect() : cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
            if (bb.area() == 0) return 0.0f;
            cv::Mat Sb = S(bb);
            float best = 0;
            for (int j : regIds) {
                const float rh = (float)c.h[j] / bb.height, rw = (float)c.w[j] / bb.width;
                if (rh < 0.85f || rh > 1.18f || rw < 0.8f || rw > 1.25f) continue;
                cv::Mat G = c.labels(cv::Rect(c.x[j], c.y[j], c.w[j], c.h[j])) == j, Gs;
                cv::resize(G, Gs, bb.size(), 0, 0, cv::INTER_NEAREST);
                int inter = cv::countNonZero(Gs & Sb), uni = cv::countNonZero(Gs | Sb);
                if (uni) best = std::max(best, (float)inter / uni);
                if (best >= 0.95f) break;
            }
            return best;
        };
        // a piece only PARTLY handwriting may be a pen stroke fused with a printed letter (the colour
        // passes split it pixel by pixel): skipped unless its non-handwriting part is pen too
        // (the colour cannot tell here, nor the shape — the printed letter is deformed by the pen
        // fused to it; the model can: it gives the printed letter under a pen 0.1-0.3 handwriting,
        // a colourless pen's own strokes 0.5-0.9)
        if (partlyHw) {
            cv::Mat rest = B & ~hw(r);
            // (only beside printed text: a letter fused with a pen stroke sits in a word; an answer
            // digit alone in its cell touching the cell's rule has no printed neighbour)
            bool besideText = false;
            for (int j : regIds) {
                int gap = std::max(c.x[j], x) - std::min(c.x[j] + c.w[j], x + w);
                int oy = std::min(c.y[j] + c.h[j], y + h) - std::max(c.y[j], y);
                if (gap <= 0.7f * cap && oy > 0.3f * std::min(c.h[j], h)) { besideText = true; break; }
            }
            if (besideText && cv::countNonZero(rest) && cv::mean(modelHw(r), rest)[0] < 0.4) continue;
        }
        float best = twinOf(B);
        float dark = (float)cv::mean(d.mean(r), B)[0];
        float model = (float)cv::mean(modelHw(r), B)[0];
        cands.push_back({i, x, y, w, h, best, dark, model, cv::countNonZero(B)});
    }
    // groups: candidates side by side on a row (a word, a number, a formula) decide together
    UnionFind uf((int)cands.size());
    for (size_t a = 0; a < cands.size(); a++)
        for (size_t b = a + 1; b < cands.size(); b++) {
            const Cand &A = cands[a], &Bc = cands[b];
            int gapx = std::max(A.x, Bc.x) - std::min(A.x + A.w, Bc.x + Bc.w);
            int oy = std::min(A.y + A.h, Bc.y + Bc.h) - std::max(A.y, Bc.y);
            if (gapx <= 0.8f * cap && oy > 0.3f * std::min(A.h, Bc.h)) uf.Join((int)a, (int)b);
        }
    std::map<int, std::vector<int>> groups;
    for (size_t a = 0; a < cands.size(); a++) groups[uf.Find((int)a)].push_back((int)a);
    for (auto &kv : groups) {
        const std::vector<int> &G = kv.second;
        int twins = 0;
        float maxTwin = 0;
        double modelSum = 0, modelN = 0;
        std::vector<float> darks;
        for (int a : G) {
            twins += cands[a].twin >= 0.7f;
            maxTwin = std::max(maxTwin, cands[a].twin);
            darks.push_back(cands[a].dark);
            modelSum += cands[a].model * cands[a].modelN; modelN += cands[a].modelN;
        }
#ifdef INK_DEBUG
        for (int a : G) fprintf(stderr, "cl-cand x=%d y=%d w=%d h=%d twin=%.2f dark=%.2f model=%.3f group=%d\n", cands[a].x, cands[a].y, cands[a].w, cands[a].h, cands[a].twin, cands[a].dark, cands[a].model, kv.first);
        fprintf(stderr, "cl-group %d n=%zu twins=%d darkMed=%.2f printDark=%.2f model=%.3f\n", kv.first, G.size(), twins, med(darks), printDark, modelN ? modelSum / modelN : -1.0);
#endif
        if (twins >= 0.5f * G.size()) continue;                  // typeset: repeats the page's glyphs
        // faint: a light watermark / footer (measured 0.32 vs 1.34 for print; a light pen 0.69)
        if (med(darks) < 0.4f * printDark) continue;
        // the model sees no handwriting at all: still handwriting when nothing about it is typeset —
        // no glyph with even a loose twin (printed headers / bold list numbers in another font: 0.57+,
        // handwritten digits <= 0.51) and not toner-dark
        if (modelN == 0 || modelSum / modelN < 0.03) {
            // (several glyphs side by side — a written number or word; a lone twinless glyph is as
            // likely a printed icon or accent — and clearly lighter than the toner core)
            float meanDark = 0;
            for (float v : darks) meanDark += v;
            meanDark /= darks.size();
            if (G.size() < 2 || maxTwin >= 0.55f || meanDark >= 0.8f * coreDark) continue;
        }
        for (int a : G) {
            const Cand &q = cands[a];
            cv::Rect r(q.x, q.y, q.w, q.h);
            found(r) |= cc.labels(r) == q.i;
        }
    }
#ifdef INK_DEBUG
    fprintf(stderr, "colourless handwriting: cap %.1f candidates %zu groups %zu found %d px\n", cap, cands.size(), groups.size(), cv::countNonZero(found));
#endif
    return found;
}

inline void DetectAtWorkingSize(const cv::Mat &src, double colorDelta, cv::Mat *outHandwriting, cv::Mat *outPrint, PenColor *outPen, cv::Mat *outFragments, cv::Mat *outMixed, cv::Mat *outPrintBand, cv::Mat *outEnclosedPrint = nullptr,
                                const cv::Mat &modelHw = cv::Mat(), cv::Mat *outColourless = nullptr) {
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
    // uniformity review: a typeset page repeats its glyphs almost exactly, a hand never does — a
    // "regular line" whose glyphs have no twin in any OTHER line is neat handwriting that happened to
    // be even and straight (a formula written between the printed lines), not print
    {
        std::vector<int> lineOf(c.count, -1);
        for (size_t li = 0; li < lines.size(); li++) for (int i : lines[li].members) lineOf[i] = (int)li;
        std::vector<int> all;
        for (int i = 1; i < c.count; i++) if (lineOf[i] >= 0) all.push_back(i);
        std::sort(all.begin(), all.end(), [&](int a, int b) { return c.h[a] < c.h[b]; });
        std::vector<cv::Mat> shapes(c.count);
        auto shape = [&](int i) -> const cv::Mat & {
            if (shapes[i].empty()) shapes[i] = c.labels(cv::Rect(c.x[i], c.y[i], c.w[i], c.h[i])) == i;
            return shapes[i];
        };
        std::vector<char> keep(lines.size(), 1);
        for (size_t li = 0; li < lines.size(); li++) {
            if (lines[li].members.size() > 40) continue;   // long lines are print (handwriting rows are short)
            int withTwin = 0, tested = 0;
            for (int i : lines[li].members) {
                if (c.h[i] < 4) continue;
                tested++;
                const cv::Mat &B = shape(i);
                bool twin = false;
                auto lo = std::lower_bound(all.begin(), all.end(), (int)std::floor(c.h[i] * 0.88f), [&](int a, int v) { return c.h[a] < v; });
                for (auto it = lo; it != all.end() && c.h[*it] <= c.h[i] * 1.14f && !twin; ++it) {
                    int j = *it;
                    if (lineOf[j] == (int)li) continue;
                    float rw = (float)c.w[j] / c.w[i];
                    if (rw < 0.85f || rw > 1.18f) continue;
                    cv::Mat G;
                    cv::resize(shape(j), G, B.size(), 0, 0, cv::INTER_NEAREST);
                    int inter = cv::countNonZero(G & B), uni = cv::countNonZero(G | B);
                    twin = uni > 0 && inter >= 0.7f * uni;
                }
                withTwin += twin;
            }
            if (tested >= 3 && withTwin < 0.3f * tested) keep[li] = 0;
#ifdef INK_DEBUG
            if (!keep[li]) fprintf(stderr, "untwinned regular line dropped: %zu members, %d/%d with twin, at %d,%d\n", lines[li].members.size(), withTwin, tested, c.x[lines[li].members[0]], c.y[lines[li].members[0]]);
#endif
        }
        std::vector<TextLine> kept;
        for (size_t li = 0; li < lines.size(); li++) if (keep[li]) kept.push_back(lines[li]);
        lines.swap(kept);
    }
    std::vector<char> regular(c.count, 0);
    for (auto &l : lines) for (int i : l.members) regular[i] = 1;
    cv::Mat regularMask = PaintComponents(c.labels, regular);
#ifdef INK_DEBUG
    cv::imwrite("/tmp/ink_regular.png", regularMask);
    cv::imwrite("/tmp/ink_workink.png", d.ink);
#endif
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
    // the band each print text line occupies: where pen crosses printed words, the print-coloured
    // pixels in these bands are print to restore (see RefineRoi)
    cv::Mat printBand = cv::Mat::zeros(src.size(), CV_8U);
    std::vector<cv::Vec4f> bandRows;   // y0, y1, x0, x1 of each print line
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
        bandRows.push_back({y0 + 0.4f * l.medianHeight, y1 - 0.1f * l.medianHeight, x0, x1});
        for (int i = 1; i < c.count; i++) {
            if (small[i] && c.cx[i] >= x0 && c.cx[i] <= x1 && c.y[i] >= y0 && c.y[i] + c.h[i] <= y1) printComp[i] = 1;
        }
    }
    // Bands span the whole text row within the page's text column: a pen stroke crossing a word
    // breaks that part of the line's regularity, so the regular pieces on either side stand in
    // for it. Regular lines that failed the colour vote count too — only non-pen-coloured pixels
    // are ever restored inside a band.
    for (auto &l : lines) {
        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
        for (int i : l.members) {
            x0 = std::min(x0, (float)c.x[i]); x1 = std::max(x1, (float)(c.x[i] + c.w[i]));
            y0 = std::min(y0, (float)c.y[i]); y1 = std::max(y1, c.bottom[i]);
        }
        bandRows.push_back({y0 - 0.2f * l.medianHeight, y1 + 0.2f * l.medianHeight, x0, x1});
    }
    if (!bandRows.empty()) {
        std::vector<float> lefts, rights;
        for (auto &b : bandRows) { lefts.push_back(b[2]); rights.push_back(b[3]); }
        float colL = Percentile(lefts, 5), colR = Percentile(rights, 95);
        for (auto &b : bandRows) {
            cv::rectangle(printBand, cv::Point((int)std::min(b[2], colL), (int)b[0]), cv::Point((int)std::max(b[3], colR), (int)b[1]),
                          cv::Scalar(255), cv::FILLED);
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
    // ...or a pen LOOP around printed text (a circled item number, e.g. "(5.)The"): its
    // pen-coloured pixels form a ring that encloses the component's other ink. A printed letter
    // made noisy by colour is never a ring with other print inside it.
    cv::Mat enclosedPrint = cv::Mat::zeros(src.size(), CV_8U);
    for (int i = 1; i < c.count; i++) {
        if (mixedStrong[i] || !printComp[i] || fracC[i] < 0.4f || fracC[i] * c.area[i] < (float)(3 * k * k)) continue;
        cv::Rect b(c.x[i] - 1, c.y[i] - 1, c.w[i] + 2, c.h[i] + 2);
        b &= cv::Rect(0, 0, W, H);
        cv::Mat comp = c.labels(b) == i;
        cv::Mat ring = comp & cand(b), rest = comp & d.ink(b) & ~cand(b);
        // enclosed = pen ring on at least 3 of the 4 sides (the ring is often open where it
        // crosses the print it circles: there the pen's pixels look print-coloured)
        const int bw = b.width, bh = b.height;
        std::vector<int> rowFirst(bh, INT_MAX), rowLast(bh, -1), colFirst(bw, INT_MAX), colLast(bw, -1);
        for (int y = 0; y < bh; y++) {
            const uchar *r = ring.ptr<uchar>(y);
            for (int x = 0; x < bw; x++) if (r[x]) {
                rowFirst[y] = std::min(rowFirst[y], x); rowLast[y] = std::max(rowLast[y], x);
                colFirst[x] = std::min(colFirst[x], y); colLast[x] = std::max(colLast[x], y);
            }
        }
        int restCount = 0, enclosed = 0;
        cv::Mat inside = cv::Mat::zeros(ring.size(), CV_8U);
        for (int y = 0; y < bh; y++) {
            const uchar *r = rest.ptr<uchar>(y);
            uchar *in = inside.ptr<uchar>(y);
            for (int x = 0; x < bw; x++) {
                if (!r[x]) continue;
                restCount++;
                int sides = (rowFirst[y] < x) + (rowLast[y] > x) + (colFirst[x] < y) + (colLast[x] > y);
                if (sides >= 3) { enclosed++; in[x] = 255; }
            }
        }
        if (enclosed >= k * k && enclosed >= 0.25 * restCount) {
            mixedStrong[i] = 1;
            // the circled print itself (the item number) stays print through refinement, where
            // the ring around it dominates every neighbourhood vote
            enclosedPrint(b) |= inside;
            // (the ring itself is marked 128: pen, never to be restored as print)
            cv::Mat ringPx = comp & cand(b) & ~inside;
            enclosedPrint(b).setTo(128, ringPx);
        }
#ifdef INK_DEBUG
        if (c.x[i] <= 75 && c.x[i] + c.w[i] >= 40 && c.y[i] <= 1056 && c.y[i] + c.h[i] >= 1025) {
            fprintf(stderr, "ring: rest=%d enclosed=%d\n", restCount, enclosed);
            cv::Mat viz; cv::merge(std::vector<cv::Mat>{inside, rest, ring}, viz);
            cv::imwrite("/tmp/ink_ring.png", viz);
        }
#endif
    }
    cv::Mat mixedPen = PaintComponents(c.labels, mixedStrong) & cand;
#ifdef INK_DEBUG
    for (int i = 1; i < c.count; i++) {
        bool at5 = c.x[i] <= 75 && c.x[i] + c.w[i] >= 40 && c.y[i] <= 1056 && c.y[i] + c.h[i] >= 1025;
        bool at1 = c.x[i] <= 48 * W / 918 && c.x[i] + c.w[i] >= 48 * W / 918 && c.y[i] <= 935 * H / 1280 && c.y[i] + c.h[i] >= 935 * H / 1280;
        if (at5 || at1) fprintf(stderr, "comp%s id=%d x=%d y=%d w=%d h=%d area=%d printComp=%d fracC=%.3f mixed=%d strong=%d W=%d\n", at5 ? "5" : "1", i, c.x[i], c.y[i], c.w[i], c.h[i], c.area[i], (int)printComp[i], fracC[i], (int)mixed[i], (int)mixedStrong[i], W);
    }
#endif
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
        fragmentMask = PaintComponents(c.labels, fragment) & d.ink;
        hw |= fragmentMask | mixedPen;
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

    if (outColourless) {
        cv::Mat colourless = ColourlessHandwriting(d, c, regular, hw, modelHw) & ~picture;
        *outColourless = colourless;
        print &= ~colourless;
    }

    *outHandwriting = hw;
    *outPrint = print;
    *outPen = pen;
    *outFragments = fragmentMask & ~picture;
    *outMixed = mixedMask & ~picture;
    *outPrintBand = printBand;
    if (outEnclosedPrint) *outEnclosedPrint = enclosedPrint & ~picture;
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
#ifdef INK_DEBUG
inline cv::Rect g_dbgBox; inline cv::Point g_dbgRoiOrigin;
#endif
#ifdef INK_DEBUG
inline cv::Mat g_dbgDarker, g_dbgBridge, g_dbgToner, g_dbgAllD, g_dbgAllB, g_dbgAllT;
#endif
inline void RefineRoi(const OpticalDensity &d, const cv::Mat &coarse, const cv::Mat &fragments, const cv::Mat &mixed, const cv::Mat &printBand, const PenColor &pen, int k,
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

        // a context-placed piece that is mostly print-coloured itself is a printed letter the pen
        // touched (e.g. "Th" next to a circled number): its non-pen pixels are print
        {
            Components cf = FindComponents(fragments & d.ink);
            std::vector<float> pc = FractionPerComponent(cf, printStrong);
            std::vector<char> printPiece(cf.count, 0);
            for (int i = 1; i < cf.count; i++) printPiece[i] = pc[i] >= 0.5f;
            carve |= PaintComponents(cf.labels, printPiece) & ~ownPen;
        }
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
#ifdef INK_DEBUG
        g_dbgDarker = overlap.clone();
#endif
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
#ifdef INK_DEBUG
        g_dbgBridge = bridge & hw;
#endif
    }
    // pen crossing PRINTED TEXT: inside a print line's band, pixels of the handwriting that are
    // NOT pen-coloured (print-coloured, or too dark to have a colour — toner cores clip) are the
    // printed letters under/around the pen — restore them
    // (faint rims of the pen have no reliable colour either, so colourless pixels only count when
    // toner-dark — sensor-clipped)
    // (only toner-dark, sensor-clipped pixels: a pen's rims can look print-coloured and would stay
    // behind as ghosts of the erased answer; its core is never as dark as toner)
    // A clipped pixel has no colour of its own, so it takes the colour of the NEAREST coloured rim
    // pixel around it: a pen's core is just as dark as toner, but its rims carry the pen's hue, a
    // printed letter's rims are print-coloured. Judged per pixel, not per blob: pen strokes crossing
    // a printed word fuse its letters into one clipped blob with mixed rims (e.g. "(GYENEMERC)").
    {
        // clipped = ALL channels dark: a saturated pen (deep red: its red channel still bright) is dark
        // on average but has plenty of colour of its own and must not borrow a print rim's
        cv::Mat minOD = cv::min(cv::min(d.od[0], d.od[1]), d.od[2]);
        cv::Mat clipped = d.ink & (d.mean > kClippedOD) & (minOD > kClippedOD);
        cv::Mat nearClipped;
        cv::dilate(clipped, nearClipped, cv::Mat::ones(7, 7, CV_8U));
        cv::Mat rimPen = nearClipped & ~clipped & ownPen;
        cv::Mat rimPrint = nearClipped & ~clipped & (ownPrint | printStrong) & ~ownPen;
        // a printed RULE the pen writes on is print-coloured right beside the pen's core: it says
        // nothing about that core (ghost specks of the answer stayed on the fill-in lines)
        {
            cv::Mat ruleLike;
            cv::morphologyEx(d.ink & ~clipped, ruleLike, cv::MORPH_OPEN, cv::Mat::ones(1, std::max(5, 4 * k) | 1, CV_8U));
            cv::dilate(ruleLike, ruleLike, cv::Mat::ones(3, 1, CV_8U));
            rimPrint &= ~ruleLike;
        }
        cv::Mat rims = rimPen | rimPrint;
        cv::Mat dist, nearestIdx;
        cv::distanceTransform(~rims, dist, nearestIdx, cv::DIST_L2, 3, cv::DIST_LABEL_PIXEL);
        // DIST_LABEL_PIXEL numbers the zero pixels (the rims) in scan order from 1
        std::vector<uchar> idxIsPrint(1, 0);
        for (int y = 0; y < rims.rows; y++) {
            const uchar *r = rims.ptr<uchar>(y), *rp = rimPrint.ptr<uchar>(y);
            for (int x = 0; x < rims.cols; x++) if (r[x]) idxIsPrint.push_back(rp[x] ? 1 : 0);
        }
        cv::Mat votePrint = cv::Mat::zeros(clipped.size(), CV_32F), votePen = cv::Mat::zeros(clipped.size(), CV_32F);
        const float reach = 2.0f * k;
        for (int y = 0; y < clipped.rows; y++) {
            const uchar *cl = clipped.ptr<uchar>(y);
            const int *ni = nearestIdx.ptr<int>(y);
            const float *dd = dist.ptr<float>(y);
            float *vp = votePrint.ptr<float>(y), *vn = votePen.ptr<float>(y);
            for (int x = 0; x < clipped.cols; x++) {
                if (!cl[x] || dd[x] > reach || ni[x] <= 0 || ni[x] >= (int)idxIsPrint.size()) continue;
                (idxIsPrint[ni[x]] ? vp[x] : vn[x]) = 1.0f;
            }
        }
        // light majority smoothing, so a lone rim pixel of the wrong colour does not decide
        cv::boxFilter(votePrint, votePrint, -1, cv::Size(3, 3), cv::Point(-1, -1), false);
        cv::boxFilter(votePen, votePen, -1, cv::Size(3, 3), cv::Point(-1, -1), false);
        // ...and only where print rims outnumber pen rims in the wider neighbourhood too: inside a
        // written word a few of the pen's own rim pixels look neutral and would come back as specks
        cv::Mat wideOk;
        {
            cv::Mat rp, rn;
            rimPrint.convertTo(rp, CV_32F, 1.0 / 255.0);
            rimPen.convertTo(rn, CV_32F, 1.0 / 255.0);
            #ifndef INK_WIDE
#define INK_WIDE 6
#endif
#ifndef INK_SPECK
#define INK_SPECK (k * k)
#endif
#ifndef INK_WIDE_RATIO
#define INK_WIDE_RATIO 0.5f
#endif
            int wide = INK_WIDE * k + 1;
            cv::boxFilter(rp, rp, -1, cv::Size(wide, wide), cv::Point(-1, -1), false);
            cv::boxFilter(rn, rn, -1, cv::Size(wide, wide), cv::Point(-1, -1), false);
            wideOk = rp > INK_WIDE_RATIO * rn;
        }
        cv::Mat tonerCore = clipped & (votePrint > 0.6f * (votePrint + votePen)) & (votePrint > 0) & wideOk;
#ifdef INK_DEBUG
        if (g_dbgBox.area() > 0) { cv::Rect bx = g_dbgBox - g_dbgRoiOrigin; bx &= cv::Rect(0, 0, tonerCore.cols, tonerCore.rows); if (bx.area() > 0) fprintf(stderr, "toner-stage pixelvote: %d\n", cv::countNonZero(tonerCore(bx))); }
#endif
        cv::Mat penCore = clipped & (votePen > votePrint);
        // ...and per blob, as before: a clipped blob whose rims are mostly print-coloured is print
        // as a whole (pixels deep inside a thick letter may have no rim within reach)
        {
            Components cc = FindComponents(clipped);
            std::vector<float> pv(cc.count, 0), nv(cc.count, 0);
            for (int y = 0; y < clipped.rows; y++) {
                const int *l = cc.labels.ptr<int>(y);
                const float *vp = votePrint.ptr<float>(y), *vn = votePen.ptr<float>(y);
                for (int x = 0; x < clipped.cols; x++) if (l[x]) { pv[l[x]] += vp[x] > vn[x]; nv[l[x]] += vn[x] > vp[x]; }
            }
            std::vector<char> printBlob(cc.count, 0);
            for (int i = 1; i < cc.count; i++) printBlob[i] = pv[i] + nv[i] >= 3 && pv[i] > 0.7f * (pv[i] + nv[i]);
            tonerCore |= PaintComponents(cc.labels, printBlob);
        }
#ifdef INK_DEBUG
        if (g_dbgBox.area() > 0) { cv::Rect bx = g_dbgBox - g_dbgRoiOrigin; bx &= cv::Rect(0, 0, tonerCore.cols, tonerCore.rows); if (bx.area() > 0) fprintf(stderr, "toner-stage blob: %d\n", cv::countNonZero(tonerCore(bx))); }
#endif
        // grown into the dark, non-pen pixels around it (a letter's body between clipped parts)
        // (only as dark as the print cores themselves: a nearly neutral second pen, a shade
        // lighter than toner, would otherwise be grown into and restored as "print")
        float coreOD = 1.0f;
        if (cv::countNonZero(tonerCore)) {
            std::vector<float> v;
            for (int y = 0; y < tonerCore.rows; y++) {
                const uchar *t = tonerCore.ptr<uchar>(y);
                const float *m = d.mean.ptr<float>(y);
                for (int x = 0; x < tonerCore.cols; x++) if (t[x]) v.push_back(m[x]);
            }
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            coreOD = std::max(1.0f, 0.85f * v[v.size() / 2]);
        }
        cv::Mat growInto = d.ink & (d.mean > coreOD) & ~ownPen & ~penCore;
        for (int it = 0; it < 2; it++) {
            cv::Mat grown;
            cv::dilate(tonerCore, grown, cv::Mat::ones(3, 3, CV_8U));
            tonerCore |= grown & growInto;
        }
#ifdef INK_DEBUG
        if (g_dbgBox.area() > 0) { cv::Rect bx = g_dbgBox - g_dbgRoiOrigin; bx &= cv::Rect(0, 0, tonerCore.cols, tonerCore.rows); if (bx.area() > 0) fprintf(stderr, "toner-stage grow: %d\n", cv::countNonZero(tonerCore(bx))); }
#endif
        // ...plus its own anti-aliased rim (grey, not pen-coloured), or restored letters come back thin
        cv::Mat withRim;
        cv::dilate(tonerCore, withRim, cv::Mat::ones(3, 3, CV_8U));
        tonerCore |= withRim & d.ink & ~ownPen & ~penCore;
#ifdef INK_DEBUG
        if (g_dbgBox.area() > 0) { cv::Rect bx = g_dbgBox - g_dbgRoiOrigin; bx &= cv::Rect(0, 0, tonerCore.cols, tonerCore.rows); if (bx.area() > 0) fprintf(stderr, "toner-stage rim: %d\n", cv::countNonZero(tonerCore(bx))); }
#endif
        // specks are not letters
        {
            Components ct = FindComponents(tonerCore);
            std::vector<char> big(ct.count, 0);
            for (int i = 1; i < ct.count; i++) big[i] = ct.area[i] >= INK_SPECK;
            tonerCore = PaintComponents(ct.labels, big);
        }
#ifdef INK_DEBUG
        if (g_dbgBox.area() > 0) { cv::Rect bx = g_dbgBox - g_dbgRoiOrigin; bx &= cv::Rect(0, 0, tonerCore.cols, tonerCore.rows); if (bx.area() > 0) fprintf(stderr, "toner-stage final: %d\n", cv::countNonZero(tonerCore(bx))); }
#endif
#ifdef INK_DEBUG
        if (g_dbgBox.area() > 0) { cv::Rect bx = (g_dbgBox - g_dbgRoiOrigin) & cv::Rect(0, 0, tonerCore.cols, tonerCore.rows); if (bx.area() > 0) fprintf(stderr, "clipped %d  inBand %d  hw %d  meanOD-max %.2f\n", cv::countNonZero(clipped(bx)), cv::countNonZero(printBand(bx)), cv::countNonZero(hw(bx)), [&]{ double mx; cv::minMaxLoc(d.mean(bx), nullptr, &mx); return mx; }()); }
#endif
        overlap |= hw & printBand & ~ownPen & tonerCore;
#ifdef INK_DEBUG
        g_dbgToner = hw & printBand & ~ownPen & tonerCore;
#endif
#ifdef INK_DEBUG
        cv::imwrite("/tmp/ink_toner_" + std::to_string(coarse.cols) + "_" + std::to_string(coarse.rows) + ".png", tonerCore);
#endif
    }
    *outHw = hw;
    *outOverlap = overlap;
    *outPrintStrong = printStrong;
}

/// Detects handwriting, the print layer to protect and the overlap (print under pen) to restore,
/// all at full resolution: layout/colour context at working size, then per-pixel refinement.
/// `modelPrint` / `modelHw`: the segmentation model's probabilities at `full`'s size (CV_32F), or
/// empty when it is unavailable. Used only as a second opinion where colour cannot decide: whether
/// print really lies under a near-black pen stroke before it is restored, and whether colourless ink
/// outside the printed lines is handwriting (see ColourlessHandwriting).
inline void DetectByInkColor(const cv::Mat &full, double colorDelta, cv::Mat *outHandwriting, cv::Mat *outPrint, cv::Mat *outOverlap,
                             const cv::Mat &modelPrint = cv::Mat(), const cv::Mat &modelHw = cv::Mat()) {
    cv::Mat work = ToWorkingSize(full);
    cv::Mat coarseHw, coarsePrint, fragWork, mixedWork, bandWork, enclosedWork, colourlessWork;
    PenColor penWork;
    cv::Mat modelHwWork;
    if (!modelHw.empty()) cv::resize(modelHw, modelHwWork, work.size(), 0, 0, cv::INTER_AREA);
    DetectAtWorkingSize(work, colorDelta, &coarseHw, &coarsePrint, &penWork, &fragWork, &mixedWork, &bandWork, &enclosedWork,
                        modelHwWork, &colourlessWork);
    // print may be restored under a pen stroke only where the model also sees print there: a
    // near-black pen's core is as colourless as toner and its rim nearly neutral, so by colour a
    // formula written between the lines looked like print under a pen (measured: model print
    // probability 0.73-0.79 at real print under pen, 0.06-0.37 at such pen strokes)
    cv::Mat printish;
    if (!modelPrint.empty() && !modelHw.empty()) {
        // (5x5 mean, or a 5x5 max for the soft rims of a pen-crossed letter as long as the mean is
        // not near zero — measured: real print under pen passes 97-100%, a pen formula between the
        // lines 0-3%)
        cv::Mat mp, mx;
        cv::boxFilter(modelPrint, mp, -1, cv::Size(5, 5));
        cv::dilate(modelPrint, mx, cv::Mat::ones(5, 5, CV_8U));
        printish = (mp >= 0.5f) | ((mx >= 0.5f) & (mp >= 0.2f));
    }
    cv::Mat printBand = ScaleMaskTo(bandWork, full.size());
#ifdef INK_DEBUG
    cv::imwrite("/tmp/ink_band.png", printBand);
    cv::imwrite("/tmp/ink_coarse.png", ScaleMaskTo(coarseHw, full.size()));
    cv::imwrite("/tmp/ink_frag.png", ScaleMaskTo(fragWork, full.size()));
    cv::imwrite("/tmp/ink_mixed.png", ScaleMaskTo(mixedWork, full.size()));
    cv::imwrite("/tmp/ink_colourless.png", ScaleMaskTo(colourlessWork, full.size()));
    cv::imwrite("/tmp/ink_enclosed.png", ScaleMaskTo(enclosedWork, full.size()));
#endif
    cv::Mat coarse = ScaleMaskTo(coarseHw, full.size());
    cv::Mat fragments = ScaleMaskTo(fragWork, full.size());
    cv::Mat mixed = ScaleMaskTo(mixedWork, full.size());
    cv::Mat print = ScaleMaskTo(coarsePrint, full.size());
    // circled print (255) and the pen ring around it (128), see DetectAtWorkingSize
    cv::Mat enclosedPrint = ScaleMaskTo(enclosedWork == 255, full.size());
    cv::Mat ringPen = ScaleMaskTo(enclosedWork == 128, full.size());
    {   // the ring's own rim is pen too, but not the rim of the print it encloses
        cv::Mat enclosedNear;
        cv::dilate(enclosedPrint, enclosedNear, cv::Mat::ones(3, 3, CV_8U));
        cv::dilate(ringPen, ringPen, cv::Mat::ones(3, 3, CV_8U));
        ringPen &= ~enclosedNear;
    }

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
#ifdef INK_DEBUG
        fprintf(stderr, "refine roi %d %d %d %d\n", roi.x, roi.y, roi.width, roi.height);
#endif
#ifdef INK_DEBUG
        g_dbgRoiOrigin = roi.tl();
        if (getenv("INK_BOX")) { int bx, by, bw, bh; sscanf(getenv("INK_BOX"), "%d,%d,%d,%d", &bx, &by, &bw, &bh); g_dbgBox = cv::Rect(bx, by, bw, bh); }
#endif
        RefineRoi(d, coarse(roi), fragments(roi), mixed(roi), printBand(roi), penRoi, k, &roiHw, &roiOverlap, &roiPrintStrong);
        if (!printish.empty()) roiOverlap &= printish(roi);
        roiHw |= ringPen(roi) & d.ink;
        roiHw &= ~enclosedPrint(roi);
        roiOverlap &= ~ringPen(roi);
        roiPrintStrong |= enclosedPrint(roi) & d.ink;
        roiHw.copyTo(hw(roi));
        roiOverlap.copyTo(overlap(roi));
#ifdef INK_DEBUG
        if (g_dbgAllD.empty()) { g_dbgAllD = cv::Mat::zeros(full.size(), CV_8U); g_dbgAllB = g_dbgAllD.clone(); g_dbgAllT = g_dbgAllD.clone(); }
        if (!g_dbgDarker.empty()) g_dbgAllD(roi) |= g_dbgDarker;
        if (!g_dbgBridge.empty()) g_dbgAllB(roi) |= g_dbgBridge;
        if (!g_dbgToner.empty()) g_dbgAllT(roi) |= g_dbgToner;
        g_dbgDarker.release(); g_dbgBridge.release(); g_dbgToner.release();
        cv::imwrite("/tmp/ink_ov_darker.png", g_dbgAllD); cv::imwrite("/tmp/ink_ov_bridge.png", g_dbgAllB); cv::imwrite("/tmp/ink_ov_toner.png", g_dbgAllT);
#endif
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
        // colourless handwriting found by layout + shape, verified by the model
        if (!colourlessWork.empty() && cv::countNonZero(colourlessWork)) {
            cv::Mat colourless = ScaleMaskTo(colourlessWork, full.size());
            cv::dilate(colourless, colourless, cv::Mat::ones(3, 3, CV_8U));
            colourless &= dAll.ink;
            hw |= colourless;
            // pieces the coarse pass had as handwriting keep their refinement's overlap / print (a
            // pen-crossed printed letter has no twin either); the newly found ones are handwriting
            // through and through, with their anti-aliased rim (left in the print layer, the rim is
            // never erased: a ghost)
            cv::Mat fresh = colourless & ~coarse, freshNear;
            overlap &= ~fresh;
            cv::dilate(fresh, freshNear, cv::Mat::ones(5, 5, CV_8U));
            print &= ~freshNear;
        }
        // review layer — straight pen strokes: a long thin ink run that slopes (a printed rule is level
        // with a deskewed page) and that the model sees as handwriting is a pen strike-through or
        // underline, whatever colour classes its pixels got (a colourless pen passed as print there)
        if (!modelHw.empty()) {
            cv::Mat smear, runs;
            cv::dilate(dAll.ink, smear, cv::Mat::ones(3, 1, CV_8U));
            cv::morphologyEx(smear, runs, cv::MORPH_OPEN, cv::Mat::ones(1, std::max(9, 12 * k) | 1, CV_8U));
            runs &= dAll.ink;
            Components cs = FindComponents(runs);
            std::vector<float> mh(cs.count, 0.f);
            std::vector<int> n(cs.count, 0);
            for (int y = 0; y < runs.rows; y++) {
                const int *l = cs.labels.ptr<int>(y);
                const float *m = modelHw.ptr<float>(y);
                for (int x = 0; x < runs.cols; x++) if (l[x]) { mh[l[x]] += m[x]; n[l[x]]++; }
            }
            // width of the whole ink stroke each run belongs to (a long design line breaks into
            // several runs where it bends)
            // (level page rules / table grid removed first: a formula written on a table row is
            // joined to the whole table through them)
            cv::Mat levelH, levelV, inkNoGrid;
            const int lg = std::max(15, 30 * k) | 1;
            cv::morphologyEx(dAll.ink, levelH, cv::MORPH_OPEN, cv::Mat::ones(1, lg, CV_8U));
            cv::morphologyEx(dAll.ink, levelV, cv::MORPH_OPEN, cv::Mat::ones(lg, 1, CV_8U));
            cv::dilate(levelH | levelV, levelH, cv::Mat::ones(3, 3, CV_8U));
            inkNoGrid = dAll.ink & ~levelH;
            Components ci = FindComponents(inkNoGrid);
            std::vector<int> strokeW(cs.count, 0);
            for (int y = 0; y < runs.rows; y++) {
                const int *l = cs.labels.ptr<int>(y), *li = ci.labels.ptr<int>(y);
                for (int x = 0; x < runs.cols; x++) if (l[x] && li[x]) strokeW[l[x]] = std::max(strokeW[l[x]], ci.w[li[x]]);
            }
            std::vector<char> strike(cs.count, 0);
            for (int i = 1; i < cs.count; i++) {
                const int t = std::max(1, cs.area[i] / std::max(1, cs.w[i]));
                // (short: a sloped printed design line — a CV header's swoosh — spans the page)
                strike[i] = cs.h[i] - t > 0.015f * cs.w[i] + 2 && strokeW[i] < 0.25f * runs.cols && n[i] && mh[i] / n[i] >= 0.4f;
            }
            cv::Mat strikeMask;
            cv::dilate(PaintComponents(cs.labels, strike), strikeMask, cv::Mat::ones(3, 3, CV_8U));
            strikeMask &= dAll.ink;
            hw |= strikeMask;
            overlap &= ~strikeMask;
            print &= ~strikeMask;
        }
        // leftover pieces of a pen stroke: ink that is neither handwriting nor print, touching the
        // handwriting (the part of a pen loop that crossed the printed line's band and so was never a
        // candidate) is the same stroke
        {
            cv::Mat hwNear, loose = dAll.ink & ~print & ~hw;
            cv::dilate(hw, hwNear, cv::Mat::ones(3, 3, CV_8U));
            Components cl = FindComponents(loose);
            std::vector<float> touch = FractionPerComponent(cl, hwNear);
            std::vector<char> take(cl.count, 0);
            for (int i = 1; i < cl.count; i++) take[i] = touch[i] > 0 && cl.area[i] <= (8 * k) * (8 * k);
            hw |= PaintComponents(cl.labels, take);
        }
    }
    *outHandwriting = hw;
    *outPrint = print;
    *outOverlap = overlap;
}

// MARK: - Erase

/// Repairs printed strokes the erase took away where the pen crossed print, by EXAMPLE — no text
/// recognition (OCR misreads exactly the letters a pen crosses, and a wrong letter is worse than a
/// gap). A printed page repeats the same glyphs, so for each erased spot next to visible print, the
/// visible surroundings (≈6 stroke widths) are matched against the whole page; where another place
/// matches them closely (same letter, same font and size), its pixels fill the erased part. A spot
/// without a convincing match is left as it is.
inline void RepairPrintByExample(cv::Mat &dst, const cv::Mat &erased, int k) {
    if (cv::countNonZero(erased) == 0) return;
    cv::Mat gray;
    cv::cvtColor(dst, gray, cv::COLOR_BGR2GRAY);
    cv::Mat small, paper;
    cv::resize(gray, small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    cv::medianBlur(small, small, std::min(21, (std::min(small.cols, small.rows) - 1) | 1));
    cv::resize(small, paper, gray.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat inkLevel;
    cv::subtract(paper, gray, inkLevel);
    cv::Mat ink = inkLevel > 60;
    cv::Mat visiblePrint = ink & ~erased;

    // erased pixels right next to visible print = where print was cut
    cv::Mat seeds, nearPrint;
    cv::dilate(visiblePrint, nearPrint, Ellipse(2 * k + 1));
    seeds = erased & nearPrint;
    if (cv::countNonZero(seeds) == 0) return;

    const int T = std::max(15, 6 * k) | 1;   // patch size
    // matched on SHAPE, not grey level: ink amount 0..1, lightly blurred so sub-pixel shifts and
    // anti-aliasing do not dominate the score
    cv::Mat inkF;
    inkLevel.convertTo(inkF, CV_32F, 1.0 / 120.0);
    cv::min(inkF, 1.0, inkF);
    cv::max(inkF, 0.0, inkF);
    cv::GaussianBlur(inkF, inkF, cv::Size(0, 0), 1.0);
    // coarse search at 1/4 resolution (1/2 for small patches), refined at full resolution around
    // the best hit (the search is most of the erase time: 1/2 resolution took seconds per page)
    const int sc = 2;
    cv::Mat inkH;
    cv::resize(inkF, inkH, cv::Size(), 1.0 / sc, 1.0 / sc, cv::INTER_AREA);
    cv::Mat erasedH;
    cv::resize(erased, erasedH, inkH.size(), 0, 0, cv::INTER_NEAREST);
    const int TH = T / sc;
    cv::Mat erasedInt;
    cv::integral(erasedH / 255, erasedInt, CV_32S);
    auto erasedIn = [&](int x, int y) {
        return erasedInt.at<int>(y + TH, x + TH) - erasedInt.at<int>(y, x + TH) - erasedInt.at<int>(y + TH, x) + erasedInt.at<int>(y, x);
    };
    // windows on a half-patch grid wherever print was cut
    std::vector<cv::Point> windows;
    const int step = std::max(1, T / 2);
    for (int cy = 0; cy < seeds.rows; cy += step) {
        for (int cx = 0; cx < seeds.cols; cx += step) {
            cv::Rect cell(cx, cy, std::min(step, seeds.cols - cx), std::min(step, seeds.rows - cy));
            if (cv::countNonZero(seeds(cell)) < k) continue;
            windows.push_back(cv::Point(std::max(0, std::min(dst.cols - T, cx + step / 2 - T / 2)),
                                        std::max(0, std::min(dst.rows - T, cy + step / 2 - T / 2))));
        }
    }
    // each window finds its twin independently (in parallel); the fills are applied afterwards in
    // window order, so the result does not depend on the thread count
    std::vector<cv::Point> twinOf(windows.size(), cv::Point(-1, -1));
    cv::parallel_for_(cv::Range(0, (int)windows.size()), [&](const cv::Range &range) {
        for (int w = range.start; w < range.end; w++) {
            const int x0 = windows[w].x, y0 = windows[w].y;
            cv::Rect R(x0, y0, T, T);
            cv::Mat known = ~erased(R);
            int knownCount = cv::countNonZero(known);
            int knownInk = cv::countNonZero(visiblePrint(R));
            if (knownCount < 0.5 * T * T || knownInk < 2 * k * k) continue;   // too little evidence
            {   // only a printed rule in view: rules are re-joined separately, nothing to learn here
                cv::Mat rowsWithInk;
                cv::reduce(visiblePrint(R), rowsWithInk, 1, cv::REDUCE_MAX);
                int top = -1, bottom = -1;
                for (int y = 0; y < rowsWithInk.rows; y++) if (rowsWithInk.at<uchar>(y)) { if (top < 0) top = y; bottom = y; }
                if (top < 0 || bottom - top + 1 <= k) continue;
            }

            // coarse search
            cv::Rect RH(x0 / sc, y0 / sc, TH, TH);
            RH &= cv::Rect(0, 0, inkH.cols, inkH.rows);
            if (RH.width < TH || RH.height < TH) continue;
            cv::Mat maskH, res;
            cv::Mat knownH = ~erasedH(RH);
            knownH.convertTo(maskH, CV_32F, 1.0 / 255.0);
            cv::matchTemplate(inkH, inkH(RH), res, cv::TM_SQDIFF, maskH);
            // the two best DISTINCT twins at search resolution: the best valid spot, then the
            // best valid spot clear of the first (not itself, not mostly erased)
            std::vector<cv::Point> twinsH;
            for (int pass = 0; pass < 2; pass++) {
                float best = FLT_MAX;
                cv::Point bp(-1, -1);
                for (int y = 0; y < res.rows; y++) {
                    const float *r = res.ptr<float>(y);
                    for (int x = 0; x < res.cols; x++) {
                        if (r[x] >= best) continue;
                        if (std::abs(x - RH.x) < TH && std::abs(y - RH.y) < TH) continue;   // itself
                        if (pass == 1 && std::abs(x - twinsH[0].x) < TH && std::abs(y - twinsH[0].y) < TH) continue;
                        if (erasedIn(x, y) > 0.02 * TH * TH) continue;
                        best = r[x];
                        bp = cv::Point(x, y);
                    }
                }
                if (bp.x < 0) break;
                twinsH.push_back(bp);
            }
            if (twinsH.size() < 2) continue;
            // full-resolution refinement ±sc px of each
            cv::Mat maskF;
            known.convertTo(maskF, CV_32F, 1.0 / 255.0);
            std::vector<cv::Point> twins;
            std::vector<double> scores;
            for (auto &th : twinsH) {
                cv::Point bp(-1, -1);
                double bf = 1e30;
                for (int dy = -sc; dy <= sc + 1; dy++) {
                    for (int dx = -sc; dx <= sc + 1; dx++) {
                        int sx = th.x * sc + dx, sy = th.y * sc + dy;
                        if (sx < 0 || sy < 0 || sx + T > dst.cols || sy + T > dst.rows) continue;
                        cv::Mat diff = inkF(cv::Rect(sx, sy, T, T)) - inkF(R);
                        double ssd = cv::sum(diff.mul(diff).mul(maskF))[0];
                        if (ssd < bf) { bf = ssd; bp = cv::Point(sx, sy); }
                    }
                }
                twins.push_back(bp);
                // normalised by the visible INK, so a mostly-blank patch cannot win on paper alone
                scores.push_back(bf / std::max(1, knownInk));
            }
            if (twins[0].x < 0 || twins[1].x < 0) continue;
            // both twins must be convincing AND agree on what lies under the pen (a real repeated
            // glyph gives the same answer twice; a chance lookalike does not)
            cv::Mat e = erased(R);
            cv::Mat f0 = ink(cv::Rect(twins[0].x, twins[0].y, T, T)) & e, f1 = ink(cv::Rect(twins[1].x, twins[1].y, T, T)) & e;
            int inter = cv::countNonZero(f0 & f1), uni = cv::countNonZero(f0 | f1);
            double agree = uni == 0 ? 1.0 : (double)inter / uni;
            if (scores[0] > 0.06 || scores[1] > 0.10 || agree < 0.6) continue;
            twinOf[w] = twins[0];
        }
    });
    // fill only the erased pixels, and only with the twin's INK (paper is already paper)
    const cv::Mat source = dst.clone();
    cv::Mat done = cv::Mat::zeros(dst.size(), CV_8U);
    for (size_t w = 0; w < windows.size(); w++) {
        if (twinOf[w].x < 0) continue;
        const int x0 = windows[w].x, y0 = windows[w].y;
        const cv::Rect Rs(twinOf[w].x, twinOf[w].y, T, T);
        for (int y = 0; y < T; y++) {
            const uchar *e = erased.ptr<uchar>(y0 + y) + x0;
            const uchar *srcInk = ink.ptr<uchar>(Rs.y + y) + Rs.x;
            const cv::Vec3b *src = source.ptr<cv::Vec3b>(Rs.y + y) + Rs.x;
            cv::Vec3b *out = dst.ptr<cv::Vec3b>(y0 + y) + x0;
            uchar *dn = done.ptr<uchar>(y0 + y) + x0;
            for (int x = 0; x < T; x++) {
                if (!e[x] || !srcInk[x] || dn[x]) continue;
                out[x] = src[x];
                dn[x] = 255;
            }
        }
    }
}

/// Re-typesets printed letters the pen crossed, from the page's OWN glyphs — no text recognition
/// (OCR misreads exactly the letters a pen crosses, and a scrambled word like "(GYENEMERC)" has no
/// language to fall back on). The erased page's clean printed letters form a glyph bank (same
/// font, size and ink as the damaged text); for each printed line segment the handwriting
/// touched, a dynamic programme places the sequence of bank glyphs that best explains the print
/// still VISIBLE in the original (pen-coloured ink is unknown and does not count). A placed glyph
/// fills only erased pixels, with its own pixels.
///
/// Two passes, because a wrong letter is worse than a gap: the first accepts the majority — glyphs
/// the visible print confirms for the most part, print by the detector and print-coloured on
/// average; the second re-scans for the exceptions with the first pass's glyphs as certain
/// evidence: print the context vote pulled into the handwriting but whose own colour is clearly
/// print ("Th" next to a circled number), or a letter the pen hides for the most part whose visible
/// remainder is print by the detector. No layout rule: handwriting may sit anywhere — on rules,
/// between or over printed words. (Measured on a worksheet and two scanned CVs: every accepted
/// glyph was the right letter; without these gates blank answer lines filled with fake letters.)
inline void RetypesetPrint(cv::Mat &dst, const cv::Mat &analysis, const cv::Mat &handwriting, const cv::Mat &printMask, int k) {
    if (cv::countNonZero(handwriting) == 0) return;
#ifdef INK_DEBUG
    int64 t0 = cv::getTickCount(); double tTable = 0, tFit = 0; int nSeg = 0, nKeys = 0;
#endif
    const int W = dst.cols, H = dst.rows;
    const float unit = k / 5.0f;   // glyph size limits below were measured at k = 5
    auto inkMap = [](const cv::Mat &bgr) {
        cv::Mat g, small, paper, out;
        cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
        cv::resize(g, small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        cv::medianBlur(small, small, std::min(21, (std::min(small.cols, small.rows) - 1) | 1));
        cv::resize(small, paper, g.size(), 0, 0, cv::INTER_LINEAR);
        paper.convertTo(paper, CV_32F);
        g.convertTo(g, CV_32F);
        out = (paper - g) * (1.0 / 140.0);
        cv::max(out, 0.0, out);
        cv::min(out, 1.0, out);
        return out;
    };
    cv::Mat inkO = inkMap(analysis), inkE = inkMap(dst);

    // --- per-pixel colour: pen-coloured ink is unknown (the handwriting mask also holds print the
    // context vote pulled in, which is evidence)
    OpticalDensity d = DensityInRoi(analysis, PaperLowRes(analysis), cv::Rect(0, 0, W, H));
    cv::Mat clipped = d.ink & (d.mean > kClippedOD), coloured = d.ink & ~clipped;
    cv::Mat w3, sums[3], cw;
    coloured.convertTo(cw, CV_32F, 1.0 / 255.0);
    cv::boxFilter(cw, w3, -1, cv::Size(3, 3), cv::Point(-1, -1), false);
    for (int c = 0; c < 3; c++) cv::boxFilter(d.od[c].mul(cw), sums[c], -1, cv::Size(3, 3), cv::Point(-1, -1), false);
    cv::Mat tot = sums[0] + sums[1] + sums[2];
    cv::max(tot, 1e-6, tot);
    cv::Mat chroma[3];
    for (int c = 0; c < 3; c++) chroma[c] = sums[c] / tot;
    cv::Mat decided = (w3 >= 2.0f) & d.ink;
    auto medianOf = [&](const cv::Mat &m, const cv::Mat &mask) {
        std::vector<float> v;
        for (int y = 0; y < H; y++) {
            const float *p = m.ptr<float>(y);
            const uchar *q = mask.ptr<uchar>(y);
            for (int x = 0; x < W; x++) if (q[x]) v.push_back(p[x]);
        }
        if (v.empty()) return 1.0f / 3;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };
    float refPr[3], refPen[3], dir[3], norm = 0;
    for (int c = 0; c < 3; c++) {
        refPr[c] = medianOf(chroma[c], decided & ~handwriting);
        refPen[c] = medianOf(chroma[c], decided & handwriting);
        dir[c] = refPen[c] - refPr[c];
        norm += dir[c] * dir[c];
    }
    norm = std::sqrt(std::max(norm, 1e-12f));
    for (int c = 0; c < 3; c++) dir[c] /= norm;
    const float thr = 0.5f * norm;
    if (thr < 0.004f) return;   // no distinguishable pen colour on this page
    cv::Mat proj = cv::Mat::zeros(H, W, CV_32F);
    for (int c = 0; c < 3; c++) proj += (chroma[c] - refPr[c]) * dir[c];
    cv::Mat penPx = decided & (proj > thr);
    cv::Mat unknown, penNear;
    cv::dilate(penPx, unknown, cv::Mat::ones(3, 3, CV_8U));
    cv::dilate(penPx, penNear, cv::Mat::ones(5, 5, CV_8U));
    unknown |= clipped & penNear;
    // handwriting that does not touch print anywhere (an answer in its own cell) holds no print to
    // re-typeset: all of it is unknown, whatever its colour (a colourless pen looked like visible
    // print and got "1"s pasted into it)
    {
        // (print TEXT only: an answer usually touches the table rule or fill-in line under it)
        cv::Mat hR, vR, textPrint, printNear;
        const int lr = std::max(9, 6 * k) | 1;
        cv::morphologyEx(printMask, hR, cv::MORPH_OPEN, cv::Mat::ones(1, lr, CV_8U));
        cv::morphologyEx(printMask, vR, cv::MORPH_OPEN, cv::Mat::ones(lr, 1, CV_8U));
        cv::dilate(hR | vR, hR, cv::Mat::ones(3, 3, CV_8U));
        textPrint = printMask & ~hR;
        cv::dilate(textPrint, printNear, cv::Mat::ones(5, 5, CV_8U));
        Components chw = FindComponents(handwriting);
        std::vector<float> touch = FractionPerComponent(chw, printNear);
        std::vector<char> alone(chw.count, 0);
        // (by pixel count, not share: a big pen loop around a number that also covers the first
        // letters of the next word touches that word with only a few pixels)
        for (int i = 1; i < chw.count; i++) alone[i] = touch[i] * chw.area[i] < 2.0f;
        cv::Mat aloneMask;
        cv::dilate(PaintComponents(chw.labels, alone), aloneMask, cv::Mat::ones(3, 3, CV_8U));
        unknown |= aloneMask;
    }
    cv::Mat known = ~unknown;
    cv::Mat erasedArea;
    cv::dilate(handwriting, erasedArea, cv::Mat::ones(5, 5, CV_8U));

    // --- glyph bank: clean components of the erased page, grouped into text lines
    cv::Mat binE = inkE > 0.45f;
    Components c = FindComponents(binE);
    cv::Mat nearErase;
    cv::dilate(erasedArea, nearErase, cv::Mat::ones(5, 5, CV_8U));
    cv::Mat nearEraseInt;
    cv::integral(nearErase / 255, nearEraseInt, CV_32S);
    struct Member { int i, x, y, w, h; bool clean; };
    std::vector<Member> comps;
    for (int i = 1; i < c.count; i++) {
        if (c.h[i] < 6 * unit || c.h[i] > 60 * unit || c.w[i] > 60 * unit || c.area[i] < 8 * unit * unit) continue;
        int x = c.x[i], y = c.y[i], w = c.w[i], h = c.h[i];
        int n = nearEraseInt.at<int>(y + h, x + w) - nearEraseInt.at<int>(y, x + w) - nearEraseInt.at<int>(y + h, x) + nearEraseInt.at<int>(y, x);
        comps.push_back({i, x, y, w, h, n == 0});
    }
    std::sort(comps.begin(), comps.end(), [](const Member &a, const Member &b) { return a.y + a.h / 2.0f < b.y + b.h / 2.0f; });
    struct Line { std::vector<Member> m; float h, cy; int base = 0, cap = 0; };
    std::vector<Line> lines;
    auto medianF = [](std::vector<float> v) { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
    for (const Member &m : comps) {
        float cy = m.y + m.h / 2.0f;
        bool joined = false;
        for (Line &L : lines) {
            if (std::fabs(cy - L.cy) < 0.5f * L.h && std::fabs(m.h - L.h) < 0.8f * L.h) {
                L.m.push_back(m);
                std::vector<float> hs, cys;
                for (auto &q : L.m) { hs.push_back((float)q.h); cys.push_back(q.y + q.h / 2.0f); }
                std::sort(hs.begin(), hs.end());
                L.h = hs[hs.size() / 2];
                L.cy = medianF(cys);
                joined = true;
                break;
            }
        }
        if (!joined) {
            Line L;
            L.m.push_back(m);
            L.h = (float)m.h;
            L.cy = cy;
            lines.push_back(L);
        }
    }
    lines.erase(std::remove_if(lines.begin(), lines.end(), [](const Line &L) { return L.m.size() < 5; }), lines.end());
    for (Line &L : lines) {
        std::vector<float> bottoms, hs;
        for (auto &q : L.m) if (q.h >= 0.7f * L.h) { bottoms.push_back((float)(q.y + q.h)); hs.push_back((float)q.h); }
        L.base = (int)medianF(bottoms);
        L.cap = (int)medianF(hs);
    }
    struct Glyph { cv::Mat G, B; cv::Point src; int dy, cap; };
    std::vector<Glyph> bank;
    {
        struct Rep { size_t first; int members; };
        std::vector<Rep> reps;
        for (const Line &L : lines) {
            for (const Member &m : L.m) {
                if (!m.clean) continue;
                int x0 = std::max(0, m.x - 1), y0 = std::max(0, m.y - 1), x1 = std::min(W, m.x + m.w + 1), y1 = std::min(H, m.y + m.h + 1);
                cv::Rect r(x0, y0, x1 - x0, y1 - y0);
                cv::Mat own = c.labels(r) == m.i;
                cv::dilate(own, own, cv::Mat::ones(3, 3, CV_8U));
                cv::Mat G = cv::Mat::zeros(r.size(), CV_32F);
                inkE(r).copyTo(G, own);
                Glyph g{G, G > 0.35f, cv::Point(x0, y0), y0 - L.base, L.cap};
                // near-duplicates (the same letter many times on a page) cost time and add nothing:
                // greedy clusters of equal-sized glyphs with overlapping shapes, two kept each
                bool dup = false;
                for (Rep &rp : reps) {
                    const Glyph &q = bank[rp.first];
                    if (std::abs(q.B.rows - g.B.rows) > 1 || std::abs(q.B.cols - g.B.cols) > 1 || std::abs(q.dy - g.dy) > 1 || q.cap != g.cap) continue;
                    int hh = std::min(q.B.rows, g.B.rows), ww = std::min(q.B.cols, g.B.cols);
                    cv::Mat a = q.B(cv::Rect(0, 0, ww, hh)), b = g.B(cv::Rect(0, 0, ww, hh));
                    int inter = cv::countNonZero(a & b), uni = cv::countNonZero(a | b);
                    if (uni && inter >= 0.8 * uni) {
                        if (rp.members < 2) { rp.members++; bank.push_back(g); }
                        dup = true;
                        break;
                    }
                }
                if (!dup) { reps.push_back({bank.size(), 1}); bank.push_back(g); }
            }
        }
    }
    if (bank.empty()) return;

#ifdef INK_DEBUG
    fprintf(stderr, "retypeset setup %.2fs bank %zu lines %zu\n", (cv::getTickCount() - t0) / cv::getTickFrequency(), bank.size(), lines.size());
#endif
    // --- two passes over the damaged line segments
    const cv::Mat source = dst.clone();
    cv::Mat sourceGray;
    cv::cvtColor(source, sourceGray, cv::COLOR_BGR2GRAY);
    cv::Mat printM = printMask.clone();
    cv::Mat painted = cv::Mat::zeros(H, W, CV_8U), paintedBoxes = cv::Mat::zeros(H, W, CV_8U);
    cv::Mat hwCols;
    // placements of the first pass per segment, reused by the second where nothing was painted
    std::map<std::pair<int, int>, std::vector<std::array<int, 3>>> firstPass;
    for (int pass = 1; pass <= 2; pass++) {
        if (pass == 2) {
            // the first pass's glyphs are now certain print and count as evidence
            inkO.setTo(1.0f, painted);
            known |= painted;
            printM |= painted;
        }
        // ink that could ever confirm a glyph: visible, print by the detector or clearly
        // print-coloured, and not a rule (a rule says nothing about which letter it is) — segments
        // without any (a pen answer on a blank line) are skipped before the costly fit
        cv::Mat evidence = known & (inkO > 0.35f) & (printM | (decided & (proj <= 0.2f * thr)));
        {
            cv::Mat runs;
            cv::morphologyEx(evidence, runs, cv::MORPH_OPEN, cv::Mat::ones(1, std::max(9, 6 * k) | 1, CV_8U));
            cv::dilate(runs, runs, cv::Mat::ones(3, 1, CV_8U));
            evidence &= ~runs;
        }
        cv::Mat evidenceInt;
        cv::integral(evidence / 255, evidenceInt, CV_32S);
        cv::Mat textEvidence = evidence & printM & ~handwriting;
        for (const Line &L : lines) {
            int top = std::max(0, L.base - (int)(1.5f * L.cap) - 2), bot = std::min(H, L.base + (int)(0.5f * L.cap) + 2);
            if (bot - top < 4) continue;
            cv::reduce(handwriting.rowRange(top, bot), hwCols, 0, cv::REDUCE_MAX);
            std::vector<std::pair<int, int>> spans;
            int s0 = -1, prev = -1;
            for (int x = 0; x < W; x++) {
                if (!hwCols.at<uchar>(0, x)) continue;
                if (s0 < 0) s0 = x;
                else if (x - prev > L.cap) { spans.push_back({s0, prev}); s0 = x; }
                prev = x;
            }
            if (s0 >= 0) spans.push_back({s0, prev});
            std::vector<int> cand;
            for (int gi = 0; gi < (int)bank.size(); gi++) if (bank[gi].cap >= 0.8f * L.cap && bank[gi].cap <= 1.25f * L.cap) cand.push_back(gi);
            if (cand.empty()) continue;
            for (auto sp : spans) {
                int a = std::max(0, sp.first - L.cap), b = std::min(W - 1, sp.second + L.cap);
                // grown over the ink right next to it (whole neighbouring letters)
                for (int it = 0; it < 3; it++)
                    for (const Member &m : L.m)
                        if (m.x < b + 0.5f * L.cap && m.x + m.w > a - 0.5f * L.cap) { a = std::max(0, std::min(a, m.x - 2)); b = std::min(W - 1, std::max(b, m.x + m.w + 2)); }
                cv::Rect strip(a, top, b - a + 1, bot - top);
                if (cv::countNonZero(erasedArea(strip)) == 0) continue;
#ifdef INK_DEBUG
                if (getenv("INK_RT_BOX")) { int bx, by, bw, bh; sscanf(getenv("INK_RT_BOX"), "%d,%d,%d,%d", &bx, &by, &bw, &bh);
                    if ((strip & cv::Rect(bx, by, bw, bh)).area() > 0) fprintf(stderr, "rt pass%d segment %d..%d y %d..%d evidence %d\n", pass, a, b, top, bot, evidenceInt.at<int>(bot, b + 1) - evidenceInt.at<int>(top, b + 1) - evidenceInt.at<int>(bot, a) + evidenceInt.at<int>(top, a)); }
#endif
                {
                    int ev = evidenceInt.at<int>(bot, b + 1) - evidenceInt.at<int>(top, b + 1) - evidenceInt.at<int>(bot, a) + evidenceInt.at<int>(top, a);
                    if (ev < 12.0f * (L.cap / 15.0f) * (L.cap / 15.0f)) continue;
                }
                const std::pair<int, int> segKey(top, a);
                const bool reuse = pass == 2 && firstPass.count(segKey) && cv::countNonZero(painted(strip)) == 0;
                const int width = strip.width;
                cv::Mat T = inkO(strip), K, KT, UT;
                known(strip).convertTo(K, CV_32F, 1.0 / 255.0);
                KT = K.mul(T);
                cv::Mat tMask;
                cv::Mat(T > 0.35f).convertTo(tMask, CV_32F, 1.0 / 255.0);
                UT = (1.0f - K).mul(tMask);   // hidden ink (under the pen)
                cv::Mat colPenM;
                cv::reduce(KT, colPenM, 0, cv::REDUCE_SUM);
                std::vector<float> colPen(width), colPenCum(width + 1, 0.f);
                for (int x = 0; x < width; x++) { colPen[x] = colPenM.at<float>(0, x); colPenCum[x + 1] = colPenCum[x] + colPen[x]; }
                // score of every candidate glyph at every x: s = sum K (3 T G - T - G) + 0.3 hidden ink covered
                struct Key { int gi, jit; std::vector<float> s; };
                std::vector<Key> table;
                std::vector<std::array<int, 3>> placements;   // (x, key gi, jit) of the fit
                if (reuse) {
                    placements = firstPass[segKey];
                } else {
                for (int gi : cand) {
                    if (bank[gi].G.cols > width) continue;
                    for (int jit = -1; jit <= 1; jit++) {
                        int gy = L.base + bank[gi].dy + jit - top;
                        if (gy < 0 || gy + bank[gi].G.rows > bot - top) continue;
                        table.push_back({gi, jit, {}});
                    }
                }
#ifdef INK_DEBUG
                int64 ta = cv::getTickCount(); nSeg++; nKeys += (int)table.size();
#endif
                // the score is linear in the image: one correlation with 3 KT - K + 0.3 UT gives it,
                // and one over a strip 2 px taller gives all three vertical jitters at once
                cv::Mat M = 3.0f * KT - K + 0.3f * UT;
                std::vector<int> firstKey(bank.size(), -1);
                for (int ki = 0; ki < (int)table.size(); ki++) if (firstKey[table[ki].gi] < 0) firstKey[table[ki].gi] = ki;
                std::vector<int> glyphs;
                for (int gi : cand) if (firstKey[gi] >= 0) glyphs.push_back(gi);
                cv::parallel_for_(cv::Range(0, (int)glyphs.size()), [&](const cv::Range &range) {
                    for (int n = range.start; n < range.end; n++) {
                        const int gi = glyphs[n];
                        const Glyph &g = bank[gi];
                        const int gh = g.G.rows, gw = g.G.cols;
                        const int y0 = L.base + g.dy - 1 - top;           // jitter -1
                        const int ya = std::max(0, y0), yb = std::min(bot - top, y0 + gh + 2);
                        if (yb - ya < gh) continue;
                        cv::Mat res;
                        cv::matchTemplate(M(cv::Rect(0, ya, width, yb - ya)), g.G, res, cv::TM_CCORR);
                        for (int ki = firstKey[gi]; ki < (int)table.size() && table[ki].gi == gi; ki++) {
                            Key &key = table[ki];
                            const int row = y0 + (key.jit + 1) - ya;
                            if (row < 0 || row >= res.rows) continue;
                            key.s.resize(res.cols);
                            const float *rp = res.ptr<float>(row);
                            for (int x = 0; x < res.cols; x++) key.s[x] = rp[x] - (colPenCum[x + gw] - colPenCum[x]);
                        }
                    }
                });
#ifdef INK_DEBUG
                int64 tb = cv::getTickCount(); tTable += (tb - ta) / cv::getTickFrequency();
#endif
#ifdef INK_DEBUG
                if (getenv("INK_RT_BOX")) { int bx, by, bw, bh; sscanf(getenv("INK_RT_BOX"), "%d,%d,%d,%d", &bx, &by, &bw, &bh);
                    if ((strip & cv::Rect(bx, by, bw, bh)).area() > 0)
                        for (int x = std::max(0, bx - a - 2); x < std::min(width, bx - a + 4); x++) {
                            std::vector<std::pair<float, int>> best;
                            for (int ki = 0; ki < (int)table.size(); ki++) if (x < (int)table[ki].s.size()) best.push_back({table[ki].s[x], ki});
                            std::sort(best.rbegin(), best.rend());
                            for (int q = 0; q < std::min(3, (int)best.size()); q++) { const Glyph &g = bank[table[best[q].second].gi]; fprintf(stderr, "rt x=%d score %.1f glyph %dx%d src %d,%d colPen@x %.1f\n", x + a, best[q].first, g.G.cols, g.G.rows, g.src.x, g.src.y, colPen[x]); }
                        } }
#endif
                // free fit: gap (1 px, pays for uncovered visible ink) or a glyph with a positive score
                const float NEG = -1e30f;
                std::vector<float> score(width + 1, NEG);
                std::vector<int> backX(width + 1, -1), backKey(width + 1, -1);
                score[0] = 0;
                for (int x = 0; x < width; x++) {
                    if (score[x] == NEG) continue;
                    float v = score[x] - colPen[x];
                    if (v > score[x + 1]) { score[x + 1] = v; backX[x + 1] = x; backKey[x + 1] = -1; }
                    for (int ki = 0; ki < (int)table.size(); ki++) {
                        const Key &key = table[ki];
                        if (x >= (int)key.s.size() || key.s[x] <= 0) continue;
                        // (no overlap of neighbouring glyph boxes, not even their empty margins: measured,
                        // a 1 px overlap let wrong letters in — "CATIO" re-typeset as "CARO")
                        int adv = bank[key.gi].G.cols;
                        float v2 = score[x] + key.s[x];
                        if (v2 > score[x + adv]) { score[x + adv] = v2; backX[x + adv] = x; backKey[x + adv] = ki; }
                    }
                }
                for (int x = width; x > 0 && backX[x] >= 0; x = backX[x])
                    if (backKey[x] >= 0) placements.push_back({backX[x], table[backKey[x]].gi, table[backKey[x]].jit});
                if (pass == 1) firstPass[segKey] = placements;
#ifdef INK_DEBUG
                tFit += (cv::getTickCount() - tb) / cv::getTickFrequency();
#endif
                }
                for (const auto &pl : placements) {
                    const Glyph &g = bank[pl[1]];
                    const int gh = g.G.rows, gw = g.G.cols;
                    const int gx = a + pl[0], gy = L.base + g.dy + pl[2];
                    if (gy < 0 || gy + gh > H || gx + gw > W) continue;
                    cv::Rect r(gx, gy, gw, gh);
                    cv::Mat need = erasedArea(r) & g.B;
                    if (pass == 2) {
                        if (cv::mean(paintedBoxes(r))[0] > 0.3 * 255) continue;
                        need &= ~painted(r);
                    }
                    if (cv::countNonZero(need) == 0) continue;
                    cv::Mat mt = known(r) & g.B & (inkO(r) > 0.35f);
                    int visInk = cv::countNonZero(known(r) & g.B);
                    int nm = cv::countNonZero(mt);
                    if (nm == 0) continue;
                    // the confirmed ink must span the glyph's height: a rule under the pen confirms
                    // only a letter's bottom bar and says nothing about which letter it is
                    cv::Mat rowsHit;
                    cv::reduce(mt, rowsHit, 1, cv::REDUCE_MAX);
                    int r0 = -1, r1 = -1;
                    for (int y = 0; y < gh; y++) if (rowsHit.at<uchar>(y)) { if (r0 < 0) r0 = y; r1 = y; }
                    if (r1 - r0 + 1 < 0.4 * gh) continue;
                    float ratio = (float)nm / std::max(1, cv::countNonZero(g.B));
                    float inPrint = (float)cv::countNonZero(mt & printM(r)) / nm;
                    cv::Mat md = mt & decided(r);
                    float mp = cv::countNonZero(md) ? (float)(cv::mean(proj(r), md)[0] / thr) : 0.0f;
                    const float area = (L.cap / 15.0f) * (L.cap / 15.0f);   // counts below measured at a 15 px cap height
                    bool ok;
                    if (pass == 1) ok = ratio >= 0.5f && inPrint >= 0.7f && mp <= 0.35f && nm >= 0.8f * visInk;
                    else {
                        // (a) needs visible print TEXT right beside the glyph on its rows (the "ings"
                        // after a pen-covered "Th"): a colourless pen formula written between the lines
                        // is print-coloured too, but its neighbours are its own pen strokes
                        bool besidePrint = false;
                        {
                            const int gap = std::max(3, L.cap / 2);
                            cv::Rect lr(std::max(0, gx - gap), gy, std::min(gap, gx), gh), rr(gx + gw, gy, std::min(gap, W - gx - gw), gh);
                            for (const cv::Rect &side : {lr, rr})
                                if (side.width > 0 && cv::countNonZero(textEvidence(side)) >= 3) besidePrint = true;
                        }
                        ok = nm >= 0.6f * visInk && ((ratio >= 0.5f && mp <= 0.2f && nm >= 20 * area && besidePrint) ||
                                                     (inPrint >= 0.8f && nm >= 12 * area && mp <= 0.3f));
                    }
#ifdef INK_DEBUG
                    if (getenv("INK_RT_BOX")) { int bx, by, bw, bh; sscanf(getenv("INK_RT_BOX"), "%d,%d,%d,%d", &bx, &by, &bw, &bh);
                        if ((r & cv::Rect(bx, by, bw, bh)).area() > 0) fprintf(stderr, "rt pass%d glyph@%d,%d %dx%d ratio %.2f inPrint %.2f mp %.2f nm %d vis %d ok %d\n", pass, gx, gy, gw, gh, ratio, inPrint, mp, nm, visInk, (int)ok); }
#endif
                    // and the glyph must leave no visible print in its box unexplained: an "e" pasted
                    // where an "a" lost its bowl under the pen still matches the visible part, but the
                    // a's own stem, visible beside it, is not the e's
                    if (ok) {
                        cv::Mat unexplained = known(r) & (inkO(r) > 0.35f) & printM(r) & ~g.B;
                        cv::Mat bDil;
                        cv::dilate(g.B, bDil, cv::Mat::ones(3, 3, CV_8U));
                        unexplained &= ~bDil;
                        if (cv::countNonZero(unexplained) > 0.12f * nm) ok = false;
                    }
                    if (!ok) continue;
                    // the glyph's own pixels, only where darker than what the erase left there
                    cv::Rect rs(g.src.x, g.src.y, gw, gh);
                    cv::Mat dstGray;
                    cv::cvtColor(dst(r), dstGray, cv::COLOR_BGR2GRAY);
                    cv::Mat m = need & (sourceGray(rs) < dstGray);
                    source(rs).copyTo(dst(r), m);
                    painted(r) |= g.B;
                    paintedBoxes(r).setTo(255);
                }
            }
        }
    }
#ifdef INK_DEBUG
    fprintf(stderr, "retypeset total %.2fs table %.2fs fit %.2fs segments %d keys %d\n", (cv::getTickCount() - t0) / cv::getTickFrequency(), tTable, tFit, nSeg, nKeys);
#endif
}


/// Last review layer — stray ink around what was erased. Every piece of ink left near an erased
/// stroke must "mean something" on this page: a typeset page repeats its glyphs (letters, digits,
/// dots and accents) almost exactly, so a piece with no twin among the page's untouched glyphs is a
/// leftover of the pen and goes. Rules and table lines are taken out first and never touched —
/// only what sticks out of them is judged, so a rule keeps its exact thickness and length. Print
/// the erase restored or re-typeset (`protectedInk`) stays. Small twins (a dot, an accent) must also
/// sit where a mark belongs: straight above or below a letter.
inline void RemoveStrayInk(cv::Mat &dst, const cv::Mat &erasedAll, const cv::Mat &handwriting, const cv::Mat &protectedInk, const cv::Mat &paperBright, int k) {
    if (cv::countNonZero(erasedAll) == 0) return;
    cv::Mat g, pg, diff;
    cv::cvtColor(dst, g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(paperBright, pg, cv::COLOR_BGR2GRAY);
    cv::subtract(pg, g, diff, cv::noArray(), CV_16S);
    cv::Mat ink = diff > 50;
    // rules / table lines at their exact thickness
    const int L = std::max(15, 8 * k) | 1;
    cv::Mat hR, vR;
    cv::morphologyEx(ink, hR, cv::MORPH_OPEN, cv::Mat::ones(1, L, CV_8U));
    cv::morphologyEx(ink, vR, cv::MORPH_OPEN, cv::Mat::ones(L, 1, CV_8U));
    // (a rule the erase re-joined in pieces is still that rule: ink exactly on a rule's rows, along
    // its line, belongs to it)
    cv::Mat hExt, vExt;
    cv::dilate(hR, hExt, cv::Mat::ones(1, 6 * L, CV_8U));
    cv::dilate(vR, vExt, cv::Mat::ones(6 * L, 1, CV_8U));
    cv::Mat rules = hR | vR | (ink & (hExt | vExt));
    cv::Mat glyphInk = ink & ~rules;
    Components c = FindComponents(glyphInk);
    // (where the erase worked: erased pixels, and print it put back)
    cv::Mat zone;
    cv::dilate(erasedAll | protectedInk, zone, Ellipse(2 * k + 1));
    std::vector<float> inZone = FractionPerComponent(c, zone);
    std::vector<float> prot = FractionPerComponent(c, protectedInk);
    // judged only where the pen was: a piece mostly on the handwriting mask (its rim included) is a
    // candidate; print beside an erased stroke is never one (small print — periods, accents, a thin
    // font — has few exact twins and must not depend on them)
    cv::Mat hwRim;
    cv::dilate(handwriting, hwRim, cv::Mat::ones(5, 5, CV_8U));
    std::vector<float> onPen = FractionPerComponent(c, hwRim);
    // the reference: untouched glyphs, and the text height
    std::vector<int> ref;
    std::vector<float> hs;
    for (int i = 1; i < c.count; i++)
        if (inZone[i] == 0 && c.area[i] >= 3) { ref.push_back(i); if (c.area[i] > (2 * k) * (2 * k)) hs.push_back((float)c.h[i]); }
    if (ref.size() < 50 || hs.empty()) return;
    std::nth_element(hs.begin(), hs.begin() + hs.size() / 2, hs.end());
    const float cap = hs[hs.size() / 2];
    std::sort(ref.begin(), ref.end(), [&](int a, int b) { return c.h[a] < c.h[b]; });
    std::vector<cv::Mat> shapes(c.count);
    auto shape = [&](int i) -> const cv::Mat & {
        if (shapes[i].empty()) shapes[i] = c.labels(cv::Rect(c.x[i], c.y[i], c.w[i], c.h[i])) == i;
        return shapes[i];
    };
    auto hasTwin = [&](int i) {
        const bool tiny = c.h[i] <= 4 || c.w[i] <= 4;
        const float need = tiny ? 0.5f : 0.6f;
        const int hLo = tiny ? c.h[i] - 1 : (int)std::floor(c.h[i] * 0.85f), hHi = tiny ? c.h[i] + 1 : (int)std::ceil(c.h[i] * 1.18f);
        auto lo = std::lower_bound(ref.begin(), ref.end(), hLo, [&](int a, int v) { return c.h[a] < v; });
        const cv::Mat &B = shape(i);
        for (auto it = lo; it != ref.end() && c.h[*it] <= hHi; ++it) {
            int j = *it;
            if (tiny ? std::abs(c.w[j] - c.w[i]) > 1 : (c.w[j] < 0.82f * c.w[i] || c.w[j] > 1.22f * c.w[i])) continue;
            cv::Mat G;
            cv::resize(shape(j), G, B.size(), 0, 0, cv::INTER_NEAREST);
            int inter = cv::countNonZero(G & B), uni = cv::countNonZero(G | B);
            if (uni && inter >= need * uni) return true;
        }
        return false;
    };
    // letters to hang marks on
    std::vector<char> stray(c.count, 0);
    for (int i = 1; i < c.count; i++) {
#ifdef INK_DEBUG
        if (getenv("INK_BOX")) { int bx, by, bw, bh; sscanf(getenv("INK_BOX"), "%d,%d,%d,%d", &bx, &by, &bw, &bh);
            if ((cv::Rect(c.x[i], c.y[i], c.w[i], c.h[i]) & cv::Rect(bx, by, bw, bh)).area() > 0)
                fprintf(stderr, "stray-cand %d,%d %dx%d a=%d zone %.2f prot %.2f onPen %.2f twin %d\n", c.x[i], c.y[i], c.w[i], c.h[i], c.area[i], inZone[i], prot[i], onPen[i], (int)hasTwin(i)); }
#endif
        // (restored print is kept — unless it is a meaningless crumb: a speck with no twin, what is
        // left of a letter the pen took and nothing could rebuild)
        const bool crumb = c.area[i] <= k * k && c.h[i] < 0.4f * cap && c.w[i] < 0.4f * cap;
        if (inZone[i] < 0.3f || (prot[i] > 0.5f && !crumb) || onPen[i] < 0.6f) continue;
        if (c.h[i] > 2.5f * cap || c.w[i] > 4 * cap) continue;             // not a stray piece
        if (!hasTwin(i)) { stray[i] = 1; continue; }
        if (c.h[i] < 0.5f * cap && c.area[i] <= (2 * k) * (2 * k)) {
            // a mark: a letter straight above or below it, within half a text height
            int gap = (int)std::ceil(0.5f * cap);
            cv::Rect col(c.x[i], std::max(0, c.y[i] - gap), c.w[i], 0);
            col.height = std::min(dst.rows, c.y[i] + c.h[i] + gap) - col.y;
            // (and centred on it: a printed accent sits over the middle of its letter, a pen speck
            // anywhere)
            bool centred = false;
            const float cx = c.x[i] + c.w[i] / 2.0f;
            cv::Mat labs = c.labels(col);
            std::vector<int> seen;
            for (int y = 0; y < labs.rows && !centred; y++)
                for (int x = 0; x < labs.cols && !centred; x++) {
                    int j = labs.at<int>(y, x);
                    if (j == 0 || j == i || c.h[j] < 0.5f * cap || c.area[j] <= k * k) continue;
                    if (std::find(seen.begin(), seen.end(), j) != seen.end()) continue;
                    seen.push_back(j);
                    // (and close: a dot below a letter within ~1/3 of the text height of its foot, a
                    // mark above within ~0.45 of its top)
                    const bool below = c.y[i] >= c.y[j] + c.h[j];
                    const int gapV = below ? c.y[i] - (c.y[j] + c.h[j]) : c.y[j] - (c.y[i] + c.h[i]);
                    const bool close = gapV <= (below ? 0.35f : 0.45f) * cap;
                    centred = close && cx >= c.x[j] + 0.2f * c.w[j] && cx <= c.x[j] + 0.8f * c.w[j];
                }
            if (!centred) stray[i] = 1;
        }
    }
    cv::Mat strayMask = PaintComponents(c.labels, stray), grown;
    // with its anti-aliased rim, but never into a rule or protected print
    cv::dilate(strayMask, grown, cv::Mat::ones(3, 3, CV_8U));
    cv::Mat rulesNear;
    cv::dilate(rules, rulesNear, cv::Mat::ones(1, 1, CV_8U));
    grown &= ~rules & (~protectedInk | strayMask) & (diff > 6);
    grown |= strayMask;
    paperBright.copyTo(dst, grown);
#ifdef INK_DEBUG
    cv::imwrite("/tmp/ink_stray.png", strayMask);
    fprintf(stderr, "stray ink: %d px removed\n", cv::countNonZero(strayMask));
#endif
}

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
    // the brightest paper around (~15 cells of 8 px): under a dense block of handwriting even the
    // local paper estimate is pulled grey, and erased strokes would be filled with that grey
    cv::Mat targetPaperSmallBright;
    cv::dilate(targetPaperSmall, targetPaperSmallBright, cv::Mat::ones(15, 15, CV_8U));
    cv::Mat targetSmall;
    cv::resize(target, targetSmall, cv::Size(), 0.125, 0.125, cv::INTER_AREA);
    cv::medianBlur(targetSmall, targetSmall, 21);
    int margin = Odd(61 * s) / 2 + Odd(13 * s) + Odd(11 * s) + 4;
    cv::Mat erasedAll = cv::Mat::zeros(target.size(), CV_8U), restoredAll = erasedAll.clone();

    // Printed rules the pen wrote ALONG (fill-in blanks): the pen fused with the rule, so the print
    // layer never had it. Found on the WHOLE page as long, thin, straight ink runs (a pen stroke is
    // never this straight for 20 stroke widths; a slight tilt is tolerated by a 1 px vertical
    // smear), judged as whole lines: a rule runs well beyond the handwriting written on it, a
    // straight hand-drawn strike-through lies (almost) entirely within the handwriting.
    cv::Mat pageRules = cv::Mat::zeros(target.size(), CV_8U);
    {
        OpticalDensity dAll = DensityInRoi(analysis, paperSmall, cv::Rect(0, 0, analysis.cols, analysis.rows));
        cv::Mat smear, runs, hwNear;
        cv::dilate(dAll.ink, smear, cv::Mat::ones(3, 1, CV_8U));
        int l2 = std::max(5, std::min(20 * k, analysis.cols / 30)) | 1;
        cv::morphologyEx(smear, runs, cv::MORPH_OPEN, cv::Mat::ones(1, l2, CV_8U));
        runs &= dAll.ink;
        cv::dilate(handwriting, hwNear, Ellipse(2 * k + 1));
        Components cr = FindComponents(runs);
        std::vector<float> inside = FractionPerComponent(cr, hwNear);
        std::vector<char> rule(cr.count, 0);
        // ...or so long and straight that no hand drew it (a blank the answer fills end to end)
        // (the length is capped by the page width: k has a floor, so on a small page 40k would
        // be longer than a whole blank)
        const int ruleLen = std::min(40 * k, analysis.cols / 18);
        // ...but a long one must still show some of itself OUTSIDE the pen strokes: a ruler-straight
        // pen underline is handwriting end to end, a rule peeks out past the answer written on it
        std::vector<float> inHw = FractionPerComponent(cr, handwriting);
        for (int i = 1; i < cr.count; i++)
            rule[i] = inside[i] <= 0.7f || (cr.w[i] >= ruleLen && inHw[i] <= 0.9f);
        // a printed rule is level with the page (a scan is deskewed to well under a degree); a pen
        // strike-through drawn with a ruler still slopes (measured: 4% on a real worksheet)
        {
            std::vector<int> thick;
            for (int i = 1; i < cr.count; i++) if (rule[i]) thick.push_back(std::max(1, cr.area[i] / std::max(1, cr.w[i])));
            std::sort(thick.begin(), thick.end());
            const int t = thick.empty() ? k : thick[thick.size() / 2];
            for (int i = 1; i < cr.count; i++)
                if (rule[i] && inside[i] > 0.7f && cr.h[i] - t > 0.015f * cr.w[i] + 2) rule[i] = 0;
        }
        pageRules = PaintComponents(cr.labels, rule);
#ifdef INK_DEBUG
        cv::imwrite("/tmp/ink_runs.png", runs);
        cv::imwrite("/tmp/ink_rules.png", pageRules);
        cv::imwrite("/tmp/ink_dink.png", dAll.ink);
        fprintf(stderr, "k=%d\n", k);
#endif
    }

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
            int l1 = std::max(3, 6 * k) | 1, l2 = std::max(5, std::min(20 * k, analysis.cols / 30)) | 1;
            cv::morphologyEx(prInk, closed, cv::MORPH_CLOSE, cv::Mat::ones(1, l1, CV_8U));
            cv::morphologyEx(closed, runs, cv::MORPH_OPEN, cv::Mat::ones(1, l2, CV_8U));
            cv::morphologyEx(runs, thick, cv::MORPH_OPEN, cv::Mat::ones(std::max(3, k) | 1, 1, CV_8U));
            cv::dilate(thick, thickNear, cv::Mat::ones(3, 3, CV_8U));
            restoreMask |= runs & ~thickNear & area;
        }
        // printed rules the pen wrote ALONG (fill-in blanks) — found page-wide, see below
        restoreMask |= pageRules(roi) & area;
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
        // (the brightest-nearby paper estimate: a median at 1/8 scale is pulled grey by a dense
        // block of handwriting, and the erased block came out as a grey smear)
        cv::resize(targetPaperSmall(sr), paperColor, roi.size(), 0, 0, cv::INTER_LINEAR);
        cv::GaussianBlur(paperColor, paperColor, cv::Size(0, 0), 8);
        cv::Mat nearArea;
        cv::dilate(area, nearArea, Ellipse(9 * s));
        cv::Mat keep = (printNear & nearArea & ~area) | restore;
        cv::Mat tmp = out.clone();
        paperColor.copyTo(tmp, keep);
        cv::Mat filled;
        cv::inpaint(tmp, paperArea, filled, 3, cv::INPAINT_TELEA);
        // inpainting pulls the pale anti-aliased edge of the stroke inwards and leaves a light
        // ghost of it (measured: grey 192-250 on white paper): erased pixels are never darker than
        // the local paper
        // (against the BRIGHTEST paper nearby, ~5 stroke-free cells wide: under a dense block of
        // handwriting even the local paper estimate comes out grey, leaving a grey smear)
        {
            cv::Mat brightSmall, bright;
            cv::dilate(targetPaperSmallBright(sr), brightSmall, cv::Mat::ones(1, 1, CV_8U));
            cv::resize(brightSmall, bright, roi.size(), 0, 0, cv::INTER_LINEAR);
            cv::GaussianBlur(bright, bright, cv::Size(0, 0), 8);
            cv::Mat fg, pg;
            cv::cvtColor(filled, fg, cv::COLOR_BGR2GRAY);
            cv::cvtColor(bright, pg, cv::COLOR_BGR2GRAY);
            cv::Mat ghost = paperArea & (fg < pg - 6);
            bright.copyTo(filled, ghost);
        }
        out.copyTo(filled, keep);
        filled.copyTo(dst(roi));
        erasedAll(roi) |= paperArea;
        restoredAll(roi) |= restore;
#ifdef INK_DEBUG
        if (roi.contains(cv::Point(55, 1040))) {
            cv::Rect q(55 - roi.x - 15, 1040 - roi.y - 16, 60, 33);
            cv::Mat viz; cv::merge(std::vector<cv::Mat>{restore(q), pr(q), paperArea(q)}, viz);
            cv::imwrite("/tmp/ink_erase5.png", viz);
            cv::imwrite("/tmp/ink_erase5_out.png", filled(q));
        }
#endif
    }
    // final review layer — leftovers around what was erased: specks of the stroke's tips and its grey
    // anti-aliased halo that no mask covered. Near erased strokes, ink that is not print (by the
    // print layer, rules included) and is either faint or a speck is set to the paper.
    {
        cv::Mat zone, printNear;
        cv::dilate(erasedAll, zone, Ellipse(3 * k + 1));
        // (restored print — rules re-joined, letters under the pen — and level rules are kept)
        cv::Mat levelRuns;
        {
            cv::Mat g0;
            cv::cvtColor(dst, g0, cv::COLOR_BGR2GRAY);
            cv::morphologyEx(g0 < 200, levelRuns, cv::MORPH_OPEN, cv::Mat::ones(1, std::max(9, 6 * k) | 1, CV_8U));
        }
        // (the print layer right next to the pen is its own dilated rim, not print: there only
        // faint ink is cleaned, specks only where the print layer does not reach)
        cv::Mat hwNear;
        cv::dilate(handwriting, hwNear, Ellipse(2 * k + 1));
        cv::dilate((print & ~hwNear) | overlap | restoredAll | levelRuns, printNear, cv::Mat::ones(3, 3, CV_8U));
        cv::Mat bright, g, pg;
        cv::Mat brightSmall;
        brightSmall = targetPaperSmallBright;
        cv::resize(brightSmall, bright, dst.size(), 0, 0, cv::INTER_LINEAR);
        cv::GaussianBlur(bright, bright, cv::Size(0, 0), 8);
        cv::cvtColor(dst, g, cv::COLOR_BGR2GRAY);
        cv::cvtColor(bright, pg, cv::COLOR_BGR2GRAY);
        cv::Mat diff;
        cv::subtract(pg, g, diff, cv::noArray(), CV_16S);
        cv::Mat darkish = diff > 6, strong = diff > 60;
        cv::Mat cand = zone & darkish & ~printNear;
        Components cs = FindComponents(cand);
        std::vector<float> strongShare = FractionPerComponent(cs, strong);
        std::vector<float> inPrint = FractionPerComponent(cs, print);
        // letter-sized ink that stays on the page: a speck within a stroke width of it may be its
        // accent or dot (Vietnamese "ộ", "ị"), a speck further away is a leftover of the pen
        cv::Mat letters;
        {
            Components cl = FindComponents(strong & ~erasedAll);
            std::vector<char> big(cl.count, 0);
            for (int i = 1; i < cl.count; i++) big[i] = cl.area[i] > (2 * k) * (2 * k);
            cv::dilate(PaintComponents(cl.labels, big), letters, Ellipse(2 * k + 1));
        }
        std::vector<float> nearLetter = FractionPerComponent(cs, letters);
        std::vector<char> clean(cs.count, 0);
        for (int i = 1; i < cs.count; i++) {
            const bool speck = cs.area[i] <= (2 * k) * (2 * k);
            clean[i] = strongShare[i] < 0.3f ||                                   // faint halo
                       (speck && inPrint[i] < 0.5f && nearLetter[i] == 0);         // a speck off the print, no letter by it
        }
        cv::Mat cleanMask = PaintComponents(cs.labels, clean);
#ifdef INK_DEBUG
        {
            cv::Rect q(560, 282, 80, 18);
            if ((q & cv::Rect(0, 0, dst.cols, dst.rows)) == q)
                fprintf(stderr, "cleanup probe: zone %d darkish %d printNear %d cand %d clean %d  bright %.0f g %.0f\n", cv::countNonZero(zone(q)), cv::countNonZero(darkish(q)), cv::countNonZero(printNear(q)), cv::countNonZero(cand(q)), cv::countNonZero(cleanMask(q)), cv::mean(pg(q))[0], cv::mean(g(q))[0]);
        }
#endif
        bright.copyTo(dst, cleanMask);
    }
    const cv::Mat beforeRepair = dst.clone();
#ifndef INK_NO_REPAIR
    RepairPrintByExample(dst, erasedAll, k);
#endif
#ifndef INK_NO_RETYPESET
    RetypesetPrint(dst, analysis, handwriting, print | overlap, k);
#endif
#ifndef INK_NO_STRAY
    {
        // print put back by the repair steps is protected, like the restored print
        cv::Mat changed, painted;
        cv::absdiff(dst, beforeRepair, changed);
        cv::cvtColor(changed, changed, cv::COLOR_BGR2GRAY);
        painted = changed > 30;
        cv::Mat bright;
        cv::resize(targetPaperSmallBright, bright, dst.size(), 0, 0, cv::INTER_LINEAR);
        cv::GaussianBlur(bright, bright, cv::Size(0, 0), 8);
        cv::Mat protect;
        cv::dilate(painted | restoredAll, protect, cv::Mat::ones(3, 3, CV_8U));
        RemoveStrayInk(dst, erasedAll, handwriting, protect, bright, k);
    }
#endif
    return dst;
}

}  // namespace inkanalysis
