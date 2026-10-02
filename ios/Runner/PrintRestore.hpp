// Post-erase restore — runs right after the eraser (InkRefine erase) on the same page. A port of
// ml/prototype/print_restore.py (keep the two in step; Android: PrintRestore.kt):
//
// 1. print the eraser took: erased strokes inside a printed-text line band; inside the
//    handwriting mask only with the model's print layer or print geometry (level with the print
//    beside it at its height, a digit on a fraction bar, a diacritic on a glyph)
// 2. rules (grid) the eraser took: long thin straight neutral runs of the original that continue
//    a surviving rule, bridged across the pen strokes crossing them
// 3. pen the eraser left: bits of mostly-erased writing, and chromatic ink, off the print lines
#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace printrestore {

namespace detail {

inline cv::Mat Paper(const cv::Mat &img, int k, int blur) {
    std::vector<cv::Mat> ch; cv::split(img, ch);
    for (auto &c : ch) { cv::dilate(c, c, cv::Mat::ones(k, k, CV_8U)); cv::medianBlur(c, c, blur); }
    cv::Mat out; cv::merge(ch, out); return out;
}

inline cv::Mat Normalize(const cv::Mat &img) {
    cv::Mat bg; Paper(img, 7, 31).convertTo(bg, CV_32FC3);
    cv::Mat f; img.convertTo(f, CV_32FC3);
    cv::max(bg, 1.0, bg);
    cv::Mat out; cv::divide(f, bg, out, 255.0);
    out.convertTo(out, CV_8UC3);
    return out;
}

/// darkness (255 - L) and the a/b chroma axes, CV_32F
inline void Lab(const cv::Mat &bgr, cv::Mat *d, cv::Mat *a, cv::Mat *b) {
    cv::Mat lab; cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch; cv::split(lab, ch);
    if (d) ch[0].convertTo(*d, CV_32F, -1, 255);
    if (a) ch[1].convertTo(*a, CV_32F, 1, -128);
    if (b) ch[2].convertTo(*b, CV_32F, 1, -128);
}

inline cv::Mat Rect(int w, int h) { return cv::getStructuringElement(cv::MORPH_RECT, cv::Size(std::max(1, w), std::max(1, h))); }
inline cv::Mat Ellipse(int d) { d = std::max(1, d); return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(d, d)); }
inline cv::Mat Dilate(const cv::Mat &m, const cv::Mat &k) { cv::Mat o; cv::dilate(m, o, k); return o; }

inline double Median(std::vector<float> v) {
    if (v.empty()) return 0;
    size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    return v[m];
}

inline double Percentile(std::vector<float> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = q / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx, hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

inline std::vector<float> Values(const cv::Mat &f, const cv::Mat &mask) {
    std::vector<float> v;
    for (int y = 0; y < f.rows; y++) {
        const float *pf = f.ptr<float>(y); const uchar *pm = mask.ptr<uchar>(y);
        for (int x = 0; x < f.cols; x++) if (pm[x]) v.push_back(pf[x]);
    }
    return v;
}

/// Connected components with their pixel lists (CSR), so per-component statistics are one pass.
struct Components {
    int n = 0;
    cv::Mat labels, stats;
    std::vector<int> start, pix;      // pixels of component i: pix[start[i] .. start[i+1])
    explicit Components(const cv::Mat &mask) {
        cv::Mat cents;
        n = cv::connectedComponentsWithStats(mask, labels, stats, cents, 8, CV_32S);
        start.assign(n + 1, 0);
        const int *lab = labels.ptr<int>();
        const int total = (int)labels.total();
        for (int i = 0; i < total; i++) start[lab[i] + 1]++;
        for (int i = 0; i < n; i++) start[i + 1] += start[i];
        pix.resize(total);
        std::vector<int> fill(start.begin(), start.end() - 1);
        for (int i = 0; i < total; i++) pix[fill[lab[i]]++] = i;
    }
    int x(int i) const { return stats.at<int>(i, cv::CC_STAT_LEFT); }
    int y(int i) const { return stats.at<int>(i, cv::CC_STAT_TOP); }
    int w(int i) const { return stats.at<int>(i, cv::CC_STAT_WIDTH); }
    int h(int i) const { return stats.at<int>(i, cv::CC_STAT_HEIGHT); }
    int area(int i) const { return stats.at<int>(i, cv::CC_STAT_AREA); }
    double Mean(int i, const cv::Mat &m8) const {          // share of component i set in a CV_8U mask
        const uchar *p = m8.ptr<uchar>(); int c = 0;
        for (int j = start[i]; j < start[i + 1]; j++) c += p[pix[j]] != 0;
        return (double)c / std::max(1, start[i + 1] - start[i]);
    }
    double MeanF(int i, const cv::Mat &f) const {
        const float *p = f.ptr<float>(); double s = 0;
        for (int j = start[i]; j < start[i + 1]; j++) s += p[pix[j]];
        return s / std::max(1, start[i + 1] - start[i]);
    }
    double MedianF(int i, const cv::Mat &f) const {
        const float *p = f.ptr<float>(); std::vector<float> v;
        v.reserve(start[i + 1] - start[i]);
        for (int j = start[i]; j < start[i + 1]; j++) v.push_back(p[pix[j]]);
        return Median(v);
    }
    void Paint(int i, cv::Mat &m8) const {
        uchar *p = m8.ptr<uchar>();
        for (int j = start[i]; j < start[i + 1]; j++) p[pix[j]] = 255;
    }
};

struct Box { float x, y, w, h; };

}  // namespace detail

/**
 * Restores print / rules the eraser took from `erased` and removes the pen it left, in place.
 *  - original: the unprocessed page (colour source; white-balanced here)
 *  - target:   the page the eraser worked on (near-greyscale, shadow-removed)
 *  - erased:   the eraser's output, same size as `target` (CV_8UC3, modified)
 *  - handwriting / print: the mask's layers (CV_8U, 255 = set), same size
 */
inline void Restore(const cv::Mat &original, const cv::Mat &target, cv::Mat &erased,
                    const cv::Mat &handwriting, const cv::Mat &print) {
    using namespace detail;
    const int H = erased.rows, W = erased.cols;
    const int k = std::max(1, (int)std::lround(std::max(H, W) / 1280.0));
    cv::Mat O = original;
    if (O.size() != erased.size()) cv::resize(original, O, erased.size(), 0, 0, cv::INTER_AREA);
    cv::Mat C = Normalize(O);
    cv::Mat dN, aN, bN, dE;
    Lab(target, &dN, nullptr, nullptr);
    Lab(C, nullptr, &aN, &bN);
    Lab(erased, &dE, nullptr, nullptr);
    const cv::Mat paperE = Paper(erased, 9, 21);   // the erased page's paper, for leftover pen
    cv::Mat hwMask = handwriting > 0, printMask = print > 127;
    cv::Mat purple = aN - bN;                       // pen axis: purple/red pen is a+ b-, print neutral
    const double INK = 40;
    cv::Mat inkN = dN > INK, inkE = dE > INK;
    cv::Mat erasedM = inkN & (dE < 0.35 * dN);      // ink the eraser took
    cv::Mat faded = inkN & (dE < 0.6 * dN);         // …or faded to a grey ghost

    // rule lines surviving in the erased page
    cv::Mat hl, vl;
    cv::morphologyEx(inkE, hl, cv::MORPH_OPEN, Rect(15 * k, 1));
    cv::morphologyEx(inkE, vl, cv::MORPH_OPEN, Rect(1, 15 * k));
    cv::Mat linesE = hl | vl;

    // printed text: glyph-sized components of the surviving non-rule ink
    cv::Mat textE = inkE & ~Dilate(linesE, cv::Mat::ones(3, 3, CV_8U));
    Components T(textE);
    std::vector<char> glyph(T.n, 0);
    std::vector<int> heights;
    for (int i = 1; i < T.n; i++) {
        glyph[i] = T.area(i) >= 12 * k * k && T.h(i) >= 6 * k && T.h(i) <= 0.03 * H && T.w(i) <= 0.05 * W;
        if (glyph[i]) heights.push_back(T.h(i));
    }
    int gh = 12 * k;
    if (!heights.empty()) { std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end()); gh = heights[heights.size() / 2]; }
    // share of the original ink around each point that the eraser took (tight window: a printed
    // word right beside an erased one stays intact; a stray pen bit sits in a cleared patch)
    int win = (gh / 2) | 1;
    cv::Mat ef, inf, erasedRatio;
    erasedM.convertTo(ef, CV_32F, 1 / 255.0); inkN.convertTo(inf, CV_32F, 1 / 255.0);
    cv::blur(ef, ef, cv::Size(win, win)); cv::blur(inf, inf, cv::Size(win, win));
    cv::max(inf, 1e-3, inf);
    cv::divide(ef, inf, erasedRatio);
    std::vector<char> intact(T.n, 0);
    for (int i = 1; i < T.n; i++) { intact[i] = T.MeanF(i, erasedRatio) < 0.5; glyph[i] &= intact[i]; }
    // a printed line = several glyphs side by side on one row (dashes, dots count as members)
    cv::Mat member = cv::Mat::zeros(H, W, CV_8U), core = cv::Mat::zeros(H, W, CV_8U);
    for (int i = 1; i < T.n; i++) {
        if (intact[i] && T.area(i) >= 4 * k * k) T.Paint(i, member);
        if (glyph[i]) T.Paint(i, core);
    }
    cv::Mat rows; cv::morphologyEx(member, rows, cv::MORPH_CLOSE, Rect(2 * gh, 1));
    Components Rw(rows);
    std::vector<char> lineOk(Rw.n, 0);
    for (int i = 1; i < Rw.n; i++) lineOk[i] = Rw.w(i) >= 6 * gh && Rw.Mean(i, core) > 0;
    std::vector<Box> gBox;
    for (int i = 1; i < T.n; i++) {
        if (!glyph[i]) continue;
        int lab = Rw.labels.at<int>(T.y(i) + T.h(i) / 2, T.x(i) + T.w(i) / 2);
        bool ok = lab > 0 && lineOk[lab];
        if (!ok) { // the glyph's centre may fall in a gap: check any of its pixels
            const int *rl = Rw.labels.ptr<int>();
            for (int j = T.start[i]; j < T.start[i + 1] && !ok; j++) ok = lineOk[rl[T.pix[j]]];
        }
        if (ok) gBox.push_back({(float)T.x(i), (float)T.y(i), (float)T.w(i), (float)T.h(i)});
        else glyph[i] = 0;
    }
    core.setTo(0);
    for (int i = 1; i < T.n; i++) if (glyph[i]) T.Paint(i, core);
    std::vector<Box> bBox;            // short surviving bars a fraction's digits hang on
    for (int i = 1; i < T.n; i++)
        if (T.w(i) <= 1.5 * gh && T.w(i) >= 0.4 * gh && T.h(i) <= std::max(2.0, 0.3 * gh))
            bBox.push_back({(float)T.x(i), (float)T.y(i), (float)T.w(i), (float)T.h(i)});

    auto onPrintLine = [&](int x, int y, int w, int h) {
        std::vector<float> cy, hh, base;
        for (auto &g : gBox)
            if (g.x < x + w + 4 * gh && g.x + g.w > x - 4 * gh && std::abs(g.y + g.h / 2 - (y + h / 2.0)) < gh) {
                cy.push_back(g.y + g.h / 2); hh.push_back(g.h); base.push_back(g.y + g.h);
            }
        if (cy.size() < 2) return false;
        double c = Median(cy), ghn = Median(hh), b = Median(base);
        bool level = std::min(std::abs(y + h / 2.0 - c), std::abs(y + h - b)) <= 0.4 * ghn;
        return level && h >= 0.6 * ghn && h <= 1.6 * ghn;
    };
    auto hangsOn = [&](const std::vector<Box> &boxes, int x, int y, int w, int h, double up, double down) {
        for (auto &g : boxes) {
            if (!(g.x < x + w && g.x + g.w > x)) continue;
            double a = g.y - (y + h), bl = y - (g.y + g.h);
            if ((a >= -1 && a <= up * gh) || (bl >= -1 && bl <= down * gh)) return true;
        }
        return false;
    };

    cv::Mat band = Dilate(core, Rect(4 * gh, gh | 1));
    cv::Mat tallBand = Dilate(core, Rect(2 * gh, (3 * gh) | 1));

    // colour models — pen from what was erased, print from surviving glyphs
    cv::Mat dark70 = dN > 70;
    double penC = Median(Values(purple, erasedM & dark70));
    double printC = Median(Values(purple, core & dark70));
    double split = (penC + printC) / 2, margin = 0.25 * std::abs(penC - printC);
    bool separable = std::abs(penC - printC) >= 4;

    // ── rules the eraser took ──
    cv::Mat neutralN = (dN > 18) & (purple < split);
    cv::Mat footprint = Dilate(erasedM, Ellipse(7 * k));
    cv::Mat missing = (dN > 18) & (dE < 0.5 * dN) & footprint;
    cv::Mat ruleGap = cv::Mat::zeros(H, W, CV_8U);
    for (int dir = 0; dir < 2; dir++) {
        cv::Mat run;
        cv::morphologyEx(neutralN, run, cv::MORPH_CLOSE, dir == 0 ? Rect(7 * k, 1) : Rect(1, 7 * k));
        cv::morphologyEx(run, run, cv::MORPH_OPEN, dir == 0 ? Rect(31 * k, 1) : Rect(1, 31 * k));
        Components R(run);
        for (int i = 1; i < R.n; i++) {
            double thick = (double)R.area(i) / std::max(R.w(i), R.h(i));
            // a real rule mostly survived the eraser and was only cut
            if (thick <= 3 * k && R.Mean(i, linesE) >= 0.3) {
                cv::Mat m = cv::Mat::zeros(H, W, CV_8U); R.Paint(i, m);
                ruleGap |= m & missing;
            }
        }
    }
    ruleGap &= ~linesE;

    // ── print the eraser took ──
    double a0 = Median(Values(aN, core & dark70)), b0 = Median(Values(bN, core & dark70));
    cv::Mat da = aN - a0, db = bN - b0, cdPx; cv::magnitude(da, db, cdPx);
    double cdPrintT = Percentile(Values(cdPx, core & dark70), 90);
    Components F(faded & ~ruleGap);
    cv::Mat restore = cv::Mat::zeros(H, W, CV_8U);
    std::vector<char> done(F.n, 0);
    for (int pass = 0; pass < 3; pass++) {           // a restored glyph extends the band
        for (int i = 1; i < F.n; i++) {
            if (done[i]) continue;
            int x = F.x(i), y = F.y(i), w = F.w(i), h = F.h(i);
            double pm = F.MedianF(i, purple);
            bool printColoured = pm < printC + margin && F.MedianF(i, cdPx) <= cdPrintT;
            if (separable && !printColoured && pm >= split) continue;          // pen-coloured
            bool inBand = F.Mean(i, band) > 0.6;
            bool glyphSized = h <= 1.5 * gh && w <= 1.5 * gh;
            bool stacked = glyphSized && F.Mean(i, tallBand) > 0.8 &&
                (hangsOn(bBox, x, y, w, h, 0.5, 0.5) || (w <= 0.6 * gh && h <= 0.6 * gh && hangsOn(gBox, x, y, w, h, 0.6, 0.4)));
            if (!(inBand || stacked)) continue;
            // beyond the handwriting mask (the eraser's halo): the band suffices. Inside it the
            // model called them handwriting — and a dark pen can be print-coloured — so it takes
            // the model's print layer, or print geometry
            bool inside = F.Mean(i, hwMask) > 0.5;
            if (!inside || F.Mean(i, printMask) > 0.5 || (printColoured && (stacked || onPrintLine(x, y, w, h)))) {
                F.Paint(i, restore); done[i] = 1;
            }
        }
        band |= Dilate(restore, Rect(4 * gh, gh | 1));
    }

    // ── pen the eraser left ──
    cv::Mat inkNf; inkN.convertTo(inkNf, CV_32F, 1 / 255.0);
    cv::Mat cdist = cdPx.clone(); cdist.setTo(0, ~inkN);
    cv::Mat cdS, wS; cv::blur(cdist, cdS, cv::Size(3, 3)); cv::blur(inkNf, wS, cv::Size(3, 3));
    cv::max(wS, 1e-3, wS); cv::divide(cdS, wS, cdS);
    double penCd = Median(Values(cdS, erasedM & dark70));
    double printCd = Percentile(Values(cdS, core & dark70), 90);
    double chromaT = (penCd + printCd) / 2;
    cv::Mat keep = band | restore | ruleGap | (printMask & ~hwMask);
    cv::Mat beside = Dilate(erasedM, Ellipse((2 * gh) | 1));
    cv::Mat rulesAll = Dilate(linesE | ruleGap, cv::Mat::ones(3, 3, CV_8U));
    cv::Mat residue = cv::Mat::zeros(H, W, CV_8U);
    for (int i = 1; i < T.n; i++) if (T.MeanF(i, erasedRatio) > 0.5) T.Paint(i, residue);
    residue &= ~keep & beside;
    if (separable) {
        cv::Mat chromaInk = inkE & (cdS > chromaT) & beside & ~rulesAll;
        residue |= Dilate(chromaInk, cv::Mat::ones(3, 3, CV_8U)) & inkE & ~keep & ~rulesAll;
    }

    // ── compose ──
    target.copyTo(erased, restore);
    std::vector<cv::Vec3b> ruleSamples;
    cv::Mat ruleSrc = linesE & (dN > 25);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (ruleSrc.at<uchar>(y, x)) ruleSamples.push_back(target.at<cv::Vec3b>(y, x));
    cv::Vec3b ruleCol(160, 160, 160);
    if (!ruleSamples.empty()) {
        for (int c = 0; c < 3; c++) {
            std::vector<float> v; v.reserve(ruleSamples.size());
            for (auto &p : ruleSamples) v.push_back(p[c]);
            ruleCol[c] = (uchar)Median(v);
        }
    }
    for (int y = 0; y < H; y++) {
        const uchar *rg = ruleGap.ptr<uchar>(y); const float *pp = purple.ptr<float>(y);
        for (int x = 0; x < W; x++)
            if (rg[x]) erased.at<cv::Vec3b>(y, x) = pp[x] < split ? target.at<cv::Vec3b>(y, x) : ruleCol;
    }
    paperE.copyTo(erased, residue);
}

}  // namespace printrestore
