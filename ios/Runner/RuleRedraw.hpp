// Last erase step: printed rules (answer lines, table grid) that handwriting was written on are
// cleared and drawn again. Pen strokes fused with a rule cannot be told from it pixel by pixel, so
// the erase leaves bits of the writing stuck to the line; instead, where the handwriting mask
// touches a rule, the band around the rule is cleared of pen leftovers and the rule is redrawn as a
// straight line of its own measured thickness and colour. Mirrors RuleRedraw.kt on Android (same
// steps, same constants).
//
// A rule is only taken when it is (on its parts without handwriting on them) long, thin, straight,
// solid (a dotted line is left alone), isolated (paper right above and below — not the top or
// bottom edge of a text line) and print in the mask (not a pen underline). Leftover pieces cleared
// must lie in the band near the rule AND in the handwriting mask — printed letters never qualify —
// and another rule crossing the band is never touched.
#pragma once

#import <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ruleredraw {

/// A rule in "rule space": along = x (page column, or page row when transposed), across = y.
struct Rule {
    bool transposed;
    double a, b, th;
    int x0, x1;
    std::vector<int> cleanX, cleanY;   // its points without handwriting, for its colour
    double y(int x) const { return a * x + b; }
};

inline int StrokeUnit(int w, int h) { return std::max(5, std::max(w, h) / 450) | 1; }

/// Rules of a page whose rules run along x. ink / pure / hwOn: CV_8U 0/1, size h x w.
inline std::vector<Rule> Detect(const cv::Mat &ink, const cv::Mat &pure, const cv::Mat &hwOn, int k, bool transposed) {
    const int w = ink.cols, h = ink.rows;
    const int minLen = std::max(8 * k, w / 8);
    cv::Mat run;
    cv::morphologyEx(ink, run, cv::MORPH_CLOSE, cv::Mat::ones(1, k, CV_8U));
    cv::morphologyEx(run, run, cv::MORPH_OPEN, cv::Mat::ones(1, minLen, CV_8U));
    cv::Mat labels, stats, cents;
    const int count = cv::connectedComponentsWithStats(run, labels, stats, cents, 8, CV_32S);
    auto at = [&](const cv::Mat &m, int x, int y) { return m.at<uchar>(std::min(h - 1, std::max(0, y)), x) != 0; };
    std::vector<Rule> rules;
    for (int l = 1; l < count; l++) {
        const int x0 = stats.at<int>(l, cv::CC_STAT_LEFT), y0 = stats.at<int>(l, cv::CC_STAT_TOP);
        const int bw = stats.at<int>(l, cv::CC_STAT_WIDTH), bh = stats.at<int>(l, cv::CC_STAT_HEIGHT);
        if (bw < minLen) continue;
        std::vector<int> top(bw, INT_MAX), bot(bw, -1);
        for (int y = y0; y < y0 + bh; y++) {
            const int *lp = labels.ptr<int>(y);
            for (int x = x0; x < x0 + bw; x++) if (lp[x] == l) {
                const int c = x - x0; top[c] = std::min(top[c], y); bot[c] = std::max(bot[c], y);
            }
        }
        // clean column: no handwriting ON the line (pen letters only touch a rule here and there)
        std::vector<char> clean(bw, 0); int nClean = 0;
        for (int c = 0; c < bw; c++) if (bot[c] >= 0 && !at(hwOn, x0 + c, (top[c] + bot[c]) / 2)) { clean[c] = 1; nClean++; }
        if (nClean < 0.3 * bw) continue;
        std::vector<double> ths; ths.reserve(nClean);
        for (int c = 0; c < bw; c++) if (clean[c]) ths.push_back(bot[c] - top[c] + 1);
        std::sort(ths.begin(), ths.end());
        const double th = ths[nClean / 2];
        if (th > 0.6 * k + 2) continue;                                       // thin
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int c = 0; c < bw; c++) if (clean[c]) {
            const double x = x0 + c, cy = (top[c] + bot[c]) / 2.0;
            sx += x; sy += cy; sxx += x * x; sxy += x * cy;
        }
        const double den = nClean * sxx - sx * sx;
        const double a = den != 0 ? (nClean * sxy - sx * sy) / den : 0;
        const double b = (sy - a * sx) / nClean;
        std::vector<double> res; res.reserve(nClean);
        for (int c = 0; c < bw; c++) if (clean[c]) res.push_back(std::abs((top[c] + bot[c]) / 2.0 - (a * (x0 + c) + b)));
        std::sort(res.begin(), res.end());
        if (res[(size_t)(0.8 * (nClean - 1))] > 1.5) continue;                // straight
        Rule r{transposed, a, b, th, x0, x0 + bw, {}, {}};
        const int gap = (int)(th / 2) + 2;
        int solid = 0, above = 0, below = 0, pureHit = 0;
        for (int c = 0; c < bw; c++) if (clean[c]) {
            const int x = x0 + c, y = std::min(h - 1, std::max(0, (int)std::lround(a * x + b)));
            r.cleanX.push_back(x); r.cleanY.push_back(y);
            if (at(ink, x, y) || at(ink, x, y - 1) || at(ink, x, y + 1)) solid++;
            if (at(ink, x, y - gap - 1) || at(ink, x, y - gap - 3)) above++;
            if (at(ink, x, y + gap + 1) || at(ink, x, y + gap + 3)) below++;
            if (at(pure, x, y) || at(pure, x, y - 1) || at(pure, x, y + 1)) pureHit++;
        }
        if (solid < 0.9 * nClean) continue;                                   // solid, not dotted
        if (above > 0.2 * nClean || below > 0.2 * nClean) continue;           // isolated
        if (pureHit < 0.5 * nClean) continue;                                 // print, not pen
        rules.push_back(std::move(r));
    }
    return rules;
}

/// original: the unprocessed page the mask was detected on (CV_8UC3); handwriting / print: the
/// mask's layers (CV_8UC1, 255 = set); dst: the erased page (CV_8UC3), updated in place.
inline void Apply(const cv::Mat &original, const cv::Mat &handwriting, const cv::Mat &print, cv::Mat &dst) {
    const int w = original.cols, h = original.rows;
    const int k = StrokeUnit(w, h);
    cv::Mat hw = handwriting > 0;  hw /= 255;
    cv::Mat pure = (print > 127) & (handwriting == 0); pure /= 255;
    cv::Mat ink;
    {
        cv::Mat g, bg;
        cv::cvtColor(original, g, cv::COLOR_BGR2GRAY);
        cv::morphologyEx(g, bg, cv::MORPH_CLOSE, cv::Mat::ones(25, 25, CV_8U));
        ink = (bg - g) > 35; ink /= 255;
    }
    cv::Mat hwOn, hwTight;
    cv::dilate(hw, hwOn, cv::Mat::ones(3, 3, CV_8U));
    cv::dilate(hw, hwTight, cv::Mat::ones(5, 5, CV_8U));
    std::vector<Rule> rules = Detect(ink, pure, hwOn, k, false);
    {
        std::vector<Rule> v = Detect(ink.t(), pure.t(), hwOn.t(), k, true);
        rules.insert(rules.end(), v.begin(), v.end());
    }
    if (rules.empty()) return;

    cv::Mat paper;
    {
        cv::Mat small;
        cv::resize(dst, small, cv::Size(), 0.125, 0.125, cv::INTER_AREA);
        cv::dilate(small, small, cv::Mat::ones(5, 5, CV_8U));
        cv::medianBlur(small, small, 5);
        cv::resize(small, paper, dst.size(), 0, 0, cv::INTER_LINEAR);
        cv::GaussianBlur(paper, paper, cv::Size(0, 0), 4);
    }
    auto px = [&](const Rule &r, int x, int y) { return r.transposed ? cv::Point(y, x) : cv::Point(x, y); };
    auto across = [&](const Rule &r) { return r.transposed ? w : h; };

    // every rule's own line (protected from the other rules' clearing)
    cv::Mat lineMask = cv::Mat::zeros(h, w, CV_8U);
    for (const Rule &r : rules) for (int x = r.x0; x < r.x1; x++) {
        const double yc = r.y(x);
        for (double d = -r.th / 2 - 1; d <= r.th / 2 + 1.01; d += 1.0) {
            const int y = (int)std::lround(yc + d);
            if (y >= 0 && y < across(r)) lineMask.at<uchar>(px(r, x, y)) = 1;
        }
    }
    // affected columns: handwriting in the band near the rule
    struct Col { const Rule *r; int x; double yc; cv::Vec3f colour; };
    cv::Mat clearZone = cv::Mat::zeros(h, w, CV_8U);
    std::vector<Col> redraw;
    for (const Rule &r : rules) {
        const int len = r.x1 - r.x0; const double band = r.th / 2 + 2 * k;
        std::vector<char> aff(len, 0);
        for (int j = 0; j < len; j++) {
            const int x = r.x0 + j; const double yc = r.y(x);
            const int y1 = std::max(0, (int)(yc - band)), y2 = std::min(across(r) - 1, (int)(yc + band));
            for (int y = y1; y <= y2; y++) if (hw.at<uchar>(px(r, x, y))) { aff[j] = 1; break; }
        }
        if (std::none_of(aff.begin(), aff.end(), [](char c) { return c != 0; })) continue;
        cv::Vec3f colour;
        for (int c = 0; c < 3; c++) {
            std::vector<int> v;
            for (size_t i = 0; i < r.cleanX.size(); i++) v.push_back(original.at<cv::Vec3b>(px(r, r.cleanX[i], r.cleanY[i]))[c]);
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            colour[c] = (float)v[v.size() / 2];
        }
        for (int j = 0; j < len; j++) {
            bool grown = false;
            for (int t = std::max(0, j - k); t <= std::min(len - 1, j + k) && !grown; t++) grown = aff[t];
            if (!grown) continue;
            const int x = r.x0 + j; const double yc = r.y(x);
            for (int d = -(int)band; d <= (int)band; d++) {
                const int y = (int)std::lround(yc + d);
                if (y >= 0 && y < across(r)) clearZone.at<uchar>(px(r, x, y)) = 1;
            }
            redraw.push_back({&r, x, yc, colour});
        }
    }
    if (redraw.empty()) return;

    // pen leftovers near the rules: erased-page ink pieces (rule lines aside) lying in the zone and
    // inside the handwriting mask; printed letters never qualify
    cv::Mat gd, gp;
    cv::cvtColor(dst, gd, cv::COLOR_BGR2GRAY);
    cv::cvtColor(paper, gp, cv::COLOR_BGR2GRAY);
    cv::Mat diff; cv::subtract(gp, gd, diff, cv::noArray(), CV_16S);
    cv::Mat pieces = (diff > 25) & (lineMask == 0);
    cv::Mat lab;
    const int count = cv::connectedComponents(pieces, lab, 8, CV_32S);
    std::vector<int> cnt(count, 0), inZone(count, 0), inHw(count, 0), isPure(count, 0);
    for (int y = 0; y < h; y++) {
        const int *lp = lab.ptr<int>(y);
        for (int x = 0; x < w; x++) {
            const int l = lp[x]; if (!l) continue;
            cnt[l]++;
            if (clearZone.at<uchar>(y, x)) inZone[l]++;
            if (hwTight.at<uchar>(y, x)) inHw[l]++;
            if (pure.at<uchar>(y, x)) isPure[l]++;
        }
    }
    std::vector<char> go(count, 0);
    for (int l = 1; l < count; l++)
        go[l] = isPure[l] <= 0.2 * cnt[l] &&
                ((inZone[l] >= 0.85 * cnt[l] && inHw[l] >= 0.6 * cnt[l]) || (inZone[l] >= 0.4 * cnt[l] && inHw[l] >= 0.9 * cnt[l]));
    for (int y = 0; y < h; y++) {
        const int *lp = lab.ptr<int>(y);
        for (int x = 0; x < w; x++) if (go[lp[x]]) dst.at<cv::Vec3b>(y, x) = paper.at<cv::Vec3b>(y, x);
    }

    // redraw each affected column: the rule's own band anti-aliased, its rim cleared to paper
    // (another rule crossing here is left alone)
    for (const Col &col : redraw) {
        const Rule &r = *col.r; const double th = r.th;
        for (int y = (int)(col.yc - th / 2 - 2); y <= (int)(col.yc + th / 2 + 2); y++) {
            if (y < 0 || y >= across(r)) continue;
            const cv::Point p = px(r, col.x, y);
            const double cov = std::min(1.0, std::max(0.0, std::min(y + 0.5, col.yc + th / 2) - std::max(y - 0.5, col.yc - th / 2)));
            const cv::Vec3b pap = paper.at<cv::Vec3b>(p);
            if (cov > 0) {
                cv::Vec3b &o = dst.at<cv::Vec3b>(p);
                for (int c = 0; c < 3; c++) o[c] = cv::saturate_cast<uchar>(cov * col.colour[c] + (1 - cov) * pap[c]);
            } else if (!lineMask.at<uchar>(p) || std::abs(y - r.y(col.x)) <= th / 2 + 1.01) {
                dst.at<cv::Vec3b>(p) = pap;
            }
        }
    }
}

}  // namespace ruleredraw
