// Straightens a photographed page before handwriting analysis — a port of
// ml/prototype/straighten.py (keep the two in step; Android: Straightener.kt).
//
// 1. rules: fit the page's printed rules (table borders, writing grids) — horizontal rules'
//    angle as a function of height, vertical rule pieces' angle as a function of x, since
//    perspective makes both drift across the page — intersect the outermost into a quad and
//    warp it square. Fixes rotation AND keystone even with the sheet's edges out of frame.
// 2. no usable rules: deskew by the angle that levels the print (neutral ink only, so a
//    coloured pen can't drag it), applied only when clearly better than leaving it.
// 3. an already-level page (scanner output) is returned unchanged.
#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace straighten {

constexpr int kWork = 1200;          // analysis size (long side)
constexpr double kMaxDeg = 8.0;
constexpr double kLevelDeg = 0.6;    // quad sides within this of the axes = already level

struct Line { cv::Point2d p, d; double length, angle; };
struct Piece { double x, angle, length; };

inline cv::Mat InkMap(const cv::Mat &small, bool neutralOnly) {
    cv::Mat g, bg;
    cv::cvtColor(small, g, cv::COLOR_BGR2GRAY);
    cv::dilate(g, bg, cv::Mat::ones(7, 7, CV_8U));
    cv::medianBlur(bg, bg, 31);
    cv::Mat diff;
    cv::subtract(bg, g, diff, cv::noArray(), CV_16S);
    cv::Mat ink = diff > 25;
    if (neutralOnly) {
        cv::Mat lab; cv::cvtColor(small, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> ch; cv::split(lab, ch);
        cv::Mat a, b; ch[1].convertTo(a, CV_32F, 1, -128); ch[2].convertTo(b, CV_32F, 1, -128);
        cv::Mat chroma; cv::magnitude(a, b, chroma);
        ink &= chroma < 12;
    }
    return ink;
}

inline void FitLine(const cv::Mat &labels, int id, const cv::Rect &box, cv::Point2d *p, cv::Point2d *d) {
    std::vector<cv::Point2f> pts;
    for (int y = box.y; y < box.y + box.height; y++) {
        const int *row = labels.ptr<int>(y);
        for (int x = box.x; x < box.x + box.width; x++) if (row[x] == id) pts.emplace_back((float)x, (float)y);
    }
    cv::Vec4f l; cv::fitLine(pts, l, cv::DIST_HUBER, 0, 0.01, 0.01);
    *d = {l[0], l[1]}; *p = {l[2], l[3]};
}

/// Long (≥ 35 % of the width), thin, nearly level printed rules.
inline std::vector<Line> LongHorizontals(const cv::Mat &ink) {
    const int W = ink.cols;
    cv::Mat lines;
    cv::morphologyEx(ink, lines, cv::MORPH_OPEN, cv::Mat::ones(1, std::max(15, W / 25), CV_8U));
    // re-join a rule the photo's slope or a pen crossing broke into pieces
    cv::morphologyEx(lines, lines, cv::MORPH_CLOSE, cv::Mat::ones(3, 3, CV_8U));
    cv::dilate(lines, lines, cv::Mat::ones(3, 1, CV_8U));
    cv::Mat labels, stats, cents;
    int n = cv::connectedComponentsWithStats(lines, labels, stats, cents, 8, CV_32S);
    std::vector<Line> out;
    for (int i = 1; i < n; i++) {
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (w < 0.35 * W || h > 0.08 * W) continue;
        cv::Point2d p, d;
        FitLine(labels, i, cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), w, h), &p, &d);
        double angle = std::atan2(d.y, d.x) * 180 / CV_PI;
        angle = std::fmod(angle + 90 + 360, 180.0) - 90;
        if (std::abs(angle) > kMaxDeg) continue;
        double norm = std::hypot(d.x, d.y);
        out.push_back({p, {d.x / norm, d.y / norm}, (double)w, angle});
    }
    return out;
}

/// Short vertical rule pieces (grid / table columns).
inline std::vector<Piece> ShortVerticals(const cv::Mat &ink) {
    const int H = ink.rows, W = ink.cols;
    cv::Mat v;
    cv::morphologyEx(ink, v, cv::MORPH_OPEN, cv::Mat::ones(std::max(12, H / 60), 1, CV_8U));
    cv::Mat labels, stats, cents;
    int n = cv::connectedComponentsWithStats(v, labels, stats, cents, 8, CV_32S);
    std::vector<Piece> out;
    for (int i = 1; i < n; i++) {
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (h < 0.04 * H || w > 0.02 * W) continue;
        cv::Point2d p, d;
        FitLine(labels, i, cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), w, h), &p, &d);
        if (std::abs(d.y) < std::abs(d.x)) continue;
        if (d.y < 0) d = -d;
        double angle = std::atan2(-d.x, d.y) * 180 / CV_PI;   // 0 = vertical
        if (std::abs(angle) <= kMaxDeg) out.push_back({cents.at<double>(i, 0), angle, (double)h});
    }
    return out;
}

/// angle = c0 * t + c1 by weighted least squares (weights squared, as numpy.polyfit's `w`);
/// with `linear` false, the length-weighted mean angle.
inline cv::Vec2d FitAngle(const std::vector<double> &t, const std::vector<double> &a, const std::vector<double> &w, bool linear) {
    double sw = 0, sa = 0;
    for (size_t i = 0; i < t.size(); i++) { sw += w[i]; sa += w[i] * a[i]; }
    if (!linear) return {0, sa / sw};
    double s2 = 0, st = 0, sa2 = 0;
    for (size_t i = 0; i < t.size(); i++) { double w2 = w[i] * w[i]; s2 += w2; st += w2 * t[i]; sa2 += w2 * a[i]; }
    double mt = st / s2, ma = sa2 / s2;
    double num = 0, den = 0;
    for (size_t i = 0; i < t.size(); i++) { double w2 = w[i] * w[i]; num += w2 * (t[i] - mt) * (a[i] - ma); den += w2 * (t[i] - mt) * (t[i] - mt); }
    double c0 = den > 1e-9 ? num / den : 0;
    return {c0, ma - c0 * mt};
}

inline bool Intersect(cv::Point2d p1, cv::Point2d d1, cv::Point2d p2, cv::Point2d d2, cv::Point2d *out) {
    double det = d1.x * -d2.y - (-d2.x) * d1.y;
    if (std::abs(det) < 1e-6) return false;
    cv::Point2d r = p2 - p1;
    double t = (r.x * -d2.y - (-d2.x) * r.y) / det;
    *out = p1 + t * d1;
    return true;
}

inline double Percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    double idx = q / 100.0 * (v.size() - 1);
    size_t lo = (size_t)std::floor(idx), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

/// Quad (in `small` coordinates) squaring the page's rule field; empty when there isn't one.
inline std::vector<cv::Point2f> RulesQuad(const cv::Mat &small) {
    cv::Mat ink = InkMap(small, false);
    const int H = ink.rows, W = ink.cols;
    auto hs = LongHorizontals(ink);
    auto vs = ShortVerticals(ink);
    if (hs.size() < 2) return {};
    std::vector<double> hy, ha, hl;
    for (auto &l : hs) { hy.push_back(l.p.y + (W / 2.0 - l.p.x) * l.d.y / l.d.x); ha.push_back(l.angle); hl.push_back(l.length); }
    double yTop = *std::min_element(hy.begin(), hy.end()), yBot = *std::max_element(hy.begin(), hy.end());
    if (yBot - yTop < 0.25 * H) return {};
    cv::Vec2d fh = FitAngle(hy, ha, hl, hs.size() >= 3);
    cv::Vec2d fv; double xL = 0.1 * W, xR = 0.9 * W;
    if (vs.size() >= 6) {
        std::vector<double> vx, va, vl;
        for (auto &p : vs) { vx.push_back(p.x); va.push_back(p.angle); vl.push_back(p.length); }
        double spread = *std::max_element(vx.begin(), vx.end()) - *std::min_element(vx.begin(), vx.end());
        fv = FitAngle(vx, va, vl, spread > 0.3 * W);
        xL = Percentile(vx, 5); xR = Percentile(vx, 95);
    } else {
        // no verticals to read: keep them square to the mean horizontal (rotation only)
        fv = FitAngle(hy, ha, hl, false);
    }
    if (xR - xL < 0.3 * W) { xL = 0.1 * W; xR = 0.9 * W; }
    auto hline = [&](double y, cv::Point2d *p, cv::Point2d *d) {
        double a = (fh[0] * y + fh[1]) * CV_PI / 180; *p = {W / 2.0, y}; *d = {std::cos(a), std::sin(a)};
    };
    auto vline = [&](double x, cv::Point2d *p, cv::Point2d *d) {
        double a = (fv[0] * x + fv[1]) * CV_PI / 180; *p = {x, H / 2.0}; *d = {-std::sin(a), std::cos(a)};
    };
    std::vector<cv::Point2f> quad;
    const double corners[4][2] = {{yTop, xL}, {yTop, xR}, {yBot, xR}, {yBot, xL}};
    for (auto &c : corners) {
        cv::Point2d p1, d1, p2, d2, x;
        hline(c[0], &p1, &d1); vline(c[1], &p2, &d2);
        if (!Intersect(p1, d1, p2, d2, &x)) return {};
        quad.emplace_back((float)x.x, (float)x.y);
    }
    return quad;
}

inline double SideAngle(cv::Point2f a, cv::Point2f b, bool horizontal) {
    cv::Point2f v = b - a;
    return horizontal ? std::atan2(v.y, v.x) * 180 / CV_PI : std::atan2(-v.x, v.y) * 180 / CV_PI;
}

/// Squares `quad` with a homography applied to the WHOLE image (nothing outside the quad is
/// cropped away); empty when the fit is degenerate.
inline cv::Mat WarpKeepPage(const cv::Mat &img, const std::vector<cv::Point2f> &q) {
    double w = (cv::norm(q[1] - q[0]) + cv::norm(q[2] - q[3])) / 2;
    double h = (cv::norm(q[3] - q[0]) + cv::norm(q[2] - q[1])) / 2;
    std::vector<cv::Point2f> dst = {q[0], {q[0].x + (float)w, q[0].y}, {q[0].x + (float)w, q[0].y + (float)h}, {q[0].x, q[0].y + (float)h}};
    cv::Mat M = cv::getPerspectiveTransform(q, dst);
    std::vector<cv::Point2f> frame = {{0, 0}, {(float)img.cols, 0}, {(float)img.cols, (float)img.rows}, {0, (float)img.rows}}, warped;
    cv::perspectiveTransform(frame, warped, M);
    float x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (auto &p : warped) { x0 = std::min(x0, p.x); y0 = std::min(y0, p.y); x1 = std::max(x1, p.x); y1 = std::max(y1, p.y); }
    cv::Size size((int)std::ceil(x1 - x0), (int)std::ceil(y1 - y0));
    if (size.width > 1.6 * img.cols || size.height > 1.6 * img.rows) return {};
    cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, -x0, 0, 1, -y0, 0, 0, 1);
    cv::Mat out;
    cv::warpPerspective(img, out, T * M, size, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    return out;
}

inline double ProjectionScore(const cv::Mat &ink, double deg) {
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(ink.cols / 2.f, ink.rows / 2.f), deg, 1), r;
    cv::warpAffine(ink, r, M, ink.size());
    cv::Mat prof; cv::reduce(r, prof, 1, cv::REDUCE_SUM, CV_64F);
    double s = 0;
    for (int i = 1; i < prof.rows; i++) { double dd = prof.at<double>(i) - prof.at<double>(i - 1); s += dd * dd; }
    return s;
}

inline cv::Mat Rotate(const cv::Mat &img, double deg) {
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(img.cols / 2.f, img.rows / 2.f), deg, 1);
    double c = std::abs(M.at<double>(0, 0)), s = std::abs(M.at<double>(0, 1));
    int nW = (int)(img.rows * s + img.cols * c), nH = (int)(img.rows * c + img.cols * s);
    M.at<double>(0, 2) += nW / 2.0 - img.cols / 2.0;
    M.at<double>(1, 2) += nH / 2.0 - img.rows / 2.0;
    cv::Mat out;
    cv::warpAffine(img, out, M, cv::Size(nW, nH), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    return out;
}

/// The straightened page; `how` = "level" (unchanged), "rules" or "rotate".
inline cv::Mat Straighten(const cv::Mat &img, std::string *how) {
    double s = (double)kWork / std::max(img.cols, img.rows);
    cv::Mat small; cv::resize(img, small, cv::Size(), s, s, cv::INTER_AREA);
    auto quad = RulesQuad(small);
    if (!quad.empty()) {
        double maxSide = std::max({std::abs(SideAngle(quad[0], quad[1], true)), std::abs(SideAngle(quad[3], quad[2], true)),
                                   std::abs(SideAngle(quad[0], quad[3], false)), std::abs(SideAngle(quad[1], quad[2], false))});
        if (maxSide < kLevelDeg) { *how = "level"; return img.clone(); }
        for (auto &p : quad) p *= (float)(1 / s);
        cv::Mat out = WarpKeepPage(img, quad);
        if (!out.empty()) { *how = "rules"; return out; }
    }
    cv::Mat ink = InkMap(small, true);
    double best = -1, bestA = 0;
    for (double a = -kMaxDeg; a <= kMaxDeg + 1e-6; a += 0.25) { double sc = ProjectionScore(ink, a); if (sc > best) { best = sc; bestA = a; } }
    double coarse = bestA; best = -1;
    for (double a = coarse - 0.25; a <= coarse + 0.25 + 1e-6; a += 0.05) { double sc = ProjectionScore(ink, a); if (sc > best) { best = sc; bestA = a; } }
    double conf = best / std::max(ProjectionScore(ink, 0), 1e-6);
    if (std::abs(bestA) >= 0.3 && conf >= 1.15) { *how = "rotate"; return Rotate(img, bestA); }
    *how = "level";
    return img.clone();
}

}  // namespace straighten
