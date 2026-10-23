// Last erase step on grid paper (Vietnamese "ô ly" exercise books): the grid drawn again where the
// erase took it with the writing. Mirrors GridRedraw.kt on Android (same steps, same constants).
//
// The grid's faint lines are found where the page was never written on, and each one is taken as
// straight, end to end: from the grid's frame to the frame, not where the writing hid it (lines
// missing between found ones are added at the grid's pitch). Through the erased area each line is
// painted with its own look — its colour and width measured beside each painted point on the
// original — and only on paper: ink the erase kept is never painted over.
//
// Only on grid paper: thin vertical lines repeating at one pitch (see VerticalRhythm).
#pragma once

#import <opencv2/opencv.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <vector>

namespace gridredraw {

/// The lines are found (and painted) at this long side at most (the grid is the same, found faster).
static const int kWorkLong = 1536;
/// At least this many lines each way: a grid, not a table or a few answer lines.
static const int kMinLines = 8;
static const int kBand = 64;
/// Prominence of the column profile's periodic peak that makes the page grid paper.
static const double kMinRhythm = 0.25;
/// A grid's pitch at the working scale at most (px); wider repeats are table columns.
static const int kMaxPitch = 40;

/// A straight line: across = a·t + b for t in [t0, t1] (t runs along it).
struct Line {
    double a, b, t0, t1;
    double At(double t) const { return a * t + b; }
};

namespace detail {

/// Kotlin's roundToInt (half up).
inline int Round(double v) { return (int)std::floor(v + 0.5); }

/// The paper's brightness: a wide median over the page with its strokes closed away (CV_32F).
inline cv::Mat Paper(const cv::Mat &g) {
    cv::Mat d;
    cv::dilate(g, d, cv::Mat::ones(7, 7, CV_8U));
    cv::medianBlur(d, d, 31);
    cv::Mat f; d.convertTo(f, CV_32F);
    return f;
}

inline std::vector<float> Floats(const cv::Mat &m) {
    cv::Mat c = m.isContinuous() ? m : m.clone();
    return std::vector<float>((const float *)c.data, (const float *)c.data + c.total());
}

inline std::vector<uchar> Bytes(const cv::Mat &m) {
    cv::Mat c = m.isContinuous() ? m : m.clone();
    return std::vector<uchar>(c.data, c.data + c.total());
}

/// The best periodic peak of the share of clean pixels per column that are darker than 3 px to
/// either side (a thin vertical line): its prominence over the lowest autocorrelation before it, and
/// its lag (the pitch). (0, 0) when there is none.
inline void VerticalRhythm(const std::vector<float> &dark, const std::vector<char> &clean, int w, int h,
                           double &prominence, int &pitch) {
    prominence = 0; pitch = 0;
    std::vector<double> prof(w, 0.0);
    for (int c = 3; c < w - 3; c++) {
        int cov = 0, hit = 0;
        for (int y = 0; y < h; y++) {
            const int i = y * w + c;
            if (!clean[i]) continue;
            cov++;
            if (dark[i] - std::max(dark[i - 3], dark[i + 3]) > 2.5f) hit++;
        }
        prof[c] = cov > 0 ? (double)hit / cov : 0.0;
    }
    double mean = 0; for (double v : prof) mean += v;
    mean /= std::max(1, w);
    for (double &v : prof) v -= mean;
    const int maxLag = 61;
    std::vector<double> ac(maxLag + 2, 0.0);
    for (int l = 0; l <= maxLag + 1; l++) {
        double sum = 0;
        for (int c = 0; c < w - l; c++) sum += prof[c] * prof[c + l];
        ac[l] = sum;
    }
    if (ac[0] <= 1e-12) return;
    const double a0 = ac[0];
    for (double &v : ac) v /= a0;
    for (int l = 6; l < maxLag; l++) {
        if (ac[l] < ac[l - 1] || ac[l] < ac[l + 1]) continue;
        double lo = 1e300; for (int q = 1; q < l; q++) lo = std::min(lo, ac[q]);
        if (ac[l] - lo > prominence) { prominence = ac[l] - lo; pitch = l; }
    }
}

/// Least-squares cs = a·ts + b.
inline void Fit(const std::vector<double> &ts, const std::vector<double> &cs, double &a, double &b) {
    const int n = (int)ts.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) { sx += ts[i]; sy += cs[i]; sxx += ts[i] * ts[i]; sxy += ts[i] * cs[i]; }
    const double den = n * sxx - sx * sx;
    a = std::abs(den) > 1e-9 ? (n * sxy - sx * sy) / den : 0.0;
    b = (sy - a * sx) / n;
}

/// Lines running along one direction (`vertical`: columns): per band across them, the share of clean
/// pixels where the page is darker than 3 px to either side, across the line; peaks of that share
/// are line points, linked into straight lines.
inline std::vector<Line> FindLines(const std::vector<float> &dark, const std::vector<char> &clean, int w, int h, bool vertical) {
    const int L = vertical ? h : w;   // along
    const int A = vertical ? w : h;   // across
    auto at = [&](int t, int c) { return vertical ? t * w + c : c * w + t; };
    std::vector<std::pair<double, double>> pts;   // (t, across)
    for (int t0 = 0; t0 + kBand <= L; t0 += kBand / 2) {
        std::vector<float> on(std::max(0, A), 0.f);
        for (int c = 3; c < A - 3; c++) {
            int cov = 0, hit = 0;
            for (int t = t0; t < t0 + kBand; t++) {
                const int i = at(t, c);
                if (!clean[i]) continue;
                cov++;
                const float side = std::max(dark[at(t, c - 3)], dark[at(t, c + 3)]);
                if (dark[i] - side > 2.5f) hit++;
            }
            on[c] = cov < kBand * 0.25 ? 0.f : (float)hit / cov;
        }
        for (int c = 3; c < A - 3; c++)
            if (on[c] >= 0.45f && on[c] >= on[c - 1] && on[c] > on[c + 1]) pts.push_back({t0 + kBand / 2.0, (double)c});
    }
    // link points into straight lines, allowing a drift of 2.5 px
    std::sort(pts.begin(), pts.end(), [](const std::pair<double, double> &p, const std::pair<double, double> &q) {
        return p.second != q.second ? p.second < q.second : p.first < q.first;
    });
    std::vector<std::pair<std::vector<double>, std::vector<double>>> groups;
    for (const auto &p : pts) {
        int best = -1;
        for (int g = 0; g < (int)groups.size(); g++) {
            const auto &ts = groups[g].first; const auto &cs = groups[g].second;
            if (std::abs(ts.back() - p.first) < 1e-6) continue;
            double pc;
            if (ts.size() >= 2) { double fa, fb; Fit(ts, cs, fa, fb); pc = fa * p.first + fb; } else pc = cs.back();
            if (std::abs(pc - p.second) <= 2.5 && std::abs(ts.back() - p.first) <= kBand * 8) { best = g; break; }
        }
        if (best < 0) groups.push_back({{p.first}, {p.second}});
        else { groups[best].first.push_back(p.first); groups[best].second.push_back(p.second); }
    }
    std::vector<Line> out;
    for (const auto &g : groups) {
        const auto &ts = g.first; const auto &cs = g.second;
        if (ts.size() < 4) continue;
        double fa, fb; Fit(ts, cs, fa, fb);
        bool straight = true;
        for (size_t i = 0; i < ts.size(); i++) if (std::abs(cs[i] - (fa * ts[i] + fb)) > 1.5) { straight = false; break; }
        if (!straight) continue;
        out.push_back({fa, fb, *std::min_element(ts.begin(), ts.end()) - kBand / 2.0, *std::max_element(ts.begin(), ts.end()) + kBand / 2.0});
    }
    return out;
}

/// Grid lines repeat at one pitch: lines missing between found ones (hidden by the writing) are added.
inline std::vector<Line> Complete(const std::vector<Line> &lines0, int along) {
    if (lines0.size() < 6) return lines0;
    const double mid = along / 2.0;
    std::vector<Line> sorted = lines0;
    std::stable_sort(sorted.begin(), sorted.end(), [mid](const Line &p, const Line &q) { return p.At(mid) < q.At(mid); });
    std::vector<Line> lines;
    for (const Line &l : sorted) {
        if (!lines.empty() && std::abs(l.At(mid) - lines.back().At(mid)) < 3) {
            if (l.t1 - l.t0 > lines.back().t1 - lines.back().t0) lines.back() = l;
        } else lines.push_back(l);
    }
    std::vector<double> gaps;
    for (size_t i = 0; i + 1 < lines.size(); i++) gaps.push_back(lines[i + 1].At(mid) - lines[i].At(mid));
    std::vector<Line> out = lines;
    for (int g = 0; g < (int)gaps.size(); g++) {
        std::vector<double> win;
        for (int j = std::max(0, g - 6); j < std::min((int)gaps.size(), g + 7); j++) if (gaps[j] >= 6) win.push_back(gaps[j]);
        if (win.empty()) continue;
        const double mn = *std::min_element(win.begin(), win.end());
        std::vector<double> singles;
        for (double v : win) if (v <= 1.4 * mn) singles.push_back(v);
        if (singles.size() < 3) continue;
        std::sort(singles.begin(), singles.end());
        const double p = singles[singles.size() / 2];
        const int q = Round(gaps[g] / p);
        if (q < 2 || q > 6 || std::abs(gaps[g] / q - p) > 0.15 * p) continue;
        const Line l0 = lines[g], l1 = lines[g + 1];
        for (int r = 1; r < q; r++) {
            const double f = (double)r / q;
            out.push_back({l0.a + (l1.a - l0.a) * f, l0.b + (l1.b - l0.b) * f, std::min(l0.t0, l1.t0), std::max(l0.t1, l1.t1)});
        }
    }
    return out;
}

/// A straight line runs whole, between the outermost lines across it that reach it (the frame).
inline void Span(std::vector<Line> &lines, const std::vector<Line> &cross) {
    for (Line &l : lines) {
        std::vector<double> pos;
        for (const Line &c : cross) {
            const double den = 1 - l.a * c.a;
            if (std::abs(den) < 1e-6) continue;
            const double t = (c.a * l.b + c.b) / den, acr = l.a * t + l.b;
            if (acr >= c.t0 - 6 && acr <= c.t1 + 6 && t >= l.t0 - 400 && t <= l.t1 + 400) pos.push_back(t);
        }
        if (pos.size() >= 2) {
            l.t0 = std::min(*std::min_element(pos.begin(), pos.end()), l.t0);
            l.t1 = std::max(*std::max_element(pos.begin(), pos.end()), l.t1);
        }
    }
}

/// One printed line found twice (its two edges): within 4 px along its run, the longer one.
inline std::vector<Line> Dedupe(const std::vector<Line> &lines, int along) {
    const double mid = along / 2.0;
    std::vector<Line> sorted = lines;
    std::stable_sort(sorted.begin(), sorted.end(), [mid](const Line &p, const Line &q) { return p.At(mid) < q.At(mid); });
    std::vector<Line> out;
    for (const Line &l : sorted) {
        if (!out.empty() && std::abs(l.At(mid) - out.back().At(mid)) < 4) {
            if (l.t1 - l.t0 > out.back().t1 - out.back().t0) out.back() = l;
        } else out.push_back(l);
    }
    return out;
}

inline int LowerBound(const std::vector<int> &v, int x) {
    return (int)(std::lower_bound(v.begin(), v.end(), x) - v.begin());
}

/// Paints `vl` / `hl` through the erased area of `dst` (BGR, all at one scale); returns the pixels painted.
inline int Paint(const cv::Mat &original, cv::Mat &dst, const cv::Mat &handwriting, const cv::Mat &print,
                 const std::vector<Line> &vl, const std::vector<Line> &hl, int k) {
    const int fw = dst.cols, fh = dst.rows, n = fw * fh;
    cv::Mat zoneF, farM;
    cv::dilate(handwriting, zoneF, cv::Mat::ones(2 * k + 1, 2 * k + 1, CV_8U));
    cv::dilate(zoneF, farM, cv::Mat::ones(5, 5, CV_8U));
    const std::vector<uchar> zone = Bytes(zoneF), nearZone = Bytes(farM);
    // the erased area cleaned to the paper's own colour first (no grey eraser haze under the lines);
    // print and the ink the erase kept stay as they are
    std::vector<float> pap[3], outF[3];
    {
        std::vector<cv::Mat> ch; cv::split(dst, ch);
        for (int c = 0; c < 3; c++) {
            pap[c] = Floats(Paper(ch[c]));
            cv::Mat f; ch[c].convertTo(f, CV_32F); outF[c] = Floats(f);
        }
    }
    cv::Mat prN; cv::dilate(print, prN, cv::Mat::ones(3, 3, CV_8U));
    const std::vector<uchar> prn = Bytes(prN);
    auto keptInk = [&]() {
        cv::Mat gm(fh, fw, CV_32F);
        float *g = (float *)gm.data;
        for (int i = 0; i < n; i++) g[i] = 0.114f * outF[0][i] + 0.587f * outF[1][i] + 0.299f * outF[2][i];
        cv::Mat g8; gm.convertTo(g8, CV_8U);
        const std::vector<float> pa = Floats(Paper(g8));
        std::vector<char> ink(n);
        for (int i = 0; i < n; i++) ink[i] = pa[i] - g[i] > 25.f;
        return ink;
    };
    std::vector<char> inkd = keptInk();
    for (int i = 0; i < n; i++)
        if (zone[i] && !prn[i] && !inkd[i]) for (int c = 0; c < 3; c++) outF[c][i] = pap[c][i];
    inkd = keptInk();
    // the original's look relative to its paper: a line's cross-section
    std::vector<float> ratio[3];
    {
        std::vector<cv::Mat> ch; cv::split(original, ch);
        for (int c = 0; c < 3; c++) {
            const std::vector<float> pa = Floats(Paper(ch[c]));
            cv::Mat fm; ch[c].convertTo(fm, CV_32F);
            const std::vector<float> f = Floats(fm);
            ratio[c].resize(n);
            for (int i = 0; i < n; i++) ratio[c][i] = f[i] / std::max(pa[i], 1.f);
        }
    }
    const int R = 2;            // half the cross-section, px
    const int W = 2 * R + 1;
    const int nearT = 120;      // clean samples this far along the line make its local profile
    const int reprofile = 8;    // the profile is measured again every this many px along the line
    const int maxW = 3;         // a grid line's width at most
    int painted = 0;
    for (int pass = 0; pass < 2; pass++) {
        const bool vert = pass == 0;
        for (const Line &ln : vert ? vl : hl) {
            const int along = vert ? fh : fw, across = vert ? fw : fh;
            const double a = ln.a, b = ln.b;
            const int ta = std::max(0, (int)ln.t0), tb = std::min(along, (int)ln.t1);
            if (tb - ta < 10) continue;
            auto idx = [&](int t, int c) { return vert ? t * fw + c : c * fw + t; };
            std::vector<int> cleanT, zoneT;
            for (int t = ta; t < tb; t++) {
                const int c = Round(a * t + b);
                if (c < R + 1 || c >= across - R - 1) continue;
                const int i = idx(t, c);
                if (zone[i]) zoneT.push_back(t); else if (!nearZone[i]) cleanT.push_back(t);
            }
            if (cleanT.size() < 20 || zoneT.empty()) continue;
            // per clean sample, the cross-section (offsets -R..R, three channels)
            std::vector<std::vector<float>> samp(cleanT.size(), std::vector<float>(W * 3));
            for (size_t j = 0; j < cleanT.size(); j++) {
                const int t = cleanT[j], c0 = Round(a * t + b);
                for (int q = 0; q < W * 3; q++) samp[j][q] = ratio[q % 3][idx(t, c0 + q / 3 - R)];
            }
            auto profileNear = [&](int t) {
                int lo = LowerBound(cleanT, t - nearT);
                const int hi = LowerBound(cleanT, t + nearT + 1);
                if (hi - lo < 8) lo = 0;
                int from = 0, to = (int)cleanT.size();
                if (hi - lo >= 8) { from = lo; to = hi; }
                std::vector<float> prof(W * 3), v;
                for (int q = 0; q < W * 3; q++) {
                    v.clear();
                    for (int j = from; j < to; j++) v.push_back(samp[j][q]);
                    if (v.empty()) { prof[q] = 1.f; continue; }
                    std::sort(v.begin(), v.end());
                    prof[q] = v[v.size() / 2];
                }
                return prof;
            };
            const std::vector<float> whole = profileNear(-1000000);
            bool visible = false;
            for (int q = 0; q < W * 3; q++) if (whole[q] <= 0.985f) { visible = true; break; }
            if (!visible) continue;   // no visible line here
            std::vector<float> prof = whole;
            int lastT = -1000000;
            for (int t : zoneT) {
                // 32-bit wrapping difference, as GridRedraw.kt computes it (Int arithmetic)
                if (t - lastT >= reprofile) { prof = profileNear(t); lastT = t; }
                // the line's own width: offsets clearly darker than the paper, around the centre
                bool dk[W];
                for (int o = 0; o < W; o++) dk[o] = std::min(prof[o * 3], std::min(prof[o * 3 + 1], prof[o * 3 + 2])) < 0.985f;
                int centre = 0, bestScore = INT_MAX;
                for (int o = 0; o < W; o++) {
                    const int score = std::abs(o - R) + (dk[o] ? 0 : 1000);
                    if (score < bestScore) { bestScore = score; centre = o; }
                }
                if (!dk[centre] || std::abs(centre - R) > std::max(1, R / 2)) continue;
                int lo = centre, hi = centre;
                while (lo - 1 >= 0 && dk[lo - 1] && hi - lo + 1 < maxW) lo--;
                while (hi + 1 < W && dk[hi + 1] && hi - lo + 1 < maxW) hi++;
                const int c0 = Round(a * t + b);
                for (int o = lo; o <= hi; o++) {
                    const int c = c0 + o - R;
                    if (c < 0 || c >= across) continue;
                    const int i = idx(t, c);
                    if (!zone[i] || inkd[i]) continue;
                    for (int ch = 0; ch < 3; ch++) outF[ch][i] = std::min(outF[ch][i], pap[ch][i] * prof[o * 3 + ch]);
                    painted++;
                }
            }
        }
    }
    std::vector<cv::Mat> chs(3);
    for (int c = 0; c < 3; c++) {
        cv::Mat m(fh, fw, CV_32F, outF[c].data());
        m.convertTo(chs[c], CV_8U);
    }
    cv::merge(chs, dst);
    return painted;
}

}  // namespace detail

/// `original`: the page before the erase (BGR); `dst`: the erased page (BGR), painted in place;
/// `handwriting` / `print`: the mask layers (CV_8UC1 255 = set), `dst`'s size.
inline void Apply(const cv::Mat &original, cv::Mat &dst, const cv::Mat &handwriting, const cv::Mat &print) {
    using namespace detail;
    if (cv::countNonZero(handwriting) == 0) return;
    const int fw = dst.cols, fh = dst.rows;
    const double s = std::min(1.0, (double)kWorkLong / std::max(fw, fh));
    const int w = std::max(1, Round(fw * s)), h = std::max(1, Round(fh * s));
    const int k = std::max(5, std::max(w, h) / 450) | 1;
    const cv::Size sz(w, h);

    // ---- find the lines at the working scale, where the page was never written on
    std::vector<float> dark;
    std::vector<char> clean;
    {
        cv::Mat o, g, gf, d, hs, zone;
        cv::resize(original, o, sz, 0, 0, cv::INTER_AREA);
        cv::cvtColor(o, g, cv::COLOR_BGR2GRAY);
        const cv::Mat p = Paper(g);
        g.convertTo(gf, CV_32F);
        cv::subtract(p, gf, d);
        cv::max(d, 0.0, d);
        dark = Floats(d);
        cv::resize(handwriting, hs, sz, 0, 0, cv::INTER_NEAREST);
        cv::dilate(hs, zone, cv::Mat::ones(2 * k + 1, 2 * k + 1, CV_8U));
        const std::vector<uchar> zb = Bytes(zone);
        clean.resize(zb.size());
        for (size_t i = 0; i < zb.size(); i++) clean[i] = zb[i] == 0;
    }
    // grid paper: the faint vertical lines repeat at one pitch across the clean page — a clear
    // periodic peak in the column profile of thin dark lines. Text, tables and answer lines have no
    // such vertical rhythm (measured on 30 pages: grid pages 0.31-0.72, the others at most 0.20)
    double prominence; int pitch;
    VerticalRhythm(dark, clean, w, h, prominence, pitch);
    if (prominence < kMinRhythm || pitch > kMaxPitch) return;
    std::vector<Line> vl = FindLines(dark, clean, w, h, true);
    std::vector<Line> hl = FindLines(dark, clean, w, h, false);
    vl = Complete(vl, h); hl = Complete(hl, w);
    Span(vl, hl); Span(hl, vl);
    vl = Dedupe(vl, h); hl = Dedupe(hl, w);
    if ((int)vl.size() < kMinLines || (int)hl.size() < kMinLines) return;

    // ---- paint at the working scale too; the pixels it changes are carried back to the page
    cv::Mat oS, dS, hS, pS;
    if (s < 1.0) {
        cv::resize(original, oS, sz, 0, 0, cv::INTER_AREA); cv::resize(dst, dS, sz, 0, 0, cv::INTER_AREA);
        cv::resize(handwriting, hS, sz, 0, 0, cv::INTER_NEAREST); cv::resize(print, pS, sz, 0, 0, cv::INTER_NEAREST);
    } else { original.copyTo(oS); dst.copyTo(dS); handwriting.copyTo(hS); print.copyTo(pS); }
    const int painted = Paint(oS, dS, hS, pS, vl, hl, k);
    if (painted == 0) return;
    if (s < 1.0) {
        // changed pixels only, carried up; elsewhere the page keeps its full resolution
        cv::Mat before, diff, changed, up, chUp;
        cv::resize(dst, before, sz, 0, 0, cv::INTER_AREA);
        cv::absdiff(dS, before, diff);
        cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
        cv::threshold(diff, changed, 0, 255, cv::THRESH_BINARY);
        cv::resize(dS, up, cv::Size(fw, fh), 0, 0, cv::INTER_LINEAR);
        cv::resize(changed, chUp, cv::Size(fw, fh), 0, 0, cv::INTER_NEAREST);
        up.copyTo(dst, chUp);
    } else dS.copyTo(dst);
}

}  // namespace gridredraw
