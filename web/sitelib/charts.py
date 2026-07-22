"""Inline-SVG chart builders.

All charts are styled through CSS classes (see style.py) so dark mode works
by variable swap; no hex lives in the SVG.  Every mark carries a <title>
tooltip.  Marks follow the house dataviz specs: 2px lines, >=8px hover
targets, recessive grid, selective direct labels, no dual axes.
"""
from __future__ import annotations

import html
from .race import value_points


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def _ticks(lo, hi, n=5):
    """Round-number tick positions covering [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** __import__("math").floor(__import__("math").log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw),
               default=10) * mag
    t = []
    v = (lo // step) * step
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            t.append(round(v, 10))
        v += step
    return t


def trajectory_chart(traj, w=780, h=280):
    """Monthly skill curve (converted to the points scale) with a 90% band.

    traj: [(month 'YYYY-MM', mean_logit, sd_logit)] chronological.
    """
    if len(traj) < 2:
        return ""
    ml, mr, mt, mb = 44, 12, 14, 30
    iw, ih = w - ml - mr, h - mt - mb
    pts = [(m, value_points(v), value_points(v - 1.645 * s), value_points(v + 1.645 * s))
           for m, v, s in traj]
    ymin = min(p[2] for p in pts); ymax = max(p[3] for p in pts)
    pad = max(0.4, (ymax - ymin) * 0.08); ymin -= pad; ymax += pad
    n = len(pts)

    def X(i): return ml + iw * i / (n - 1)
    def Y(v): return mt + ih * (1 - (v - ymin) / (ymax - ymin))

    grid, axis = [], []
    for tv in _ticks(ymin, ymax, 5):
        y = Y(tv)
        grid.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}"/>')
        axis.append(f'<text class="axis" x="{ml - 6}" y="{y + 3.5:.1f}" text-anchor="end">{tv:+g}</text>')
    for i, (m, *_rest) in enumerate(pts):
        if m.endswith("-01") or i == 0:
            axis.append(f'<text class="axis" x="{X(i):.1f}" y="{h - 8}" text-anchor="middle">{m[:4]}</text>')
            grid.append(f'<line class="grid" x1="{X(i):.1f}" y1="{mt}" x2="{X(i):.1f}" y2="{mt + ih}"/>')
    zero = Y(0)
    if ymin < 0 < ymax:
        grid.append(f'<line class="zeroline" x1="{ml}" y1="{zero:.1f}" x2="{w - mr}" y2="{zero:.1f}"/>')

    band_up = " ".join(f"{X(i):.1f},{Y(p[3]):.1f}" for i, p in enumerate(pts))
    band_dn = " ".join(f"{X(i):.1f},{Y(p[2]):.1f}"
                       for i, p in reversed(list(enumerate(pts))))
    line = " ".join(f"{X(i):.1f},{Y(p[1]):.1f}" for i, p in enumerate(pts))
    dots = []
    for i, (m, mid, lo, hi) in enumerate(pts):
        label = f"{m} · {mid:+.1f} pts (90% band {lo:+.1f} to {hi:+.1f})"
        dots.append(f'<circle class="hoverdot" cx="{X(i):.1f}" cy="{Y(mid):.1f}" r="7">'
                    f'<title>{esc(label)}</title></circle>')
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Monthly skill trajectory with uncertainty band">'
            f'{"".join(grid)}'
            f'<polygon class="band" points="{band_up} {band_dn}"/>'
            f'<polyline class="s1line" points="{line}"/>'
            f'{"".join(dots)}{"".join(axis)}'
            f'<text class="axis" transform="translate(12 {mt + ih / 2:.0f}) rotate(-90)" '
            f'text-anchor="middle">PICKLE score (vs avg pair)</text></svg>')


def gamelog_chart(log, w=780, h=260, max_games=200):
    """Actual point share per game (win/loss dots) vs the model's expected
    share (line).  Even x-spacing by game index; year ticks."""
    entries = log[-max_games:]
    if len(entries) < 5:
        return ""
    ml, mr, mt, mb = 44, 12, 14, 30
    iw, ih = w - ml - mr, h - mt - mb
    n = len(entries)

    def X(i): return ml + iw * i / max(n - 1, 1)
    def Y(v): return mt + ih * (1 - v)

    grid, axis = [], []
    for tv in (0.0, 0.25, 0.5, 0.75, 1.0):
        cls = "zeroline" if tv == 0.5 else "grid"
        grid.append(f'<line class="{cls}" x1="{ml}" y1="{Y(tv):.1f}" x2="{w - mr}" y2="{Y(tv):.1f}"/>')
        axis.append(f'<text class="axis" x="{ml - 6}" y="{Y(tv) + 3.5:.1f}" text-anchor="end">{int(tv * 100)}%</text>')
    seen_years = set()
    for i, e in enumerate(entries):
        y4 = e["date"][:4]
        if y4 not in seen_years:
            seen_years.add(y4)
            axis.append(f'<text class="axis" x="{X(i):.1f}" y="{h - 8}">{y4}</text>')
            grid.append(f'<line class="grid" x1="{X(i):.1f}" y1="{mt}" x2="{X(i):.1f}" y2="{mt + ih}"/>')

    expline = " ".join(f"{X(i):.1f},{Y(e['exp']):.1f}" for i, e in enumerate(entries)
                       if e["exp"] is not None)
    dots = []
    for i, e in enumerate(entries):
        cls = "windot" if e["won"] else "lossdot"
        tip = (f"{e['date']} {e['tour']} · {e['score']}"
               f"{' (OT)' if e['ot'] else ''} · vs {e['opp_names']}"
               f" · w/ {e['partner_name']}")
        if e["exp"] is not None:
            tip += f" · model expected {e['exp'] * 100:.0f}% of points"
        dots.append(f'<circle class="{cls}" cx="{X(i):.1f}" cy="{Y(e["share"]):.1f}" r="3.2">'
                    f'<title>{esc(tip)}</title></circle>')
    legend = (f'<g class="legend" transform="translate({ml + 6} {mt + 4})">'
              f'<circle class="windot" cx="5" cy="5" r="3.2"/><text x="13" y="9">win</text>'
              f'<circle class="lossdot" cx="52" cy="5" r="3.2"/><text x="60" y="9">loss</text>'
              f'<line class="s2line" x1="98" y1="5" x2="118" y2="5"/>'
              f'<text x="124" y="9">model expectation</text></g>')
    note = f'last {n} games' if len(log) > max_games else ""
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Point share per game vs model expectation">'
            f'{"".join(grid)}<polyline class="s2line" points="{expline}"/>'
            f'{"".join(dots)}{"".join(axis)}{legend}'
            f'<text class="axis" x="{w - mr}" y="{mt + 10}" text-anchor="end">{note}</text>'
            f'<text class="axis" transform="translate(12 {mt + ih / 2:.0f}) rotate(-90)" '
            f'text-anchor="middle">share of points</text></svg>')


def interval_cell(lo, mid, hi, xmin, xmax, w=240, h=20):
    """Small dot-and-interval SVG for ranking rows (shared scale per table)."""
    def X(v):
        v = min(max(v, xmin), xmax)
        return 6 + (w - 12) * (v - xmin) / (xmax - xmin)
    zero = X(0)
    return (f'<svg class="ivl" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'<line class="grid" x1="{zero:.1f}" y1="2" x2="{zero:.1f}" y2="{h - 2}"/>'
            f'<line class="s1line" x1="{X(lo):.1f}" y1="{h / 2}" x2="{X(hi):.1f}" y2="{h / 2}"/>'
            f'<circle class="s1dot" cx="{X(mid):.1f}" cy="{h / 2}" r="4">'
            f'<title>{mid:+.1f} pts (90% {lo:+.1f} to {hi:+.1f})</title></circle></svg>')


def calibration_chart(buckets, w=420, h=340):
    """Predicted-probability buckets vs realized win rate + diagonal."""
    ml, mr, mt, mb = 46, 14, 12, 34
    iw, ih = w - ml - mr, h - mt - mb

    def X(v): return ml + iw * (v - 0.5) / 0.5
    def Y(v): return mt + ih * (1 - (v - 0.5) / 0.5)

    parts = []
    for tv in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        parts.append(f'<line class="grid" x1="{X(tv):.1f}" y1="{mt}" x2="{X(tv):.1f}" y2="{mt + ih}"/>'
                     f'<line class="grid" x1="{ml}" y1="{Y(tv):.1f}" x2="{w - mr}" y2="{Y(tv):.1f}"/>'
                     f'<text class="axis" x="{X(tv):.1f}" y="{h - 16}" text-anchor="middle">{int(tv * 100)}%</text>'
                     f'<text class="axis" x="{ml - 6}" y="{Y(tv) + 3.5:.1f}" text-anchor="end">{int(tv * 100)}%</text>')
    parts.append(f'<line class="zeroline" x1="{X(0.5):.1f}" y1="{Y(0.5):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}"/>')
    for k, v in sorted(buckets.items()):
        pmid = float(k) + 0.05
        act, nn = v["actual"], v["n"]
        parts.append(f'<circle class="s1dot" cx="{X(pmid):.1f}" cy="{Y(min(act, 1)):.1f}" r="5">'
                     f'<title>forecast {int(float(k) * 100)}–{int(float(k) * 100) + 9}%: '
                     f'won {act * 100:.0f}% of {nn} games</title></circle>')
    parts.append(f'<text class="axis" x="{ml + iw / 2:.0f}" y="{h - 2}" text-anchor="middle">forecast win probability</text>')
    parts.append(f'<text class="axis" transform="translate(12 {mt + ih / 2:.0f}) rotate(-90)" '
                 f'text-anchor="middle">actual win rate</text>')
    return (f'<svg class="chart capped" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Calibration: forecast probability vs actual win rate">{"".join(parts)}</svg>')

