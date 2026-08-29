"""Backend agreement guard — the spec's pre-named validation.

coverage_spec.md: before trusting the RTMPose fleet, extract a handful
of rallies with BOTH backends and confirm the coverage metrics agree
within a few percent; where they disagree, the ViTPose number wins and
the discrepancy gets diagnosed before any scale-out.

    # 1. extract the same rallies twice (GPU box for vitpose):
    python vision/pose_extract.py --video vod.mp4 --labels /nonexistent.csv \
        --windows data/vision/coverage_windows_<vod>.csv --rallies 20,21,22,23,24 \
        --backend rtmpose --fps 10 --out-dir /tmp/ab_rtm
    python vision/pose_extract.py ... --backend vitpose --device cuda \
        --fps 10 --out-dir /tmp/ab_vit
    # 2. compare:
    python vision/coverage_ab.py --pose-a /tmp/ab_rtm --pose-b /tmp/ab_vit \
        --court data/vision/court_<vod>.json \
        --windows data/vision/coverage_windows_<vod>.csv \
        --lineup data/vision/lineup_<id8>.csv [--cam cam.csv]

Prints per-player ellipse-area and width-share deltas plus a PASS/FLAG
verdict (thresholds: 8% relative on area, 0.02 absolute on width share
— generous for "a few percent" but tight enough that a systematic
keypoint bias shows).  python vision/coverage_ab.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage as C

AREA_REL_TOL = 0.08
SHARE_ABS_TOL = 0.02


def run_one(pose_dir, a, tag, tmp):
    C.DATA = Path(tmp)
    ns = types.SimpleNamespace(
        pose_dir=str(pose_dir), court=a.court, windows=a.windows,
        lineup=a.lineup, cam=a.cam or "", spotcheck="",
        vod=f"ab_{tag}", event="", date="", match_id="")
    C.run(ns)
    rows = {}
    f = Path(tmp) / "coverage_players.csv"
    if f.exists():
        for r in csv.DictReader(open(f)):
            if r["vod"] == f"ab_{tag}":
                rows[(r["game"], r["player_uuid"])] = r
    return rows


def compare(rows_a, rows_b):
    """-> (report lines, n_flagged).  Pure, so the selftest can drive it."""
    lines, flagged = [], 0
    keys = sorted(set(rows_a) & set(rows_b))
    if not keys:
        return ["no overlapping player-games — nothing to compare"], 1
    for k in keys:
        ra, rb = rows_a[k], rows_b[k]
        try:
            aa, ab = float(ra["ellipse_area_ft2"]), float(rb["ellipse_area_ft2"])
            rel = abs(aa - ab) / max(ab, 1e-9)
        except ValueError:
            rel = float("nan")
        d_share = ""
        share_bad = False
        if ra["width_share"] and rb["width_share"]:
            ds = abs(float(ra["width_share"]) - float(rb["width_share"]))
            d_share = f"{ds:.3f}"
            share_bad = ds > SHARE_ABS_TOL
        bad = (rel == rel and rel > AREA_REL_TOL) or share_bad
        flagged += bad
        lines.append(f"  g{k[0]} {ra['player']:>22}  area {aa:7.1f} vs "
                     f"{ab:7.1f} ({rel:5.1%})  dshare {d_share or '  -  '}"
                     f"{'  <-- FLAG' if bad else ''}")
    only = (set(rows_a) ^ set(rows_b))
    if only:
        lines.append(f"  ({len(only)} player-games present in only one "
                     f"backend — identity/coverage differs there)")
        flagged += 1
    return lines, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-a", help="rtmpose npz dir (production spine)")
    ap.add_argument("--pose-b", help="vitpose npz dir (gold standard)")
    ap.add_argument("--court")
    ap.add_argument("--windows")
    ap.add_argument("--lineup")
    ap.add_argument("--cam", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        ra = {("1", "u1"): {"vod": "ab_a", "player": "X", "game": "1",
                            "player_uuid": "u1", "ellipse_area_ft2": "100.0",
                            "width_share": "0.410"}}
        rb = {("1", "u1"): {"vod": "ab_b", "player": "X", "game": "1",
                            "player_uuid": "u1", "ellipse_area_ft2": "104.0",
                            "width_share": "0.415"}}
        lines, n = compare(ra, rb)
        assert n == 0, lines
        rb[("1", "u1")]["ellipse_area_ft2"] = "150.0"
        lines, n = compare(ra, rb)
        assert n == 1, lines
        rb[("1", "u1")]["ellipse_area_ft2"] = "104.0"
        rb[("1", "u1")]["width_share"] = "0.460"
        lines, n = compare(ra, rb)
        assert n == 1, lines
        rb[("2", "u2")] = dict(rb[("1", "u1")])
        lines, n = compare(ra, rb)
        assert n >= 2, lines
        print("SELFTEST OK")
        return
    for req in ("pose_a", "pose_b", "court", "windows", "lineup"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_', '-')} required")
    with tempfile.TemporaryDirectory() as tmp:
        rows_a = run_one(a.pose_a, a, "a", tmp)
        rows_b = run_one(a.pose_b, a, "b", tmp)
    lines, flagged = compare(rows_a, rows_b)
    print("backend A/B (a = production spine, b = gold standard):")
    print("\n".join(lines))
    print("VERDICT:", "PASS — scale out on rtmpose" if not flagged else
          f"FLAG x{flagged} — the ViTPose number wins; diagnose before "
          f"scale-out (coverage_spec.md backend guard)")


if __name__ == "__main__":
    main()
