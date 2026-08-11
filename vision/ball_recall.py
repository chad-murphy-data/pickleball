"""Is the ball NOT DETECTED, or detected and MIS-ASSOCIATED?

The end-to-end pipeline recovers ~1 shot in 12.  That number alone cannot
say which stage is at fault, and the two answers point at completely
different fixes: better linking is an afternoon, a learned detector is a
GPU weekend and a labelling job.  This script separates them on data
already on disk.

THE TRICK: A CONFIDENT BALL SEGMENT, FOR FREE
    A track that crosses the net is the ball.  Nothing else in frame does —
    players hold their side, colour noise sits still.  So net-crossing
    tracks are a label we get without annotating anything, and we can ask
    what the detector was doing in the frames AROUND them.

THREE MEASUREMENTS
    1. IN-TRACK RECALL.  Inside a confident track's own span, what fraction
       of frames actually carry a detection?  Linking tolerates gaps of up
       to `max_gap`, so this is not forced to 100% — but it IS
       survivorship-biased upward, because a run of misses longer than the
       gap tolerance ends the track and removes those frames from the
       measurement.  Read it as a ceiling.

    2. EXTRAPOLATION PROBE.  Fit the tail of a confident track, predict
       where the ball goes in the next few frames, and ask whether ANY
       candidate sits near the prediction.  These are exactly the frames
       where tracking failed, so this is the unbiased view.
       Scored against a DISPLACED null — the same prediction pushed a fixed
       distance in a random direction — because "a candidate was near my
       prediction" means nothing until you know how often a candidate is
       near an arbitrary point in the same neighbourhood.  (A validation
       that cannot fail is what produced the last set of wrong conclusions
       in this project; not repeating it.)

    3. BRIDGEABLE GAPS.  When two confident tracks follow each other with a
       short gap, does the first one's extrapolation land on the second
       one's start?  If yes, the ball was visible on both sides and the
       LINKER simply failed to join them — association, not detection.

READING THE RESULT
    extrapolation ~ in-track          -> association problem, fix linking
    extrapolation ~ null              -> detection problem, needs a learned
                                         detector; better linking is wasted
                                         effort

    python vision/ball_recall.py --ball pfx_ball.csv --rallies windows.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shots as SH                                           # noqa: E402
import court as C                                            # noqa: E402

NET_Y = C.NET_Y


def fit_extrapolate(f, x, y, targets, deg=2):
    """Quadratic in frame number, per image axis.

    A ballistic flight is not exactly a parabola in IMAGE space under
    perspective, but over the handful of frames probed here the difference
    is far below the detection noise.
    """
    deg = min(deg, len(f) - 1)
    if deg < 1:
        return None
    f0 = f[-1]
    cx = np.polyfit(f - f0, x, deg)
    cy = np.polyfit(f - f0, y, deg)
    t = np.asarray(targets, float) - f0
    return np.polyval(cx, t), np.polyval(cy, t)


def nearest_candidate(byframe, frame, px, py):
    a = byframe.get(int(frame))
    if a is None or not len(a):
        return np.inf
    return float(np.min(np.hypot(a[:, 0] - px, a[:, 1] - py)))


def run(ball_csv, rallies_csv, horizon=5, tail=8, min_len=15,
        null_offset=120.0, radii=(20.0, 35.0), seed=0, max_bridge=15):
    b = SH.load_csv(ball_csv, ["frame", "t_s", "x_img", "y_img", "area",
                               "strength", "x_ft", "y_ft"], play_region=True)
    rallies = [{"rally": int(r["rally"]), "t0": float(r["t0"]),
                "t1": float(r["t1"])}
               for r in csv.DictReader(open(rallies_csv))]
    rng = np.random.default_rng(seed)

    # every candidate, indexed by frame, for the "is anything there?" query
    byframe = {}
    order = np.argsort(b["frame"])
    fr_all = b["frame"][order]
    xy_all = np.stack([b["x_img"][order], b["y_img"][order]], 1)
    bounds = np.searchsorted(fr_all, np.unique(fr_all), side="left")
    uniq = np.unique(fr_all)
    for i, f in enumerate(uniq):
        lo = bounds[i]
        hi = bounds[i + 1] if i + 1 < len(bounds) else len(fr_all)
        byframe[int(f)] = xy_all[lo:hi]

    in_hits = in_span = 0
    n_conf = 0
    conf_frames = 0
    rally_frames = 0
    probe = {h: {"d": [], "null": [], "hold": []}
             for h in range(1, horizon + 1)}
    bridge_ok = bridge_tot = 0
    bridge_null = 0
    gaps = []

    for r in rallies:
        m = (b["t_s"] >= r["t0"]) & (b["t_s"] < r["t1"])
        sel = np.nonzero(m)[0]
        if len(sel) < 8:
            continue
        tracks, (F, T, X, Y) = SH.link_tracks(
            b["frame"][sel], b["t_s"][sel], b["x_img"][sel], b["y_img"][sel])
        ordr = np.argsort(b["frame"][sel])
        CY = b["y_ft"][sel][ordr]
        rally_frames += int(F.max() - F.min() + 1) if len(F) else 0

        conf = []
        for tr in tracks:
            if len(tr) < min_len:
                continue
            if np.sum(np.diff(np.sign(CY[tr] - NET_Y)) != 0) < 1:
                continue
            conf.append(tr)
        conf.sort(key=lambda tr: F[tr[0]])
        n_conf += len(conf)

        for tr in conf:
            span = int(F[tr[-1]] - F[tr[0]] + 1)
            in_hits += len(tr)
            in_span += span
            conf_frames += span

            # ---- 2. extrapolate past the end -------------------------
            tail_idx = tr[-tail:]
            ff = F[tail_idx].astype(float)
            if len(np.unique(ff)) < 3:
                continue
            tgt = [F[tr[-1]] + h for h in range(1, horizon + 1)]
            pred = fit_extrapolate(ff, X[tail_idx], Y[tail_idx], tgt)
            if pred is None:
                continue
            for h, (pxp, pyp) in enumerate(zip(*pred), start=1):
                fq = F[tr[-1]] + h
                if fq > F.max():
                    continue
                d = nearest_candidate(byframe, fq, pxp, pyp)
                th = rng.uniform(0, 2 * np.pi)
                dn = nearest_candidate(byframe, fq,
                                       pxp + null_offset * np.cos(th),
                                       pyp + null_offset * np.sin(th))
                # CONFOUND: tracks often end AT A CONTACT, where forward
                # extrapolation is wrong by construction.  But a ball that
                # merely changed direction is still NEAR where the track
                # ended, so holding the last observed position tests
                # non-detection independently of any motion model.
                dh = nearest_candidate(byframe, fq, X[tr[-1]], Y[tr[-1]])
                probe[h]["d"].append(d)
                probe[h]["null"].append(dn)
                probe[h]["hold"].append(dh)

        # ---- 3. can consecutive confident tracks be bridged? ----------
        for a, z in zip(conf, conf[1:]):
            gap = int(F[z[0]] - F[a[-1]])
            if not (1 <= gap <= max_bridge):
                continue
            tail_idx = a[-tail:]
            ff = F[tail_idx].astype(float)
            if len(np.unique(ff)) < 3:
                continue
            pred = fit_extrapolate(ff, X[tail_idx], Y[tail_idx], [F[z[0]]])
            if pred is None:
                continue
            d = float(np.hypot(pred[0][0] - X[z[0]], pred[1][0] - Y[z[0]]))
            th = rng.uniform(0, 2 * np.pi)
            dn = float(np.hypot(pred[0][0] + null_offset * np.cos(th) - X[z[0]],
                                pred[1][0] + null_offset * np.sin(th) - Y[z[0]]))
            bridge_tot += 1
            bridge_ok += d < null_offset * 0.5
            bridge_null += dn < null_offset * 0.5
            gaps.append(gap)

    # ---------------- report ----------------------------------------
    print(f"confident (net-crossing, >={min_len} frame) tracks: {n_conf} "
          f"over {len(rallies)} rallies = {n_conf/len(rallies):.1f}/rally")
    print(f"they cover {100*conf_frames/max(rally_frames,1):.1f}% of rally frames\n")

    print("1. IN-TRACK RECALL (ceiling — survivorship-biased upward)")
    print(f"   {in_hits}/{in_span} frames inside a confident track carry a "
          f"detection = {100*in_hits/max(in_span,1):.1f}%\n")

    print("2. EXTRAPOLATION PROBE — frames just past a confident track")
    print("   horizon  n     median: extrap / hold-last / null      "
          + "  ".join(f"<{int(R)}px extrap (hold, null)" for R in radii))
    med = lambda a: (np.median(a[np.isfinite(a)])                  # noqa: E731
                     if np.isfinite(a).any() else np.inf)
    for h in range(1, horizon + 1):
        d = np.array(probe[h]["d"])
        nu = np.array(probe[h]["null"])
        ho = np.array(probe[h]["hold"])
        if not len(d):
            continue
        cols = [f"{100*np.mean(d < R):5.1f}% ({100*np.mean(ho < R):4.1f}%,"
                f" {100*np.mean(nu < R):4.1f}%)" for R in radii]
        print(f"   +{h:<7d} {len(d):<5d} {med(d):7.1f} / {med(ho):7.1f} / "
              f"{med(nu):7.1f} px    " + "   ".join(cols))

    print("\n3. BRIDGEABLE GAPS — does one confident track predict the next?")
    if bridge_tot:
        print(f"   {bridge_ok}/{bridge_tot} = {100*bridge_ok/bridge_tot:.0f}% "
              f"of consecutive confident tracks join up "
              f"(null {100*bridge_null/bridge_tot:.0f}%)")
        print(f"   gap length: median {np.median(gaps):.0f} frames, "
              f"p90 {np.percentile(gaps,90):.0f}")
    else:
        print("   no consecutive confident-track pairs found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball", required=True)
    ap.add_argument("--rallies", required=True)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-len", type=int, default=15)
    a = ap.parse_args()
    run(a.ball, a.rallies, horizon=a.horizon, min_len=a.min_len)


if __name__ == "__main__":
    main()
