"""Where the wind actually blows: subgroup scoring of the cross-fit-by-event
wind challenger (arm 2 of wind_holdout.py).

If wind carries forecast information the gain must concentrate in the windy
outdoor games.  Scoring the whole archive dilutes a real effect by ~22:1
(966 outdoor games at >=14 mph out of 21,335 outdoor / 30,982 total), so
this re-scores the SAME cross-fit predictions inside wind bins.  Also runs
the arm-2 power check on a 2-fold (odd/even event) split.

  python model/weather_review/wind_subgroups.py [--power]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windlib import Racer, build, nelder_mead  # noqa: E402
from wind_holdout import (SEED, cluster_z, features, fit, metrics,  # noqa: E402
                          paired_boot, predict)

OUT = Path(__file__).resolve().parent
BINS = [(0, 8), (8, 14), (14, 20), (20, 99)]


def crossfit(names, n_folds=10, label_source="override"):
    games, eta = build(label_source=label_source, rating="traj")
    F = features(games, True)
    T = np.array([g["T"] for g in games])
    won = np.array([g["won"] for g in games])
    ev = np.array([g["event"] for g in games])
    uniq = np.array(sorted(set(ev)))
    rng = np.random.default_rng(SEED)
    fold_of = {e: i for i, e in zip(rng.permutation(len(uniq)) % n_folds, uniq)}
    folds = np.array([fold_of[e] for e in ev])
    P = {}
    for name in names:
        p = np.empty(len(games))
        for k in range(n_folds):
            te = folds == k
            tr = ~te
            x = fit(name, eta[tr], {kk: vv[tr] for kk, vv in F.items()},
                    T[tr], won[tr], Racer_)
            p[te] = predict(name, x, eta[te],
                            {kk: vv[te] for kk, vv in F.items()}, T[te], Racer_)
        P[name] = p
    return games, eta, F, T, won, ev, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--sims", type=int, default=80)
    args = ap.parse_args()
    global Racer_
    Racer_ = Racer()
    names = ["scale_only", "a_skill_x_wind", "c_gust_x_skill", "d_threshold",
             "e_prob_temper"]
    games, eta, F, T, won, ev, P = crossfit(names)
    wind = F["raw_wind"]
    outdoor = F["outdoor"] == 1
    res = {"bins": {}}
    for lo, hi in BINS:
        m = outdoor & (wind >= lo) & (wind < hi)
        if m.sum() < 30:
            continue
        row = {"n": int(m.sum()), "n_events": int(len(set(ev[m]))),
               "base": metrics(P["scale_only"][m], won[m])}
        for nm in names[1:]:
            row[nm] = dict(**metrics(P[nm][m], won[m]),
                           vs_scale=paired_boot(P[nm][m], P["scale_only"][m],
                                                won[m], clusters=ev[m]))
        res["bins"][f"{lo}-{hi}"] = row
        print(f"wind {lo}-{hi} mph  n={row['n']:5d} ev={row['n_events']:3d} "
              f"base_brier={row['base']['brier']:.5f}")
        for nm in names[1:]:
            v = row[nm]["vs_scale"]
            print(f"    {nm:16s} dB={v['d_brier']:+.5f} "
                  f"CI[{v['brier_ci'][0]:+.5f},{v['brier_ci'][1]:+.5f}] "
                  f"P(better)={v['p_better']:.3f}")

    # outdoor-only overall (the arena where the term can act at all)
    m = outdoor
    res["outdoor_all"] = {"n": int(m.sum()),
                          "base": metrics(P["scale_only"][m], won[m])}
    for nm in names[1:]:
        v = paired_boot(P[nm][m], P["scale_only"][m], won[m], clusters=ev[m])
        res["outdoor_all"][nm] = dict(**metrics(P[nm][m], won[m]), vs_scale=v)
        print(f"OUTDOOR ALL {nm:16s} dB={v['d_brier']:+.6f} "
              f"CI[{v['brier_ci'][0]:+.6f},{v['brier_ci'][1]:+.6f}] "
              f"P(better)={v['p_better']:.3f}")

    if args.power:
        res["power_arm2"] = arm2_power(games, eta, F, T, ev, args.sims)
    (OUT / "wind_subgroups_summary.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT / 'wind_subgroups_summary.json'}")


def arm2_power(games, eta, F, T, ev, n_sims):
    """2-fold (odd/even event) cross-fit power on the WHOLE archive: inject a
    true skill x wind effect d, regenerate outcomes, and see how often the
    wind challenger clears the project's bar against the no-wind control.
    2-fold is less efficient than the 10-fold used for the point estimate, so
    the MDE reported here is conservative."""
    uniq = np.array(sorted(set(ev)))
    half = {e: i % 2 for i, e in enumerate(uniq)}
    side = np.array([half[e] for e in ev])
    rows = []
    for d in (0.0, -0.05, -0.10, -0.15, -0.25):
        p_true = Racer_.win(eta * (1.0 + d * F["W"]), T)
        rng = np.random.default_rng(SEED + int(abs(d) * 1000))
        hits = 0
        bs = []
        for s in range(n_sims):
            y = rng.random(len(p_true)) < p_true
            pc = np.empty(len(y))
            pb = np.empty(len(y))
            for k in (0, 1):
                te, tr = side == k, side != k
                xa = fit("a_skill_x_wind", eta[tr],
                         {kk: vv[tr] for kk, vv in F.items()}, T[tr], y[tr], Racer_)
                xs = fit("scale_only", eta[tr],
                         {kk: vv[tr] for kk, vv in F.items()}, T[tr], y[tr], Racer_)
                bs.append(float(xa[1]))
                sub = {kk: vv[te] for kk, vv in F.items()}
                pc[te] = predict("a_skill_x_wind", xa, eta[te], sub, T[te], Racer_)
                pb[te] = predict("scale_only", xs, eta[te], sub, T[te], Racer_)
            hits += (cluster_z(pc, pb, y, ev) <= -1.645)
        rows.append(dict(d=d, detect_rate=hits / n_sims,
                         b_hat_mean=float(np.mean(bs)),
                         b_hat_sd=float(np.std(bs)), n_sims=n_sims))
        print(f"  arm2 power d={d:+.2f} detect={hits / n_sims:.3f} "
              f"b_hat={np.mean(bs):+.3f} +- {np.std(bs):.3f}")
    return rows


if __name__ == "__main__":
    main()
