"""TASK C2 — does knowing the wind make the FORECAST better?

The project gates every model change on out-of-sample performance
(model/v2_holdout.py, model/spec_shootout.py: v2 = 77.4% / 0.165 Brier on
the frozen post-2026-06-01 holdout; a challenger is adopted only if the
paired bootstrap on the Brier difference puts P(better) >= 0.95).  The
weather thread never faced that bar.  This does.

Protocol (copied from spec_shootout.py, family A):
  * base prediction = the FROZEN v2 _train fit (posterior-mean plugin
    eta = sum + gamma|gap| + chem) priced through the exact win-by-2 race DP;
  * every free parameter is fit on data the scored games never touch;
  * differences vs the base get a paired bootstrap on Brier.

Two arms:
  ARM 1 (protocol-exact): wind params fit on 2026-01-01..2026-05-31 train
    games, scored on the frozen June+ holdout.  This is literally the
    spec_shootout family-A pipeline with a wind term bolted on.
  ARM 2 (power extension): 10-fold cross-fit BY EVENT over the whole
    2024-2026 archive using month-appropriate v2 trajectory values, so the
    wind coefficient is never estimated on the games it scores.

The base v2 ratings are in-sample in BOTH arms (they were fit on the train
period and, in arm 2, on the scored games themselves).  What is validated
here is therefore the INCREMENTAL value of wind: the leak is identical in
both arms, so the difference is clean even though neither arm's absolute
level is an honest accuracy estimate.

Usage:  python model/weather_review/wind_holdout.py [--power] [--quick]
Writes model/weather_review/wind_holdout_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windlib import (DATA, RECENT, SPLIT, Racer, build, nelder_mead)  # noqa: E402

OUT = Path(__file__).resolve().parent
SEED = 20260731
N_BOOT = 4000
WIND_CENTER = 8.0     # mph; centering only moves the scale, not the wind coef
GUST_CENTER = 14.0
TEMP_CENTER = 75.0
THRESH = 14.0         # the published "windy" bin edge


# ------------------------------------------------------- feature matrix --

def features(games, outdoor_only=True):
    """Wind features, zeroed where wind cannot physically act (indoor) unless
    outdoor_only=False, in which case the covariate is label-free."""
    w = np.array([g["wind"] for g in games], float)
    gu = np.array([g["gust"] for g in games], float)
    tp = np.array([g["temp"] if g["temp"] is not None else TEMP_CENTER
                   for g in games], float)
    out = np.array([g["setting"] == "outdoor" for g in games], float)
    m = out if outdoor_only else np.ones_like(out)
    return dict(
        W=m * (w - WIND_CENTER) / 10.0,
        G=m * (gu - GUST_CENTER) / 10.0,
        TH=m * (w >= THRESH),
        TP=m * (tp - TEMP_CENTER) / 10.0,
        raw_wind=w, outdoor=out)


# --------------------------------------------------------- challengers --
# Each returns (eta', prob-post-transform or None).  p = race(eta'); a
# challenger may instead warp the calibrated probability (family e).

CHALLENGERS = {
    # name: (n_params, x0, description)
    "base_v2":      (0, [], "frozen v2 plugin through the race DP (reference)"),
    "scale_only":   (1, [1.0], "free overall scale on the v2 eta — the "
                               "no-wind calibration control"),
    "a_skill_x_wind": (2, [1.0, 0.0],
                       "(a) skill x wind: eta -> s*eta*(1 + b*(wind-8)/10), "
                       "outdoor only"),
    "b_wind_main":  (2, [1.0, 0.0],
                     "(b) wind main effect on eta magnitude: "
                     "eta -> s*eta + b*sign(eta)*(wind-8)/10"),
    "c_gust_x_skill": (2, [1.0, 0.0],
                       "(c) gust version of (a): s*eta*(1 + b*(gust-14)/10)"),
    "d_threshold":  (2, [1.0, 0.0],
                     "(d) threshold: s*eta*(1 + b*1[wind >= 14 mph])"),
    "e_prob_temper": (2, [1.0, 0.0],
                      "(e) probability temperature: logit(p) -> "
                      "logit(p)*(1 + b*(wind-8)/10), flattens toward 0.5"),
    "f_heat_x_skill": (2, [1.0, 0.0],
                       "(f bonus) heat version of (a): s*eta*(1 + b*(T-75)/10)"),
}


def predict(name, x, eta, F, T, racer):
    if name == "base_v2":
        return racer.win(eta, T)
    if name == "scale_only":
        return racer.win(x[0] * eta, T)
    if name == "a_skill_x_wind":
        return racer.win(x[0] * eta * (1.0 + x[1] * F["W"]), T)
    if name == "b_wind_main":
        return racer.win(x[0] * eta + x[1] * np.sign(eta) * F["W"], T)
    if name == "c_gust_x_skill":
        return racer.win(x[0] * eta * (1.0 + x[1] * F["G"]), T)
    if name == "d_threshold":
        return racer.win(x[0] * eta * (1.0 + x[1] * F["TH"]), T)
    if name == "e_prob_temper":
        p = np.clip(racer.win(x[0] * eta, T), 1e-9, 1 - 1e-9)
        z = np.log(p / (1 - p)) * (1.0 + x[1] * F["W"])
        return 1.0 / (1.0 + np.exp(-z))
    if name == "f_heat_x_skill":
        return racer.win(x[0] * eta * (1.0 + x[1] * F["TP"]), T)
    raise KeyError(name)


def fit(name, eta, F, T, won, racer, x0=None):
    n_par, d0, _ = CHALLENGERS[name]
    if n_par == 0:
        return np.array([])
    x0 = np.array(x0 if x0 is not None else d0, float)

    def nll(x):
        p = np.clip(predict(name, x, eta, F, T, racer), 1e-9, 1 - 1e-9)
        return -np.mean(np.where(won, np.log(p), np.log(1 - p)))

    return nelder_mead(nll, x0)


# ------------------------------------------------------------- metrics --

def metrics(p, won):
    p = np.asarray(p, float)
    won = np.asarray(won, bool)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(n=int(len(p)),
                accuracy=float(np.mean((p > 0.5) == won)),
                brier=float(np.mean((p - won) ** 2)),
                log_loss=float(np.mean(-np.where(won, np.log(pc), np.log(1 - pc)))))


def paired_boot(p, ref, won, clusters=None, seed=SEED, n=N_BOOT):
    """Bootstrap the Brier and log-loss differences (challenger - base);
    negative favours the challenger.  clusters=None -> resample games (the
    spec_shootout protocol); clusters given -> resample EVENTS (the brief's
    requirement, since games in an event share weather)."""
    p, ref, won = np.asarray(p, float), np.asarray(ref, float), np.asarray(won, bool)
    db = (p - won) ** 2 - (ref - won) ** 2
    pc, rc = np.clip(p, 1e-9, 1 - 1e-9), np.clip(ref, 1e-9, 1 - 1e-9)
    dl = (-np.where(won, np.log(pc), np.log(1 - pc))
          + np.where(won, np.log(rc), np.log(1 - rc)))
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(db), size=(n, len(db)))
        mb, ml = db[idx].mean(axis=1), dl[idx].mean(axis=1)
    else:
        uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv, minlength=len(uniq))
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        sb = np.add.reduceat(db[order], starts) if len(uniq) else np.array([])
        sl = np.add.reduceat(dl[order], starts)
        pick = rng.integers(0, len(uniq), size=(n, len(uniq)))
        tot = counts[pick].sum(axis=1)
        mb = sb[pick].sum(axis=1) / tot
        ml = sl[pick].sum(axis=1) / tot
    return dict(
        d_brier=float(db.mean()),
        brier_ci=[float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))],
        d_log_loss=float(dl.mean()),
        ll_ci=[float(np.percentile(ml, 2.5)), float(np.percentile(ml, 97.5))],
        p_better=float(np.mean(mb < 0)))


def cluster_z(p, ref, won, clusters):
    """One-sided z for mean Brier difference with a cluster-robust se —
    the fast stand-in for the bootstrap used inside the power loop."""
    d = (p - won) ** 2 - (ref - won) ** 2
    uniq, inv = np.unique(clusters, return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(uniq))
    c = np.bincount(inv, minlength=len(uniq)).astype(float)
    n = len(d)
    m = d.mean()
    var = np.sum((s - c * m) ** 2) / n ** 2
    return m / np.sqrt(var) if var > 0 else 0.0


# ------------------------------------------------------------- the arms --

def arm1(racer, results, label_source="override", outdoor_only=True, tag="",
         end=None):
    games, eta = build(label_source=label_source, rating="now")
    fitset = [i for i, g in enumerate(games) if RECENT <= g["date"] < SPLIT]
    hold = [i for i, g in enumerate(games)
            if g["date"] >= SPLIT and (end is None or g["date"] < end)]
    F = features(games, outdoor_only)
    T = np.array([g["T"] for g in games])
    won = np.array([g["won"] for g in games])
    ev = np.array([g["event"] for g in games])

    def sl(idx, d):
        return {k: v[idx] for k, v in d.items()}

    Ff, Fh = sl(fitset, F), sl(hold, F)
    ef, eh = eta[fitset], eta[hold]
    Tf, Th = T[fitset], T[hold]
    wf, wh = won[fitset], won[hold]
    evh = ev[hold]

    base_p = predict("base_v2", None, eh, Fh, Th, racer)
    x_scale = fit("scale_only", ef, Ff, Tf, wf, racer)
    scale_p = predict("scale_only", x_scale, eh, Fh, Th, racer)
    out = {}
    for name in CHALLENGERS:
        x = fit(name, ef, Ff, Tf, wf, racer)
        p = predict(name, x, eh, Fh, Th, racer)
        row = dict(**metrics(p, wh), params=[round(float(v), 4) for v in x],
                   desc=CHALLENGERS[name][2])
        if name != "base_v2":
            row["vs_base_event_cluster"] = paired_boot(p, base_p, wh, clusters=evh)
            row["vs_base_by_game"] = paired_boot(p, base_p, wh)
        if name not in ("base_v2", "scale_only"):
            # PRIMARY: nested against the same model without the wind term
            row["vs_scale_event_cluster"] = paired_boot(p, scale_p, wh, clusters=evh)
            row["vs_scale_by_game"] = paired_boot(p, scale_p, wh)
            row["z_vs_scale_event_cluster"] = float(cluster_z(p, scale_p, wh, evh))
        out[name] = row
        vs = row.get("vs_scale_event_cluster") or row.get("vs_base_event_cluster")
        print(f"  {name:18s} n={row['n']:4d} acc={row['accuracy']:.4f} "
              f"brier={row['brier']:.5f} ll={row['log_loss']:.4f} "
              f"par={row['params']}"
              + ("" if vs is None else
                 f"  dB={vs['d_brier']:+.6f} "
                 f"CI[{vs['brier_ci'][0]:+.5f},{vs['brier_ci'][1]:+.5f}] "
                 f"P(better)={vs['p_better']:.3f}"))
    results[f"arm1{tag}"] = dict(
        n_fit=len(fitset), n_hold=len(hold), n_events_hold=int(len(set(evh))),
        label_source=label_source, outdoor_only=outdoor_only,
        fit_window=[RECENT, SPLIT], split=SPLIT, hold_end=end,
        reference_note="vs_scale_* is the PRIMARY wind test (nested: same "
                       "model, wind term dropped); vs_base_* also shown",
        models=out)
    return games, eta, F, T, won, ev, fitset, hold


def arm2(racer, results, label_source="override", n_folds=10, n_boot_coef=600):
    games, eta = build(label_source=label_source, rating="traj")
    F = features(games, True)
    T = np.array([g["T"] for g in games])
    won = np.array([g["won"] for g in games])
    ev = np.array([g["event"] for g in games])
    uniq = np.array(sorted(set(ev)))
    rng = np.random.default_rng(SEED)
    fold_of = {e: i for i, e in zip(rng.permutation(len(uniq)) % n_folds, uniq)}
    folds = np.array([fold_of[e] for e in ev])

    out = {}
    for name in CHALLENGERS:
        p = np.empty(len(games))
        pars = []
        for k in range(n_folds):
            te = folds == k
            tr = ~te
            x = fit(name, eta[tr], {kk: vv[tr] for kk, vv in F.items()},
                    T[tr], won[tr], racer)
            pars.append([round(float(v), 4) for v in x])
            p[te] = predict(name, x, eta[te],
                            {kk: vv[te] for kk, vv in F.items()}, T[te], racer)
        row = dict(**metrics(p, won), fold_params=pars,
                   desc=CHALLENGERS[name][2])
        if name == "base_v2":
            base_p = p
        elif name == "scale_only":
            scale_p = p
            row["vs_base_event_cluster"] = paired_boot(p, base_p, won, clusters=ev)
        else:
            row["vs_base_event_cluster"] = paired_boot(p, base_p, won, clusters=ev)
            row["vs_scale_event_cluster"] = paired_boot(p, scale_p, won, clusters=ev)
            row["z_vs_scale_event_cluster"] = float(cluster_z(p, scale_p, won, ev))
        out[name] = row
        vs = row.get("vs_scale_event_cluster") or row.get("vs_base_event_cluster")
        print(f"  {name:18s} n={row['n']:5d} acc={row['accuracy']:.4f} "
              f"brier={row['brier']:.5f} ll={row['log_loss']:.4f}"
              + ("" if vs is None else
                 f"  dB={vs['d_brier']:+.6f} "
                 f"CI[{vs['brier_ci'][0]:+.5f},{vs['brier_ci'][1]:+.5f}] "
                 f"P(better)={vs['p_better']:.3f}"))

    # full-sample coefficient + event-cluster bootstrap CI (the BOUND)
    coefs = {}
    for name in ("a_skill_x_wind", "d_threshold", "c_gust_x_skill",
                 "f_heat_x_skill"):
        xf = fit(name, eta, F, T, won, racer)
        ev_idx = {e: np.flatnonzero(ev == e) for e in uniq}
        bs = []
        r2 = np.random.default_rng(SEED)
        for i in range(n_boot_coef):
            pick = uniq[r2.integers(0, len(uniq), len(uniq))]
            idx = np.concatenate([ev_idx[e] for e in pick])
            xb = fit(name, eta[idx], {k: v[idx] for k, v in F.items()},
                     T[idx], won[idx], racer, x0=xf)
            bs.append(float(xb[1]))
        bs = np.array(bs)
        coefs[name] = dict(b=float(xf[1]), s=float(xf[0]),
                           ci=[float(np.percentile(bs, 2.5)),
                               float(np.percentile(bs, 97.5))],
                           n_boot=n_boot_coef)
        print(f"  coef {name:18s} b={xf[1]:+.4f} "
              f"CI [{coefs[name]['ci'][0]:+.4f}, {coefs[name]['ci'][1]:+.4f}]")

    results["arm2"] = dict(n=len(games), n_events=int(len(uniq)),
                           n_folds=n_folds, label_source=label_source,
                           models=out, coefficients=coefs)
    return games, eta, F, T, won, ev


# ---------------------------------------------------------------- power --

def power(racer, results, n_sims=200, ds=(0.0, -0.05, -0.10, -0.15, -0.20,
                                          -0.30, -0.40, -0.60)):
    """Inject a TRUE skill x wind effect of size d into both the fitting
    window and the holdout, regenerate outcomes from the model, rerun the
    identical arm-1 pipeline, and record how often the challenger clears the
    project's bar (event-clustered one-sided z <= -1.645, i.e. P(better)>=.95).
    """
    games, eta = build(label_source="override", rating="now")
    F = features(games, True)
    T = np.array([g["T"] for g in games])
    ev = np.array([g["event"] for g in games])
    fitset = np.array([i for i, g in enumerate(games)
                       if RECENT <= g["date"] < SPLIT])
    hold = np.array([i for i, g in enumerate(games) if g["date"] >= SPLIT])
    Ff = {k: v[fitset] for k, v in F.items()}
    Fh = {k: v[hold] for k, v in F.items()}
    ef, eh, Tf, Th, evh = eta[fitset], eta[hold], T[fitset], T[hold], ev[hold]

    rows = []
    for d in ds:
        p_true_f = racer.win(ef * (1.0 + d * Ff["W"]), Tf)
        p_true_h = racer.win(eh * (1.0 + d * Fh["W"]), Th)
        hits = 0
        bs = []
        rng = np.random.default_rng(SEED + int(abs(d) * 1000))
        for s in range(n_sims):
            wf = rng.random(len(p_true_f)) < p_true_f
            wh = rng.random(len(p_true_h)) < p_true_h
            x = fit("a_skill_x_wind", ef, Ff, Tf, wf, racer)
            bs.append(float(x[1]))
            pc = predict("a_skill_x_wind", x, eh, Fh, Th, racer)
            xs = fit("scale_only", ef, Ff, Tf, wf, racer)
            pb = predict("scale_only", xs, eh, Fh, Th, racer)
            z = cluster_z(pc, pb, wh, evh)
            hits += (z <= -1.645)
        rows.append(dict(d=d, detect_rate=hits / n_sims,
                         b_hat_mean=float(np.mean(bs)),
                         b_hat_sd=float(np.std(bs)), n_sims=n_sims))
        print(f"  d={d:+.2f}  detect={hits / n_sims:.3f}  "
              f"b_hat={np.mean(bs):+.3f} +- {np.std(bs):.3f}")
    results["power_arm1"] = dict(
        challenger="a_skill_x_wind", rule="event-clustered z <= -1.645",
        grid=rows,
        note="d is the fractional change in the per-point skill edge per "
             "+10 mph of match-hour wind, outdoor games only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sims", type=int, default=200)
    args = ap.parse_args()
    racer = Racer()
    results = {}
    print("ARM 1 — protocol-exact frozen holdout (audited venue labels)")
    arm1(racer, results)
    print("ARM 1-canonical — holdout truncated to the spec_shootout snapshot")
    arm1(racer, results, tag="_canonical", end="2026-07-18")
    print("ARM 1b — same, HEURISTIC venue labels (what published tests used)")
    arm1(racer, results, label_source="heuristic", tag="_heuristic")
    print("ARM 1c — same, label-free wind (wind applied to every game)")
    arm1(racer, results, outdoor_only=False, tag="_labelfree")
    if not args.quick:
        print("ARM 2 — 10-fold cross-fit by EVENT over the whole archive")
        arm2(racer, results)
    if args.power:
        print("POWER — synthetic-effect injection on the frozen holdout")
        power(racer, results, n_sims=args.sims)
    (OUT / "wind_holdout_summary.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'wind_holdout_summary.json'}")


if __name__ == "__main__":
    main()
