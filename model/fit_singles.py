"""Singles rating suite — Bayesian, doubles-informed, DreamBreaker-aware.

    python model/fit_singles.py    # -> data/singles_players.csv
                                   #    + model/singles_model.json

Three stages (all pure python, no deps, ~15 s):

1. PURE fit — unchanged from the first pass: each PPA singles game is a
   per-point Binomial race, points ~ Binomial(total, sigmoid(v_i - v_j)),
   v ~ N(0, SD_PRIOR), exponential recency (half-life 12 months). These
   values ship in the `singles_value_pure` column and stay doubles-blind —
   the singles~doubles regression and the finding-12 surplus diagnostics
   must keep using them (a doubles-informed value in that seat is circular).

2. HYPERPRIORS — regress pure values on v2 doubles values over the
   penalty-free population (>= 60 singles games), PER GENDER: men and women
   never meet in singles, so the two singles scales (and the two models'
   cross-gender conventions) are prior-linked only — the intercepts differ
   by ~0.2 logit and pooling would smear that into tau. 60+ only because
   the selection penalty (below) demonstrably persists into the 10-59
   range, which would tilt a regression fit on it.

3. SUITE fit — refit everything with a per-player prior
       v_p ~ N(a_g + b_g*d_p - c*fade(games_p),
               tau_g^2 + (b_g*sd_d(p))^2 + (C_SD*fade)^2)
   where c is the DreamBreaker-measured selection penalty for players with
   no real singles record (model/db_impute.md: they underperform their
   doubles-implied value by ~0.35 logit; refreshed 2026-08-16 on 114
   validated DBs: 0.348, CI [0.05, 0.71]) and fade(games) is EMPIRICALLY
   CALIBRATED per evidence bin: f(0)=1 fixed (that is db_impute's measured
   point), f(60+)=0, and the bins between are adjusted until the mean
   posterior-minus-prior residual in each bin is ~0 — i.e. the prior is
   unbiased at every record depth (db_impute's threshold sensitivities,
   0.33/0.35/0.39 at >=1/>=10/>=30, independently show the penalty fading
   far slower than a naive ramp to the ranked line). Bin-unbiased priors
   also kill the component-location drift that a mis-specified fade
   injects (a hard-won lesson: with optimistic thin-tier priors the whole
   connected component inflates, the imputation line is left behind, and
   no gauge trick fixes it without crushing someone — see singles_suite.md).
   Players with no doubles value keep the zero-centered pure prior. PLUS
   DreamBreaker rallies as direct evidence: same-gender attributed rallies
   (data/db_rallies.csv, validated referee-log parse) enter as
   Bernoulli(sigmoid(K_DB * (v_i - v_j))) with the measured attenuation
   K_DB (db_impute rally slope 0.49), same recency weighting. Cross-gender
   DB rallies (~2%) are EXCLUDED from the likelihood (house rule: the M/W
   offset is a prior convention, never data-identified here).

Output rows cover every player with singles evidence AND every v2 player:
tier `fitted` (>= RANKED_GAMES singles games), `blended` (1-9 games or any
DB rallies), `imputed` (prior only). `singles_sd` is the diagonal-Laplace
posterior sd at the MAP — for imputed players it equals the prior sd, so
the uncertainty of a projection is explicit. DB rally counts in the CSV
are record-book accounting (ALL attributed rallies, cross-gender included).

Gate: model/singles_holdout.py (pre-June train / post-June eval) — the
suite must not lose to the pure fit on singles games and must beat it on
DreamBreakers before consumers trust it. Results: model/singles_suite.md.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SD_PRIOR = 0.6          # per-point logit; pure-fit prior (singles spread wide)
HALF_LIFE_DAYS = 365.0
RANKED_GAMES = 10       # the "real singles record" line (db_impute.md)
HYPER_MIN_GAMES = 60    # penalty-free population for the hyperprior line
# Selection penalty for no-record players, measured on DB rallies
# (db_impute.md primary fit; env override for sensitivity runs):
C_NONRANKED = float(os.environ.get("SINGLES_C", "0.35"))
C_SD = 0.17             # c's own uncertainty (CI half-width / 2) — widens
                        # projection sd, honestly, by ~0.03
# fade bins between the fixed endpoints f(0 games)=1 and f(60+)=0;
# starting values only — calibrated in stage 3:
FADE_BINS = ((1, 4), (5, 9), (10, 29), (30, 59))
FADE_INIT = [0.9, 0.7, 0.45, 0.2]
# Attenuation of singles-value gaps in DB rallies (db_impute empirical
# rally slope b1 = 0.488 ± 0.067 on the 2026-08-16 refresh, 114 validated
# DBs; the team-mean-level constant 0.42 lives in consumers):
K_DB = float(os.environ.get("SINGLES_K_DB", "0.49"))
DATE_BEFORE = os.environ.get("SINGLES_DATE_BEFORE")   # holdout: train cutoff
REF_DATE = None          # default: newest game/rally loaded


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def load_games(before=None):
    games = []
    for r in csv.DictReader((DATA / "singles_games.csv").open()):
        if r["is_forfeit"] != "False":
            continue
        if before and r["date"] >= before:
            continue
        games.append((r["p1"], r["p2"], int(r["s1"]), int(r["s2"]),
                      r["date"], r["context"], r["p1_name"], r["p2_name"]))
    return games


def load_doubles(path=None):
    """player -> (d, sd_d, gender, name) from v2_players.csv (or a
    train-cutoff variant, for the holdout gate)."""
    out = {}
    for r in csv.DictReader((path or DATA / "v2_players.csv").open()):
        out[r["player_id"].lower()] = (float(r["value_now_mean"]),
                                       float(r["value_now_sd"]),
                                       r["gender"], r["full_name"])
    return out


def load_db_rallies(before=None):
    """[(p1, p2, y1, date)] from the validated DreamBreaker parse; dates via
    dreambreakers.csv. Missing files -> empty (suite degrades gracefully)."""
    dates = {}
    dbf = DATA / "dreambreakers.csv"
    if dbf.exists():
        for r in csv.DictReader(dbf.open()):
            if len(r.get("match_id") or "") == 36:
                dates[r["match_id"].lower()] = r["date"]
    path = DATA / "db_rallies.csv"
    if not path.exists():
        return []
    out = []
    for r in csv.DictReader(path.open()):
        d = dates.get(r["match_id"].lower())
        if d is None or (before and d >= before):
            continue
        out.append((r["player_team1"].lower(), r["player_team2"].lower(),
                    int(r["team1_won"]), d))
    return out


def fit_map(idx, obs_games, obs_rallies, prior_mean, prior_var,
            iters=300, verbose=True, v0=None):
    """Preconditioned gradient-ascent MAP + diagonal-Laplace sd.

    obs_games:   (i, j, s1, s2, w)  per-point binomial at sigmoid(v_i - v_j)
    obs_rallies: (i, j, y, w)       Bernoulli at sigmoid(K_DB * (v_i - v_j))
    """
    n = len(idx)
    v = list(v0) if v0 is not None else list(prior_mean)
    tpts = [0.0] * n
    for i, j, s1, s2, w in obs_games:
        tpts[i] += w * (s1 + s2)
        tpts[j] += w * (s1 + s2)
    for i, j, _y, w in obs_rallies:
        tpts[i] += w * K_DB * K_DB
        tpts[j] += w * K_DB * K_DB
    precon = [0.25 * t + 1.0 / prior_var[k] for k, t in enumerate(tpts)]
    gnorm = float("inf")
    for it in range(iters):
        grad = [(prior_mean[k] - v[k]) / prior_var[k] for k in range(n)]
        ll = 0.0
        for i, j, s1, s2, w in obs_games:
            p = sigmoid(v[i] - v[j])
            g = w * (s1 - (s1 + s2) * p)
            grad[i] += g
            grad[j] -= g
            ll += w * (s1 * math.log(max(p, 1e-12))
                       + s2 * math.log(max(1 - p, 1e-12)))
        for i, j, y, w in obs_rallies:
            p = sigmoid(K_DB * (v[i] - v[j]))
            g = w * K_DB * (y - p)
            grad[i] += g
            grad[j] -= g
            ll += w * math.log(max(p if y else 1 - p, 1e-12))
        for k in range(n):
            v[k] += grad[k] / precon[k]
        if it % 100 == 99:
            gnorm = math.sqrt(sum(g * g for g in grad))
            if verbose:
                print(f"  iter {it + 1}: ll {ll:,.1f}  |grad| {gnorm:.3f}")
    # diagonal Laplace at the MAP
    h = [1.0 / prior_var[k] for k in range(n)]
    for i, j, s1, s2, w in obs_games:
        p = sigmoid(v[i] - v[j])
        q = w * (s1 + s2) * p * (1 - p)
        h[i] += q
        h[j] += q
    for i, j, _y, w in obs_rallies:
        p = sigmoid(K_DB * (v[i] - v[j]))
        q = w * K_DB * K_DB * p * (1 - p)
        h[i] += q
        h[j] += q
    sd = [1.0 / math.sqrt(hk) for hk in h]
    return v, sd


def regress(pairs):
    """OLS y ~ a + b*x with residual sd (df n-2) and r."""
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    b = sxy / sxx
    a = my - b * mx
    tau = math.sqrt(sum((y - a - b * x) ** 2 for x, y in pairs) / (n - 2))
    return {"a": a, "b": b, "tau": tau, "r": sxy / math.sqrt(sxx * syy),
            "n": n}


def fit_all(before=None, doubles_path=None, verbose=True):
    """Run all three stages; returns everything main() needs (also used by
    model/singles_holdout.py with a train cutoff + train doubles values)."""
    games = load_games(before)
    rallies_all = load_db_rallies(before)
    doubles = load_doubles(doubles_path)

    ref = REF_DATE or max([g[4] for g in games]
                          + [r[3] for r in rallies_all] or ["2026-01-01"])
    ref_ord = date.fromisoformat(ref).toordinal()

    def weight(d):
        return 0.5 ** ((ref_ord - date.fromisoformat(d).toordinal())
                       / HALF_LIFE_DAYS)

    # -- roster: singles players + DB players + every v2 player -------------
    idx, names, gender = {}, {}, {}
    counts = defaultdict(int)

    def add(p, name=None, g=None):
        if p not in idx:
            idx[p] = len(idx)
        if name and p not in names:
            names[p] = name
        if g in ("M", "F") and p not in gender:
            gender[p] = g

    for p1, p2, s1, s2, d, ctx, n1, n2 in games:
        g_ = "F" if ctx == "womens_singles" else "M"
        add(p1, n1, g_)
        add(p2, n2, g_)
        counts[p1] += 1
        counts[p2] += 1
    db_stats = defaultdict(lambda: [0, 0])          # all attributed: [n, wins]
    for p1, p2, y, d in rallies_all:
        for p in (p1, p2):
            if p in doubles:
                add(p, doubles[p][3], doubles[p][2])
            else:
                add(p)
        db_stats[p1][0] += 1
        db_stats[p1][1] += y
        db_stats[p2][0] += 1
        db_stats[p2][1] += 1 - y
    for u, (dv, sdv, g_, nm) in doubles.items():
        add(u, nm, g_)

    n = len(idx)
    obs_games = [(idx[p1], idx[p2], s1, s2, weight(d))
                 for p1, p2, s1, s2, d, *_ in games]
    # likelihood rallies: same-gender, both players known
    obs_rallies = [(idx[p1], idx[p2], y, weight(d))
                   for p1, p2, y, d in rallies_all
                   if gender.get(p1) and gender.get(p1) == gender.get(p2)]

    # -- stage 1: pure fit (doubles-blind) -----------------------------------
    if verbose:
        print(f"stage 1 — pure fit: {len(games)} games, {n} players")
    zero = [0.0] * n
    var0 = [SD_PRIOR ** 2] * n
    v_pure, _sd_pure = fit_map(idx, obs_games, [], zero, var0,
                               verbose=verbose)

    # -- stage 2: per-gender hyperpriors (penalty-free 60+ population) --------
    hyper = {}
    pooled_pairs = []
    for g_ in ("M", "F"):
        pairs = [(doubles[p][0], v_pure[idx[p]])
                 for p in idx
                 if counts[p] >= HYPER_MIN_GAMES and p in doubles
                 and gender.get(p) == g_]
        hyper[g_] = regress(pairs)
        pooled_pairs += pairs
    hyper["pooled"] = regress(pooled_pairs)
    if verbose:
        for g_, h in hyper.items():
            print(f"stage 2 — {g_}: singles ≈ {h['a']:+.3f} + {h['b']:.3f}·d"
                  f"  tau {h['tau']:.3f}  r {h['r']:.3f}  (n={h['n']})")

    # -- stage 3: suite fit with an empirically calibrated fade --------------
    # The fade is CALIBRATED, not assumed: with an optimistic thin-tier
    # fade (e.g. a linear ramp to zero at 10 games) the thin players' data
    # systematically undershoots their priors, the residual tension lifts
    # the whole connected component's location away from the imputation
    # line, and no post-hoc anchor/translation can repair that without
    # stranding some tier (measured: +0.15 drift, and the gauge-translation
    # "fix" crushed non-v2 qualifiers by up to −1.2). Bin-unbiased priors
    # remove the drift at the source. f(0)=1 stays fixed (db_impute's
    # directly measured point — and the 0-games-with-DB-rallies bin sat at
    # +0.03 already under it), f(60+)=0 by the stage-2 construction.
    inv = {i: p for p, i in idx.items()}
    fade_vals = list(FADE_INIT)

    def fade_of(g_ct):
        if g_ct <= 0:
            return 1.0
        if g_ct >= HYPER_MIN_GAMES:
            return 0.0
        for (lo, hi), f in zip(FADE_BINS, fade_vals):
            if lo <= g_ct <= hi:
                return f
        return 0.0

    v_suite, sd_suite = None, None
    n_iter = 0
    for round_ in range(10):
        pmean, pvar = [0.0] * n, [SD_PRIOR ** 2] * n
        for i in range(n):
            p = inv[i]
            if p in doubles:
                dv, sdv, g_, _nm = doubles[p]
                key = gender.get(p) or g_
                h = hyper.get(key) or hyper["pooled"]
                fd = fade_of(counts[p])
                pmean[i] = h["a"] + h["b"] * dv - C_NONRANKED * fd
                pvar[i] = (h["tau"] ** 2 + (h["b"] * sdv) ** 2
                           + (C_SD * fd) ** 2)
        v_suite, sd_suite = fit_map(idx, obs_games, obs_rallies, pmean, pvar,
                                    iters=300 if round_ == 0 else 150,
                                    verbose=verbose and round_ == 0,
                                    v0=v_suite)
        n_iter = round_ + 1
        resids = []
        for bi, (lo, hi) in enumerate(FADE_BINS):
            rs = [v_suite[idx[p]] - pmean[idx[p]] for p in idx
                  if p in doubles and lo <= counts[p] <= hi]
            resids.append(sum(rs) / len(rs) if rs else 0.0)
        if verbose:
            print(f"  fade round {n_iter}: bin residuals "
                  + " ".join(f"{r:+.3f}" for r in resids)
                  + "  fade " + " ".join(f"{f:.2f}" for f in fade_vals))
        if max(abs(r) for r in resids) < 0.02:
            break
        for bi, r in enumerate(resids):
            fade_vals[bi] = min(2.0, max(0.0,
                                         fade_vals[bi] - r / C_NONRANKED))

    if verbose:
        for g_ in ("M", "F"):
            top = [v_suite[idx[p]] - (hyper[g_]["a"]
                                      + hyper[g_]["b"] * doubles[p][0])
                   for p in idx if counts[p] >= HYPER_MIN_GAMES
                   and p in doubles and gender.get(p) == g_]
            drift = [v_suite[idx[p]] - v_pure[idx[p]] for p in idx
                     if counts[p] >= HYPER_MIN_GAMES and gender.get(p) == g_]
            print(f"stage 3 — {g_}: 60+ tier vs line {sum(top)/len(top):+.3f}"
                  f", vs pure {sum(drift)/len(drift):+.3f}")
        print(f"stage 3 — suite fit ({n_iter} calibration rounds): "
              f"+{len(obs_rallies)} same-gender DB rallies "
              f"(of {len(rallies_all)} attributed), K_DB {K_DB}, "
              f"c {C_NONRANKED}, fade "
              + " ".join(f"{lo}-{hi}:{f:.2f}"
                         for (lo, hi), f in zip(FADE_BINS, fade_vals)))

    return {
        "idx": idx, "inv": inv, "names": names, "gender": gender,
        "counts": counts, "db_stats": db_stats, "doubles": doubles,
        "hyper": hyper, "fade": fade_vals, "fade_rounds": n_iter,
        "ref": ref,
        "v_pure": v_pure, "v_suite": v_suite, "sd_suite": sd_suite,
        "n_games": len(games), "n_rallies_used": len(obs_rallies),
        "n_rallies_all": len(rallies_all),
    }


def tier_of(singles_games, db_rallies):
    if singles_games >= RANKED_GAMES:
        return "fitted"
    if singles_games > 0 or db_rallies > 0:
        return "blended"
    return "imputed"


def main():
    F = fit_all(before=DATE_BEFORE)
    idx, inv = F["idx"], F["inv"]
    counts, db_stats = F["counts"], F["db_stats"]
    doubles, names, gender = F["doubles"], F["names"], F["gender"]

    rows = []
    for p, i in idx.items():
        db_n, db_w = db_stats.get(p, (0, 0))
        has_singles = counts[p] > 0
        rows.append({
            "player_id": p,
            "full_name": names.get(p, ""),
            "gender": gender.get(p, ""),
            "singles_games": counts[p],
            "db_rallies": db_n,
            "db_rally_wins": db_w,
            "singles_value": round(F["v_suite"][i], 4),
            "singles_sd": round(F["sd_suite"][i], 4),
            "tier": tier_of(counts[p], db_n),
            "singles_value_pure": (round(F["v_pure"][i], 4)
                                   if has_singles else ""),
        })
    rows.sort(key=lambda r: -r["singles_value"])
    with (DATA / "singles_players.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tiers = defaultdict(int)
    for r in rows:
        tiers[r["tier"]] += 1
    print(f"\nwrote data/singles_players.csv: {len(rows)} players "
          f"({dict(tiers)})")

    for g_ in ("M", "F"):
        pool = [r for r in rows if r["gender"] == g_
                and r["tier"] != "imputed"]
        print(f"\ntop {g_} singles (rated):")
        for r in pool[:8]:
            print(f"  {r['full_name']:26s} {r['singles_value']:+.2f} "
                  f"±{r['singles_sd']:.2f}  ({r['singles_games']} g, "
                  f"{r['db_rallies']} DB rallies, {r['tier']})")

    h = F["hyper"]
    summary = {
        "ref_date": F["ref"],
        "n_games": F["n_games"],
        "n_db_rallies_used": F["n_rallies_used"],
        "n_db_rallies_attributed": F["n_rallies_all"],
        "n_players": len(rows),
        "sd_prior_pure": SD_PRIOR,
        "half_life_days": HALF_LIFE_DAYS,
        "ranked_games": RANKED_GAMES,
        "hyper_min_games": HYPER_MIN_GAMES,
        "c_nonranked": C_NONRANKED,
        "c_sd": C_SD,
        "k_db": K_DB,
        "fade_bins": [list(b) for b in FADE_BINS],
        "fade": [round(f, 3) for f in F["fade"]],
        "fade_rounds": F["fade_rounds"],
        "hyper": {g: {k: round(v, 4) for k, v in hh.items()}
                  for g, hh in h.items()},
        # zero-evidence closed form (per gender): value = a - c + b*d —
        # the consumer fallback for UUIDs missing from singles_players.csv
        "impute": {g: {"a": round(h[g]["a"] - C_NONRANKED, 4),
                       "b": round(h[g]["b"], 4)} for g in ("M", "F")},
        "date_before": DATE_BEFORE,
    }
    (ROOT / "model" / "singles_model.json").write_text(
        json.dumps(summary, indent=1))
    print(f"\nwrote model/singles_model.json (ref {F['ref']})")


if __name__ == "__main__":
    main()
