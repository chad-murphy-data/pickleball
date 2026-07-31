"""ADVERSARIAL VERIFICATION of B2a (corrected-venue-label re-run).

Independent re-implementation: builds the game table from source CSVs with its
own code, estimates the H4a skill x wind interaction d with (i) plain OLS,
(ii) an EVENT-CLUSTERED SANDWICH SE (analytic, no bootstrap), (iii) a
STRATIFIED cluster bootstrap (outdoor and indoor events resampled within arm,
unlike the tester's union resample), and (iv) a delete-one-event jackknife.

Then attacks the two load-bearing claims:
  (a) paired arm-a -> arm-c change in outdoor d = +0.039 [+0.004,+0.080]
  (b) within corrected-outdoor pool: verified-outdoor d = +0.114 vs unaudited
      d = -0.111, gap 0.225 at z ~ 4

    python model/weather_review/v_b2a_verify.py
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import sigmoid, team_eta  # noqa: E402

SEED = 20260731


def rd(p):
    with open(ROOT / p) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ build
def build_games():
    v2 = {r["player_id"]: float(r["value_now_mean"]) for r in rd("data/v2_players.csv")}
    hourly = {}
    for r in rd("data/event_weather_hourly.csv"):
        w = fnum(r["windspeed_10m"])
        if w is not None:
            hourly[(r["event_id"], r["local_time"][:13])] = (w, fnum(r["temperature_2m"]))
    hour, planned = {}, set()
    for r in rd("data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            hour[r["match_id"]] = ts[:13]
            if not r["start_local"]:
                planned.add(r["match_id"])
    out = []
    for g in rd("data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        hw = hourly.get((g["event_id"], hour.get(g["match_id"], "")))
        if hw is None:
            continue
        wind, temp = hw
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if any(v is None for v in vals):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        sk = sigmoid(team_eta(*vals)) - 0.5
        out.append(dict(ev=g["event_id"], mid=g["match_id"], tour=g["tour"],
                        date=g["date"], fmt=g["scoring_format"],
                        y=s1 / (s1 + s2) - 0.5, skill=sk, wind=wind,
                        w=wind / 10.0, sw=sk * wind / 10.0, temp=temp,
                        planned=g["match_id"] in planned))
    return out


def labels():
    geo = {r["event_id"]: r["setting"] for r in rd("data/event_geo.csv")}
    ov = {r["event_id"]: r for r in rd("data/venue_overrides.csv")}
    arms = {}
    for a in "abcd":
        m = {}
        for ev, heur in geo.items():
            o = ov.get(ev)
            if o is None:
                m[ev] = heur
                continue
            s, conf = o["setting"], o["confidence"]
            if a == "a":
                m[ev] = heur
            elif a == "b":
                m[ev] = (s if s in ("indoor", "outdoor") else None) if conf == "high" else heur
            else:
                if s == "unknown":
                    m[ev] = None
                elif s == "mixed":
                    m[ev] = "outdoor" if a == "d" else None
                elif conf in ("high", "medium"):
                    m[ev] = s
                else:
                    m[ev] = heur
        arms[a] = m
    return geo, ov, arms


# ------------------------------------------------------------------ estimators
def design(rows):
    X = np.array([[1.0, r["skill"], r["w"], r["sw"]] for r in rows])
    y = np.array([r["y"] for r in rows])
    ev = np.array([r["ev"] for r in rows])
    return X, y, ev


def ols_cluster(rows, kidx=3):
    """OLS coefficients + event-clustered sandwich SEs.  kidx = coefficient
    index of interest (3 = skill x wind)."""
    X, y, ev = design(rows)
    XtX = X.T @ X
    XtXi = np.linalg.inv(XtX)
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for e in np.unique(ev):
        m = ev == e
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    G = len(np.unique(ev))
    n, k = X.shape
    adj = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    V = XtXi @ meat @ XtXi * adj
    return beta, np.sqrt(np.diag(V)), G


def d_of(rows):
    if len(rows) < 50:
        return None
    X, y, _ = design(rows)
    try:
        return float(np.linalg.solve(X.T @ X, X.T @ y)[3])
    except np.linalg.LinAlgError:
        return None


def strat_boot(pool_rows, nboot=2000, seed=SEED):
    """Cluster bootstrap resampling events WITHIN the pool (stratified)."""
    cl = defaultdict(list)
    for r in pool_rows:
        cl[r["ev"]].append(r)
    keys = list(cl)
    rng = random.Random(seed)
    draws = []
    for _ in range(nboot):
        s = []
        for _ in keys:
            s.extend(cl[rng.choice(keys)])
        v = d_of(s)
        if v is not None:
            draws.append(v)
    draws.sort()
    return (draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))],
            float(np.std(draws, ddof=1)), len(draws))


def jack_events(pool_rows):
    """Delete-one-event jackknife: point spread + jackknife SE."""
    cl = defaultdict(list)
    for r in pool_rows:
        cl[r["ev"]].append(r)
    full = d_of(pool_rows)
    keys = list(cl)
    vals = {}
    for k in keys:
        sub = [r for kk, v in cl.items() if kk != k for r in v]
        vals[k] = d_of(sub)
    v = np.array([x for x in vals.values() if x is not None])
    G = len(v)
    se = math.sqrt((G - 1) / G * float(np.sum((v - v.mean()) ** 2)))
    worst = sorted(vals.items(), key=lambda kv: -abs((kv[1] or full) - full))[:3]
    return full, se, [(k, vals[k]) for k, _ in worst]


def within_event_d(rows, min_games=40):
    """Composition-free estimator: absorb event-specific INTERCEPT and
    event-specific SKILL slope (FWL), then regress residual y on residual
    w and residual skill*w.  d is then identified only from WITHIN-event
    covariation of wind with the skill loading."""
    by = defaultdict(list)
    for r in rows:
        by[r["ev"]].append(r)
    Y, W, SW, EV = [], [], [], []
    for ev, rs in by.items():
        if len(rs) < min_games:
            continue
        A = np.array([[1.0, r["skill"]] for r in rs])
        if np.std([r["wind"] for r in rs]) < 1e-9:
            continue          # no within-event wind variation -> no information
        P = A @ np.linalg.pinv(A)
        M = np.eye(len(rs)) - P
        Y.append(M @ np.array([r["y"] for r in rs]))
        W.append(M @ np.array([r["w"] for r in rs]))
        SW.append(M @ np.array([r["sw"] for r in rs]))
        EV += [ev] * len(rs)
    y = np.concatenate(Y); w = np.concatenate(W); sw = np.concatenate(SW)
    X = np.column_stack([w, sw])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    ev = np.array(EV)
    meat = np.zeros((2, 2))
    for e in np.unique(ev):
        m = ev == e
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    G = len(np.unique(ev))
    V = XtXi @ meat @ XtXi * (G / (G - 1.0))
    return float(beta[1]), float(math.sqrt(V[1, 1])), G, len(y)


def main():
    games = build_games()
    geo, ov, arms = labels()
    rep = {}
    P = print

    P(f"games with match-hour wind join: {len(games)}")
    for a in "abcd":
        o = [r for r in games if arms[a].get(r["ev"]) == "outdoor"]
        i = [r for r in games if arms[a].get(r["ev"]) == "indoor"]
        P(f"  arm {a}: outdoor {len(o)} ({len({r['ev'] for r in o})} ev)  "
          f"indoor {len(i)} ({len({r['ev'] for r in i})} ev)  "
          f"dropped {len(games)-len(o)-len(i)}")

    # ---------------- 1. arm-level d with three inference routes -------------
    P("\n=== H4a interaction d: OLS + clustered sandwich + stratified boot ===")
    pools = {}
    for a in "abcd":
        for s in ("outdoor", "indoor"):
            rows = [r for r in games if arms[a].get(r["ev"]) == s]
            pools[(a, s)] = rows
            beta, se, G = ols_cluster(rows)
            lo, hi, bse, nb = strat_boot(rows, nboot=1500, seed=SEED + hash(a + s) % 1000)
            P(f" arm {a} {s:7s} n={len(rows):6d} G={G:3d}  b={beta[1]:+.3f} "
              f"d={beta[3]:+.4f}  clusterSE={se[3]:.4f} "
              f"[{beta[3]-1.96*se[3]:+.3f},{beta[3]+1.96*se[3]:+.3f}]  "
              f"stratboot [{lo:+.3f},{hi:+.3f}] bse={bse:.4f}")
            rep[f"d_{a}_{s}"] = dict(n=len(rows), G=G, d=float(beta[3]),
                                     b=float(beta[1]), clusterse=float(se[3]),
                                     boot=[lo, hi], bootse=bse)

    # difference outdoor - indoor, independent events => SEs add in quadrature
    P("\n outdoor - indoor difference (independent event pools):")
    for a in "abcd":
        do, di = rep[f"d_{a}_outdoor"], rep[f"d_{a}_indoor"]
        diff = do["d"] - di["d"]
        se = math.hypot(do["clusterse"], di["clusterse"])
        seb = math.hypot(do["bootse"], di["bootse"])
        P(f"  arm {a}: {diff:+.4f}  clusterSE {se:.4f} [{diff-1.96*se:+.3f},"
          f"{diff+1.96*se:+.3f}]   bootSE {seb:.4f} [{diff-1.96*seb:+.3f},"
          f"{diff+1.96*seb:+.3f}]")
        rep[f"diff_{a}"] = dict(diff=diff, clusterse=se, bootse=seb)

    # ---------------- 2. decompose the arm-a -> arm-c outdoor change ---------
    P("\n=== decomposition of the arm-a -> arm-c OUTDOOR change ===")
    def cls_of(r):
        heur = geo[r["ev"]]
        o = ov.get(r["ev"])
        return f"{heur}->{o['setting'] if o else '(unaudited)'}"
    groups = defaultdict(list)
    for r in games:
        groups[cls_of(r)].append(r)
    for k in sorted(groups, key=lambda k: -len(groups[k])):
        g = groups[k]
        d = d_of(g)
        P(f"  {k:26s} n={len(g):6d} ev={len({r['ev'] for r in g}):3d} "
          f"MLP={100*sum(1 for r in g if r['tour']=='MLP')/len(g):4.0f}% "
          f"meanwind={sum(r['wind'] for r in g)/len(g):4.1f} "
          f"d={'   n/a' if d is None else f'{d:+.4f}'}")
    core = groups["outdoor->(unaudited)"] + groups["outdoor->outdoor"]
    removed = groups["outdoor->indoor"] + groups["outdoor->mixed"] + groups["outdoor->unknown"]
    added = groups["indoor->outdoor"]
    P(f"\n  arm a outdoor d = {d_of(core+removed):+.4f}  (n={len(core+removed)})")
    P(f"  core only       d = {d_of(core):+.4f}  (n={len(core)})   "
      f"[= arm a minus the removed games]")
    P(f"  arm c outdoor d = {d_of(core+added):+.4f}  (n={len(core+added)})")
    P(f"  removed subset  d = {d_of(removed):+.4f}  (n={len(removed)})")
    P(f"  added subset    d = {d_of(added):+.4f}  (n={len(added)})")

    # paired bootstrap of the a-minus-c change, different seed + jackknife
    cl = defaultdict(list)
    for r in games:
        cl[r["ev"]].append(r)
    keys = list(cl)
    def a_minus_c(rows):
        ao = [r for r in rows if arms["a"].get(r["ev"]) == "outdoor"]
        co = [r for r in rows if arms["c"].get(r["ev"]) == "outdoor"]
        da, dc = d_of(ao), d_of(co)
        return None if da is None or dc is None else da - dc
    pt = a_minus_c(games)
    rng = random.Random(SEED)
    draws = []
    for _ in range(1000):
        s = []
        for _ in keys:
            s.extend(cl[rng.choice(keys)])
        v = a_minus_c(s)
        if v is not None:
            draws.append(v)
    draws.sort()
    P(f"\n  paired a-c change (my seed, 1000 resamples): {pt:+.4f} "
      f"[{draws[int(.025*len(draws))]:+.4f},{draws[int(.975*len(draws))]:+.4f}]")
    # jackknife over events for the same paired statistic
    jv = []
    for k in keys:
        sub = [r for kk, v in cl.items() if kk != k for r in v]
        v = a_minus_c(sub)
        if v is not None:
            jv.append(v)
    jv = np.array(jv); G = len(jv)
    jse = math.sqrt((G - 1) / G * float(np.sum((jv - jv.mean()) ** 2)))
    P(f"  paired a-c change jackknife SE over {G} events: {jse:.4f} "
      f"-> [{pt-1.96*jse:+.4f},{pt+1.96*jse:+.4f}]")
    rep["paired_a_minus_c"] = dict(point=pt, boot=[draws[int(.025*len(draws))],
                                   draws[int(.975*len(draws))]], jse=jse)

    with open("/tmp/claude-0/-home-user-pickleball/"
              "a427a3a4-6690-5ae8-9453-094c68f7122d/scratchpad/v_b2a_part1.json",
              "w") as f:
        json.dump(rep, f, indent=1, default=float)


if __name__ == "__main__":
    main()
