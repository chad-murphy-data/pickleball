"""ADVERSARIAL VERIFICATION of phase-2 test B1 (heat).

Independent re-derivation, written from the raw CSVs rather than by importing
heat_test.py, of:
  (1) the sample definition (games with a match-hour weather join, by label set)
  (2) the PUBLISHED binned obs-pred statistic under heuristic AND audited labels
      -- including the 92F+ outdoor bin's event composition, which is the
      load-bearing claim under review
  (3) a continuous dose-response alternative to the bins (no thin-bin problem)
  (4) the primary antisymmetric skill x heat interaction d, by an independent
      closed-form route (no-intercept OLS on the odd design == symmetrization)

    python model/weather_review/v_b1_heat_verify.py
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import sigmoid, team_eta, game_win_prob  # noqa: E402


def rd(p):
    with open(ROOT / p) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ load
geo = {r["event_id"]: r for r in rd("data/event_geo.csv")}
ov = {r["event_id"]: r for r in rd("data/venue_overrides.csv")}
set_heur = {e: r["setting"] for e, r in geo.items()}
set_aud = {e: (ov[e]["setting"] if e in ov else r["setting"])
           for e, r in geo.items()}

hourly = {}
for r in rd("data/event_weather_hourly.csv"):
    try:
        hourly[(r["event_id"], r["local_time"][:13])] = (
            float(r["temperature_2m"]), float(r["windspeed_10m"]))
    except (TypeError, ValueError):
        pass

start = {}
for r in rd("data/match_times.csv"):
    ts = r["start_local"] or r["planned_start_local"]
    if ts:
        start[r["match_id"]] = (ts[:13], bool(r["start_local"]))

v2 = {r["player_id"]: float(r["value_now_mean"]) for r in rd("data/v2_players.csv")}

# cache the race DP (it is the slow part)
_pc = {}


def gwp(eta, T):
    k = (round(eta, 4), T)
    if k not in _pc:
        _pc[k] = game_win_prob(eta, T)
    return _pc[k]


G = []
drop = defaultdict(int)
for g in rd("data/games.csv"):
    if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
        continue
    sh = start.get(g["match_id"])
    if not sh:
        drop["no_start_time"] += 1
        continue
    w = hourly.get((g["event_id"], sh[0]))
    if w is None:
        drop["no_hourly"] += 1
        continue
    vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
    if not all(v is not None for v in vals):
        drop["unrated"] += 1
        continue
    s1, s2 = int(g["t1_score"]), int(g["t2_score"])
    if s1 + s2 < 11:
        drop["short_game"] += 1
        continue
    Tt = int(g["scoring_format"].rsplit("_", 1)[1])
    eta = team_eta(*vals)
    G.append({
        "ev": g["event_id"], "mid": g["match_id"], "date": g["date"],
        "tour": g["tour"], "T": Tt,
        "aud": set_aud.get(g["event_id"]), "heur": set_heur.get(g["event_id"]),
        "temp": w[0], "wind": w[1], "hour": int(sh[0][11:13]),
        "actual": sh[1],
        "y": s1 / (s1 + s2) - 0.5,
        "skill": sigmoid(eta) - 0.5, "eta": eta,
        "won1": 1.0 if s1 > s2 else 0.0,
    })
for r in G:
    p1_11 = gwp(r["eta"], 11)                 # tester's pred (always T=11)
    p1_T = gwp(r["eta"], r["T"])              # published pred (format-aware)
    r["p_fav11"] = max(p1_11, 1 - p1_11)
    r["p_favT"] = max(p1_T, 1 - p1_T)
    r["favwon"] = 1.0 if (r["won1"] == 1.0) == (r["eta"] >= 0) else 0.0

print(f"games kept: {len(G)}   drops: {dict(drop)}")
for lab, key in (("audited", "aud"), ("heuristic", "heur")):
    c = defaultdict(int)
    for r in G:
        c[r[key]] += 1
    print(f"  {lab:10s} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
n15 = sum(1 for r in G if r["T"] != 11)
print(f"  games with target != 11: {n15} "
      f"({sum(1 for r in G if r['T'] != 11 and r['aud']=='outdoor')} outdoor-audited)")

BINS = [("<70", -99, 70), ("70-82", 70, 82), ("82-92", 82, 92), ("92+", 92, 999)]


def binned(rows, predkey, nboot=4000, seed=11):
    """Bin edges + cluster bootstrap over EVENTS. No minimum-n drop rule:
    every resample contributes, and a resample with an empty bin simply omits
    that bin's draw (recorded so we can report how often it happens)."""
    cl = defaultdict(list)
    for r in rows:
        cl[r["ev"]].append(r)
    keys = list(cl)
    # per-event, per-bin sufficient stats: (n, sum_won, sum_pred)
    stat = {k: {} for k in keys}
    for k in keys:
        for lab, lo, hi in BINS:
            sub = [r for r in cl[k] if lo <= r["temp"] < hi]
            stat[k][lab] = (len(sub), sum(r["favwon"] for r in sub),
                            sum(r[predkey] for r in sub))
    obs = {}
    for lab, lo, hi in BINS:
        n = sum(stat[k][lab][0] for k in keys)
        if not n:
            continue
        obs[lab] = (n, sum(stat[k][lab][1] for k in keys) / n,
                    sum(stat[k][lab][2] for k in keys) / n)
    rng = random.Random(seed)
    dr = defaultdict(list)
    ddr = defaultdict(list)
    empty = defaultdict(int)
    nk = len(keys)
    for _ in range(nboot):
        acc = {lab: [0, 0.0, 0.0] for lab, _, _ in BINS}
        for _ in range(nk):
            s = stat[keys[rng.randrange(nk)]]
            for lab, _, _ in BINS:
                a, b, c = s[lab]
                t = acc[lab]
                t[0] += a
                t[1] += b
                t[2] += c
        e = {}
        for lab, _, _ in BINS:
            if acc[lab][0] == 0:
                empty[lab] += 1
                continue
            e[lab] = (acc[lab][1] - acc[lab][2]) / acc[lab][0]
        for lab in e:
            dr[lab].append(e[lab])
            if "<70" in e:
                ddr[lab].append(e[lab] - e["<70"])
    return obs, dr, ddr, empty, nk


def q(v, a):
    v = sorted(v)
    return v[int(a * len(v))]


print("\n=== (2) BINNED obs-pred FAVOURITE EDGE, match-hour temperature ===")
for labelset, key in (("HEURISTIC", "heur"), ("AUDITED", "aud")):
    for predkey, pname in (("p_favT", "format-aware pred (published spec)"),
                           ("p_fav11", "T=11 pred (tester's spec)")):
        for setting in ("outdoor", "indoor"):
            rows = [r for r in G if r[key] == setting]
            if len(rows) < 300:
                continue
            obs, dr, ddr, empty, nk = binned(rows, predkey)
            print(f"\n-- {labelset} / {setting} / {pname} "
                  f"({len(rows)} games, {nk} events)")
            for lab, _, _ in BINS:
                if lab not in obs:
                    continue
                n, o, p = obs[lab]
                d_obs = o - p
                lo, hi = q(dr[lab], .025), q(dr[lab], .975)
                s = (f"  {lab:>6}F n={n:6d} pred={p:.3f} obs={o:.3f} "
                     f"edge={d_obs:+.3f} [{lo:+.3f},{hi:+.3f}]")
                if lab != "<70" and ddr[lab]:
                    dd = d_obs - (obs['<70'][1] - obs['<70'][2])
                    s += (f"   vs<70: obs {dd:+.3f} boot-mean "
                          f"{np.mean(ddr[lab]):+.3f} "
                          f"[{q(ddr[lab], .025):+.3f},{q(ddr[lab], .975):+.3f}]")
                if empty[lab]:
                    s += f"  (bin empty in {empty[lab]}/4000 resamples)"
                print(s)

# ---- event composition of the tail bin -------------------------------
print("\n=== (2b) EVENT COMPOSITION of the 92F+ bin ===")
for labelset, key in (("HEURISTIC", "heur"), ("AUDITED", "aud")):
    for setting in ("outdoor", "indoor"):
        sub = [r for r in G if r[key] == setting and r["temp"] >= 92]
        if not sub:
            continue
        byev = defaultdict(list)
        for r in sub:
            byev[r["ev"]].append(r)
        print(f"\n-- {labelset}/{setting}: {len(sub)} games in "
              f"{len(byev)} events")
        for ev, rs in sorted(byev.items(), key=lambda kv: -len(kv[1])):
            o = sum(r["favwon"] for r in rs) / len(rs)
            p = sum(r["p_favT"] for r in rs) / len(rs)
            g = geo.get(ev, {})
            print(f"   n={len(rs):4d} edge={o-p:+.3f} {rs[0]['tour']:4s} "
                  f"{rs[0]['date']}  heur={set_heur.get(ev)} "
                  f"aud={set_aud.get(ev)}  {g.get('venue','')[:52]}")

# ---- (3) continuous dose-response ------------------------------------
print("\n=== (3) CONTINUOUS dose-response: (favwon - pred) ~ a + s*(T-75)/10 ===")


def slope_boot(rows, predkey, nboot=4000, seed=23):
    cl = defaultdict(list)
    for r in rows:
        cl[r["ev"]].append(r)
    keys = list(cl)
    suf = {}
    for k in keys:
        x = np.array([(r["temp"] - 75.0) / 10.0 for r in cl[k]])
        y = np.array([r["favwon"] - r[predkey] for r in cl[k]])
        suf[k] = np.array([len(x), x.sum(), (x * x).sum(), y.sum(), (x * y).sum()])
    tot = sum(suf.values())

    def fit(t):
        n, sx, sxx, sy, sxy = t
        den = n * sxx - sx * sx
        return (n * sxy - sx * sy) / den, (sy * sxx - sx * sxy) / den
    s0, a0 = fit(tot)
    rng = np.random.default_rng(seed)
    arr = np.array([suf[k] for k in keys])
    idx = rng.integers(0, len(keys), size=(nboot, len(keys)))
    ss = arr[idx].sum(axis=1)
    n, sx, sxx, sy, sxy = ss.T
    den = n * sxx - sx * sx
    sl = (n * sxy - sx * sy) / den
    return s0, a0, np.percentile(sl, [2.5, 97.5]), len(keys)


for labelset, key in (("HEURISTIC", "heur"), ("AUDITED", "aud")):
    for setting in ("outdoor", "indoor"):
        rows = [r for r in G if r[key] == setting]
        if len(rows) < 300:
            continue
        s0, a0, ci, nk = slope_boot(rows, "p_favT")
        print(f"  {labelset:10s} {setting:8s} n={len(rows):6d} ev={nk:3d} "
              f"slope={s0:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] per +10F  "
              f"(edge at 75F {a0:+.3f})")

# ---- (4) primary antisymmetric d, independent closed form -------------
print("\n=== (4) PRIMARY: share ~ b*skill + d*skill*h, ODD design ===")
print("    (no-intercept OLS on [skill, skill*h] == the tester's symmetrization)")


def odd_fit(rows, nboot=4000, seed=17, tempkey="temp"):
    cl = defaultdict(list)
    for r in rows:
        cl[r["ev"]].append(r)
    keys = list(cl)
    suf = []
    for k in keys:
        s = np.array([r["skill"] for r in cl[k]])
        h = np.array([(r[tempkey] - 75.0) / 10.0 for r in cl[k]])
        y = np.array([r["y"] for r in cl[k]])
        X = np.column_stack([s, s * h])
        suf.append(np.concatenate([(X.T @ X).ravel(), X.T @ y]))
    suf = np.array(suf)

    def fit(t):
        A = t[:4].reshape(2, 2)
        b = t[4:]
        return np.linalg.solve(A, b)
    pt = fit(suf.sum(axis=0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(nboot, len(keys)))
    ss = suf[idx].sum(axis=1)
    out = np.array([fit(t) for t in ss])
    return pt, np.percentile(out[:, 1], [2.5, 97.5]), len(keys)


for labelset, key in (("AUDITED", "aud"), ("HEURISTIC", "heur")):
    for setting in ("outdoor", "indoor"):
        rows = [r for r in G if r[key] == setting]
        if len(rows) < 300:
            continue
        pt, ci, nk = odd_fit(rows)
        print(f"  {labelset:10s} {setting:8s} n={len(rows):6d} ev={nk:3d} "
              f"b={pt[0]:.4f} d={pt[1]:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]")

# seed sensitivity on the primary
rows = [r for r in G if r["aud"] == "outdoor"]
print("  seed sweep (audited outdoor d CI):", end=" ")
for sd in (1, 17, 99, 2718):
    pt, ci, _ = odd_fit(rows, nboot=2000, seed=sd)
    print(f"[{ci[0]:+.4f},{ci[1]:+.4f}]", end=" ")
print()

# season splits, independent
print("\n=== (4b) season split of the primary d (audited outdoor) ===")
for yr in ("2024", "2025", "2026"):
    rs = [r for r in rows if r["date"][:4] == yr]
    pt, ci, nk = odd_fit(rs)
    print(f"  {yr} n={len(rs):6d} ev={nk:3d} d={pt[1]:+.4f} "
          f"[{ci[0]:+.4f},{ci[1]:+.4f}]")
