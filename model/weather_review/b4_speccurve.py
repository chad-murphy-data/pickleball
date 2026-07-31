"""B4 parts 3-4 — SPECIFICATION CURVES for H4 (skill x wind interaction)
and H1 (serve-point rate vs wind).

    python model/weather_review/b4_speccurve.py

Enumerates the reasonable analyst choices and reports the DISTRIBUTION of
the estimate over the full grid, plus where the PUBLISHED specification
sits inside its own curve.

H4 grid (864 specs):
  wind timing  : match hour, actual-or-planned start (PUBLISHED)
                 match hour, ACTUAL start only
                 match hour, PLANNED start only
                 event-day daily maximum
  wind metric  : sustained 10 m (PUBLISHED) | gusts
  game sample  : all games (PUBLISHED) | game 1 only | deciders excluded
  labels       : heuristic event_geo (PUBLISHED) | corrected_all |
                 corrected_hi | audited_hi   (data/venue_overrides.csv)
  outcome      : point share − ½ (PUBLISHED) | win indicator − ½ | margin
  fixed effects: none (PUBLISHED) | tour dummy | EVENT fixed effects

H1 grid (384 specs): same wind choices/labels, outcome = serve-point rate
  (n_points / n_rallies per match), weighting rally-weighted (PUBLISHED)
  vs unweighted, minimum 20 (PUBLISHED) vs 40 rallies, same FE ladder.

Estimator plumbing: every spec is computed from per-EVENT Gram matrices,
so all label/FE/outcome variants are cheap linear algebra on the same
single pass over the data. Inference is the cluster-robust (CR1) sandwich
by event — the analytic twin of the event cluster bootstrap the published
tests used; agreement with a 2,000-replicate cluster bootstrap is checked
explicitly for the published specs and printed.

Stdlib only, deterministic. Writes model/weather_review/b4_speccurve.md
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(HERE))
from sitelib.race import sigmoid, team_eta  # noqa: E402
import b2b_lib as L  # noqa: E402

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------ linear algebra
def solve(A, b):
    n = len(b)
    m = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def inv(A):
    n = len(A)
    m = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            return None
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        for k in range(2 * n):
            m[c][k] /= d
        for r in range(n):
            if r != c:
                f = m[r][c]
                for k in range(2 * n):
                    m[r][k] -= f * m[c][k]
    return [row[n:] for row in m]


def fit_spec(grams, ns, xcols, ycol, demean):
    """grams: {event: 8x8 Gram of v=[1,skill,w,sw,mlp,y_share,y_win,y_margin]}
    Returns (beta, V_cr1, N, K) with CR1 cluster-robust covariance by event."""
    p = len(xcols)
    A = [[0.0] * p for _ in range(p)]
    c = [0.0] * p
    per = {}
    N = 0
    for e, G in grams.items():
        n = ns[e]
        if n <= 1:
            continue
        wsum = G[0][0]          # sum of weights (== n when unweighted)
        if demean:
            g = [[G[i][j] - G[0][i] * G[0][j] / wsum for j in range(8)]
                 for i in range(8)]
        else:
            g = G
        Ge = [[g[xcols[i]][xcols[j]] for j in range(p)] for i in range(p)]
        ce = [g[xcols[i]][ycol] for i in range(p)]
        for i in range(p):
            c[i] += ce[i]
            for j in range(p):
                A[i][j] += Ge[i][j]
        per[e] = (Ge, ce)
        N += n
    beta = solve(A, c)
    if beta is None:
        return None
    Ai = inv(A)
    if Ai is None:
        return None
    K = len(per)
    meat = [[0.0] * p for _ in range(p)]
    for e, (Ge, ce) in per.items():
        u = [ce[i] - sum(Ge[i][j] * beta[j] for j in range(p)) for i in range(p)]
        for i in range(p):
            for j in range(p):
                meat[i][j] += u[i] * u[j]
    dof = (K / max(K - 1, 1)) * ((N - 1) / max(N - p - (K if demean else 0), 1))
    V = [[dof * sum(Ai[i][a] * meat[a][b] * Ai[b][j]
                    for a in range(p) for b in range(p))
          for j in range(p)] for i in range(p)]
    return beta, V, N, K


# ------------------------------------------------------------------ data
def load_wind_sources():
    hourly = {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        try:
            hourly[(r["event_id"], r["local_time"][:13])] = (
                float(r["windspeed_10m"]), float(r["windgusts_10m"]))
        except (TypeError, ValueError):
            pass
    daily = {}
    for r in read_csv(ROOT / "data/event_weather.csv"):
        try:
            daily[(r["event_id"], r["date"])] = (
                float(r["windspeed_10m_max"]), float(r["windgusts_10m_max"]))
        except (TypeError, ValueError):
            pass
    hours = {}
    for r in read_csv(ROOT / "data/match_times.csv"):
        hours[r["match_id"]] = (
            (r["start_local"] or "")[:13] or None,
            (r["planned_start_local"] or "")[:13] or None)
    return hourly, daily, hours


TIMINGS = ["hour_pub", "hour_actual", "hour_planned", "daily_max"]
METRICS = ["sustained", "gust"]


def wind_for(timing, metric, ev, date, mid, hourly, daily, hours):
    k = 0 if metric == "sustained" else 1
    if timing == "daily_max":
        v = daily.get((ev, date))
        return v[k] if v else None
    act, plan = hours.get(mid, (None, None))
    h = {"hour_pub": act or plan, "hour_actual": act,
         "hour_planned": plan}[timing]
    if not h:
        return None
    v = hourly.get((ev, h))
    return v[k] if v else None


def game_rows():
    """One pass over games.csv; returns list of dicts with everything the
    grid needs (all four wind variants pre-joined)."""
    hourly, daily, hours = load_wind_sources()
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}
    rows = []
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        eta = team_eta(*vals)
        gn = int(g["game_number"] or 0)
        bo = int(g["best_of"] or 0)
        rec = {"ev": g["event_id"], "mlp": 1.0 if g["tour"] == "MLP" else 0.0,
               "skill": sigmoid(eta) - 0.5,
               "y_share": s1 / (s1 + s2) - 0.5,
               "y_win": (1.0 if s1 > s2 else 0.0) - 0.5,
               "y_margin": float(s1 - s2),
               "g1": gn == 1,
               "decider": (bo == 3 and gn == 3) or (bo == 5 and gn == 5)}
        for t in TIMINGS:
            for m in METRICS:
                rec[(t, m)] = wind_for(t, m, g["event_id"], g["date"],
                                       g["match_id"], hourly, daily, hours)
        rows.append(rec)
    return rows


def build_grams(rows, timing, metric, sample):
    """Per-event 8x8 Gram of v=[1,skill,w,sw,mlp,y_share,y_win,y_margin]."""
    grams, ns = {}, defaultdict(int)
    for r in rows:
        if sample == "g1_only" and not r["g1"]:
            continue
        if sample == "no_deciders" and r["decider"]:
            continue
        wv = r[(timing, metric)]
        if wv is None:
            continue
        w = wv / 10.0
        v = (1.0, r["skill"], w, r["skill"] * w, r["mlp"],
             r["y_share"], r["y_win"], r["y_margin"])
        e = r["ev"]
        G = grams.get(e)
        if G is None:
            G = grams[e] = [[0.0] * 8 for _ in range(8)]
        for i in range(8):
            vi = v[i]
            if vi == 0.0:
                continue
            Gi = G[i]
            for j in range(8):
                Gi[j] += vi * v[j]
        ns[e] += 1
    return grams, ns


# ------------------------------------------------------------------ H4 grid
YCOL = {"share": 5, "win": 6, "margin": 7}
FE_COLS = {"none": ([0, 1, 2, 3], False), "tour": ([0, 1, 2, 3, 4], False),
           "event": ([1, 2, 3], True)}
PUBLISHED_H4 = ("hour_pub", "sustained", "all", "published", "share", "none")


def pct(vals, q):
    v = sorted(vals)
    if not v:
        return float("nan")
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def summarize(specs, key, name, published_key=None):
    vals = [s[key] for s in specs]
    say(f"\n**{name}** — {len(specs)} specs")
    say(f"\n| median | IQR | min | max | frac CI excludes 0 | "
        f"frac negative (compression) |")
    say("|---|---|---|---|---|---|")
    nsig = sum(1 for s in specs if s[key + "_lo"] > 0 or s[key + "_hi"] < 0)
    nneg = sum(1 for s in specs if s[key] < 0)
    say(f"| {pct(vals,0.5):+.4f} | [{pct(vals,0.25):+.4f}, {pct(vals,0.75):+.4f}] "
        f"| {min(vals):+.4f} | {max(vals):+.4f} | {nsig}/{len(specs)} "
        f"({nsig/len(specs)*100:.0f}%) | {nneg}/{len(specs)} "
        f"({nneg/len(specs)*100:.0f}%) |")
    if published_key is not None:
        p = published_key[key]
        rank = sum(1 for v in vals if v < p) / len(vals)
        say(f"\nPUBLISHED spec = {p:+.4f} → percentile {rank*100:.0f} of its "
            f"own curve ({'null-most edge' if rank < 0.15 or rank > 0.85 else 'middle'}"
            f"; 0 = most negative/compression end, 100 = most positive).")
    return vals


def h4():
    say("\n# Part 3 — specification curve, H4 outdoor skill×wind interaction\n")
    say("Model per spec (OUTDOOR games only): "
        "`y = a + b·skill + c·(wind/10) + d·skill·(wind/10)`, skill = "
        "v2-expected point share − ½. **d** is the compression coefficient "
        "(negative = wind flattens the favourite). Because the three "
        "outcomes live on different scales, the scale-free curve statistic "
        "is **r = d/b**, the fraction of the favourite's skill edge erased "
        "per +10 mph; the raw-d curve is also given for the 216 "
        "point-share specs, directly comparable to the published "
        "+0.002 [−0.060, +0.064].\n")
    rows = game_rows()
    arms = L.label_arms()
    specs = []
    pub_rec = None
    for timing in TIMINGS:
        for metric in METRICS:
            for sample in ("all", "g1_only", "no_deciders"):
                grams, ns = build_grams(rows, timing, metric, sample)
                for arm_name, arm in arms.items():
                    keep = {e: G for e, G in grams.items()
                            if arm.get(e) == "outdoor"}
                    if len(keep) < 8:
                        continue
                    kns = {e: ns[e] for e in keep}
                    for out_name, ycol in YCOL.items():
                        for fe_name, (xcols, dm) in FE_COLS.items():
                            res = fit_spec(keep, kns, xcols, ycol, dm)
                            if not res:
                                continue
                            beta, V, N, K = res
                            ib = xcols.index(1)
                            idd = xcols.index(3)
                            b, d = beta[ib], beta[idd]
                            sd_d = math.sqrt(max(V[idd][idd], 0))
                            r = d / b if abs(b) > 1e-6 else float("nan")
                            var_r = (V[idd][idd] / b ** 2
                                     + d ** 2 * V[ib][ib] / b ** 4
                                     - 2 * d * V[idd][ib] / b ** 3)
                            sd_r = math.sqrt(max(var_r, 0))
                            rec = {"timing": timing, "metric": metric,
                                   "sample": sample, "labels": arm_name,
                                   "outcome": out_name, "fe": fe_name,
                                   "N": N, "K": K, "b": b,
                                   "d": d, "d_lo": d - 1.96 * sd_d,
                                   "d_hi": d + 1.96 * sd_d,
                                   "r": r, "r_lo": r - 1.96 * sd_r,
                                   "r_hi": r + 1.96 * sd_r}
                            specs.append(rec)
                            if (timing, metric, sample, arm_name, out_name,
                                    fe_name) == PUBLISHED_H4:
                                pub_rec = rec
    say(f"Grid: {len(specs)} estimable specifications.\n")
    say(f"PUBLISHED spec reproduces at d = {pub_rec['d']:+.4f} "
        f"[{pub_rec['d_lo']:+.4f}, {pub_rec['d_hi']:+.4f}] (CR1), "
        f"vs published +0.002 [−0.060, +0.064] (event bootstrap) — "
        f"n={pub_rec['N']}, {pub_rec['K']} events.")

    summarize(specs, "r", "FULL CURVE — r = d/b (fraction of favourite's "
              "edge erased per +10 mph)", pub_rec)
    share = [s for s in specs if s["outcome"] == "share"]
    summarize(share, "d", "POINT-SHARE outcome only — raw d", pub_rec)

    say("\n## Curve by analyst choice (median r, and how many of that "
        "slice's CIs exclude zero)\n")
    say("| dimension | level | specs | median r | IQR | CI≠0 |")
    say("|---|---|---|---|---|---|")
    for dim in ("timing", "metric", "sample", "labels", "outcome", "fe"):
        for lvl in sorted({s[dim] for s in specs}):
            sub = [s for s in specs if s[dim] == lvl]
            vals = [s["r"] for s in sub]
            nsig = sum(1 for s in sub if s["r_lo"] > 0 or s["r_hi"] < 0)
            say(f"| {dim} | {lvl} | {len(sub)} | {pct(vals,0.5):+.4f} "
                f"| [{pct(vals,0.25):+.4f}, {pct(vals,0.75):+.4f}] "
                f"| {nsig}/{len(sub)} |")

    say("\n## The composition question — event fixed effects\n")
    for fe in ("none", "tour", "event"):
        sub = [s for s in specs if s["fe"] == fe]
        vals = [s["r"] for s in sub]
        widths = [s["r_hi"] - s["r_lo"] for s in sub]
        nsig = sum(1 for s in sub if s["r_lo"] > 0 or s["r_hi"] < 0)
        say(f"- **FE = {fe}**: median r {pct(vals,0.5):+.4f}, "
            f"IQR [{pct(vals,0.25):+.4f}, {pct(vals,0.75):+.4f}], "
            f"range [{min(vals):+.4f}, {max(vals):+.4f}], "
            f"median CI width {pct(widths,0.5):.4f}, "
            f"{nsig}/{len(sub)} CIs exclude zero.")

    say("\n### Extreme specs (the ones that would make a headline)\n")
    say("| r | 95% CI | timing | metric | sample | labels | outcome | FE | n |")
    say("|---|---|---|---|---|---|---|---|---|")
    ordered = sorted(specs, key=lambda s: s["r"])
    for s in ordered[:4] + ordered[-4:]:
        say(f"| {s['r']:+.4f} | [{s['r_lo']:+.4f}, {s['r_hi']:+.4f}] "
            f"| {s['timing']} | {s['metric']} | {s['sample']} | {s['labels']} "
            f"| {s['outcome']} | {s['fe']} | {s['N']} |")

    # ------- bootstrap cross-check of the CR1 interval on the published spec
    say("\n### CR1 vs event cluster bootstrap (published spec, 2,000 reps)")
    grams, ns = build_grams(rows, "hour_pub", "sustained", "all")
    keep = {e: G for e, G in grams.items()
            if arms["published"].get(e) == "outdoor"}
    kns = {e: ns[e] for e in keep}
    keys = list(keep)
    rng = random.Random(2026)
    draws = []
    for _ in range(2000):
        A = [[0.0] * 4 for _ in range(4)]
        c = [0.0] * 4
        for _ in keys:
            e = keys[rng.randrange(len(keys))]
            G = keep[e]
            for i in range(4):
                c[i] += G[i][5]
                for j in range(4):
                    A[i][j] += G[i][j]
        bb = solve(A, c)
        if bb:
            draws.append(bb[3])
    draws.sort()
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
    say(f"\nbootstrap d CI [{lo:+.4f}, {hi:+.4f}] vs CR1 "
        f"[{pub_rec['d_lo']:+.4f}, {pub_rec['d_hi']:+.4f}] — the analytic "
        f"intervals used across the grid are the honest twin of the "
        f"published bootstrap.")
    return specs, pub_rec


# ------------------------------------------------------------------ H1 grid
def h1():
    say("\n\n# Part 4 — specification curve, H1 serve-point rate vs wind "
        "(outdoor)\n")
    say("Model per spec: `serve_rate = a + s·(wind/10)` on OUTDOOR matches "
        "with rally logs; **s** is the slope per +10 mph (published "
        "+0.0030 [−0.0009, +0.0072] at match hour, +0.0017 daily). "
        "serve_rate = n_points / n_rallies.\n")
    hourly, daily, hours = load_wind_sources()
    ev_of = {}
    for g in read_csv(ROOT / "data/games.csv"):
        ev_of.setdefault(g["match_id"], (g["event_id"], g["date"],
                                         g["tour"]))
    recs = []
    for r in read_csv(ROOT / "data/match_rally_summary.csv"):
        if r["discipline"] != "doubles":
            continue
        nr = int(r["n_rallies"])
        if nr < 20:
            continue
        meta = ev_of.get(r["match_id"])
        if not meta:
            continue
        ev, date, tour = meta
        rec = {"ev": ev, "nr": nr, "mlp": 1.0 if tour == "MLP" else 0.0,
               "rate": int(r["n_points"]) / nr}
        for t in TIMINGS:
            for m in METRICS:
                rec[(t, m)] = wind_for(t, m, ev, date, r["match_id"],
                                       hourly, daily, hours)
        recs.append(rec)
    arms = L.label_arms()
    specs, pub = [], None
    FE1 = {"none": ([0, 1], False), "tour": ([0, 1, 2], False),
           "event": ([1], True)}
    for timing in TIMINGS:
        for metric in METRICS:
            for minr in (20, 40):
                for weight in ("rallies", "equal"):
                    grams, ns = {}, defaultdict(int)
                    for r in recs:
                        if r["nr"] < minr:
                            continue
                        wv = r[(timing, metric)]
                        if wv is None:
                            continue
                        wt = r["nr"] if weight == "rallies" else 1.0
                        v = (1.0, wv / 10.0, r["mlp"], r["rate"], 0.0, 0.0,
                             0.0, 0.0)
                        G = grams.get(r["ev"])
                        if G is None:
                            G = grams[r["ev"]] = [[0.0] * 8 for _ in range(8)]
                        for i in range(8):
                            vi = v[i] * wt
                            if vi == 0.0:
                                continue
                            for j in range(8):
                                G[i][j] += vi * v[j]
                        ns[r["ev"]] += 1
                    for arm_name, arm in arms.items():
                        keep = {e: G for e, G in grams.items()
                                if arm.get(e) == "outdoor"}
                        if len(keep) < 8:
                            continue
                        kns = {e: ns[e] for e in keep}
                        for fe_name, (xcols, dm) in FE1.items():
                            res = fit_spec(keep, kns, xcols, 3, dm)
                            if not res:
                                continue
                            beta, V, N, K = res
                            iw = xcols.index(1)
                            # w is already wind/10, so beta_w IS per +10 mph
                            s = beta[iw]
                            sd = math.sqrt(max(V[iw][iw], 0))
                            rec = {"timing": timing, "metric": metric,
                                   "minr": minr, "weight": weight,
                                   "labels": arm_name, "fe": fe_name,
                                   "N": N, "K": K, "s": s,
                                   "s_lo": s - 1.96 * sd, "s_hi": s + 1.96 * sd}
                            specs.append(rec)
                            if (timing, metric, minr, weight, arm_name,
                                    fe_name) == ("hour_pub", "sustained", 20,
                                                 "rallies", "published",
                                                 "none"):
                                pub = rec
    say(f"Grid: {len(specs)} specifications. PUBLISHED spec reproduces at "
        f"s = {pub['s']:+.4f} [{pub['s_lo']:+.4f}, {pub['s_hi']:+.4f}] "
        f"(CR1) vs published +0.0030 [−0.0009, +0.0072].\n")
    vals = [s["s"] for s in specs]
    nsig = sum(1 for s in specs if s["s_lo"] > 0 or s["s_hi"] < 0)
    say("| median | IQR | min | max | frac CI excludes 0 | published pctile |")
    say("|---|---|---|---|---|---|")
    rank = sum(1 for v in vals if v < pub["s"]) / len(vals)
    say(f"| {pct(vals,0.5):+.4f} | [{pct(vals,0.25):+.4f}, "
        f"{pct(vals,0.75):+.4f}] | {min(vals):+.4f} | {max(vals):+.4f} "
        f"| {nsig}/{len(specs)} ({nsig/len(specs)*100:.0f}%) "
        f"| {rank*100:.0f} |")
    say("\n| dimension | level | specs | median s | IQR | CI≠0 |")
    say("|---|---|---|---|---|---|")
    for dim in ("timing", "metric", "minr", "weight", "labels", "fe"):
        for lvl in sorted({s[dim] for s in specs}, key=str):
            sub = [s for s in specs if s[dim] == lvl]
            v = [s["s"] for s in sub]
            ns_ = sum(1 for s in sub if s["s_lo"] > 0 or s["s_hi"] < 0)
            say(f"| {dim} | {lvl} | {len(sub)} | {pct(v,0.5):+.4f} "
                f"| [{pct(v,0.25):+.4f}, {pct(v,0.75):+.4f}] | {ns_}/{len(sub)} |")
    ordered = sorted(specs, key=lambda s: s["s"])
    say("\n| s | 95% CI | timing | metric | min rallies | weight | labels "
        "| FE | n |")
    say("|---|---|---|---|---|---|---|---|---|")
    for s in ordered[:3] + ordered[-3:]:
        say(f"| {s['s']:+.4f} | [{s['s_lo']:+.4f}, {s['s_hi']:+.4f}] "
            f"| {s['timing']} | {s['metric']} | {s['minr']} | {s['weight']} "
            f"| {s['labels']} | {s['fe']} | {s['N']} |")
    return specs, pub


def main():
    say("# B4 (3-4) — specification curves\n")
    h4()
    h1()
    (HERE / "b4_speccurve.md").write_text("\n".join(OUT) + "\n")
    print("\nwrote model/weather_review/b4_speccurve.md")


if __name__ == "__main__":
    main()
