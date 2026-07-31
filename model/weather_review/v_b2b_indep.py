"""ADVERSARIAL independent re-derivation of B2b.

Everything here is written from the raw CSVs without importing b2b_lib, so a
shared bug in the tester's loader cannot propagate.  Adds:
  * FE estimator via explicit within-event demeaned OLS (closed form) as a
    cross-check on the weighted-mean formula
  * a CORRECT cluster bootstrap (multiplicity preserved) next to the tester's
    dict-keyed one (which silently de-duplicates resampled events)
  * wild cluster bootstrap (Rademacher) at event level
  * exact randomization test, own RNG / own seeds
  * ratio-of-means aggregation (sum sq / sum noise) as a different estimand
  * MLP vs PPA-decider decomposition of the switch arm
  * wind_source (hourly vs daily-max) composition by bin
  * event overlap between the DiD's switch and placebo arms
"""
from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRATCH = Path("/tmp/claude-0/-home-user-pickleball/"
               "a427a3a4-6690-5ae8-9453-094c68f7122d/scratchpad")
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import team_eta  # noqa: E402


def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ labels
def labels():
    geo = {r["event_id"]: r["setting"] for r in rd(ROOT / "data/event_geo.csv")}
    pub = {k: (v if v in ("indoor", "outdoor") else None) for k, v in geo.items()}
    call, chi = dict(pub), dict(pub)
    for r in rd(ROOT / "data/venue_overrides.csv"):
        e, s, c = r["event_id"], r["setting"], r["confidence"]
        call[e] = s if s in ("indoor", "outdoor") else None
        chi[e] = s if (c == "high" and s in ("indoor", "outdoor")) else None
    return {"published": pub, "corrected_all": call, "corrected_hi": chi}


# ------------------------------------------------------------------ context
def context(setmap):
    daily = {}
    for r in rd(ROOT / "data/event_weather.csv"):
        try:
            daily[(r["event_id"], r["date"])] = float(r["windspeed_10m_max"])
        except ValueError:
            pass
    hourly = {}
    for r in rd(ROOT / "data/event_weather_hourly.csv"):
        try:
            hourly[(r["event_id"], r["local_time"][:13])] = float(r["windspeed_10m"])
        except ValueError:
            pass
    hour = {}
    for r in rd(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            hour[r["match_id"]] = ts[:13]
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in rd(ROOT / "data/v2_players.csv")}
    bym = defaultdict(list)
    for g in rd(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        bym[g["match_id"]].append(g)
    out = {}
    for mid, gs in bym.items():
        g0 = gs[0]
        st = setmap.get(g0["event_id"])
        w = daily.get((g0["event_id"], g0["date"]))
        if st is None or w is None:
            continue
        src = "daily"
        hw = hourly.get((g0["event_id"], hour.get(mid, "")))
        if hw is not None:
            w, src = hw, "hour"
        vals = [v2.get(g0[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        out[mid] = dict(setting=st, wind=w, src=src, tour=g0["tour"],
                        event=g0["event_id"], date=g0["date"],
                        best_of=int(g0["best_of"] or 0),
                        eta=team_eta(*vals) if all(v is not None for v in vals) else 0.0,
                        fmt={int(g["game_number"]): g for g in gs})
    return out


def bin_of(m):
    if m["setting"] == "indoor":
        return "INDOOR"
    w = m["wind"]
    if w < 8:
        return "calm"
    if w < 14:
        return "moderate"
    return "windy"


def z2(xa, n1, ya, n2):
    x, y = xa / n1, ya / n2
    p = (xa + ya) / (n1 + n2)
    noise = p * (1 - p) * (1 / n1 + 1 / n2)
    sq = (x - y) ** 2
    return sq, noise, (sq / noise if noise > 0 else 0.0)


# ------------------------------------------------------------------ units
def build(splits, ctx, arm):
    """arm in {'switch','noswitch'}; returns list of dicts."""
    out = []
    for r in splits:
        mid, gn = r["match_id"], int(r["game_number"])
        m = ctx.get(mid)
        if not m:
            continue
        g = m["fmt"].get(gn)
        if g is None:
            continue
        if m["tour"] == "MLP":
            if gn != 1:
                continue
            sw = True
        else:
            sw = (m["best_of"] == 3 and gn == 3) or (m["best_of"] == 5 and gn == 5)
        if arm == "switch" and not sw:
            continue
        if arm == "noswitch" and (sw or m["tour"] == "MLP"):
            continue
        pre = int(r["pa_pre"]) + int(r["pb_pre"])
        post = int(r["pa_post"]) + int(r["pb_post"])
        if pre < 5 or post < 5:
            continue
        b = bin_of(m)
        if b is None:
            continue
        sq, noise, z = z2(int(r["pa_pre"]), pre, int(r["pa_post"]), post)
        out.append(dict(ev=m["event"], bin=b, wind=m["wind"], src=m["src"],
                        tour=m["tour"], mid=mid, gn=gn, z=z, sq=sq,
                        noise=noise, eta=m["eta"], fmt=g["scoring_format"],
                        date=m["date"], seq_ok=r.get("seq_ok"),
                        boundary_ok=r.get("boundary_ok")))
    return out


# ------------------------------------------------------------- estimators
def paired_cells(units, trt, ref="calm"):
    T, C = defaultdict(list), defaultdict(list)
    for u in units:
        if u["bin"] == trt:
            T[u["ev"]].append(u["z"])
        elif u["bin"] == ref:
            C[u["ev"]].append(u["z"])
    return {e: (T[e], C[e]) for e in T if e in C}


def wgt(nt, nc, w):
    if w == "unit":
        return 1.0
    if w == "att":
        return float(nt)
    return nt * nc / (nt + nc)


def paired(cells, w="fe"):
    num = den = 0.0
    for e, (t, c) in cells.items():
        if not t or not c:
            continue
        ww = wgt(len(t), len(c), w)
        num += ww * (sum(t) / len(t) - sum(c) / len(c))
        den += ww
    return num / den if den else float("nan")


def fe_ols(cells):
    """Within-event demeaned OLS of z on the treated dummy. Closed form."""
    num = den = 0.0
    for e, (t, c) in cells.items():
        n = len(t) + len(c)
        if n < 2 or not t or not c:
            continue
        allz = t + c
        d = [1.0] * len(t) + [0.0] * len(c)
        md, mz = sum(d) / n, sum(allz) / n
        for di, zi in zip(d, allz):
            num += (di - md) * (zi - mz)
            den += (di - md) ** 2
    return num / den if den else float("nan")


def boot_correct(cells, stat, n=4000, seed=1):
    ks = list(cells)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        smp = {}
        for i in range(len(ks)):
            smp[i] = cells[ks[rng.randrange(len(ks))]]
        v = stat(smp)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))], vals


def boot_dedup(cells, stat, n=4000, seed=1):
    """Reproduces the tester's b2b_did.py bootstrap (dict comprehension
    silently collapses duplicate events)."""
    ks = list(cells)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        picked = [ks[rng.randrange(len(ks))] for _ in ks]
        smp = {e: cells[e] for e in picked}
        v = stat(smp)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))], vals


def randomization(cells, w="fe", n=20000, seed=99):
    rng = random.Random(seed)
    obs = paired(cells, w)
    ge = 0
    ge2 = 0
    for _ in range(n):
        sh = {}
        for e, (t, c) in cells.items():
            pool = t + c
            rng.shuffle(pool)
            sh[e] = (pool[:len(t)], pool[len(t):])
        v = paired(sh, w)
        if v >= obs:
            ge += 1
        if abs(v) >= abs(obs):
            ge2 += 1
    return obs, (ge + 1) / (n + 1), (ge2 + 1) / (n + 1)


def wildboot(cells, w="fe", n=4000, seed=5):
    """Rademacher wild cluster bootstrap-t on the event-level differences
    (imposing the null d_e = 0 by sign-flipping the demeaned d_e)."""
    ds, ws = [], []
    for e, (t, c) in cells.items():
        ds.append(sum(t) / len(t) - sum(c) / len(c))
        ws.append(wgt(len(t), len(c), w))
    G = len(ds)
    W = sum(ws)
    est = sum(a * b for a, b in zip(ws, ds)) / W

    def tstat(dd):
        m = sum(a * b for a, b in zip(ws, dd)) / W
        v = sum(a * a * (b - m) ** 2 for a, b in zip(ws, dd)) * G / (G - 1) / W / W
        return m / math.sqrt(v) if v > 0 else 0.0
    t0 = tstat(ds)
    # impose null: centre the differences
    cen = [d - est for d in ds]
    rng = random.Random(seed)
    ts = []
    for _ in range(n):
        dd = [c * (1 if rng.random() < .5 else -1) for c in cen]
        ts.append(tstat(dd))
    p1 = (sum(1 for t in ts if t >= t0) + 1) / (n + 1)
    p2 = (sum(1 for t in ts if abs(t) >= abs(t0)) + 1) / (n + 1)
    # CI by inverting: percentile-t
    ts.sort()
    se = est / t0 if t0 else float("nan")
    lo = est - ts[int(.975 * n)] * se
    hi = est - ts[int(.025 * n)] * se
    return est, t0, p1, p2, (lo, hi)


def loo(cells, w="fe"):
    base = paired(cells, w)
    rows = []
    for e in cells:
        sub = {k: v for k, v in cells.items() if k != e}
        rows.append((paired(sub, w) - base, e, len(cells[e][0])))
    rows.sort()
    return base, rows


def fmt(x):
    return f"{x:+.3f}"


def main():
    splits = rd(ROOT / "data/decider_splits.csv")
    reb = rd(SCRATCH / "rebuilt_splits.csv")
    L = labels()
    print("=" * 78)
    print("PART 1 — Design B windy-vs-calm, committed decider_splits.csv")
    print("=" * 78)
    for arm in ("published", "corrected_all", "corrected_hi"):
        ctx = context(L[arm])
        u = build(splits, ctx, "switch")
        cnt = defaultdict(int)
        for x in u:
            cnt[x["bin"]] += 1
        print(f"\n--- labels={arm}  n={len(u)}  bins={dict(cnt)}")
        # wind source composition
        for b in ("calm", "moderate", "windy", "INDOOR"):
            s = [x for x in u if x["bin"] == b]
            if not s:
                continue
            hr = sum(1 for x in s if x["src"] == "hour") / len(s)
            print(f"    {b:9s} n={len(s):5d} hour-joined={hr:5.1%} "
                  f"meanwind={statistics.mean(x['wind'] for x in s):5.2f} "
                  f"meanz2={statistics.mean(x['z'] for x in s):.3f} "
                  f"ratio-of-means={sum(x['sq'] for x in s)/sum(x['noise'] for x in s):.3f}")
        cells = paired_cells(u, "windy")
        nt = sum(len(t) for t, c in cells.values())
        nc = sum(len(c) for t, c in cells.values())
        print(f"    paired events G={len(cells)}  windy n={nt}  calm-at-those-events n={nc}")
        print(f"    paired FE (weighted mean)={fmt(paired(cells,'fe'))}   "
              f"FE via demeaned OLS={fmt(fe_ols(cells))}")
        print(f"    paired ATT={fmt(paired(cells,'att'))}  unit={fmt(paired(cells,'unit'))}")
        for w in ("fe", "att", "unit"):
            los, his = [], []
            for sd in (1, 2, 3):
                lo, hi, _ = boot_correct(cells, lambda c, w=w: paired(c, w),
                                         n=4000, seed=sd)
                los.append(lo)
                his.append(hi)
            o, p1, p2 = randomization(cells, w, n=20000, seed=99)
            est, t0, wp1, wp2, wci = wildboot(cells, w, n=4000, seed=5)
            print(f"    [{w:4s}] est={fmt(o)} boot95=[{fmt(min(los))}..{fmt(max(los))}, "
                  f"{fmt(min(his))}..{fmt(max(his))}] rand p1={p1:.4f} p2={p2:.4f} "
                  f"| wild p1={wp1:.4f} p2={wp2:.4f} wildCI=[{fmt(wci[0])},{fmt(wci[1])}]")
        b0, rows = loo(cells, "fe")
        print("    LOO (FE) most influential:",
              ", ".join(f"{e[:8]}({n_}g){d:+.3f}" for d, e, n_ in rows[:2] + rows[-2:]))
        # MLP vs PPA decomposition of the switch arm
        for tour in ("MLP", "PPA"):
            su = [x for x in u if x["tour"] == tour]
            cc = paired_cells(su, "windy")
            if not cc:
                print(f"    {tour}-only: no paired events")
                continue
            ntt = sum(len(t) for t, c in cc.values())
            print(f"    {tour}-only paired: G={len(cc)} windy_n={ntt} "
                  f"FE={fmt(paired(cc,'fe'))} ATT={fmt(paired(cc,'att'))}")
        # trimming / outlier sensitivity (drop top 1% of z2 overall)
        allz = sorted(x["z"] for x in u)
        cut = allz[int(.99 * len(allz))]
        u2 = [x for x in u if x["z"] <= cut]
        cc = paired_cells(u2, "windy")
        print(f"    winsor-drop top1% z2 (>{cut:.1f}, {len(u)-len(u2)} games): "
              f"FE={fmt(paired(cc,'fe'))} ATT={fmt(paired(cc,'att'))}")

    print()
    print("=" * 78)
    print("PART 2 — DiD switch vs no-switch placebo (rebuilt splits)")
    print("=" * 78)
    for arm in ("published", "corrected_all"):
        ctx = context(L[arm])
        sw = [x for x in build(reb, ctx, "switch") if x["fmt"] == "sideout_11"]
        ns = [x for x in build(reb, ctx, "noswitch") if x["fmt"] == "sideout_11"]
        cs = paired_cells(sw, "windy")
        cn = paired_cells(ns, "windy")
        print(f"\n--- labels={arm}")
        print(f"    switch n={len(sw)} placebo n={len(ns)}")
        print(f"    switch paired G={len(cs)} windy_n={sum(len(t) for t,_ in cs.values())}"
              f" FE={fmt(paired(cs,'fe'))}")
        print(f"    placebo paired G={len(cn)} windy_n={sum(len(t) for t,_ in cn.values())}"
              f" FE={fmt(paired(cn,'fe'))}")
        print(f"    events: switch-only={len(set(cs)-set(cn))} both={len(set(cs)&set(cn))} "
              f"placebo-only={len(set(cn)-set(cs))}  union={len(set(cs)|set(cn))}")
        # composition of the switch arm's windy games
        wsw = [x for x in sw if x["bin"] == "windy" and x["ev"] in cs]
        print("    switch windy by tour:",
              dict((t, sum(1 for x in wsw if x["tour"] == t)) for t in ("MLP", "PPA")))
        est = paired(cs, "fe") - paired(cn, "fe")
        allev = sorted(set(cs) | set(cn))

        def stat_dict(smp):
            """smp: {key: event_name} preserving multiplicity."""
            a, b = {}, {}
            for k, e in smp.items():
                if e in cs:
                    a[k] = cs[e]
                if e in cn:
                    b[k] = cn[e]
            if not a or not b:
                return float("nan")
            return paired(a, "fe") - paired(b, "fe")
        cells_ev = {e: e for e in allev}
        loC, hiC, _ = boot_correct(cells_ev, stat_dict, n=4000, seed=1)

        def stat_ded(smp):
            a = {e: cs[e] for e in smp.values() if e in cs}
            b = {e: cn[e] for e in smp.values() if e in cn}
            if not a or not b:
                return float("nan")
            return paired(a, "fe") - paired(b, "fe")
        loD, hiD, _ = boot_dedup(cells_ev, lambda c: stat_ded({i: e for i, e in enumerate(c)}),
                                 n=4000, seed=1)
        print(f"    DiD={fmt(est)}  CORRECT cluster boot95=[{fmt(loC)},{fmt(hiC)}]  "
              f"tester's dedup boot95=[{fmt(loD)},{fmt(hiD)}]")
        # randomization on DiD
        rng = random.Random(777)
        ge = 0
        N = 20000
        for _ in range(N):
            pa, pb = {}, {}
            for cell, dst in ((cs, pa), (cn, pb)):
                for e, (t, c) in cell.items():
                    pool = t + c
                    rng.shuffle(pool)
                    dst[e] = (pool[:len(t)], pool[len(t):])
            if paired(pa, "fe") - paired(pb, "fe") >= est:
                ge += 1
        print(f"    DiD randomization p1={(ge+1)/(N+1):.4f}")


if __name__ == "__main__":
    main()
