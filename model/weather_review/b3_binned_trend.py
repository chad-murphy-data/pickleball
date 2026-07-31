"""B3 part 3 — is the outdoor binned skill-compression a real dose-response?

rally_favorites_extras.py found the outdoor rally-level skill slope b
declining monotonically across wind bins
    0.4941 / 0.4917 / 0.4889 / 0.4739 / 0.4087
while the continuous adv x wind interaction is only -0.019 [-0.048,+0.013].
This script tests whether that decline is real, pre-specifying:

  T1 trend       precision-weighted OLS slope of b_k on the bin's mean wind,
                 recomputed inside a cluster bootstrap over EVENTS.
                 SIGNAL = negative with a CI excluding 0.
  T2 monotone    P(b_1 > b_2 > b_3 > b_4 > b_5) in the bootstrap.
                 SIGNAL = well above the 1/120 = 0.8% chance rate.
  T3 top-vs-rest b(16-40) - b(0-12), same bootstrap.
  T4 falsify     difference-in-differences: T3 outdoor - T3 indoor.  Indoor
                 courts cannot feel the wind, so an indoor top-bin drop of
                 the same size means the effect is composition, not weather.
                 SIGNAL = outdoor drop clearly larger than indoor.
  T5 events      name the outdoor 16-40 mph events; leave-one-out refit.
  T6 within      restrict to events that straddle the top bin and the calm
                 bins, and take the contrast inside those events only.

Cluster bootstrap over events throughout (games in an event share weather).
Deterministic: seeded.

    python model/weather_review/b3_binned_trend.py <scratch>
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rally_favorites_allmatches as P1  # noqa: E402

R_BOOT = 1000
SEED = 20260731
BINS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 40)]


# ------------------------------------------------------------- fitting ----
def fit2(rows, b0=None, iters=25):
    """Binomial logit  logit p = a + b*x  on rows of (wins, n, x)."""
    a, b = (b0 if b0 else (0.0, 0.0))
    exp = math.exp
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for wins, n, x in rows:
            z = a + b * x
            if z > 30.0:
                z = 30.0
            elif z < -30.0:
                z = -30.0
            p = 1.0 / (1.0 + exp(-z))
            res = wins - n * p
            w = n * p * (1.0 - p)
            g0 += res
            g1 += res * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            return None
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a += da
        b += db
        if abs(da) < 1e-10 and abs(db) < 1e-10:
            break
    return a, b


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    i = int(q * (len(v) - 1))
    return v[i]


def ci(v):
    return pct(v, 0.025), pct(v, 0.975)


def wls_slope(xs, ys, ws):
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    num = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    return num / den if den else float("nan")


# ---------------------------------------------------------------- main ----
def main():
    cells, _ = P1.build_cells()
    out = []
    say = lambda s="": (print(s), out.append(s))

    # event metadata for naming
    geo = {r["event_id"]: r for r in P1.read_csv(ROOT / "data/event_geo.csv")}
    ovr = {r["event_id"]: r for r in P1.read_csv(ROOT / "data/venue_overrides.csv")}

    def binof(w):
        for i, (lo, hi) in enumerate(BINS):
            if lo <= w < hi:
                return i
        return None

    for c in cells:
        c["bin"] = binof(c["wind"])

    say("# B3 part 3 — is the outdoor binned skill compression a dose-response?\n")
    say("Pre-specified before looking: SIGNAL requires (a) a negative binned")
    say("trend slope whose cluster-bootstrap CI excludes 0, (b) an outdoor")
    say("top-bin drop that the INDOOR arm does not reproduce, and (c) a top")
    say("bin that is not one tournament in disguise (leave-one-out stable).")
    say("Any one of those failing downgrades it to a composition artifact.\n")

    arms = {}
    for setting in ("outdoor", "indoor"):
        arms[setting] = [c for c in cells
                         if c["setting"] == setting and c["bin"] is not None]

    # ---- composition of each bin ---------------------------------------
    say("## 0. bin composition (what actually differs between bins)\n")
    for setting in ("outdoor", "indoor"):
        say(f"### {setting}")
        say(f"{'bin':>10} {'rallies':>8} {'cells':>6} {'events':>6} "
            f"{'meanW':>6} {'sd(adv)':>8} {'mean|adv|':>9} {'%MLP':>6} "
            f"{'%to-15':>7} {'medN':>5}")
        for i, (lo, hi) in enumerate(BINS):
            sub = [c for c in arms[setting] if c["bin"] == i]
            if not sub:
                continue
            n = sum(c["n"] for c in sub)
            evs = {c["ev"] for c in sub}
            mw = sum(c["n"] * c["wind"] for c in sub) / n
            ma = sum(c["n"] * c["adv"] for c in sub) / n
            va = sum(c["n"] * (c["adv"] - ma) ** 2 for c in sub) / n
            maa = sum(c["n"] * abs(c["adv"]) for c in sub) / n
            mlp = sum(c["n"] for c in sub if c["tour"] == "MLP") / n
            f15 = sum(c["n"] for c in sub if "11" not in c["fmt"]) / n
            ns = sorted(c["n"] for c in sub)
            say(f"{lo:>4}-{hi:<5} {n:>8} {len(sub):>6} {len(evs):>6} "
                f"{mw:>6.1f} {math.sqrt(va):>8.3f} {maa:>9.3f} {mlp:>6.1%} "
                f"{f15:>7.1%} {ns[len(ns)//2]:>5}")
        say("")

    # ---- point estimates + joint cluster bootstrap ----------------------
    say("## 1. binned skill slopes, jointly bootstrapped (T1/T2/T3/T4)\n")

    rows = {}      # (setting, bin) -> list of (wins, n, adv)
    byev = defaultdict(lambda: defaultdict(list))   # ev -> (setting,bin) -> rows
    xbar = {}
    for setting in ("outdoor", "indoor"):
        for i in range(len(BINS)):
            sub = [c for c in arms[setting] if c["bin"] == i]
            key = (setting, i)
            rows[key] = [(c["wins"], c["n"], c["adv"]) for c in sub]
            nn = sum(c["n"] for c in sub)
            xbar[key] = (sum(c["n"] * c["wind"] for c in sub) / nn) if nn else 0.0
            for c in sub:
                byev[c["ev"]][key].append((c["wins"], c["n"], c["adv"]))

    # "rest" pool = bins 0..2 (0-12 mph), the calm reference
    for setting in ("outdoor", "indoor"):
        rows[(setting, "rest")] = sum((rows[(setting, i)] for i in range(3)), [])
        for ev in byev:
            r = sum((byev[ev].get((setting, i), []) for i in range(3)), [])
            if r:
                byev[ev][(setting, "rest")] = r

    hat = {}
    for key, r in rows.items():
        f = fit2(r) if len(r) > 5 else None
        hat[key] = f

    say(f"{'arm':>9} {'bin':>10} {'meanW':>6}   b (point est)")
    for setting in ("outdoor", "indoor"):
        for i, (lo, hi) in enumerate(BINS):
            f = hat[(setting, i)]
            if f:
                say(f"{setting:>9} {lo:>4}-{hi:<5} {xbar[(setting,i)]:>6.1f}   "
                    f"{f[1]:+.4f}")
        f = hat[(setting, "rest")]
        say(f"{setting:>9} {'0-12 pool':>10} {'':>6}   {f[1]:+.4f}")
        say("")

    # bootstrap: resample EVENTS once, refit every (arm, bin) on the resample
    events = sorted(byev)
    rng = random.Random(SEED)
    draws = defaultdict(list)
    trend = {"outdoor": [], "indoor": []}
    mono = {"outdoor": 0, "indoor": 0}
    top_rest = {"outdoor": [], "indoor": []}
    did = []
    nok = 0
    for _ in range(R_BOOT):
        pick = [rng.choice(events) for _ in events]
        pool = defaultdict(list)
        for ev in pick:
            for key, r in byev[ev].items():
                pool[key].extend(r)
        bs = {}
        ok = True
        for setting in ("outdoor", "indoor"):
            for i in list(range(len(BINS))) + ["rest"]:
                key = (setting, i)
                r = pool.get(key)
                if not r or len(r) < 6 or hat[key] is None:
                    ok = False
                    continue
                f = fit2(r, b0=hat[key], iters=6)
                if f is None:
                    ok = False
                    continue
                bs[key] = f[1]
                draws[key].append(f[1])
        if not ok:
            continue
        nok += 1
        for setting in ("outdoor", "indoor"):
            ys = [bs[(setting, i)] for i in range(len(BINS))]
            xs = [xbar[(setting, i)] for i in range(len(BINS))]
            ws = [sum(n for _, n, _ in rows[(setting, i)]) for i in range(len(BINS))]
            trend[setting].append(wls_slope(xs, ys, ws) * 10.0)  # per +10 mph
            if all(ys[j] > ys[j + 1] for j in range(len(ys) - 1)):
                mono[setting] += 1
            top_rest[setting].append(bs[(setting, 4)] - bs[(setting, "rest")])
        did.append(top_rest["outdoor"][-1] - top_rest["indoor"][-1])

    say(f"cluster bootstrap over {len(events)} events, {nok}/{R_BOOT} usable\n")

    # point estimates of the derived statistics
    for setting in ("outdoor", "indoor"):
        ys = [hat[(setting, i)][1] for i in range(len(BINS))]
        xs = [xbar[(setting, i)] for i in range(len(BINS))]
        ws = [sum(n for _, n, _ in rows[(setting, i)]) for i in range(len(BINS))]
        t = wls_slope(xs, ys, ws) * 10.0
        lo, hi = ci(trend[setting])
        say(f"T1 trend  [{setting:>7}]  d b / d(10 mph) = {t:+.4f}  "
            f"[{lo:+.4f}, {hi:+.4f}]")
        say(f"T2 monotone-decreasing across all 5 bins: "
            f"{mono[setting]/max(nok,1):.1%}  (chance 0.8%)")
        tr = hat[(setting, 4)][1] - hat[(setting, "rest")][1]
        lo, hi = ci(top_rest[setting])
        say(f"T3 top(16-40) - calm(0-12) = {tr:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        say("")
    d0 = (hat[("outdoor", 4)][1] - hat[("outdoor", "rest")][1]
          - (hat[("indoor", 4)][1] - hat[("indoor", "rest")][1]))
    lo, hi = ci(did)
    say(f"T4 FALSIFICATION diff-in-diff (outdoor drop - indoor drop) = "
        f"{d0:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    say("")

    # per-bin bootstrap CIs for reference
    say("per-bin bootstrap CIs")
    for setting in ("outdoor", "indoor"):
        for i, (lo_, hi_) in enumerate(BINS):
            v = draws[(setting, i)]
            lo, hi = ci(v)
            say(f"    {setting:>7} {lo_:>2}-{hi_:<2}: {hat[(setting,i)][1]:+.4f} "
                f"[{lo:+.4f}, {hi:+.4f}]")
    say("")

    # ---- T5: who is in the outdoor top bin? -----------------------------
    say("## 2. T5 — is the outdoor 16-40 mph bin one tournament in disguise?\n")
    top = [c for c in arms["outdoor"] if c["bin"] == 4]
    ev_n = defaultdict(int)
    ev_cells = defaultdict(int)
    ev_dates = defaultdict(set)
    for c in top:
        ev_n[c["ev"]] += c["n"]
        ev_cells[c["ev"]] += 1
        ev_dates[c["ev"]].add(c["date"])
    tot = sum(ev_n.values())
    say(f"{'rallies':>8} {'share':>6} {'cells':>6} {'days':>5}  event")
    for ev, n in sorted(ev_n.items(), key=lambda kv: -kv[1]):
        g = geo.get(ev, {})
        o = ovr.get(ev, {})
        say(f"{n:>8} {n/tot:>6.1%} {ev_cells[ev]:>6} {len(ev_dates[ev]):>5}  "
            f"{g.get('event_name','?')} | {g.get('city','?')}, "
            f"{g.get('state','?')} | {min(ev_dates[ev])} | "
            f"conf={o.get('confidence','heuristic')}")
    say("")

    say("leave-one-event-out refit of the outdoor 16-40 slope "
        f"(full-sample {hat[('outdoor',4)][1]:+.4f}):")
    for ev, n in sorted(ev_n.items(), key=lambda kv: -kv[1]):
        r = [(c["wins"], c["n"], c["adv"]) for c in top if c["ev"] != ev]
        f = fit2(r) if len(r) > 5 else None
        nm = geo.get(ev, {}).get("event_name", ev)[:44]
        if f:
            say(f"    drop {nm:<46} n={sum(x[1] for x in r):>6}  b={f[1]:+.4f}")
        else:
            say(f"    drop {nm:<46} (no fit)")
    say("")

    # ---- T6: within-event contrast --------------------------------------
    say("## 3. T6 — within-event contrast (events straddling top and calm)\n")
    for setting in ("outdoor", "indoor"):
        sub = arms[setting]
        evs_top = {c["ev"] for c in sub if c["bin"] == 4}
        evs_calm = {c["ev"] for c in sub if c["bin"] is not None and c["bin"] <= 2}
        strad = evs_top & evs_calm
        st = [c for c in sub if c["ev"] in strad]
        rt = [(c["wins"], c["n"], c["adv"]) for c in st if c["bin"] == 4]
        rc = [(c["wins"], c["n"], c["adv"]) for c in st if c["bin"] <= 2]
        if len(rt) < 6 or len(rc) < 6:
            say(f"[{setting}] too few straddling events")
            continue
        ft, fc = fit2(rt), fit2(rc)
        # bootstrap over the straddling events only
        bev = defaultdict(lambda: ([], []))
        for c in st:
            if c["bin"] == 4:
                bev[c["ev"]][0].append((c["wins"], c["n"], c["adv"]))
            elif c["bin"] <= 2:
                bev[c["ev"]][1].append((c["wins"], c["n"], c["adv"]))
        ks = sorted(bev)
        r2 = random.Random(SEED + 7)
        dd = []
        for _ in range(R_BOOT):
            pk = [r2.choice(ks) for _ in ks]
            a1 = sum((bev[k][0] for k in pk), [])
            a2 = sum((bev[k][1] for k in pk), [])
            if len(a1) < 6 or len(a2) < 6:
                continue
            f1, f2 = fit2(a1, b0=ft, iters=6), fit2(a2, b0=fc, iters=6)
            if f1 and f2:
                dd.append(f1[1] - f2[1])
        lo, hi = ci(dd)
        say(f"[{setting}] {len(strad)} straddling events; "
            f"top n={sum(x[1] for x in rt)} b={ft[1]:+.4f}, "
            f"calm n={sum(x[1] for x in rc)} b={fc[1]:+.4f}")
        say(f"          within-event top-minus-calm = {ft[1]-fc[1]:+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}]  ({len(dd)} draws)")
    say("")

    # ---- the first four bins alone --------------------------------------
    say("## 4. the 0-16 mph range on its own (the well-observed part)\n")
    for setting in ("outdoor", "indoor"):
        ys = [hat[(setting, i)][1] for i in range(4)]
        xs = [xbar[(setting, i)] for i in range(4)]
        ws = [sum(n for _, n, _ in rows[(setting, i)]) for i in range(4)]
        t = wls_slope(xs, ys, ws) * 10.0
        tv = []
        for j in range(len(trend[setting])):
            pass
        say(f"[{setting}] trend over bins 1-4 only: {t:+.4f} b per +10 mph "
            f"(compare continuous d: outdoor -0.0191, indoor -0.0131)")
    say("")

    (Path(__file__).parent / "b3_binned_trend.txt").write_text("\n".join(out) + "\n")
    print("\nwrote model/weather_review/b3_binned_trend.txt")


if __name__ == "__main__":
    main()
