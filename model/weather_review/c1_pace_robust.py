"""C1b — stress-test the PACE result (the one measure that moved).

    python model/weather_review/c1_pace_robust.py

Runs, all on corrected venue labels with the indoor arm as control:
  A. spec ladder (no FE -> event FE -> event x date FE, + court + stage)
  B. filter sensitivity for the duration sanity window
  C. finer wind bins (is it a gradient or a calm-vs-breeze step?)
  D. outcome transforms (log s/point, trimmed mean, median-ish)
  E. MATCH-level decomposition: seconds per RALLY vs rallies per point,
     using data/match_rally_summary.csv — separates "rallies got shorter"
     from "there was less dead time between points"
  F. leave-one-event-out jackknife on the headline slope
  G. a placebo: wind at the SAME hour on a different day of the same event
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_lib import (ROOT, event_tz, get_tz, label_arms, load_hourly,  # noqa
                    local_hour_key, parse_utc, read_csv)
from c1_build_pace import END_COLS, calibrate_offsets, naive_local  # noqa: E402
from c1_texture import (SCRATCH, cluster_boot_ols, design, fmt,  # noqa: E402
                        load_pace)

NB = 1000


def attach(rows):
    """add hour, court, stage, day key."""
    tzs = event_tz()
    mt = {r["match_id"]: r for r in read_csv(ROOT / "data/match_times.csv")}
    stage = {}
    for g in read_csv(ROOT / "data/games.csv"):
        stage[g["game_id"]] = g["stage"] or ""
    for r in rows:
        t = mt.get(r["match_id"])
        ts = parse_utc(t[END_COLS[r["gn"] - 1]]) if t else None
        hk = local_hour_key(ts, tzs[r["event_id"]]) if ts else None
        r["hour"] = int(hk[11:13]) if hk else -1
        r["court"] = (t or {}).get("court", "") or "?"
        st = stage.get(r["game_id"], "")
        r["round"] = st.split("—")[-1].strip()[:18] if st else "?"
        r["evday"] = r["event_id"] + "|" + r["date"]
    return rows


def keep(rows, arm, dmin=120.0, dmax=2700.0, smin=8.0, smax=200.0):
    out = []
    for r in rows:
        if r["corrected_all"] != arm:
            continue
        if not (dmin <= r["dur"] <= dmax):
            continue
        if not (smin <= r["spp"] <= smax):
            continue
        if math.isnan(r["temp"]):
            continue
        out.append(r)
    return out


BASE = [("const", lambda r: 1.0),
        ("wind10", lambda r: r["wind"] / 10.0),
        ("temp10", lambda r: r["temp"] / 10.0),
        ("inv_points", lambda r: 1.0 / r["points"]),
        ("first_game", lambda r: float(r["first_game"])),
        ("fmt15", lambda r: 1.0 if r["fmt"] == "sideout_15" else 0.0),
        ("gn3plus", lambda r: 1.0 if r["gn"] >= 3 else 0.0)]

HR = ("hr", lambda r: r["hour"])
EV = ("ev", lambda r: r["event_id"])
ED = ("evday", lambda r: r["evday"])
CT = ("court", lambda r: r["court"])
RD = ("round", lambda r: r["round"])


def absorb(rows, fe_fns, mats):
    """Alternating within-group demeaning (Frisch-Waugh) for >=1 FE sets.

    mats: list of 1-D arrays (regressors and outcome) to be residualised on
    the FE dummies. Returns the residualised copies. Used for the
    high-dimensional event-DAY specs, where explicit dummies would make the
    bootstrap intractable; the cluster bootstrap then treats the absorbed FE
    as fixed (a mild understatement of uncertainty, noted in the writeup).
    """
    groups = []
    for _, fn in fe_fns:
        vals = [fn(r) for r in rows]
        lv = {v: i for i, v in enumerate(sorted(set(vals)))}
        groups.append(np.fromiter((lv[v] for v in vals), int, len(vals)))
    out = [m.astype(float).copy() for m in mats]
    for _ in range(30):
        for g in groups:
            for m in out:
                cnt = np.bincount(g)
                s = np.bincount(g, weights=m)
                m -= (s / cnt)[g]
    return out


def run(sub, cats, y=None, want="wind10", nb=NB, cl_key="event_id",
        absorb_fe=None):
    if y is None:
        y = np.array([r["spp"] for r in sub])
    if absorb_fe:
        X, names = design(sub, BASE, cats)
        keepc = [i for i, n in enumerate(names) if n != "const"]
        mats = [X[:, i] for i in keepc] + [np.asarray(y, float)]
        res = absorb(sub, absorb_fe, mats)
        Xa = np.column_stack(res[:-1])
        ya = res[-1]
        names = [names[i] for i in keepc]
        return cluster_boot_ols(Xa, ya, [r[cl_key] for r in sub], names,
                                [want], nboot=nb)[want]
    X, names = design(sub, BASE, cats)
    res = cluster_boot_ols(X, y, [r[cl_key] for r in sub], names, [want], nboot=nb)
    return res[want]


def section(title):
    print("\n" + "-" * 74 + "\n" + title + "\n" + "-" * 74)


def main():
    rows = attach(load_pace())
    arms = {a: keep(rows, a) for a in ("outdoor", "indoor")}

    section("A. SPEC LADDER — s/point per +10 mph (cluster boot over events)")
    ladders = [
        ("no FE (pooled)", [], None),
        ("hour-of-day FE", [HR], None),
        ("event + hour FE  [headline]", [HR, EV], None),
        ("event + hour + court FE", [HR, EV, CT], None),
        ("event + hour + round FE", [HR, EV, RD], None),
        ("event-DAY + hour FE (absorbed)", [], [ED, HR]),
        ("event-DAY + hour + court (abs)", [], [ED, HR, CT]),
    ]
    for arm in ("outdoor", "indoor"):
        print(f"\n[{arm}] n={len(arms[arm])}", flush=True)
        for lab, cats, abs_fe in ladders:
            print("   %-32s %s"
                  % (lab, fmt(run(arms[arm], cats, absorb_fe=abs_fe),
                              "s/pt/10mph", 1.0, 3)), flush=True)

    section("B. DURATION FILTER SENSITIVITY (outdoor, event+hour FE)")
    for dmin, dmax in ((0.1, 1e9), (60, 3600), (120, 2700), (180, 2100),
                       (240, 1800)):
        sub = keep(rows, "outdoor", dmin, dmax)
        print("   dur in [%6.0f,%7.0f]s  n=%5d  %s"
              % (dmin, dmax, len(sub), fmt(run(sub, [HR, EV]), "s/pt/10mph",
                                           1.0, 3)))

    section("C. FINER WIND BINS (outdoor, event+hour FE; ref = 0-2 mph)")
    edges = [0, 2, 4, 6, 8, 10, 12, 14, 18, 99]
    sub = arms["outdoor"]

    def blab(r):
        for i in range(len(edges) - 1):
            if edges[i] <= r["wind"] < edges[i + 1]:
                return "%02d-%s" % (edges[i], edges[i + 1] if edges[i + 1] < 99 else "+")
        return "?"
    cols = [c for c in BASE if c[0] != "wind10"]
    X, names = design(sub, cols, [HR, EV, ("wb", blab)])
    want = [n for n in names if n.startswith("wb=")]
    y = np.array([r["spp"] for r in sub])
    res = cluster_boot_ols(X, y, [r["event_id"] for r in sub], names, want,
                           nboot=NB)
    cnt = Counter(blab(r) for r in sub)
    print("   ref bin 00-2  n=%d" % cnt["00-2"])
    for w in want:
        b, lo, hi = res[w]
        print("      %-6s n=%5d  %+.2f [%+.2f, %+.2f]" % (w[3:], cnt[w[3:]], b, lo, hi))

    section("D. OUTCOME TRANSFORMS (outdoor / indoor, event+hour FE)")
    for arm in ("outdoor", "indoor"):
        sub = arms[arm]
        y = np.array([r["spp"] for r in sub])
        print("   [%s] raw        %s" % (arm, fmt(run(sub, [HR, EV], y),
                                                  "s/pt/10mph", 1.0, 3)))
        b, lo, hi = run(sub, [HR, EV], np.log(y))
        print("   [%s] log        %+.2f%% [%+.2f%%, %+.2f%%] per +10 mph"
              % (arm, 100 * b, 100 * lo, 100 * hi))
        lo_q, hi_q = np.quantile(y, [0.02, 0.98])
        s2 = [r for r in sub if lo_q <= r["spp"] <= hi_q]
        print("   [%s] 2%%-trimmed %s" % (arm, fmt(run(s2, [HR, EV]),
                                                   "s/pt/10mph", 1.0, 3)))

    section("E. MATCH-LEVEL DECOMPOSITION: s/rally vs rallies/point")
    match_decomp()

    section("F. LEAVE-ONE-EVENT-OUT JACKKNIFE (outdoor headline)")
    sub = arms["outdoor"]
    evs = sorted({r["event_id"] for r in sub})
    base = run(sub, [HR, EV], nb=1)[0]
    jk = []
    for e in evs:
        s2 = [r for r in sub if r["event_id"] != e]
        jk.append((run(s2, [HR, EV], nb=1)[0], e, sum(1 for r in sub
                                                      if r["event_id"] == e)))
    jk.sort()
    print("   full-sample %+.3f ; LOO range %+.3f .. %+.3f over %d events"
          % (base, jk[0][0], jk[-1][0], len(evs)))
    print("   most influential (largest |shift|):")
    for b, e, n in sorted(jk, key=lambda t: -abs(t[0] - base))[:5]:
        print("      drop %s (n=%4d) -> %+.3f" % (e[:8], n, b))

    section("G. PLACEBO — wind at the same hour, WRONG DAY of the same event")
    placebo(rows)


def match_decomp():
    """Match-level: total playing time vs rallies and points."""
    mt_rows = read_csv(ROOT / "data/match_times.csv")
    tzs = event_tz()
    offsets = calibrate_offsets(mt_rows, tzs)
    hourly = load_hourly()
    arms = label_arms()
    summ = {r["match_id"]: r for r in read_csv(ROOT / "data/match_rally_summary.csv")}
    rows = []
    for t in mt_rows:
        ev = t["event_id"]
        tzname = tzs.get(ev, "")
        tz = get_tz(tzname)
        if tz is None or ev not in offsets:
            continue
        s = summ.get(t["match_id"])
        if not s or s["discipline"] != "doubles":
            continue
        ends = [parse_utc(t[c]) for c in END_COLS]
        ends = [e for e in ends if e]
        sl = naive_local(t["start_local"])
        if not ends or sl is None:
            continue
        start = (sl - offsets[ev]).replace(tzinfo=tz).astimezone(dt.timezone.utc)
        dur = (ends[-1] - start).total_seconds()
        try:
            nral, npts = int(s["n_rallies"]), int(s["n_points"])
        except ValueError:
            continue
        if npts < 15 or nral < 20 or not (300 <= dur <= 7200):
            continue
        hk = local_hour_key(ends[0], tzname)
        wx = hourly.get((ev, hk))
        if not wx or wx["wind"] is None or wx["temp"] is None:
            continue
        rows.append({"event_id": ev, "hour": int(hk[11:13]), "wind": wx["wind"],
                     "temp": wx["temp"], "dur": dur, "npts": npts, "nral": nral,
                     "s_per_point": dur / npts, "s_per_rally": dur / nral,
                     "rallies_per_point": nral / npts,
                     "arm": arms["corrected_all"].get(ev) or ""})
    cols = [("const", lambda r: 1.0), ("wind10", lambda r: r["wind"] / 10.0),
            ("temp10", lambda r: r["temp"] / 10.0),
            ("inv_points", lambda r: 1.0 / r["npts"])]
    cats = [HR, EV]
    for arm in ("outdoor", "indoor"):
        sub = [r for r in rows if r["arm"] == arm]
        if len(sub) < 300:
            print(f"   [{arm}] n={len(sub)} too thin")
            continue
        X, names = design(sub, cols, cats)
        cl = [r["event_id"] for r in sub]
        print("   [%s] n=%d matches, %d events"
              % (arm, len(sub), len({r["event_id"] for r in sub})))
        for v, dec in (("s_per_point", 3), ("s_per_rally", 3),
                       ("rallies_per_point", 4)):
            y = np.array([r[v] for r in sub])
            res = cluster_boot_ols(X, y, cl, names, ["wind10"], nboot=NB)
            b, lo, hi = res["wind10"]
            print("      %-18s mean %7.3f  %+.*f [%+.*f, %+.*f] per +10 mph"
                  "  (%+.1f%% [%+.1f%%, %+.1f%%])"
                  % (v, y.mean(), dec, b, dec, lo, dec, hi,
                     100 * b / y.mean(), 100 * lo / y.mean(), 100 * hi / y.mean()))


def placebo(rows):
    """Reassign each game the wind observed at the same local hour but on a
    DIFFERENT day of the same event. A real effect should vanish; a
    confound with event/hour composition should survive."""
    hourly = load_hourly()
    by_ev_hour = defaultdict(list)
    for (ev, hk), wx in hourly.items():
        if wx["wind"] is not None:
            by_ev_hour[(ev, hk[11:13])].append((hk[:10], wx["wind"]))
    rng = np.random.default_rng(4242)
    sub = keep(rows, "outdoor")
    fake = []
    for r in sub:
        cands = [w for d, w in by_ev_hour.get((r["event_id"], "%02d" % r["hour"]), [])
                 if d != r["date"]]
        if not cands:
            continue
        r2 = dict(r)
        r2["wind"] = float(rng.choice(cands))
        fake.append(r2)
    print("   n=%d (placebo wind drawn from other days, same event+hour)" % len(fake))
    print("   %s" % fmt(run(fake, [HR, EV]), "s/pt/10mph", 1.0, 3))
    print("   true-wind same subsample: %s"
          % fmt(run([r for r in sub
                     if by_ev_hour.get((r["event_id"], "%02d" % r["hour"]))],
                    [HR, EV]), "s/pt/10mph", 1.0, 3))


if __name__ == "__main__":
    main()
