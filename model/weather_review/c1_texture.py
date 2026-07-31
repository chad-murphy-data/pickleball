"""C1 — does wind change the TEXTURE of play (not just who wins)?

    python model/weather_review/c1_texture.py            # all measures
    python model/weather_review/c1_texture.py pace       # one measure

Four measures, each run OUTDOOR (corrected labels) with INDOOR as the
falsification control, each with a cluster bootstrap over EVENTS:

  1. PACE        seconds per point vs wind at the game's actual hour
  2. SIDE-OUTS   side-outs per point and k_match vs wind (match level)
  3. SCORE SHAPE blowout / deuce / margin dispersion at fixed skill gap
  4. GUSTS       measures 1 and 3 re-run on gust speed

PRE-SPECIFIED (written before looking at any output):
  * PACE signal  = outdoor slope >= +1.0 s/point per +10 mph with a 95%
    cluster-bootstrap CI excluding 0, while the indoor control slope is
    smaller in magnitude and/or spans 0.  (+1 s/point ~ +20 s on a
    20-point game ~ 2% of game length; below that nobody would notice.)
  * SIDE-OUT signal = outdoor slope >= +0.02 side-outs per point per
    +10 mph, CI excluding 0, indoor arm smaller.  (0.02 ~ +0.4 side-outs
    in a game to 11 — one extra rotation.)
  * SCORE-SHAPE signal = >= 2.0 pp change in blowout rate OR in
    deuce rate per +10 mph, CI excluding 0, indoor arm smaller; or a
    >= 5% change in residual margin variance.
  * Anything smaller than these thresholds, or matched by the indoor
    control, is a NULL and gets reported with its MDE.

Everything reads committed data only. Deterministic (all RNG seeded).
"""
from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_lib import (ROOT, event_tz, game_eta, get_tz, label_arms,  # noqa: E402
                    load_games, load_hourly, load_v2, local_hour_key,
                    parse_utc, read_csv)
import datetime as dt  # noqa: E402
from c1_build_pace import calibrate_offsets, naive_local, END_COLS  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-user-pickleball/"
               "a427a3a4-6690-5ae8-9453-094c68f7122d/scratchpad")
NBOOT = 1000
SEED = 20260731

# duration sanity window: a game to 11 that took under 2 min or over 45 min
# is a clock artefact (court change, rain delay, feed glitch), not pace.
DUR_MIN, DUR_MAX = 120.0, 2700.0
SPP_MIN, SPP_MAX = 8.0, 200.0


# --------------------------------------------------------------- inference
def cluster_boot_ols(X, y, clusters, names, want, nboot=NBOOT, seed=SEED,
                     ridge=1e-6):
    """OLS + cluster bootstrap over `clusters`; returns {name: (b, lo, hi)}.

    Uses per-cluster cross-products so a resample is a weighted sum of
    precomputed p x p matrices (fast enough for 1000 draws x ~140 cols).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    keys = sorted(set(clusters))
    idx = {k: i for i, k in enumerate(keys)}
    cid = np.fromiter((idx[c] for c in clusters), int, len(clusters))
    C = len(keys)
    S = np.zeros((C, p, p))
    T = np.zeros((C, p))
    for c in range(C):
        m = cid == c
        Xc = X[m]
        S[c] = Xc.T @ Xc
        T[c] = Xc.T @ y[m]
    lam = ridge * np.trace(S.sum(0)) / p
    I = np.eye(p) * lam

    def solve(counts):
        A = np.tensordot(counts, S, axes=(0, 0)) + I
        b = counts @ T
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    beta = solve(np.ones(C))
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(C, np.full(C, 1.0 / C), size=nboot).astype(float)
    boots = np.empty((nboot, len(want)))
    wi = [names.index(w) for w in want]
    for i in range(nboot):
        boots[i] = solve(draws[i])[wi]
    out = {}
    for j, w in enumerate(want):
        col = np.sort(boots[:, j])
        out[w] = (beta[names.index(w)],
                  col[int(.025 * nboot)], col[int(.975 * nboot)])
    return out


def fmt(res, unit, scale=1.0, dec=3):
    b, lo, hi = res
    se = (hi - lo) / (2 * 1.959964)
    m = 2.802 * se
    return ("%+.*f [%+.*f, %+.*f] %s   (MDE80 %.*f)"
            % (dec, b * scale, dec, lo * scale, dec, hi * scale, unit,
               dec, m * scale))


def design(rows, cols, cats):
    """Build X from numeric col fns + categorical FE fns. Returns X, names."""
    names, mats = [], []
    for nm, fn in cols:
        names.append(nm)
        mats.append(np.array([fn(r) for r in rows], float))
    for nm, fn in cats:
        vals = [fn(r) for r in rows]
        levels = sorted(set(vals))[1:]  # drop reference level
        for lv in levels:
            names.append(f"{nm}={lv}")
            mats.append(np.array([1.0 if v == lv else 0.0 for v in vals]))
    return np.column_stack(mats), names


# ------------------------------------------------------------------ MEASURE 1
def load_pace():
    path = SCRATCH / "c1_pace.csv"
    if not path.exists():
        raise SystemExit("run model/weather_review/c1_build_pace.py first")
    rows = []
    for r in csv.DictReader(open(path)):
        r["dur"] = float(r["dur"])
        r["points"] = int(r["points"])
        r["wind"] = float(r["wind"])
        r["gust"] = float(r["gust"]) if r["gust"] else float("nan")
        r["temp"] = float(r["temp"]) if r["temp"] else float("nan")
        r["first_game"] = int(r["first_game"])
        r["gn"] = int(r["game_number"])
        r["spp"] = r["dur"] / r["points"]
        rows.append(r)
    return rows


def pace_rows(rows, arm, label_col="corrected_all"):
    out = []
    for r in rows:
        if r[label_col] != arm:
            continue
        if not (DUR_MIN <= r["dur"] <= DUR_MAX):
            continue
        if not (SPP_MIN <= r["spp"] <= SPP_MAX):
            continue
        if math.isnan(r["temp"]):
            continue
        out.append(r)
    return out


PACE_COLS = [
    ("const", lambda r: 1.0),
    ("wind10", lambda r: r["wind"] / 10.0),
    ("temp10", lambda r: r["temp"] / 10.0),
    ("inv_points", lambda r: 1.0 / r["points"]),
    ("first_game", lambda r: float(r["first_game"])),
    ("fmt15", lambda r: 1.0 if r["fmt"] == "sideout_15" else 0.0),
    ("gn3plus", lambda r: 1.0 if r["gn"] >= 3 else 0.0),
]


def hour_of(r):
    return "NA"


def measure_pace(rows, tag="wind10", var="wind"):
    print("\n" + "=" * 74)
    print("MEASURE 1 — PACE: seconds per point vs %s (game's actual hour)"
          % ("sustained wind" if var == "wind" else "gusts"))
    print("=" * 74)
    cols = [c for c in PACE_COLS if c[0] != "wind10"]
    cols.insert(1, (tag, lambda r, v=var: r[v] / 10.0))
    for arm in ("outdoor", "indoor"):
        sub = pace_rows(rows, arm)
        if len(sub) < 400:
            print(f"\n[{arm}] n={len(sub)} — too thin, skipped")
            continue
        cats = [("hr", lambda r: r["hour"]), ("ev", lambda r: r["event_id"])]
        X, names = design(sub, cols, cats)
        y = np.array([r["spp"] for r in sub])
        res = cluster_boot_ols(X, y, [r["event_id"] for r in sub], names,
                               [tag, "temp10"])
        ev = len({r["event_id"] for r in sub})
        print(f"\n[{arm}]  n={len(sub)} games, {ev} events, "
              f"mean {y.mean():.1f} s/point (sd {y.std():.1f})")
        print("   slope  %s" % fmt(res[tag], "s/point per +10 mph", 1.0, 3))
        print("   (temp) %s" % fmt(res["temp10"], "s/point per +10 F", 1.0, 3))
        # secondary: hold total points fixed as well
        cols2 = cols + [("points", lambda r: float(r["points"]))]
        X2, n2 = design(sub, cols2, cats)
        r2 = cluster_boot_ols(X2, y, [r["event_id"] for r in sub], n2, [tag])
        print("   +points-controlled  %s"
              % fmt(r2[tag], "s/point per +10 mph", 1.0, 3))
        # binned dose-response (event+hour FE absorbed, wind entered as bins)
        binned_pace(sub, cols, cats, y, var)


def binned_pace(sub, cols, cats, y, var):
    bins = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 99)]

    def blab(r):
        w = r[var]
        for lo, hi in bins:
            if lo <= w < hi:
                return f"{lo}-{hi if hi < 99 else '+'}"
        return "?"
    cols_nb = [c for c in cols if c[0] not in ("wind10", "gust10")]
    cats_b = cats + [("wb", blab)]
    X, names = design(sub, cols_nb, cats_b)
    want = [n for n in names if n.startswith("wb=")]
    if not want:
        return
    res = cluster_boot_ols(X, y, [r["event_id"] for r in sub], names, want)
    cnt = Counter(blab(r) for r in sub)
    print("   dose-response vs reference bin (s/point):")
    for w in want:
        lab = w[3:]
        b, lo, hi = res[w]
        print("      %-6s n=%5d  %+.2f [%+.2f, %+.2f]" % (lab, cnt[lab], b, lo, hi))


# ------------------------------------------------------------------ MEASURE 2
def build_match_hours():
    """match_id -> (event_id, local hour key, hour int) from TRUE UTC stamps
    where available, else the calibrated local start."""
    mt_rows = read_csv(ROOT / "data/match_times.csv")
    tzs = event_tz()
    offsets = calibrate_offsets(mt_rows, tzs)
    out = {}
    src = Counter()
    for r in mt_rows:
        tzname = tzs.get(r["event_id"], "")
        tz = get_tz(tzname)
        if tz is None:
            continue
        ends = [parse_utc(r[c]) for c in END_COLS]
        ends = [e for e in ends if e]
        if ends:
            hk = local_hour_key(ends[0], tzname)
            src["game-end (true UTC)"] += 1
        else:
            sl = naive_local(r["start_local"] or r["planned_start_local"])
            if sl is None or r["event_id"] not in offsets:
                continue
            hk = (sl - offsets[r["event_id"]]).strftime("%Y-%m-%dT%H")
            src["start/planned local" if r["start_local"] else "planned local"] += 1
        out[r["match_id"]] = (r["event_id"], hk, int(hk[11:13]))
    return out, src


def measure_sideouts():
    print("\n" + "=" * 74)
    print("MEASURE 2 — SIDE-OUT TEXTURE: side-outs per point and k, vs wind")
    print("=" * 74)
    hours, src = build_match_hours()
    print("   match-hour source:", dict(src))
    hourly = load_hourly()
    arms = label_arms()
    rows = []
    for r in read_csv(ROOT / "data/match_rally_summary.csv"):
        h = hours.get(r["match_id"])
        if not h:
            continue
        ev, hk, hr = h
        wx = hourly.get((ev, hk))
        if not wx or wx["wind"] is None or wx["temp"] is None:
            continue
        try:
            npts, nso, nral = int(r["n_points"]), int(r["n_sideouts"]), int(r["n_rallies"])
            k = float(r["k_match"])
        except (ValueError, TypeError):
            continue
        if r["discipline"] != "doubles" or npts < 15 or nral < 20:
            continue
        rows.append({"event_id": ev, "hour": hr, "wind": wx["wind"],
                     "gust": wx["gust"], "temp": wx["temp"],
                     "so_per_point": nso / npts, "serve_rate": npts / nral,
                     "k": k, "npts": npts,
                     "arm": arms["corrected_all"].get(ev) or ""})
    cols = [("const", lambda r: 1.0), ("wind10", lambda r: r["wind"] / 10.0),
            ("temp10", lambda r: r["temp"] / 10.0)]
    cats = [("hr", lambda r: r["hour"]), ("ev", lambda r: r["event_id"])]
    for arm in ("outdoor", "indoor"):
        sub = [r for r in rows if r["arm"] == arm]
        if len(sub) < 300:
            print(f"\n[{arm}] n={len(sub)} — too thin")
            continue
        X, names = design(sub, cols, cats)
        cl = [r["event_id"] for r in sub]
        print(f"\n[{arm}]  n={len(sub)} matches, "
              f"{len({r['event_id'] for r in sub})} events")
        for out_var, unit, dec, scale in (
                ("so_per_point", "side-outs per point per +10 mph", 4, 1.0),
                ("k", "k (P(hold)) per +10 mph", 4, 1.0),
                ("serve_rate", "serve-point rate per +10 mph", 4, 1.0)):
            y = np.array([r[out_var] for r in sub])
            res = cluster_boot_ols(X, y, cl, names, ["wind10"])
            print("   %-13s mean %.3f   %s"
                  % (out_var, y.mean(), fmt(res["wind10"], unit, scale, dec)))


# ------------------------------------------------------------------ MEASURE 3
def build_score_rows():
    hours, src = build_match_hours()
    hourly = load_hourly()
    arms = label_arms()
    v2 = load_v2()
    tzs = event_tz()
    # game-hour where we have a true-UTC end stamp for that game
    game_hour = {}
    mt_rows = read_csv(ROOT / "data/match_times.csv")
    for r in mt_rows:
        tzname = tzs.get(r["event_id"], "")
        if get_tz(tzname) is None:
            continue
        for i, c in enumerate(END_COLS, 1):
            ts = parse_utc(r[c])
            if ts:
                game_hour[(r["match_id"], i)] = local_hour_key(ts, tzname)
    rows = []
    for mid, gs in load_games().items():
        h = hours.get(mid)
        if not h:
            continue
        ev, mhk, _ = h
        for g in gs:
            if g["scoring_format"] != "sideout_11":
                continue
            gn = int(g["game_number"])
            hk = game_hour.get((mid, gn), mhk)
            wx = hourly.get((ev, hk))
            if not wx or wx["wind"] is None or wx["temp"] is None:
                continue
            eta = game_eta(g, v2)
            if eta is None:
                continue
            a, b = int(g["t1_score"]), int(g["t2_score"])
            hi, lo = max(a, b), min(a, b)
            rows.append({"event_id": ev, "hour": int(hk[11:13]),
                         "day": hk[:10],
                         "wind": wx["wind"], "gust": wx["gust"],
                         "temp": wx["temp"], "eta": abs(eta),
                         "margin": hi - lo,
                         "blowout": 1.0 if lo <= 4 else 0.0,
                         "deuce": 1.0 if hi > 11 else 0.0,
                         "arm": arms["corrected_all"].get(ev) or ""})
    return rows


def measure_score_shape(rows, tag="wind10", var="wind"):
    print("\n" + "=" * 74)
    print("MEASURE 3 — SCORE SHAPE at fixed skill gap (%s)"
          % ("sustained wind" if var == "wind" else "gusts"))
    print("=" * 74)
    cols = [("const", lambda r: 1.0),
            (tag, lambda r, v=var: r[v] / 10.0),
            ("temp10", lambda r: r["temp"] / 10.0),
            ("gap", lambda r: r["eta"]),
            ("gap2", lambda r: r["eta"] ** 2)]
    cats = [("hr", lambda r: r["hour"]), ("ev", lambda r: r["event_id"])]
    for arm in ("outdoor", "indoor"):
        sub = [r for r in rows if r["arm"] == arm]
        if len(sub) < 500:
            print(f"\n[{arm}] n={len(sub)} — too thin")
            continue
        X, names = design(sub, cols, cats)
        cl = [r["event_id"] for r in sub]
        print(f"\n[{arm}]  n={len(sub)} games, "
              f"{len({r['event_id'] for r in sub})} events")
        for out_var, unit, scale, dec in (
                ("blowout", "pp per +10 mph", 100.0, 2),
                ("deuce", "pp per +10 mph", 100.0, 2),
                ("margin", "points per +10 mph", 1.0, 3)):
            y = np.array([r[out_var] for r in sub])
            res = cluster_boot_ols(X, y, cl, names, [tag])
            print("   %-8s mean %.3f   %s"
                  % (out_var, y.mean(), fmt(res[tag], unit, scale, dec)))
        # residual margin dispersion: squared residual from the gap model
        y = np.array([r["margin"] for r in sub])
        Xr, nr = design(sub, [c for c in cols if c[0] != tag], cats)
        bhat = np.linalg.lstsq(Xr, y, rcond=None)[0]
        r2 = (y - Xr @ bhat) ** 2
        res = cluster_boot_ols(X, r2, cl, names, [tag])
        b, lo, hi = res[tag]
        m = r2.mean()
        print("   margin-var  mean %.2f pts^2   %s   = %+.1f%% [%+.1f%%, %+.1f%%]"
              % (m, fmt(res[tag], "pts^2 per +10 mph", 1.0, 3),
                 100 * b / m, 100 * lo / m, 100 * hi / m))


# ----------------------------------------------------------------------- main
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    np.random.seed(SEED)
    if which in ("all", "pace", "gust"):
        pace = load_pace()
        # attach the local hour of the game (from the pace table's own key)
        tzs = event_tz()
        mt = {r["match_id"]: r for r in read_csv(ROOT / "data/match_times.csv")}
        for r in pace:
            t = mt.get(r["match_id"])
            ts = parse_utc(t[END_COLS[r["gn"] - 1]]) if t else None
            r["hour"] = (int(local_hour_key(ts, tzs[r["event_id"]])[11:13])
                         if ts else -1)
    if which in ("all", "pace"):
        measure_pace(pace)
    if which in ("all", "sideouts"):
        measure_sideouts()
    if which in ("all", "score", "gust"):
        srows = build_score_rows()
    if which in ("all", "score"):
        measure_score_shape(srows)
    if which in ("all", "gust"):
        print("\n\n#################  GUSTS  #################")
        for r in pace:
            if math.isnan(r["gust"]):
                r["gust"] = r["wind"]
        measure_pace(pace, tag="gust10", var="gust")
        measure_score_shape(srows, tag="gust10", var="gust")


if __name__ == "__main__":
    main()
