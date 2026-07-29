"""TASK B2a — corrected-venue-label re-run of H1 (serve rate vs wind) and
H4 (favorites x wind), under four label arms.

    python model/weather_review/label_arms_rerun.py

Arms
  a  heuristic            — data/event_geo.csv `setting` (what every published
                            test used); CONTROL, must reproduce the published
                            point estimates.
  b  high-conf flips      — heuristic, overridden ONLY where
                            venue_overrides.confidence == 'high'
                            (mixed/unknown at high confidence are dropped).
  c  all non-low, strict  — apply high+medium overrides; DROP every event whose
                            verified label is mixed or unknown (any
                            confidence); low-confidence indoor/outdoor events
                            keep the heuristic label.
  d  sensitivity          — as (c) but mixed -> outdoor (unknown still dropped).
  Events absent from venue_overrides.csv (33 small heuristic-outdoor events)
  keep the heuristic label in every arm.

Tests (identical estimators to the committed scripts, re-implemented here so
model/weather_report.py and model/favorites_wind.py stay untouched)
  H1  serve-point-rate WLS slope on match-hour wind, weight = n_rallies
      (weather_report.py cut A, hour version)
  H4a game-level OLS share-1/2 = a + b*skill + c*w + d*skill*w   -> d
      (favorites_wind.py regression 1)
  H4b favorite-minus-underdog serve-rally-rate gap = a + c*w     -> c
      (favorites_wind.py regression 2)

Everything is reported per setting AND as the OUTDOOR-minus-INDOOR difference,
which is the actual falsification statistic; CIs are cluster bootstraps over
EVENTS (a single event resample drives both arms of the difference, so the
difference CI is exact, not a subtraction of two intervals).
"""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import sigmoid, team_eta  # noqa: E402

NBOOT = 1500


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- estimators
def wls_slope(rows, xkey, ykey, wkey):
    sw = sx = sy = sxx = sxy = 0.0
    for r in rows:
        w, x, y = r[wkey], r[xkey], r[ykey]
        sw += w; sx += w * x; sy += w * y
        sxx += w * x * x; sxy += w * x * y
    den = sxx - sx * sx / sw if sw else 0.0
    return (sxy - sx * sy / sw) / den if den else None


def ols(rows, ykey, xkeys):
    p = len(xkeys) + 1
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for r in rows:
        x = [1.0] + [r[k] for k in xkeys]
        y = r[ykey]
        for i in range(p):
            xty[i] += x[i] * y
            for j in range(p):
                xtx[i][j] += x[i] * x[j]
    m = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r_: abs(m[r_][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            return None
        for r_ in range(p):
            if r_ != col:
                f = m[r_][col] / m[col][col]
                for c_ in range(col, p + 1):
                    m[r_][c_] -= f * m[col][c_]
    return [m[i][p] / m[i][i] for i in range(p)]


# ---------------------------------------------------------------- label arms
def load_labels():
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    ov = {r["event_id"]: r for r in read_csv(ROOT / "data/venue_overrides.csv")}

    def arm(name):
        out = {}
        for ev, heur in geo.items():
            o = ov.get(ev)
            if o is None:
                out[ev] = heur
                continue
            s, conf = o["setting"], o["confidence"]
            if name == "a":
                out[ev] = heur
            elif name == "b":
                if conf == "high":
                    out[ev] = s if s in ("indoor", "outdoor") else None
                else:
                    out[ev] = heur
            elif name in ("c", "d"):
                if s == "unknown":
                    out[ev] = None
                elif s == "mixed":
                    out[ev] = "outdoor" if name == "d" else None
                elif conf in ("high", "medium"):
                    out[ev] = s
                else:                      # low-confidence indoor/outdoor
                    out[ev] = heur
        return out

    return geo, ov, {k: arm(k) for k in "abcd"}


# ---------------------------------------------------------------- data build
def build():
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}
    hourly = {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        w = fnum(r["windspeed_10m"])
        if w is not None:
            hourly[(r["event_id"], r["local_time"][:13])] = (
                w, fnum(r["temperature_2m"]))
    start_hour, planned_only = {}, set()
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]
            if not r["start_local"]:
                planned_only.add(r["match_id"])
    rally = {r["match_id"]: r
             for r in read_csv(ROOT / "data/match_rally_summary.csv")
             if r["discipline"] == "doubles" and int(r["n_rallies"]) >= 20}

    match_rows, game_rows, match_meta = [], [], {}
    seen = set()
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        ev, mid = g["event_id"], g["match_id"]
        hw = hourly.get((ev, start_hour.get(mid, "")))
        if hw is None:
            continue
        wind, temp = hw
        # H1 row: one per match with rally logs
        if mid not in seen and mid in rally:
            seen.add(mid)
            rs = rally[mid]
            nr = int(rs["n_rallies"])
            match_rows.append({"ev": ev, "tour": g["tour"], "wind": wind,
                               "temp": temp, "n_rallies": nr, "w": float(nr),
                               "serve_rate": int(rs["n_points"]) / nr})
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        eta = team_eta(*vals)
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        game_rows.append({"ev": ev, "tour": g["tour"],
                          "y": s1 / (s1 + s2) - 0.5,
                          "skill": sigmoid(eta) - 0.5, "w": wind / 10.0,
                          "sw": (sigmoid(eta) - 0.5) * wind / 10.0,
                          "wind": wind, "temp": temp,
                          "planned": mid in planned_only})
        match_meta[mid] = (ev, wind, eta, g["tour"])

    gap_rows = []
    for r in read_csv(ROOT / "data/decider_serve_splits.csv"):
        meta = match_meta.get(r["match_id"])
        if not meta:
            continue
        ev, wind, eta, tour = meta
        if abs(eta) < 0.1:
            continue
        ra = int(r["ra_pre"]) + int(r["ra_post"])
        wa = int(r["wa_pre"]) + int(r["wa_post"])
        rb = int(r["rb_pre"]) + int(r["rb_post"])
        wb = int(r["wb_pre"]) + int(r["wb_post"])
        if ra < 8 or rb < 8:
            continue
        gap = (wa / ra - wb / rb) if eta > 0 else (wb / rb - wa / ra)
        gap_rows.append({"ev": ev, "tour": tour, "y": gap, "w": wind / 10.0,
                         "wind": wind})
    return match_rows, game_rows, gap_rows


# ---------------------------------------------------------------- inference
def paired_boot(rows_by_setting, stat, nboot=NBOOT, seed=11):
    """Cluster bootstrap over events. Returns dict with point + CI for
    outdoor, indoor and the outdoor-minus-indoor difference, where every
    replicate resamples ONE pool of events and recomputes all three."""
    clusters = defaultdict(list)
    setting_of = {}
    for setting, rows in rows_by_setting.items():
        for r in rows:
            clusters[r["ev"]].append(r)
            setting_of[r["ev"]] = setting
    keys = list(clusters)
    point = {s: stat(rows) for s, rows in rows_by_setting.items()}
    pdiff = (point["outdoor"] - point["indoor"]
             if point.get("outdoor") is not None and point.get("indoor") is not None
             else None)
    rng = random.Random(seed)
    draws = {"outdoor": [], "indoor": [], "diff": []}
    for _ in range(nboot):
        pool = {"outdoor": [], "indoor": []}
        for _ in keys:
            k = rng.choice(keys)
            pool[setting_of[k]].extend(clusters[k])
        vals = {}
        for s in ("outdoor", "indoor"):
            vals[s] = stat(pool[s]) if len(pool[s]) > 20 else None
            if vals[s] is not None:
                draws[s].append(vals[s])
        if vals["outdoor"] is not None and vals["indoor"] is not None:
            draws["diff"].append(vals["outdoor"] - vals["indoor"])
    res = {}
    for k in ("outdoor", "indoor", "diff"):
        v = sorted(draws[k])
        res[k] = (v[int(0.025 * len(v))], v[int(0.975 * len(v))]) if v else (None, None)
    return point, pdiff, res


def split(rows, labels):
    out = {"outdoor": [], "indoor": []}
    for r in rows:
        s = labels.get(r["ev"])
        if s in out:
            out[s].append(r)
    return out


def fmt(x, d=4):
    return "  n/a" if x is None else f"{x:+.{d}f}"


# ---------------------------------------------------------------- main
def main():
    geo, ov, arms = load_labels()
    match_rows, game_rows, gap_rows = build()
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    say("# B2a — corrected-venue-label re-run (H1 serve rate, H4 favorites)\n")
    say(f"rows: {len(match_rows)} matches (H1), {len(game_rows)} games (H4a), "
        f"{len(gap_rows)} games (H4b); all with a match-hour wind join.\n")

    # ---- label bookkeeping ------------------------------------------------
    say("## Label arms — game counts (H4a sample)\n")
    say("| arm | outdoor games | indoor games | dropped | outdoor events | indoor events |")
    say("|---|---|---|---|---|---|")
    for a in "abcd":
        sp = split(game_rows, arms[a])
        evo = len({r["ev"] for r in sp["outdoor"]})
        evi = len({r["ev"] for r in sp["indoor"]})
        say(f"| {a} | {len(sp['outdoor'])} | {len(sp['indoor'])} | "
            f"{len(game_rows)-len(sp['outdoor'])-len(sp['indoor'])} | {evo} | {evi} |")

    # confounding with tour + climate
    say("\n## What the correction is confounded with\n")
    say("Flip class = heuristic label -> verified label (arm c mapping), "
        "game-weighted over the H4a sample.\n")
    cls = defaultdict(lambda: {"n": 0, "MLP": 0, "PPA": 0, "wind": 0.0,
                               "temp": 0.0})
    for r in game_rows:
        o = ov.get(r["ev"])
        heur = geo[r["ev"]]
        ver = o["setting"] if o else "(unaudited)"
        k = f"{heur} -> {ver}"
        c = cls[k]
        c["n"] += 1
        c[r["tour"]] += 1
        c["wind"] += r["wind"]
        c["temp"] += r["temp"] if r["temp"] is not None else 0.0
    say("| heuristic -> verified | games | %MLP | mean match-hour wind | mean temp |")
    say("|---|---|---|---|---|")
    for k in sorted(cls, key=lambda k: -cls[k]["n"]):
        c = cls[k]
        say(f"| {k} | {c['n']} | {100*c['MLP']/c['n']:.0f}% | "
            f"{c['wind']/c['n']:.1f} mph | {c['temp']/c['n']:.0f} °F |")

    say("\n| arm | pool | games | %MLP | mean wind | P(wind>=14) | mean temp |")
    say("|---|---|---|---|---|---|---|")
    for a in "abcd":
        sp = split(game_rows, arms[a])
        for s in ("outdoor", "indoor"):
            rows = sp[s]
            n = len(rows)
            say(f"| {a} | {s} | {n} | "
                f"{100*sum(1 for r in rows if r['tour']=='MLP')/n:.0f}% | "
                f"{sum(r['wind'] for r in rows)/n:.1f} mph | "
                f"{100*sum(1 for r in rows if r['wind']>=14)/n:.1f}% | "
                f"{sum(r['temp'] for r in rows)/n:.0f} °F |")

    # ---- H1 ---------------------------------------------------------------
    say("\n## H1. Serve-point rate vs match-hour wind (WLS slope per +10 mph)\n")
    say("| arm | outdoor slope [95% CI] | n_out | indoor slope [95% CI] | n_in "
        "| OUTDOOR − INDOOR [95% CI] |")
    say("|---|---|---|---|---|---|")
    stat1 = lambda rows: (None if len(rows) < 20 else
                          (lambda s: None if s is None else s * 10.0)(
                              wls_slope(rows, "wind", "serve_rate", "w")))
    for a in "abcd":
        sp = split(match_rows, arms[a])
        pt, pd_, ci = paired_boot(sp, stat1, seed=101)
        say(f"| {a} | {fmt(pt['outdoor'])} [{fmt(ci['outdoor'][0])}, "
            f"{fmt(ci['outdoor'][1])}] | {len(sp['outdoor'])} "
            f"| {fmt(pt['indoor'])} [{fmt(ci['indoor'][0])}, {fmt(ci['indoor'][1])}] "
            f"| {len(sp['indoor'])} | {fmt(pd_)} [{fmt(ci['diff'][0])}, "
            f"{fmt(ci['diff'][1])}] |")

    # ---- H4a --------------------------------------------------------------
    say("\n## H4a. Game level: share−½ = a + b·skill + c·w + d·skill·w  (d)\n")
    say("| arm | outdoor d [95% CI] | b_out | indoor d [95% CI] | b_in "
        "| OUTDOOR − INDOOR d [95% CI] |")
    say("|---|---|---|---|---|---|")
    statd = lambda rows: (None if len(rows) < 50 else
                          (lambda c: None if c is None else c[3])(
                              ols(rows, "y", ["skill", "w", "sw"])))
    statb = lambda rows: (None if len(rows) < 50 else
                          (lambda c: None if c is None else c[1])(
                              ols(rows, "y", ["skill", "w", "sw"])))
    for a in "abcd":
        sp = split(game_rows, arms[a])
        pt, pd_, ci = paired_boot(sp, statd, seed=202)
        bo, bi = statb(sp["outdoor"]), statb(sp["indoor"])
        say(f"| {a} | {fmt(pt['outdoor'],3)} [{fmt(ci['outdoor'][0],3)}, "
            f"{fmt(ci['outdoor'][1],3)}] | {bo:.3f} | {fmt(pt['indoor'],3)} "
            f"[{fmt(ci['indoor'][0],3)}, {fmt(ci['indoor'][1],3)}] | {bi:.3f} "
            f"| {fmt(pd_,3)} [{fmt(ci['diff'][0],3)}, {fmt(ci['diff'][1],3)}] |")

    # ---- H4b --------------------------------------------------------------
    say("\n## H4b. Rally level: favourite−underdog serve-rate gap slope per +10 mph\n")
    say("| arm | outdoor c [95% CI] | n_out | indoor c [95% CI] | n_in "
        "| OUTDOOR − INDOOR [95% CI] |")
    say("|---|---|---|---|---|---|")
    statc = lambda rows: (None if len(rows) < 50 else
                          (lambda c: None if c is None else c[1])(
                              ols(rows, "y", ["w"])))
    for a in "abcd":
        sp = split(gap_rows, arms[a])
        pt, pd_, ci = paired_boot(sp, statc, seed=303)
        say(f"| {a} | {fmt(pt['outdoor'])} [{fmt(ci['outdoor'][0])}, "
            f"{fmt(ci['outdoor'][1])}] | {len(sp['outdoor'])} | "
            f"{fmt(pt['indoor'])} [{fmt(ci['indoor'][0])}, {fmt(ci['indoor'][1])}] "
            f"| {len(sp['indoor'])} | {fmt(pd_)} [{fmt(ci['diff'][0])}, "
            f"{fmt(ci['diff'][1])}] |")

    # ---- binned favourite edge (the number the published verdict quoted) ---
    say("\n## H4c (context). Favourite obs−pred edge by match-hour wind bin\n")
    say("| arm | setting | 0–8 | 8–14 | 14–20 | 20+ |")
    say("|---|---|---|---|---|---|")
    # need p_fav/fav_won: recompute quickly from game rows' skill & y sign
    for a in "abcd":
        sp = split(game_rows, arms[a])
        for s in ("outdoor", "indoor"):
            cells = []
            for lo, hi in ((0, 8), (8, 14), (14, 20), (20, 99)):
                sub = [r for r in sp[s] if lo <= r["wind"] < hi]
                if len(sub) < 30:
                    cells.append("  —")
                    continue
                # favourite share edge: observed share of the favoured side
                # minus v2-expected share (share metric, not win metric)
                obs = sum((r["y"] if r["skill"] >= 0 else -r["y"]) for r in sub) / len(sub)
                pred = sum(abs(r["skill"]) for r in sub) / len(sub)
                cells.append(f"{obs-pred:+.4f} ({len(sub)})")
            say(f"| {a} | {s} | " + " | ".join(cells) + " |")

    p = ROOT / "model/weather_review/label_arms_rerun.md"
    p.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
