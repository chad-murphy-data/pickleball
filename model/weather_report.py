"""Weather vs outcomes — first pass, DAY-level join (2026-07-28).

    python model/weather_report.py        # prints + writes model/weather_report.md

Joins games.csv to data/event_weather.csv (daily, from scraper/weather.py)
by (event_id, date). Three cuts, each run outdoor vs indoor so the indoor
arm serves as a placebo (a "wind effect" indoors would mean the labels or
the method are broken):

  A. Serve-point rate (n_points / n_rallies from data/match_rally_summary.csv)
     vs daily max wind. The physics question: does wind shorten rallies /
     change the serve-vs-return balance?
  B. Do v2 favorites underperform in wind? Observed favorite win rate vs
     the race-DP predicted rate, plus Brier, by wind bin.
  C. Same by daily max temperature (heat).

Honest caveats, stated up front:
  - DAY grain: a 25 mph daily max may have been calm at match time. This
    attenuates real effects toward zero; hourly join is the follow-up
    (data/event_weather_hourly.csv is already on disk waiting for
    per-match start times from raw/ localDateMatch*).
  - v2 values are CURRENT form applied retroactively (lookahead). Fine for
    a weather *interaction* cut (form drift shouldn't correlate with wind),
    not a headline accuracy number.
  - indoor/outdoor labels are tour-default + venue keywords
    (data/event_geo.csv setting column); a few Life Time stops do use
    outdoor courts. Curate data/venue_overrides.csv as broadcasts confirm.
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import game_win_prob, team_eta  # noqa: E402

WIND_BINS = [(0, 8), (8, 14), (14, 20), (20, 99)]
TEMP_BINS = [(0, 70), (70, 82), (82, 92), (92, 150)]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def bin_label(v, bins, unit):
    for lo, hi in bins:
        if lo <= v < hi:
            return f"{lo}–{hi if hi < 99 else '+'} {unit}" if hi < 99 else f"{lo}+ {unit}"
    return None


def boot_ci(pairs, stat, n=2000, seed=7):
    """Cluster bootstrap over event-days. pairs = {cluster: [rows]}."""
    keys = list(pairs)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = []
        for _ in keys:
            sample.extend(pairs[rng.choice(keys)])
        vals.append(stat(sample))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def wls_slope(rows, xkey, ykey, wkey):
    """Weighted least-squares slope of y on x."""
    sw = sx = sy = sxx = sxy = 0.0
    for r in rows:
        w, x, y = r[wkey], r[xkey], r[ykey]
        sw += w; sx += w * x; sy += w * y
        sxx += w * x * x; sxy += w * x * y
    den = sxx - sx * sx / sw
    return (sxy - sx * sy / sw) / den if den else 0.0


def main():
    geo = {r["event_id"]: r for r in read_csv(ROOT / "data/event_geo.csv")}
    wx = {(r["event_id"], r["date"]): r
          for r in read_csv(ROOT / "data/event_weather.csv")}
    v2 = {r["player_id"]: (float(r["value_now_mean"]), float(r["value_now_sd"]))
          for r in read_csv(ROOT / "data/v2_players.csv")}
    rally = {r["match_id"]: r
             for r in read_csv(ROOT / "data/match_rally_summary.csv")
             if r["discipline"] == "doubles" and int(r["n_rallies"]) >= 20}

    # ---- assemble game-level analysis rows -------------------------------
    games, matches_seen = [], set()
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        w = wx.get((g["event_id"], g["date"]))
        ge = geo.get(g["event_id"])
        if not w or not ge:
            continue
        wind, tmax = fnum(w["windspeed_10m_max"]), fnum(w["temperature_2m_max"])
        if wind is None or tmax is None:
            continue
        row = {"setting": ge["setting"], "tour": g["tour"],
               "cluster": g["event_id"] + g["date"],
               "wind": wind, "gust": fnum(w["windgusts_10m_max"]),
               "tmax": tmax, "precip": fnum(w["precipitation_sum"])}

        # serve-point rate: one row per MATCH (not per game)
        if g["match_id"] not in matches_seen and g["match_id"] in rally:
            matches_seen.add(g["match_id"])
            rs = rally[g["match_id"]]
            row_m = dict(row)
            row_m["n_rallies"] = int(rs["n_rallies"])
            row_m["serve_rate"] = int(rs["n_points"]) / int(rs["n_rallies"])
            games.append(("match", row_m))

        # favorite check: needs all 4 players rated
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if all(vals):
            eta = team_eta(vals[0][0], vals[1][0], vals[2][0], vals[3][0])
            T = int(g["scoring_format"].rsplit("_", 1)[1])
            if T < 11:
                continue
            p1 = game_win_prob(eta, T)
            p_fav = max(p1, 1 - p1)
            fav_won = (int(g["t1_score"]) > int(g["t2_score"])) == (p1 >= 0.5)
            row_g = dict(row)
            row_g.update(p_fav=p_fav, fav_won=1.0 if fav_won else 0.0,
                         brier=(p_fav - (1.0 if fav_won else 0.0)) ** 2)
            games.append(("game", row_g))

    out = []
    say = lambda s="": (print(s), out.append(s))

    n_m = sum(1 for k, _ in games if k == "match")
    n_g = sum(1 for k, _ in games if k == "game")
    say("# Weather report — first pass (day-level join)\n")
    say(f"Joined rows: {n_m} matches with rally logs, {n_g} games with "
        f"full v2 ratings.\n")

    for setting in ("outdoor", "indoor"):
        say(f"\n## {setting.upper()}"
            + ("  *(placebo arm — nothing should move)*" if setting == "indoor" else ""))

        # ---- A: serve rate vs wind ---------------------------------------
        rows = [r for k, r in games if k == "match" and r["setting"] == setting]
        say(f"\n### A. Serve-point rate vs daily max wind ({len(rows)} matches)")
        say("| wind | matches | rallies | serve-point rate |")
        say("|---|---|---|---|")
        for lo, hi in WIND_BINS:
            sub = [r for r in rows if lo <= r["wind"] < hi]
            nr = sum(r["n_rallies"] for r in sub)
            if not nr:
                continue
            rate = sum(r["serve_rate"] * r["n_rallies"] for r in sub) / nr
            lbl = f"{lo}–{hi}" if hi < 99 else f"{lo}+"
            say(f"| {lbl} mph | {len(sub)} | {nr} | {rate:.4f} |")
        if rows:
            for r in rows:
                r["w"] = r["n_rallies"]
            slope = wls_slope(rows, "wind", "serve_rate", "w")
            clusters = defaultdict(list)
            for r in rows:
                clusters[r["cluster"]].append(r)
            lo_ci, hi_ci = boot_ci(clusters,
                                   lambda s: wls_slope(s, "wind", "serve_rate", "w"))
            say(f"\nWLS slope: {slope*1000:+.3f} pp serve-rate per 1000×mph "
                f"→ per +10 mph: {slope*10:+.4f} "
                f"(95% cluster-bootstrap CI [{lo_ci*10:+.4f}, {hi_ci*10:+.4f}])")

        # ---- B/C: favorites vs wind and heat -----------------------------
        rows = [r for k, r in games if k == "game" and r["setting"] == setting]
        for name, key, bins, unit in (("B. Favorites vs wind", "wind", WIND_BINS, "mph"),
                                      ("C. Favorites vs heat", "tmax", TEMP_BINS, "°F")):
            say(f"\n### {name} ({len(rows)} games)")
            say(f"| {key} | games | predicted fav % | observed fav % | edge (obs−pred) | Brier |")
            say("|---|---|---|---|---|---|")
            for lo, hi in bins:
                sub = [r for r in rows if lo <= r[key] < hi]
                if len(sub) < 30:
                    continue
                pred = sum(r["p_fav"] for r in sub) / len(sub)
                obs = sum(r["fav_won"] for r in sub) / len(sub)
                br = sum(r["brier"] for r in sub) / len(sub)
                clusters = defaultdict(list)
                for r in sub:
                    clusters[r["cluster"]].append(r)
                lo_ci, hi_ci = boot_ci(
                    clusters,
                    lambda s: (sum(r["fav_won"] for r in s) / len(s)
                               - sum(r["p_fav"] for r in s) / len(s)))
                lbl = f"{lo}–{hi}" if hi < 99 and hi < 150 else f"{lo}+"
                say(f"| {lbl} {unit} | {len(sub)} | {pred:.3f} | {obs:.3f} "
                    f"| {obs-pred:+.3f} [{lo_ci:+.3f},{hi_ci:+.3f}] | {br:.4f} |")

    say("\n---\n*Caveats: day-level weather (attenuates), v2 current-form "
        "values applied retroactively (fine for interactions, not levels), "
        "indoor/outdoor labels are heuristic (see scraper/weather.py). "
        "Hourly join is the designed next step.*")

    (ROOT / "model/weather_report.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/weather_report.md")


if __name__ == "__main__":
    main()
