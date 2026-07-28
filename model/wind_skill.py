"""Is there a WIND SKILL dimension? (the F1-rain-driver hypothesis)

    python model/wind_skill.py       # prints + writes model/wind_skill.md

Distinct from the mean effect (wind compressing everyone's edges, see
weather_report.py): here we ask whether SPECIFIC players systematically
outperform their expectation in wind — a stable, orthogonal skill, like
rain driving in F1.

Method (mirrors the clutch/durability split-half design from the spec
shootout):
  1. Per outdoor game with known match-hour wind and full v2 ratings:
     residual = actual point share − sigmoid(eta), signed per player
     (+residual for team-1 players, −residual for team-2).
  2. Per player with enough such games: OLS slope of residual on wind =
     that player's "wind slope."
  3. Existence test: split each player's games into halves
     (chronologically alternating), estimate the slope in each half
     independently, correlate slope₁ vs slope₂ across players (weighted
     by games). A real dimension → same players wind-positive in both
     halves → r > 0. Null band from permuting wind within player
     (breaks the wind link, preserves everything else).

Power note, stated honestly: per-game share residuals are noisy
(sd ≈ 0.11) and match-hour wind has modest spread, so per-player slope
estimates carry large errors — this test can detect a moderate-or-larger
dimension, not a whisper.
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
from sitelib.race import sigmoid, team_eta  # noqa: E402

MIN_GAMES = 40      # per player, outdoor with known match-hour wind
N_PERMS = 20
N_SPLITS = 20


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ols_slope(pts):
    n = len(pts)
    if n < 10:
        return None
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den < 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / den


def wcorr(pairs):
    """Weighted Pearson corr of (x, y, w) triples."""
    sw = sum(w for _, _, w in pairs)
    mx = sum(x * w for x, _, w in pairs) / sw
    my = sum(y * w for _, y, w in pairs) / sw
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in pairs)
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pairs)
    syy = sum(w * (y - my) ** 2 for _, y, w in pairs)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def split_half_r(players, rng, permute=False):
    """One split: r of half-slopes across players. players =
    {pid: [(wind, resid), ...] chronological}."""
    pairs = []
    for pid, games in players.items():
        g = list(games)
        if permute:
            winds = [x for x, _ in g]
            rng.shuffle(winds)
            g = [(w, y) for w, (_, y) in zip(winds, g)]
        idx = list(range(len(g)))
        rng.shuffle(idx)
        h1 = [g[i] for i in idx[::2]]
        h2 = [g[i] for i in idx[1::2]]
        s1, s2 = ols_slope(h1), ols_slope(h2)
        if s1 is not None and s2 is not None:
            pairs.append((s1, s2, len(g)))
    return wcorr(pairs), len(pairs)


def main():
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}
    names = {r["player_id"]: r["full_name"]
             for r in read_csv(ROOT / "data/v2_players.csv")}
    hourly = {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        try:
            hourly[(r["event_id"], r["local_time"][:13])] = \
                float(r["windspeed_10m"])
        except (TypeError, ValueError):
            pass
    start_hour = {}
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]

    players = defaultdict(list)   # pid -> [(wind, signed residual)]
    n_games = 0
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        if geo.get(g["event_id"]) != "outdoor":
            continue
        wind = hourly.get((g["event_id"], start_hour.get(g["match_id"], "")))
        if wind is None:
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        eta = team_eta(*vals)
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        resid = s1 / (s1 + s2) - sigmoid(eta)
        n_games += 1
        for k in ("t1_p1", "t1_p2"):
            players[g[k]].append((wind, resid))
        for k in ("t2_p1", "t2_p2"):
            players[g[k]].append((wind, -resid))

    players = {p: games for p, games in players.items()
               if len(games) >= MIN_GAMES}
    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# Wind skill — is there an orthogonal dimension? (F1-rain test)\n")
    say(f"{n_games} outdoor games with match-hour wind + full ratings; "
        f"{len(players)} players with ≥{MIN_GAMES} such games.\n")

    rng = random.Random(42)
    obs = [split_half_r(players, rng)[0] for _ in range(N_SPLITS)]
    obs_mean = sum(obs) / len(obs)
    nulls = [split_half_r(players, rng, permute=True)[0]
             for _ in range(N_PERMS)]
    nulls_sorted = sorted(nulls)
    say(f"Split-half reliability of per-player wind slopes "
        f"(mean of {N_SPLITS} random splits): **r = {obs_mean:+.3f}**")
    say(f"Permutation null (wind shuffled within player, {N_PERMS} runs): "
        f"mean {sum(nulls)/len(nulls):+.3f}, "
        f"range [{nulls_sorted[0]:+.3f}, {nulls_sorted[-1]:+.3f}]\n")
    say("Reference points: clutch and durability came in at split-half "
        "r ≈ 0.15/0.13 (faint but real). A wind-skill dimension of "
        "similar or larger size should clear the permutation band.\n")

    # full-sample slopes, for the leaderboard (only meaningful if the
    # reliability test passes; printed regardless, labeled accordingly)
    slopes = []
    for pid, games in players.items():
        s = ols_slope(games)
        if s is not None:
            slopes.append((s, pid, len(games)))
    slopes.sort(reverse=True)
    verdict = "SUGGESTIVE — interpret names with care" \
        if obs_mean > max(nulls) else \
        "NOT ESTABLISHED — the list below is mostly noise, do not publish"
    say(f"Reliability verdict: {verdict}\n")
    say("| best in wind (per +10 mph, share) | n | worst in wind | n |")
    say("|---|---|---|---|")
    for i in range(5):
        t, b = slopes[i], slopes[-(i + 1)]
        say(f"| {names.get(t[1], t[1])} {t[0]*10:+.3f} | {t[2]} "
            f"| {names.get(b[1], b[1])} {b[0]*10:+.3f} | {b[2]} |")

    say("\n---\n*Residual = actual point share − v2-expected, so the mean "
        "wind-compression effect is shared; the test targets player-"
        "specific deviations. Current-form v2 values applied "
        "retroactively; outdoor labels heuristic; power limited — this "
        "detects a moderate dimension, not a whisper.*")

    (ROOT / "model/wind_skill.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/wind_skill.md")


if __name__ == "__main__":
    main()
