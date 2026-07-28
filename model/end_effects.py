"""Court-END (good side / bad side) effects WITHOUT knowing the ends.

    python model/end_effects.py       # prints + writes model/end_effects.md

Referee logs never record which physical end a team occupies — but the
rules move teams across ends on a known schedule, so end effects leave a
statistical fingerprint:

  Design A — consecutive games (PPA doubles, same four players).
    Teams switch ends between games. Write team 1's margin in game g as
        m_g = s + (−1)^(g+1) e + noise,
    s = match-level skill/form edge, e = end advantage held in game 1.
    Then cov(m1, m2) = Var(s) − Var(e): end effects DEPRESS the
    consecutive-game covariance. Var(s) is unknown, so the LEVEL of the
    correlation is not interpretable — but CONTRASTS are: comparing calm
    vs windy outdoor days (or outdoor vs indoor) differences out Var(s)
    and identifies the DIFFERENCE in end-effect variance:
        cov_calm − cov_windy = Var(e|windy) − Var(e|calm).
    Margins are residualized on the v2-predicted margin first, which
    removes most of Var(s) and sharpens the contrast.

  Design B — the mid-game switch in deciders (game 3 of a PPA best-of-3).
    Teams switch ends when the first team reaches 6. If ends matter, a
    team's point share before the switch anti-correlates with its share
    after (they surrender the good end mid-game). Skill continuity pushes
    the pre/post correlation positive; end effects push it negative; the
    same contrast logic (windy vs calm, outdoor vs indoor) isolates the
    end component. Data: data/decider_splits.csv, aggregated from the
    Supabase pb_rally table (points by side, before/after the score
    first reaches 6).

Per the house stance (2026-07-28): indoor is NOT assumed end-effect-free —
more controlled, not fully (drafts, lighting, backdrops). Indoor gets its
own estimate; it is a comparison arm, not a zero.

MLP is excluded from both designs: each game in a matchup is a different
discipline with different players, so consecutive games share no lineup.
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
from sitelib.race import race_dist, sigmoid, team_eta  # noqa: E402

WIND_GROUPS = [("calm <8 mph", 0, 8), ("moderate 8–14", 8, 14),
               ("windy 14+", 14, 99)]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def corr(pairs):
    n = len(pairs)
    if n < 3:
        return float("nan")
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def cov(pairs):
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    return sum((x - mx) * (y - my) for x, y in pairs) / (n - 1)


def boot(clustered, stat, n=4000, seed=11):
    keys = list(clustered)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = []
        for _ in keys:
            s.extend(clustered[rng.choice(keys)])
        v = stat(s)
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def load_context():
    """match_id -> (setting, wind, event_id) for PPA doubles matches.
    Wind = daily max, upgraded to wind AT THE MATCH START HOUR whenever
    scraper/extract_match_times.py has produced data/match_times.csv."""
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    wx = {(r["event_id"], r["date"]): r for r in read_csv(ROOT / "data/event_weather.csv")}
    hourly, start_hour = {}, {}
    if (ROOT / "data/match_times.csv").exists():
        for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
            try:
                hourly[(r["event_id"], r["local_time"][:13])] = \
                    float(r["windspeed_10m"])
            except (ValueError, TypeError):
                pass
        for r in read_csv(ROOT / "data/match_times.csv"):
            ts = r["start_local"] or r["planned_start_local"]
            if ts:
                start_hour[r["match_id"]] = ts[:13]
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}
    matches = {}
    by_match = defaultdict(list)
    for g in read_csv(ROOT / "data/games.csv"):
        if g["tour"] != "PPA" or g["is_dreambreaker"] == "True" \
                or g["is_forfeit"] == "True":
            continue
        by_match[g["match_id"]].append(g)
    for mid, gs in by_match.items():
        g0 = gs[0]
        w = wx.get((g0["event_id"], g0["date"]))
        setting = geo.get(g0["event_id"])
        if not w or not setting:
            continue
        try:
            wind = float(w["windspeed_10m_max"])
        except (ValueError, TypeError):
            continue
        hw = hourly.get((g0["event_id"], start_hour.get(mid, "")))
        if hw is not None:
            wind = hw
        matches[mid] = {"setting": setting, "wind": wind,
                        "event": g0["event_id"], "games": sorted(
                            gs, key=lambda r: int(r["game_number"])),
                        "best_of": int(g0["best_of"] or 0)}
    return matches, v2


def group_of(m):
    if m["setting"] == "indoor":
        return "INDOOR (all wind — no exposure expected)"
    for lbl, lo, hi in WIND_GROUPS:
        if lo <= m["wind"] < hi:
            return f"OUTDOOR {lbl}"
    return None


def main():
    matches, v2 = load_context()
    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# End effects (good side / bad side) — inferred from switch structure\n")
    say("Levels are NOT interpretable alone (they mix skill continuity with "
        "end effects); read the CONTRASTS between rows. End effects push "
        "every statistic DOWN.\n")

    # ---------------- Design A: consecutive-game residual correlation ----
    say("## Design A — corr of game-1 vs game-2 residual margins "
        "(same 4 players, ends switched)\n")
    rows = defaultdict(list)   # group -> [(cluster,(r1,r2))]
    for mid, m in matches.items():
        gs = m["games"]
        if len(gs) < 2 or gs[0]["game_number"] != "1" or gs[1]["game_number"] != "2":
            continue
        g1, g2 = gs[0], gs[1]
        if g1["scoring_format"] != "sideout_11":
            continue
        vals = [v2.get(g1[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        eta = team_eta(*vals)
        exp = race_dist(round(sigmoid(eta), 4), 11)["exp_margin"]
        r1 = int(g1["t1_score"]) - int(g1["t2_score"]) - exp
        r2 = int(g2["t1_score"]) - int(g2["t2_score"]) - exp
        grp = group_of(m)
        if grp:
            rows[grp].append((m["event"], (r1, r2)))

    say("| group | matches | corr(r1, r2) [95% CI] | cov (pts²) |")
    say("|---|---|---|---|")
    for grp in sorted(rows):
        data = rows[grp]
        pairs = [p for _, p in data]
        clustered = defaultdict(list)
        for ev, p in data:
            clustered[ev].append(p)
        c = corr(pairs)
        lo, hi = boot(clustered, corr)
        say(f"| {grp} | {len(pairs)} | {c:+.3f} [{lo:+.3f}, {hi:+.3f}] "
            f"| {cov(pairs):+.2f} |")
    say("\ncov(calm) − cov(windy) estimates Var(end adv | windy) − "
        "Var(end adv | calm) in points²; its square root is the typical "
        "per-game end advantage the wind adds.\n")

    # ---------------- Design B: decider pre/post switch -------------------
    say("## Design B — decider game 3: point share before vs after the "
        "end switch at 6\n")
    splits = read_csv(ROOT / "data/decider_splits.csv")
    rows_b = defaultdict(list)
    for r in splits:
        m = matches.get(r["match_id"])
        if not m:
            continue
        gn = int(r["game_number"])
        if not (m["best_of"] == 3 and gn == 3) and \
           not (m["best_of"] == 5 and gn == 5):
            continue
        pre = int(r["pa_pre"]) + int(r["pb_pre"])
        post = int(r["pa_post"]) + int(r["pb_post"])
        if pre < 5 or post < 5:
            continue
        x = int(r["pa_pre"]) / pre - 0.5
        y = int(r["pa_post"]) / post - 0.5
        grp = group_of(m)
        if grp:
            rows_b[grp].append((m["event"], (x, y)))

    say("| group | deciders | corr(pre, post) [95% CI] |")
    say("|---|---|---|")
    for grp in sorted(rows_b):
        data = rows_b[grp]
        pairs = [p for _, p in data]
        clustered = defaultdict(list)
        for ev, p in data:
            clustered[ev].append(p)
        c = corr(pairs)
        lo, hi = boot(clustered, corr)
        say(f"| {grp} | {len(pairs)} | {c:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    say("\nBinomial noise in the point shares attenuates all rows toward "
        "zero equally; again, read contrasts, not levels.\n")

    say("---\n*Caveats: day-level wind (attenuates the windy-vs-calm "
        "contrast); indoor/outdoor labels heuristic; Design A assumes "
        "match-level form variance is similar across groups. Hour-level "
        "wind (data/match_times.csv + event_weather_hourly.csv) is the "
        "designed upgrade.*")

    (ROOT / "model/end_effects.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/end_effects.md")


if __name__ == "__main__":
    main()
