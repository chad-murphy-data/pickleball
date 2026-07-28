"""Court-END (good side / bad side) effects WITHOUT knowing the ends.

    python model/end_effects.py       # prints + writes model/end_effects.md

Referee logs never record which physical end a team occupies — but the
rules move teams across ends on a known schedule, so end effects leave a
statistical fingerprint:

  Why not a simple difference of means? Because nobody records which end
  is the good one: per game the observable swing (performance on one end
  minus the other) is SIGN-SYMMETRIC — team A's good end is team B's bad
  end and the assignment is effectively a coin flip — so its mean is zero
  with or without end effects. The paired difference is still the right
  object (the skill term cancels exactly, which correlation never
  manages); the information just lives in its VARIANCE: an end effect
  makes the swing bigger than sampling noise, and wind should make the
  excess grow.

  Design A — consecutive games (PPA doubles, same four players).
    Teams switch ends between games. Write team 1's margin in game g as
        m_g = s + (−1)^(g+1) e + noise.
    Paired difference d = m1 − m2 = 2e + noise: s cancels. Var(d) =
    4 Var(e) + 2 Var(noise), so the windy-vs-calm contrast
        [Var(d|windy) − Var(d|calm)] / 4 = Var(e|windy) − Var(e|calm)
    (game noise assumed weather-invariant — the serve-rate cut backs
    this). Margins are residualized on the v2-predicted margin first
    (affects nothing in d, kept for the secondary correlation table).

  Design B — the mid-game switch in deciders (game 3 of a PPA best-of-3).
    Teams switch ends when the first team reaches 6. Per decider, the
    swing d = (point share pre-switch) − (point share post-switch) has
    E[d] = 0 by the symmetry above; excess of d² over the binomial noise
    p(1−p)(1/n_pre + 1/n_post) estimates 4 Var(e) in share² units. The
    LEVEL of the excess is inflated by serve-streak clustering (rallies
    are not iid), so again read the windy-vs-calm CONTRAST. Data:
    data/decider_splits.csv, aggregated from the Supabase pb_rally table
    (points by side, before/after the score first reaches 6).

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
from sitelib.winprob import K_DOUBLES, serve_probs  # noqa: E402

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


def sim_game(kA: float, kB: float, rng: random.Random, T: int = 11):
    """One side-out game under the winprob.py rally model — serve-state
    Markov clustering only, NO momentum, NO end effects. Returns
    (pre-switch points [A,B], post-switch points, final a, final b) with
    the switch defined as in the decider design (rally-start max < 6)."""
    a = b = 0
    team = 0 if rng.random() < 0.5 else 1     # who serves first
    num = 2                                    # first-server exception
    pre, post = [0, 0], [0, 0]
    while True:
        period = pre if max(a, b) < 6 else post
        k = kA if team == 0 else kB
        if rng.random() < k:
            period[team] += 1
            if team == 0:
                a += 1
            else:
                b += 1
            if (a >= T or b >= T) and abs(a - b) >= 2:
                return pre, post, a, b
        elif num == 1:
            num = 2
        else:
            team, num = 1 - team, 1


def zsq(x_pts: int, n1: int, y_pts: int, n2: int):
    """(swing², binomial noise, z²) for share x_pts/n1 vs y_pts/n2."""
    x, y = x_pts / n1, y_pts / n2
    p_hat = (x_pts + y_pts) / (n1 + n2)
    noise = p_hat * (1 - p_hat) * (1 / n1 + 1 / n2)
    sq = (x - y) ** 2
    return sq, noise, (sq / noise if noise > 0 else 0.0)


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
        vals = [v2.get(g0[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        eta = team_eta(*vals) if all(v is not None for v in vals) else None
        matches[mid] = {"setting": setting, "wind": wind, "eta": eta,
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
    hourly_mode = (ROOT / "data/match_times.csv").exists()
    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# End effects (good side / bad side) — inferred from switch structure\n")
    say("Wind source: **{}**.\n".format(
        "at the match start hour (data/match_times.csv)" if hourly_mode
        else "daily max (run scraper/extract_match_times.py for hour-level)"))
    say("End effects push the paired-swing VARIANCE up (primary tables) and "
        "the correlations down (secondary). Levels are contaminated — "
        "Design A by game noise, Design B by serve-streak clustering — so "
        "read CONTRASTS between rows; the contaminants have no reason to "
        "vary with wind.\n")

    # ---------------- Design A: paired swing across the between-game switch
    say("## Design A — game 1 vs game 2 (same 4 players, ends switched "
        "between games)\n")
    rows = defaultdict(list)     # group -> [(cluster,(r1,r2))] margins, rated
    rows_a2 = defaultdict(list)  # group -> [(cluster, rec)] share swing, all
    for mid, m in matches.items():
        gs = m["games"]
        if len(gs) < 2 or gs[0]["game_number"] != "1" or gs[1]["game_number"] != "2":
            continue
        g1, g2 = gs[0], gs[1]
        if g1["scoring_format"] != "sideout_11":
            continue
        grp = group_of(m)
        if not grp:
            continue
        s11, s12 = int(g1["t1_score"]), int(g1["t2_score"])
        s21, s22 = int(g2["t1_score"]), int(g2["t2_score"])
        if s11 + s12 >= 11 and s21 + s22 >= 11:
            sq, noise, z2 = zsq(s11, s11 + s12, s21, s21 + s22)
            rows_a2[grp].append((m["event"],
                                 {"sq": sq, "noise": noise, "z2": z2,
                                  "eta": m["eta"] or 0.0}))
        if m["eta"] is None:
            continue
        exp = race_dist(round(sigmoid(m["eta"]), 4), 11)["exp_margin"]
        rows[grp].append((m["event"], (s11 - s12 - exp, s21 - s22 - exp)))

    def var_d(pairs):
        ds = [x - y for x, y in pairs]
        m = sum(ds) / len(ds)
        return sum((d - m) ** 2 for d in ds) / (len(ds) - 1)

    say("Primary: variance of the paired swing d = margin(g1) − margin(g2) "
        "(skill cancels; Var(d) = 4·Var(end adv) + 2·Var(game noise)).\n")
    say("| group | matches | Var(d) pts² [95% CI] | vs calm: implied "
        "end-adv sd (pts/game) |")
    say("|---|---|---|---|")
    calm_v = None
    stats_a = {}
    for grp in sorted(rows):
        data = rows[grp]
        pairs = [p for _, p in data]
        clustered = defaultdict(list)
        for ev, p in data:
            clustered[ev].append(p)
        v = var_d(pairs)
        lo, hi = boot(clustered, var_d)
        stats_a[grp] = (v, lo, hi)
        if "calm" in grp:
            calm_v = v
    for grp in sorted(rows):
        v, lo, hi = stats_a[grp]
        if calm_v is None or "calm" in grp:
            implied = "— (baseline)" if "calm" in grp else "—"
        else:
            dv = (v - calm_v) / 4
            implied = (f"+{math.sqrt(dv):.2f}" if dv > 0
                       else f"≤0 (Var Δ {dv*4:+.2f})")
        say(f"| {grp} | {len(rows[grp])} | {v:.2f} [{lo:.2f}, {hi:.2f}] "
            f"| {implied} |")

    say("\nSecondary (older view — levels not interpretable, contrasts only):")
    say("\n| group | corr(r1, r2) [95% CI] | cov (pts²) |")
    say("|---|---|---|")
    for grp in sorted(rows):
        data = rows[grp]
        pairs = [p for _, p in data]
        clustered = defaultdict(list)
        for ev, p in data:
            clustered[ev].append(p)
        c = corr(pairs)
        lo, hi = boot(clustered, corr)
        say(f"| {grp} | {c:+.3f} [{lo:+.3f}, {hi:+.3f}] | {cov(pairs):+.2f} |")
    say("")

    # ------- Design A2: between-games share swing (all matches, z²) ------
    say("## Design A2 — game-1 vs game-2 POINT SHARES (share/z² framework, "
        "no ratings needed)\n")
    say("Same object as Design B but across the between-game end switch: "
        "swing = team 1's share of points in game 1 minus game 2 "
        "(scores ARE the point counts). 'null z²' = the same statistic in "
        "games simulated from the winprob.py serve-state model (k = "
        f"{K_DOUBLES}, match etas, NO momentum, NO end effects) — the "
        "mechanical serve-clustering floor.\n")
    rngs = random.Random(20260728)
    say("| group | matches | mean z² [95% CI] | null z² (sim) | "
        "mean excess ×10³ [95% CI] |")
    say("|---|---|---|---|---|")
    REPS_A = 12
    for grp in sorted(rows_a2):
        data = rows_a2[grp]
        recs = [d for _, d in data]
        clustered = defaultdict(list)
        for ev, d in data:
            clustered[ev].append(d)
        mean_z2 = lambda s: sum(d["z2"] for d in s) / len(s)
        mean_ex = lambda s: sum(d["sq"] - d["noise"] for d in s) / len(s)
        z2 = mean_z2(recs)
        zlo, zhi = boot(clustered, mean_z2)
        me = mean_ex(recs)
        elo, ehi = boot(clustered, mean_ex)
        null_sum = null_n = 0.0
        for d in recs:
            kA, kB = serve_probs(d["eta"])
            for _ in range(REPS_A):
                _, _, a1, b1 = sim_game(kA, kB, rngs)
                _, _, a2, b2 = sim_game(kA, kB, rngs)
                _, _, z = zsq(a1, a1 + b1, a2, a2 + b2)
                null_sum += z
                null_n += 1
        say(f"| {grp} | {len(recs)} | {z2:.2f} [{zlo:.2f}, {zhi:.2f}] "
            f"| {null_sum/null_n:.2f} "
            f"| {me*1000:+.2f} [{elo*1000:+.2f}, {ehi*1000:+.2f}] |")
    say("")

    # ---------------- Design B: decider pre/post switch -------------------
    say("## Design B — decider game 3: point share before vs after the "
        "mid-game end switch at 6\n")
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
        x = int(r["pa_pre"]) / pre
        y = int(r["pa_post"]) / post
        p_hat = (int(r["pa_pre"]) + int(r["pa_post"])) / (pre + post)
        noise = p_hat * (1 - p_hat) * (1 / pre + 1 / post)
        excess = (x - y) ** 2 - noise
        grp = group_of(m)
        if grp:
            rows_b[grp].append((m["event"],
                                {"x": x - 0.5, "y": y - 0.5,
                                 "sq": (x - y) ** 2, "noise": noise,
                                 "excess": excess, "eta": m["eta"] or 0.0}))

    say("Primary: the swing = TEAM A's point share on its first end minus "
        "its share on its second end (the 6-0-then-5-7 comparison; team B "
        "is the mirror image, so side A alone carries all the "
        "information). Its mean is 0 by end-assignment symmetry, so the "
        "tests are (i) mean of swing² − binomial noise (= 4·Var(end adv) "
        "in share²) and (ii) mean z², each game standardized by its own "
        "sampling noise — 1.00 under the null, so games with decisive "
        "halves count for more. LEVELS are inflated by serve-streak "
        "clustering; read contrasts.\n")
    say("| group | deciders | RMS swing | noise RMS | mean excess ×10³ "
        "[95% CI] | mean z² [95% CI] | null z² (sim) |")
    say("|---|---|---|---|---|---|---|")
    rngs_b = random.Random(20260729)
    REPS_B = 40
    for grp in sorted(rows_b):
        data = rows_b[grp]
        recs = [d for _, d in data]
        clustered = defaultdict(list)
        for ev, d in data:
            clustered[ev].append(d)
        mean_excess = lambda s: sum(d["excess"] for d in s) / len(s)
        # noise == 0 only when one team took every point of the game
        # (x == y exactly), which carries no swing information — z² := 0
        mean_z2 = lambda s: sum(
            d["sq"] / d["noise"] if d["noise"] > 0 else 0.0
            for d in s) / len(s)
        me = mean_excess(recs)
        lo, hi = boot(clustered, mean_excess)
        z2 = mean_z2(recs)
        zlo, zhi = boot(clustered, mean_z2)
        rms = math.sqrt(sum(d["sq"] for d in recs) / len(recs))
        nrms = math.sqrt(sum(d["noise"] for d in recs) / len(recs))
        null_sum = null_n = 0.0
        for d in recs:
            kA, kB = serve_probs(d["eta"])
            for _ in range(REPS_B):
                pre, post, _, _ = sim_game(kA, kB, rngs_b)
                n1, n2 = sum(pre), sum(post)
                if n1 < 5 or n2 < 5:
                    continue
                _, noise, z = zsq(pre[0], n1, post[0], n2)
                if noise > 0:
                    null_sum += z
                null_n += 1
        say(f"| {grp} | {len(recs)} | {rms:.3f} | {nrms:.3f} "
            f"| {me*1000:+.2f} [{lo*1000:+.2f}, {hi*1000:+.2f}] "
            f"| {z2:.2f} [{zlo:.2f}, {zhi:.2f}] | {null_sum/null_n:.2f} |")

    say("\nSecondary (older correlation view):\n")
    say("| group | corr(pre, post) [95% CI] |")
    say("|---|---|")
    for grp in sorted(rows_b):
        data = rows_b[grp]
        pairs = [(d["x"], d["y"]) for _, d in data]
        clustered = defaultdict(list)
        for ev, d in data:
            clustered[ev].append((d["x"], d["y"]))
        c = corr(pairs)
        lo, hi = boot(clustered, corr)
        say(f"| {grp} | {c:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    say("")

    say("---\n*Caveats: indoor/outdoor labels heuristic; Design A assumes "
        "match-level form variance is similar across groups."
        + ("" if hourly_mode else " Day-level wind attenuates the "
           "windy-vs-calm contrast — run scraper/extract_match_times.py "
           "for the hour-level version.") + "*")

    (ROOT / "model/end_effects.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/end_effects.md")


if __name__ == "__main__":
    main()
