"""Does wind flatten the favorite's edge? — fully data-referenced test.

    python model/favorites_wind.py    # prints + writes model/favorites_wind.md

The nulls here are regression nulls, not simulations (house note
2026-07-28: sim nulls only where no within-data reference exists):
  * the skill x wind INTERACTION is its own null (zero = no compression),
  * INDOOR is the falsification arm (same regression, no wind exposure),
  * v2 ratings are used to MEASURE skill or LABEL the favorite — never to
    supply a model-expected value that observed data is graded against.

Regression 1 — game level (max power, games.csv scores):
    share_i − ½ = b·skill_i + d·(skill_i × w_i)
  per setting, where skill = sigmoid(eta) − ½ (v2 expected share, centered)
  and w = match-hour wind / 10 mph. Favorites hypothesis: d < 0 outdoors
  (each unit of skill buys fewer points as wind rises), d ≈ 0 indoors.
  b is the skill slope at 0 mph; b + 1.5·d is the slope at 15 mph.

  ORIENTATION (fixed 2026-08-09; the original fit carried an intercept a
  and a wind main effect c, and both had to go). A game and its mirror
  image are the same game: relabelling the sides flips y and skill but
  leaves w alone. So a and c are ODD under the relabel and are identically
  zero on any symmetric panel — but games.csv is NOT symmetric. t1 wins
  67.8% of rows because PPA orders the match winner first, which is
  selection ON THE OUTCOME, so t1's residual is positive by construction
  (+0.017 mean share). Fitting a and c let that leak into d: the INDOOR
  estimate was −0.080 with them and is −0.031 without. Dropping them
  reproduces the symmetrised-panel estimate exactly, while keeping one row
  per game so the cluster bootstrap CIs stay honest (duplicating each game
  in both orientations would halve the apparent variance).
  Regression 2 below was always safe — it is oriented by which side v2
  calls the favorite, which is a function of the prediction, not the label.

Regression 2 — rally level, favorite breakdown (decider_serve_splits):
    gap_i = favorite side's serve-rally win rate − underdog's, full game
    gap_i = a + c·w_i        (per setting; v2 only picks WHO the favorite is)
  Compression → c < 0 outdoors. Complementary: serve-rate gap isolates
  rally-level dominance; sample is deciders + all MLP games (the games
  with committed serve splits), so levels are close-match-selected — the
  wind SLOPE is the object, not the level.

Cluster bootstrap by event throughout.
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


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ols(rows, ykey, xkeys, intercept=True):
    """OLS via normal equations (few regressors).

    intercept=False is required for the game-level fit — see the ORIENTATION
    note in the module docstring.  Coefficients are returned with the
    intercept first when it is fitted, so callers index accordingly.
    """
    p = len(xkeys) + (1 if intercept else 0)
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for r in rows:
        x = ([1.0] if intercept else []) + [r[k] for k in xkeys]
        y = r[ykey]
        for i in range(p):
            xty[i] += x[i] * y
            for j in range(p):
                xtx[i][j] += x[i] * x[j]
    # gaussian elimination
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


def boot_coefs(clustered, ykey, xkeys, n=2000, seed=17, intercept=True):
    keys = list(clustered)
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        s = []
        for _ in keys:
            s.extend(clustered[rng.choice(keys)])
        b = ols(s, ykey, xkeys, intercept)
        if b:
            draws.append(b)
    cis = []
    for i in range(len(draws[0])):
        v = sorted(d[i] for d in draws)
        cis.append((v[int(0.025 * len(v))], v[int(0.975 * len(v))]))
    return cis


def load_wind():
    hourly, start_hour = {}, {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        try:
            hourly[(r["event_id"], r["local_time"][:13])] = \
                float(r["windspeed_10m"])
        except (TypeError, ValueError):
            pass
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]
    return hourly, start_hour


def main():
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}
    hourly, start_hour = load_wind()
    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# Does wind flatten the favorite's edge? (data-referenced nulls)\n")

    # ---------------- Regression 1: game level ---------------------------
    rows_by = defaultdict(list)
    match_meta = {}
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        setting = geo.get(g["event_id"])
        if setting not in ("outdoor", "indoor"):
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
        skill = sigmoid(eta) - 0.5
        w = wind / 10.0
        rows_by[setting].append({"ev": g["event_id"],
                                 "y": s1 / (s1 + s2) - 0.5,
                                 "skill": skill, "w": w,
                                 "sw": skill * w})
        match_meta[g["match_id"]] = (setting, wind, eta, g["event_id"])

    say("## 1. Game level: share − ½ = b·skill + d·skill·(wind/10)\n")
    say("skill = v2-expected share − ½. d is the test: negative = wind "
        "compresses the favorite's conversion of skill into points. "
        "b at 0 mph vs b + 1.5d at 15 mph shows the size.\n")
    say("No intercept and no wind main effect: both are odd under a "
        "side relabel and so are zero for a symmetric panel, but games.csv "
        "is not symmetric (t1 wins 67.8% — PPA orders the match winner "
        "first, which is selection on the outcome). Fitting them let that "
        "ordering leak into d; the indoor estimate was −0.080 with them "
        "and is −0.031 without. Corrected 2026-08-09.\n")
    say("| setting | games | b (skill slope) | d (skill×wind) [95% CI] "
        "| slope at 15 mph |")
    say("|---|---|---|---|---|")
    for setting in ("outdoor", "indoor"):
        rows = rows_by[setting]
        # No intercept and no wind MAIN effect: both are odd under the
        # side relabel and so are identically zero for a symmetric panel.
        # Fitting them lets the games.csv t1 ordering leak into d.
        coefs = ols(rows, "y", ["skill", "sw"], intercept=False)
        clustered = defaultdict(list)
        for r in rows:
            clustered[r["ev"]].append(r)
        cis = boot_coefs(clustered, "y", ["skill", "sw"], intercept=False)
        b, d = coefs[0], coefs[1]
        dlo, dhi = cis[1]
        say(f"| {setting} | {len(rows)} | {b:.3f} | {d:+.3f} "
            f"[{dlo:+.3f}, {dhi:+.3f}] | {b + 1.5*d:.3f} |")
    say("")

    # ---------------- Regression 2: rally level, favorite gap ------------
    say("## 2. Rally level: favorite−underdog SERVE-RALLY win-rate gap "
        "vs wind\n")
    say("v2 only labels which side is the favorite (|eta| ≥ 0.1 required); "
        "gap uses full-game serve tallies from decider_serve_splits "
        "(deciders + all MLP games — close-match-selected, so read the "
        "wind slope, not the level).\n")
    gap_rows = defaultdict(list)
    for r in read_csv(ROOT / "data/decider_serve_splits.csv"):
        meta = match_meta.get(r["match_id"])
        if not meta:
            continue
        setting, wind, eta, ev = meta
        if abs(eta) < 0.1:
            continue
        ra = int(r["ra_pre"]) + int(r["ra_post"])
        wa = int(r["wa_pre"]) + int(r["wa_post"])
        rb = int(r["rb_pre"]) + int(r["rb_post"])
        wb = int(r["wb_pre"]) + int(r["wb_post"])
        if ra < 8 or rb < 8:
            continue
        rate_a, rate_b = wa / ra, wb / rb
        gap = (rate_a - rate_b) if eta > 0 else (rate_b - rate_a)
        gap_rows[setting].append({"ev": ev, "y": gap, "w": wind / 10.0})

    say("| setting | games | mean gap | wind slope c per +10 mph [95% CI] |")
    say("|---|---|---|---|")
    for setting in ("outdoor", "indoor"):
        rows = gap_rows[setting]
        coefs = ols(rows, "y", ["w"])
        clustered = defaultdict(list)
        for r in rows:
            clustered[r["ev"]].append(r)
        cis = boot_coefs(clustered, "y", ["w"])
        mean_gap = sum(r["y"] for r in rows) / len(rows)
        say(f"| {setting} | {len(rows)} | {mean_gap:+.3f} "
            f"| {coefs[1]:+.4f} [{cis[1][0]:+.4f}, {cis[1][1]:+.4f}] |")

    # -------- Regression 3: rally-level binomial logit -------------------
    say("\n## 3. Rally level proper: P(server wins THIS rally) — binomial "
        "logit\n")
    say("Each serve rally is a 0/1; with covariates constant within a "
        "match-side the Bernoulli series collapses losslessly to its "
        "(wins, attempts) sufficient statistic, so this fits the rally "
        "series exactly while weighting every rally once (fixing the "
        "equal-weight-per-game approximation of regression 2): "
        "logit p = a + b·adv + c·(wind/10) + d·adv·(wind/10), where adv = "
        "serving team's v2 eta advantage (signed). d < 0 = wind erodes "
        "the better team's rally edge. Cluster bootstrap by event.\n")

    def fit_logit(rows):
        """Newton-Raphson binomial logit; rows carry wins, n, covariates."""
        beta = [0.0, 0.0, 0.0, 0.0]
        for _ in range(25):
            grad = [0.0] * 4
            hess = [[0.0] * 4 for _ in range(4)]
            for r in rows:
                x = (1.0, r["adv"], r["w"], r["adv"] * r["w"])
                z = sum(b_ * x_ for b_, x_ in zip(beta, x))
                p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
                res = r["wins"] - r["n"] * p
                wgt = r["n"] * p * (1 - p)
                for i in range(4):
                    grad[i] += res * x[i]
                    for j in range(4):
                        hess[i][j] += wgt * x[i] * x[j]
            m = [hess[i][:] + [grad[i]] for i in range(4)]
            for col in range(4):
                piv = max(range(col, 4), key=lambda r_: abs(m[r_][col]))
                m[col], m[piv] = m[piv], m[col]
                if abs(m[col][col]) < 1e-10:
                    return None
                for r_ in range(4):
                    if r_ != col:
                        f = m[r_][col] / m[col][col]
                        for c_ in range(col, 5):
                            m[r_][c_] -= f * m[col][c_]
            step = [m[i][4] / m[i][i] for i in range(4)]
            beta = [b_ + s_ for b_, s_ in zip(beta, step)]
            if max(abs(s_) for s_ in step) < 1e-9:
                break
        return beta

    rally_rows = defaultdict(list)
    for r in read_csv(ROOT / "data/decider_serve_splits.csv"):
        meta = match_meta.get(r["match_id"])
        if not meta:
            continue
        setting, wind, eta, ev = meta
        for side, sgn in (("a", 1.0), ("b", -1.0)):
            n = int(r[f"r{side}_pre"]) + int(r[f"r{side}_post"])
            wins = int(r[f"w{side}_pre"]) + int(r[f"w{side}_post"])
            if n < 4:
                continue
            rally_rows[setting].append({"ev": ev, "n": n, "wins": wins,
                                        "adv": sgn * eta,
                                        "w": wind / 10.0})

    say("| setting | rallies | b (adv) | d (adv×wind) [95% CI] |")
    say("|---|---|---|---|")
    rng3 = random.Random(23)
    for setting in ("outdoor", "indoor"):
        rows = rally_rows[setting]
        beta = fit_logit(rows)
        clustered = defaultdict(list)
        for r in rows:
            clustered[r["ev"]].append(r)
        keys = list(clustered)
        draws = []
        for _ in range(600):
            s = []
            for _ in keys:
                s.extend(clustered[rng3.choice(keys)])
            bb = fit_logit(s)
            if bb:
                draws.append(bb[3])
        draws.sort()
        n_rallies = sum(r["n"] for r in rows)
        say(f"| {setting} | {n_rallies} | {beta[1]:.3f} | {beta[3]:+.3f} "
            f"[{draws[int(0.025*len(draws))]:+.3f}, "
            f"{draws[int(0.975*len(draws))]:+.3f}] |")

    say("\n---\n*All nulls are within-data: the interaction's zero, the "
        "indoor arm, and calm games. No simulation used. Caveats: "
        "current-form v2 retroactive (fine for interactions); outdoor "
        "labels heuristic; regressions 2–3 use the close-match-selected "
        "serve-splits sample (deciders + MLP) — slopes/interactions are "
        "the objects, not levels.*")

    (ROOT / "model/favorites_wind.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/favorites_wind.md")


if __name__ == "__main__":
    main()
