"""Is WIND SKILL a rare trait? — the minority-aimed battery, per finding 10.

    python model/wind_rare.py     # prints + writes model/wind_rare.md

model/wind_skill.py asked the POPULATION question (split-half reliability,
max|z| scan) and found nothing. The clutch thread (finding 10) showed those
are the wrong instruments for a trait carried by a FEW players: 500 zero
players dilute 40 real ones, and split-half r on a mostly-zero field sits
near zero even when the minority is real. Clutch itself only showed up once
the minority-aimed tests ran: spike-and-slab LR, tail counts vs a proper
null, and select-then-verify across eras.

This runs the same battery on per-player wind slopes. The data panel is
IDENTICAL to wind_skill.py (same filters, same residuals, same MIN_GAMES),
so any change of verdict is attributable to the instrument, not the data.

  1. SPIKE AND SLAB. b_p ~ (1-pi)*delta_0 + pi*N(mu, sigma^2) observed
     through per-player OLS noise. The LR against pi=0 is the test; the
     same fit runs on permutation replicates (wind shuffled within
     player-era) because heavy-tailed noise alone can manufacture pi > 0.
  2. TAIL COUNT. Players past |z| bars vs the same count on null seasons.
     Both tails reported: wind-strong and wind-fragile.
  3. SELECT THEN VERIFY. Name the top K wind players on 2024-25 alone,
     measure that fixed roster's slope on 2026 alone, against the identical
     procedure on null replicates. The only test that establishes
     individuals.

Reference outcome (clutch, same battery): doubles LR 72.2 vs null 1.0±1.3;
tail z>2.5 6 vs 2; STV K=40 z=3.77 vs 0.29±1.00.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import sigmoid, team_eta  # noqa: E402

MIN_GAMES = 40        # per player, full panel (matches wind_skill.py)
MIN_GAMES_ERA = 20    # per era for select-then-verify
N_REPS = 60           # permutation replicates (the null "seasons")
N_SS_REPS = 20        # replicates that get a full spike-slab fit
ERA_SPLIT = "2026-01-01"
SEED = 42


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_panel():
    """pid -> list of (wind, signed_residual, date), chronological.
    Byte-for-byte the wind_skill.py construction, plus the date."""
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    v2 = {r["player_id"]: float(r["value_now_mean"])
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

    players = defaultdict(list)
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
            players[g[k]].append((wind, resid, g["date"]))
        for k in ("t2_p1", "t2_p2"):
            players[g[k]].append((wind, -resid, g["date"]))

    players = {p: gs for p, gs in players.items() if len(gs) >= MIN_GAMES}
    return players, n_games


def slope_se(w, y):
    """OLS slope + se. Returns (b, se) or None."""
    n = len(w)
    if n < 10:
        return None
    w = np.asarray(w); y = np.asarray(y)
    mw = w.mean(); my = y.mean()
    sxx = float(((w - mw) ** 2).sum())
    if sxx < 1e-9:
        return None
    b = float(((w - mw) * (y - my)).sum()) / sxx
    ssr = float(((y - my - b * (w - mw)) ** 2).sum())
    if n <= 2:
        return None
    se = math.sqrt(ssr / (n - 2) / sxx)
    return (b, se) if se > 0 else None


def build_arrays(players, rng):
    """Observed + replicate (b, se) per player per arm.

    Null replicates permute wind within (player, era) — breaks any
    player-wind link, preserves each era's wind marginal, residual
    autocorrelation, and the panel structure exactly."""
    pids = sorted(players)
    arms = ("full", "pre26", "y26")
    obs = {a: {"b": [], "se": [], "n": []} for a in arms}
    rep = {a: {"b": np.full((N_REPS, len(pids)), np.nan),
               "se": np.full((N_REPS, len(pids)), np.nan)} for a in arms}

    for j, pid in enumerate(pids):
        gs = players[pid]
        eras = {"pre26": [(w, y) for w, y, d in gs if d < ERA_SPLIT],
                "y26": [(w, y) for w, y, d in gs if d >= ERA_SPLIT]}
        eras["full"] = eras["pre26"] + eras["y26"]
        for a in arms:
            r = slope_se([w for w, _ in eras[a]], [y for _, y in eras[a]])
            if r is None:
                obs[a]["b"].append(np.nan); obs[a]["se"].append(np.nan)
            else:
                obs[a]["b"].append(r[0]); obs[a]["se"].append(r[1])
            obs[a]["n"].append(len(eras[a]))
        for k in range(N_REPS):
            perm = {}
            for e in ("pre26", "y26"):
                winds = [w for w, _ in eras[e]]
                rng.shuffle(winds)
                perm[e] = [(w, y) for w, (_, y) in zip(winds, eras[e])]
            perm["full"] = perm["pre26"] + perm["y26"]
            for a in arms:
                r = slope_se([w for w, _ in perm[a]], [y for _, y in perm[a]])
                if r is not None:
                    rep[a]["b"][k, j] = r[0]
                    rep[a]["se"][k, j] = r[1]

    for a in arms:
        obs[a] = {k: np.array(v, float) for k, v in obs[a].items()}
    return pids, obs, rep


def stats(obs, rep, arm, min_n, r=None):
    """Centered b + se for one arm; r=None -> observed, else replicate r
    scored against the remaining replicates (clutch_rare protocol)."""
    br = rep[arm]["b"]
    if r is None:
        b, se, n = obs[arm]["b"], obs[arm]["se"], obs[arm]["n"]
        others = np.arange(N_REPS)
    else:
        b, se = br[r], rep[arm]["se"][r]
        n = obs[arm]["n"]
        others = np.array([i for i in range(N_REPS) if i != r])
    bm = np.nanmean(br[others], axis=0)
    ok = (np.isfinite(b) & np.isfinite(se) & (n >= min_n)
          & np.isfinite(bm))
    sdm = np.nanstd(br[others], axis=0, ddof=1)
    se_t = np.sqrt(se ** 2 + sdm ** 2 / len(others))
    return np.where(ok, b - bm, np.nan), np.where(ok, se_t, np.nan), ok


def spike_slab(b, se):
    """ML fit of (pi, mu, sigma); LR against pi=0. (clutch_rare.spike_slab
    with slope-scale starting points.)"""
    def nll(th):
        pi = 1.0 / (1.0 + np.exp(-th[0]))
        mu, sig = th[1], np.exp(th[2])
        f0 = norm.pdf(b, 0.0, se)
        f1 = norm.pdf(b, mu, np.sqrt(se ** 2 + sig ** 2))
        return -np.sum(np.log(np.maximum((1 - pi) * f0 + pi * f1, 1e-300)))

    best, bx = None, None
    for p0 in (-3.0, -1.5, 0.0):
        for s0 in (np.log(0.001), np.log(0.004)):
            r = minimize(nll, [p0, 0.0, s0], method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-9,
                                  "fatol": 1e-9})
            if best is None or r.fun < best:
                best, bx = r.fun, r.x
    pi = 1.0 / (1.0 + np.exp(-bx[0]))
    mu, sig = bx[1], np.exp(bx[2])
    f0 = norm.pdf(b, 0.0, se)
    f1 = norm.pdf(b, mu, np.sqrt(se ** 2 + sig ** 2))
    post = pi * f1 / np.maximum((1 - pi) * f0 + pi * f1, 1e-300)
    ll0 = np.sum(np.log(np.maximum(norm.pdf(b, 0.0, se), 1e-300)))
    return {"pi": pi, "mu": mu, "sigma": sig, "post": post,
            "lr": 2 * (-best - ll0)}


def main():
    import random
    rng = random.Random(SEED)
    players, n_games = load_panel()
    names = {r["player_id"]: r["full_name"]
             for r in read_csv(ROOT / "data/v2_players.csv")}
    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# Wind skill as a RARE trait — spike-and-slab / tail / "
        "select-then-verify\n")
    say(f"{n_games} outdoor games with match-hour wind; {len(players)} "
        f"players with ≥{MIN_GAMES} games (identical panel to "
        f"wind_skill.py); {N_REPS} within-player-era permutation "
        "replicates as the null.\n")

    pids, obs, rep = build_arrays(players, rng)

    # calibration: are OLS slope se's honest? var(z) on null replicates
    zs_null = []
    for r in range(N_REPS):
        b, se, ok = stats(obs, rep, "full", MIN_GAMES, r=r)
        zs_null.append(np.nanvar((b / se)[ok]))
    say(f"Null calibration: var(z) across replicates = "
        f"{np.mean(zs_null):.2f} (should be ≈1; the observed-vs-null "
        "comparisons below are valid either way since both sides get "
        "identical treatment).\n")

    # ---- 1. spike and slab -----------------------------------------
    say("## [1] Spike and slab — what fraction of players have ANY wind "
        "skill?\n")
    say("| arm | players | pi | slab mu (/mph) | slab sd | LR | null LR |")
    say("|---|---|---|---|---|---|---|")
    b, se, ok = stats(obs, rep, "full", MIN_GAMES)
    f = spike_slab(b[ok], se[ok])
    nlr, npi = [], []
    for r in range(N_SS_REPS):
        bn, sn, okn = stats(obs, rep, "full", MIN_GAMES, r=r)
        fn = spike_slab(bn[okn], sn[okn])
        nlr.append(fn["lr"]); npi.append(fn["pi"])
    say(f"| full | {int(ok.sum())} | {f['pi']:.3f} | {f['mu']:+.5f} "
        f"| {f['sigma']:.5f} | **{f['lr']:.1f}** "
        f"| {np.mean(nlr):.1f}±{np.std(nlr):.1f} "
        f"(max {np.max(nlr):.1f}) |")
    say(f"\n(clutch, same test: LR 72.2 doubles / 18.2 singles vs null "
        f"≈1. pi alone is not quotable — the LR against its own null is "
        "the test.)\n")

    # ---- 2. tail counts --------------------------------------------
    say("## [2] Tail counts — players past |z| bars vs null seasons\n")
    say("| tail | bar | observed | null median | null 95% | p |")
    say("|---|---|---|---|---|---|")
    b, se, ok = stats(obs, rep, "full", MIN_GAMES)
    zz = (b / se)[ok]
    for sign, tag in ((+1, "wind-strong (z>bar)"), (-1, "wind-fragile (z<-bar)")):
        for bar in (2.0, 2.5, 3.0):
            obs_n = int((sign * zz > bar).sum())
            nulls = []
            for r in range(N_REPS):
                bn, sn, okn = stats(obs, rep, "full", MIN_GAMES, r=r)
                nulls.append(int((sign * (bn / sn)[okn] > bar).sum()))
            nulls = np.array(nulls)
            p = float((nulls >= obs_n).mean())
            ptxt = f"<{1/len(nulls):.3f}" if p == 0 else f"{p:.3f}"
            say(f"| {tag} | {bar} | {obs_n} | {np.median(nulls):.0f} "
                f"| {np.percentile(nulls, 95):.0f} | {ptxt} |")

    # ---- 3. select then verify -------------------------------------
    say("\n## [3] Select then verify — name them on 2024-25, test on 2026\n")
    say("(both directions: 'strong' selects the top wind-positive tail, "
        "'fragile' the wind-negative tail; obs z > null in EITHER row "
        "means the trait persisted across eras)\n")
    say("| tail K | obs z | null mean | null sd | p | n avail |")
    say("|---|---|---|---|---|---|")
    b1, s1, o1 = stats(obs, rep, "pre26", MIN_GAMES_ERA)
    b2, s2, o2 = stats(obs, rep, "y26", MIN_GAMES_ERA)
    both = o1 & o2
    idx = np.where(both)[0]

    def verify(bb1, ss1, bb2, ss2, K, sign=+1):
        """sign=+1: select top wind-strong, expect positive 2026 slope.
        sign=-1: select most wind-fragile, expect negative."""
        order = np.argsort(-sign * (bb1 / ss1))
        sel = order[:K]
        w = 1.0 / ss2[sel] ** 2
        est = np.sum(w * bb2[sel]) / np.sum(w) * math.sqrt(np.sum(w))
        return sign * est   # positive = trait persisted, either direction

    for sign, tag in ((+1, "strong"), (-1, "fragile")):
        for K in (5, 10, 20, 40):
            if K > len(idx):
                continue
            obs_z = verify(b1[idx], s1[idx], b2[idx], s2[idx], K, sign)
            nl = []
            for r in range(N_REPS):
                n1, m1, k1 = stats(obs, rep, "pre26", MIN_GAMES_ERA, r=r)
                n2, m2, k2 = stats(obs, rep, "y26", MIN_GAMES_ERA, r=r)
                j = np.where(k1 & k2)[0]
                if len(j) >= K:
                    nl.append(verify(n1[j], m1[j], n2[j], m2[j], K, sign))
            nl = np.array(nl)
            p = float((nl >= obs_z).mean())
            ptxt = f"<{1/len(nl):.3f}" if p == 0 else f"{p:.3f}"
            say(f"| {tag} {K} | {obs_z:+.2f} | {np.mean(nl):+.2f} "
                f"| {np.std(nl):.2f} | {ptxt} | {len(idx)} |")
    say("\n(clutch, same test: K=40 z=3.77 vs null 0.29±1.00; K=5 3.31 "
        "vs 0.02±0.97.)\n")

    # ---- 4. who (only meaningful if [1]-[3] found something) --------
    say("## [4] Top posterior P(slab) — DO NOT PUBLISH unless the LR/"
        "tail/STV tests above cleared their nulls\n")
    b, se, ok = stats(obs, rep, "full", MIN_GAMES)
    uu = np.array(pids)[ok]
    g = obs["full"]["n"][ok]
    order = np.argsort(-f["post"])
    say("| player | games | slope/10mph | z | P(slab) |")
    say("|---|---|---|---|---|")
    for i in order[:8]:
        say(f"| {names.get(uu[i], uu[i][:8])} | {int(g[i])} "
            f"| {b[ok][i]*10:+.4f} | {(b/se)[ok][i]:+.2f} "
            f"| {f['post'][i]:.2f} |")

    # honest power statement
    med_se = float(np.nanmedian(se[ok]))
    say(f"\n## Power\n\nMedian per-player slope se = {med_se*10:.4f} "
        f"share per +10 mph. A wind specialist worth +0.02 share at "
        f"+10 mph (≈ +0.4–0.5 points in a 22-point game) sits at "
        f"z ≈ {0.002/med_se:.1f} for a median player — individually "
        "invisible, but the battery aggregates: the spike-slab LR and "
        "tail counts see a 10-15% minority of that size if it exists, "
        "and select-then-verify sees whether the SAME names repeat "
        "across eras.")
    say("\n---\n*Panel identical to wind_skill.py: current-form v2 "
        "values applied retroactively, outdoor labels heuristic, "
        "match-hour wind. Null = wind permuted within player-era. "
        "Verdict changes vs wind_skill.py are attributable to the "
        "instrument, not the data.*")

    (ROOT / "model/wind_rare.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/wind_rare.md")


if __name__ == "__main__":
    main()
