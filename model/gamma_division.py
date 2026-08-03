"""Is the weakest-link gamma the same in men's, women's, and mixed?

    python model/gamma_division.py    # prints + writes model/gamma_division.md

Background: v2 fits ONE pooled gamma (posterior mean -0.183 logit) for
team = a + b + gamma*|a-b|.  Algebraically that is a weighted sum
0.41*better + 0.59*worse (normalized), the "0.42/0.58" weighting.  Nobody
has checked whether the three divisions share it.

Method: profile likelihood with player values FIXED at their v2
month-of-game means (trajectories where dynamic, static mean otherwise) —
the same conditioning used for GWAE (finding 9).  Per division, maximize
the per-point Binomial likelihood over gamma alone; CIs from a
match-cluster bootstrap (the match random effect makes plain binomial
CIs overconfident).  Caveat, stated up front: values were fitted jointly
with the POOLED gamma, so each division's values have partially adapted
to it; this test detects divergence conditional on those values, and a
large divergence would warrant a full per-division-gamma refit of v2.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

N_BOOT = 1000
SEED = 42


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_values():
    static = {}
    for r in read_csv(DATA / "v2_players.csv"):
        static[r["player_id"]] = float(r["value_now_mean"])
    traj = defaultdict(dict)
    for r in read_csv(DATA / "v2_trajectories.csv"):
        traj[r["player_id"]][r["month"]] = float(r["value_mean"])
    return static, traj


def value_at(pid, month, static, traj):
    t = traj.get(pid)
    if t:
        if month in t:
            return t[month]
        months = sorted(t)
        if month < months[0]:
            return t[months[0]]
        return t[months[-1]]
    return static.get(pid)


def prep():
    static, traj = load_values()
    rows = defaultdict(list)   # division -> list of games
    n_skipped = 0
    for g in read_csv(DATA / "games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        div = g["context"]
        if div not in ("mens", "womens", "mixed"):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 6:
            continue
        month = g["date"][:7]
        vals = [value_at(g[k], month, static, traj)
                for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if any(v is None for v in vals):
            n_skipped += 1
            continue
        a1, a2, b1, b2 = vals
        base = (a1 + a2) - (b1 + b2)
        g1, g2 = abs(a1 - a2), abs(b1 - b2)
        rows[div].append((base, g1 - g2, s1, s2, g["match_id"], (g1 + g2) / 2))
    return rows, n_skipped


def fit_gamma(base, dgap, s1, s2):
    """1-D binomial MLE of gamma given fixed values."""
    def nll(gam):
        eta = base + gam * dgap
        # -sum[ s1*log(sig(eta)) + s2*log(sig(-eta)) ]
        return float(np.sum((s1 + s2) * np.logaddexp(0.0, -eta) + s2 * eta))
    r = minimize_scalar(nll, bounds=(-1.5, 1.0), method="bounded",
                        options={"xatol": 1e-6})
    return r.x


def boot_gammas(games, rng):
    """Match-cluster bootstrap distribution of gamma-hat."""
    by_match = defaultdict(list)
    for row in games:
        by_match[row[4]].append(row)
    matches = list(by_match.values())
    out = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(matches), len(matches))
        sample = [row for i in idx for row in matches[i]]
        base = np.array([r[0] for r in sample])
        dgap = np.array([r[1] for r in sample])
        s1 = np.array([r[2] for r in sample], float)
        s2 = np.array([r[3] for r in sample], float)
        out.append(fit_gamma(base, dgap, s1, s2))
    return np.array(out)


def weights(gam):
    """Normalized (better, worse) partner weights implied by gamma."""
    return (1 + gam) / 2, (1 - gam) / 2


def main():
    rows, n_skipped = prep()
    rng = np.random.default_rng(SEED)
    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# Weakest-link gamma by division\n")
    say(f"Games with month-of-game v2 values for all four players "
        f"({n_skipped} skipped for missing values):")
    for div in ("mens", "womens", "mixed"):
        say(f"  {div}: {len(rows[div])}")
    say("")
    say("| division | games | gamma (MLE) | 95% CI (match bootstrap) | "
        "better wt | worse wt |")
    say("|---|---|---|---|---|---|")

    results = {}
    all_games = [r for div in ("mens", "womens", "mixed") for r in rows[div]]
    for label, games in (("pooled", all_games),
                         ("mens", rows["mens"]),
                         ("womens", rows["womens"]),
                         ("mixed", rows["mixed"])):
        base = np.array([r[0] for r in games])
        dgap = np.array([r[1] for r in games])
        s1 = np.array([r[2] for r in games], float)
        s2 = np.array([r[3] for r in games], float)
        gam = fit_gamma(base, dgap, s1, s2)
        bs = boot_gammas(games, rng)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        wb, ww = weights(gam)
        results[label] = (gam, lo, hi, bs)
        say(f"| {label} | {len(games)} | {gam:+.3f} | [{lo:+.3f}, {hi:+.3f}] "
            f"| {wb/(wb+ww):.3f} | {ww/(wb+ww):.3f} |")

    # pairwise differences via the shared bootstrap draws
    say("\nPairwise differences (bootstrap of the difference):\n")
    say("| contrast | Δgamma | 95% CI |")
    say("|---|---|---|")
    for a, b in (("mens", "womens"), ("mens", "mixed"), ("womens", "mixed")):
        d = results[a][3] - results[b][3]
        dm = results[a][0] - results[b][0]
        lo, hi = np.percentile(d, [2.5, 97.5])
        say(f"| {a} − {b} | {dm:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")

    say("\n*Values fixed at v2 month-of-game means (fitted with the pooled "
        "gamma), so divisions' values have partially adapted to the shared "
        "gamma — this is a conditional test. Match-cluster bootstrap "
        f"(n={N_BOOT}) absorbs the match random effect. Per-division "
        "|gap| spread differs (mixed pairs are wider), which is the power "
        "driver.*")

    # context: how much |gap| identification each division carries
    say("\n| division | mean team |gap| | sd of Δgap (identifying spread) |")
    say("|---|---|---|")
    for div in ("mens", "womens", "mixed"):
        games = rows[div]
        dgap = np.array([r[1] for r in games])
        mg = np.mean([r[5] for r in games])
        say(f"| {div} | {mg:.3f} | {np.std(dgap):.3f} |")

    (ROOT / "model/gamma_division.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/gamma_division.md")


if __name__ == "__main__":
    main()
