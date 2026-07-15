"""Unshrunk ("fixed effects") dyad chemistry estimates.

OLS: margin ~ player FEs (+1 for team-1 members, -1 for team-2) + tour
intercepts + ONE dyad dummy, cluster-robust SEs by match (CR1). The dyad
coefficient is the pair's points-per-game above the sum of its parts, with
no shrinkage prior — the "just look at the data" number the Bayesian model
pulls toward zero.

Run once per dyad of interest (cheap: FWL-style, one shared base projection).
Estimated for every dyad with >= MIN_G games; results to
data/results_dyads_fe.csv and printed for the focal pairs.

Caveat printed in analysis.md: for a player whose games in a context are all
with one partner (Bright in mixed), the dyad dummy is collinear with their
personal context shift — the FE estimate is the SUM of the two.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MIN_G = int(__import__("os").environ.get("FE_MIN_G", 30))
FOCAL_PAIRS = [
    ("Anna Bright", "Hayden Patriquin"),
    ("Anna Leigh Waters", "Ben Johns"),
    ("Anna Leigh Waters", "Anna Bright"),
    ("Ben Johns", "Gabriel Tardio"),
    ("Christian Alshon", "Hayden Patriquin"),
    ("Jorja Johnson", "Tyra Hurricane Black"),
    ("Noe Khlif", "Will Howells"),
]


def main():
    rows = list(csv.DictReader((DATA / "model_data.csv").open()))
    dyads = list(csv.DictReader((DATA / "model_dyads.csv").open()))
    n = len(rows)
    a = np.array([[int(r[k]) for k in ("a1", "a2", "a3", "a4")] for r in rows])
    d1 = np.array([int(r["dyad1"]) for r in rows])
    d2 = np.array([int(r["dyad2"]) for r in rows])
    match = np.array([int(r["match_idx"]) for r in rows])
    tour = np.array([int(r["tour_idx"]) for r in rows])
    y = np.array([float(r["margin"]) for r in rows])
    n_players = a.max() + 1

    # base design: player FEs + tour intercepts
    ii, jj, vv = [], [], []
    for slot, sign in ((0, 1), (1, 1), (2, -1), (3, -1)):
        ii.extend(range(n)); jj.extend(a[:, slot]); vv.extend([sign] * n)
    # tour intercept columns
    ii.extend(range(n)); jj.extend(n_players + tour); vv.extend([1.0] * n)
    X = sparse.csr_matrix((vv, (ii, jj)), shape=(n, n_players + 2))

    XtX = (X.T @ X).toarray()
    # two flat directions (all-players +c; all-women +c) -> pseudoinverse
    XtX_inv = np.linalg.pinv(XtX, rcond=1e-10)
    Xty = X.T @ y
    beta0 = XtX_inv @ Xty
    resid0 = y - X @ beta0

    dyad_games = np.zeros(len(dyads), dtype=int)
    np.add.at(dyad_games, d1, 1)
    np.add.at(dyad_games, d2, 1)

    out = []
    for di in np.where(dyad_games >= MIN_G)[0]:
        z = (d1 == di).astype(float) - (d2 == di).astype(float)
        # FWL: residualize z on X, then beta = <z~, y> / <z~, z~> using resid0
        gamma = XtX_inv @ (X.T @ z)
        z_t = z - X @ gamma
        denom = float(z_t @ z_t)
        if denom < 1e-8:
            continue  # dummy collinear with FEs — not estimable at all
        beta = float(z_t @ resid0) / denom
        # cluster-robust (CR1) SE by match on the FWL regression
        e = resid0 - beta * z_t
        scores = np.zeros(match.max() + 1)
        np.add.at(scores, match, z_t * e)
        g = len(np.unique(match))
        meat = float((scores ** 2).sum()) * g / (g - 1)
        se = np.sqrt(meat) / denom
        dy = dyads[di]
        out.append({
            "p1_name": dy["p1_name"], "p2_name": dy["p2_name"],
            "context": dy["context"], "games": int(dyad_games[di]),
            "fe_estimate": round(beta, 3), "fe_se": round(se, 3),
            "t": round(beta / se, 2) if se > 0 else 0.0,
        })

    out.sort(key=lambda r: -r["fe_estimate"])
    with (DATA / "results_dyads_fe.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    ts = [r["t"] for r in out]
    print(f"dyads estimated (>= {MIN_G} games): {len(out)}")
    print(f"t-stat spread: mean {np.mean(ts):+.2f}, sd {np.std(ts):.2f} "
          "(≈1 expected if chemistry were pure noise)")
    print()
    for want in FOCAL_PAIRS:
        for r in out:
            if {r["p1_name"], r["p2_name"]} == set(want):
                print(f'{r["p1_name"]} + {r["p2_name"]} ({r["context"]}, {r["games"]}g): '
                      f'{r["fe_estimate"]:+.2f} ± {r["fe_se"]:.2f}  (t={r["t"]})')


if __name__ == "__main__":
    main()
