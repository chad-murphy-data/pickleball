"""C3(A) robustness — is the player-FE compression a skill-NONLINEARITY artefact?

The only specification in c3a_fixed_effects.py whose CI excludes zero is the
player-FE one.  A mechanical route to a negative d without any wind effect:
if the true response to measured skill is not exactly linear (an odd,
compressive function), and windy hours systematically contain bigger or
smaller skill gaps than calm hours, the fitted skill slope will differ by
wind bin for purely functional-form reasons.

Test: add odd nonlinear terms in skill (skill^3, skill^5) AND their wind
interactions, so the wind-varying part of the skill response is estimated
flexibly, and re-read d.  Also report the composition fact the artefact
would need: mean |skill| by wind bin.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c3a_fixed_effects as fx  # noqa: E402
from c3_lib import absorb, cluster_se, load_frame  # noqa: E402


def run(spec, rows, order=3, B=600, seed=9):
    D = fx.pack(rows)
    n = len(rows)
    s, w = D["skill"], D["w"]
    cols = [s, w, s * w]
    names = ["skill", "w", "skill*w"]
    for k in (3, 5)[:max(0, (order - 1) // 2)]:
        cols += [s ** k, s ** k * w]
        names += [f"skill^{k}", f"skill^{k}*w"]
    M = np.column_stack([D["y"]] + cols)
    if spec != "L0":
        fe = fx.make_fe(spec, D, np.zeros(n, np.int64))
        M, _ = absorb(fe, M)
    X = np.column_stack([np.ones(n)] + [M[:, i] for i in range(1, M.shape[1])])
    beta = np.linalg.solve(X.T @ X, X.T @ M[:, 0])
    cov, G = cluster_se(X, M[:, 0], beta, D["ev"])
    by = defaultdict(list)
    for i, e in enumerate(D["ev"]):
        by[int(e)].append(i)
    idxs = {k: np.array(v) for k, v in by.items()}
    keys = list(idxs)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        sel = np.concatenate([idxs[keys[i]]
                              for i in rng.integers(0, len(keys), len(keys))])
        Xb, yb = X[sel], M[sel, 0]
        try:
            draws.append(np.linalg.solve(Xb.T @ Xb, Xb.T @ yb)[3])
        except np.linalg.LinAlgError:
            pass
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"  {spec} order={order}: d={beta[3]:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"CRse={np.sqrt(cov[3,3]):.4f}  ({', '.join(names)})")


if __name__ == "__main__":
    allrows = load_frame()
    for pool in ("outdoor", "indoor"):
        rows = [r for r in allrows if r["setting_c"] == pool]
        print(f"\n== {pool}: {len(rows)} games")
        sk = np.array([abs(r["skill"]) for r in rows])
        wd = np.array([r["wind"] for r in rows])
        for lo_, hi_ in ((0, 8), (8, 14), (14, 99)):
            m = (wd >= lo_) & (wd < hi_)
            print(f"   wind {lo_}-{hi_} mph: n={m.sum():6d} "
                  f"mean|skill|={sk[m].mean():.4f}")
        for spec in ("L0", "L2", "L3"):
            for order in (1, 3, 5):
                run(spec, rows, order=order)
