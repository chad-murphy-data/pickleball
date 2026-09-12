"""How many CLICKED RALLIES does the emission learner need?  DIAGNOSTIC ONLY.

`learner_curve.py` answered a different axis: it subsampled positives WITHIN
one rally (flat -> the learner is not count-limited inside a rally).  What was
never measured is the axis that actually varies when the owner clicks another
rally: the number of distinct TRAIN RALLIES pooled.  This sweeps it,
leave-one-rally-out over the seven clicked train rallies.

    python3 label_curve.py            -> label_curve.txt

No knob is tuned, no gate is run, no seal is touched, nothing is written back
to a model file.  Eval rallies (r9 / r10) and every temporal-gate holdout
rally are excluded by construction -- TRAIN is the literal rally list below.

Result 2026-09-03 (label_curve.txt): FLAT.  k=1 -> k=6 moves AUC 0.9377 ->
0.9461 (+0.008, inside the +/-0.015 between-rally spread) and the operational
metric -- negatives surviving at 97% recall -- does not improve at all
(0.555 -> 0.558).  Read with learner_curve.txt, both label axes are saturated:
more clicked ball-path rallies do NOT improve this model class.  Caveat: this
grades the emission SCORER in isolation, not the whole path-first + gap-fill
stack, and it says nothing about conditioning (r5's dbody/crowd failure is a
regime problem, not a volume problem).
"""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emission as em  # noqa: E402

TRAIN = [2, 3, 4, 5, 6, 7, 17]
MAXDRAW = 12          # random subsets per (k, held-out rally) when combos explode
CACHE = HERE / "label_curve_packs.npz"    # gitignored; regenerate in ~4 min


def packs():
    """{rally: (features, labels)} for every train rally, cached on disk."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {r: (z[f"X{r}"], z[f"y{r}"]) for r in TRAIN}
    out = {r: em.harvest_train(r) for r in TRAIN}
    np.savez_compressed(CACHE, **{f"X{r}": out[r][0] for r in TRAIN},
                        **{f"y{r}": out[r][1] for r in TRAIN})
    return out


def fit_eval(P, tr_rs, te_r):
    """Pool tr_rs, fit the production logistic, score the held-out rally.
    Returns (AUC, fraction of negatives kept at 97% positive recall)."""
    X = np.vstack([P[r][0] for r in tr_rs])
    y = np.concatenate([P[r][1] for r in tr_rs])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w = np.where(y == 1, (y == 0).sum() / max(y.sum(), 1), 1.0)
    th = em.fit_logistic((X - mu) / sd, y, w)
    Xte, yte = P[te_r]
    sc = ((Xte - mu) / sd) @ th[:-1] + th[-1]
    pos = np.sort(sc[yte == 1])
    thr = pos[max(0, int(0.03 * len(pos)) - 1)]
    return float(em.auc(sc, yte)), float((sc[yte == 0] >= thr).mean())


def main():
    P = packs()
    for r in TRAIN:
        X, y = P[r]
        print(f"  r{r:<3d} {len(y):7d} labeled cands  {int(y.sum()):4d} pos")

    rng = np.random.default_rng(0)
    print("\nleave-one-rally-out: pool k train rallies, test on the held-out one")
    print(f"{'k':>2}  {'AUC':>16}   {'neg kept @97% recall':>22}   draws")
    curve = {}
    for k in range(1, len(TRAIN)):
        A, N = [], []
        for te in TRAIN:
            pool = [r for r in TRAIN if r != te]
            combos = list(itertools.combinations(pool, k))
            if len(combos) > MAXDRAW:
                combos = [combos[i] for i in
                          rng.choice(len(combos), MAXDRAW, replace=False)]
            for c in combos:
                a, n = fit_eval(P, list(c), te)
                A.append(a)
                N.append(n)
        A, N = np.array(A), np.array(N)
        curve[k] = (A.mean(), N.mean())
        print(f"{k:>2}  {A.mean():.4f} ± {A.std():.4f}   "
              f"{N.mean():.3f} ± {N.std():.3f}   n={len(A)}")

    print("\nmarginal gain per added rally (neg kept, lower is better):")
    ks = sorted(curve)
    for i in range(1, len(ks)):
        d_a = curve[ks[i]][0] - curve[ks[i - 1]][0]
        d_n = curve[ks[i]][1] - curve[ks[i - 1]][1]
        print(f"  {ks[i-1]} -> {ks[i]}:  AUC {d_a:+.4f}   neg kept {d_n:+.4f}")


if __name__ == "__main__":
    main()
