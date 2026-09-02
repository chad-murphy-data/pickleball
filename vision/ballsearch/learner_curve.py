"""Diagnostic only (no gate, no seal): learning curve on r6/r7.
Train on one rally with its positives SUBSAMPLED (all negatives kept),
test on the other; both the logistic and cfg-A trees. Answers "is the
learner label-limited?" — if AUC / tail are still rising at the full
positive count, more clicks move it; if flat, the model class does.
    python3 learner_curve.py   -> learner_curve.txt
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import emission as em
import learner as L

packs = {r: em.harvest_train(r) for r in (6, 7)}
FRACS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (1, 2, 3, 4, 5)
for tr_r, te_r in ((6, 7), (7, 6)):
    X, y = packs[tr_r]
    Xte, yte = packs[te_r]
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    print(f"train r{tr_r} ({len(pos)} pos) -> test r{te_r} ({int(yte.sum())} pos)")
    for fr in FRACS:
        n = max(10, int(round(fr * len(pos))))
        rows = {"logit": [], "trees": []}
        for s in SEEDS:
            rng = np.random.default_rng(s)
            idx = np.concatenate([rng.choice(pos, n, replace=False), neg])
            Xs, ys = X[idx], y[idx]
            mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
            w = np.where(ys == 1, (ys == 0).sum() / max(ys.sum(), 1), 1.0)
            th = em.fit_logistic((Xs - mu) / sd, ys, w)
            sc = ((Xte - mu) / sd) @ th[:-1] + th[-1]
            st = L.xval_stats(sc, yte)
            rows["logit"].append((st["auc"], st["neg_kept"]))
            m = L.fit(Xs, ys, L.CONFIGS["A"])
            st = L.xval_stats(m.predict_proba(Xte)[:, 1], yte)
            rows["trees"].append((st["auc"], st["neg_kept"]))
            if fr == 1.0:
                break
        for k, v in rows.items():
            v = np.asarray(v)
            print(f"  {int(fr*100):3d}% pos (n={n:3d}) {k:5s}: AUC {v[:,0].mean():.4f} ± {v[:,0].std():.4f}"
                  f"   neg_kept@97% {v[:,1].mean():.3f} ± {v[:,1].std():.3f}")
