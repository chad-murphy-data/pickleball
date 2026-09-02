# Better learner — pre-registration (2026-09-02, written before any number)

Owner's go (2026-09-02, night): "Go ahead and do the better learner",
after the procedure was described as: replace the per-candidate
logistic with gradient-boosted trees on the SAME 14 features, judged on
r6/r7 cross-fold first; a new scorer changes the p-caches, so the
path-first cell is re-tuned on r6/r7 and shot ONCE on r9/r10 under the
same bars. That go covers the shot; it is disclosed here as the THIRD
r9/r10 shot for the tracker layer (path-first, hand-off, this).

## What changes and what does not
- Features: the 14 in `emission.py` (`FEATS`), unchanged. Harvest,
  positives (owner V clicks within R_POS = 6 px), ignore zones (any
  click within R_IGN = 22 px), negatives, train rallies r6 + r7 only:
  ALL unchanged — `learner.py` calls `emission.harvest_train`.
- Model: sklearn `HistGradientBoostingClassifier`, balanced sample
  weights exactly as the logistic (pos weight = n_neg / n_pos). Three
  fixed configurations, chosen before any number, selected by MEAN
  cross-rally AUC (6→7, 7→6), ties → the smaller model:
  A: max_leaf_nodes 15, max_iter 300, learning_rate 0.05
  B: max_leaf_nodes 31, max_iter 300, learning_rate 0.05
  C: max_leaf_nodes 15, max_iter 600, learning_rate 0.03
  (l2 1.0, min_samples_leaf 20, early stopping OFF, random_state 7.)
- Tracker: `pathfirst.py` code untouched except an env hook
  (`PF_PXS`) that appends a suffix to the p-cache filename; default
  behaviour identical. Caches from the new scorer are written under
  `_gbt` (`p_r{9,10}_{cc,peak}_14_gbt.npz` from the pooled model,
  `p_r{6,7}_{cc,peak}_14_x_gbt.npz` cross-fold: r6 scored by r7-only,
  r7 by r6-only, as today). The incumbent caches are not overwritten.

## Gate 1 — the scorer (r6/r7 cross-fold, no seal)
Incumbent logistic: AUC 0.9042 (6→7) / 0.9394 (7→6); negatives kept at
97 % held-out positive recall 0.7229 / 0.3192.
PASS only if the selected configuration has cross-rally AUC ≥ the
logistic in BOTH directions AND negatives-kept ≤ the logistic in BOTH
directions. Otherwise DEAD: no caches, no tune, no shot.

## Gate 2 — the tracker on the new caches (r6/r7, no seal)
Same 12-cell grid as pathfirst_gate.md (P_SEED {0.4, 0.6} × S_MIN
{4, 6, 8} × GAP {6, 12}), same code path (`pathfirst.run`). Rule: max
pooled r@12 on r6+r7 subject to pooled prec@12 ≥ 0.807 − 0.03 and both
nulls ≤ 3; ties → larger S_MIN, smaller GAP, larger P_SEED. The winner
must beat the path-first incumbent on its own caches (263 @ 0.807)
strictly, else DEAD. Written to `pathfirst_tune_gbt.json`; the frozen
`pathfirst_tune.json` is never touched.

## The one shot (r9, r10) — bars, never loosened
ADOPT only if on BOTH rallies: r@12 strictly above the incumbent
(537 / 422); prec@12 ≥ incumbent − 0.02 (0.85 / 0.86); displaced and
time-shift nulls r@12 ≤ 3; and the adopted events layer (v3 cell,
unchanged) re-run on the new track keeps F1 ≥ adopted − 0.03
(.701 / .645). Anything else = NOT ADOPTED, recorded, incumbent stays.
Secondary (reported, not bars): r@8, at-click coverage, V/S strata,
events recall/precision, flights count.
Owner clicks on r9/r10 are used for grading only.

## Results — gate 1 (2026-09-02, learner_train.txt): DEAD

| direction | logistic AUC / neg_kept@97 | trees A | trees B | trees C |
|---|---|---|---|---|
| 6→7 (198 pos → 161) | 0.9042 / 0.723 | 0.9060 / 0.694 | 0.9059 / 0.739 | 0.9085 / 0.554 |
| 7→6 (161 pos → 198) | 0.9394 / 0.319 | 0.9464 / **0.635** | 0.9329 / 0.928 | 0.9407 / 0.745 |

Selected A (mean AUC 0.9262). AUC is above the logistic in both
directions, by +0.002 and +0.007 — noise-sized on 161–198 positives.
The tail bar fails: at 97 % held-out positive recall the trees keep
0.635 of r6's negatives where the logistic keeps 0.319. That threshold
is set by the ~5 lowest-scored true balls of 198; trees fit on r7's 161
positives score r6's oddest positives (blur, near-body) far lower than
a smooth logistic surface does, and the threshold has to drop to the
floor to catch them. Rule says DEAD: no caches, no tune, no shot. The
`PF_PXS` hook stays (inert when unset).

## Diagnostic — learning curve (learner_curve.py → learner_curve.txt)
Train positives subsampled at 25/50/75/100 %, all negatives kept, 5
seeds, tested on the other rally. NOT a gate.

| train pos | logistic AUC 6→7 / 7→6 | trees AUC 6→7 / 7→6 | logistic tail 7→6 | trees tail 7→6 |
|---|---|---|---|---|
| 25 % | 0.907 / 0.922 | 0.869 / 0.929 | 0.58 | 0.82 |
| 50 % | 0.898 / 0.931 | 0.877 / 0.923 | 0.40 | 0.94 |
| 75 % | 0.904 / 0.940 | 0.916 / 0.944 | 0.34 | 0.71 |
| 100 % | 0.904 / 0.939 | 0.906 / 0.946 | 0.32 | 0.64 |

Reading: the logistic's AUC is FLAT from a quarter of the labels to all
of them — on these 14 features it is saturated, and more clicks will
not move it much (its tail does keep improving with labels). The trees
are still CLIMBING at the full count, in AUC and in the tail, and only
overtake the logistic at ≥ 75 % of the positives. So the learner is
label-limited for the model that could beat the incumbent, and
feature-limited for the incumbent. The next learner step is clicks,
not code: at roughly double the positives (three to five more labeled
rallies, `label_picks.md`) the curve says trees pass this gate. A
patch-appearance model (the real step change) sits further up the same
curve. Re-run `learner.py train` when new train rallies land; the
gate text above does not change.
