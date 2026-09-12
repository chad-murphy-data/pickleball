# Learned contact classifier: first experiment

A small ridge-regularized logistic classifier combines ball/paddle distance,
incoming/outgoing image motion, fit quality, arm reach, and torso-relative wrist
motion. Four rallies train each model; the fifth supplies its evaluation events.
Scaling and missing-value imputation use training data only.

There was one fixed run: 20 Hz features, a class-balanced loss, score threshold
0.5, and 0.2-second peak suppression. No threshold search. The score is not a
calibrated probability. Training uses contact neighborhoods as positives,
distant frames as provisional negatives, and excludes ambiguous flanks.
Evaluation retains the entire supplied clip, including those flanks.

| Measure | Proximity | Motion rule | Learned classifier |
|---|---:|---:|---:|
| Labeled contacts | 78 | 78 | 78 |
| Matched | 52 | 56 | 71 |
| Missed | 26 | 22 | 7 |
| Extra detections | 27 | 55 | 62 |
| Precision | 65.8% | 50.5% | 53.4% |
| Recall | 66.7% | 71.8% | 91.0% |
| Correct hitter among matches | 49 | 54 | 66 |
| Wrong hitter among matches | 0 | 0 | 0 |
| Unknown hitter among matches | 3 | 2 | 5 |

Per-rally matched/missed/extra: r6 6/1/2; r7 8/1/10; r8 7/0/4;
r9 28/1/27; r10 22/4/19.

Interpretation: improved contact recall, but not a deployable replacement.
More detections alone are not proof of better discrimination, and the models
have different operating points. Any threshold selection must happen within
training folds before evaluating an outer held-out rally. Do not choose a
threshold from these pooled test outcomes and call the result held-out.

This is an exploratory same-match comparison. These rallies informed feature
choice; human identities and contact-derived clip windows remain inputs; rally
10 includes upstream ball-training frames. The whole system is not held out.
Additional untouched rallies are required before claiming generalization.

The next useful diagnostic is whether false candidates are duplicate detections
around real hits, ball-tracking jumps, or genuinely confusing non-contact motion.
Do not presume that raising a threshold or adding another model solves them.

```bash
python3 vision/learned_contact_experiment.py --pose-dir pose \
  --predictions-zip ball_contact_review_inputs.zip \
  --hitter-report data/vision/ball_hitter_run1/report.json \
  --baseline-report data/vision/auto_contact_run1/report.json \
  --out data/vision/learned_contact_run1
```

Report includes source hashes, fold memberships, scaler values, coefficients,
events and detailed errors. No source labels or existing detector defaults change.

## Nested threshold follow-up

Fixed grid 0.50 through 0.95 in steps of 0.05. For each outer test rally,
the other four rallies undergo an inner leave-one-rally-out pass. Choose the
threshold maximizing pooled event F0.5 (precision weighted more than recall),
with ties preferring higher thresholds. The outer test rally is excluded from
all threshold selection and preprocessing. Refit on the four training rallies,
then evaluate the selected threshold on the outer rally once.

| Measure | Fixed 0.5 classifier | Nested threshold | Original proximity |
|---|---:|---:|---:|
| Matched contacts | 71 | 51 | 52 |
| Missed | 7 | 27 | 26 |
| Extra | 62 | 26 | 27 |
| Precision | 53.4% | 66.2% | 65.8% |
| Recall | 91.0% | 65.4% | 66.7% |
| Correct hitter among matches | 66 | 46 | 49 |
| Unknown hitter among matches | 5 | 5 | 3 |
| Wrong hitter among matches | 0 | 0 | 0 |

Outer rally thresholds: r6 0.80, r7 0.80, r8 0.80, r9 0.90, r10 0.75.
Inner objectives pool event counts, so longer rallies contribute more.
Class-balanced scores remain uncalibrated. The same upstream and feature-choice
caveats apply; nested selection does not turn development footage into a final
untouched test set.

Decision: no clear improvement over the original proximity baseline. Keep the
experiment optional; do not replace existing defaults or continue selecting
thresholds from outer test results. Add `--nested-threshold` to the reproduction
command and choose a new output directory to reproduce this run.
