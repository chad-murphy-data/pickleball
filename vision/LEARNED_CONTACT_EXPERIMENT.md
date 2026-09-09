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
