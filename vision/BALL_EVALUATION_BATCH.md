# Fixed 120-frame evaluation batch

Generate after expanded-model overlays for rallies 25 and 27 exist:

```bash
python3 vision/ball_eval_batch.py generate --video full_match.mp4.webm
open data/vision/ball_eval1/r25.html
open data/vision/ball_eval1/r27.html
```

Label one page at a time with V/S/I/N/O as in training batch 1. Each page has
60 samples: 40 temporal-stratum midpoints between first and last recorded
contact, plus 20 nonoverlapping challenge samples. The challenge sampler
prefers both sides of >100-pixel jumps in ten temporal bins, then fills from
the largest remaining jumps. This targets instability, not independently
verified net/body overlap. It excludes the long post-contact tails.

The generator reads data/vision/temporal_batch1_review/r25/predictions.csv and
r27/predictions.csv, freezes their hashes in plan.json, and preserves all raw
files. Both groups are mixed chronologically on each page, with the cohort
hidden from the labeler. Model proposals are disabled unconditionally. Progress
uses separate evaluation cache keys. Exports: ball_eval1_r25.csv and
ball_eval1_r27.csv. Evaluation schema ball-eval-v1 is rejected by the training
loader. Output folders must be new; do not regenerate to cherry-pick samples.

Systematic sampling gives broad temporal coverage, not an independent random
sample or a representative estimate across matches. Forty samples per rally
are sparse and cannot characterize every short failure. The two groups are
reported separately by rally (no misleading 120-frame pooled accuracy).
Challenge selection depends on the expanded model and is therefore biased
toward its unstable predictions. Both rallies have been reviewed; neither is
now an untouched holdout. These caveats travel in the report.

## Comparison after labeling

The original model already has rally 25 predictions under temporal_review_r25.
Produce its missing rally 27 predictions, without training:

```bash
python3 vision/ball_temporal_overlay.py --video full_match.mp4.webm --checkpoint data/vision/temporal_r18_run3/model.pt --rallies 27 --pre-seconds 2 --post-seconds 15 --out data/vision/temporal_original_r27
```

Copy the completed evaluation exports from Downloads into data/vision/ and run:

```bash
python3 vision/ball_eval_batch.py score --plan data/vision/ball_eval1/plan.json --labels data/vision/ball_eval1_r25.csv data/vision/ball_eval1_r27.csv --old-r25 data/vision/temporal_review_r25/r25/predictions.csv --old-r27 data/vision/temporal_original_r27/r27/predictions.csv --new-r25 data/vision/temporal_batch1_review/r25/predictions.csv --new-r27 data/vision/temporal_batch1_review/r27/predictions.csv --out data/vision/ball_eval1_results
```

The scorer requires every planned label and prediction, exact decoded-frame
agreement, unchanged new-model prediction hashes, unseen-context metadata,
valid presence categories and in-image coordinates. It exports paired.csv and
report.json. Localization errors use V/S only; I/N/O remain descriptive counts.
Missing predictions are errors rather than silently dropped examples. Paired
improvement/regression counts use the same visible labels at a 10-pixel threshold.
No presence classification accuracy is claimed: both models always output a
position. No training, trajectory selection, smoothing or interpolation occurs.

Selection, label validation, rejection by training, metric denominators, and
generated JavaScript syntax are tested. Video/browser interaction still requires
the local source video. No model accuracy is known until human labels are supplied.
