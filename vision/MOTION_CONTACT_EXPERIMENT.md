# Motion contact experiment, first run

Development experiment on Chicago rallies 6–10, using the same uploaded
predictions and poses as the proximity baseline. No model retraining and no
threshold sweep. Human contact times are passed only to the evaluator;
human identities and contact-derived clip boundaries remain inputs.

The experiment fits robust image-space lines before and after a candidate
time, requires a velocity change and improved fit near an estimated paddle,
and rejects large discontinuities and poor fits. Hitter assignment is unchanged.
These are exploratory thresholds, not general physical laws.

| Measure | Proximity baseline | Motion experiment |
|---|---:|---:|
| Labeled contacts | 78 | 78 |
| Matched contacts | 52 | 56 |
| Missed contacts | 26 | 22 |
| Extra detections | 27 | 55 |
| Precision | 65.8% | 50.5% |
| Recall | 66.7% | 71.8% |
| Correct hitter among matched contacts | 49 | 54 |
| Wrong hitter among matched contacts | 0 | 0 |
| Unknown hitter among matched contacts | 3 | 2 |

Decision: do not replace the baseline. Four additional net matches cost 28
additional extra detections. This result does not reject trajectory information
in general; it rejects adopting this particular standalone rule on these data.
Possible sources include bounces, projection, tracking noise, and overlapping
player geometry. The report alone does not identify the cause of each extra.

Four synthetic tests check a straight flight, a turn, a teleport, and a turn far
from a paddle. They verify intended mechanics, not accuracy on match footage.

Reproduce using the five-prediction ZIP, original pose files, prior hitter report,
and prior automatic-contact report:

```bash
python3 vision/motion_contact_experiment.py --pose-dir pose \
  --predictions-zip ball_contact_review_inputs.zip \
  --hitter-report data/vision/ball_hitter_run1/report.json \
  --baseline-report data/vision/auto_contact_run1/report.json \
  --out data/vision/motion_contact_run1
```

The script verifies pose/prediction hashes against both earlier reports and
records source hashes in its output. The Ting Chieh correction comes from the
prior hitter report. No source labels or existing detector defaults are changed.
