# Player-specific contact experiment

Each feature row now represents one player at one instant. All paddle proximity,
arm reach, and torso-relative wrist motion in that row belong to that player.
Ball motion is shared context. Names and track IDs are not numeric model inputs.
Multiple track fragments for a player remain supported. Both arms are still
aggregated within that player.

At labeled contacts only the labeled hitter is positive; other players are
negative. The existing ambiguous-time exclusion buffer remains. Feature
definitions, optimizer, regularization, event matching, and peak suppression
are reused. This changes both representation and supervision to player-level
contact classification, so it is not a pure feature-only ablation.

Five outer rally folds, each trained on the other four rallies. The threshold
uses inner rally validation and pooled event F0.5 on the same fixed 0.50–0.95
grid. No threshold sweep on outer results. Hitter attribution remains the
existing proximity method; classifier player suggestions are recorded separately.

| Measure | Original proximity | Prior classifier, nested | Player-specific, nested | Player-specific, fixed 0.5 |
|---|---:|---:|---:|---:|
| Labeled contacts | 78 | 78 | 78 | 78 |
| Matched | 52 | 51 | 54 | 76 |
| Missed | 26 | 27 | 24 | 2 |
| Extra detections | 27 | 26 | 30 | 87 |
| Precision | 65.8% | 66.2% | 64.3% | 46.6% |
| Recall | 66.7% | 65.4% | 69.2% | 97.4% |
| Correct hitter among matches | 49 | 46 | 52 | 71 |
| Wrong hitter among matches | 0 | 0 | 0 | 0 |
| Unknown hitter among matches | 3 | 5 | 2 | 5 |

Outer thresholds: r6 .90, r7 .95, r8 .85, r9 .90, r10 .95. Outcomes vary
substantially by rally: nested r10 finds 11/26 while r9 finds 26/29.

Decision: no clear improvement warranting replacement of the baseline.
At 0.5, 163 proposed events recover 76 real contacts, making this a possible
review-candidate generator, but review burden and missed-event recovery remain
unvalidated. Do not call it a reliable automatic-contact system.

Scope remains exploratory same-match work, with human identities and
contact-derived clip boundaries. Upstream ball detection is not held out;
rally 10 includes ball-training data, and all five rallies informed feature
choice. The nested evaluation does not remove those limitations.

```bash
python3 vision/player_contact_experiment.py --pose-dir pose \
  --predictions-zip ball_contact_review_inputs.zip \
  --hitter-report data/vision/ball_hitter_run1/report.json \
  --baseline-report data/vision/auto_contact_run1/report.json \
  --out data/vision/player_contact_run1
```

The report includes source hashes, per-fold model parameters, threshold-selection
curves, events, and detailed errors. Existing source labels and detector defaults
are unchanged.
