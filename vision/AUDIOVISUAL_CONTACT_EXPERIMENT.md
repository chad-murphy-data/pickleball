# Audiovisual contact experiment

Four fixed audio features record the maximum broadband spectral-flux percentile
within +/-25 ms at delays of +80, +105, +130, and +155 ms from each visual
candidate time. The model chooses weights using training rallies. Contact labels
and outer test rallies do not select an offset. These delays were informed by the
prior diagnostic on the same five rallies, so the experiment is developmental.

The visual features, logistic optimizer, 20 Hz candidate grid, peak separation,
and positive/negative time windows are unchanged from the first learned contact
experiment. Each outer rally is excluded from model fitting. The decision
threshold is selected with inner rally validation on a fixed 0.50–0.95 grid,
maximizing pooled event F0.5. The existing ball-proximity hitter method remains
unchanged.

| Measure | Visual proximity baseline | Visual learned, nested | Audiovisual, nested |
|---|---:|---:|---:|
| Labeled contacts | 78 | 78 | 78 |
| Matched | 52 | 51 | 55 |
| Missed | 26 | 27 | 23 |
| Extra detections | 27 | 26 | 12 |
| Precision | 65.8% | 66.2% | 82.1% |
| Recall | 66.7% | 65.4% | 70.5% |
| Correct hitter among matches | 49 | 46 | 51 |
| Wrong hitter among matches | 0 | 0 | 0 |
| Unknown hitter among matches | 3 | 5 | 4 |

Per-rally audiovisual matched/missed/extra: r6 7/0/0; r7 5/4/3; r8 5/2/0;
r9 19/10/4; r10 19/7/5. Every outer fold selected threshold 0.90.

Audio-only ablation under the same fold structure and nested threshold method:
67 matched, 11 missed, 49 extras (57.8% precision, 85.9% recall). This supports
complementarity: audio raises candidate recall, while visual context removes many
unrelated transients. It does not prove that every accepted sound is a paddle hit.

Decision: best precision-recall result so far and worth carrying forward as the
current development candidate. It is not ready for automatic statistics. Twenty-
three missed contacts remain, and these five rallies informed the features and
audio delay range. Human identities and contact-derived clip boundaries remain
inputs. The upstream ball model is not held out, and rally 10 includes ball model
training frames. Generalization requires untouched rallies and eventually another
match with different acoustics and camera placement.

```bash
python3 vision/audiovisual_contact_experiment.py \
  --audio contact_audio_first330.flac --pose-dir pose \
  --predictions-zip ball_contact_review_inputs.zip \
  --hitter-report data/vision/ball_hitter_run1/report.json \
  --baseline-report data/vision/auto_contact_run1/report.json \
  --out data/vision/audiovisual_contact_run1
```

The output records source hashes, event-level errors, every fold's model and inner
threshold curve. Three tests cover fixed delay alignment, label-independent audio
feature construction, and explicit missing audio. No existing model or source
label is modified.
