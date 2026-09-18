# Batch 1: targeted visual training expansion

The experiment changes labeled data, retaining the same model architecture,
RGB preprocessing, heatmaps, optimizer defaults and random seed. No trajectory
selector is involved in the first comparison.

Training: the existing complete rally 18 export plus these new source-video windows:

| Rally | Source seconds | Expected samples at 60 fps | Reason from contact labels |
|---|---|---|---|
| 3 | 71–75 | 157 | Smash, drive, counters, speed-up |
| 10 | 299.5–303.5 | 145 | Speed-up and counters |
| 23 | 551–555 | 156 | Fast exchanges transitioning to slow play |

458 new samples, selected before model prediction inspection. These are candidate
examples of difficult play, not visually verified net/paddle/occlusion coverage.
The human labeler should report what conditions actually occur. Sampling remains
native-rate near recorded contacts and every third frame elsewhere. No proposals
are loaded and the human trail starts hidden. Original schema/pages are unchanged.

Rally 25 remains the familiar development comparison, not training data or a clean
holdout. Reserve rally 27 for a fresh same-video check; do not preview it first.
This does not test transfer to another match/camera.

## Generate and label

With the existing virtual environment active, from the repository root:

```bash
python3 vision/ball_label_batch.py --video full_match.mp4.webm
open data/vision/ball_batch1/r3.html
```

Then open r10.html and r23.html in that folder, one at a time. Each page loads
the local full_match.mp4.webm as before. Progress has a batch-specific cache key.
Exports are ball_batch1_r3.csv, ball_batch1_r10.csv, ball_batch1_r23.csv.

- Click visible ball: V. S then click: smear (use its visual center consistently).
- I then click: inferred in-image position during occlusion, if reasonably inferable.
- N: position unknown. O: definitely off-screen/absent.
- O exports visibility N plus presence absent; ordinary N exports presence unknown.
- Do not guess a boundary position when the ball is outside the image.

Export each completed page (and periodically while working). No need to redo
rally 18. Verify source/shown frame agreement as before. Send the three exports
for validation before training. Generated folders are never overwritten.

## Prepare combined data after label validation

Copy the three completed exports to data/vision/ with their names unchanged.
The existing data/vision/ball_labels_v2_r18.csv must be the 153-row final export.

```bash
python3 vision/ball_temporal_baseline.py prepare --video full_match.mp4.webm --labels data/vision/ball_labels_v2_r18.csv data/vision/ball_batch1_r3.csv data/vision/ball_batch1_r10.csv data/vision/ball_batch1_r23.csv --plan data/vision/ball_batch1/plan.json --out data/vision/temporal_batch1
python3 vision/ball_temporal_baseline.py train --data data/vision/temporal_batch1 --all-samples --epochs 90 --out data/vision/temporal_batch1_run1
```

The plan checks source identity, expected frame membership/completeness, and
training-rally membership. Duplicate/overlapping exports are rejected. Each
source CSV is hashed in the prepared manifest. Decoding happens once from the
start of the source, with shared context frames stored once. Prepared images
can occupy substantial memory; archives/checkpoints should stay out of git.

--all-samples avoids silently training just the first 32 examples. --epochs 90
means approximately 90 random-sampling exposures per V/S example, not 90
deterministic full passes. This is similar per-example exposure to the earlier
3,000-step run on 132 targets (~91). More examples means more total updates;
it is not an equal-compute comparison. Training starts from scratch with seed 17.
I, absent and unknown labels are preserved but excluded from localization loss.
No presence head or inferred-position loss is added in this experiment.

The report provides training-set diagnostics by rally; it does not demonstrate
generalization. Preserve temporal_r18_run3/model.pt and use both checkpoints on
identical rally 25 windows with raw predictions. Run the fresh rally 27 comparison
after fixing the experiment settings. All these evaluations remain same-video.

## Verification status

Label-presence preservation, split/frame validation, multi-rally overlay windows,
generated JavaScript syntax and existing tests were checked locally. The original
single-rally checkpoint architecture and command behavior remain supported.
The new batch labeling UI and combined extraction/training require the user's
local video/runtime; no completed training or accuracy improvement is claimed.
