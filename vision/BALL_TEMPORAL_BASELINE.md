# Temporal localization smoke test

This isolated experiment leaves existing tracking code unchanged. Five RGB frames
predict the middle frame's ball heatmap. The default run memorizes 32 V/S examples
from rally 18. Results are training-set diagnostics, not generalization evidence.
There is no presence head yet. I/N labels are retained but have no localization loss.

User clarification: rally 18 sample indices 40–51 (page samples 41–52), source
frames 27414–27447, have N meaning ball absent. The other N means position unknown.
The original CSV remains unchanged; the manifest adds a presence field.

Rally 17 is deliberately quarantined: v2.0 exported timestamps could be stale
because callbacks were not tied to completed seeks. Deduplication alone does not
prove pixel/label alignment. Do not call these labels verified without reviewing
extracted frames. This is more conservative than the initial recovery proposal.

## Mac setup (from repository root)

Save the final rally 18 CSV as data/vision/ball_labels_v2_r18.csv first.

```bash
python3 -m venv .venv-ball
source .venv-ball/bin/activate
python3 -m pip install torch numpy opencv-python
python3 vision/ball_temporal_baseline.py prepare --video full_match.mp4.webm --labels data/vision/ball_labels_v2_r18.csv --out data/vision/temporal_r18
python3 vision/ball_temporal_baseline.py train --data data/vision/temporal_r18 --out data/vision/temporal_r18_run1
```

Preparation decodes sequentially from the beginning to avoid seek ambiguity.
This baseline assumes constant-frame-rate, zero-origin source indexing; compare
extracted frames against the labeler before trusting alignment on other videos.
Frames are resized to 640x360; the output heatmap is 160x90. Errors are reported
in original-video pixels. CPU is supported; Apple MPS/CUDA are selected when available.
No model download or paid compute is required. Existing output folders are never
overwritten. Use a new output name for another run.

Return report.json and predictions.csv from the run folder. model.pt is a local
checkpoint; do not commit checkpoints or prepared frame archives. Runs are offline,
use future context, and are not suitable for claiming causal live performance.

## Verification and limitations

```bash
python3 -m unittest discover -s vision -p test_ball_temporal_baseline.py
```

Label validation and syntax were tested by the assistant. PyTorch and the source
video were unavailable there, so numerical training and video extraction still
require a local smoke test. No achieved training accuracy is claimed. Begin with
32 samples; --limit 132 includes all currently labeled V/S examples. No holdout
split, presence evaluation, contact attribution, or physical speed inference is
implemented in this initial experiment.

## Saved-checkpoint visual review

Run from the repository root with the existing virtual environment activated:

```bash
python3 vision/ball_temporal_overlay.py --video full_match.mp4.webm --checkpoint data/vision/temporal_r18_run3/model.pt --rallies 18 19 --out data/vision/temporal_review_r18_r19
```

This loads the existing checkpoint without training or changing it. The model
architecture is shared with training and preserves the original checkpoint keys.
Requires ffmpeg with libx264 (already typically available alongside ffprobe).
Outputs r18/overlay.mp4, r19/overlay.mp4, per-rally predictions.csv and report.json.
The MP4 files use H.264 for QuickTime and have no audio.

Magenta hollow circles are predictions; smaller green hollow circles are V/S
human labels, orange circles are inferred labels. Marker centers are transparent
and all text is below the image so the ball remains visible. No smoothing, hiding
low scores, or interpolating labels is applied. Off-screen predictions remain
visible; peak mass is not a calibrated ball-presence probability.

Rally 18 includes every frame between its first and last labels, including gaps
and explicitly absent samples. Distinguish training targets from frames that
only appeared in five-frame input context. Rally 19's review window comes from
accepted contacts plus one second on either side, not verified rally boundaries.
It is same-video qualitative testing with no human localization score. The report
counts any training-context overlap, including neighboring input frames.

Look first at whether the green circles match the ball in rally 18. That checks
image/label alignment. Then watch where magenta circles go on off-screen frames
and in rally 19. Return report.json, both CSVs, and the videos if practical.

The decode-window, training-exposure and missing-label logic has automated tests.
End-to-end inference/encoding still needs local verification with the checkpoint
and source video; these are not available in the assistant workspace.
