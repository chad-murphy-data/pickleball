# Native versus resized condition audit

Generate after the two evaluation-label CSVs are copied into data/vision:

```bash
python3 vision/ball_condition_audit.py --video full_match.mp4.webm
open data/vision/ball_condition_audit/review.html
```

Defaults use ball_eval1/plan.json, ball_eval1_r25.csv, ball_eval1_r27.csv and
temporal_batch1_review predictions. Source hashes must match the saved evaluation
plan. No model inference or training occurs. The generator sequentially decodes
from frame zero, using OpenCV already installed in the virtual environment.

The page includes 24 items: six temporally spread >20-pixel misses and six
<=10-pixel controls per rally, where available. Controls preferentially match the
evaluation cohort, then the nearest available time. They are not yet matched
on visual conditions, which are unknown before this review. All selected items
were originally labeled V/S. I/N are not silently treated as visual misses.

The native crop is 256x256. The model-view crop is 128x128 AFTER resizing the whole
1280x720 frame to 640x360 with the same linear interpolation used in training.
Crop origins are aligned to even source coordinates; both depict the same area.
Both display at 384x384 with nearest-neighbor scaling; no sharpening, enhancement,
or markers are applied to these crops. The center full-frame context has only a
hollow ring at the HUMAN label. Each crop pair can step through offsets -2..+2,
matching the five input frames. Classify the center, using neighbors as context.

Conditions can overlap (net tape, net mesh, player, paddle). Open flight and
Uncertain are exclusive choices. Also select native visibility and whether the
ball remains distinguishable after resizing. Corrections to visibility are audit
annotations only; the original evaluation labels and scores are not overwritten.
The page hides prediction location, error magnitude, miss/control status and pair
identity. An external manifest.json preserves these for later analysis.

Click Save and next, export periodically and at the end. The page is self-contained
and works locally without a server; browser progress is a convenience, not a
backup. The export is ball_condition_review.json. Return that file and manifest.json
from the output folder. The manifest links each review to selection provenance.

This outcome-balanced case/control sample diagnoses failure mechanisms; it cannot
estimate population error rates by condition. Conditions must be interpreted in
light of the prior systematic/challenge sampling too. Human legibility after
resizing is evidence about information loss, not proof of what the network can
learn. Net-tape failures require both image evidence and model-error analysis.

Tests check crop geometry, sample uniqueness/balance, inferred-label exclusion,
and JavaScript syntax. End-to-end image extraction and browser review still need
the local source video. Do not commit generated image-heavy pages to git.
