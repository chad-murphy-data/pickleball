# Offline candidate selection experiment

Run on the existing three diagnostic windows, with no inference or retraining:

```bash
python3 vision/ball_temporal_select.py --review data/vision/temporal_candidates_r25 --out data/vision/temporal_selected_r25
```

Open r25_window1/selected.mp4, r25_window2/selected.mp4, and
r25_window3/selected.mp4 under the output directory. Playback speed is inherited
from the input report, so these remain quarter-speed. Source timestamps are
unchanged. Original magenta and cyan circles remain; the selected candidate has
a larger hollow yellow ring. No center dots are added. GAP is displayed in the
footer when no candidate is selected. No ring is drawn for a gap.

Each folder has selected.csv with raw and selected positions; report.json
records settings, the source report/checkpoint hash, and counts of rank-1,
alternative, and gap frames. Counts are descriptive, not accuracy measurements.
Inputs and checkpoints are not modified; output folders must be new. Rendering
uses existing annotated MP4 frames (one additional encoding generation), OpenCV
and ffmpeg. --csv-only skips rendering and needs only Python's standard library.

## Scoring

An exact dynamic program minimizes total cost over each complete review window:

- Candidate cost: -0.5 * log(peak_mass), floored at mass 1e-12.
- Movement cost: squared displacement divided by a 32-pixel scale squared,
  adjusted for source FPS and frame spacing (32 pixels per 1/60 second).
- Gap cost: 3 per frame. Entering or leaving a gap costs 0.75.

These are starting heuristics, not calibrated physical limits or probabilities.
CLI options --strength-weight, --motion-scale, --gap-cost, --gap-switch-cost
expose them; the experiment does not tune them against rally 25 labels.
There is no hard maximum speed, no acceleration penalty, no interpolation,
no named-object exclusion, and no court mask. Sharp direction changes are
allowed. A gap resets the motion connection; reacquisition can choose any
candidate. Gap does not mean the ball was absent. Selected candidates still
require visual validation, especially on net tape or player overlap.

This is offline full-window lookahead, not a causal/live tracker. Each window
is optimized independently without outside context. A consistent ad or shoe can
win over an actual ball; a ball missing from all candidates cannot be recovered.
The movement cost may discourage fast true motion. Both outcomes must be checked.

## Review

For the same three windows, check:

1. Does yellow follow the visible ball when magenta jumps away?
2. Does yellow introduce errors where magenta was correct?
3. Does yellow settle on an ad/shoe or remain through actual occlusion?
4. Does it retain sharp bounces, contacts, and fast flight?

Keep the raw comparison visible. Do not score visually plausible interpolations
as detections (this implementation deliberately emits none). Rally 25 is a
development example after repeated inspection; a later untouched rally is needed.

## Validation

```bash
python3 -m unittest discover -s vision -p 'test_ball_temporal*.py'
```

Tests cover candidate recovery after an isolated teleport, retained bounces and
correct raw predictions, empty/weak-candidate gaps, FPS scaling, invalid input,
and agreement with exhaustive enumeration on a small sequence. The local video
render still needs the user's diagnostic clips; no measured improvement is claimed.
