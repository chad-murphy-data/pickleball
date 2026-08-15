# Gate C — the pose-stream contact ceiling (pre-registered 2026-08-15)

**Status: OPEN — bars frozen in this file BEFORE any timestamped label
exists. Written the same day the program closed, on an explicit user
decision.**

## Provenance, so the contradiction with POSTMORTEM.md is not a mystery

`vision/POSTMORTEM.md` (2026-08-15, morning) closed the vision program:
Gate A (ball) killed on physical findability, Gate B (swing∧pop proxy)
killed on a sound measurement frame. Later the same day the user
reopened **the swing thread only**, dropping the zero-training constraint
that every prior instrument ran under: *"let's drop zero-training — what
are our options for me giving human-coded training data to it?"*

What changes: human-labeled **contact times** become available, which
v1 never had (the 203 labels are ordinal — order, hitter, type — because
the audio pop train was supposed to carry timing, and audio's premise
failed). What does not change: the ball thread stays closed (training
data cannot exist where a human cannot see the ball — 36% of in-play
frames); the footage is still the condensed 720p Chicago VOD; the
championship-court sample bias and the n=4 freeze-out caveat are
permanent.

Why a new gate is justified rather than a re-litigation of Gate B:
v1's measured recall (0.442) was computed on events required to
coincide with an audio pop within 0.25 s, and pops were shown to be
uncorrelated with true shots (r ≈ 0.0–0.2; gated survivors coincide at
base rate). Coincidence with an uncorrelated signal is a near-random
thinning, so 0.442 ≈ (pose recall) × (chance of a nearby pop) — the
first factor was never measured alone. The tell: serve recall was 33%
on the sport's most stereotyped, stationary, full-arm motion. Gate C
measures that first factor directly, with zero training, before anything
else is built.

## The question

Does the pose stream contain a **candidate event** near enough to each
true contact that a downstream classifier could recover it? A classifier
can only fire where candidates exist; this gate bounds achievable recall
before any training. It is the same "measure the wall before building
the ladder" move that closed Gate A cheaply (findability before
fine-tune) — pointed at the instrument we now propose to train.

## Pre-registered instrument (vision/contact_ceiling.py)

- **Signal**: per-track **torso-relative** wrist speed on
  **identity-continuous** tracks (`vision/pose_extract.py`: greedy-IoU
  association, track death after a 0.6 s gap). Wrist velocity minus
  hip-center velocity, normalized by torso scale (shoulder-mid to
  hip-mid distance; fallback 0.35 × box height), max over the two
  wrists, 3-point smooth. This is the v2 feature the Gate B autopsy
  specified — absolute wrist speed registers locomotion; torso-relative
  cancels it.
- **Candidates**: local maxima, strongest-first refractory 0.25 s,
  budgeted per rally per image-side to **mult × that side's labeled
  contacts** (ranked by peak height). Primary mult = **2×**; 4×
  reported as secondary. The budget is what makes "a peak exists"
  falsifiable — unbudgeted, dense noise peaks cover any label set.
- **Match**: a true contact is COVERED if a surviving candidate on the
  **hitter's image side** lies within **±0.30 s** (ceiling-side; the
  headline). Ceiling-any (any track) reported as diagnostic.
- **Team→image-side mapping** (teams switch ends; the log does not say
  which end): per rally, take whichever of the two mappings scores more
  covered contacts — and apply the IDENTICAL maximization to every null
  draw, so the selection optimism is absorbed by the null, not by the
  verdict.
- **Null**: circular time-shift of each rally's label set by a uniform
  offset in [3 s, L−3 s] (window length L; [2, L−2] when L < 8 s),
  wrapped within the window, 200 draws, identical budgets and mapping
  maximization. Reported as mean and p95 of the null ceiling-side.
- **Denominator**: labeled contacts only (`contact=1`); whiffs
  (swing-and-miss, `contact=0`) are excluded from recall and reported
  separately. Fast stratum = `{speed-up, counter, smash}` — frozen from
  `vision/swing_score.py` (FAST_TYPES), unchanged from Gate B.
- **Timing ground truth**: `t_refined_s` where present, else `t_tap_s`,
  from `data/vision/contact_labels_chicago0725.csv` (the new
  instrument's export). Scored rallies are the **core 16** (rally_cum
  1–16, womens game), whose serve instants were already hand-pinned on
  this same video — the measurement frame is airtight by construction,
  which was Gate B's hard-bought lesson.

## Pre-registered bars (verdict printed by contact_ceiling.py)

At mult = 2×, tolerance ±0.30 s, ceiling-side:

- **PROCEED**: overall ≥ **0.85** AND fast stratum ≥ **0.75** AND
  null mean ≤ **0.55**.
- **Fallback (one shot, only if null mean > 0.55)**: tolerance tightens
  to ±0.20 s and the same bars apply. No other knob moves after data.
- **KILL**: overall < **0.75** — the pose stream physically does not
  carry the shots at 720p; no classifier can recall what is not there;
  the thread closes again, this time with the confound removed.
- **MIDDLE** [0.75, 0.85): judgment zone. Default: proceed only if the
  fast stratum cleared 0.75 (the stratum that killed everything else),
  and say so out loud in the write-up.

Jitter calibration rides along for free: the shot-1 taps on the core 16
are re-marks of serves already pinned in
`rally_windows_chicago0725_v4.csv`, so tap−pin deltas measure the
labeler's timing noise. **Training guard band = max(0.4 s, 3 × p95
|tap − pin|)** — set from measurement, not assumption.

## Sequencing (what is deliberately NOT built yet)

1. **Label**: re-tap the core 16 with timestamps
   (`data/vision/contact_audit_chicago0725.html`, pass 1; ~1 evening).
2. **Extract**: `python vision/pose_extract.py --video full_match.mp4
   --rallies labeled` (rtmpose-balanced, the pre-registered backend —
   see the amendment below; overnight-class on CPU for the core 16,
   minutes on a GPU with `--device cuda`; `--fast` is a smoke preset
   and NOT the gate).
3. **Gate**: `python vision/contact_ceiling.py` → verdict against the
   bars above.
4. **Only on PROCEED**: label the pilot corpus (~40–60 rallies across
   all four games, longest first), and only then build the trainer
   (whole-skeleton temporal windows, guard-banded negatives, matched
   same-frame negatives from the three non-hitters, whiffs as their own
   class). The trainer is deliberately unbuilt today: building it before
   the wall is measured is the exact anti-pattern Gate A avoided by
   skipping the fine-tune.
5. **Final gate (Gate D, if the pilot's learning curve earns it)**: v1's
   original bars — 75% overall recall / 60% fast / 90% precision,
   side-alternation junk line 0.45 — scored on FRESH rallies never used
   in training. The core 16 and the pilot corpus are train/dev forever;
   nothing gates on data that tuned it.

## Prior odds, stated before data

The record priced a third-instrument revival at ~1-in-5 with hand-tuned
features. With trained discrimination and the random-thinning argument
above: pilot-shows-signal near even, ultimately-clears-Gate-D ~1-in-3.
Writing these down now so the eventual verdict can be checked against
what we believed going in.

## Amendment 2026-08-15 (same day, STILL PRE-LABEL): pose backend

Made while zero timestamped labels exist, so the instrument remains
frozen-before-data; nothing below was chosen with any Gate C output in
hand.

- **Primary backend = RTMPose via rtmlib, `balanced` mode** (top-down
  two-stage: detector → per-person crop → pose), wired as
  `pose_extract.py --backend rtmpose` (the default). Rationale: top-down
  normalizes every person crop to a fixed input size, so the ~40 px far
  pair stops being small to the keypoint head — exactly Gate B's failure
  surface — and RTMPose-m sits ~15 COCO-AP above the yolov8s-pose the v1
  probe used (~75 vs ~60), with the gap widest on small people. Install
  verified: `pip install rtmlib onnxruntime`, ONNX models auto-download,
  returns the same 17 COCO keypoints the npz schema and scorer already
  use.
- **Secondary backend = yolov8s-pose (`--backend yolo`), diagnostic
  only.** It may be run for A/B context; it is NEVER the verdict, and
  the better of two backends is NEVER selected after seeing results —
  that disjunction is exactly the forking path this amendment exists to
  foreclose. The gate verdict comes from rtmpose-balanced, full stop.
- `contact_ceiling.py` warns when the pose meta records any other
  backend/mode; `--fast` (lightweight models) is a smoke preset, not the
  gate.
- MediaPipe was considered and rejected without being run: BlazePose is
  a single-person, near-field pipeline (fitness-app shape); our frame
  has four-plus people with the far pair at ~40-120 px. It was never
  used anywhere in this project.
