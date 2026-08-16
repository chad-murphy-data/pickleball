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
   --device cuda` on the GPU box (ViTPose-plus-huge at native fps — the
   verdict instrument per Amendment 2 below, which supersedes
   Amendment 1's RTMPose-primary). Optionally also `--backend rtmpose
   --out-dir data/vision/pose_rtm` for the production-spine A/B.
   `--fast` is a smoke preset and NOT the gate.
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

## Amendment 2, 2026-08-15 (same day, STILL PRE-LABEL): gold-standard
## instrument — user directive

User, verbatim intent: *"I don't want to kill this because we took an
easy route... I'm relying heavily on you to get gold standard of what is
reasonable rather than reaching for the quickest, easiest tool."* Made
while zero timestamped labels exist. The principle that follows: the
gate is a ONE-SHOT ~15k-frame measurement, so inference speed is nearly
irrelevant to it — the verdict instrument must be the strongest model
that practically runs, and production convenience is a separate,
post-gate concern. Amendment 1's RTMPose pick conflated those two
questions; this amendment separates them.

- **VERDICT instrument = ViTPose-plus-huge** (top-down, ~81 COCO AP —
  the strongest practical 2D pose model; HF transformers path, RT-DETR
  person boxes, court-gated before pose, `pose_extract.py --backend
  vitpose --device cuda`). Install + inference of both the base and the
  MoE `plus` code paths verified in a fresh environment before this
  amendment was written. Needs the GPU box; that is a feature of the
  choice, not a bug.
- **RTMPose-balanced is demoted to PRODUCTION SPINE**: it is what a
  500-rally pipeline would run at scale, so its ceiling is REPORTED
  next to the verdict (named A/B, `--out-dir data/vision/pose_rtm`) to
  price the production gap — but it is never the verdict.
  yolov8s-pose remains a diagnostic. The better-of-N-backends
  disjunction stays foreclosed: the verdict number comes from
  ViTPose-plus-huge, decided before any label existed.
- **Candidate signal widened to the whole arm**: max of torso-relative
  speed over wrists AND elbows, no coefficients. Motion blur destroys
  the most distal joint first and a swing moves the whole arm; wrists
  dominate whenever visible, elbows carry through blur frames. The
  selftest plants blur-degraded contacts that only the elbow channel
  can cover.
- **Native fps**: extraction runs at the VOD's native frame rate
  (auto-detected, `--fps 0` default) — no temporal subsampling on a
  measurement whose object is ~100 ms events.
- **No super-resolution, ever, for measurement**: SR hallucinates
  pixels, and claims scored on hallucinated pixels are unfalsifiable —
  the same failure class that poisoned the auto-label fine-tune. Native
  720p in, verdict on native 720p.
- **Escalation ladder, pre-named to foreclose post-hoc reaching**: if
  the verdict is MIDDLE [0.75, 0.85), ONE escalation is permitted —
  Sapiens-1B pose (Meta's 300M-image-pretrained model, the exotic tier
  above ViTPose) — and its result is final either way. A KILL at
  ViTPose-plus-huge is final for this footage: a >10-point gap to the
  bar is not a model-size gap at 720p; it is the footage, which is the
  postmortem's stated only door. MediaPipe is not on the ladder
  (single-person, near-field — wrong shape, never used in this
  project).

## Labeling protocol note, 2026-08-16: prefill demoted — the pop-counting
## era reached the labels

During the first timestamped session the user found rallies whose prefill
diverged from the screen mid-rally ("the server is correct, but the rest
aren't"; rallies 3-4, while 1-2 tracked). Data-side audit before touching
anything: labels × log × pins × page payload are all CONSISTENT at offset
0 — all 16 coded shot-1 hitters equal the log server (shifted alignments
score 8/14 at ±1 vs 14/14 at 0), pins are monotone with plausible gaps,
and the user confirms the correct server at the pin instants. So rally
IDENTITY, the serve pins, and the time base are sound; the divergence is
in the old ORDINAL sequences themselves. Mechanism: the v1 audit tool
anchored shots 1-2 from the referee log and advised counting the rest
"at 0.5× with sound on — the pops make it easy" — and the pops were later
proven uncorrelated with shots. The audio premise failure degraded the
answer key, not just the detector it graded.

Consequences:
- **The prefill is a suggestion, not an authority.** Protocol: scorebug-
  verify shot 1 (the header shows the log start_score; the on-screen bug
  must match — the score IS the rally's identity), ⏎ only while the video
  visibly tracks the prefill, "✕ prefill" at the first divergence and
  keys 1-4 from there. The screen always wins. Tool updated accordingly
  (scorebug banner, ✕ prefill toggle that materializes already-stamped
  types, auto-pause clamped to the next rally's serve).
- **Gate C is unaffected**: its ground truth is the NEW timestamped CSV,
  which depends on nothing from the old sequences; windows derive from
  the new serve stamps. Per-rally identity is now VERIFIED via scorebug
  rather than inherited — an upgrade to the measurement frame.
- **The jitter reference (pins) remains valid** — pins were user-
  confirmed correct at the serve instants.
- **Historical caveat, recorded not re-litigated**: shot_labels_
  chicago0725.csv's reliable content = rally identity + shots 1-2 +
  approximate counts/type mix; mid-rally order and per-shot types carry
  errors at an unknown rate. Gate B's 0.442 recall / 0.871 precision
  were scored against those sequences — one more unquantified reason
  0.442 was never the pose ceiling. Once the new labels cover the core
  16, rescoring Gate B against corrected sequences is a cheap curiosity
  for the record; it carries zero decision weight (the thread was
  already reopened on stronger grounds).

Addendum to the 2026-08-16 note — **same-file check is now mechanical**:
the labels export stamps the loaded video's name and duration
(`video_name`/`video_dur_s` on shot-1 rows + the meta JSON), and
`pose_extract.py` refuses to extract when the file it is given differs
by >2 s from the stamp (`--force-video` to override, only when certain).
Rationale: a tap IS the video's own clock, so timestamped labels can
only desync from the footage if the FILE changes between labeling and
extraction — the one sync failure mode taps cannot self-detect, now
checked by machine instead of memory. This also answers, for the
record, the user's standing worry that past kills were sync artifacts:
the kills that stand rest on sync-proof evidence (Gate B = detector-
stream side-alternation + the label-free side-repair autopsy; Gate A =
direct human observation of in-play frames), the one sync-corrupted
verdict (the first 0.467 swing kill) was caught and retracted at the
time, and the sync-vulnerable number that survived in folklore (0.442
recall) is precisely the one Gate C refuses to inherit.
