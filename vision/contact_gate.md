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

## Amendment 3, 2026-08-16: the pin-identity scare, resolved — and the
## replay rule

Sequence of events, so the record reads straight: the user found the
scorebug reading 1-0 at the rally-3 pin where 0-0 was expected. An
ultracode workflow (two independent solvers + adversarial verifier +
surface audit; data/vision/pin_realignment.md is the verdict document)
plus a non-circular parity tiebreak (teams alternate shots, so the
user's typed hitters on odd shots ≥3 identify the serving team; row 6 =
5/5 Utah-served, impossible under the shift hypothesis) established:

- **Pin mapping is IDENTITY** — rows 1-16 pin and describe rallies
  1-16. The single anomaly: row 3's pin marks a full-speed broadcast
  REPLAY of rally 3 (the match's first point); the live airing is
  ~56-79 s. The audit tool now drops that pin and seeks the live
  region; 15 pins remain valid.
- **The jitter reference is REINSTATED over those 15 pins** (the
  2026-08-16 note's blanket "pins remain valid" was an overclaim when
  written, and is now true again by measurement, minus rally 3). Guard
  band formula unchanged.
- **Correction to my own 2026-08-16 note**: its "labels x log x pins
  consistent at offset 0" check was partly CIRCULAR — the old tool
  prefilled shots 1-2 from the log per row, so shot-1-hitter matches
  were the log agreeing with itself, and server identity cannot
  distinguish rallies inside a same-server run. The checks that
  actually carry weight are the parity signature (typed hitters,
  shots ≥3), shot-count-vs-duration capacity, and pin-gap arithmetic —
  all of which the realignment used.
- **Replay rule added to the labeling protocol**: a serve whose bug
  shows a LATER score than the header is a replay (replays display the
  replayed rally's END state); the live serve is earlier. This is the
  third face of one recurring lesson — the broadcast stream contains
  non-live segments (replays, filler), and every alignment that ignored
  them (cheer join, scorebug chain, the pinning session) tripped on
  exactly that.
- Label-quality bonus from the tiebreak: the typed sequences are
  parity-PERFECT in every disputed row, so pop-era damage is bounded to
  shot types and occasional count slips — the ordinal labels are
  structurally sounder than the 2026-08-16 note feared.

## Addendum, 2026-08-17 (mid-labeling): the VOD stitches REPLAYS —
## content is the identity authority, not the bug

User observations while timestamping the core 16: rally 3's live
airing was where the realignment put it; rally 4's pin lands past
rally 4; rally 11's banner score (3-2-1) appears NOWHERE near its pin
while the bug there reads 4-2-2 — which is the log's post-rally-12
state, i.e. exactly what a replay of the 29-shot marathon aired after
rally 12 would display. Corrected picture: pin_realignment.md's
mechanism (replay-pinned rows) was right but its scope ("one replay
pin") was low — this condensed VOD stitches FULL-SPEED REPLAYS after
notable rallies, multiple pins sit on them, and at least one live
airing (rally 11's) may not exist in the VOD at all. What SURVIVES
untouched is the load-bearing claim: row k's CONTENT is rally k
(nothing else can hold 29 coded shots but rally 11), which is what the
labels' validity rests on.

Protocol (supersedes the bug-must-match rule where they conflict):
- **CONTENT is the authority.** If the prefilled sequence matches what
  plays, it IS that rally — a later-score bug only means you are on
  its replay. CODE IT THERE: a full-speed replay contains the rally's
  real swings, taps land in the same video pose extraction reads, so
  labels and pose stay window-consistent. Add "REPLAY" to the rally
  note (the record + any coverage-era logic needs to know).
- Skip via ⛔ + note only if the replay is slow-motion or starts
  mid-rally — never stamp slo-mo.
- Window machinery is unaffected: pins/stamps stay monotone, so the
  serve-to-serve bounds hold.

Whiff convention (user, recorded): W was stamped when BOTH partners
went for a ball and only one connected — so a whiff on the same team
adjacent to a contact = both-went-for-it. Handling unchanged (whiffs
are contact=0, excluded from every Gate C denominator) and these are
premium trainer-era events: real swings with no contact, the exact
hard-negative class W exists for.

Correction to the addendum above (user, same evening): there were NO
replays in the VOD — the replay mechanism is NOT established and the
bug/banner mismatches at pins 3/4/11 remain mechanically unexplained
(candidate: a mid-game score correction making the log's reconstructed
chain differ from the broadcast bug; candidate: residual pin drift).
What held, and what Gate C actually rests on: CONTENT-first
identification — the user coded rallies whose action matched the
prefilled sequences, with timestamps in the same footage pose
extraction reads. The ceiling consumes timestamps + hitter TEAM +
pose windows and never touches scores, so the unresolved mechanism has
zero effect on the gate. Do not re-assert the replay story; if the
mechanism ever matters (it doesn't for Gate C), resolve it against the
raw referee log's correction events.

## A/B diagnostic result, 2026-08-17 (rtmpose-balanced, 10 rallies —
## NOT the verdict; interpretation written BEFORE the ViTPose run)

data/vision/contact_ceiling_report_rtm.json. Headline: ceiling-side
45.1% [37.6, 52.7] vs null 30.4% -> KILL on the frozen definition.
n = 162 contacts / 10 rallies (a deviation from the pre-registered 16;
rallies 11-16 unlabeled at run time — r11+r12 alone hold ~55 contacts
and most of the missing fast stratum).

What is INSIDE the number matters more than the verdict line:

- **The stream sees every violent swing**: speed-up 9/9, smash 9/10.
  The pose channel is near-perfect exactly where Gate B's fast-stratum
  fear lived.
- **What fails is small-amplitude shots under the frozen candidate
  RANKING**: dink 41%, counter 32%, other 34%. The budget selects the
  top-2x peaks BY RAW HEIGHT, so small dink/counter peaks lose their
  budget slots to big swings and residual noise. Direct evidence the
  selector, not the stream, binds: ceiling@4x = 64% (+19 pts from
  budget relaxation alone) and ceiling-any = 71%. (Caveat: no null was
  computed at 4x, so 64% has no chance-floor beside it.)
- **The KILL clause's stated interpretation ("the pose stream
  physically does not carry the shots; no classifier can recall what
  is not there") is NOT licensed by this evidence.** The licensed
  conclusion is narrower: a height-ranked 2x-budget candidate
  generator cannot reach 85% coverage at 720p. A trained
  discriminator — the thing Gate C was gating — would not rank
  candidates by raw peak height; the frozen ceiling under-proxies it
  for small motions. This gap between the clause's wording and its
  evidence must carry into any closure write-up.
- Fast stratum 61% on n=41 vs bar 75% — the nearest margin, and
  diluted: 58/162 contacts are typed "other" (some are surely
  counters/speed-ups), and the missing r11/r12 are fast-heavy.
- **Jitter line is VOID**: median |tap-pin| = 19.8 s confirms the v4
  pins were never serve references for these rallies (consistent with
  the unresolved bug/banner mechanism and "no replays"); the printed
  83 s guard band is meaningless. Trainer-era guard band must come
  from refine-pass deltas or the 0.4 s default — never from pins.

Predictions, stated before the verdict run: ViTPose-plus-huge improves
keypoint quality and thus peak-ranking cleanliness, but a 40-point gap
to the 85% bar does not close on model quality; realistic verdict =
KILL or low MIDDLE on the frozen definition, with the attack stratum
staying near-perfect. Pre-registered options unchanged: the verdict
run stands; MIDDLE buys one Sapiens shot; KILL closes THIS instrument.
A candidate-definition v2 (e.g. learned or per-type-aware ranking) is
NOT a knob-turn on Gate C — it would be a new pre-registration,
developed on the now-spent 10 rallies and gated on rallies never used
for development (11-16 + pilot). Independently of any verdict: the
attack-shot layer (smash/speed-up events at ~95-100% coverage) is a
real, immediately usable capability, and the 162+7 timestamped labels
are permanent assets.

## VERDICT RUN, 2026-08-17 (ViTPose-plus-huge, Colab T4, native 60 fps):
## KILL — Gate C closed, final for this footage

Same-file check passed (labels 4820.0 s vs file 4820.0 s). 10 rallies /
162 contacts — the deviation from the pre-registered 16 stands exactly
as in the A/B (rallies 11–16 unlabeled at run time). Report JSON lives
in the user's Drive (`contact_ceiling_report.json`); commit it beside
the RTM one as `contact_ceiling_report_vit.json` when shared in-thread.

    ceiling-side  40.7%  [33.5, 48.4]   (bar 85)
    fast stratum  53.7%  on n=41        (bar 75)
    null          29.1%  p95 34.0       (max 55)
    @4x 59.9%   ceiling-any @2x 66.0%

**The A/B question — "does a stronger backbone put the soft contacts
into the stream?" — is answered NO.** ViTPose-plus-huge ≤ RTMPose on
every summary number: side 40.7 vs 45.1, fast 53.7 vs 61.0, any 66.0
vs 71.0, @4x 59.9 vs 64.2. Per type, ViT gains counters (11/22 vs
7/22) but LOSES attacks (smash 5/10 vs 9/10, speed-up 6/9 vs 9/9).
CIs overlap heavily, so the licensed conclusion is narrow: **the pose
backbone is not the binding constraint on this footage** — not "RTM is
better". (One speculative line, not investigated: ViT-huge may localize
motion-blurred fast arms worse than RTMPose's training mix.)

Corroboration from the exploration layer (swing_explore, same run):
the learned-scorer lift is the SAME SIZE on both streams (+11.8 on ViT,
40.7→52.5; +11.1 on RTM, 45.1→56.2) — learning improves the selector,
the stream sets the level, and both streams plateau at the same soft-
kitchen wall. Oracle-orientation = headline on both (orientation is
solved). Prep-window ablation ≈ 0 on ViT (−0.6 pts).

**Escalation clause resolves: KILL, not MIDDLE → no Sapiens shot.
Gate C is CLOSED, final for this footage.** The A/B addendum's caveat
carries into this closure unchanged: the licensed reading is "a 2×
height-budget candidate ceiling on THIS 720p condensed VOD is too low
for the gated trainer" — the KILL clause's "nothing in the stream"
wording remains unlicensed (attacks near-ceiling in at least one
stream; ceiling-any 66–71%).

Pre-named KILL follow-ups: (1) this closure; (2) footage outreach —
different video, not more labels or compute, is the ball-thread door
and now also the swing-ceiling door; (3) the label-scale route
(`vision/labeling_protocol.md`, split frozen 2026-08-17) feeds a
TEMPORAL model class — that is NOT a Gate C knob-turn; it requires a
fresh pre-registration scored on untouched holdout rallies.

## Measured anomaly, 2026-08-17 (recorded, deliberately NOT adjudicated):
## r9/r10 label spans vs log durations

Surfaced by the user asking "are we sure it gets the right rallies?"
Label-window lengths (extraction log) vs the v4 log durations:
r9 41.1 s vs 17.0 s (+24), r10 47.0 s vs 24.0 s (+23); every other
rally is within +4/−13 (r1 is −13 the other way: window 29.8 vs dur
43.0). 29 contacts inside a 17 s log-rally is physically impossible
(0.6 s cadence including dinks), so at least one of {per-rally log
durations, label rally attribution} is wrong for r9/r10 — and log
durations err in BOTH directions across the set, which points at them.

What this does NOT touch: the verdict. Scoring windows derive from the
labels themselves (internally consistent regardless of identity), and
excising r9+r10 entirely (56/162 contacts) moves learned coverage only
52.5→61.3 (r1–r8) — still far under every bar; KILL is robust. Against
the wrong-rally hypothesis: serve-side mapping agrees with the log on
9/10 rallies (only r4 flips, at zero oracle cost), the ±0.8 s drift
sweep is flat on r9/r10, and the strong rallies score 2.5–2.8× the
shifted-label null — impossible on wrong-rally windows.

Per the postmortem lesson (a frame bug and a detector failure are
observationally identical in label-free diagnostics), this is settled
by HUMAN EYEBALL, not by more reasoning: the Colab debug frames
`pose/debug/r0009_f*.png` / `r0010_f*.png` should show scorebugs
2-3-1 / 2-3-2, and rally 9 on screen should run ~30 s, not ~11 s.
If the scorebugs mismatch → real identity problem → remap before any
TRAINING use of r9/r10. Until checked, r9/r10 carry an asterisk for
training purposes only.

**Addendum 2026-09-01 — the eyeball happened, via the ball passes.**
The owner clicked the ball frame-by-frame through both rallies for the
ball gate (ball_path_r9/r10.csv): r9 is one CONTINUOUS point running
252.60→282.53 clip time ≈ 30 s serve-to-out — the "should run ~30 s"
criterion, met by direct human observation (29 contacts / 30 s ≈ 1/s
cadence, physically sane for a dink rally, where 29 in 17 s was not).
r10's play span 294.30→318.45 = 24.2 s MATCHES its log duration 24.0 —
its 47.0 label window was extraction padding, not play; r10 was never
actually anomalous. r10 identity is further pinned to the main VOD by
the owner's own bounce checks (clip offset 292.7 maps 296.50→4:46 etc.,
verified at 4:58.9 / 5:00.6 / 5:15.0 during the 2026-08-31 bounce
dispute). Verdict: the log-side per-rally DURATION field is the wrong
member of the pair for r9 (errs both directions across the set, as the
original note suspected). Remaining formality before TRAINING use: the
r9 scorebug still (expect 2-3-1); everything else about the asterisk is
resolved.
