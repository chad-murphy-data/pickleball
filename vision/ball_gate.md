# Ball Gate — pre-registration for the constrained ball tracker

**STATUS: DRAFT (2026-08-30).** Not frozen. No tracker code may exist
until the user reviews this document and explicitly says "freeze it",
after which a dated FROZEN stamp is committed and the bars below become
immutable. Building first is the anti-pattern every gate in this
project exists to prevent.

## Why this reopens a closed question, and what changed

Automated ball detection was closed 2026-08-15 on the ball-visibility
measurement: 64% in-play findability, whole CI under the 0.8 kill
line. Two things changed, both measured since:

1. **The 64% was an isolated-frame instrument.** The user's dense
   rally-1 ball pass (683 frames, blink-compare, frame-by-frame)
   measured 92% locatable when frames are labeled contiguously —
   continuity, not per-frame appearance, is what makes the ball
   findable. The oracle test (make_ball_audit.py, frozen constants,
   circular-shift null) PASSED on that stream.
2. **The constraint side now exists.** Contact times (user labels),
   hitter identity (verified features, 76/82 on 84 contacts), player
   floor positions, court DLT, and the piecewise ballistic-drag arc
   model (court3d.py) with net-crossing / containment / player-anchor
   priors — i.e., a weak detector no longer has to carry the problem
   alone. This is the recursive ball↔hitter loop's ball half.

The old closure stands for what it measured: a STANDALONE per-frame
detector on this 720p VOD. What is licensed for testing here is a
CONSTRAINED tracker. A KILL below closes automated ball tracking on
this footage again, this time with the constraints on the table —
the next reopening would need new footage, not a new argument.

## The instrument under test

Two stages, both gated behind the freeze:

- **Candidate stage**: per-frame ball candidate proposals with recall
  prioritized over precision (classical motion/blob methods and/or a
  lightweight detector; multiple candidates per frame allowed; "no
  candidate" allowed).
- **Decoding stage**: piecewise ballistic arcs (court3d's linear-drag
  model) selected/fit over candidates, anchored on labeled contact
  times and hitter positions, with the established priors
  (net-crossing per side vote, court containment, player-geometry
  anchors, bounce continuity). Output: per-frame (t, x, y) in the
  1280x720 pose frame plus arc segmentation, and the 3D lift via
  court3d.

Anchors come from LABELS during gating (contact taps + track-assign
clicks). Swapping in model-derived anchors is out of scope for this
gate and would be a separate registration.

## Ground truth and split

User ball passes via make_ball_audit.py (classes: V visible click,
S smear click, I inferred/hidden, N not visible).

- **Spent / development**: rally 1's existing ball path
  (data/vision/ball_path_r1.csv) — it tuned the 3D reconstruction and
  is free for development without restriction.
- **Train**: ball passes on rallies **6 and 7** (manual contact
  anchors exist; short rallies, cheap passes). Tuning, iteration, and
  diagnostics unrestricted here.
- **Holdout**: a ball pass on rally **8**, SEALED — the user labels
  it and commits it, and no tracker run touches it until grading.
  One grading run, ever, per the consequences below. (Rallies 9/10
  may be added as secondary sealed rallies later under the same
  rules; their training asterisk does not affect use as holdout.)

The user labels holdout rally 8 BEFORE the first tracker run on any
rally, so the seal is real. More train rallies may be added at any
time (user's call — ball passes are cheap); the holdout set can only
grow, never shrink or swap.

## Scoring (frozen at freeze time)

Panel: all frames of the holdout rally from first labeled contact to
0.5 s after the last, at the label pass's frame rate.

- **Scoreable frames** = V frames (primary) and S frames (secondary).
  I and N frames are EXCLUDED from scoring — a claim where no human
  can see the ball is unfalsifiable, the exact failure that poisoned
  the 2026-08 auto-label fine-tune. Coverage on I/N is reported as a
  diagnostic only.
- **Hit** = predicted position within **25 px** of the user's click
  (V), or within the smear's clicked position + 40 px (S).
- **Primary metric** = hit rate on V frames.
- **NULL (the bar that matters)**: the anchors-only path — piecewise
  LINEAR interpolation in pixel space between the hitter's wrist
  position at each labeled contact (same anchors the decoder gets, no
  pixels, no physics). The tracker must beat what its own anchors
  give away for free. A ballistic-arc-through-anchors variant (still
  candidate-blind) is the second, harder null, reported alongside.

### Bars (draft numbers — user may adjust before freeze; immutable after)

- **PASS**: V hit rate ≥ **70%** AND ≥ linear-anchor null + **15
  points** AND ≥ candidate-blind ballistic null + **10 points**.
- **KILL**: V hit rate < linear-anchor null + **5 points**.
- **MIDDLE** (between): ONE further train-only iteration is allowed,
  then one re-grade on a NEWLY labeled sealed rally (not rally 8,
  which is burned by the first grading). Its result is final.

### Consequences (pre-committed)

- **PASS** → the constrained tracker is a licensed channel: it may
  feed the 3D replay, ball-derived stats, and (via the separate
  temporal-gate amendment) the temporal model.
- **KILL** → automated ball tracking on this VOD is closed AGAIN,
  now with constraints tested. Recorded in STATUS.md; next door is
  new footage. Human ball passes remain licensed regardless.
- **MIDDLE exhausted without PASS** → same as KILL.

## What may be built before freeze

Nothing of the two gated stages. Existing instruments (court3d,
make_ball_audit, verify_hitter_features, labeling tools) may be
maintained. Writing this document is not building.

## Amendment rule

Before freeze: anything may change. After freeze: amendments follow
the contact_gate.md convention — dated, appended, and only ever
tightening or clarifying; bars and the holdout seal never loosen.
