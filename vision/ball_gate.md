# Ball Gate — pre-registration for the automated ball tracker

**STATUS: DRAFT v2 (2026-08-30).** Not frozen. No tracker code may
exist until the user reviews this document and explicitly says
"freeze it"; a dated FROZEN stamp is then committed and the bars
become immutable. v1 (same day) graded the tracker against
anchors-only interpolation with contact taps given — retired after
discussion because it tested the regime where the ball is least
needed. v2 encodes the user's actual research question:

> **"Can we track the ball?"** — specifically, well enough to
> replicate the 3D reconstruction (court3d / the replay) without
> hand-labeled ball clicks or contact taps.

That question is the strict version: the 3D model's skeleton is the
arc structure and arcs are born at contacts, so a path good enough to
rebuild the replay necessarily recovers the turns — the contact-time
channel (dead since Gate C) comes along as a byproduct, scored by the
instrument that killed it.

## Why this reopens a closed question, and what changed

Automated ball detection was closed 2026-08-15 on isolated-frame
findability (64%, CI under the 0.8 kill line). Since then, measured:
(1) the user's dense rally-1 ball pass found the ball in 92% of
frames — continuity, not per-frame appearance, is what makes it
findable, and the oracle test (make_ball_audit.py, frozen constants,
circular-shift null) PASSED on that stream; (2) the physics side
exists (court3d piecewise ballistic-drag arcs, net-crossing /
containment / player-anchor priors) so a weak per-frame detector no
longer has to carry the problem alone. The old closure stands for
what it measured — a standalone per-frame detector. A KILL below
closes automated ball tracking on this footage again, with the
physics constraints on the table; the next door is new footage.

## The instrument under test

Gated behind the freeze: a candidate stage (per-frame ball proposals,
recall over precision, multiple or zero candidates per frame allowed)
plus a decoding stage (piecewise ballistic arcs selected/fit over
candidates with the established priors; the decoder chooses its own
arc boundaries = its claimed contacts). Output: per-frame (t, x, y)
in the 1280x720 pose frame, arc segmentation, and the 3D lift.

**Inputs at grade time** (deployment-realistic, nothing hand-labeled
about the ball, the contacts, or the people):
- the VOD and the rally window (referee logs supply these),
- the serve pin (logs carry serve timing),
- court calibration (one-time, 11 clicks per venue/camera — the only
  persistent human input; per camera, not per rally),
- **the person channel, FULLY AUTOMATED** (amended 2026-08-30, user
  question "can the gate include automated person labeling?"): pose
  tracks as extracted, with junk/crowd filtering, fragment handling,
  and side classification done by the pipeline itself. Occlusion
  reasoning and the player-anchor prior use these freely. Note the
  ball physics needs anonymous bodies-with-sides only — NAMES are
  not required for this gate (identity is the hitter-features
  channel's problem, not the ball's).
- NOT the user's ball clicks; NOT the contact taps; NOT the user's
  track_assign clicks.

Contact taps, ball clicks, and track_assign clicks on TRAIN rallies
may be used freely during development, including anchored diagnostic
runs. Surviving the raw track mess (crowd tracks, fragments,
garbage-keypoint rows) unaided is part of what is being graded.

## Ground truth and split

User ball passes (make_ball_audit.py; classes V/S/I/N) and, for the
replication check, the court3d reconstruction built from them.

- **Spent / development**: rally 1's ball path
  (data/vision/ball_path_r1.csv) and its reconstruction — free for
  development without restriction.
- **Train**: ball passes on rallies **6 and 7** (short rallies, cheap
  passes; manual contact taps already exist there for anchored
  diagnostics). More train rallies may be added any time.
- **Holdout**: a ball pass on rally **8**, SEALED — labeled and
  committed BEFORE the first tracker run on any rally, untouched
  until grading, graded once. The holdout set can only grow, never
  shrink or swap. (Rallies 9/10 may join as later sealed rallies;
  their training asterisk does not affect holdout use.)

## Scoring (frozen at freeze time) — three checks, one graded run

Panel: the sealed rally from first contact to 0.5 s after the last.

1. **PATH — V-frame hit rate**: fraction of the user's
   visible-ball (V) frames where the predicted position is within
   **25 px** of the click. S (smear) frames scored at 40 px,
   reported separately. I and N frames are NEVER scored — a claim
   where no human can see the ball is unfalsifiable (the auto-label
   poison lesson); coverage there is a diagnostic only.
2. **TURNS — the oracle battery, unchanged**: the tracker's decoded
   path is scored by make_ball_audit's frozen instrument (turns
   within ±0.15 s of the user's contact taps, recall bar 0.80,
   circular-shift null at 95%, same constants: TURN_DEG 25.0,
   MIN_SEP_S 0.25, N_NULL 1000). The human-labeled path passed this
   battery; the tracker must pass the same test without the human.
   This carries the null that matters — placement beyond rhythm.
3. **REPLICATION — the deliverable**: court3d run on the tracked
   path vs court3d run on the user's path, same anchors policy on
   both sides (player-geometry priors from AUTOMATED person data; no
   tap anchors and no track_assign clicks on the tracked side): median 3D distance between matched impact
   points ≤ **3.0 ft** (the measured monocular floor is ~1.3-1.5 ft
   per reconstruction), all decoded arcs satisfy the net-crossing
   check, and bounce count within ±1 of the human-path
   reconstruction.

### Readiness rule (the seal cannot be spent on an unfinished pipeline)

The sealed rally is graded only after the pipeline PASSES the full
three-check battery on at least one TRAIN rally (scored against that
rally's labels, which are of course not inputs). Train iteration is
unlimited and free — the outer-space-arcs class of bug (court3d's
own history: fixed by two iterations that a premature verdict would
have foreclosed) gets caught there, never on the seal.

### Bars

- **PASS**: all three — V hit rate ≥ **70%**, oracle battery PASS,
  replication check met.
- **FAIL — attribution required, see the autopsy**: V hit rate
  < **40%**, or the decoded path fails the oracle battery's
  circular-shift null (its turns carry no placement information
  beyond rhythm).
- **MIDDLE** (anything between): ONE further train-only iteration,
  then one re-grade on a NEWLY labeled sealed rally (rally 8 is
  burned by the first grading).

### The autopsy (pre-registered failure attribution)

**This gate cannot kill the ball CHANNEL: the oracle test already
proved the information exists in this footage** (a human extracted
the path and it carried the timing; that result stands regardless of
what any tracker does). A failing grade is therefore a verdict about
THIS BUILD, and must be attributed before any consequence attaches.
After any non-PASS grade, these arms run on the now-spent sealed
rally (grading is over; the seal's purpose is spent):

1. **Oracle substitution** — the user's ball clicks fed through the
   SAME decoder as the candidate stream. Fails even on perfect
   candidates → **DECODER** bug; the candidate stage was never
   tested.
2. **Person ablation** — re-run with train-quality person data
   (user track assigns / hand-curated tracks). Scores recover →
   **PERSON** channel is the binding constraint. Consequence: fix or
   kill the person module; the ball question is UNDETERMINED.
3. **Frame checks** — clock alignment, window integrity, label
   sanity. A broken frame → verdict **VOID** (the swing-proxy
   lesson: a frame bug and a detector failure are observationally
   identical in label-free diagnostics).
4. Only the remaining case — decoder passes on oracle candidates,
   person channel exonerated, frame sound, but the candidate stage
   cannot find the ball in pixels — is a **CANDIDATES** verdict, the
   one outcome that speaks about automated ball extraction on this
   footage.

### Consequences (pre-committed)

- **PASS** → the tracker is a licensed channel: automated 3D
  replays, ball-derived stats, and recovered contact times for
  unlabeled rallies; feeding the temporal model still goes through
  the separate temporal-gate amendment.
- **DECODER / PERSON / VOID attribution** → fix the named component;
  a re-attempt is licensed under a short dated addendum naming what
  changed, graded on a NEWLY labeled sealed rally. The user is the
  rate limiter — each re-attempt costs one fresh ball pass — so
  retries are priced, not forbidden, and a component failure never
  converts into a false kill of the ball channel.
- **CANDIDATES verdict** (confirmed on the MIDDLE re-grade if one
  was available) → automated ball extraction with this instrument
  class is closed on this VOD, recorded in STATUS.md NEXT TO the
  standing oracle result: the information is there, this machine
  couldn't get it. The next door is a different candidate approach
  under a fresh registration, or new footage. Human ball passes
  remain licensed regardless.

## What may be built before freeze

Nothing of the gated stages. Existing instruments (court3d, the
audit tools, verify_hitter_features) may be maintained. This
document is not building.

## Amendment rule

Before freeze: anything may change. After freeze: amendments follow
contact_gate.md convention — dated, appended, tightening or
clarifying only; bars and the holdout seal never loosen.
