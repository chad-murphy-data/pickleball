# Ball Gate — pre-registration for the automated ball tracker

**STATUS: FROZEN 2026-08-31.** The user reviewed v2 with all
amendments (tap-free grading, automated person channel, readiness
rule, failure autopsy) and approved: "Done, I think we're good to do
a few more rallies" (2026-08-31). Bars are now immutable; amendments
follow the rule at the end of this document. Build order: train ball
passes (rallies 6-7) may be labeled in any order, but rally 8's pass
is committed and SEALED before the first tracker run on any rally.
v1 (same day) graded the tracker against
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

## Amendment 1 — 2026-08-31 (user-approved): check 2 is HUMAN-MATCHED

Measured on TRAIN before any decoder existed: the frozen turns
battery saturates below ~15 impacts. Rally 6 — 7 physical contacts in
5.6 s, ~14 path events, ±0.15 s tolerance — gives the circular-shift
null a 95th percentile of 100%, so the user's own hand-labeled path
scored a PERFECT 100% recall and still failed (rally 7: obs 77.8%,
null 95th 77.8%; rally 1 passed only because 25 contacts over 23 s
left the null room to fail). As frozen, check 2 was unwinnable on the
sealed rally class regardless of tracker quality — the exact
instrument-manufactured false-fail the autopsy section exists for,
caught upstream.

Check 2 therefore reads, per the gate's stated intent ("the same test
without the human"):
- On the graded rally, the tracker's decoded path must (a) recall at
  least as many PHYSICAL contacts (whiffs excluded — Amendment 1a
  below) as the user's hand-labeled path recalls on that same rally
  under the same frozen battery, and (b) sit at a circular-shift null
  percentile >= the human path's on that rally. On rally 6 this bar
  is a perfect score; nothing is softened in disguise.
- POOLED ABSOLUTE TEST: once the sealed set holds >= 20 physical
  contacts across rallies, the original absolute bars (recall >= 0.80,
  obs > null 95th) run POOLED across all sealed rallies and become
  binding alongside the per-rally human-matched check. Strictly a
  tightening over time.

Amendment 1a (same day, bug fix): contact=0 WHIFF taps are never
recall targets on any rally — a whiff is a no-contact swing, the ball
correctly shows no turn. Rally 1's frozen answer key is untouched.

Owner-reopen clause (user, same approval: "I reserve the right to
unkill things if I feel you were too harsh"): closures and KILL
verdicts bind the PROCESS — the agent may not reopen them. They do
not bind the OWNER, who may reopen any closure with an explicit call
plus a fresh registration (this gate itself is the precedent: the
ball thread was owner-reopened exactly that way). Reopening is never
silent and never a knob-turn of the failed instrument.

## Graded-run configuration note — 2026-08-31, recorded BEFORE the run

Owner authorized breaking the seal ("let's break it", 2026-08-31,
after the readiness report). Frozen here, before any rally-8 output
exists:

- **Serve pin** = t0s of the v4 windows row covering rally 8's time
  (189.2 s; the CSV's rally_cum numbering is shifted by one from the
  state labels for rallies >= 6 — mapping is BY TIME, the known
  anomaly).
- **Rally window end** = the user's point_dead mark (198.0 s).
  Clarification of "the rally window (referee logs supply these)":
  the windows CSV's t1s is NOT a log timestamp — it is the
  serve_pin_windows.py estimate min(next_serve-2, serve+1.5*shots+5),
  padded 6-12 s past the rally on every train rally (past the CLIP
  end on rally 6). Deployment referee logs DO carry the rally-end
  moment tightly. The point_dead mark is therefore the same stand-in
  category as the serve pin, which the gate licenses on exactly this
  reasoning ("logs carry serve timing"); it is not a contact tap,
  which remain excluded. This also matches the train configuration
  the readiness rule validated (decode trimmed at the dead mark).
- Pipeline as merged on main at grading (PRs #106-#107): candidate
  stage -> hitter chain (pose + blur gap-fill, wrist-position
  anchors) -> decoder (full-gap edge search, serve/end trims) ->
  arc refit; check 3 via ball_replicate (largest-angle anchor
  claims, crossing demotion, automated ankle-floor anchors).
  Anchor generation for rally 8 bypasses hitter_chain's train
  score() so the sealed ball pass is read ONLY by the scoring
  harness.
- One run. Pre-stated expectation (from the rally-6-shaped
  character of rally 8): MIDDLE is the modal outcome; check 3 is
  the underdog.

## GRADED RESULT — 2026-08-31, rally 8 (seal spent)

One run, configuration as in the note above (`vision/ball_grade.py`;
dry-run on rally 7 in the identical configuration passed end-to-end
before the seal was touched).

- **CHECK 1: 58.1%** V hit rate (75/129; S 54.3%) — between the
  bars (PASS >= 70, FAIL < 40).
- **CHECK 2: FAIL** — recall 85.7% vs the human path's 100% on the
  same rally; null pct 85 vs the human's 98; and the decoded path
  does NOT beat its own circular-shift null (85.7 vs 95th = 100).
  The human path beat its own null on this rally, so this is a fair
  loss under Amendment 1, not instrument saturation.
- **CHECK 3: PASS** — 3/7 impact points matched at median 2.82 ft
  (bar 3.0), net crossings 3/3, bounces 3 vs 2 (bar +/-1). The
  deliverable check survived out of sample.
- Pre-stated expectation (MIDDLE modal, check 3 the underdog) was
  wrong in both directions; recorded.

**VERDICT: FAIL, by check 2's null clause. Autopsy run same day:**

1. Oracle substitution — the user's clicks through the same decoder:
   V 94.6%, turns 85.7% at pct 98 beating its own null (95th 71.4).
   Decoder SOUND on clean input.
2. Person ablation — the automated 4-longest tracks were the SAME
   four tracks as the user's assigns; curated anchors reproduce
   V 58.1% exactly. PERSON channel exonerated.
3. Frame checks — candidate recall 84.5% at +/-1 frame proves clock
   alignment; window sane. Not VOID.
4. Candidate stage on the sealed rally: V recall 84.5%, S 100% —
   train-range. The arm-4 condition ("cannot find the ball in
   pixels") is NOT met.

**ATTRIBUTION: DECODER — association under clutter.** Every
component passes in isolation; the failure is the decoder's
selection among junk candidates, which arm 1 cannot test (no junk)
and which the miss geography shows directly: misses cluster in
5 runs during the fast exchange with candidates present (7/7,
12/13 of missed frames had the true ball in the stream), the path
sitting 39-161 px away on limb chains — the identical failure class
measured on train rally 6, which is rally 8's shape. The ball
channel itself is NOT killed (the autopsy's premise stands: arm 1
re-confirmed the information is in the stream).

**Consequence (per the frozen consequences): DECODER fix licensed;
re-attempt under a short dated addendum naming what changed, graded
once on a NEWLY labeled sealed rally.** Rally 8 is spent and is now
train data. The user is the rate limiter: the re-grade costs one
fresh sealed ball pass.

## Decoder-fix addendum — 2026-08-31 (owner-approved re-attempt)

Owner approved the two-regime split ("I'm good for two-regime split.
Let's go ahead there", 2026-08-31). What changed, and only this:

**Two-regime decode** (`ball_decoder.py`, wired into `ball_grade.py`):
one candidate stream, two decodes of it.
- The **POSITION stream** is the graded-run production config,
  byte-for-byte unchanged. Checks 1 and 3 read it, exactly as before.
- The **TIMING stream** re-decodes the same candidates with the
  position stream's own turns (>= 40 deg, from the forward AND
  reverse decodes, unioned) as additional turn anchors, plus a
  turn-cost hardening factor 2.5x away from all anchors, ONE feedback
  round. At grade time the hitter-chain anchors union in as before.
  Check 2 reads this stream through the same frozen battery. Directly
  aimed at the attributed failure: anchors waive the turn cost where a
  contact is plausible, hardening prices the junk turns everywhere
  else.

Bars, panel, battery constants, licensed inputs, autopsy arms:
unchanged. Scoring two streams was put to the owner explicitly (the
registration's-spirit question flagged in the 2026-08-31 afternoon
notes) and approved. Convention clarification recorded: check 2's
human reference is scored on the full label span (what ball_grade.py
and the graded run already did); score_train was panel-filtering and
now matches.

Train evidence for the frozen constants (sweep of 33 configs; full
tables in swing_explore_notes 2026-08-31 evening): with NO pose
anchors, the timing stream passes the human-matched check on ALL
THREE train rallies — r6 100@96.3 (human 100@94.5), r7 77.8@98.4
(77.8@91.4), r8 100@98.5 (100@98.5) — the rally class that failed
the grade now scores at the human path's own level. Dev rally 1
(long-rally screen) reads 88@99.3 vs human 92@99.5 without pose
anchors; the grade configuration unions the hitter-chain anchors in
(on r1 those carried the missile run to 96% recall). Measured dead
ends recorded in the module docstring (second feedback round,
hardening ladders, intersection anchors, fwd/rev-adaptive hardness,
30/60-deg anchor gates) so they are not re-tried.

**Rally designations:**
- Rally 8: spent -> train (SEALED sets cleared in all four
  harnesses).
- Rally 9's ball pass (delivered 2026-08-31, committed untouched;
  885 frames, 66V/22S/6I/6N) — designation was OWNER-PENDING with
  the recommendation on record: r9 -> TRAIN, because its 34-frame
  off-frame lob excursion (~1.1 s, 3x the decoder's GAP_MAX) is a
  regime no current train rally exhibits and the current decoder
  structurally cannot bridge, and r9 carries the log-span-anomaly
  asterisk; seal r10 instead.
- **DESIGNATED 2026-08-31 (owner)**: the owner labeled and delivered
  rally 10's ball pass in response to that recommendation — **r10 is
  the SEALED re-grade rally** (26 physical contacts, which by itself
  arms Amendment 1's pooled absolute test), **r9 is TRAIN**. r10's
  pass was committed sealed; its human bar stays unpeeked until
  grading. SEALED = {10} in all four harnesses.
- The re-grade runs once, on rally 10, after the readiness rule
  passes in the full grade configuration.

Dated corrections recorded BEFORE any rally-10 run:

- **2026-08-31, timing-stream inputs (readiness measurement):** the
  addendum's "at grade time the hitter-chain anchors union in as
  before" was tested by the readiness rule and MEASURED HARMFUL —
  r7 in grade config posted timing 77.8@86 with the union vs
  77.8@98.4 without (hitter anchors add cheap-turn zones at fake
  swings; the inflated event set eats the null percentile). The
  timing stream therefore runs on its SELF-FEEDBACK anchors ONLY —
  the exact configuration the train sweep froze and passed 3/3 —
  and the hitter-chain anchors remain inputs to the position stream
  (checks 1/3), where the same readiness run showed them working
  (replication 1.70 ft, crossings 5/5). Bars untouched; this
  narrows the grade config to the swept one.
- **2026-08-31, owner label caveat on the sealed r10 pass (recorded
  verbatim-in-substance, before any run):** the owner reports rally
  10 contains "another weird lob segment in the middle" where they
  were "mostly sure" they clicked the correct object. Logged now so
  the pre-registered frame-checks/label-sanity autopsy arm can weigh
  it AFTER grading if the verdict turns on that segment; it is not
  grounds to touch, score, or re-label the sealed pass beforehand.
- **2026-08-31, sealed pass finalized (owner delivery, pre-run):**
  the owner supplied the completed version of the same pass — "same
  clicks but stopped one frame after the ball bounced out of play"
  (672 rows, final row t=318.447; the first delivery carried ~11
  post-dead rows). The trimmed version is the sealed answer key of
  record; its final row IS the owner's point-dead mark. No tracker
  has run on rally 10; only the file's last row was read, as the
  rally-end mark.

## Graded-run configuration note — rally 10, recorded BEFORE the run

Frozen here before any rally-10 output exists (r8 precedent):

- **Readiness**: PASSED 2026-08-31 on train rally 7 in this exact
  configuration — V 82.3%, check 2 human-matched PASS (77.8@95 vs
  77.8@91), replication 1.70 ft / crossings 5/5 / bounces 2v2.
- **Serve pin** = 294.30 (v4 windows row covering rally 10's time;
  mapping BY TIME per the known numbering anomaly; the rally's serve
  tap 295.78 sits inside the window).
- **Rally window end** = 318.45, the owner's point-dead mark (the
  sealed pass's final row — "one frame after the ball bounced out of
  play"). Same licensing category as the r8 run's point_dead mark:
  a stand-in for the rally-end moment deployment referee logs carry.
- **Clip** = owner-cut r10_clip.mp4, 1980 frames @ 60 fps, offset
  292.7 (verified against the cut command). Candidates extracted
  post-readiness under --graded-run (78,645 candidates, 39.7/frame,
  committed as data/vision/ball_candidates_r10.csv.gz). Person
  channel = r0010.npz, automated as licensed.
- **Pipeline** = the merged two-regime state: position stream with
  hitter-chain anchors (checks 1/3), timing stream self-feedback
  anchors only (check 2), per the dated corrections above. Anchor
  generation bypasses hitter_chain's train score() as in the r8 run.
- **One run**, upon explicit owner authorization — GIVEN 2026-09-01
  ("break the seal, let's go"), recorded before the run.
- **Run attempt 1 (2026-09-01): CRASHED PRE-ANSWER-KEY, not a graded
  outcome.** hitter_chain.blur_gap_fill unpacked a no-wrist row
  (w=None; r10's pose has a wristless stretch r7 never exercised)
  and raised before anchor generation finished — before any decode,
  before the sealed pass was read, before any rally-10 output
  existed. Zero information about the seal was produced or consumed.
  Neutral guard added (a row with no measurable wrist cannot
  nominate a blur event; r7's path never reached the branch, so its
  readiness result is unchanged by construction). The one graded run
  proceeds under the same frozen configuration. No pre-stated
  expectation this time: the r8 note's forecast was wrong in both
  directions, and the honest prior is the train table plus the r1
  long-rally screen (r10 is long-rally class, 26 contacts — the
  null will have real power).

## GRADED RESULT — 2026-09-01, rally 10 (seal spent)

One graded run (attempt 2; attempt 1 crashed pre-answer-key, recorded
above). Configuration exactly as frozen in the note.

- **CHECK 1: 66.2%** V (314/474; S 85.0%) — between the bars
  (PASS >= 70, FAIL < 40), 3.8 points short of PASS.
- **CHECK 2: PASS — the tracker BEAT THE HUMAN PATH.** Timing stream
  recall 73.1% at null pct 98 vs the human's 65.4% at pct 91, and it
  beats its own shift-null 95th (65.4) outright — the clause that
  failed the r8 grade. 26 physical contacts; the null had real
  power. First instrument in program history to outscore human
  labels on a sealed rally's contact times.
- **CHECK 3: FAIL — on segmentation, not accuracy.** Matched impacts
  16/26 at median 2.73 ft (the 3.0 bar is MET); net crossings 10/19
  drawn segments (bar: all); bounces 3 vs the human path's 10 (bar
  +/-1). The rally is dink-heavy: bounce-rich. The tracker
  over-fragments (32 segments vs human 26) and under-recovers
  bounces; where impacts matched, the 3D agreement held.
- **Pooled absolute test (Amendment 1, first time binding — 26
  sealed physical contacts):** pooled recall 73.1% < 0.80 — SHORT,
  recorded. Noted alongside: the human path itself scores 65.4% on
  this rally under the same frozen battery (the lob class is hard
  for humans too), the first human pass under the 0.80 absolute bar.
- Owner's pre-recorded lob-segment caveat: to be weighed by the
  autopsy arms against the check-3 miss geography.

**VERDICT: MIDDLE, per the frozen bars — one further train-only
iteration, then one re-grade on a NEWLY labeled sealed rally.**
Rally 10 is spent -> train (a long-rally, bounce-rich, lob-bearing
training case — the exact regimes the iteration must address).
Autopsy arms run post-verdict on the spent rally per the
pre-registration; results appended when complete.
