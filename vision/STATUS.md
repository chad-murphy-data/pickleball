# Vision channels — the ledger

Written 2026-08-20 because summaries of this work kept regressing to
whatever was measured most recently. Every "where are we" answer should
be generated FROM THIS FILE, not from memory of the last session. If a
summary disagrees with this table, the summary is wrong.

Update the table when a measurement lands. Do not update it to match a
mood.

## What works

| channel | number | chance / null | status |
|---|---|---|---|
| hitter SIDE | 19/20 = 95% | 50% flat | **works** |
| hitter 4-WAY | 17/20 = 85% | 25% flat | **works** |
| play / no-play | 30/30 | real binary null | **works** |
| shot COUNTS (alternation decoder) | 161/162 | — | **works** |
| ball circles, spatial | 25/27 = 93% precision | user-audited | **works** |
| side alternation | 0 violations / 229 contacts | — | **exact** |
| team assignment from alternation | 0 contradictions | — | **exact, pooled** |
| court homography | 0.06 ft median residual | — | **solved** |
| player identity from referee logs | 99.25% / 45,689 rallies | — | **solved** |
| HUMAN ball path -> contact times (oracle) | 23/25 = 92% @ ±0.15s | shift-null 95th: 76% | **PASSED, licensed** |

These share one property: **a flat chance baseline, or no tolerance to
widen.** That is exactly why the 2026-08-20 null work could not touch
them. It killed placement in TIME; none of these is a timing claim.

## What is dead

| thing | verdict | why |
|---|---|---|
| contact TIMESTAMPS, every instrument | dead | see below |
| ball detection / tracking (DETECTOR) | closed | 64% was isolated-frame; dense contiguous = 92% in-pixels (2026-08-30), but the detector question stays closed — the oracle licenses HUMAN labels only |
| poach / intent measures | retracted | needs the ball |

Timestamp placement, all four grid arms measured against a
random-phase null: 3x3, 5x5, 6x6 marked, 6x6 marked re-run. Every one
posts 83-93% at +/-0.5s, every one lands 27-53% at +/-1 cell, and not
one clears its own shifted calls at 95%. +/-0.5s is 3.3 cells wide at
these packings, so it measures "roughly the right region", which rhythm
alone answers. The 3x3 arm DOES clear its uniform null (92.9% vs 78.6%,
98.1th) and falls short only on the shift null (82.5%, 93.6th) — it is
the closest thing to a survivor and it is not established.

The classical tracker is the same story from the other side: 68% at
+/-0.5s against a 78.5% null (0th percentile), but a median timing
error of 0.08s, which the tolerance cannot see. Its tight-tolerance
recall is unmeasured and is the open question.

## Graded and attributed

**Automated ball tracker, ball_gate.md graded run (2026-08-31, rally 8
sealed -> spent):** VERDICT FAIL by check 2's null clause — V 58.1%
(bars 70/40), turns 85.7% vs human 100% and under its own shift-null
(95th 100), **check 3 PASS** (3D replication: median impact agreement
2.82 ft vs 3.0 bar, crossings 3/3, bounces 3v2). Autopsy attribution:
**DECODER (association under clutter)** — oracle substitution 94.6% V /
null-beating turns, person channel exonerated (auto tracks = user's
assigns), frame sound, candidate recall 84.5%/100% on the sealed rally.
Readiness had passed on rally 7 (V 82.3 / turns 100 at pct 99 /
replication 1.70 ft); rally 8 is rally-6-shaped and failed the same way
rally 6 does on train. Re-attempt licensed after a decoder fix, one new
sealed ball pass required (gate record: ball_gate.md, graded-run note +
result). Train state of record: r7 full battery PASS, r1 checks 1-2
PASS, r6 fails check 3.

**Decoder fix BUILT + constants frozen (2026-08-31 evening,
owner-approved addendum in ball_gate.md): TWO-REGIME decode.** Position
stream = graded config unchanged (checks 1/3); timing stream re-decodes
with the position stream's own >= 40-deg turns (fwd AND rev decodes,
union) as turn anchors + 2.5x turn-cost hardening away from anchors,
one feedback round (check 2). Train, with NO pose anchors: PASSES the
human-matched check on ALL THREE train rallies — r6 100@96.3 (human
100@94.5), r7 77.8@98.4 (77.8@91.4), r8 100@98.5 (100@98.5) — and
screens 88@99.3 vs human 92@99.5 on long dev rally 1 (grade config
adds hitter anchors; on r1 those carried the missile to 96%).
Measured dead ends (do NOT re-try; module docstring + notes): a second
feedback round (r8 -> 85.7), hardening ladders (fix r8, break r6 —
soft-round junk gets anchor protection), INTERSECTION anchors (r8
14.3@0.8 — fwd/rev disagree exactly at contacts, the co-location
geometry), fwd/rev-agreement adaptive hardness (picks wrong on r8),
60-deg anchor gate (hardens dink turns away on long rallies).
DESIGNATED 2026-08-31: r10 SEALED (owner labeled + delivered its
pass; 26 physical contacts, arming the pooled absolute test), r9 ->
TRAIN (its off-frame lob — 34 frames N, 3x GAP_MAX — is the named
next decoder capability). READINESS PASSED
2026-08-31 on r7 in the exact grade config (V 82.3 / check 2
human-matched 77.8@95 vs 77.8@91 / replication 1.70 ft) — after the
readiness rule caught and reversed one conjecture: hitter-chain
anchors in the TIMING stream are measured harmful (77.8@86 with vs
77.8@98.4 without; fake-swing anchors bloat the event set), so the
timing stream runs self-feedback only and hitter anchors stay in the
position stream. r10 staged, owner authorized, and GRADED
2026-09-01: **VERDICT MIDDLE** — check 2 **PASS, BEATING THE HUMAN
PATH** (timing 73.1@98 vs human 65.4@91, beats own null 95th — first
instrument in program history to outscore human labels on sealed
contact times); check 1 66.2 V (between bars); check 3 FAIL on
segmentation not accuracy (matched impacts 2.73 ft UNDER the 3.0
bar; bounces 3 vs 10, crossings 10/19). Pooled absolute (first time
binding): 73.1 < 80, short — the human path also fails 80 on this
rally class (65.4). AUTOPSY (arm validated on the r8 reference
first): candidates EXONERATED (93.6%, full lob-window coverage —
the owner's label caveat resolves in the labels' favor), frame
sane, person N/A; **DECODER, two named defects** — (a) no
off-frame-excursion capability (the flagged lob = the one
over-GAP_MAX label gap, 24 frames; clean-input decode truncates at
exactly 50% of the window there), (b) bounce segmentation on
dink-heavy rallies. Per the frozen bars: ONE train-only iteration,
RUN 2026-09-01 (full record: ball_gate.md iteration record + notes).
KEPT: excursion edges (r10 oracle 50->95.8), anchors-never-touch-a-
decode, looser top-band margin, check-3 bounds from the timing
stream. State at freeze: check 1 PASSES r6/r7/r10 (80.3/84.8/73.6),
check 2 PASSES r7/r10 (+r6/r8 train windows), r10 replication at
20/26 matched impacts / 2.15 ft (best ever) — but CHECK 3 FAILS on
all three rallies, a different component each (r7 bounce over-gen
5v2, r10 under 5v10 + crossings 14/21, r6 accuracy). Named residual
causes were hitter-anchor precision and bounce-vs-junk triage;
decoder knob-turning deliberately stopped at the overfitting line.
FIX #1 LANDED same day (fit-demanded bounces: interior search grid +
restitution in court3d.fit_segment; motivated by measured bounce
occlusion — far court 32% I/N at floor-proximity moments, the
white-dress contrast wall): r7 now passes every check-3 component
except bounce count (6v4), r10 20/26 at 2.42 ft with bounces 11v13 —
both off by exactly 2. Remaining failures all trace to BOUND
recall/precision = the anchor-precision front. Bounce mechanism no
longer the blocker. ANCHOR DEDUPE landed same day (two tiebreaks
measured; timing-turn proximity frozen): rally 7 posted the FIRST
FULL THREE-CHECK PASS under the z-tiebreak; the frozen config holds
r10 at 16/26 / 2.98 ft / 14-15 crossings / 8v13 bounces with r7 one
impact from PASS. STOPPED at the see-saw: three configs trade r7 vs
r10 on single impacts (n=5-9 matched) — further claiming arithmetic
is curve-fitting. Next levers, evidence-ordered: hitter_chain anchor
QUALITY (fake swings out-z real dinks), the r9 clip (missing 6th
battery rally + second lob case), more tapped rallies. r20 re-grade NOT recommended yet; owner fork
recorded in the gate file: one anchor-precision/bounce-triage train
pass first, or re-register a narrower checks-1+2 contact-times
channel (touch share / tempo / categories) as a fresh
pre-registration. r10 spent -> train.
**Owner truth verdicts + r9 baseline (2026-09-01, later):** all 5
disputed r10 bounce calls verified REAL (kitchen short-hops; one
occluded) -> r10 truth = 13, the human-side fitter was exactly right;
hard velocity gates REVERTED (premise falsified), BOUNCE_MARGIN kept.
Truth rerun: **r7 FULL three-check PASS** (2.36 ft / 4-4 / 3v3); r10
human fit = truth 13/13 on 26/26 segments — the bounce mechanism is
exact given human positions; r10 tracked side 16/26 @ 2.98 ft / 8v13.
Owner uploaded the r9 clip: candidates 92.8k @ 95.4 V-recall;
baseline battery check 1 PASS **86.1 V, best on record — the
r10-built lob machinery generalizes unseen**; check 2 beats own null
at pct 100 (79v83 human near-miss); check 3 FAIL on a third profile
(bounds 22/28 + crossings 23/23 strong; median 3.75 ft from ~0.1
s-early fast-shot bounds; bounces 18v14). All three check-3 failures
now sit on the anchor/claim front. Anchor stage quantified
(anchor_diag): recall 5/7, 6/9, 18/29, 17/26 with 3-37 fakes and
fake-z overlapping or beating real-z — dink RECALL and z ORDERING
are the two measured defects; anchor-quality session in progress
(torso-relative speed, asymmetry discount, alternation rescoring).

## Built, producing numbers

**Ball-path corridor tracker, learned-emission upgrade (2026-09-01;
full arc + numbers in swing_explore_notes.md, instruments in the
session scratchpad).** Per-corridor DP over motion candidates with
body-extremity cost, graded against owner path clicks (r@12 hits +
precision vs displaced-anchor nulls; nulls 0–5 against real arms in
the hundreds). Owner lifted the no-training rule for the ball thread:
a logistic emission scorer fit on r6/r7 clicks ONLY (AUC 0.90/0.94
cross-rally; S clicks = ignore-zones, never positives) now enters the
DP as a SOFT cost 25·(1−p) — weight tuned blind on r6/r7 cross-fold,
rule frozen first, then one-shot on the evaluation rallies: r9 431
r@12 @ prec 0.69 (incumbent 388/0.55), r10 320 @ 0.67 (282/0.51);
oracle arms 406/0.69 and 309/0.61. Hard p-filtering REJECTED — it
kills whole chains on faint fast drives; soft > hard > none. The
owner-designed repertoire trail matcher (52-shot book) rides the same
p as a precision/bridging arm (r9 oracle 177 r@12 @ 0.74). Product
questions (bounce ledger, eviction) not yet re-graded on this stream.
**Fusion (2026-09-01, `vision/ballsearch/fusion.py`): the three-part
model (emission + trail proposals + DP as one cost) was built,
self-tested and tuned blind on r6/r7 under a pre-registered rule —
DEAD: best cell 381 @ 0.620 vs incumbent 376 @ 0.633, recall falls
monotonically with trail weight, r9/r10 never run. Train autopsy: the
trail helps only as a candidate POOL (W≈0 arm beats every cost cell),
hurts as a per-frame cost, bridges sit > 12 px off — the corridor
geometry error it inherits binds first (HANDOFF item 2).**

**Coverage model** (branch `claude/court-coverage-model-8rg94l`), one
match, 90 of 141 rallies: width share (Alshon .549 / Black .451),
90%-contour area (Alshon 261 > Black 232 > Patriquin 204 > Bright 188
ft2), off-court fraction (Black 6.8% vs Alshon 3.3%), depth and
kitchen-band occupancy. Three findings already withdrawn by its own
checks (deep poach, crossing rate, ellipse area) — the instrument
polices itself.

## Measured, not yet a channel

**Paddle visibility near a tracked wrist** (2026-08-29, user eyeball
of PR #65's `paddle_probe.py` n=24 contact sheet — the pre-build
sanity check): paddles visible in ~all crops; all but 2 hitters in
frame and both misses are the probe's own wrong-wrist centering, not
invisibility; hitters' paddles mostly motion-blurred at contact; ball
absent from the vast majority of contact-instant crops (independent
corroboration of the ball closure's "misses pile at contacts").
Sanity gate PASSED — a bounded paddle detector is feasible; usefulness
untested. Presence is non-discriminative (everyone holds one): any
signal is position/attitude/blur. Full reading + process constraints:
swing_explore_notes.md 2026-08-29.
**2026-08-31 follow-ups, measured:** ball goes invisible AT the paddle
(I-frames 17-32 px from a wrist, 88-100% within 60 px); paddle point =
wrist + 0.5*(wrist-elbow) beats the wrist as contact-position estimate
(13.5-15 px vs 18-20; occluded 12.6-13.7 vs 18) — shipped as an anchors
column; a full paddle DETECTOR is unnecessary for that use. Blur
(smear mass near wrist) is a weak standalone contact channel
(24-57% recall) but complementary — ships as gap-fill only.
Forward/backward decode agreement is a label-free confidence map
(agreeing frames err 4-7 px, disagreeing 32-34).

## Buildable now, glue unbuilt

**Touch share** — how many balls each player actually hits. Needs no
timestamps: alternation gives every shot's SIDE exactly, so the only
open call per shot is WHICH PARTNER, a 50/50 the VLM makes at ~89%.
Direct complement to coverage: coverage measures SPACE, touch share
measures BALLS, and where they diverge is the "who is carrying this
team" question. Costs the 3x3 rung.

## Cost, and how soft it is

$44/match for 3x3 = 5.5M input tokens. MEASURED: the 1568px downscale
cap and the /750 token rule, the 0.15s step, 1.13 contacts/s.
ASSUMED: 47 min of rally time per match, $15/$75 per M tokens, batch at
half price. UNPRICED: retries, and the 30-40% of a condensed VOD that
is not main camera — which the coverage work measured and this model
bills as usable anyway. Order of magnitude, plausibly 1.5x low.
