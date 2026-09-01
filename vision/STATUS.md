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
dink-heavy rallies. Per the frozen bars: ONE train-only iteration
(specced: excursion edges + bounce-aware segmentation; train
coverage r9/r10 lobs + r10's 10 bounces), then one re-grade on a
newly sealed rally (r20 identified). r10 spent -> train.

## Built, producing numbers

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
