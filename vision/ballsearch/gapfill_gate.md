# Gap fill by arc extension — pre-registration (2026-09-02, before any tune number)

## Why
Owner's framing (2026-09-02): "We're never going to see the ball when
it's behind a paddle or a player. So infer a point ... from where the
ball disappears to the first discovered frame." Autopsy of the adopted
path-first track (scratch `occl_autopsy.py`, fixed geometry, no sweep)
agrees with the premise: on r9 124 of 242 misses and on r10 169 of 235
sit in an inter-flight gap AND within half a body height of a wrist or
paddle proxy; 23 of r9's 33 gaps hold an owner-tapped contact. This
layer is ONLY for frames that do not exist except through inference.
It does not touch the in-flight misses (arc/depth error inside tracked
flights) or the long lost flights; those are other work.

## Rule (gapfill.py; pathfirst.py and events.py untouched)
Pass 1 = the adopted path-first run (frozen cell, unchanged).
For consecutive selected flights A, B (sorted by start frame) with a
gap of g = fa(B) − fb(A) − 1 ≥ 1 frames and g/60 ≤ GAP_MAX seconds:
- extend A's fitted arc FORWARD and B's fitted arc BACKWARD over every
  frame of the gap (image plane, `pathfirst.arc_px`);
- f_c = the gap frame (inclusive of both ends) where the two extended
  arcs are closest in the image plane; d_c = that distance in px;
- if d_c ≤ D_MEET: fill frames fb(A)+1 … f_c from A's arc and f_c+1 …
  fa(B)−1 from B's arc, and set fb(A) = f_c, fa(B) = f_c + 1, so the
  layer above sees two flights meeting at one frame. Otherwise the arcs
  do not meet and the gap is LEFT OPEN (no straight line, no paddle
  waypoint: the paddle proxy is wrist + half a forearm from the pose
  model and picks the switch time worse than the arcs do — measured in
  the autopsy).
Filled frames are tagged `inferred` in the returned track so a viewer
or demo can draw them differently; the grade does not distinguish them.
No paddle is used anywhere in the fill. The arcs carry the inference.

## Tuning (r6 + r7 ONLY, cross-fold p-caches)
Grid: GAP_MAX ∈ {0.5, 0.8} s, D_MEET ∈ {20, 40, ∞} px. 6 cells.
Selection rule: highest pooled r@12 on r6+r7 subject to pooled prec@12
≥ incumbent path-first prec − 0.03 and both nulls (displaced,
time-shift) r@12 ≤ 3; ties → smaller GAP_MAX, then smaller D_MEET.
Incumbent path-first on r6+r7: 263 @ 0.807. No cell beats it → DEAD.

## One shot (r9, r10) — bars, never loosened (same bars as handoff_gate)
ADOPT only if on BOTH rallies: r@12 strictly above the incumbent
(537 / 422); prec@12 ≥ incumbent − 0.02 (0.85 / 0.86); displaced and
time-shift nulls r@12 ≤ 3; and the adopted events v3 layer re-run on
the filled flights keeps F1 ≥ adopted − 0.03 (.701 / .645). Anything
else = NOT ADOPTED, recorded, incumbent stays. Secondary (reported):
at-click coverage, V/S strata, number of gaps filled / left open,
events recall and precision, filled-frame precision alone.

## Disclosure (honesty about what was seen before this file)
The autopsy that motivated this layer ran ONE fixed geometry (arc
extension, switch at closest approach, GAP_MAX 0.8, no D_MEET) on all
four rallies, r9/r10 included, and reported recall gains of +64 / +74
clicks at 12 px. Precision, nulls and the events layer were NOT looked
at. So the r9/r10 recall bar is informed for that one cell; the
selection here is by the r6/r7 rule above regardless, and the
precision, null and events bars are unseen. The next CLEAN seal for
this line is r20 / r21 once labeled (label_picks.md).

Owner clicks on r9/r10 are used for grading only. Owner go for the
build and the shot: 2026-09-02 ("Let's go ahead and do that").

## Results — tune (2026-09-02, r6 + r7, 6 cells, gapfill_tune.txt)
Every cell keeps both nulls at 0/2. Recall rises with GAP_MAX and with
D_MEET; precision falls with D_MEET: no limit on the meeting distance
fills 8 of r7's 9 gaps and costs 8 points of precision (0.730). Under
the rule: **gap_max 0.8, d_meet 20 → 273 @ 0.782** vs the incumbent
263 @ 0.807 (+10 frames, +4%; the 279 @ 0.730 cell fails the precision
floor). Selected per the rule; the one shot follows.

## Results — the one shot (2026-09-02, r9 + r10, gapfill_grade_r{9,10}.txt)
Cell gap_max 0.8 / d_meet 20, frozen from the tune above.

| | r9 | r10 |
|---|---|---|
| path-first (adopted) r@12 / prec | 537 / 0.87 | 422 / 0.88 |
| gap-fill r@12 / prec | **556 / 0.866** (+19, +3.5%) | **443 / 0.850** (+21, +5.0%) |
| r@8 | 416 → 429 | 327 → 339 |
| at-click coverage | 616 → 642 / 779 | 479 → 521 / 657 |
| inferred frames alone: at-click / r@12 / prec | 46 / 30 / 0.65 | 59 / 34 / 0.58 |
| gaps filled / no-meet / long | 13 / 12 / 1 | 9 / 12 / 1 |
| nulls displaced / time-shift r@12 | 0 / 2 | 0 / 1 |
| events v3 on the filled flights F1 (bar) | **0.756** (≥ .701 ✓; adopted .731) | **0.684** (≥ .645 ✓; adopted .675) |
| events recall / prec | 0.756 / 0.756 | 0.692 / 0.675 |
| verdict | PASS | **FAIL** (precision bar, by 0.010) |

**NOT ADOPTED.** The bars were written as "both rallies"; r10 misses
the precision bar (0.850 vs 0.86). The incumbent path-first track and
the v3 events layer stay production. Recorded, not re-run, no bar
loosened. `grade` is spent for r9/r10 on this line.

What the numbers say, honestly. The tracked frames are untouched by
construction, so every change is the inferred frames: they are right
at 12 px about 60% of the time (0.65 r9, 0.58 r10), which is the
measured honesty of "extend both arcs through the occlusion" on this
footage — and it is what pulls the pooled precision under the bar on
r10. The layer above LIKES the fill: events F1 rises on BOTH rallies
(.731 → .756, .675 → .684), the opposite of the hand-off result,
because meeting arcs give the seam rule an arrive/depart pair at one
frame instead of two loose ends. Half the gaps are left open on both
rallies because the two arcs never come within 20 px of each other:
those are the gaps where at least one flight end is a fragment whose
extrapolation is wrong, not occlusions the fill can bridge.

The precision bar was written for a product where every frame claims
to be a detection. An inferred frame is a different product (the owner's
framing: "data that don't exist other than through inference"); the
honest way to ship it is TAGGED, graded as its own stratum with its own
bar and its own displaced null (HANDOFF item 2 already proposed exactly
this for the off-frame stratum). That is a NEW registration on the next
clean seal (r20 / r21), not a re-read of this one.

## v2 — inferred frames as their own product (2026-09-02, owner go to re-use r9/r10)

Owner, after the v1 shot: "I'm ok using the previous rallies to try
again, let's not be as sweaty here, we'll have chances to check it
again." So r9/r10 are re-used for THIS registration, disclosed; r20/r21
(label_picks.md) are the later clean check.

What is different: the PRODUCT. v1 graded the filled track as if every
frame were a detection and failed the pooled precision bar by 0.010.
v2 says an inferred frame is a different kind of claim: the track is
the incumbent's tracked frames, bit-identical, PLUS a tagged set of
inferred frames that any consumer can drop and get the incumbent back.
Each half is graded on its own.

Instrument unchanged (`gapfill.fill`): same arc extension, same switch
rule, same tag. Tune rule re-stated for the product (r6/r7 only, same
6-cell grid): highest pooled INFERRED r@12 subject to inferred prec@12
≥ 0.5 and both nulls on the inferred frames alone ≤ 3; ties smaller
gap_max, smaller d_meet. (The v1 cell gap_max 0.8 / d_meet 20 is the
fallback if the new rule picks nothing.)

Bars on r9 and r10, BOTH rallies, written here before `grade2` runs:
1. tracked frames bit-identical to the incumbent (asserted; r@12 stays
   537 / 422 on the tracked half);
2. inferred stratum: prec@12 ≥ 0.5 (an inferred frame must be more
   likely right than wrong — the principled floor, and the one v1's
   numbers 0.65 / 0.58 already sit above; disclosed) AND inferred r@12
   ≥ 10 AND displaced + time-shift nulls on the inferred frames ≤ 3;
3. events v3 (frozen cell) on the filled flights: F1 ≥ adopted − 0.03
   (.701 / .645), as in v1.
Anything else = NOT ADOPTED. Adoption means: consumers may use the
filled flights WITH the tag (viewer draws inferred frames dashed;
events run on the filled flights; stats too); the graded track number
quoted for the tracker stays the tracked half's.

### v2 results (2026-09-02, gapfill_tune2.txt, gapfill_grade2_r{9,10}.txt)
Tune under the v2 rule (r6/r7, inferred stratum alone): prec falls
monotonically with d_meet (0.708 / 0.621 / 0.413 at gap 0.5; 0.553 /
0.512 / 0.383 at gap 0.8); the no-limit cells fail the 0.5 floor.
Selected **gap_max 0.8 / d_meet 40 → inferred 22 @ 0.512** (v1's cell,
d_meet 20, is 21 @ 0.553 — the rule takes the extra frame).

| | r9 | r10 |
|---|---|---|
| tracked half (bit-identical) r@12 / prec | 537 / 0.87 ✓ | 422 / 0.88 ✓ |
| inferred frames / at-click | 114 / 57 | 154 / 69 |
| inferred r@12 / prec@12 (bar ≥ 10, ≥ 0.5) | **38 / 0.667** ✓ | **41 / 0.594** ✓ |
| inferred r@8 / r@20 | 30 / 49 | 29 / 57 |
| inferred nulls displaced / time-shift | 0 / 0 | 0 / 1 |
| tracked + inferred r@12 / prec (reported, not a bar) | 563 / 0.86 | 447 / 0.85 |
| events v3 on filled flights F1 (bar) | **0.764** (≥ .701; adopted .731) | **0.658** (≥ .645; adopted .675) |
| verdict | PASS | PASS |

**ADOPTED as a TAGGED product** (2026-09-02). The tracker's graded
number stays the tracked half's (537 / 422 @ 0.87 / 0.88); the inferred
frames are a second, tagged layer that is right at 12 px two times in
three on r9 and three in five on r10, above chance by every null, and
that any consumer can drop. Events on the filled flights: up on r9
(.731 → .764), down 0.017 on r10 (.675 → .658) — within the bar, and
note the v1 cell (d_meet 20) had r10 events at .684; the v2 rule chose
recall in the inferred stratum over that. Consumers wired with the tag:
`rally_3d.py` / court3d viewer (inferred segments dashed, captioned),
`render_court3d.py` (same in the video), `rally_stats.py` (events and
hits on the filled flights). Clean re-check on r20 / r21 when labeled
(bars 1–3 as written, no re-tune).
