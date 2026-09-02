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

## v3 — HIT-ANCHORED fill of the gaps v2 leaves open (written 2026-09-02 before any tune number; owner go: "Go ahead!")

**Why.** Autopsy of the gaps the adopted v2 product leaves open (r9 12 gaps
/ 5.7 s / 97 V clicks; r10 11 / 5.1 s / 116 — about a fifth of all
clicks) against the owner's tapped contacts: all but four hold exactly ONE
contact, usually at the gap's end, and half also hold a bounce. They are
HIT gaps: A is the ball arriving, B the ball leaving, two different
flights meeting at the paddle, so v2's closest-approach switch is the
wrong rule (the extensions miss by 46–350 px) and a bounce on A's side
breaks the extension outright. The owner's framing: "LL hit the ball and
UL hit it, the ball must travel between them; if it got there quickly it
travelled straighter" — a boundary-value arc between two hits.

**Anchors are production, never truth.** Contact TIME = the production
approach detector (`corridor_lab.prod_contacts`, the r6/r7-tuned
pathfirst "prod" arm) — a contact within ±TOL = 0.10 s of the open gap is
an anchor (r6 1/1, r7 4/4, r9 12/12, r10 10/11 open gaps have one; counts
of detections only, no clicks read). The hitter = the tracked player
whose paddle proxy (wrist + half forearm, the existing series) at that
time is nearest the midpoint of A's last pixel and B's first pixel; the
anchor PIXEL is that paddle proxy; its DEPTH is the hitter's floor
position (ankle midpoint through the z=0 homography) and the anchor's 3D
point is the pixel lifted at that depth with z clamped to a paddle range
[0.5, 9] ft. The owner's `imps` are never read by the fill.

**Rule per open gap** (A, B consecutive v2 flights, anchors c_1..c_k
inside the gap ±TOL, frames clipped to the gap):
1. A' = A's own projected pixels on its tracked frames (weight 1) refit
   with the anchor: pixel residual at t_c weighted W_ANC (px per px) plus
   a depth residual (y at t_c − hitter's y) at 1 px/ft. Fills
   A.fb+1 .. f_c1. Kept only if A' reproduces A's pixels within
   RMS_MAX = 3 px (else those frames stay open).
2. BOUNCE (grid arm): if A's UNANCHORED arc reaches the floor inside
   (ta − 0.05, t_c1), the fill is A to the floor and then a drag-free
   boundary-value arc from the bounce point to the anchor's 3D point
   over the remaining time — the owner's rule in its exact physical
   form. Mirror on B's side (B' back to the floor after t_ck).
3. c_i → c_{i+1}: drag-free boundary-value arc between the two lifted
   anchor points (≥ 3 frames).
4. B' = mirror of 1, fills f_ck+1 .. B.fa−1.
Every filled frame is tagged `inferred3`; v2's tracked frames and v2's
inferred frames are bit-identical to the adopted product (asserted).
pathfirst.py, events.py, gapfill v2 untouched.

**Grid** (4 cells; the train set has only 5 open gaps / 24 clicks, so the
grid is small on purpose): W_ANC ∈ {2, 6} × BOUNCE ∈ {off, on}. TOL,
RMS_MAX, the depth weight, the z clamp are FIXED above.
**Selection** on r6+r7 (cross-fold `_x` p-caches): max pooled inferred3
r@12 s.t. inferred3 prec@12 ≥ 0.5 and own nulls ≤ 3; ties → smaller
W_ANC, then BOUNCE off. If no cell passes: DEAD, nothing graded.
**Bars on r9/r10 (both rallies; second re-use of the eval rallies,
under the owner's 2026-09-02 relaxation, disclosed)**:
1. tracked + v2-inferred frames bit-identical to the v2 product;
2. inferred3 prec@12 ≥ 0.5 (a v3 frame is more likely right than wrong);
3. inferred3 r@12 ≥ 10;
4. inferred3 displaced + time-shift nulls ≤ 3;
5. events v3 F1 on the v3-filled flights ≥ adopted − 0.03 (r9 ≥ .701,
   r10 ≥ .645 — the same reference as v2).
ADOPT ⇒ `gapfill.product3(ctx)` becomes the consumer product, with
inferred3 as its own tag (consumers may draw it distinctly or drop it).
FAIL on either rally ⇒ recorded, v2 stays, no re-tune on r9/r10.
Clean re-check on r20/r21 as written.

### v3 as registered: DEAD on train (2026-09-02, `gapfill_tune3_v3a_DEAD.txt`)

All four cells: pooled inferred3 r@12 = 1 at prec 0.077 (r6 1/13 on 28
frames; r7 0/0 on 46 frames — the anchored refits were rejected by
RMS_MAX on every single-anchor gap, only between-anchor BVPs survived).
Nothing graded on r9/r10. Diagnosis on TRAIN ONLY (r6/r7 clicks and
truth contacts, which the rules allow): the anchor PIXEL is the failure.
The paddle proxy (wrist + half forearm) sits 80–200 px from the ball at
contact — a paddle length away, and on r6's first anchor a different
player from the hitter — so a point anchor is wrong by construction, and
forcing A through it costs > 3 px on A's own pixels. The detector's
TIMES are fine (within 2–4 frames of the truth contacts on every train
gap that has one). Two more instrument facts: r6's open gap holds two
contacts with a BOUNCE between them, where a plain boundary-value arc is
wrong, and r7's 3-anchor gap is detector false positives with the ball
off-frame.

## v3b — re-registered on train only (written before any v3b number)

Changes, all from the train diagnosis above; r9/r10 still untouched:
1. Anchor = contact TIME (detector, as before) + hitter's DEPTH (floor y,
   as before) + a paddle REGION: a hinge at R_ANC = 2 ft (at the local
   px/ft) around the paddle proxy, weight W_ANC, instead of a point.
2. The two flights are refit JOINTLY: A's own pixels + B's own pixels +
   a MEET residual |A(t_c) − B(t_c)| in px at W_MEET = 2 (fixed) + the
   depth and region terms on both. Accepted only if BOTH still reproduce
   their own pixels within RMS_MAX = 3 px. A' fills A.fb+1 .. f_c, B'
   fills f_c+1 .. B.fa−1. Contact point X_c = the mean of A'(t_c), B'(t_c).
3. BOUNCE arm unchanged in spirit: if A's unanchored arc reaches the
   floor before t_c, A runs to the floor and a drag-free BVP runs from
   the bounce point to X_c (mirror on B).
4. Gaps with ≥ 2 anchors: A' anchored alone to the first, B' alone to the
   last (region + depth, no meet); the span between them is filled by a
   drag-free BVP between A'(t_c1) and B'(t_ck) ONLY if that arc stays
   above the floor (a volley-to-volley exchange); a BVP that dips under
   the floor means a bounce we cannot place from no data — left open.
Grid, selection rule, bars, consequences: exactly as v3.

### v3b as registered: DEAD on train (2026-09-02, `gapfill_tune3.txt` / `.json`)

All four cells fill ZERO frames: every joint refit (single-anchor gaps)
and every anchored single refit (multi-anchor gaps) is rejected at
RMS_MAX 3 px. Per-gap autopsy (scratch, reproduced in
`gapfill_explore3.txt` part 1): the meet residual at the detector's
contact time starts at 137–244 px on the three short r7 gaps and the
optimiser can only close it by wrecking the arcs (own-pixel rms 8–43 px
at convergence); on r6 / the long r7 gap the refit_one arms never get
within 3 px either. The constraint set is unsatisfiable, not mis-coded.

## v3c — train-only exploration, NOT a registration (2026-09-02, `gapfill_explore3.py` → `gapfill_explore3.txt`)

Before writing the hit-anchored idea off, every kink rule that uses only
what the anchor reliably carries — the contact TIME and the hitter's
floor DEPTH, never the paddle pixel — was run on r6/r7 with the gap
filled A-side / kink / B-side by drag-free BVPs (bounce arm included):
X_c = A forward, B backward, their mean, mean at hitter depth, the 3D
chord at t_c, and the pixel of either extrapolation re-lifted at the
hitter's depth. r9/r10 were never loaded. Best cell: A-forward, 7/41
right at 12 px (prec 0.17), all others 1–4/41; the bounce arm never
fires because the depth-degenerate arcs are already under the floor at
the gap edge or never cross it. Nothing is near the 0.5 / ≥10 bar, so
no v3c is registered.

**Why (part 1 of the printout, the finding that closes this line):**
1. **Three of r7's four open gaps start on a flight whose TAIL is off
   the ball** — A's own last frame sits 50–110 px from the owner's
   click (f142: 109 px, f346: 50 px, f535: 92 px), i.e. the tracker
   ended those flights on a non-ball candidate near a bounce or a
   body. A kink anchored in time cannot repair a wrong starting point;
   B's backward arc is right to ~10 px in all three, which is exactly
   what v2's meet rule could not see because A never comes to it.
2. **r6's open gap is a two-contact fast exchange** (owner contacts at
   5.70 and 6.07 s, 8 streak clicks between them). A forward tracks the
   truth to ≤ 9 px right up to the first contact — that is where the 7
   hits come from — and B backward is exact from the second; the 34
   frames between are a drive the arcs cannot bend into, and a BVP
   between two depth-degenerate 3D points projects onto the wrong
   curve (the streak runs almost straight in pixels; the 3D heights
   the arcs carry there are +10 ft and −3 ft).
3. The long r7 gap (1.2 s) is the ball off-frame with three detector
   false positives; nothing to fill.

So the owner's human logic ("LL hit it, UL hit it, the ball must travel
between them, quickly = straighter") is right about the r6 gap and is
the one that would pay: it needs the fill drawn in PIXEL space between
the two contact pixels (a near-straight line for a fast shot), not a
3D BVP, and it needs the contact TIMES to ~1 frame (the detector's are
2 and 1 frames off there). The r7 gaps need the tracker to stop ending
flights on junk — hand-off/tail work, not inference. Neither is a
knob on v3; both are new registrations. **Verdict: v3 line closed on
train; v2 stays the adopted fill; r9/r10 untouched by anything v3.**
`gapfill.py`'s v3/v3b code stays as the reproducible instrument behind
these records (`tune3`, `selftest3`; `grade3` must never be run — the
rule has no passing cell to shoot).
