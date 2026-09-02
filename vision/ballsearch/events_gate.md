# Events from flight ends — pre-registration (frozen 2026-09-02, before any number)

Owner's read of the path-first overlay (2026-09-02): every real hit shows
TWO labels (ball arriving at the paddle = end of one flight, ball leaving
it = start of the next), a lob is split at its apex and the seam gets a
label, and the flight ends were not typed. HANDOFF to-do 0c. This layer
turns the adopted path-first flights into ONE event per moment the ball
changes flight. The pixel track is NOT touched: `events.py` re-runs
`pathfirst.run` with the frozen cell (`pathfirst_tune.json`) and reads the
flights; `grade` asserts r@12 equals the adopted numbers (r9 537, r10 422).

## Instrument: `events.py`

For consecutive selected flights A, B (sorted by start frame):
- dt = start(B) − end(A) in seconds (≤ OVERLAP_SEL frames can be ≤ 0).
- Image-plane seam error e = mean of |proj_A(t_B0) − track_B(t_B0)| and
  |proj_B(t_A1) − track_A(t_A1)| in px — each arc extrapolated across the
  gap to the other's first/last tracked frame. Image plane, not 3D: the
  owner confirmed the wrong-side-of-net read was depth only.
- Direction change a = angle (deg) between the image-plane velocity of A
  at its last frame and of B at its first frame.
- SEAM (no event) if dt ≤ DT_SEAM = 0.12 s AND e ≤ R_SEAM AND a ≤ A_SEAM.
- Else if dt ≤ DT_PAIR: ONE event at t* = the time in
  [end(A) − 0.02, start(B) + 0.02] (grid 1/240 s) where the two arcs'
  image-plane extrapolations are closest — the arrive/depart pair
  collapsed to one moment.
- Else (the tracker lost the ball for longer than DT_PAIR): TWO events,
  at end(A) ("arrive", flagged gap) and start(B) − OFF ("depart", flagged
  gap; OFF corrects the seed delay after a contact — a new flight cannot
  start until the ball is clear of the paddle).
- The FIRST flight's start is an event at start − OFF (the serve). The
  LAST flight's end is NOT an event (the ball dying, or the tracker
  losing it, is not a hit; the truth ledger has no event there).
- Type (SECONDARY, reported only, not gated): "bounce" if the 3D height
  of A at its end ≤ BOUNCE_Z = 0.3 ft, else "hit". Bounce typing is a
  later registration; the same rule as pathfirst.boundaries.

## Truth and metric

Truth events per rally = oracle paddle contacts `c["imps"]` ∪ human
bounce times (`h_segs` with kind == "bounce", field `ts`): r6 10, r7 13,
r9 45, r10 39. Match events to truth greedily one-to-one by |dt| within
TOL = 0.10 s. recall = matched / truth, precision = matched / emitted,
F1 = harmonic mean. Per rally and pooled.

Baselines, same metric, same truth:
- RAW = every flight boundary as an event (what the overlay showed;
  the state before this layer).
- PROD = the production contact detector (`prod_contacts(c, series, 0.5)`)
  — contacts only, graded against the same contacts∪bounces truth AND
  against contacts-only (reported both ways so the bounce share of the
  truth is visible).

## Tune (r6 + r7 only, oracle contacts and human bounces of the TRAIN rallies)

Grid: R_SEAM ∈ {8, 16, 30} px × A_SEAM ∈ {20, 40} deg × DT_PAIR ∈
{0.25, 0.40} s × OFF ∈ {0, 0.06} s — 24 cells. DT_SEAM fixed 0.12 s,
TOL fixed 0.10 s, BOUNCE_Z fixed 0.3 ft.
Rule: max pooled F1 over r6 + r7; ties → smaller R_SEAM, smaller A_SEAM,
smaller DT_PAIR, OFF 0 (the cell that changes least). The chosen cell
must beat RAW's pooled F1 on r6 + r7, else DEAD and `grade 9|10` refuses.

## Bars (one shot each on r9, r10 — no re-tune, no knob override)

PRIMARY: F1 ≥ RAW's F1 on the same rally AND recall ≥ RAW's recall − 0.05
on BOTH rallies (a merge layer can only lose recall by collapsing two
true events; that is bounded, precision is where it must win).
NULL: the emitted event list circularly time-shifted by U(2, 4) s within
[serve − 0.4, end + 0.2], 200 shifts, seed 20260902: report mean and sd
of null F1; the measured F1 must exceed mean + 3 sd. Event density makes
chance matching substantial (tolerance 0.2 s wide against ~1 truth event
per 0.6 s), so the null is a floor to print, not a formality.
SECONDARY (reported): typed hit/bounce counts vs the ledger (29/16 on
r9, 26/13 on r10); per-event table (t, kind, dt to nearest truth, flag);
median |dt| of matched events; the PROD baseline both ways.

## Discipline

r9/r10 truth read once each by `grade`. No tracker output is a label.
The adopted track is not modified; if it changes, the assert fails and
the run is void. Temporal-gate holdout untouched.

## v1 results (2026-09-02; events_grade_r{9,10}.txt) — FAIL on the recall guard

Tune: RAW pooled F1 0.542 on r6+r7. The seam knobs (R_SEAM, A_SEAM) were
INERT on both train rallies — no consecutive flights there fall within
DT_SEAM with a small seam error — so the rule froze them at their
smallest values (8 px, 20°) untested; the two knobs that moved were
OFF (0 → 0.06 lifted r6 from 4 to 6 matched) and DT_PAIR (0.25 beat
0.40). Frozen: r_seam 8, a_seam 20, dt_pair 0.25, off 0.06; pooled F1
0.667. `events_tune.json`.

| rally | truth (C+B) | RAW | events v1 | PROD (all / contacts-only) | null F1 |
|---|---|---|---|---|---|
| r9 | 29+16 = 45 | 67 emitted, 35 matched, R .778 P .522 F1 .625 | 44 emitted, 32 matched, R .711 P .727 **F1 .719** | .525 / .594 | .297 ± .063 |
| r10 | 26+13 = 39 | 55, 29, R .744 P .527 F1 .617 | 42, 27, R .692 P .643 **F1 .667** | .627 / .556 | .326 ± .085 |

Track check passed both (r@12 537 / 422 == adopted). F1 bar and null
bar pass on both; the RECALL guard fails on both — r9 0.711 vs the
floor 0.728, r10 0.692 vs 0.694 (3 and 2 truth events lost). Bars
never loosen: **v1 is NOT adopted; RAW boundaries remain the state on
record.** Typed (secondary): bounce right/wrong 4/5 on r9, 3/1 on r10 —
the z-rule is a coin flip, as expected before its own registration.

Autopsy (READ FROM THE r9/r10 TABLES — this is evaluation truth being
used to form a hypothesis, disclosed as such): every lost truth event
sits under a "pair" whose seam error e is 100–360 px (r9 256.51 e 226,
259.92 e 184, 262.05 e 241, 265.99 e 228, 272.62 e 357; r10 295.05
e 116, 299.03 e 114). A short gap whose two arcs do NOT meet in the
image plane is not one hit seen twice; it is two events (a bounce and
the hit that follows it within a quarter second). The arrive/depart
pairs that ARE one hit have e ≤ ~80 px. Separately, "depart" events
after a long gap are late by more than OFF (the late re-acquisition
the owner saw): that is coverage (to-do 0b), not this layer.

## v2 registration (frozen 2026-09-02, before any v2 number)

One added check: a gap ≤ DT_PAIR is ONE event only if e ≤ E_PAIR; a
short gap whose arcs do not meet holds TWO events (arrive at end(A),
depart at start(B) − OFF), exactly as a long gap does. Everything else
as v1, with v1's r_seam 8 / a_seam 20 / dt_pair 0.25 carried frozen.
Grid: E_PAIR ∈ {40, 80, 150} px × OFF ∈ {0.06, 0.10, 0.14} s (OFF
re-opened because r6's serve and lone departs read late at 0.06;
tuned on r6/r7 only). Rule: max pooled F1 on r6+r7; ties larger E_PAIR
(fewer splits), smaller OFF; must beat BOTH RAW (0.542) and the v1
cell's pooled F1 on r6+r7, else DEAD. Bars on r9/r10 unchanged from
v1 (F1 ≥ RAW, recall ≥ RAW − 0.05, F1 > null mean + 3 sd, both
rallies). The hypothesis was formed on the r9/r10 tables; the knobs are
tuned on r6/r7 only; the r9/r10 shot is the SECOND on this truth for
this layer and is recorded as such. `events.py tune --v2`,
`grade <r> --v2`.

## v2 results (2026-09-02): DEAD on r6/r7, r9/r10 NOT run

RAW 0.542, v1 cell 0.667; best v2 cell 0.653 (E_PAIR 80). Splitting on
seam error hurts r7: a hard volley reverses the ball, so the two arcs
extrapolate far apart even when it is ONE hit (r7 emitted 13 → 17 for
10 vs 9 matched). Extrapolation error is a direction-change measure,
not a "two events" measure. `events_tune_v2.json`.

## v3 registration (frozen 2026-09-02, before any v3 number) — owner's framing

Owner, watching the overlay: "the double hit should maybe be physical
distance rather than temporal distance". A ball on a paddle does not
travel; a bounce-then-hit does. So: a gap is ONE event iff the distance
between where A's ball ARRIVED (A's last tracked point) and where B's
ball LEFT (B's first tracked point) is ≤ D_PAIR feet, converted from
pixels with the local px/ft scale at A's end point (a 1-ft offset
projected at that 3D location — depth error barely moves the scale,
unlike the 3D position itself, per the owner's confirmed wrong-side
read), AND the gap ≤ DT_CAP = 0.5 s (sanity only: a dink exchange can
bring the ball back near the same spot after ~0.6 s+). Otherwise two
events (arrive at end(A), depart at start(B) − OFF). Seam rule as v1
(r_seam 8, a_seam 20, frozen, still untested on train — no lobs in r6/r7,
owner-confirmed). Grid: D_PAIR ∈ {1.5, 2.5, 4} ft × OFF ∈ {0.06, 0.10}.
Rule: max pooled F1 on r6+r7; ties smaller D_PAIR, smaller OFF; must beat
RAW (0.542) AND the v1 cell (0.667), else DEAD. Bars on r9/r10 as v1.
This would be the second r9/r10 shot for this layer (v2 never fired).
`events.py tune --v3`, `grade <r> --v3`.
