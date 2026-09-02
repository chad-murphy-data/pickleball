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
