# Path-first ball tracker — pre-registration (frozen 2026-09-01, before any number)

Owner's framing (2026-09-01): "slide the library paths over the raw
candidate blobs with no contact guess at all, find the stretches where a
book path lines up with a run of blobs, and then read the contacts off
the ends of the matched path." Every tracker so far is corridor-first:
contacts are guessed from pose (paddle nearest the decode), a line is
drawn between them, a box is wrapped around it, and the ball is searched
inside the box. The geometry fix (geom_fix.py) showed the box is the
miss on lobs and bounces and that a mis-merged corridor (missed contact)
destroys a whole flight. Path-first removes the contact detector from the
loop entirely: flights are found from the blobs, contacts are their ends.

## Instrument: `pathfirst.py`

Inputs (nothing from contacts, nothing from pose except the body
damping already in the incumbent): the whole-frame candidate cache
`cands_r{r}_cc_14.npz` with learned p (`p_r{r}_cc_14{_x}.npz`, cross-fold
`_x` on r6/r7), the shared camera P (identical across the four rallies),
the pose-extremity points (candidates within 16 px of an extremity are
damped exactly as the incumbent damps them).

1. HYPOTHESES. Seeds = top-N_SEED candidates per frame by p with p >=
   P_SEED, not near a body extremity. For every seed at frame f1, every
   seed at f3 = f1 + D (D in {12, 24, 40} frames), and every seed at
   f2 = f1 + D/2 + {-2, 0, +2}: solve the unique drag-free ballistic arc
   through the three image points at their three times (6 linear
   equations in p0, v0). Keep arcs whose launch state is physical: z in
   [-0.5, 12] ft over the span, speed 10-110 ft/s, inside the court
   volume [-10,30] x [-10,54] ft.
2. SUPPORT. Project each arc at every frame of its span (extended by
   D/2 each way); support = sum over frames of the p-weighted kernel of
   the nearest candidate (kernel = (1 - d/8 px)+ * (0.2 + 0.8 p), body
   damping 0.3), minus the per-frame random-probe baseline (spaghetti's
   frame_base idea: what a random pixel collects). Keep arcs with support
   >= S_MIN; non-max suppress arcs whose spans overlap > 60 % and whose
   projections agree within 12 px.
3. GROW + REFINE. Each surviving arc is refit with drag
   (court3d.fit_arc, 7 params) on its inlier candidates (within 8 px,
   weight p) and grown frame by frame in both directions: a candidate
   within R_GROW = 10 px of the extrapolated arc joins; refit every 6
   joins; stop after GAP consecutive frames without one. The grown span
   is the flight; its ends are the flight's boundaries.
4. SELECT. Greedy by support density (support / span), no two selected
   flights overlapping by more than 3 frames. The TRACK is the projected
   arc at every frame of every selected flight. CONTACTS = flight
   boundaries; a boundary where the arc reaches z <= 0.3 ft and the next
   flight starts within 0.15 s and 2 ft is a BOUNCE, not a contact.
5. PRIOR. Only a weak launch-state plausibility prior from the r6/r7
   entries of the shot book (13 launches; the book's r9/r10 entries are
   NOT used — see leak note). The physics bounds above do the real work.

## Knobs and the rule (tuned on r6/r7 ONLY, cross-fold p)

Grid: P_SEED in {0.4, 0.6} x S_MIN in {4, 6, 8} x GAP in {6, 12}
(N_SEED = 4 fixed). 12 cells. Selection RULE: max total r@12 over r6+r7
subject to pooled prec@12 >= the incumbent PROD arm on the same caches
(r6 prod 90/125 + r7 prod 115/204 = 205 @ 0.623); ties -> larger S_MIN,
then smaller GAP, then larger P_SEED (the more conservative cell). None
-> DEAD, `grade 9|10` refuses to run. The comparison arm is the
incumbent PROD track (dp-ccS+body, W_P_SOFT=25, its own contact
detector): path-first uses no contacts, so the production contact
detector is the fair opponent; the oracle arm is reported for context
only.

## Bars (one shot each on r9, r10 — no re-tune, no knob override)

PRIMARY: r@12 and prec@12 vs the incumbent prod arm on the same caches
(r9 431 @ 0.69; r10 325 @ 0.69 on the re-cut clip). ADOPT only if r@12
>= incumbent AND prec@12 >= incumbent - 0.02 on BOTH rallies. SPLIT or
worse -> record, incumbent stays.
NULLS (both required <= 5 % of clicks): (a) the emitted track displaced
by (+-U(160,240), +-U(80,140)) px, seed 20260901; (b) the emitted track
circularly time-shifted by U(2,4) s (a rhythm-only tracker would survive
(a) and fail (b)).
SECONDARY (reported, not gating): contact recovery — for each oracle
contact time (c["imps"]) the |dt| to the nearest path-first boundary
(median, share within 0.10 s), next to the same numbers for the
production contact detector (prod_contacts). Bounce count and locations
vs the human bounce ledger on r6/r7.
STRATA (reported): r@12 by the incumbent's own strata (cand / nocand /
outwin / nocor) so it is visible WHERE path-first wins or loses.

## Leak note

`launch_prior.json` (2026-09-01) was harvested from rallies 6, 7, 9 AND
10 (r9 25 launches / 12 bounces, r10 23 / 10, r6 5 / 2, r7 8 / 3).
Spaghetti's mode prior (`prior_pen`, W_MODE) therefore reads the
evaluation rallies' own shots; spaghetti's r9/r10 numbers on record are
optimistic by an unmeasured amount. Path-first uses the r6/r7 entries
only. The book should be rebuilt r6/r7-only before any spaghetti
re-registration.

## Discipline

Truth (ball_path_r{r}.csv V/S clicks) is used for grading only; the
tune reads r6/r7 truth, r9/r10 truth is read once each by `grade`.
No tracker output is ever used as a label. Temporal-gate holdout rows
untouched.

## Results (2026-09-01, after the freeze above; numbers in pathfirst_grade_r{9,10}.txt)

Self-test: planted drag-free arc among 20 junk blobs/frame recovered
61/61 frames (one fix during bring-up, BEFORE any tune number: the
drag refit was seeded at k=0.3, which bent a drag-free hypothesis off
its own inliers; seeded at k=1e-4 instead, and a refit needs >= 6
inliers). No other change between the freeze and the shots.

Tune (12 cells, r6+r7, cross-fold p): EVERY cell clears the incumbent
prod arm (205 @ 0.623); range 220 @ 0.815 .. 263 @ 0.807. Rule picked
**p_seed=0.4 s_min=6 gap=6 (263 @ 0.807)** (tie with s_min=4 broken
to the larger s_min). `pathfirst_tune.json` holds the grid + verdict.

One shot each:

| rally | incumbent prod | path-first | V / S | nulls disp / tshift | at-click pts |
|---|---|---|---|---|---|
| r9 (779 clicks) | 431 @ 0.69 | **537 @ 0.87** | 421/587 @ 0.94, 116/192 @ 0.69 | 0 / 1 | 616 vs 628 |
| r10 (657, re-cut clip) | 325 @ 0.69 | **422 @ 0.88** | 316/487 @ 0.92, 106/170 @ 0.78 | 0 / 0 | 479 vs 472 |

Both PRIMARY bars pass on both rallies (r@12 +106 / +97, prec +18 pp /
+19 pp); both nulls <= 1 % of clicks. **ADOPTED: path-first is the
incumbent.** Strata (incumbent prod geometry): r9 cand 441 vs 431,
nocand 12 vs 0, outwin 82/131 vs 0, nocor 2/2; r10 cand 338 vs 325,
nocand 10 vs 0, outwin 74/97 vs 0 — the gain is the out-of-window
clicks (lobs, bounces) the corridor box could never reach, plus a
small selection gain inside the box.

SECONDARY (not gating, reported as promised): oracle-contact recovery
|dt| to nearest path-first boundary — r9 median 0.046 s, 24/29 within
0.10 s (prod detector 0.069 s, 19/29); r10 0.048 s, 19/26 (prod 0.074,
15/26). Caveats that keep this secondary: (i) it is RECALL of oracle
contacts only — path-first emits 64 / 56 boundaries against 29 / 26
oracle contacts, so many boundaries are bounces, flight fragments or
misses, and no precision is claimed; (ii) the bounce rule (z <= 0.3 ft,
next start <= 0.15 s and <= 2 ft) fired 2 times on r9 and 0 on r10
against a human ledger of 13 bounces on r10 — bounce TYPING is
essentially unbuilt; the flights break at bounces, the labels do not
say so. r6 smoke (train, overrides allowed): 0 bounces typed.
Runtime: ~3 s per rally.

What this does NOT establish: track quality on frames with NO click
(the V/S ledger is the only truth; between-click frames are unscored),
contact precision, bounce typing, r20 (the seal is untouched and needs
owner authorization).
