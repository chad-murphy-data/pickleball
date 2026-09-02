# Hand-off seeding — pre-registration (2026-09-02, before any number)

## Why
Owner's overlay observations on the adopted path-first track: the ball
is lost for about two exchanges per rally, always CLOSE TO A PLAYER, and
the down-the-line speed-up is lost in both eval rallies. Two frozen
design choices cause this, not the knobs: a flight cannot SEED from a
blob within NEAR_BODY (16 px) of a pose extremity (the zero-false-track
guarantee), and the shortest seed span is 12 frames (0.2 s), which a
fast short flight never offers cleanly. The events grade agrees:
nearly every remaining miss is an arrive/depart pair around a lost gap.

## Rule (handoff.py; pathfirst.py untouched)
Pass 1 = the adopted path-first run (frozen cell, unchanged).
Pass 2 = for each pass-1 flight END (both ends of every flight), a
hand-off ZONE: frames within W of the end, candidates within R_ZONE px
of the flight's last tracked point. Inside the zone the near-body
exclusion is LIFTED for the first seed of a hypothesis and the seed
p-threshold is P_HAND; the outer-seed spans are DS_HAND = (6, 8, 12)
frames. Hypotheses are solved, checked and supported exactly as in
pass 1 (solve_arcs / plausible / support / nms / grow with the frozen
gap). New flights are admitted only where they do not overlap a pass-1
flight by more than OVERLAP_SEL frames, greedily by density (pf.select
on the new flights alone, then appended). The track is the projected
arc as before. The door opens only next to where the ball was last
seen, for a short time.

## Tuning (r6 + r7 ONLY, cross-fold p-caches)
Grid: R_ZONE ∈ {40, 70} px, W ∈ {18, 30} frames, P_HAND ∈ {0.25, 0.4},
S_MIN_HAND ∈ {3, 4} (support threshold for the short hypotheses).
16 cells. Selection rule: highest pooled r@12 on r6+r7 subject to pooled
prec@12 ≥ incumbent path-first prec − 0.03 and both nulls (displaced,
time-shift) r@12 ≤ 3; ties → smaller R_ZONE, then smaller W.
Incumbent path-first on r6+r7: 263 @ 0.807.

## One shot (r9, r10) — bars, never loosened
ADOPT only if on BOTH rallies: r@12 strictly above the incumbent
(537 / 422); prec@12 ≥ incumbent − 0.02 (0.85 / 0.86); displaced and
time-shift nulls r@12 ≤ 3; and the adopted events layer re-run on the
new flights keeps F1 ≥ adopted − 0.03 (.701 / .645) — the layer above
must survive. Anything else = NOT ADOPTED, recorded, incumbent stays.
Secondary (reported, not a bar): at-click coverage, r@12 in the V and
S strata, number of new flights, events recall.
Owner clicks on r9/r10 are used for grading only.

## Results — tune (2026-09-02, r6 + r7, 16 cells, handoff_tune.txt)
Every cell keeps precision (0.771–0.810) and both nulls at 0/2. The
gain is SMALL and all of it is r6: r6 goes 125/127 → 134/138 in the
r_zone=70 cells (+3 flights), r7 stays at 143/204 with 0 new flights in
12 of 16 cells (+1 flight, 146/225, in the two w=30 / s_min 3 cells,
which drop precision to 0.771). Pooled best under the rule:
**r_zone 70, w 18, p_hand 0.25, s_min_hand 3 → 277 @ 0.810** vs the
incumbent 263 @ 0.807 (+14 frames, +5%). Selected per the rule; the one
shot on r9/r10 follows (handoff_grade_r{9,10}.txt). Read honestly before
the shot: the door opens but very little walks through it on the
training rallies — the short-span seeds near players mostly do not
survive plausibility + support at these thresholds.

## Results — the one shot (2026-09-02, r9 + r10, handoff_grade_r{9,10}.txt)
Cell r_zone 70 / w 18 / p_hand 0.25 / s_min_hand 3, frozen from the
tune above. Five new flights per rally, all 0.13–0.47 s long.

| | r9 | r10 |
|---|---|---|
| path-first (adopted) r@12 / prec | 537 / 0.87 | 422 / 0.88 |
| hand-off r@12 / prec | **563 / 0.87** (+26, +4.8%) | **446 / 0.89** (+24, +5.7%) |
| r@8 | 416 → 439 | 327 → 350 |
| at-click coverage | 616 → 650 / 779 | 479 → 503 / 657 |
| V stratum r@12 (prec) | 439 (0.93) | 340 (0.93) |
| S stratum r@12 (prec) | 124 (0.70) | 106 (0.78) |
| nulls displaced / time-shift r@12 | 0 / 2 | 0 / 1 |
| events v3 on the new track F1 (bar) | **0.758** (≥ .701 ✓; adopted .731) | **0.636** (≥ .645 ✗; adopted .675) |
| events recall / prec | 0.800 / 0.720 | 0.718 / 0.571 |
| verdict | PASS | **FAIL** (events bar, by 0.009) |

**NOT ADOPTED.** The bars were written as "both rallies"; r10 misses
the events bar. The incumbent path-first track and the v3 events layer
stay production. Recorded, not re-run, no bar loosened.

What the numbers say, honestly. The TRACK improved on both evaluation
rallies at unchanged precision and clean nulls, r8 (the tight radius)
included, so the new flights are real ball and sit where the ball is —
the coverage-near-players mechanism works. What it costs is the layer
above: five short flights add ten flight ENDS, and the events layer
pairs and labels those ends; on r9 that turned lost gaps into matched
arrive/depart events (recall .711 → .800, precision held), on r10 the
new ends fired more unmatched events than they matched (precision
.643 → .571) and recall gained only .692 → .718. The events layer was
tuned (v3) on the incumbent's flight-end statistics; a short flight
ending next to a player is a different kind of end than the ones it
learned its seam rule on. A re-tune of events on the hand-off track
(r6/r7, then one shot) is the obvious next gate if this is ever
reopened; that is a NEW pre-registration, not a knob turn here.

Consumers are NOT wired to the hand-off track. handoff_tune.json stays
on disk as the record of the tuned cell; `handoff.py grade` is spent
for r9/r10.
