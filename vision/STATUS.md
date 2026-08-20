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

These share one property: **a flat chance baseline, or no tolerance to
widen.** That is exactly why the 2026-08-20 null work could not touch
them. It killed placement in TIME; none of these is a timing claim.

## What is dead

| thing | verdict | why |
|---|---|---|
| contact TIMESTAMPS, every instrument | dead | see below |
| ball detection / tracking | closed | 64% in-play findability, physical |
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

## Built, producing numbers

**Coverage model** (branch `claude/court-coverage-model-8rg94l`), one
match, 90 of 141 rallies: width share (Alshon .549 / Black .451),
90%-contour area (Alshon 261 > Black 232 > Patriquin 204 > Bright 188
ft2), off-court fraction (Black 6.8% vs Alshon 3.3%), depth and
kitchen-band occupancy. Three findings already withdrawn by its own
checks (deep poach, crossing rate, ellipse area) — the instrument
polices itself.

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
