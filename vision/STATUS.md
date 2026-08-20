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
match, 90 of 141 rallies. Numbers below are frame-weighted over the
MATCH and re-read from the committed CSVs on 2026-08-20; two figures in
the first version of this entry were stale and are corrected here.

| measure | value | source |
|---|---|---|
| width share | Alshon .552 / Black .448; Patriquin .561 / Bright .440 | coverage_players.csv |
| 90% contour area | Alshon 261 > Black 232 > Patriquin 204 > Bright 188 ft2 | coverage_heatmap |
| off-court fraction | Black 4.7%, Alshon 3.5%, Bright 3.0%, Patriquin 1.5% | coverage_dominance.csv |
| depth (median from net) | Patriquin 10.7 ft, Bright 12.0, Alshon 15.5, Black 15.5 | coverage_heatmap |
| kitchen-band occupancy | Patriquin 28.8%, Alshon 14.8%, Bright 9.7%, Black 4.6% | coverage_heatmap |

CORRECTED HERE: width share was quoted as .549/.451, which is GAME 1,
not the match; off-court was quoted as "Black 6.8% vs Alshon 3.3%",
which is the 63-rally figure from before anchor-free identity took the
sample to 90 — the committed table says 4.73% vs 3.51%, narrowing that
gap from 2.1x to 1.35x. The 6.8% number had already been retracted in
the PR body before this file was written; it regressed into the ledger
that exists to stop exactly that. Re-read the CSVs, do not copy prose.

FOUR findings withdrawn by the instrument's own checks — deep poach
(counts crossings, not intent), the 22.8% crossing rate (stacking
contaminated the anchor; 12.8% under a settled anchor), ellipse area
(overstates 1.14-1.54x; a court is bimodal and one Gaussian spans the
gap), and the off-court gap above. Three of the four were caught by the
USER looking at overlays or questioning a definition, not by the code.

IDENTITY IS INDEPENDENTLY CHECKED (`vision/coverage_idcheck.py`):
label-swapping a fraction q of rallies gives TV = |1-2q| x TV_true
between occupancy maps, and cross-team pairs are clean at 97-100%, so q
is identifiable. Partners measure 0.8060 and 0.7716 against a clean
man/woman reference of 0.7920 — implied q of -0.9% and +1.3%, no
detectable swap, independent of Gate A and of the classifier that
assigned the labels. POWER WARNING: this works because mixed partners
differ by ROLE. Same-gender pairs sit at TV ~0.25, so in men's or
women's doubles the check loses ~3x power AND the human eye loses power
for the same reason, simultaneously — the two-big-men-on-one-MLP-team
case (Bar/McGuffin, Staksrud/Sock) is where every channel fails at once.

STILL PROVISIONAL: three user verification gates are open (anchor-free
overlay spot-check, v4 overlay check, match watch). And the shipped
numbers need a gitignored pose extraction that does not survive a fresh
container — see coverage_spec.md.

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
