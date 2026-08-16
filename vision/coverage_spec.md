# Court coverage per player — spec (2026-08-16, user request; unbuilt)

**The ask**: track all four players and measure how much court space each
takes up during a match. How much court does an average man cover in a
mixed match? A giant like Christian Alshon? Anna Leigh Waters' partner?

**Why this is the FIRST vision analytics with no research risk**: it
needs only the layers that are already measured and working —
identity-continuous tracks (`vision/pose_extract.py`, selftested;
RTMPose production spine is the right backend at scale), court
homography (`vision/court.py`, 0.06 ft median residual), and player
identity from the referee log (`vision/lineup.py`, 99.25% — which half
each player owns at every serve). No ball (dead thread), no contact
detection (Gate C-gated), no training. Coverage is a slow AGGREGATE, so
it is robust to the per-frame blips that make fast-event work hard.

**And it is not just a stat — it is finding 11's dial, observed.**
v2's weakest-link γ is a court-coverage dial: w = (1+γ)/2 = 0.4085 is
the stronger player's inferred share of the court, applied uniformly to
every team. The gap-exploit thread (null for a persistent skill, blind
to episodic tactics) could only test w through SCORES. This measures w
from VIDEO: width share at the kitchen line per pairing per match — the
freeze-out, the mixed gender split, the per-team deviation from 0.41 /
0.59, all as direct observation. If measured width share correlates
with the model's inferred within-pair rating gap, that is independent
validation of the weakest-link structure; where specific pairings
deviate (episodic game plans), that is exactly what the null test could
not see.

## Pipeline (every stage exists; the glue is the build)

1. Rally windows for a VOD (scorebug sync / stamped serves where
   available).
2. `pose_extract.py --backend rtmpose` full-game pass → tracks with
   near/far sides (GPU: tens of minutes per match).
3. Foot point per detection: ankle midpoint where confident, else
   box bottom-center.
4. Homography (court.py, calibrated once per broadcast setup) → court
   coordinates in feet.
5. Identity: near/far from tracks × which-half-at-serve from lineup.py;
   within-side disambiguation by track continuity + height prior
   (mixed) + serve-position anchors. Confidence-weighted; drop
   ambiguous spans (aggregate metrics tolerate gaps).
6. Per player per rally: occupancy distribution on the court plane.

## Metrics (report per player per match, aggregate per season)

- **Coverage area**: 90% occupancy ellipse, ft² (in-rally frames only).
- **Width share at the kitchen**: fraction of the team's 20 ft the
  player patrols — THE w observable. Team pairs sum to 1 by
  construction.
- Depth range (baseline↔kitchen transitions), distance traveled per
  rally, static stance vs dynamic coverage split.
- Cuts: gender within mixed; by pairing (Waters/partner across
  partners); size outliers (Alshon); serve vs receive.

## Caveats (carried from the record, not new)

- Championship-court VODs only — the permanent sample bias.
- Condensed VODs cover most rallies of a broadcast match; per-rally
  coverage is enough (no need for uncut footage).
- Freeze-out QUESTION (does targeting Waters' partner work) stays n=4
  per matchup — but coverage itself generalizes across every televised
  match; it measures the geometry, not the outcome.
- Camera-side occlusion inflates near-pair uncertainty slightly;
  aggregate metrics + confidence weighting absorb it.

**Status**: specced, unbuilt. Sequenced BEHIND the Gate C evening (do
not preempt the contact thread's labeling/verdict). Natural build
moment: right after the Gate C verdict, either way — coverage does not
depend on it.
