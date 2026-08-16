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

## Backend choice — gold standard is PER-TASK, decided on merit

Use **RTMPose-balanced** (`pose_extract.py --backend rtmpose`) — and the
reasoning must be stated so a future thread doesn't mistake it for the
convenience pick the contact thread had to correct (contact_gate.md
Amendment 2): Gate C is a ONE-SHOT ~15k-frame measurement whose verdict
hinges on wrist precision during motion blur, so its instrument is the
strongest model that runs (ViTPose-plus-huge). Coverage is the opposite
regime on both axes: a BULK pass over full matches (eventually many
VODs — ViT-huge would cost GPU-days for nothing), measuring a SLOW
AGGREGATE of foot positions where per-frame keypoint noise averages
out and the binding accuracy terms are homography calibration and
identity/side attribution, not keypoint AP. RTMPose is the merit
choice here, not the easy one.

**Pre-named validation guard (run before trusting the fleet)**: extract
a handful of rallies with BOTH backends and confirm the coverage
metrics (ellipse area, width share) agree within a few percent. If
they disagree, the ViTPose number wins and the discrepancy gets
diagnosed before any scale-out.

## Prerequisites for a fresh session

- **PR #57 must be merged** (or the session works on branch
  `claude/pickleball-swing-detection-6mw3xk`): pose_extract, the gate
  docs, and this spec all live there — a fresh clone of main has none
  of it.
- **Rally windows at scale come from the scorebug reader**
  (`vision/scorebug_read.py`, frame-exact flip sync — the keeper
  asset), NOT from the cheer join or grammar chains (both failed; see
  data/vision/pin_realignment.md).
- **EXCLUDE REPLAYS** — the recurring alignment trap (2026-08-16
  lesson): broadcasts re-air rallies at full speed, and a replayed
  rally would double-count coverage frames. Detection: a segment whose
  scorebug state duplicates or fails to advance the score sequence is
  a replay; only score-advancing segments are live. The scorebug
  reader makes this check mechanical.
- Sequencing: do NOT touch the contact-thread instruments
  (contact_gate.md is pre-registered and mid-measurement); coverage is
  a parallel read-only consumer of pose_extract.

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

## Verification overlay (user-requested, 2026-08-16 — build it FIRST)

Render an annotated copy of the video the user can just watch: boxes +
skeletons on all four players, each labeled with the RESOLVED PLAYER
NAME (team-colored), track id, near/far side, and the foot point — plus
a small schematic-court inset showing the four projected dots moving in
court coordinates (that inset verifies homography AND identity at once,
and court coordinates are where every coverage metric lives). Dim/flag
a player's label when identity confidence is low, so the eye goes
straight to the moments the machine is guessing; banner replay-excluded
segments visibly. Optional 0.5x render for checking.

This is the identity layer's human-verification instrument, in the
house pattern (validate on the cases being scored; ask the human where
the machine is guessing): the ONE failure coverage cannot self-detect
is within-side partner swaps mid-rally, and a name label jumping
between partners is instantly visible to a human who knows the
players. Make the check a MEASUREMENT, not a vibe: sample ~10 random
rallies per game for the user to watch and record swaps found → a
per-rally identity error rate with a denominator, before any
leaderboard is quoted. Precedent: swing_probe's --smoke debug frames
(same drawing, single frame). Overlay videos are broadcast-derived
imagery: LOCAL ONLY, never committed (same rule as data/vision/*.png).

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
