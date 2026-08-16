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
5. Identity — the SERVE-ANCHOR chain, re-derived here so no session
   has to: the referee log names the server and receiver of every
   rally, and lineup.py's state machine supplies those two names for
   all 45k rallies at 99.25% measured accuracy. At each serve, the
   rules position-pin both of them: the server is the track serving
   from the rule-determined half (behind the baseline, hits shot 1),
   the receiver is the diagonal track. Each remaining track is simply
   "the partner" on its side — so all four names resolve WITHOUT
   predicting where partners stand (stacking swaps partners' halves
   but never touches identity, because tracks give physical position
   and the anchor only needs server + receiver). Track continuity then
   carries the names through the rally, and the NEXT serve re-anchors
   — identity errors are rally-local by construction and can never
   propagate. Mixed doubles adds a redundant height-prior check.
   Confidence-weighted; ambiguous spans (camera cut mid-rally, anchor
   frame unclear) are dropped, not guessed — aggregate metrics
   tolerate gaps.
6. Per player per rally: occupancy distribution on the court plane.

## Frame hygiene (the gaps a cold session would fall into)

- **Main-camera gating**: broadcasts cut to close-ups/low angles
  mid-rally; those frames must be DETECTED and EXCLUDED before
  projection (the homography only holds for the main elevated angle).
  Detection: scene-change + court-line reprojection sanity (the main
  camera is static — frame-to-frame motion median 0.64 grey levels per
  the POC — so a cut is loud). Validate the gate on a hand-checked
  sample before trusting it.
- **Rally-active span**: coverage counts ONLY frames between the serve
  anchor and the rally's end (scorebug flip, frame-exact); between-
  rally wandering poisons every metric. This needs the spec's ONE new
  algorithmic piece: an **anchor-frame finder** for unstamped rallies
  (within each inter-flip window, the frame where a serving-side track
  occupies the service stance zone in court coordinates with low
  all-player motion). Ground truth to validate it exists already: the
  contact thread's stamped serve times — measure the finder's error
  distribution there before scaling out.
- **Phases**: the serve stance (server frozen at the baseline) is not
  "coverage" — either exclude the first ~2 s of each rally or report
  serve-phase and rally-phase separately; pick before data.
- **Glitch rejection**: a physical speed gate on court-plane
  trajectories (feet do not move 7+ m/s) drops keypoint teleports
  before they enter any aggregate.
- **DreamBreakers are EXCLUDED** (house rule: DBs are singles and
  never enter doubles models; singles coverage would be its own,
  separate cut).

## Metrics (report per player per match, aggregate per season)

Pre-register the formulas BEFORE the first real number is looked at
(house style); the definitions below are the proposal to freeze:

- **Coverage area**: 90% occupancy ellipse of rally-active foot
  positions, ft², per player per game.
- **Width share** (the w observable): per frame, the partner midpoint's
  lateral offset from the team's court centerline, time-averaged and
  mapped to a [0,1] share; secondary definition = the player's fraction
  of the team's combined ellipse area. Team pairs sum to 1 by
  construction. HONESTY CLAUSE: v2's w = 0.4085 is a responsibility
  weighting inferred from OUTCOMES; the video quantity is a geometric
  PROXY. The deliverable is the correlation between the two across
  pairings — never a claim that they are the same number.
- Depth range (baseline↔kitchen transitions), distance traveled per
  rally, static stance vs dynamic coverage split.
- Cuts: gender within mixed; by pairing (Waters/partner across
  partners); size outliers (Alshon); serve vs receive.

## Publication gates (same bar every other trait had to clear)

- **Reliability first**: before any player-level coverage number is
  published as a trait, it passes the house split-half battery (random
  halves + across-eras where data allows) — the same gate clutch and
  the singles surplus cleared and wind skill failed. Single-match
  leaderboards are never quoted; report per-player values with
  between-match sd and a minimum-matches threshold.
- **Cross-gender scope (user correction, 2026-08-16)**: width share
  and every other coverage quantity are DIRECT OBSERVABLES — measured
  off the video with no prior in the derivation — so they compare
  across genders and are publishable as fact once validated ("men
  cover X% more ground than women in mixed" is a fair, reasonable
  statement of measurement). The likelihood-flat house rule protects
  RATINGS (outcome-inferred, offset-conventional), not measurements.
  Two boundaries that do stand: (1) coverage never feeds ratings —
  it is descriptive, a player-page axis like the singles surplus, not
  a model input; (2) any analysis correlating coverage against a
  RATING-DERIVED quantity in mixed (e.g. within-pair rating gaps)
  inherits the offset convention through the ratings side — run those
  within gender or flag the convention.

## Output data model

Committed: data/coverage_players.csv (one row per player-game:
ellipse area, width share, depth range, distance, phase splits,
identity-error rate from the overlay spot-check, frames kept/dropped
by each gate) + data/coverage_events.csv (per-VOD ledger: homography
calibration residual, main-camera gate stats, rallies covered/excluded
with reasons). Pose npz and overlay renders stay local (gitignored /
broadcast-imagery rule).

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

**Status**: BUILT 2026-08-16 (branch claude/court-coverage-model-8rg94l;
first real-VOD run pending — see the PR + build record below). The
contact-thread instruments were not touched.

## Build record (2026-08-16)

- `vision/coverage_windows.py` — flip-train x referee-timeline windows
  for ANY fresh VOD (scorebug_windows.align reused; approx flags carry
  missed-flip neighbourhoods, DP instability, and claimed replay
  inserts). `--selftest`.
- `vision/coverage.py` — camera gate (`--scan-camera`), off-court +
  speed gates, anchor-frame finder (last pre-flip freeze, motion over a
  0.5 s baseline, per-window adaptive quiet threshold), serve-anchor
  identity chain (geometry + logged names only; lineup halves, mixed
  height prior, and the per-game end-map are report-only checks),
  pre-registered metrics (frozen in its docstring BEFORE any real
  number existed), committed-CSV writers. `--selftest` covers the
  previous-rally-freeze trap, handoffs, the 0.30/0.70 width-share
  reproduction, and the ellipse against the analytic value.
- `vision/coverage_overlay.py` — the verification instrument (built
  first per this spec): names/boxes/skeletons/court-inset, dimmed
  low-confidence labels, exclusion banners, `--sample N` + spotcheck
  template that coverage.py folds back in as the identity error rate.
- `vision/coverage_ab.py` — the backend agreement guard, mechanical
  (run before any scale-out; ViTPose wins disagreements).
- `bash vision/coverage_pipeline.sh <vod> <match_uuid> <vod_id> ...`
  runs the whole chain idempotently (test target: the 2026-01-25 PPA
  Indoor Nationals mixed final, YouTube SQg2mHBPHC0, match c4eb30d0 —
  Bright/Patriquin vs Black/Alshon, three of the spec's named cuts in
  one match).
- FOUND while wiring: PPA referee logs can DESYNC the lineup state
  machine (c4eb30d0: receiver prediction 65.2% overall, but the misses
  are two long runs — games 1-2 — with game 3 at 36/37 and only one
  impossible row, so the LOGGED server/receiver names stay sound; the
  99.25% figure is MLP-2026-measured). The identity chain is built to
  survive this: it consumes logged names + geometry only, and the R/L
  halves check weights itself by the machine's local receiver_ok
  agreement.
