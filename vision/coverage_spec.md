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
identity-error rate from the overlay spot-check) +
data/coverage_events.csv (per-VOD ledger: homography calibration
residual, main-camera gate stats, rallies covered/excluded with
reasons, and the per-gate FRAME drop counts — frame gating precedes
identity resolution, so dropped frames cannot be attributed to a
player and live at the event level; build decision 2026-08-16). Pose npz and overlay renders stay local (gitignored /
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

- WINDOWS INSTRUMENT **SOLVED** (2026-08-17 night;
  bug_state_windows.py): 113/141 rallies with SCORE-VERIFIED windows
  (g1 36/48, g2 51/56, g3 26/37), orientation margin 113:16, all five
  check anchors frame-verified (r8 2-2·1 at 376-384, r10 3-2·2 at
  411-416, r74 5-2·1 at 1979-1984, r83 6-6·1 at 2163-2181, r84 7-6·1
  at 2180-2199). What it took, in order (each measured, none guessed):
  (1) canonical per-layout boxes — the locator wobbles ±2 px on a
  static overlay and every windowed-median drift faked a state change;
  the bug's three x0 modes (370/400/428) are COMPLETED-GAME COLUMNS
  appended after each game, so LAYOUT INDEX = GAME, and one canonical
  box per layout makes plateau crops pixel-identical (566→85 runs);
  (2) raw crop cache (.crops3.npz, gray+greenmask region stream) so
  every later iteration is offline; (3) dots are NAME-ANCHORED AND
  STATIC (region x 45-105), not score-column-relative — the relative
  band read the wrong place in 2 of 3 layouts; width fallback for
  touching dots (single 4-6 cols, double 8-13); (4) per-game ABSOLUTE
  value matching replaces global symbol alignment — leader-cluster the
  cell states, rank clusters by WEIGHTED INTERVAL SCHEDULING (true
  score states occupy disjoint time blocks; junk score-flash clusters
  recur across blocks and drop out — first-appearance ranking let one
  junk cluster shift every later value one off), rank k = k-th
  distinct logged value, rallies match by (top,bot,row,server#) dict
  key, so drift is structurally impossible; (5) dot-changepoint run
  splitting recovers side-out/second rallies (~half the match) whose
  only on-bug change is the dots; runs carry REAL per-frame timestamps
  and close at >8 s absences (linear interpolation over gap-spanning
  sparse runs fabricated overlapping pieces — a 67→12 collapse traced
  to exactly that). The 28 unmatched rallies stay approx=1/unmatched
  and are EXCLUDED from extraction (drop-don't-guess).
  LESSON THAT INVALIDATED OLD 'TRUTHS': the spacing-window positions
  for r8/r10 were ~100 s off, yet identity hand-checks 'confirmed'
  them — the same four players serve nearby rallies in similar
  directions, so identity checks CANNOT validate rally IDENTITY; only
  score reads off the bug can.

- ANCHOR-FINDER VALIDATION (2026-08-17, vs the Chicago serve pins,
  court-gated extractor at 10 fps): now resolves 14/15; median +1.41 s,
  IQR [+0.62, +2.57], 8/14 within 2 s. Upgrades that got there:
  serving_config3 (relaxed 2+1/1+2 config, FINDER ONLY — 5/15 true
  freezes run with ≤3 tracks and the strict 2+2 never saw them),
  post-freeze signatures gating every candidate (receiver return-rush
  surge ≥4.5 ft net-ward within 4 s; sustained play-onset motion when
  no receiver track survives; two-bounce hold-deep — the serving pair
  must stay deep 1.5 s after a real serve), and a signatureless
  fallback that reruns the OLD rule on strict-config runs only at half
  qual. Residual: 3 late anchors (+12..27 s) on the WIDE tuning
  windows, whose tails contain the next rally's pre-serve period by
  construction — production bug-state windows end at the score flip
  and structurally exclude that class; the survivors ride at reduced
  qual for downstream drop rules. Also: the frozen Gate C extractor
  anchors only 4/15 here (far-baseline starvation, see the 0.30 H
  cross-finding above) — anchor validation MUST run on
  coverage_extract output.
- CROSS-FINDING for the contact thread (2026-08-17, from the anchor
  validation): pose_extract's pixel gate rejects persons whose box
  bottom sits above 0.30 H, and BOTH test broadcasts frame the far
  baseline at ~0.26 H — far-baseline players (the server or receiver
  of every rally) never reach the pose model until they step forward.
  Chicago anchor validation only worked on the court-projection-gated
  coverage extractor. May bear on Gate C's 33% serve recall tell
  (v1 measured through the same gate); noted, instrument untouched.


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
  (run before any scale-out; ViTPose wins disagreements); every
  players/events row records backend provenance off the pose dir's
  meta.json.
- `vision/coverage.py --validate-anchor` — the anchor finder's
  ground-truth check against the 15 Chicago serve pins (rally 3
  excluded; needs the Gate C pose npzs + a Chicago court fit, i.e. the
  machine that holds the Chicago VOD). Numbers are an UPPER bound:
  label windows open 1.5 s pre-serve vs 6-20 s in coverage windows.
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

## Yield ceiling — MEASURED 2026-08-19: it is the SOURCE VIDEO

The first real run resolved 63 of 141 rallies and the obvious reading was
"the instruments are weak". Three candidate fixes were tested against
that reading. Two are now falsified and the third is redirected; record
the numbers so no future session re-buys them.

**The false diagnosis.** Cached tracks show a median 16 distinct track
ids per rally for 4 players and a median track lifetime of 7% of the
rally — which reads as catastrophic tracker shattering. It is not: the
MEDIAN track is junk. The four biggest tracks already live 81% of the
rally, and >=4 real tracks (alive >=1 s) are present in 92% of frames.
Any per-track average over this population is dominated by fragments.
Measure the top-4 tracks, never the median.

**A — 30 fps (from 10).** NULL. Top-4 lifetime 81% -> 83%, junk share
26% -> 25%, "4 alive" unchanged, at 3x the extraction cost. Do not.

**B — appearance-aware association** (`pose_extract` APP_* constants,
opt-in; `coverage_extract` defaults it on). REAL BUT SMALL: top-4
lifetime 81% -> 84% at 10 fps, 86% with 30 fps, junk 26% -> 23%.
Mechanism verified firing, not silently degrading: descriptors return
18/18 valid dims on real far-court bodies, 1% of comparisons unusable,
similarity median 0.18 across mixed pairs vs p90 0.98 same-body. Keep
it; it is nearly free and reduces the repair burden. It is NOT a yield
fix — on 16 rallies that failed in the baseline, A+B rescued 1
(extrapolates to 63 -> ~66 of 141).

**The real constraint.** Rallies that fail are the rallies the broadcast
was not showing. Main-camera fraction over the first 4 s (the serve, and
so the identity anchor):

    FAILED  rallies (n=50)   mean 0.48   median 0.64   48% below half
    COVERED rallies (n=91)   mean 0.67   median 0.74   22% below half

Whole-VOD main-camera fraction is 0.49 — half of a condensed broadcast
is replays, close-ups, crowd and graphics. A serve shown as a close-up
has no anchor frame to find, and no pose model, frame rate, tracker or
training set can recover a court that was never on screen. ~45% rally
yield is close to the STRUCTURAL ceiling for condensed VODs.

Consequences for anyone planning work here:
  * yield scales with FOOTAGE, not compute — an uncondensed broadcast or
    a fixed court camera is the only large lever;
  * GPU/stronger backend remains worth testing for track QUALITY (that
    is `coverage_ab.py`'s question) but must not be sold as a yield fix;
  * hand labels buy nothing on this axis: identity naming is already
    self-labelled from the serve anchors;
  * autonomy = an auto-accept gate (appearance-vs-geometry agreement,
    the LORO >= 0.85 bar) routing only below-bar matches to a human, NOT
    a higher yield per VOD.

## Anchor-free identity — BUILT 2026-08-19 (user's idea), 63 -> 90 rallies

If the broadcast never showed the serve there is no anchor, and the
geometry chain has nothing to name the players with.  The user's
proposal: learn each player's appearance FROM THEIR SERVICE POINTS —
the rallies that DO have anchors, where geometry names them for free —
then carry that model into the anchor-less ones.  No hand labelling:
the training labels are the geometry chain's own output.
`vision/coverage_anchorfree.py`, consumed by `coverage.py --anchor-free`,
`coverage_dominance.py`, and `coverage_overlay.py`.

What makes it tractable is a structural constraint, not a better model:
partners stand on the SAME side, so the near/far split already in the
pose npz partitions tracks into TEAMS, and the per-game team->end map
turns a 4-way identification into two 2-way ones.

GATE A (pre-registered at 0.90 before the first number): hold out a
resolved rally, refit on that game's others, name the held-out rally
from MID-RALLY appearance alone under the side->team constraint, and
compare to geometry.  Measured: **game 1 96.1%** (5,577 dets),
**game 2 100.0%** (15,253), **game 3 46.7%** (4,935) — g3 fails exactly
as its kit change predicts and contributes NOTHING, by construction.
Result: 27 rallies admitted, coverage 63 -> **90 of 141**.

RESIDUAL ERROR MODE, and it is not caught by the margin gate: two
game-1 rallies (r16, r39) come back ~50% right, i.e. one team correctly
named and the other swapped — and they carry LARGE margins (2.46, 2.58).
Both are rallies where the swap audit itself was a coin flip
(unanimity 0.50/0.60) on exactly that team.  Confidence margin does not
predict a whole-team swap; nothing internal can, since a coherent A<->B
swap is self-consistent.  This is the same non-identifiability the
project keeps meeting, and it is the price of these rallies.

WHAT THE BIGGER SAMPLE DID TO THE EARLIER CLAIMS (this is the point of
adding it — the extra data tested them):
  * width shares HELD.  Largest move 0.018 (Alshon g1 .567 -> .549,
    Black .433 -> .451, complements); games 2 and 3 moved <= 0.003.
    The 0.56/0.44 men-vs-women split is robust to a 43% larger sample.
  * deep-poach ordering FLIPPED and was therefore noise: Alshon 26.2%
    -> 23.3%, Patriquin 20.0% -> 24.7%.  Do NOT quote "Alshon poaches
    deeper than Patriquin"; peak share is what held (0.671 vs 0.662).
  * solo-vs-paired crossings weakened but survived at TEAM level:
    Alshon 16/0 -> 19/1 solo/paired, Patriquin 9/3 -> 17/4.  The
    defensible statement is the pair contrast — Bright/Patriquin
    switch together 8 times to Black/Alshon's 2 — not "all solo".

Serve-phase handling: the serve instant is unknown by construction, so
the frozen "first SERVE_PHASE_S after the serve" mask cannot be
evaluated.  Anchor-free rallies exclude the first SERVE_PHASE_S of
RETAINED frames instead — deliberately conservative (it discards good
mid-rally play rather than admit serve-stance frames), and no
serve-phase ellipse is claimed for them.

Every row carries `anchor_free_rallies` and `anchor_free_frac`, so any
number can be recomputed without them; publication must state which.
These rallies are additions to the SAMPLE and never evidence about the
chain — a rally named by appearance has no geometry to disagree with,
so it must never feed the appearance-vs-geometry swap audit.

## Where identity is actually hard (2026-08-19, reviewer questions)

Two reviewer points, both tested. One was right about the model, the
other found a forward-looking bug.

**"If we can tell Tyra from Bright we should be able to tell Alshon from
Patriquin."** Correct, and the model already does. Pairwise
leave-one-out separability on anchor crops:

    Alshon vs Patriquin  opponents   98.6% / 97.4% / 91.8%  (g1/g2/g3)
    Bright vs Black      opponents  100.0% / 97.1% / 89.9%
    Alshon vs Bright     opponents   98.5% / 98.4% / 86.8%
    Patriquin vs Black   opponents  100.0% / 97.7% / 89.0%
    Alshon vs Black      PARTNERS    82.0% / 76.0% / 88.2%
    Bright vs Patriquin  PARTNERS    64.7% / 73.6% / 58.7%

Every CROSS-TEAM pair is 97-100%; the whole problem is PARTNERS, and
Bright-vs-Patriquin is barely above chance. This is why the side->team
constraint bought only +1.1pp: it removes calls the model already got
right and leaves the ones it cannot.
CAVEAT on those cross-team numbers: near/far is itself 77% predictable
from the colour descriptor alone, so part of that separability is the
IMAGING difference (a near body is large and bright, a far one small and
dim), not the person. Partners are imaged identically — same side, same
instant, same distance — so their comparison is the honest one.

**HEIGHT DOES NOT RESCUE PARTNERS — do not retry it.** The obvious idea
(a man and a woman on the same team differ in stature) fails on measured
data. Depth-corrected h_ft medians: Black 5.34, Patriquin 5.23,
Bright 5.17, Alshon 5.09 ft — absolute values wrong for pro athletes,
and the WOMAN measures taller than her male partner in BOTH teams.
Per-rally, the taller-of-the-two matches the taller-overall 56% and 49%
of the time, i.e. chance. Bounding-box height measures POSTURE (players
are crouched and split-stepping), not stature. This is also why
coverage's mixed gender-height check reads 59/126.

**END SWITCHES ARE NOW FITTED, NOT ASSUMED** (the reviewer's second
point). Teams change ends mid-game under rules consistent WITHIN a
league and different BETWEEN them — MLP switches at 6 in every game,
PPA only in a decider. The team->end map was keyed per GAME, which
would be wrong for half of every MLP game once this runs on MLP
footage. `coverage.fit_end_segments` now splits a game into at most two
segments by fitting the changepoint from the data (side comes from box
heights, names from geometry, so neither owes anything to the appearance
model the map constrains), and both `coverage_appearance.stage2` and
`coverage_anchorfree` key on (game, segment).

On THIS match the fitter correctly finds NO mid-game switch, and Gate A
is unchanged (96.1% / 100.0% / 46.7%). What looked like a switch in
game 3 was rally 107 alone disagreeing with the eleven after it — a
single mis-identified rally, which is also the rally Gate A scores at
0%. That is why a segment must hold MIN_END_SEG = 3 rallies of evidence
before it is believed; the selftest plants exactly that outlier and
asserts it is not promoted to a switch.

### The appearance channel is a TEAM detector, not a player detector

Why partner separation is weak, measured rather than guessed (reviewer
question 2026-08-19: "colour of clothes seems to be the big difference
here?" — yes, and that is precisely the problem).

Mean 8-bit Lab of the torso region, games 1-2 (a = b = 128 is neutral):

    Alshon      L 91   a 165   b  77      <- saturated red
    Black       L 92   a 156   b  83      <- saturated red
    Bright      L 99   a 130   b 129      <- essentially NEUTRAL
    Patriquin   L 75   a 134   b 129      <- essentially NEUTRAL

Team A's red survives compression and distance (a ~160, far off
neutral).  Team B's green top and black top BOTH collapse to a ~132 /
b ~129 — colorimetrically almost the same garment — so only lightness
(99 vs 75) separates that pair at all.  Hence the pattern in the
separability table: the cross-team gap is enormous and the within-team
gap is tiny.  Doubles partners wear matching kit BY DESIGN, so the
strongest signal the feature carries is the one that cannot answer the
question the pipeline actually needs answered.

Consequences, all measured:
  * Fisher LDA does NOT beat nearest centroid on the partner call
    (-0.4% Alshon/Black, -1.4% Bright/Patriquin).  The signal is weak,
    not badly weighted — do not go looking for a better classifier.
  * What separates partners is INCIDENTAL, not team colour.  Best
    region for Alshon vs Black is LEGS (78.0%, vs torso 65.4%) — she
    wears a skirt, he wears shorts.  For Bright vs Patriquin it is
    torso LIGHTNESS (71.1%).  Both are match-specific accidents.
  * AGGREGATION is the working ingredient, and it needs INDEPENDENT
    looks.  Per-crop partner accuracy ~75%; averaged over a track of
    >=8 mid-rally frames it reaches the 96.1%/100.0% Gate A numbers;
    but averaging the 4-8 SAME-INSTANT anchor crops of one rally only
    reaches 80.9%/66.3%, because correlated samples do not average
    down.  Any future aggregation must pool across TIME, not within a
    frame.

SCALE-OUT WARNING: MLP teammates wear IDENTICAL numbered uniforms, so
the within-team colour gap there will be smaller than this match's, not
larger — while the cross-team gap this feature is good at is already
free from the geometry side constraint.  Expect partner separation by
colour to degrade on MLP.  The generalizable channel is JERSEY NUMBER
OCR (numbers survive matching kit, and the scorebug OCR machinery in
this repo is the obvious starting point) — specced here, deliberately
not built, because it is a build and not a tweak.

## RETRACTION: "deep poach" counts crossings, not poaches (2026-08-19)

The user watched the overlays and reported ZERO Tyra Black poaches. The
measure said she deep-crossed in 13-16% of rallies. The user is right
and the measure is wrong; recording why, because the geometry looks
convincing and a future session will otherwise rebuild it.

Measure 2/5's incursion test — both partners on one side of the
centreline, the nearer one is the "incursor" — cannot distinguish
    (a) I invaded your half and pushed you to the sideline, from
    (b) you went wide and I slid across to cover the middle.
They are the SAME GEOMETRY. Which one happened depends on who moved
because of the ball, and there is no ball in this stack.

Measured, at each player's deepest crossing, how far from the centreline
the PARTNER was standing:

    Black      8.9 ft   (86% of her crossings had him beyond 7 ft)
    Alshon     9.0 ft   (65%)
    Patriquin  7.9 ft   (62%)
    Bright     7.0 ft   (33%)

So most "deep poaches" happen while the partner is at the sideline —
case (b), the player merely held the middle. For Black it is 86%, which
is exactly why the eyeball saw none.

INITIATION TEST, pre-registered before computing and REPORTED AS
FAILING: over the 2.0 s ending at the peak, call the player who reaches
50% of their own lateral displacement first the initiator (a poacher
initiates; a court-holder reacts). It does not separate them —
Alshon 27% initiated, Black 33%, Patriquin 53%, Bright 40%, with 7/18
and 2/11 unclassified for too little movement. Do not resurrect this
without a genuinely different signal.

The user also identified a second false positive by eye: rally 19 at
1:05 of the anchor-free overlay is Alshon poaching and Tyra COVERING
BEHIND him, which scores as a 4.6 ft crossing for HER. Only 3 of her 14
crossings are of that shape (Alshon crossed first in all 3), so
cover-behind is a minority of the artifact; partner-went-wide is the
bulk.

WHAT SURVIVES — space, never intent:
  * width share (time-average) — the finding-11 observable, no intent
    claim, robust to the 63 -> 90 sample change;
  * off-court fraction — Black 6.8% of frames beyond a sideline vs
    Alshon 3.3%, a direct observation;
  * occupancy ellipse and Voronoi share — direct.
WHAT IS RETRACTED: "Alshon deep-poached in 26% of rallies", the
Alshon-vs-Patriquin poach ordering (already noise), and any framing of
incursion counts as poaching. Report them as CROSSINGS if at all.
Intent needs the ball; the ball is closed (see POSTMORTEM).
