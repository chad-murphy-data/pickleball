# Vision thread adjudication — the "90%+" vs "46%" accounts (2026-08-11)

*Litigating the two accounts on branch
`claude/pickleball-vision-match-analysis-dow5ds` (PR #52): the 08-10/early-08-11
"the instrument works, we get 90%+ of rallies" story vs the late-08-11
"46% is the best we measured, STOPPED" story. Verdict below; every number
re-checked against the branch's code and docs, none re-derived from video.*

## Verdict

**Amended same day, before merge:** the vote below covers the
BALL-TRACKING program, and it is unchanged. A channel neither account
ever tested — contacts from PLAYER SWING MOTIONS rather than the ball —
was raised by the user afterward and survives the same scrutiny; it gets
its own pre-registered gate in §Re-entry gates (Gate B). The distinction
matters: every measured wall in this file is a fact about a ~9 px ball,
not about the shot-level goal itself.

**The STOP is upheld. Do not pursue the shot-level vision program.** The
pessimistic account wins not because it is gloomier but because its
instruments can fail and the optimistic account's cannot: every
"validation" behind the 90%+ story is a population-level statistic that a
junk-dominated event stream passes, while the pessimistic account's three
tests (side-alternation checksum, displaced-null extrapolation probe,
hold-last-position control) directly measure the thing that matters and
came back unambiguous. Two of the pessimist's claims are amended below —
including "46% is the upper bound", which is NOT established — but neither
amendment flips the decision, because the decision was never really about
detection: even a solved detector buys a biased, incomplete shot archive
on a handful of condensed championship-court VODs, whose motivating
analysis (the Mid-Season freeze-out) is n = 4 matchups and stays a story,
not a finding, at any tracking quality.

## The two accounts, precisely

**Account A (optimist)** — `vision/README.md` verdict of 2026-08-10 +
`vision/interval_results.md` (morning 08-11): POC positive, full-VOD run
done, interval histogram shows a fast mode at 0.15–0.25 s "validated"
three ways without labels, recall "~50–65%", rally alignment 180/193
(93%), 0 empty rallies in 180, in-track recall 89.1%. This is where
"we can get 90%+ of rallies" lives, and at the layer it describes it is
TRUE (see reconciliation).

**Account B (pessimist)** — `vision/mvp_findings.md` + final HANDOFF
(evening 08-11): side-alternation 9% against a 50% chance floor (~100%
for real contacts), ~1.1 identified contacts/rally vs ~12 played,
confident-track coverage 21.1% of rally frames, ball absent from the
candidate set outside tracks (nearest candidate 239 px vs displaced null
243 px; hold-last control 238 px closes the ends-at-contact confound),
TrackNet CPU probe 46% of frames, sequence metrics need p ≈ 0.93,
auto-labels biased 42%-in-kitchen vs 14% base rate. STOPPED.

## Reconciliation: they measured different layers

The stack has an outer half (which rally is on screen when, who is
standing where) and an inner half (where is the ball, who hit it).

| layer | measured by | number | status |
|---|---|---|---|
| rally↔video alignment | A (cheer/scorebug DP) | 180/193 = **93%**, edit depth recovered blind (27.0 vs 26.9 min) | genuinely solved |
| court geometry | A/B agree | 0.06 ft median residual | genuinely solved |
| player identity | A/B agree | 99.25% over 45,689 rallies, log-only | genuinely solved |
| ball / contacts | **B only** (A had proxies) | ~10–21% usable, alternation 9% | **the wall** |

"90%+ of rallies" is a true statement about the OUTER layer. "46%" (really
~10–21%, see below) is about the INNER layer. Every payoff metric this
project wanted — touch share, speed-up roles, forced/unforced, the
freeze-out — lives on the inner layer, so the binding number is B's.
Where both accounts actually measured the same thing, they agree; the
disagreement exists only where A extrapolated and B measured.

## Cross-examination of Account A's three validations

All three pass without any real contacts in the stream:

1. **"0 empty rallies in 180" is guaranteed by construction.** The
   pipeline builds *one gated greedy chain per rally window* over a
   candidate stream averaging ~5/frame (691,884 over 80 min). A chain
   seeded inside every window yields events in every window by design.
   Likewise corr(contacts, duration) = +0.70 is what ANY roughly-constant
   event rate produces; and the median 12 contacts/rally count-match is a
   tuned quantity (the honest ledger itself records a 4× inflated config
   and a merge window that erased the fast mode).
2. **The fast mode sits exactly at the track-fragmentation period.**
   Median track length p50 = 14 frames = **0.233 s**; the histogram's fast
   peak is at **0.23 s**, and "chain reappearance starts" are counted as
   contacts — so drop-and-reacquire cycling alone produces a 0.2 s mode.
   Account B flags this in one line; it deserves headline status. The
   genuine ~200 ms physics prediction and the artifact coincide, which is
   what made the histogram feel confirmed.
3. **The blind gender split is confounded in the observed direction.**
   Men hit faster balls → more motion blur → tracks fragment more → more
   reappearance events at the ~0.23 s period → higher "fast share", with
   zero real interval measurement. Women dink more → detector holds the
   ball (42% of confident detections sit in the kitchen band vs 14% base)
   → fewer fragments → lower fast share. Observed: women 10.4% < mixed
   14.1/17.2 < men 16.9% — exactly the artifact's prediction, so the
   split cannot certify the signal. (The gate did reject two grossly
   artifactual configs — it has power against injection, not the power to
   validate a survivor.)

Meanwhile B's alternation checksum is close to airtight: every legal
return crosses the net, so consecutive real contacts alternate sides at
~100%; chance is 50%; the stream returns 9% (same-side oscillation), and
even the net-crossing-selected subset only reaches 33–37% — still below
chance, majority junk. The "50–65% recall" was physics-referenced, never
measured; the blind 10-rally human audit specced to measure it
(`vision/recall_audit.md`) was never filled in.

## Where Account B overstates (amendments to the record)

1. **"46% is the upper bound" is NOT established.** The probe
   (`vision/tracknet_probe.py`) ran tennis weights (yastrebksv/TrackNet),
   on CPU, at the net's fixed 640×360 input, on **48 mid-rally frames**,
   and 46% counts *any* above-threshold blob (2–200 px) with **no
   verification that the blob is the ball** — it is not a recall number
   and is not apples-to-apples with the colour detector's ball-verified
   21.1%. It is a smoke test whose only legitimate conclusion is
   "off-the-shelf cross-sport transfer is insufficient". TrackNet-family
   models fine-tuned in-domain publish high-80s to mid-90s F1 on
   broadcast tennis/badminton — i.e. the needed range — so the approach
   is gated on labels, not dead physics. B's own directional claims
   (fine-tuning on the 9,101 auto-labels would teach dinks and miss
   speed-ups; hand labels are the only clean path) remain correct.
2. **Chain corruption is mostly detectable, so the bar is softer than a
   hard 0.93.** A missed shot flips side-parity (catches all odd-count
   misses) and stretches the inter-contact interval (catches most
   even-count misses, ambiguously at dink→speed-up transitions). With
   certification, silent corruption becomes visible coverage loss; the
   real requirement is roughly p ≥ 0.85–0.9 *on fast shots specifically*,
   where current coverage is anti-correlated with speed. At p = 0.46 the
   distinction is academic: certified 3-chains ≈ 10% coverage, biased
   toward the slow rallies.

## Why "no" even though detection might be solvable

- **The motivating analysis cannot be rescued by tracking quality.** The
  freeze-out question is four matchups, one loss. A touch-share
  difference between the final and three controls is one team's one bad
  afternoon; this project's own standards (clutch thread, wind thread)
  would never publish it as more than an anecdote.
- **The capture ceiling is structural.** Broadcasts are condensed (80.3
  min of video for 107.2 of wall clock, cuts inside games), frame out
  servers/deep players, and exist mainly for championship courts — the
  sample bias is permanent, and it binds even a perfect detector.
- **Opportunity cost.** The referee-log warehouse (216k rallies in
  `pb_rally`) keeps producing publishable findings at zero marginal
  infrastructure; the vision program competes with that for the same
  hobby hours and loses on expected receipts per hour.

## Re-entry gates (two, both pre-registered; run 0. first in either)

Not "more tuning" — the branch is right that tuning is dead. Step 0 for
either gate: fill in the blank 10-rally blind human count
(`vision/recall_audit.md`, ~15 min) — the cheapest reality anchor, still
unmeasured.

### Gate A — the ball, via hand labels (as originally specced)

1. Hand-label ~300–500 frames stratified BY SHOT SPEED (oversample
   drives/speed-ups; label blur streaks at their centroid), ~2–3 h with
   any click tool.
2. Fine-tune (or at minimum run full-resolution, GPU, in-domain-threshold)
   TrackNet; a few dollars of compute.
3. Score with the yardsticks built before the detector
   (`vision/ball_recall.py`, the alternation checksum), reading recall
   **on the fast stratum**, not overall.

Decision rule: fast-shot recall < 0.8 → the wall is physical (720p
YouTube compression), dead at any label budget, said with a measurement;
≥ 0.9 → the sequence layer revives with the entire downstream stack
already built. In between → touch-share-grade output only.

### Gate B — the swing proxy (added 2026-08-11; never tested by either account)

Contacts from player swing events instead of ball reversals: pretrained
pose estimation per player (wrist-speed peak = swing), gated by the audio
pop train, attributed by `lineup.py`. Four asymmetries vs the ball, each
anchored to something already measured on the branch:

- **Person-scale, not ball-scale.** The wall was a ~9 px object among ~75
  distractor tracks/rally; a swing is a 40–80 px/frame wrist arc on the
  near side (~half that far side), and the branch measured that player
  detection succeeds *whenever players are in frame* — the shortfall was
  broadcast framing, not vision.
- **Off-the-shelf transfer runs the right direction.** Tennis TrackNet →
  pickleball ball failed (46% unverified); pretrained pose → athletes is
  in-domain (humans are the best-covered class in vision). The
  hand-label blocker was ball-specific.
- **Attribution is free.** The swing IS the attribution (which box swung);
  `lineup.py` names the box (99.25%). No net-crossing inference at all.
- **Audio revives, and the failure modes are complementary.** Pops
  survive the mix in fast exchanges (measured: broadband stripes at shot
  cadence) but died standalone on precision (applause). Swing∧pop
  coincidence (±150 ms) kills the applause channel. Vision-weak = small
  fast counters; audio-weak = soft dink pops — opposite strata, unlike
  the ball detector whose bias aligned AGAINST the measurand.

Known failure modes, ranked, all measurable in the probe: far-side
counter-blocks (small + fast + half-resolution + net occlusion — the
residual anti-correlation risk, hence the fast-stratum requirement);
fakes (false contact — the audio gate should kill them; a fake has no
pop); arm-pump while chasing lobs (require wrist-relative-to-torso
velocity, flag high-run-speed frames); both-partners-lunge (rare, and
adjudicable: one pop + next contact's team + log winner); serves/returns
cropped behind the baseline (identities are LOG-KNOWN — splice shots 1–2
from the log, only their exact timing is lost). Team-alternation
certification transfers intact: swing contacts carry team identity by
construction. For touch share, the mid-game end switch at 6 (all MLP
games) balances near/far recall asymmetry within every game.

Pre-registered thresholds (set before any data; amend only before
looking), scored on the Chicago VOD already downloaded, against the
committed windows/lineups:

- **G1**: team-alternation of audio-gated swing contacts ≥ 85%
  (chance 50%, truth ≈ 100%).
- **G2**: blind 10-rally audit — overall recall ≥ 75%, fast-exchange
  recall ≥ 60%, precision ≥ 90%.
- **G3**: contacts-vs-duration slope and serve/return identity checks as
  SANITY ONLY — the tuning-to-plausibility trap is documented above; do
  not tune to them.

Decision rule: pass G1+G2 → build the swing pipeline as vision MVP v2
(payoff: `hitter_player_uuid` — "the prize" field; touch share = the
DIRECT measurement of finding 1's coverage dial w; speed-up
initiator/finisher roles + the punished-selection ledger via audio
intervals + log outcomes; scales across every start-marked VOD with no
labels). Fast recall 40–60% with G1 passing → touch-share-only scope.
Alternation < 70% or overall recall < 60% → the second wall is measured
and the vision thread closes permanently, this time with both channels
dead by measurement. What the swing channel can NEVER recover: placement,
landing taxonomy, contact height — those need the ball, i.e. Gate A.

Cost: one overnight CPU run (or ~1 GPU-hour) of pretrained pose over
rally windows at 720p + one evening of scoring on the existing harness.
No labels, no fine-tune, no new evaluation code.

**The user's hand-coding offer (2026-08-11) and where it plugs in.** The
user offered to hand-code some points as drive/drop/dink/speed-up. That
is the most valuable label form for Gate B — NOT ball clicks: ~15–20
rallies on the Chicago VOD, each shot in order typed
serve/return/drive/drop/dink/speed-up/counter (~30–45 min of scrubbing;
extends the blank `vision/recall_audit.md` from a count to a typed
sequence). It converts G2 from "recall" into recall PER SHOT TYPE — the
fast-stratum kill threshold measured directly — and later seeds a
swing-kinematics → shot-type classifier. House discipline: these labels
live on the EVALUATION side of the wall only (the probe is scored
against them, never tuned on them; same contamination rule the POC
already wrote down). **The instrument is BUILT**:
`data/vision/shot_audit_chicago0725.html` (generator
`vision/make_shot_audit.py`) — open it next to the local VOD, tap
who-hit + shot-type per shot, download the CSV, commit as
`data/vision/shot_labels_chicago0725.csv`. Scrub times derive from the
committed cheer↔rally join and validate against all ten hand-scrubbed
recall_audit anchors at median |err| 1 s (worst +3 s); 191/193 rallies
timed, 20-rally core set = the original blind ten + the longest matched
rallies per game (five per game). No timestamps are asked of the human —
order suffices; the audio pop train carries timing and labels align by
sequence. The page embeds referee-log facts only (no tracker output), so
the blind rule holds by construction.

## What survives regardless (already flagged in the branch, confirmed)

- **The lineup state machine** (`vision/lineup.py`, 99.25%) — a fact
  about referee logs, not vision; usable for formation/stacking questions
  across the whole corpus with no camera.
- `vision/court.py` validated (a day saved if ever revisited), and the
  evaluation yardsticks (`ball_recall.py`, alternation test) — the gate
  above runs on them unchanged.
- The crowd null (`vision/crowd_leverage.md`): cheers track neither
  leverage nor allegiance — a finished, self-contained negative result.
