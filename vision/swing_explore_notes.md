# swing_explore — exploration-mode results (2026-08-16/17)

**Status: EXPLORATION, not a verdict.** This is the "take away the
hard-science rigor for a minute" thread the user opened after the Gate C
RTMPose diagnostic came back KILL (ceiling 45.1% @2x budget, ±0.30 s —
`data/vision/contact_ceiling_report_rtm.json`). Everything here is fit
and evaluated on the same 10 labeled rallies (162 contacts,
`data/vision/contact_labels_chicago0725.csv`) with leave-one-rally-out
cross-validation, RTMPose-balanced pose stream (`pose_rtm/` on the
user's Mac). Nothing in this file passes or fails contact_gate.md; any
result we come to believe graduates to a fresh pre-registration on
untouched rallies (11–16+).

## What the script does

`vision/swing_explore.py` (BUILD banner is versioned; run history below).
Three layers:

1. **Learned swing scorer** — logistic regression (numpy, no sklearn) on
   jitter-aware pose features per candidate peak: an *early* window
   [−0.75, −0.35] s (preparation — the user's "I can tell who's swinging
   next many frames before contact") and a *core* window [−0.35, +0.35] s,
   torso-relative arm-joint speeds, near/far aware. Negatives are
   guard-banded non-contact moments + matched other-side windows +
   locomotion-peak hard negatives; whiffs excluded from both classes.
   Metric: coverage@2× budget, ranked by learned score instead of raw
   peak height.
2. **Alternation-prior decoder** (the user's "assume alternating swings
   until the rally is over" idea) — serve-anchored side mapping, then a
   DP over candidates that enforces strict side alternation, cadence
   priors on inter-swing gaps, ghost placeholders (parity-keeping,
   max 2), and a span constraint forcing the path to reach the last
   confident candidate in the rally.
3. **Drift sweep** — labels slid ±0.8 s per rally to test whether weak
   rallies are misalignment artifacts.

## Run history (user's Mac, python3, pose_rtm stream)

| build | change | headline |
|---|---|---|
| v1 | first learned scorer | coverage@2× **54.9%** (vs 45.1% height-ranked) |
| v2 | numeric guards, serve-anchored mapping, prep/strike ablation | 46.9% (ablation leaked; optimizer hot) |
| v3/v3.1 | optimizer fixed; Accelerate BLAS warning noise suppressed | **56.2%**, stable; drift sweep FLAT (weak rallies r9/r10 not alignment artifacts) |
| v4 | alternation decoder | sequence precision 72.9% but quits early (r9: 1 of 29 decoded) |
| v5 | span constraint (alternate until the rally is over) | decoded coverage 43.2%, precision 51.1%, ghosts 24, **decoded count 161 vs 162 true** |

v5 per-rally count deltas: r1 −5, r2 +2, r3 +7, r4 +3, r5 −3, r6 −2,
r7 +7, r8 +6, r9 −10, r10 −6.

## The stable picture (held across every configuration)

- **Attacks are learnable and near-ceiling**: speed-up 9/9, smash 9/10
  in the raw ceiling; the learned scorer keeps them ~80–100%.
- **The soft kitchen game is the wall**: dinks ~41%, counters ~32% raw;
  ~45–50% under every scorer variant. This is genuinely stream-side at
  RTMPose/720p — orientation oracle ≈ headline (killed that hypothesis),
  drift sweep flat (killed misalignment), candidate-any 71% vs @2× 45%
  says the *selector* binds at budget but the missing soft contacts are
  weak or absent in the arm-speed stream itself.
- **Count vs placement split (v5's lesson)**: the alternation decoder
  with a span constraint gets rally *shot counts* nearly exact
  (161 vs 162 total) while only ~half the decoded events land on a true
  contact within tolerance. Turn-order/shot-count analytics are much
  closer than exact-timestamp contact attribution.
- Prediction ledger: I predicted v5 decoded coverage would beat 56.2%
  and precision would rise — **both wrong**. The span constraint trades
  precision for completeness; it commits through stretches where the
  stream has nothing.

## ViTPose A/B result (2026-08-17, Colab T4 — lever 1 RESOLVED: backbone ruled out)

Same script, same labels, `--pose-dir pose` (ViTPose-plus-huge stream):

| metric | RTMPose | ViTPose-huge |
|---|---|---|
| raw ceiling-side @2× | 45.1% | 40.7% |
| learned coverage @2× | 56.2% | 52.5% |
| learned lift | +11.1 | +11.8 |
| fast stratum (raw) | 61.0% | 53.7% |
| ceiling-any @2× | 71.0% | 66.0% |
| decoded count | 161 / 162 | 139 / 162 |
| smash (raw) | 9/10 | 5/10 |
| speed-up (raw) | 9/9 | 6/9 |
| counter (raw) | 7/22 | 11/22 |

The stronger backbone does NOT put the soft contacts into the stream —
it's slightly worse overall and clearly worse on attacks (CIs overlap;
licensed claim = "backbone not the binding constraint", not "RTM
better"). The learned lift is the same size on both streams: learning
improves the selector, the stream sets the level, both plateau at the
same soft-kitchen wall. Oracle-orientation = headline on both; prep
ablation ≈ 0 on ViT (−0.6). Gate C's ViT verdict run printed KILL —
gate formally CLOSED, final for this footage (contact_gate.md).
Production spine for anything downstream: RTMPose (faster AND ≥).

## Remaining levers (post-A/B)

1. **Labels at scale** (`labeling_protocol.md` — split frozen, chained
   seek shipped) feeding a **temporal model class** (sequence over
   frames, not per-peak logistic). Not gated by the Gate C KILL — a
   different instrument — but requires fresh pre-registration on
   untouched holdout. The prep-ablation ≈ 0 result is mild evidence the
   easy temporal win isn't free; the counter-evidence is that per-peak
   features structurally can't use context a sequence model can.
2. **New footage** (higher-res / uncondensed) — the postmortem's
   standing door, now for the swing ceiling too.

Also: r9/r10 label spans exceed their log durations by ~24 s (others
±4/−13) — recorded in contact_gate.md's anomaly addendum; asterisk on
r9/r10 for TRAINING use until the debug-frame scorebug check clears it.

Constant-tuning further on these 10 rallies is over — the numbers above
are the plateau.

## Candidate-feature checks: shoulders, feet, gaze (2026-08-18)

Separate from the two levers above — a feature-value question, not a
model-tuning one. Earlier ranking called shoulder rotation the weakest
of three candidate signals (behind footwork, poaching) on the reasoning
"dinks are compact, low-rotation shots." User pushback: "dink" isn't
one mechanic — down-the-line dinks stay square, but cross-court ones
(especially backhand rolls, prominent in the pro game right now)
should show real rotation from body mechanics (backhand contact sits
across the midline) plus the active pronation a roll needs for
topspin; committed shots (drives/smashes) may be MORE square than
assumed given pickleball's short paddle lever. If true, rotation isn't
a weak uniform signal, it's a high-variance/bimodal one — usually more
useful to a classifier than a small effect present everywhere.

No forehand/backhand or cross-court/down-the-line label exists
(checked `make_shot_audit.py`'s `SHOT_TYPES` — not there), so this
can't attribute WHICH dinks rotate, only whether the distribution's
SHAPE is consistent with a mixed population. `dsho` (shoulder-line
angular velocity) already exists in `track_series` as one of the six
learned-scorer channels — nothing new extracted, just looked at.

`vision/shoulder_check.py` (original): pulled `dsho` core/early-window
peaks for already-labeled dink+counter contacts vs drive/speed-up/smash,
same hitter-track and window conventions as `rally_instances`/
`window_feats` (reused, not reimplemented). Prints per-type
distributions, a text histogram, and Sarle's bimodality coefficient
(BC = (skew²+1)/kurtosis, >0.555 rule-of-thumb bimodal) — a rough
heuristic, read next to the histogram, not instead of it. Selftest
(`--selftest`, no files needed) caught a real bug before this ran on
anything: a hand-rolled 1D-2-means gap/spread statistic was the first
design and is mathematically unable to discriminate (pooled spread
already contains the between-cluster gap, so the ratio caps near 2.0
even for infinitely-separated clusters, and a plain unimodal Gaussian
split down the middle already scores ~1.6) — replaced before ever
pointing it at real labels.

Not run against real pose data yet (needs the user's Mac / pose_rtm).
If the soft-shot group reads bimodal, next step is a schema addition
(optional forehand/backhand or direction tag) to test the attribution
directly — a discussion with the user, not a unilateral change to the
frozen labeling tool. If it's high-but-unimodal, or matches the
committed-shot group, rotation stays out. Same rule as everything else
here: a result worth believing graduates to fresh pre-registration on
untouched holdout, never folded back into Gate C.

**First real run, pre-fix (2026-08-18, RTMPose, the user's Mac)**:
n=43 soft / 23 committed. Medians close (0.131 vs 0.146) but soft's p75
reached higher (0.377 vs 0.227), and BC read 0.933 for soft — looked
strongly bimodal at a glance. The histogram is why you don't stop at
the summary number: both groups had a handful of values sitting right
at ~3.13, and `dsho` is a wrapped angle difference — pi (≈3.14159) is
its hard mathematical ceiling, not a real "fast rotation" reading.

Traced the cause: unlike every other channel in `track_series`, `dsho`
had no confidence or frame-gap gate. A low-confidence/occluded shoulder
keypoint (an L/R swap under noise) reads as a near-180° flip between
two consecutive frames — impossible physically at native fps, but
exactly what pins the ceiling mathematically. Fixed: `dsho` now gates
on >=0.2 confidence on both shoulder keypoints at both endpoints, plus
the same `ok_dt` gap check every other channel already used. Selftest
added directly to `track_series` (not just shoulder_check.py) proving
a glitch frame zeroes out while real gradual rotation still registers.

Net effect on the read: BC=0.933 shouldn't be trusted as-is — it's a
kurtosis-based statistic and a few pi-pinned outliers can inflate it
on their own. Excluding that top bin by eye, there's a smaller but
real-looking asymmetry: soft had a few contacts (~4 of 43) in the
1.2–2.7 range that committed (0 of 23) had none of — directionally
consistent with "only some dinks/counters carry real rotation," but
nowhere near enough events to lean on. Re-run against the fixed code
before drawing any conclusion; if the intermediate band survives and
BC still clears 0.555 with the ceiling artifact gone, that's a real
result worth taking to the schema-addition conversation.

**Extended to feet and gaze, renamed `vision/feature_check.py`
(2026-08-18)**: the user asked directly for footwork and eyes alongside
shoulders — the other two candidates from the "what else could a
swing-detector look at" discussion (footwork was independently agreed
as the strongest candidate; poaching/court-crossing was reframed earlier
as a decoder-identity aid, not a detection feature, and stays unbuilt).
Two new `track_series` channels, both riding on keypoints
`pose_extract.py` already saves and neither wired into the trained
scorer (exploration only, no re-tune):

- `leg`: hip-relative ankle+knee speed, identical math to the existing
  `arm` channel (same confidence/gap gate from day one — no glitch class
  to fix here, since it copies an already-correct pattern rather than
  inventing new angle math). Headlined on the EARLY window, not core —
  the hypothesis ("feet moving = getting ready to hit") is about the
  PREP arc, unlike shoulder rotation which is about the strike itself.
- `dgaze`: ear-line angular velocity, identical math to the
  now-fixed `dsho` (same >=0.2-confidence, both-endpoints, both-keypoint
  gate from the start). Ears, not eyes — same head-yaw geometry, but a
  pose model localizes an ear from head shape/context and doesn't need
  to resolve anything as fine as an eye at broadcast distance.

`feature_check.py` reports raw nose/eye/ear keypoint CONFIDENCE/coverage
first, before any behavioral number — a cheap, decisive answer to the
user's stated skepticism about eyes specifically (grounded prior:
`ball_visibility.py` found the ball findable in only 64% of in-play
frames on this same condensed-VOD footage; faces are smaller). If eye/
ear coverage reads low, `dgaze` should be treated as noisy-to-unusable
regardless of what its distribution looks like — the report says so
inline. Both new channels got the same selftest rigor as the dsho fix:
a synthetic 6-frame rig with INDEPENDENT glitch frames for shoulders
(frame 3) and ears (frame 4) proving the two gates don't leak into each
other, plus a real-motion-preserved check for each.

**First real run, post confidence-gate fix (2026-08-18)**: identical
output to the pre-fix run — same max (3.129/3.126), same histogram, same
BC (0.933), to the last digit. The confidence gate did LITERALLY
NOTHING. Face-keypoint coverage came back 100.0% mean/median on every
one of nose/L eye/R eye/L ear/R ear, zero exceptions across 66 contacts
— also the tell, not a clean result: real tracking on this footage
doesn't read that saturated (`ball_visibility.py` found the ball itself
findable in only 64% of in-play frames).

Root cause, unifying both symptoms: keypoint confidence measures
EXISTENCE ("a keypoint is probably here"), not PRECISION ("this pixel
is right"). A pose model can be fully confident it found two
shoulder-shaped blobs while being wrong about which one is left —
exactly the L/R-swap failure the gate was built for, and exactly the
kind of error confidence doesn't flag. Fix: `MAX_ROT_RAD = 1.2` (~69°),
a hard physical-plausibility cap on dsho/dgaze's per-step reading,
independent of and in addition to the confidence gate — no human
shoulder or head turns further than that between two valid consecutive
frames (`ok_dt` already bounds the gap at <=2.5 frames), but an L/R
swap always lands near pi regardless of reported confidence. New
selftest specifically constructs a CONFIDENTLY-reported swap (0.9, not
low) and proves the cap catches it where the confidence gate alone
passed it — this is the exact case that shipped and reached the user
before it was caught. `feature_check.py`'s face-coverage section
rewritten to say what it actually measures and to print an explicit
saturation warning when every keypoint reads >97% with no spread.

**Second real run, with the cap (2026-08-18)**: cap clearly active this
time — dsho max dropped from 3.129 to 1.112 (comfortably under 1.2, no
more pi-pinned values), BC 0.849. `leg` unchanged from the pre-cap run
(expected — it was never touched by this fix, same already-correct
math as `arm`).

But `dgaze` max came back 1.194 — 99.5% of the 1.2 cap — with 7/43 soft
contacts in the top histogram bin, right up against the ceiling. Genuine
ambiguity, not obviously artifact this time: a head is lighter than a
shoulder girdle and can plausibly rotate faster, so MAX_ROT_RAD=1.2
(chosen from shoulder-turn physics) is NOT independently justified for
ears — it could be truncating a real fast head-turn as easily as it
could be softening an L/R ear swap short of pi. Added `dsho_nocap`/
`dgaze_nocap` (confidence/gap gated, NOT plausibility-capped) so this
is checkable without re-running pose extraction at a different
threshold: `feature_check.py` now reports, per channel, how many
suppressed soft-group events sit in the ambiguous 1.2-2.0 band (could
be real) vs unambiguously past 2.0 (nothing legitimate reads that close
to pi), plus how many KEPT values sit within 10% of the cap itself.
Selftest extended: `dsho_nocap` must preserve the raw reading a capped
`dsho` suppresses, and must equal the capped value when nothing was
suppressed. Not yet re-run — this is the next thing to look at before
reading dgaze's soft-vs-committed gap as real.

## Does leg/dgaze actually improve swing DETECTION? (2026-08-18)

Different question from everything above. Every `feature_check.py`
result answers "among KNOWN shots, does movement look different by shot
type" — useful for deciding a feature is worth trying, but silent on
whether it helps the thing this file's trained scorer is actually FOR:
telling a real swing apart from a non-swing. `vision/channel_ablation.py`
answers that directly, same LORO/coverage@2x/TOL_S=0.30s methodology
already validated in this file, run twice and compared:

    BASE = arm/lw/rw/le/re/dsho   (the v1-v5 pipeline's channels)
    EXT  = BASE + leg + dgaze

`window_feats`/`rally_instances`/`score_rally` in `swing_explore.py`
now take an optional `channels=` param (default `CHANNELS_BASE`,
preserving the exact v1-v5 vector shape/behavior for every existing
caller). `strike_only` guarded with an explicit shape assertion rather
than silently mis-ablating if ever pointed at EXT-shaped vectors (it
isn't, currently — this comparison doesn't touch the prep/strike
ablation question).

IMPORTANT: BASE here is NOT the historical 56.2%. That number predates
the dsho confidence-vs-precision fix — trained on values that don't
exist anymore. channel_ablation.py re-measures BASE fresh on the same
rallies so the comparison is apples to apples regardless of the old
number.

Selftest has real teeth (contact_ceiling.py's own standard: a control
that would fail if the mechanism didn't work, not just "doesn't
crash"): zeroes arm/lw/rw/le/re — everything BASE has except an already-
uninformative-by-construction dsho — on 2 of 4 synthetic rallies while
keeping a planted dgaze signal intact throughout. BASE genuinely
struggles (75% vs its 100% ceiling unhandicapped); EXT recovers fully
(100%) via dgaze alone. First version of this test only asserted
"EXT >= BASE" and both tied at 100% — true but proved nothing about
whether the mechanism does any work; caught before shipping.

Not yet run against real data.
