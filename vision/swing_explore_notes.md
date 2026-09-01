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

**First real run (2026-08-18)**: BASE 57.4% overall (162 contacts) —
the honest re-baseline, a small move from the stale 56.2% and NOT
comparable to it directly, consistent with the dsho fix touching a
modest slice of contacts rather than the whole model. EXT 54.9% — a
small net NEGATIVE, not an improvement. Per-type deltas mostly single-
digit and noise-range on the well-powered rows (other n=58, dink n=32,
counter n=22); every large-looking delta (drop +33%, speed-up −11%,
return −10%, serve +10%) turned out to be exactly ONE contact flipping
on a sample of 3-10 — textbook small-n noise, not signal, exactly what
the printed footer warns about. Plainest read: adding 14 columns
(50->64) to a logistic model trained on ~150 examples per LORO fold
costs more in estimation variance than these two channels are currently
paying back. Doesn't kill leg/dgaze as features — it says the channels
aren't strong enough YET to earn their keep at this label count, which
points back to labels-at-scale (`labeling_protocol.md`) as the more
load-bearing lever than more feature engineering right now.

## Footwork window sweep (2026-08-18)

User question: is the footwork ("leg") channel actually looking at the
right window before contact, or did it just inherit PRE_S=0.75s from
an unrelated earlier decision (the arm/prep-arc timing ablation, which
picked 0.75s to sit safely beyond tap-jitter contamination on a totally
different channel)? Answer: yes, inherited, never re-derived for
footwork specifically. `feature_check.py --sweep-leg` adds a sliding
0.40s window swept from 1.55s down to 0.75s before contact (the
closest point exactly reproduces EARLY as a checkpoint — same window,
computed two ways, asserted equal in the selftest).

Also reports CONTAMINATION per window: whether it reaches back past the
PREVIOUS contact (any player) via `sweep_contaminated()` — past that
point a "prep" window is measuring recovery-from-the-last-shot as much
as getting-ready-for-this-one, and fast hands exchanges make that a
real risk the further back you look. Reported, not silently corrected
— read the far-back rows against their contamination rate, don't just
trust the number.

Not yet run against real data.

## feature_check.py repurposed: shot vs NON-shot (2026-08-18)

User call, arriving in two beats: "change this to just classify
whether something is a shot or not a shot, rather than trying to
decide whether it's shot type A or B" — then, sharpening it, "maybe
it's a two-step process: (1) swing/no-swing, (2) shot type?" That
two-step framing is exactly how the pipeline is already built on the
detection side (the trained scorer IS binary swing/no-swing; per-type
numbers were only ever recall slices), but feature_check.py was the
one tool still framing its whole analysis as type A vs type B
(soft dink/counter vs committed drive/speed-up/smash medians, plus
the dink-bimodality question). Rewritten:

- SHOT rows = every labeled contact, all types pooled (type survives
  as a passenger column in the CSV for eventual step-2 work).
- NON-SHOT rows = a deterministic 0.5s grid of in-play anchors
  between each rally's first and last contact, guarded by the SAME
  constants the real detector's negatives train with (GUARD_S=0.5
  from own-side contacts, WHIFF_GUARD_S=0.6 from whiffs — imported
  from swing_explore, not re-invented). Opponent-contact instants
  pass on purpose: the other side hitting IS a non-shot moment for
  this side, and rally_instances builds its matched negatives at
  exactly those instants.
- The hitter-proxy selection (max prep-arc arm energy) is applied to
  BOTH classes, so a channel merely correlated with the selection
  rule can't fake a gap.
- Separation metric = AUC with MIDRANK tie handling — gated channels
  emit exact zeros, and a naive rank AUC (argsort order) would break
  zero-ties by concatenation order and shade the number. Selftest
  pins exact values incl. the tie case (auc([0,0,1],[0,0,0]) = 2/3).
- --sweep-leg reframed the same way: per window, shot vs non-shot
  medians + AUC + contamination % for each class ("did this window
  reach past the most recent prior contact" — for shots that's
  recovery polluting prep; for non-shots it's real shot movement
  leaking into the quiet reading, pushing AUC toward 0.5).
- Removed: SOFT/COMMITTED constants and the bimodality coefficient.
  The dink-mechanics question's results-so-far stay recorded in the
  earlier sections above; if it ever reopens, it reopens as step-2
  work after step 1 is usable.
- channel_ablation.py default output likewise reduced to pure
  swing/no-swing (overall coverage + delta); --per-type restores the
  sliced tables for debugging.

Step 2 (shot type) is deliberately NOT built: at 45-57% step-1
coverage, classifying the type of swings we mostly can't find is
premature. The typed contact labels being produced under
labeling_protocol.md are exactly the training data step 2 will need,
so nothing about the labeling habit changes.

Not yet run against real data in the new form.

**First real run of the shot-vs-non-shot form (2026-08-18, user's Mac,
123 shot rows / 193 non-shot rows; 123 of 162 contacts scored, rest
lost to track/window coverage)**:

- **dsho AUC 0.745** — the one genuinely shot-correlated channel of
  the three, and it's the one ALREADY in the trained detector. Shot
  median 0.118 vs non-shot 0.063.
- **leg AUC 0.464** — nothing, and directionally NEGATIVE: medians
  identical (0.108/0.109) with the non-shot p75 HIGHER (0.228 vs
  0.165). Feet move slightly MORE when no shot is happening —
  repositioning/locomotion, i.e. exactly the "motion peaks away from
  contacts" hard-negative class rally_instances already trains
  against. Footwork is not a tell; it's mild anti-signal.
- **Sweep: flat at every lead time** (AUC 0.470-0.530 across all five
  windows) — answers the user's original window question decisively:
  there is no window where footwork separates shot from non-shot, so
  the inherited PRE_S=0.75 wasn't hiding a better one. Structural
  bonus finding: contamination hits 90%/93% at the [-1.55,-1.15]
  window — i.e. ~90% of anchors have the PREVIOUS contact within
  1.55s. **A quiet "one second before the shot" mostly does not exist
  in this sport's rally cadence**; prep windows further back than
  ~0.75s are measuring the previous exchange, full stop.
- **dgaze AUC 0.680** — moderate separation but mechanism suspect:
  30/123 shot rows (24%) had unambiguous >2.0-rad L/R-ear-swap
  artifacts suppressed, vs 15/123 (12%) for dsho — the artifact RATE
  itself is plausibly shot-correlated (motion blur peaks at strikes),
  so part of the capped signal may still be swap-adjacent garbage
  rather than gaze behavior. Also plausibly redundant with dsho (head
  turns with the shoulders). Both consistent with channel_ablation's
  zero incremental value.
- Face existence confidence saturated ~100% again (the printed
  saturation NOTE fired as designed) — uninformative, as expected.

Closes the loop with the channel_ablation run: EXT failed to help
because the two added channels are a nothing (leg) and a weaker,
artifact-suspect cousin of a channel already in the model (dgaze).
Three candidate channels tested end to end; feature engineering on
this pose stream looks tapped out. Labels at scale remains the
load-bearing lever.

**Follow-up question (user, 2026-08-18): "so shoulders added something
to arms/wrists?"** — flagged as not yet actually known. The AUCs above
are MARGINAL (each channel alone vs nothing), not INCREMENTAL (on top
of the channels already in the model); dsho's detector slot has never
been re-measured since the MAX_ROT_RAD bug fix, so the v3-era lift was
earned partly on artifact values. Two additions to answer it with
data: (1) feature_check.py now prints an "arm" REFERENCE row (the
detector's primary channel) so candidate AUCs read against the
yardstick they'd have to add to — with the caveat printed that track
selection keys on arm prep energy for both classes, so arm's own AUC
has a selection tailwind; (2) channel_ablation.py --drop CH compares
BASE-minus-CH vs full BASE under the same LORO (e.g. --drop dsho).
Selftest teeth: a dsho-only channel set on the synth (whose signal
lives in the arm family) must do clearly worse than full BASE, proving
loro_eval honors the reduced set (measured 66.7% vs 75.0%,
deterministic). Neither run against real data yet.

**--drop dsho real run + a reproducibility catch (2026-08-18)**: the
user's run printed BASE-minus-dsho 54.9% vs BASE(full) 54.3% — dsho
currently adds −0.6 points, i.e. nothing (noise around zero; its 0.745
marginal AUC is evidently redundant with the arm family). BUT
BASE(full) had printed 57.4% earlier the same day, and that discrepancy
had to be run down before believing anything. Code CLEARED: git diff
between the two runs' channel_ablation versions touches only
print/CLI/selftest (loro_eval/assemble_rallies identical;
swing_explore/contact_ceiling untouched since 67449e0), and
deterministic probes on the handicapped synth show BASE repeat SAME
in-process, BASE-after-reduced SAME as fresh, across two
PYTHONHASHSEEDs — no order dependence, no state leak, no hash
sensitivity. Remaining explanation: the LABEL ROWS changed between the
runs (count constant at 162/10 pose-covered rallies, so retimed
existing rows — consistent with ongoing labeling edits/re-exports;
candidate suspect: an r9/r10 anomaly fix; unconfirmed, asked the
user). Consequences: (1) WITHIN-run comparisons stay valid — run 2's
−0.6 is apples-to-apples; (2) every cross-run delta involving the
older 57.4/54.9 numbers is void; (3) channel_ablation now prints a
LABELS FINGERPRINT (md5 of the evaluated contact/whiff rows, selftest
covers determinism + 10ms-edit sensitivity) — only same-fingerprint
runs compare. Echo of the measurement-frame lesson: validate the
inputs being compared, not just the code.

Provisional channel picture on current labels, three instruments
agreeing: the detector is carried by the arm/wrist family + tail
features; dsho's slot adds ~0; leg/dgaze add ~0 or slightly negative.

**Reproducibility hunt continued (2026-08-18)**: user reports labels
NOT touched — which un-clears everything and forced the rigorous pass.
Full unfiltered diff 67449e0..754bdac: genuinely CLI/print/selftest
only. Nondeterminism grep over swing_explore + contact_ceiling: every
default_rng is seeded, no glob/listdir (load_rally is exact-filename
r{cum:04d}.npz), frozensets are membership-only. So no identified
mechanism for 57.4% -> 54.3% on identical inputs. Added
pose_fingerprint (frame counts + coordinate sums per rally, from
arrays already in memory) next to labels_fingerprint so BOTH inputs
are pinned from here on. Decisive battery handed to the user: (a)
paste run 1's BASE block from terminal scrollback (verifies 57.4 was
even real — cheapest check of all), (b) same --drop command twice
back-to-back (flap = cross-process nondeterminism, e.g. BLAS; stable =
one-time input/state change), (c) default mode once (does BASE
first-position reproduce 54.3 or return to 57.4 — order sanity on real
data; synth probes say order is innocent). Numbers involved: run 1
BASE 93/162, EXT 89/162; run 2 minus-dsho 89/162, BASE 88/162 — a
±5-contact band, so whatever this is, it moves borderline contacts,
not the structure. The dsho conclusion (adds ~nothing) is robust to
the whole band; the open question is instrument hygiene, not the
finding.

**POSITION BUG FOUND AND FIXED (2026-08-18): rally_instances' training
set depended on evaluation order.** The user's decisive battery: BASE
prints 57.4% whenever evaluated FIRST in a process (default mode, 2/2
runs) and 54.3% whenever evaluated SECOND (--drop mode, 3/3 runs,
including once on the pre-fingerprint file) — deterministic per
position, labels fingerprint identical throughout (efb79d5003).
Mechanism, confirmed by direct probe: window_feats computed the
_peaks cache (hard-negative motion peaks) BEHIND its window-coverage
guard, and rally_instances' "ensure cached" priming call probed the
rally midpoint. FRAGMENT TRACKS — too short/misaligned for ≥11 frames
in any contact/probe window — therefore contributed no hard negatives
cold; but score_rally's dense pass probes each track at its OWN
timestamps, where a 15-frame fragment passes, filling the cache. So
fold k+1 trained on rallies scored in folds ≤k with MORE negatives
than the same rallies contributed before scoring (intra-eval
gradient), and a second loro_eval in the same process saw everything
warm (inter-eval jump). Probe: a 15-frame fragment parked in a window
gap yields 13 negatives cold / 14 warm from IDENTICAL calls — and the
subtler intra-call variant appeared too (a random-negative draw
landing on the fragment cached _peaks after its own hard-negative
loop had already run empty). Synthetic probes missed it twice because
synth tracks span the whole rally and pass the guard everywhere —
same lesson as the serve-pin affair: the null your rig can't
represent is the bug you can't see.
FIX: track_peaks(ser) — unconditional, no coverage guard — replaces
both the priming hack and window_feats' inline computation; regression
test in swing_explore --selftest (fragment rig, instances must be
identical before/after a scoring pass). Consequences: (1) the
54.3-style warm regime is closer to the INTENDED estimator (every
track's peaks as hard negatives) — the historical 56.2/57.4-era
numbers were the accidental cold mixture and post-fix fresh runs will
not reproduce them exactly; (2) BOTH printed ablation deltas (EXT
−2.5, dsho −0.6) were cross-regime comparisons and are VOID — re-run
both with fixed swing_explore.py; (3) registered prediction for the
re-run: BASE prints the SAME number in default and --drop mode.

**Post-fix re-run (2026-08-18, user's Mac, fingerprints efb79d5003 /
bce74d0f26 both runs): prediction CONFIRMED, ablations final at this
label count.** BASE prints 54.3% (88/162) in BOTH default and --drop
mode — first and second position — so order independence holds on the
real data that exposed the bug. The apples-to-apples channel answers:
dsho −0.6 (removing it gains exactly one contact: 88 -> 89), leg+dgaze
+0.6 (adding them gains exactly one contact: 88 -> 89). Every delta is
±1 contact of 162 — noise by the file's own footer. Note the flatness:
5 channels (minus-dsho) 54.9, 6 channels (BASE) 54.3, 8 channels (EXT)
54.9. Channel picture, now on a clean instrument: the detector is the
arm/wrist family; dsho, leg, and dgaze are each within one contact of
irrelevant at 162 labels. dsho STAYS in BASE — a 1-contact "gain" from
dropping it licenses nothing under the house rule (changes graduate
via fresh pre-registration on untouched holdout), and the channel set
gets re-chosen from scratch when the temporal model class is
pre-registered anyway. Post-fix BASE 54.3 vs the void first-position
57.4: the lower number is the INTENDED estimator (all tracks
contribute hard negatives); the higher one was the accidental cold
mixture. Feature-engineering round closed for real this time: three
instruments (marginal AUCs, clean incremental ablation, shot-vs-
non-shot descriptives) agree — the lever is labels at scale, not more
channels.

## 2026-08-18 — fastslow_check.py: the step-2 coarse-split instrument (built + selftested; awaiting first real run)

User proposal same day: step 2 as "if not fast, then slow." The idea
is stronger than a default class: detection misses concentrate in
soft shots while decoded shot COUNTS are near-exact (161/162), so
"slow = total − fast" makes fast the only class needing direct
recognition, and the binary is the smallest taxonomy that supports
the firefight/speed-up analytics. New instrument `fastslow_check.py`
measures the coarse split on TRUE contact windows (labeled times,
picked hitter track) — the step-2 analogue of contact_ceiling:
placement error removed, does the type signal exist at all.

TYPE MAPPING FROZEN before any real-data result: fast = smash /
speed-up / drive / counter (+ literal "fast"); slow = dink / drop /
lob / reset (+ literal "slow"); serve/return excluded (rally position
identifies them downstream); ""/"other" excluded with a counted
nudge — tagging the 58 "other" contacts just fast/slow (coarse tags
accepted) adds them without full typing. Unknown vocabulary prints
LOUDLY, never silently drops. Train split only; holdout untouched.

Three feature sets on the SAME instance set, because the answer
directs the build: CADENCE (gaps to prev/next labeled contact, capped
3 s — downstream this is decoder timestamps, no pose needed), POSE
(window_feats EXT at the true time; carries its own label-free
cadence proxy), POSE+CAD (deployable; gets the per-type table and
confusion). If CADENCE ≈ the rest, fast/slow ships on timing alone.

REGISTERED PREDICTIONS (before the first real run, n≈84: 46 fast /
38 slow at current labels): (1) POSE+CAD accuracy 85–95% (point ~90)
— the fast class rides the same arm-speed signature that already
finds attacks at 88–100% detection; (2) CADENCE alone materially
above the 55% majority baseline, point ~75% — counters live in tight
exchanges; (3) errors concentrate at the speed-up/counter ↔ dink
boundary (the transition shots), smash/drive near-perfect; (4) the
true-window number is a CEILING — deployed accuracy sits below it by
the placement-error tax.

RIG TRAP RECORDED (cost: one selftest round each): (a) capped edge
gaps — first/last contacts carry the 3.0 s cap, and a class cycle
that puts a fixed class at rally edges lets cadence "separate" a
separable-by-nothing arm (66.7%); bracket synth rallies with
serve/other like real rallies. (b) FRAME-GRID ALIASING × DETERMINISTIC
SLOT PATTERNS: with contacts at exact 2.2 s multiples on a 30 fps
grid, float rounding flips single frames in/out of window masks as a
deterministic function of slot index k; with class ALSO a function of
k repeated across rallies, LORO transfers the wobble — a
nothing-to-find null separated at 85% (arm_cmax d=1.09 on sd=0.012
features). Fix: shuffle class assignment per rally (seeded) AND
jitter times off-grid. General form: under LORO, any deterministic
feature-of-slot-index wobble becomes learnable whenever class is also
a function of slot index — synth nulls must randomize the LABEL
channel, not just equalize the signal channel. Post-fix null: 57%
acc / 0.56 AUC. Same family as the track_peaks incident: the rig
telling the truth only after its own blind spot got a test.

## 2026-08-18 — fastslow v1 real run: REGISTERED PREDICTIONS FALSIFIED; v2 diagnostics built

**The v1 run (user's Mac, fingerprints efb79d5003 / bce74d0f26,
n=72: 41 fast / 31 slow, majority 56.9%) falsified every registered
number: POSE+CAD 58.3% (registered 85–95, point ~90) — ONE contact
above majority; CADENCE 54.2% (registered ~75) — BELOW majority;
pose AUC 0.545, cadence AUC 0.600, combined 0.586.** Confusion:
model over-calls fast (45/72 predicted fast; true slow -> 17 fast /
14 slow; dink only 10/25 correct). The registered predictions
existed precisely so this moment is unambiguous: the "fast class
rides the arm signature detection already finds" assumption DOES NOT
TRANSFER from detection to type. Recorded as-is; no re-litigating
the numbers.

Candidate mechanisms (v2's job is to discriminate, not assume):
(a) CROSS-TRACK SCALE VARIANCE — detection compares a moment against
the SAME track's quiet moments; fast-vs-slow compares magnitudes
ACROSS players/depths/tracking quality, which raw features were
never built to survive. (b) TEMPO-FAST ≠ SWING-FAST — counters are
18/41 of the fast class and a counter is a compact punch block,
plausibly dink-like in arm speed; the mapping (frozen on tempo
grounds) may have diluted the kinematic class. (c) MOTION BLUR AT
CONTACT — the faster the swing, the worse the wrist tracking at the
strike (POSTMORTEM's blur finding), biasing fast arm readings DOWN
and compressing exactly the gap the classifier needs.

v2 additions (selftested; layout-pin, scale-invariance, jitter-
rescue, extended null all green): model-free per-feature descriptives
(arm_cmax/arm_emax/dsho/leg/dgaze fast-vs-slow AUC + medians;
gap_prev/gap_next AUC; per-TYPE arm_cmax + gap_prev medians;
per-image-side arm_cmax AUC), and POSE-N — each speed stat divided by
that channel's own track-level top-5% mean, plus POSE-N+CAD.

REGISTERED PREDICTIONS FOR THE FIRST v2 RUN (before it exists):
(1) primary diagnostic is raw arm_cmax fast-vs-slow AUC; I predict
0.60–0.70 — weak but not dead. (2) mechanism (b) shows up: smash and
drive arm_cmax medians >= 1.5x dink's, while COUNTER sits within
~1.3x of dink — "fast" as mapped is two different things, and the
kinematic signal is real but diluted. (3) POSE-N+CAD gains over
POSE+CAD's 0.586 AUC by +3–8pp — a partial rescue, NOT a return to
the falsified 85–95; if it gains >= +10pp, mechanism (a) was
dominant. (4) within-side arm_cmax AUCs exceed the pooled arm_cmax
AUC (depth/scale confound visible without POSE-N). (5) counters'
gap_prev median is the lowest of any type (tempo IS their identity).
If (2) holds, the likely end state is a TWO-AXIS story — swing size
from pose (smash/drive put-aways), tempo from cadence/decoder
(firefight segmentation) — and the product's "fast" for firefight
analytics rides tempo, not pose. Decided by the run, not here.

Practical: user's flat folder lacks label_split.csv (v1 printed the
loud warning). Harmless TODAY — the 10 pose-covered labeled rallies
are all in the frozen train block (game-1 rallies 1–21) — but it
must be present before labels grow past rally 21; file sent
alongside v2.

## 2026-08-18 — fastslow v2 real run + LABEL SEMANTICS correction (user)

V2 run (user's Mac; fingerprints efb79d5003 / bce74d0f26; split file
now present, holdout untouched; the three v1-shared feature sets
reproduce v1's numbers exactly, so the v2 edits moved nothing):
- Descriptives: raw arm_cmax fast-vs-slow AUC **0.445 — INVERTED**
  (slow med 0.5032 > fast 0.4745); arm_emax 0.485; leg_cmax 0.596
  (best raw pose single feature); dsho/dgaze ~0.5. gap_prev 0.349 /
  gap_next 0.242 as P(fast>slow) = 0.651 / **0.758 in the coded
  direction — gap_next (time to NEXT contact) is the strongest single
  feature anywhere in the instrument**. Within-side arm_cmax 0.433 /
  0.459 ≈ pooled (no depth signature in the raw magnitudes). ALL gap
  medians sub-1.3 s (dink 0.93 s) — real kitchen tempo; the 3.0 s cap
  almost never binds and "quiet windows" barely exist, consistent
  with feature_check's contamination sweep.
- LORO (fold-honest): CADENCE 54.2%/0.600, POSE 55.6%/0.545,
  POSE+CAD 58.3%/0.586, POSE-N 52.8%/0.571, **POSE-N+CAD 61.1%/0.632
  (best; +4.2pp over the 56.9% majority)**.
- CAVEAT, self-inflicted: the descriptive AUCs are POOLED, not
  fold-honest — they can ride between-rally composition (fast-heavy
  rallies are fast-tempo rallies everywhere), which LORO correctly
  refuses to credit. gap_next 0.758 pooled vs CADENCE 0.600 LORO
  quantifies exactly that gap. The docstring's "nothing to overfit"
  oversold it; between-rally composition is a confound the pooled
  number keeps.

Prediction scorecard (registered pre-run): (1) arm_cmax 0.60–0.70 →
WRONG — 0.445, inverted; magnitude deader than the registered floor.
(2) smash/drive >=1.5x dink → WRONG (smash 1.05x; drive 0.49x, the
LOWEST of all types); counter <1.3x dink → right (0.63x). (3) POSE-N
+3–8pp AUC → RIGHT (+4.6pp, 0.586→0.632). (4) within-side > pooled →
WRONG (equal). (5) counter lowest gap_prev → RIGHT (0.61 s).

**LABEL SEMANTICS (user, same day) — applies RETROACTIVELY to every
per-type table in feature_check / channel_ablation / fastslow_check:**
smash, counter, and speed-up were coded interchangeably ("basically
the same for me" — no brightline attempted). Treat them as ONE class,
kitchen-fast (n=36 here); differences among those three rows are
uninterpretable — the "speed-up reads loud (0.81), initiation from
stationary posture" detail from the first v2 read is RETRACTED as
uninterpretable. drive = FAST FROM THE BASELINE; drop = SLOW FROM THE
BASELINE. The taxonomy actually coded is PACE x LOCATION. The
fast/slow BINARY was coded consistently, so the binary instrument is
immune to the trio noise — the headline AUCs stand; per-type tables
are the only casualty. Baseline shots (drive/drop) sit at the depth
extremes of the frame, and that is precisely where POSE-N helped
(drive 2/5 → 4/5 correct) — n tiny, direction consistent with the
scale mechanism biting hardest at depth.

Mechanism ranking after v2: (c) BLUR/TRACKING-AT-SPEED front-runner —
fast contacts read no louder than slow at true windows, within sides
too, which clipping-at-the-strike produces and scale variance alone
does not; (a) scale variance real but modest (POSE-N gain inside the
registered band; the drive fix); (b) tempo-vs-swing conflation now
untestable at type level (labels interchangeable) and moot for the
binary. NEW unmeasured confound (d): WRONG-HITTER-AT-SPEED — at
0.6 s any-team gaps the partner's follow-through occupies the very
prep window pick_hitter selects on, so fast-exchange rows may often
measure the wrong player's track. Untested (tracks carry side, not
within-side identity); cheap proxy if ever needed: pick-instability
rate (prep-pick vs core-pick disagreement) as a function of gap_prev.

Standing read: per-contact pace from pose on this stream ≈ 61%/0.63
at TRUE times — a ceiling, before the placement tax. The live product
path for firefight analytics is PHASE/TEMPO SEGMENTATION from decoder
gap sequences (aggregates consecutive gaps; needs no per-contact
pose); pose's remaining per-contact role gets re-tested after
"other" tagging roughly doubles n. Labeling implication offered to
user (not yet edited into labeling_protocol.md): bless the coarse
vocabulary — the trio can be typed as any of its three words or just
"fast"; drive/drop/lob/dink stay; literal fast/slow tags are enough
for the "other" backlog.

## 2026-08-18 — phase_grader.py built: the 2000-rally decision instrument (selftested; bands + predictions registered BEFORE the first real run)

User call: build the best phase-structure grader with everything
learned (alternation decoder, pose, kitchen-located tempo), see how
we do, then a verdict on whether 2000 coded rallies are worth it.

WHAT IT DOES. Grades the product stats — has_fast, first-fast time
(the speed-up moment, ±1.0 s), initiating TEAM, per-rally fast
share — in two arms whose DIFFERENCE is the verdict: LEVEL C applies
pace classifiers at TRUE labeled times (ceiling; placement removed);
LEVEL B runs the real pipeline (LORO swing scorer -> dense scoring ->
alternation decoder with ghost-aware ordinals -> pace on decoded
events, ghost-adjusted gaps, ordinals 0/1 = opening). Level A
supplements: decoded<->label matching (±0.5 s, same team) + pace acc
on matched contacts. Classifiers, both trained at true times on
train-fold paced contacts: GAP (min-gap threshold, fit per fold) and
FULL (logistic: POSE-N + directional gaps + NEW kitchen context —
per-rally net-line from the two sides' feet-line distributions,
hitter + all-player proximity; ynorm IS the feet line, box[:,3]).

PRE-REGISTERED VERDICT BANDS (FULL classifier): has_fast >=80%,
first-fast<=1s >=70% of certain fast rallies, init-team >=75%,
share corr >=+0.6. Regimes: B clears -> ship on current decoder,
2000 rallies NOT needed for training; C clears but B misses -> 
placement binds, temporal model justified, 2000 rallies are its
fuel; C misses -> structure not recoverable on this footage,
counts-only product. DEFERRAL CLAUSE: with 58 unpaced 'other'
contacts across 10 rallies, most rallies will be boundary-UNCERTAIN
and excluded from first-fast/team grading — if fewer than 4 certain
fast rallies survive, the STRUCTURE verdict defers to a post-tagging
re-run; per-contact (A/C) numbers and the match rate stand either
way.

REGISTERED PREDICTIONS (run 1): 5–8 of 10 rallies boundary-uncertain.
Level C per-contact: GAP 58–66%, FULL 62–70% (kitchen context is the
new hope over fastslow's 61%). Match rate (±0.5 s) 55–70%. Level B
structure visibly below Level C (first-fast hit −10 to −20pp).
Most likely regime read: C partially clears, B misses — leaning
"2000 rallies justified" — but the deferral clause probably
activates on run 1. BOUNDARY-SHOT LESSON (derived analytically,
encoded in the selftest before any real run): the min-gap heuristic
MUST misattribute the initiator — the last slow shot before the
speed-up has a short gap_next, so GAP calls the defender fast one
shot early (first-fast time still hits; TEAM wrong). Directional
gaps fix it (synth: FULL 4/4 teams vs GAP 0/4). Expect GAP
init-team to lag FULL on real data for exactly this reason.

BUILD TRAPS (both caught by selftest before real data): (i) synth
boxes were in a flat [x,y,w,h] layout inherited from the probes —
inert for arm-only tests, fatal for position (track_series boxes are
CORNER format; ynorm = box[:,3] = feet line); phase_grader's builder
now uses corners, fastslow's stays as-is (its tests make no position
claims). (ii) grid aliasing AGAIN: unjittered phase synths with one
slot->class pattern let the 68-dim FULL model ride frame-grid float
wobble (share corr −0.99 on features designed to contain nothing);
jittering the synth times restored honest behavior (56/56, teams
4/4). Second sighting same day — treat "synth on exact grid times +
class repeated across rallies" as a banned combination in this
codebase.

## 2026-08-18 — phase_grader run 1: VERDICT — 2000 rallies NOT licensed by this evidence

Run (user's Mac, fingerprints efb79d5003 / bce74d0f26, split
respected): truth 10 rallies / 6 with paced fast / 5 boundary-
uncertain (ff_n = 5 certain fast rallies — above the <4 deferral
line, barely). LEVEL C (true times): per-contact GAP 63.1%
(53/84), FULL 59.5% — FULL BELOW GAP, kitchen context added
nothing; structure GAP hf 6/6, ff<=1s 3/5 (med 0.00 s), team 4/5,
share corr −0.86; FULL hf 6/6, ff 4/5 (med 0.59 s), team 2/5, corr
+0.43. LEVEL B (decoded): 135 events across 10 rallies (one rally
decoded 3), match rate 48/84 = 57.1%, matched-contact pace acc
52.1/54.2% (gap/full — barely above coin), structure FULL hf 6/6,
ff 2/5 (med 1.44 s), team 4/5, corr +0.39.

BANDS (FULL): C cleared 2/4 (has_fast 100% ✓, ff 80% ✓; team 40% ✗,
corr +0.43 ✗). B cleared a DIFFERENT 2/4 (hf ✓, team 80% ✓; ff 40%
✗, corr ✗). Registered logic: C misses -> "structure not
recoverable; counts-only product" -> 2000 rallies NOT justified.
This OVERRULES the registered lean ("most likely C clears, B
misses; leaning justified") — the bands beat the lean, which is
what pre-registration is for.

Prediction scorecard: uncertain 5 (band 5–8 ✓ at the edge); C GAP
63.1 (58–66 ✓); C FULL 59.5 (62–70 ✗, and < GAP); match 57.1
(55–70 ✓); B-below-C direction ✓ but size under-called (ff hit
−40pp, predicted −10 to −20). Boundary-shot lesson (GAP
misattributes initiator) INVERTED at Level C on real data (GAP 4/5
vs FULL 2/5) while matching the prediction at Level B (1/5 vs 4/5)
— i.e., at n=5 the team stat flips arm-to-arm; treat every
structure row as ±20pp.

What run 1 actually established: (1) THIRD independent confirmation
that per-contact pace saturates ~60% on this stream — fastslow raw,
fastslow POSE-N, and now FULL-with-kitchen all land 59–63% at TRUE
times; kitchen context is a null addition (FULL < GAP at C).
(2) Structure stats BEAT per-contact (aggregation forgives): hf
6/6, ff medians 0.0–0.6 s at C. (3) The placement tax is real and
specific: matched acc 63→52, ff med 0.59→1.44 s — decoded timing
noise corrupts the gap features that ARE the signal. That is the
only regime-2 evidence, and it is capped by (1): perfect placement
buys back only the C ceiling, which itself missed bands.

FORWARD (before ANY mass labeling — both cheap, both could overturn
run 1): (a) tag the 58 'other' contacts fast/slow (~15 min):
restores the 5 deferred rallies, roughly doubles paced n, rerun the
SAME grader/bands; (b) sequence-level segmenter — the classifiers
graded here are PER-CONTACT thresholds; the actual phase model is a
2-state segmentation over the GAP SEQUENCE (run-length/HMM-style),
buildable with zero new labels and gradeable by this same
instrument. If C clears bands after (a)+(b), revisit the 2000-rally
question with a real case. Standing hypothesis for the product
regardless: CONTINUOUS tempo stats (share of sub-0.7 s gaps, tempo
curves, rally length) sidestep classification entirely and need
only decoded TIMING FIDELITY — which is also the one thing the
temporal model demonstrably improves. If a labeling investment is
ever justified, it will be by timing fidelity for continuous
stats, not by the binary classifier.

## 2026-08-18 — phase_grader V2: SEQ (2-state HMM over the gap sequence) — predictions registered BEFORE the first run

User (correctly) called out that run 1 graded per-contact
classifiers when the sequence model was the stated design. V2 adds
SEQ into the SAME grader so all three land side by side, same folds,
same bands: a supervised 2-state HMM — one hidden state per contact
= the state of the gap it PRODUCES (a fast shot forces a fast
reply; fastslow v2 measured produced-gap (gap_next) as the strongest
single feature, 0.758 coded direction). Log-normal emissions per
state fit on train-fold labeled gaps (clip [0.15, 6.0] s, log-sd
floor 0.15); transitions from ADJACENT both-labeled pairs within
rallies (the run-length prior, learned); Viterbi over the full
sequence including unlabeled/opening gaps; rally-ending shot (no
produced gap) takes the transition prior. GAP-ONLY EMISSIONS BY
DESIGN — run 1 measured 68 features losing to one threshold at
n=84; the HMM's edge is the temporal prior, not more channels.
Level B runs SEQ on decoded events with ghost-adjusted gaps;
fallback to the GAP threshold if a train fold lacks 8 labeled gaps
per state.

RETRACTION (synth physics): the run-1 selftest's "min-gap
misattributes the initiator to the defender one shot early" lesson
was an ARTIFACT of unphysical synth timing (speed-up arriving 0.45 s
after the last dink). Under produced-gap physics — the speed-up
ARRIVES on a dink-paced gap and PRODUCES the first short gap, which
real data confirms (speed-up gap_prev med 0.98 ~ dink 0.93) — the
min-gap failure mode is the RESET (slow shot arriving on a fast
gap), one per firefight, and min-gap gets the initiator RIGHT.
Run 1's real Level C (GAP teams 4/5) agreed with physics, not with
my synth. Corrected synth: dinks -> firefight -> reset + cool-down,
jittered; derived and confirmed: GAP 64/68 (the 4 resets), SEQ
68/68 (reset produces a slow gap; rally-ender inherits its run),
FULL 68/68, all teams 4/4. Viterbi pinned to brute-force
enumeration; context-flip test pins the HMM's actual value on
overlapping real-data emissions (same 0.85 s gap reads slow in a
1.3 s run, fast in a 0.6 s run).

REGISTERED PREDICTIONS (before the first real V2 run): C SEQ
per-contact 66–74% (point ~70; beats GAP 63.1 — right feature
direction + run prior); C SEQ ff>=3/5, init-team >=3/5, share corr
positive; B SEQ matched acc >= GAP's 52.1, ff median <= 1.0 s (vs
FULL's 1.44 — smoothing noisy decoded gaps is where the prior
should pay most). KNOWN LIMITATION registered: transitions need
ADJACENT both-paced pairs, and 58 unpaced holes thin them — if
fitted stickiness comes out near-uniform, SEQ degenerates toward a
gn-threshold; tagging the 'other' backlog fixes this too.
REGISTRATION AMENDMENT (pre-run): the verdict bands now read on the
BEST of the three classifiers (expected SEQ), not FULL specifically;
regime logic unchanged.

## 2026-08-18 — phase_grader run 2 (SEQ): predictions MOSTLY FALSIFIED; V3 posterior readout; MODEL-ITERATION FREEZE

Run 2 (same fingerprints/split): C SEQ per-contact 65.5% (55/84) —
best of the three families but ONE CONTACT below the registered
66–74 floor; C SEQ structure COLLAPSED (has_fast 5/6, ff<=1s 1/5
med +3.25 s, team 2/5, share corr +0.03); B SEQ catastrophic
(matched acc 50.0%, has_fast 1/6, ff found once at +12.86 s).
Scorecard: per-contact graze-miss; every structure prediction
MISSED; "B ff med <= 1.0 s" missed by 12x. The registered
limitation (hole-thinned adjacent-pair transitions) fired in the
worse direction: not degenerate-to-threshold but OVER-STICKY toward
slow.

DIAGNOSIS (mechanism, visible in the numbers): Viterbi's MAP path
suppresses the rare state. With overlapping real emissions
(fast-produced med 0.68 vs slow 1.05, wide sd) one short gap earns
less log-likelihood than the fitted slow->fast transition costs, so
the MAP path enters firefights late or never — first-fast ~3 shots
late at C, has_fast 1/6 at B. MAP optimizes whole-path probability;
nothing we grade is that loss.

V3 (same file): readout switched to forward-backward POSTERIOR
marginals at 0.5. Pinned in selftest against brute-force path
enumeration (1e-9); context-flip holds under marginals; clean-synth
end-to-end unchanged (68/68). Registered expectation: V3 fixes the
SUPPRESSION pathology (has_fast, ff lateness should recover toward
GAP/FULL levels) but does NOT move the ~63–65% per-contact ceiling
— the information per gap is what it is.

**MODEL-ITERATION FREEZE (house discipline).** Three model families
— min-gap threshold, 68-feature logistic, 2-state HMM — have now
been evaluated against the SAME 72 paced contacts, landing 59.5 /
63.1 / 65.5 at Level C. The sample is exhausted for model
selection; further variants against it are knob-turning on a frozen
tiny sample. NO new pace models until the label base changes.
The next run that carries evidential weight is THE POST-TAGGING RUN:
user tags the 58 'other' contacts fast/slow (~15 min, the coarse
judgment they trust), then this same V3 grader, same pre-registered
bands, now on ~130 paced contacts, ~10 certain rallies, and
repaired transition adjacency. PRE-REGISTERED READING OF THAT RUN:
if best-classifier Level C structure clears the bands -> the
2000-rally question REOPENS with a real case (placement tax is
demonstrably the remaining binder). If C structure still misses ->
the pace-classification thread CLOSES on this footage and the
product ships counts + continuous tempo stats (rally length, gap
distributions, tempo curves — no classification needed). The
2000-rally position TODAY, three families deep: NOT licensed,
unchanged from run 1.

## 2026-08-18 — audit tool build 2026-08-18a: PACE PASS (two-pass labeling UX)

User request: easy UX for coding player + fast/slow — pass 1 player
(as now), pass 2 fast/slow with a rewind to rally start. Built into
make_contact_audit.py / contact_audit_chicago0725.html:

- P rewinds the rally (existing openSeek chain: serve stamp > pin >
  chained-seek) and replays; F/S tag the armed contact and advance;
  auto-exits when nothing is left; toast + orange F/S badge + "N
  await pace pass" counter surface the queue. Backspace in pace mode
  un-tags (restoring 'other' exactly, not blanking); tap-deletion
  undo is gated off while pacing, as is stamping (stray 1-4/Enter
  during the replay can't add taps).
- SKIP RULE = paceNeeds(): a real contact whose effective type is
  empty or 'other'. Serves/returns (position rule), whiffs, and
  granular types skip — so the 58-contact 'other' backlog IS the
  pace queue on old rallies, and fresh rallies need exactly shots 3+.
- SCHEMA-STABLE: tags are literal fast/slow in the existing
  shot_type column (fastslow_check's frozen mapping already accepts
  them); export/import/consumers untouched. fast/slow added to the
  type dropdown so manual round-trips work.
- done() unchanged and now MEANS both passes: a fresh rally
  completes only after hitters + pace are in.
- Tested against the REAL built script under node with a stubbed DOM
  (scratch pace_test.mjs, 12 assertions): skip rule, arm/advance,
  auto-exit, exact 'other' restore on undo, serve/return rule
  intact, done() semantics, stamp gating, prefilled-core no-op.
  Build banner bumped 2026-08-17a -> 2026-08-18a.

## 2026-08-18 — LUNGE RULE (user policy, implemented as its own word; tool build 2026-08-18b)

User policy: "lunged for a ball but made a forced error" must not
contaminate the pace dataset. Implemented WITH one modification,
explained to the user: the exclusion is right, but coding it "other"
would (a) fight the pace-pass UX (the F/S badge and armed queue treat
'other' as awaiting judgment — no skip exists) and (b) destroy the
unjudged-vs-judged distinction right after the workflow was built to
drain 'other' to zero. So: new first-class type **lunge** = contact
WITHOUT a real swing (desperate reach/stretch/stab, usually a forced
error). Semantics wired end to end:
- audit tool 2026-08-18b: X during the pace pass (and a row X button
  + dropdown entry); lunge rows are settled — no badge, done() ok;
  node test extended to 15 assertions (tag, un-tag restoring 'other',
  not-paceable) — green.
- fastslow_check: classify_type gains 'nonswing' (NONSWING =
  {lunge}); excluded from pace classes, counted separately in the
  exclusion line, never nudged, never an unknown-vocabulary warning.
- phase_grader truth_structure: lunges carry pace None but are
  JUDGMENTS — they never create boundary uncertainty (a lunge cannot
  be the rally's first attack), unlike untyped holes; n_nonswing
  reported; selftest cases added (lunge before first fast -> certain;
  all-slow + lunge -> certain). gap fitting already excludes None
  labels; lunge contacts still count for gaps/counts/alternation —
  they are real contacts and the decoder depends on them.
Detector-training note: rally_instances POSITIVES deliberately keep
lunge contacts (contact events are what step 1 detects; dropping them
would corrupt counts). The distinct word preserves the option to
filter them from any future swing-KINEMATICS training set.

## 2026-08-19 — POST-TAGGING CHECKPOINT RUN (fingerprint a0d9248a35): REGIME 2 FIRES — placement binds, the labeling program is justified

The pre-registered decision run (user's Mac; 10 pose rallies, backlog
fully drained: 0 untyped, 3 lunges, 0 boundary-uncertain — the pace
pass + lunge rule worked exactly as designed on first contact; 82
holdout rows quarantined automatically; 25 rallies labeled total,
11-21 awaiting pose extraction, 22-25 holdout).

LEVEL C (true times), n=138 paced: **GAP 74.6%** (from 63.1), SEQ
71.7%, FULL 58.7. Structure GAP: has_fast 10/10, ff<=1s 8/10 with
**median 0.00 s**, init-team 8/10, share corr +0.55. BANDS
ADJUDICATION (pre-registered: >=80 / >=70 / >=75 / >=+0.6 on the
best classifier): three HARD CLEARS (100%, 80%, 80%) and share corr
0.55 vs 0.60 — at n=10 rallies a correlation carries se ~0.3, and
the guide's own grain warning covers it. CALLED AS A CLEAR, with the
marginal criterion noted here permanently.

LEVEL B (decoded): match rate **45.7%** (down from 57.1 — the tagged
backlog is disproportionately the hard-to-see contacts, including
the 32 no-pose rows, exactly where the decoder struggles), matched
pace acc 41-56%, structure ff<=1s <=4/10 with medians 0.79-5.04 s,
init-team <=4/10. B misses the bands decisively under every
classifier.

**=> Pre-registered routing: C clears, B misses — PLACEMENT BINDS.
The temporal-model labeling program is justified; the 2000-rally
answer flips to a scoped YES.** Scope per the standing plan: start
with full Chicago (~188 windowed rallies / ~3k contacts — at the
user's measured 25 rallies/day, about a week of casual labeling)
feeding a temporal model class under a FRESH pre-registration whose
success metric is now measured, not aspirational: CLOSE THE B->C
STRUCTURE GAP (ff<=1s 4/10 -> ~8/10, init-team -> ~8/10, match rate
45.7% -> toward the C ceiling). Multi-VOD 2000-rally scale stays
CONTINGENT on the Chicago-trained model actually closing that gap.
Before the big push: pose_extract rallies 11-21 and rerun — a free
stability check of the C structure stats on 2x rallies, no new
labels needed.

Supporting findings, now at n=106-138: (1) POSE IS TRIPLE-DEAD for
pace — alone 0.493 AUC (coin), SUBTRACTS when added to cadence
(0.689 -> 0.578), and the arm inversion DEEPENED in the fresh coarse
tags (newly tagged 'fast' others: arm_cmax 0.265, the lowest of any
type; 'slow' others: 0.628, the highest — the ambiguous contacts are
fast-exchange touches where tracking fails hardest, consistent with
the blur/wrong-hitter mechanisms). gap_next strengthened to 0.778 in
the coded direction. (2) SIMPLICITY WON: the fitted min-gap
threshold now BEATS the HMM at C (74.6/8-teams vs 71.7/6) — GAP is
the pace classifier of record; SEQ retired from that role (the
posterior readout did fix the run-2 suppression pathology: B
has_fast 7/10 vs Viterbi's 1/6, C ff found 10/10 med 0.39 s).
(3) Prior SEQ registrations: C 66-74 predicted -> 71.7 IN BAND;
"B SEQ matched >= GAP" -> missed (50.8 vs 55.6). (4) The
model-iteration freeze STANDS for pace classifiers; the temporal
model is the placement program under fresh pre-registration, which
the freeze explicitly allows.

## 2026-08-19 — TEMPORAL GATE FROZEN (`vision/temporal_gate.md`) — before any temporal code exists

The fresh pre-registration the checkpoint called for, written same
day, while the label archive is still the checkpoint's 25 rallies
and pose still covers only 1–10 — i.e. before any data the temporal
model will train on beyond what the routing decision already saw,
and before a single line of temporal-model code. One verdict run on
the full Chicago holdout (76 rallies; BURNS it for the whole swing
thread), three systems same panel same run (C ceiling / B baseline
refit on final train / T frozen), per-stat closure bars
(PASS = match ≥65% + ≥2 of 3 structure closures ≥2/3;
KILL = match <55% or median closure <1/3), a train-only per-division
pre-check so an untested mixed-division ceiling can't masquerade as
decoder failure, and a <40-rally power clause that caps small panels
at MIDDLE. The S4-floor rationale is recorded in the gate: structure
gains without placement gains = the decoder painting its own prior.
Consequences pre-committed: PASS licenses multi-VOD; MIDDLE gets one
amendment cycle + final shot on future-VOD holdout; KILL falls back
to the labeled-matches (Level C) product with labels keeping full
value. Registered design input carried over: joint decode where the
pace state sets the expected next-gap distribution.

## 2026-08-19 — REGISTERED PREDICTIONS: pose-doubling stability run (before the user runs it)

User labeled through rally 27; **rallies 11–12 are not in the VOD**
(user report), so the pre-registered core-16 tops out at 14 in
practice — the fast-heavy pair the dev set hoped for doesn't exist
in this footage; fast coverage now comes from the wider train corpus.
Plan: pose_extract the newly labeled rallies (13–21 train; 22–27
holdout may be extracted too — mechanical data prep, scripts never
read holdout rows), then rerun phase_grader → train panel doubles to
~19 pose rallies.

Registered BEFORE the run (Level C = GAP classifier of record):
- per-contact pace: 68–80% (checkpoint 74.6)
- has_fast ≥ 0.85; first-fast ≤1 s 0.60–0.90; init-team 0.60–0.90
- fast-share corr +0.35 to +0.70 (se shrinks ~0.3 → ~0.2)
- SEQ prior printout: P(fast→fast) 0.70–0.88 (center 0.80 — the
  user guessed 80% cold), P(slow→slow) 0.80–0.92; median gap after
  a fast shot 0.4–0.7 s, after a slow shot 1.2–2.0 s
- Level B match rate 40–55% (same regime, no reason to move)

STABILITY READING (registered): C structure holding Regime-2 levels
(has_fast ≥ 0.85 AND ff ≥ 0.6 AND teams ≥ 0.6 AND GAP ≥ 65%) =
STABLE → labeling + temporal build proceed as planned. Any stat
collapsing below its floor = pause and diagnose before further
labeling investment. This is a monitoring read on the ROUTING
decision, not a verdict — the only verdict instrument for the
temporal model is temporal_gate.md.

## 2026-08-19 — LABELS-ONLY C PREVIEW (labels committed, fingerprint 1165ddb642): floors grazed, diagnosed to TWO exhaustive blind spots; "C's ceiling is softer on the true rally mix"

User shipped the 25-rally archive (323 contacts; 1–10, 13–27; ZERO
untyped; committed to data/vision/). Since Level C + SEQ prior need
only label times, the C side ran here ahead of the Mac run (scratch
c_preview.py mirrors run_all's C loop verbatim, pose shimmed to
empty tracks so FULL degrades to GAP per classify()'s documented
fallback; Level B and FULL await the Mac run, which remains THE
record). Holdout 22–27 quarantined from all stats; holdout rows were
touched only by file-integrity QC (row counts, time monotonicity,
team-alternation parity — no pace/structure content read).

INSTRUMENT VALIDATED: restricted to rallies 1–10 the preview
reproduces the checkpoint EXACTLY (74.6%, 10/10, 8/10 med 0.00,
8/10, +0.55) — labels for 1–10 unchanged, shim faithful.

ALL-TRAIN (19 rallies / 187 paced, th 0.597): GAP 73.8%, has_fast
16/19 (0.84), ff≤1s 9/16 (0.56), init-team 11/16 (0.69), share corr
+0.01. SEQ: 73.3%, ff 11/16 — seq BEAT gap on ff on the doubled
panel (checkpoint was the reverse); recorded, NO switching (freeze).
vs REGISTERED: pace 73.8 IN 68–80; teams 0.69 IN; has_fast 0.84
GRAZES the 0.85 floor; ff-gap 0.56 MISSES the 0.60 floor; share
corr +0.01 FALSIFIES +0.35–0.70 badly. Prior: P(f→f) 0.73 (in band,
user's cold 80% guess close), P(s→s) 0.74 FALSIFIES my 0.80–0.92;
gap after fast 0.65 s in band; after slow 1.00 s FALSIFIES 1.2–2.0
(real dink cadence is 1.0 s, my prior was wrong).

DIAGNOSIS (per the registered pause-and-diagnose): the drop is the
NEW COHORT (13–21: 65.3% pace, has_fast 6/9, corr −0.51), which
adds 4 serve-out/tiny rallies + short broken rallies — the real
rally mix, vs the structure-friendly curated core. On ≥5-paced
rallies (13): 74.3%, has_fast 12/13, ff 8/13 (0.62), teams 10/13,
corr +0.35 — near checkpoint except ff. The fast-miss decomposition
is EXHAUSTIVE (36 misses of 89 truth-fast, zero unexplained):
  (a) RALLY-ENDERS 6/10 missed — the putaway winner produces no
      next gap; min-gap falls back to the arrival gap (often
      dink-paced). The rally-ending smash is structurally invisible
      to produced-gap logic.
  (b) LONG-PRODUCED-GAP FASTS 30/45 missed — fully HALF of
      truth-fast contacts produce a gap > th: baseline drives
      (flight time ∝ distance), smashes absorbed by defensive
      resets, exchange-ending counters. Not taxonomy (misses spread
      counter 10 / fast 13 / smash 5 / drive 5 / speed-up 3) —
      physics of the reply.
  Sharpest form: the THIRD-SHOT DRIVE is the modal first-fast AND
  exactly the gap-invisible type — that is why ff suffers most
  (r14/r17 shapes). 15/45 long-gap fasts are rescued only because
  they SIT on a fast arrival gap (mid-firefight min-gap).

IMPLICATIONS (no classifier iteration — freeze stands; these are
temporal-model DESIGN INPUT + gate bookkeeping):
  1. C's tempo-only ceiling on the true mix: has_fast/teams/
     per-contact hold; ff ~0.56–0.62 and share corr soften. The
     checkpoint's 10 rallies over-stated ff/corr at C.
  2. The temporal model should carry NON-tempo channels for the two
     modes: position (kitchen feats — baseline-ness IS the drive
     signal; FULL may already show this on the Mac run) and
     rally-end context from logs (the ender's "no next gap" is
     itself observable — last contact + short time-to-rally-end).
  3. temporal_gate PRE-CHECK TENSION, flagged not resolved: womens
     train ff at GAP = 0.56 < the 0.6 pre-check bar frozen
     yesterday — as frozen, womens could fail its own ceiling
     pre-check (on ≥5-paced it passes at 0.62). PENDING AMENDMENT
     QUESTION (decide AFTER the Mac run shows FULL-at-C, BEFORE any
     temporal code): whether C's readout in the gate should be the
     best already-graded classifier on train (gap/full/seq) rather
     than GAP-only. Recorded here so the amendment, if made, is
     visibly pre-temporal-code.
  4. Label-integrity: r26 (holdout) has a hitter mis-key — shots
     5/6 hitters swapped breaks team alternation twice; swapping
     restores parity. Flagged to the user for a tool fix +
     re-export; pace content not read.

## 2026-08-19 — USER TAXONOMY INSIGHT CONFIRMED: the drive exchange is not the attack; ATTACK-ONSET semantics prototyped (train only)

User claim: "3rd-shot drives shouldn't count as first-fast — the meta
is 3: drive, 4: counter, 5: dink." MEASURED on train (fast shot-3
rallies, n=6): 4 follow the meta EXACTLY (fast-fast-slow: r2 drive/
drive/drop, r5 drive/counter/lob, r14 drive/counter/drop, r20), one
is a shot-3 ender (r18), one ESCALATES (r3: the drive gets SMASHED
back — shots 3-6 all fast, a genuine drive-ignited firefight). ZERO
drives are answered slow — the counter is forced, which is exactly
why drive-counter is transition pace, not a kitchen break. Bonus
pattern preserved: shot-4 fast by the RETURNING team after a slow 3
= attacking the weak third-shot drop (r6/r8/r9/r17/r21) — a real
attack, not transition.

ATTACK-ONSET definition prototyped (scratch; not yet in the
grader): fast runs over paced contacts (lunges/whiffs transparent);
a run is an ATTACK iff it ends the rally, OR length>=2 starting at
ordinal>=4, OR length>=3 starting at ordinal 3 (escalation clause —
data support n=1, r3). Onset = first attack run's first contact.
Applied symmetrically to truth and to classifier calls at true
times (GAP, LORO): has_attack agreement 15/19, onset<=1s 8/14,
init-team 10/14. Headline rates ~unchanged vs first-fast (0.57/0.71
vs 0.56/0.69) BUT the misses change character: r2 flips to an EXACT
hit (truth now the shot-8 speed-up the classifier always found —
the 4.4s "miss" was the truth definition, not the model); r5 and
r20 become correctly attack-free (drive exchange + lone lob-smash /
dink marathon); remaining misses are the ENDERS (r14/r18/r21) plus
a new small mode (r20 pred: two quick dinks under threshold form a
fake length-2 run). Net: the semantics change grades the thing the
product cares about and EXPOSES the ender blind spot cleanly
instead of burying it in definition mismatch. It does not rescue
the rates — enders need a non-tempo channel regardless.

GATE IMPLICATION queued with the C-readout question: S2's truth
definition is part of the frozen bars, so adopting attack-onset
means SUPERSEDING temporal_gate v1 with a dated v2 (no temporal
code exists, holdout untouched — supersession now is clean; a
strained "amendment" would not be). USER DECISION pending.
→ RESOLVED same day: user approved ("Agreed"); v2 CUT — see the
next entry.

## 2026-08-19 — TEMPORAL GATE v2 CUT (user-approved) + attack-onset wired into phase_grader

`vision/temporal_gate.md` v2 supersedes v1 same day (v1 preserved
at 479c7e6; still zero temporal code, holdout untouched, all
evidence train-only). Three changes, logged in the doc's changelog:
(1) S1–S3 graded on ATTACK-ONSET (frozen definition = the
prototyped rule: fast runs bridge lunges/whiffs; attack iff ender,
len≥2 @ord≥4, or len≥3 @ord3); (2) C's readout = best-of
{gap,full,seq} by train per-contact accuracy at verdict time
(deterministic, train-only — resolves the C-readout question);
(3) pre-check refined to ≥2-of-3 bars per division (wholesale
exclusion was built for collapsed ceilings; womens' single soft
stat — onset 0.57 — now narrows the verdict via the informative
guard instead of evicting the division). First-fast demoted to a
secondary continuity stat.

phase_grader implementation (same commit): attack_onset() +
truth_attack() (grade()-compatible key names, documented);
Catt/Batt outputs at both levels for all three classifiers
(decoded ordinals ghost-aware, ord+1 = shot number); report()
prints an attack-onset block under each level + rewritten verdict
guide pointing at the gate. Selftests: 9 unit cases (meta
exclusion, escalation, enders incl. shot-3 putaway and
trailing-lunge enderhood, lone-reset skip, attack-the-3rd,
lunge-bridge, uncertainty rules) + end-to-end synth asserts (gap
reset errors append to the run and cannot move the onset; all-dink
rally attack-free) + determinism extended to Catt. Module verified
to reproduce the prototype EXACTLY on real train labels (15/19,
8/14 med 0.00, 10/14). ALL OK.

## 2026-08-19 — VLM/API CHANNEL OPENED (user question): registered predictions + blind test kit

User: "could we run this through an AI API to detect things? ... if I
showed you screenshots could you tell who was hitting a ball?" Grep of
vision/ + vision_adjudication: NEVER CONSIDERED — a real gap, not
settled ground.

ASSESSMENT (what the record already licenses): (a) DETECTION is walled
for any pixel instrument, API or not — the user found no ball in 36% of
in-play frames with a loupe and unlimited time, so a model cannot detect
what is not in the pixels; (b) the binder is PLACEMENT (temporal
localization over ~145k frames) and VLMs are recognition engines —
video-native APIs sample ~1 fps against 0.45 s firefight gaps, i.e.
structurally blind to the shots that matter most; (c) HF models solve
step 2 (stroke classification on segmented strokes, ball tracking on
visible balls) while we are stuck on step 1, and the transfer is
measured (TrackNet 46% probe). BUT (d) the pose channel may be failing
on FEATURE CRUDENESS, not information: our features are wrist velocity
in a window; a VLM reads whole-body configuration. Live hypothesis
worth testing: a VLM is a better SWING-READER than our engineered
features, usable as a scorer over candidate windows (gate-legal — the
temporal model MAY use any model class).

EASY-CASE DEMO (user-run, 2026-08-19): a pre-serve screenshot, hitter
called correctly (near-left, serving) — but the reasoning that settled
it was RULES, not vision: Chicago at 5 (odd) must serve from the left,
which pins the server, and the diagonal pins the receiver; the Utah-
serving alternative was ruled out because it requires a deep receiver
on the near right and nobody is standing there. LESSON, and it is the
reason the test kit excludes serves: serve/return hitters are inferable
from score parity + the lineup state machine (99.25%, no camera), so
scoring them would inflate any VLM result with cases no model is needed
for.

REGISTERED BEFORE ANY BLIND FRAME IS SEEN: 4-way hitter call 60-80%
(chance 25%; 50% treating team as given), and specifically 15-25pp
WORSE on fast contacts than slow — the predicted failure mechanism is
motion blur plus posture ambiguity in kitchen exchanges, i.e. the SAME
mechanism that killed the pose channel (finding: pose alone 0.493 AUC,
arm inversion deepest in fast tags). If the fast/slow gap is small AND
overall clears 80%, the VLM sees something the engineered features do
not and the scorer hypothesis is live. If it lands ~65% with a big fast
hole, the VLM inherits the same wall — cost: a few screenshots.

KIT: `vision/vlm_frame_sample.py` — seeded balanced draw (default 20,
10 fast / 10 slow) from TRAIN rallies only, serves/returns/whiffs/
lunges excluded, third shots deliberately kept (the gap-invisible
drive is the population of interest), each question cut as a 3-frame
vstacked strip (t-0.1/t/t+0.1 — a lone contact frame is often
blurred), shuffled question order, answer key written to a separate
file. Dry-run against the committed labels: 187 eligible contacts (89
fast / 98 slow) over 16 rallies; draw is deterministic, dupe-free,
seed-sensitive, and asserted to leak neither holdout rows nor
openings. NOTE the honest limit on value: even at 90% this does not
speed up labeling (the bottleneck is watching and tapping, which a
hitter pre-fill never touches) — the value would be as a component
inside the temporal model, not an assistant beside the labeler.

## 2026-08-19 — LABEL SEMANTICS DISCLOSURE (user, unprompted): ambiguous contacts were coded by OUTCOME, and it re-frames the whole pace channel

During the blind VLM test the user disclosed how the hard cases were
coded: "if it successfully took the pace off the ball it was slow but
if it popped up it was fast. That's not really discernible." This
applies to the defensive-reach subclass — stretched, low, off-balance
contacts where the physical action is identical either way. It was
volunteered AFTER my calls were locked and BEFORE the key was sent, so
it cannot have been shaped by scoring.

Four consequences, in order of how much they matter.

1. **THE PACE LABEL IS NOT A STROKE LABEL — IT IS RALLY STATE.** The
   user's rule is, independently, EXACTLY the produced-gap semantics
   frozen in phase_grader ("a contact's pace = the state of the gap it
   PRODUCES"): pace taken off ⇒ long next gap ⇒ slow; popped up ⇒
   opponent attacks immediately ⇒ short next gap ⇒ fast. A human
   arrived at the frozen definition without being told it — good
   evidence the definition matches what "fast" means to someone
   watching. But stop calling this shot classification in any writeup;
   it is rally-state estimation, and the honest name matters because
   of (2).

2. **PARTIAL CIRCULARITY IN THE GAP RESULT, now on the record.** For
   this subclass the LABEL is derived from the outcome and the GAP
   FEATURE is the gap that outcome produces — two views of one latent
   quantity, so agreement there is partly definitional rather than
   independent validation. Scale: the subclass looked like ~4/20 of
   the blind sample (~20%), so the pooled Level-C 74.6% is plausibly
   a few points optimistic as a measure of "reading the shot". It is
   NOT wrong as a measure of "recovering rally state", which is what
   the product ships. Both readings should be quoted with their
   scope. Does not disturb the Regime-2 routing (that turned on the
   B-vs-C gap, and the circularity applies equally to both levels).

3. **IT IS A HARD CEILING ON EVERY INSTANT-BASED INSTRUMENT — and an
   argument FOR the temporal model.** If the label is a function of
   the ball's subsequent trajectory, then no pose quality, backbone,
   resolution, or VLM at the contact instant can recover it: the
   information is not in the frame, it is in the future. That single
   fact retro-explains Gate C's fast-shot deficit, the pose channel's
   arm inversion on fast tags, and my own blind-test skew (7 fast
   called where 10 exist). The temporal model is the FIRST instrument
   in this program whose input actually contains the label's
   information. Registered as the strongest a-priori argument yet for
   temporal_gate v2's program.

4. **OPEN SEMANTIC QUESTION for attack-onset (S3), flagged not
   resolved.** A pop-up is coded fast, so pop-up → opponent putaway
   is a length-2 fast run and attack_onset attributes the ONSET to
   the team that popped it up. Defensible as "when the rally turned"
   (the product's question), misleading as "who attacked". The two
   are distinguishable in principle — a fast contact struck from full
   stretch is a pop-up, one struck from a balanced kitchen position is
   a speed-up, and the kitchen/position features already in pace_row
   carry that — but that is a post-hoc refinement needing data, NOT a
   change to the frozen v2 definition. Revisit only with the full
   Chicago corpus, under an amendment, before the verdict run.

PRE-REGISTERED SCORING STRATIFICATION for the blind VLM test, fixed
BEFORE the key is seen: primary metrics stay as registered (4-way
hitter accuracy, overall pace accuracy, fast-vs-slow gap). SECONDARY,
clearly labelled: pace accuracy split by whether the truth row is a
defensive-reach/outcome-coded type versus a plainly-struck one. The
prediction that follows from (3): my pace accuracy should be near
chance on the outcome-coded subclass and materially better elsewhere;
hitter accuracy should NOT vary between them, since who touched the
ball is visible in the frame either way.

## 2026-08-19 — BLIND VLM TEST SCORED (key: data/vision/vlm_test_key_20260819.csv; reproduce with `python3 vision/vlm_score.py <key>`)

Calls locked in-thread before the key was uploaded; predictions
registered before any blind frame was seen. Effectively a 3-player
field — Allyce Jones was never drawn (0/20), a luck-of-the-draw fact,
not a design choice.

RESULT (verifiable halves):
  SIDE / TEAM   **19/20 = 95%** (chance 50%). The one miss is q19,
    rally 20 shot 4 — a post-end-switch frame where a far player's
    big overhead motion pulled the call away from the true near-side
    hitter. Note the failure mode: I was seduced by the LARGEST motion
    in the frame, which is exactly what a magnitude-based pose feature
    does. Same bug, different substrate.
  PACE          **15/20 = 75%**, and the split is the finding:
    truth-slow 9/10 (90%), truth-fast 6/10 (60%), a 30pp gap.
    REGISTERED: "15-25pp worse on fast" — DIRECTION RIGHT, MAGNITUDE
    UNDERESTIMATED (30pp, outside the band).

THE FIVE PACE ERRORS ARE ONE MECHANISM, and it is the label-semantics
finding made visible: FOUR of five are "called slow, truly fast", and
every one is an attack launched from a low or defensive-looking body
position — q18 a speed-up from a dink stance, q05/q09 low counters,
q10 the THIRD-SHOT DRIVE (the modal gap-invisible type from the same
day's blind-spot decomposition). The fifth (q04) is the mirror: an
upward lift read as a smash windup. At the contact instant a speed-up
and a dink are the same picture; what separates them is what the ball
does next. Registered prediction from the semantics entry — "near
chance on the outcome-coded subclass" — holds in spirit: 60% on fast.

WITHIN-TEAM LEFT/RIGHT — DISCRIMINATION ESTABLISHED, ORIENTATION NOT.
The key names hitters; the calls name court positions; no committed
artifact maps between them. But the calls partition players
near-bijectively: Tuionetoa near-left 7/8, Wei far-right 6/7, Nelson
far-left 4/5. On the 10 far-side (Chicago) questions — 5 Nelson,
5 Wei — all 4 of my "far-left" calls landed on Nelson,
hypergeometric **p = 0.024**. So the left/right split carries real
identity information; I was tracking individuals, not guessing.
Whether my "left" is their left cannot be settled from the key: 4-way
accuracy is ~85% if correctly oriented, ~15% if consistently mirrored.
TRAP RECORDED: serve geometry does NOT resolve it — the score-parity
rule fixes the SERVER's court, but MLP teams STACK, so post-serve
positions routinely invert it. Rally 8 (Chicago at 1, odd → server
camera-right) contradicts my read; rally 1 agrees with it. Both
inferences are void. Resolution costs one human glance at one frame.

READING. The hitter question is answered well above the registered
60-80% on its verifiable half (95% side), and the pace question is
answered at 75% with the miss concentrated exactly where the theory
says it must be. This does NOT license VLM labeling at scale: the test
handed me pre-located contacts, so it measured recognition given
placement, and placement is the binder. What it DOES license is the
narrower claim: a VLM reads individual identity out of these frames at
better than chance, which is the channel the pose stack throws away
(pick_hitter featurizes one track of four). Next question, unbuilt and
un-run: give it windows NOT centred on contacts and ask how many
contacts they contain and when. That measures the binder directly.

## 2026-08-19 — ORIENTATION CONFIRMED (4-way 85%) + LOCALIZATION TEST BUILT, PREDICTIONS REGISTERED

User eyeballed the frames and confirmed the position calls are
correctly oriented, settling the one thing the key could not:
**4-way hitter = 17/20 = 85%** (misses q11, q13, q19), against a
REGISTERED 60-80% — falsified in the FLATTERING direction, which gets
the same suspicion as the unflattering ones. Chance is ~33% (three
players actually drawn).

WHAT THE HITTER TEST LICENSES — narrow, and worth stating exactly:
  YES: on this footage a VLM recovers WHO struck a contact, given the
    contact's location, at 95% side / 85% four-way. That is the
    attribution channel the pose stack THROWS AWAY — pick_hitter
    featurizes one track of four and the scorer is a per-candidate
    binary, so "which of these four is swinging" cannot even be
    expressed. Identity is recoverable from these pixels; the failure
    was never that the frames are empty.
  YES: a bounded negative on pace — 60% on truth-fast, with every miss
    an attack from a defensive-looking posture. A VLM must NOT be
    trusted for pace on ambiguous contacts, for the reason already on
    the record: the label is a function of the ball's future.
  NO: labeling at scale. The test handed over pre-located contacts, so
    it measured recognition GIVEN placement. Placement is the binder.
  NO: AI labels as training data. My errors are SYSTEMATIC (fast
    under-call, largest-motion capture), which is the poisoning shape,
    not averaging noise.
  NO: anything about temporal_gate v2. Exploration only; holdout
    untouched.

DIVISION OF LABOUR THIS SUGGESTS (design input, not a decision): the
decoder/temporal model owns WHEN (placement) and pace (needs the
future); a VLM owns WHO (attribution). WHO is also exactly what the
user's track-identity thread is chasing, and what upgrades every
team-level product stat to an individual one. Two threads, one target.

`vision/vlm_localize_sample.py` (built, selftested, NOT yet run) asks
the unanswered question — user proposal: "was there a shot or not,
strip by strip". 3x3 grid of 9 frames at 0.15s (1.2s window), cropped
to the playing area; grid not strip because a 9-frame vertical stack
renders each cell ~30px tall after the ~1568px downscale, while a
square grid keeps players ~80-100px. Windows are drawn uniformly from
[first_contact - 4s, last_contact] and are NOT centred on contacts;
the pre-serve dead time is deliberate, because only 0-contact windows
measure hallucinated shots. Default n=30 draws 7 empty / 18 single /
5 double = 28 true contacts (placement recall se ~9pp).

REGISTERED BEFORE ANY WINDOW IS SEEN:
  - exact contact COUNT right in 50-70% of windows
  - 0-contact windows correctly called empty in 70-90% (dead time
    should look obviously different — players walking, no rally)
  - per-contact PLACEMENT recall at +/-0.5s: 45-65%
  - errors concentrate in the 2-contact windows (fast exchanges),
    same axis as every other failure in this program
READING RULE, fixed now: the decoder's 45.7% match rate is the
reference, but the comparison is NOT fair to the VLM — the decoder
sees a whole rally, knows its span, and carries an alternation prior,
while a 1.2s window has no context and no alternation. So beating
45.7% is strong evidence; failing to is AMBIGUOUS, not a kill.

## 2026-08-19 — PACE IS DOWNSTREAM ARITHMETIC, and frame-counting is AT its information ceiling

User pitch, repeated and under-credited by me until now: "go through
every frame, count the frames between swings — 10 frames = fast, 25 =
slow." I had been hearing this as a restatement of the GAP classifier
(which it is) and missing the consequence (which is the point).

THE CONSEQUENCE I MISSED: if pace is read off the frame count BETWEEN
DETECTED SWINGS, then no instrument ever has to JUDGE pace. The VLM's
measured pace weakness (60% on truth-fast, every miss an attack from a
defensive posture) does not block the pipeline, because pace is not
the VLM's job — placement is, and pace falls out as arithmetic. That
collapses two problems into one and makes localization the whole game.
Corollary, and it is quantitative: placement errors map into pace
errors deterministically — a MISSED swing merges two gaps and reads
SLOW where the truth was fast-fast; a SPURIOUS swing splits a gap and
reads FAST where the truth was slow. So the localization metric IS the
pace metric, transformed. There is no separate pace evaluation to run.

THE USER'S INTUITION IS CALIBRATED (again — cf. the 80% transition
prior guess vs the fitted 0.73). Fitted from the 25-rally archive at
30 fps: threshold **20 frames**, their bracket midpoint 17.5. Gap after
a fast shot 0.65 s = **20 frames**; after a slow shot 1.00 s = **30
frames**. The real separation is TIGHTER than 10-vs-25, and that
tightness is the whole story.

CEILING COMPUTED (new, and it settles a standing question): with
log-normal gaps at mu_fast=ln 0.65 / mu_slow=ln 1.00 and log-sd ~0.35,
the BAYES-OPTIMAL accuracy of any rule that sees only the gap is
**73.2%**. The fitted threshold measures 74.6% LORO. **Frame-counting
is already at its information ceiling** — the distributions simply
overlap, and no smarter gap rule, HMM, or learned tempo model can beat
it. This retroactively justifies the MODEL-ITERATION FREEZE on pace
classifiers (it was preventing knob-turning on an exhausted channel,
which is exactly right) and it bounds the temporal model: better
placement improves pace only by making the gaps CORRECT, never by
classifying them better. Improving pace past ~75% requires a
DIFFERENT channel — position (drive vs dink from the same tempo) or
the ball's post-contact trajectory — not a better tempo rule.

## 2026-08-19 — LOCALIZATION TEST SCORED: 93% recall, 30/30 play-detection — the binder is NOT what we thought

Reproduce: `python3 vision/vlm_loc_score.py data/vision/vlm_loc_key_20260819.csv`

CONTAMINATION DISCLOSED FIRST (my error): the realized
contacts-per-window distribution ({0:7, 1:18, 2:5}) was PRINTED while
sizing the draw for power, so the counts were known before looking.
Should have computed power without printing realized counts. Discount
accordingly; the calls diverge sharply from what was known (37 called
vs 28 real, 14 doubles called vs 5), which argues against anchoring,
and the headline metric (placement recall) is not a count metric.

RESULT vs REGISTERED:
  play / no-play decision   **30/30 = 100%** — all 7 dead-time windows
    called empty, all 23 live windows called live. Registered 70-90%
    on the empty arm; got 100% with zero hallucinated rallies.
  exact contact COUNT       19/30 = 63%  (registered 50-70%) IN BAND
  placement RECALL +/-0.5s  **26/28 = 93%**  (registered 45-65%)
    — FALSIFIED, flatteringly and by a wide margin
  precision                 26/37 = 70%
  median timing error       **0.15 s** = one cell
  tighter tolerances        75% at +/-0.3s, 57% at +/-0.2s
  serves/returns            5/5 recall (the 'position' rows)
  1-contact windows         recall 18/18 — every single contact found

THE COMPARISON THAT MATTERS: the decoded pipeline matches **45.7%** of
labeled contacts. This is **93%** — roughly double — and it was done
UNDER the handicap registered before the run: no rally context, no
alternation prior, no knowledge of the rally span, 1.2 s of video per
judgement, against a decoder that sees a whole rally and decodes
globally. The registered reading rule said beating 45.7% is strong
evidence. It is beaten by 2x.

PRECISION IS A WINDOWING ARTIFACT, NOT A DETECTION FAILURE — and this
was predicted BEFORE scoring, from the calls alone (43% of calls sat
at a window edge). **10 of the 11 false positives are at 0.00 or
>=1.05**: a ball arriving at a player in the last cell, or departing
in the first, cannot be assigned to inside-or-outside the window from
1.2 s of isolated context. The fix is OVERLAPPING windows in a tiling
scan — a contact at one window's edge sits mid-window in its
neighbour — NOT dropping edge calls (that trades to 71% recall / 95%
precision, the wrong direction for a channel whose job is to find
events).

CONSEQUENCE FOR THE PROGRAM. Regime 2 said "placement binds", and the
temporal-model gate exists to fix placement. That routing was correct
about WHERE the problem was and possibly wrong about WHAT SOLVES IT:
placement appears to be recoverable at 93% from short frame grids by a
general vision model, with no training, no labels at inference, and no
ball tracker. This does NOT touch temporal_gate v2 (exploration; the
holdout is untouched; the gate's verdict instrument is unchanged) and
it is n=28 contacts on one match with a disclosed contamination. But
it is the strongest single result in the swing thread, and it makes
one question urgent: what does a VLM-placed contact stream score on
the gate's own stats? That is answerable on TRAIN without burning
anything.

ALSO OBSERVED: w18 was a BROADCAST CLOSE-UP — the camera cut off the
court entirely to a tight shot of a player preparing to serve. Called
empty, correctly (truth 0). Any production scan must detect and skip
cutaways; a court-geometry check is the obvious gate and we already
have homography at 0.06 ft.

## 2026-08-19 — BALL FINDABILITY BY VLM: 27/36 cells (75%), and the slow/fast split is 94% vs 56%

User question after the localization result: "would you be able to
identify the ball in these stills, circle it or say I can't see — I
feel like I'm able to see the ball in most of these." Marked four
localization grids cell by cell (36 frames), positions recorded as
in-cell fractions (`vision/vlm_ball_calls.py`), rendered with circles
and sent for audit. UNVERIFIED — a wrong circle is worse than a
missing one, so these are claims pending the user's check.

  w27 (one slow contact)      9/9
  w01 (one slow contact)      8/9
  w24 (fast + a lunge)        6/9
  w13 (two FAST contacts)     4/9
  TOTAL                       27/36 = 75%
  slow-contact windows 17/18 = 94%  |  fast-contact windows 10/18 = 56%

WHY THIS MATTERS AGAINST THE CLOSED BALL THREAD. The closure rests on
64.1% [59, 69] in-play per-frame findability by a human with a loupe,
against a pre-registered 0.8 kill line. Two things this adds:
  (1) 75% here is not directly comparable — different sample, different
      observer, n=36 vs 306, and the human panel was a scattered draw
      across the match while these are four contiguous 1.2 s windows.
      It is NOT a refutation of 64%.
  (2) But the closure's ARGUMENT was per-frame ("a perfect detector
      returns ~64% of positions with the misses piled at the
      contacts"), and the user's counter-framing is TRAJECTORY: a
      contact is the direction change between two fitted segments and
      needs no ball AT the contact. On contiguous frames that
      distinction is now visible: w27 gives 9 consecutive positions
      over 1.2 s and w01 gives 8 — trivially fittable — while even the
      firefight (w13, 4/9) yields ~2 points per inter-contact segment,
      which is a line. The frames the trajectory framing needs are the
      ones between contacts, and those are exactly the ones that are
      findable.
CAVEAT that keeps this honest: this measures what a VLM can SEE, not
what a detector can FIND, and the two are different instruments; the
auto-label poisoning result stands. Same 720p condensed VOD the thread
was closed on. n=36. And the fast/slow split (94/56) is the same axis
every other failure in this program runs along.

FORWARD (unbuilt, not licensed): the honest next step remains the
ORACLE test specced earlier — label the ball in EVERY frame of ~10
contiguous inter-contact intervals and ask whether trajectory fitting
recovers the known contact times from human-quality positions. That is
decisive in the negative and needs no detector. What is new is that a
VLM could now supply the candidate positions for that test cheaply,
instead of a labelling session.

## 2026-08-19 — BALL CIRCLES AUDITED BY THE USER: 93% precision, conservative recall, and TWO DIAGNOSTIC ERRORS

The user checked every circle. Result: **25 of 27 circles correct
(93%)**, ~8 findable balls marked "can't see" (so recall ~76% of what
is actually there), and human findability on these contiguous in-play
frames ~33/36 (~92%). Per window: w27 perfect 9/9, w24 6/6 circles
right, w13 4/4 right, w01 6/9 with the two errors below.

I DO NOT HALLUCINATE BALLS — the failure mode is the opposite,
under-calling. That flips the reading of the 75% self-report: the
pixels contain MORE than I claimed, not less.

THE TWO WRONG CIRCLES ARE BOTH WORTH KEEPING, because each names a
failure class any ball tracker will hit:
  (1) w01 cell 7 — the ball was OCCLUDED BY THE PLAYER and I marked
      where the TRAJECTORY said it should be. I inferred a position and
      reported it as an observation. This is the dangerous one: a
      Kalman/gating tracker does exactly this by construction, the
      output looks like a clean detection, and nothing downstream can
      tell an interpolated point from a seen one. Any trajectory work
      must carry a seen/inferred flag per point and never fit on
      inferred points.
  (2) w01 cell 9 — circled a SHOE. Light shoe on blue court is the
      classic ball false-positive class and would fool a colour/blob
      detector identically. Argues for motion-differenced candidates
      over appearance alone (which is what the Tennis Vision writeup
      does with 3-frame stacking).

WHAT THIS DOES AND DOES NOT SAY ABOUT THE CLOSED BALL THREAD. ~92%
human findability on contiguous in-play frames vs the closure's 64.1%
[59, 69] per-frame figure is a big gap, and the plausible mechanism is
exactly the user's original argument: CONTIGUITY. Finding a ball in an
isolated frame is a search; finding it in frame k+1 when you saw it in
frame k is a lookup. But the gap is CONFOUNDED and I will not quote it
as a refutation: I selected these four windows, and selected them from
ones whose trajectory I had already traced while scoring localization.
That biases findability up by an unknown amount.
THE UNBIASED TEST IS FREE AND ALREADY IN THE USER'S HANDS: the 30
localization windows were drawn at RANDOM (seeded, dead time included).
Marking ball-visible / not-visible on those 270 cells — or a random 10
windows = 90 cells — gives a contiguous-frame findability estimate
directly comparable to the 64%, with no selection by me.
PROCESS NOTE: the ball thread's frozen re-entry condition is NEW
FOOTAGE. This is not new footage; it is a new FRAMING (contiguous vs
isolated). Re-opening on a reframing is legitimate but needs an
explicit user call and a fresh pre-registration, exactly as the
temporal gate did — not a drift back in on the strength of 36
self-selected frames.

## 2026-08-19 — THE 64% RE-EXAMINED AGAINST ITS OWN DATA: dead time was already excluded, but the number is still too low, for TWO other reasons

User: "my 64% was WAY too low because so many of the frames were bad —
sloppy frames with players walking around, very few were 'the ball
should be here but I don't see it'." Recomputed from
data/vision/ball_labels_chicago0725.csv rather than either memory.

THE STATED REASON DOES NOT HOLD — the correction was already applied.
Findability by seconds-since-window-open:
    0-3 s   52/146 = 36%   <- pre-serve dead time
    3-6 s   36/67  = 54%   <- still the ~6 s log lead
    6-10 s  55/77  = 71%
   10-20 s  62/102 = 61%
   20+  s   12/24  = 50%
  IN-PLAY (>=6 s): 129/203 = **63.5%** [57, 70] — this IS the 64.1% on
  record. The walking-around frames are the 0-6 s bins and they were
  already binned out.

BUT THE NUMBER IS STILL TOO LOW, on two grounds the closure missed:
 (1) THE MIRROR CONTAMINATION. Windows were built as t0 = t1 - duration,
     so the dead-time lead sits at the START — which the binning caught.
     Nothing caught the END: the 20+ s bin runs 50% and 10-20 s runs
     61%, against 71% in the most confidently in-play 6-10 s bin. A
     declining tail is what post-rally dead time looks like. The
     headline pools all of it. Best-confidence in-play is ~71%, not 64%.
 (2) ISOLATION, and this is the big one. All 416 frames were judged
     ALONE, scattered 3-5 s apart. Finding a ball in an isolated frame
     is a SEARCH; finding it in frame k+1 having seen it in frame k is a
     LOOKUP. The contiguous grids ran ~92% (user-audited, n=36, my
     window selection). The user's original argument was exactly this
     and it survives its own data.
NULL RESULT WORTH KEEPING: no stratum effect in play — fast 81/127 =
64%, random 48/76 = 63%. The closure's suspicion that true fast frames
would be worse is NOT visible here.
Unbiased test unchanged and still free: mark visible/not across the 30
RANDOM localization windows.

## 2026-08-19 — SCOPE CALL (user): interpolation is acceptable, and the wanted outputs are "who's next to hit" + "fast/slow", NOT ball position

Two consequences, one of which cuts against the ball thread.
 (a) THE INTERPOLATION OBJECTION LARGELY DISSOLVES, for a precise
     reason. Interpolating BETWEEN two confident observations is
     bounded; EXTRAPOLATING past the last one is not, and the
     dangerous case was always the contact itself — the ball vanishes
     into the player exactly at the discontinuity you are trying to
     detect, so smoothing through it erases the event. But we no
     longer need the ball to find contacts: VLM localization does that
     at 93% recall. With contact times supplied by another channel,
     the ball is only ever asked a BETWEEN-CONTACTS question, anchored
     at both ends. That is the safe kind of interpolation. The
     seen/inferred flag still stands as hygiene.
 (b) HONEST TWIST — the ball got LESS urgent today, not more. For the
     two outputs the user actually named, both are already served:
     "fast/slow" is downstream ARITHMETIC of contact times (and
     frame-counting is at its Bayes ceiling, 73.2% optimal vs 74.6%
     measured, so a ball adds nothing there), and "who's next to hit"
     is the hitter-ID channel measured at 95% side / 85% four-way.
     What a ball uniquely adds is WHERE SHOTS LAND — depth, direction,
     placement quality — which is precisely the thing the user said
     they do not need. Recorded so nobody re-opens the ball thread on
     the strength of an argument whose payload is already delivered.

## 2026-08-20 — "COULD A LOCAL MODEL / A REGULAR CLASSIFIER DO THIS?" — honest decomposition, and a probe

User question, at a natural reflection point. Answered by decomposing
what the VLM actually did rather than by asserting.

WHAT WAS ACTUALLY MAGIC, AND WHAT WAS NOT. Re-reading my own
localization work: **the ball was the primary cue in nearly every
window; posture was corroboration.** Stated at the time and it holds up
("the single most useful cue was, by a wide margin, the ball"). So the
93% recall decomposes into:
  find a small bright blob -> follow it across frames -> its direction
  reverses -> that is a contact
Steps 2-4 are geometry we can already write. Step 1 is a detector. NONE
of that requires a language model. The parts that genuinely used
learned visual priors were secondary: reading a paddle blur, judging
"is this a ready stance or a walk", spotting the broadcast cutaway.

CONSEQUENT ARCHITECTURE for a fully local pipeline, all classical:
  motion-differenced candidates -> trajectory linking -> direction
  change = CONTACT TIME -> nearest player to the ball at contact
  (we already have all four pose tracks) = HITTER -> gap between
  contacts = PACE (already at its Bayes ceiling)
Note what this fixes: pick_hitter currently guesses the hitter from
wrist-velocity within one side and is the weakest link in the stack.
NEAREST-PLAYER-TO-BALL is a far better estimator and it is arithmetic —
but it needs the ball.

SELF-CORRECTION vs yesterday. I wrote that the ball had become
REDUNDANT because the VLM supplies hitter ID at 85% and pace is
arithmetic. That is true GIVEN A VLM IN THE LOOP. For a LOCAL,
VLM-free pipeline the ball is not redundant, it is load-bearing: it is
how contacts are found AND how the hitter is attributed. The ball
thread's value depends on which architecture is chosen, and yesterday's
"less urgent" verdict was scoped to the wrong one.

`vision/ball_candidates.py` (built, selftested, NOT run — needs the
video) probes the ONE step that could fail outright: at each of the 25
USER-VERIFIED ball positions, does a plain 3-frame motion difference
put the ball in the top-K candidates? No training, no weights, no
labels at inference. Deliberately dumb — no colour model, no shape
prior. Motion differencing is also the principled fix for my own worst
false positive: a SHOE travels with its player, the ball does not, so
the class that fooled me is gated out by construction (asserted in the
selftest with a synthetic static-shoe/moving-ball scene).
READING RULE, fixed before the run: high top-25 recall means the ball
IS in the candidate set and everything downstream is geometry; low
recall means the classical route needs a learned detector and the
probe said so for the price of one script.

## 2026-08-20 — CLASSICAL BALL-CANDIDATE PROBE RUN: 76% presence, ranks bimodal, misses STRUCTURED

User ran `ball_candidates.py` (1280x720 source, 25 user-verified
positions). Headline as printed: top-1 36%, top-5 64%, top-25 76%,
median 103 candidates/frame.

**36% TOP-1 IS THE WRONG NUMBER TO READ.** Rank only matters to a
detector that must pick blind. A tracker gates: given the previous
positions it predicts a small region and asks only whether the ball is
in the candidate set NEAR THERE. So the operative figure is PRESENCE
AT ANY RANK = **19/25 = 76%**, and 103 candidates/frame collapses to
~1-3 inside a gate.

SIGNATURE WORTH KEEPING — the ranks are BIMODAL: of the 19 found,
**14 sit at rank <=2**, and the rest of the mass is misses, not
mid-ranks. The dumb detector either sees the ball cleanly or not at
all; it is rarely "confused". That is the profile of an occlusion /
motion-blur limit, not of a weak scoring function — and it means
better SCORING buys little while better VISIBILITY buys a lot.

THE REAL WORRY, and it is specific: per window 5/6, 4/4, **3/6**, 7/9
— and w24's three misses are CONSECUTIVE cells (0.45/0.60/0.75 s).
Under independent 24% misses, the chance that ANY of the four windows
shows a 3-run is ~5.4% (simulated). So the misses are STRUCTURED: the
ball is genuinely undetectable for a stretch, not randomly dropped.
Structured gaps are exactly the case where interpolation stops being
bounded — which is the failure mode already on record from my own w01
cell-7 error. w24 is the fast window; same axis as everything else.

TWO LIMITS OF THE PROBE, both stated before any next step:
 (1) it samples every 0.15 s; a tracker runs at 30 fps and gets 4.5x
     more chances per second. Per-frame recall at native rate is
     UNMEASURED and could be better (more shots at a moving ball) or
     worse (motion blur between sampled instants).
 (2) the detector uses **NO COLOUR MODEL AT ALL** — the ball is bright
     yellow-green on a blue court and nothing in the code knows that.
     This is the largest untried lever and it is ~5 lines.

DECISION — DO NOT TUNE THE DETECTOR ON THESE 25. Two reasons: it
optimises a PROXY (candidate recall) rather than the thing we care
about, and it burns the only hand-verified ball positions we have. The
disciplined move is END-TO-END: build the tracker + direction-change
contact detector and score it on CONTACT TIMES against the 323 labeled
contacts already in the repo, using the same +/-0.5 s match rate that
gives the decoder 45.7% and the VLM 93%. That metric exists, has
independent ground truth, needs NO new labeling, and burns NO holdout.
If the classical pipeline lands near 93% it replaces the VLM locally;
if it lands near 45% it is the decoder again with extra steps; either
way the answer is a number, not an argument.

## 2026-08-20 — CLASSICAL CONTACT DETECTOR BUILT (`vision/ball_track.py`): three design failures on the way, predictions registered

End-to-end classical pipeline, no model and no labels at inference:
stream frames -> motion-difference candidates -> velocity-gated beam
tracker -> flight SEGMENTS -> contact = the JOIN between consecutive
segments -> scored against the 323 labeled contacts at +/-0.5 s, the
same metric that gives the decoded pose pipeline 45.7% and the VLM 93%.

THREE DESIGN FAILURES, each found by a test and each worth keeping:
 (1) GATE TOO TIGHT TO CROSS THE EVENT. v1 sized the association gate
     for smooth flight, so at a reversal the ball sat ~2x speed from
     the constant-velocity prediction and every track died at every
     kink. The tracker could not survive the one event it existed to
     detect.
 (2) EARLY-ENDING TRACKS DISCARDED. track_ball only returned
     hypotheses alive at the FINAL frame, so a segment that ends at a
     contact — i.e. every segment — was thrown away. Fixed by keeping
     the best track ending ANYWHERE.
 (3) LENGTH BEAT QUALITY. Scoring the association gain as a fraction
     of the gate let a 35-frame wandering clutter chain (straightness
     0.16) outscore the real ball AND consume its candidates. Fixed by
     scoring departure from the constant-velocity prediction in
     ABSOLUTE px (ACC_SCALE), plus a tortuosity gate on kept segments.
 RESOLUTION OF (1) vs the tortuosity gate: they were incompatible — a
 track spanning several legs is by construction not straight. The
 physics settles it. A segment IS one flight between contacts; tracks
 END at contacts by design, and the contact is the join. That also
 makes the straightness test meaningful instead of self-defeating.
Discipline kept: a coasted stretch is linear by construction, so a
contact inside one is reported at the gap midpoint and FLAGGED
inferred, never quoted as a precise time (the w01 cell-7 lesson).

SELFTESTS WITH TEETH (synthetic paths with planted kinks, degraded to
the measured conditions): clean 2/2 kinks 0 spurious; 24% dropout 2/2
0 spurious; 5-frame blackout ACROSS a kink — asserted that the
single-track path FAILS there (0/1) and the segment-join path rescues
it (2/2, flagged inferred), so it cannot pass by coincidence the way
the first version of that test did; pure clutter 0 contacts.

KNOWN LIMIT, stated before the run: synthetic clutter is UNIFORM, real
motion-difference clutter is CONCENTRATED ON THE PLAYERS. At real
candidate density (~103/frame over 1280x720 = 1.1e-4/px2) the uniform
synth degrades badly (40-clutter -> 0/2 kinks). Whether reality is
kinder — a ball over empty court has few nearby competitors — or
harsher is exactly what the run measures. Do not tune on the synth.

REGISTERED BEFORE THE RUN: contact recall 35-65% at +/-0.5 s;
precision 30-60% (over-segmentation is the expected failure);
detected contacts outnumbering true ones. Meaningful bars: beating
45.7% means the classical route matches the pose decoder for free and
locally; approaching 93% means it replaces the VLM. Below ~30% means
candidate quality binds and the colour lever (still unpulled) is the
next thing to try.

## 2026-08-20 — ball_track.py ON REAL VIDEO: 0/229. Six real bugs found; then STOPPED, because I was tuning against a clutter model I invented

User ran it: **0% recall, every rally "track 0 seen pts"**. Not a
performance number — a plumbing/behaviour failure, and the signature
(tracks formed, none kept) pointed straight at the tortuosity gate
rejecting everything.

FIRST, THE PROCESS FAILURE THAT CAUSED IT. The selftests exercised the
ALGORITHM on a 60-frame synthetic and never the CONDITIONS: 700-frame
windows, ~103 candidates/frame, and clutter that MOVES SMOOTHLY
(limbs). "Length beats quality" had already been found and fixed once
— at 60 frames it is a mild bug, at 700 frames with dense clutter it
is fatal, because a smooth 300-frame arm chain scores ~200 while a
real 20-frame ball flight scores ~18, wins the greedy extraction, and
MAX_TRACKS ran out before reaching the ball. Same class as the
vision-postmortem lesson: validate the inner layer, assume the outer
one, get an observationally identical failure.

REPRODUCED FIRST, THEN FIXED (a realistic synth — 600 frames, 106
cand/frame, smoothly-moving limb distractors — reproduces 0/22 exactly).
Six bugs, all real, all now covered:
 (1) MAX_TRACKS=10 exhausted by clutter before reaching the ball -> 45.
 (2) No length cap: chains spanned whole rallies and won on score ->
     MAX_SEG_FRAMES=48 (~1.6 s), which is what a flight actually is.
 (3) Straightness applied only AFTER extraction -> moved INSIDE the
     beam, so wanderers die instead of being filtered later.
 (4) **A HARD FLOOR GETS SATURATED.** With the in-beam floor at 0.60
     the beam produced chains at exactly 0.60-0.74 — maximally
     wandering while still legal — which then all failed the 0.78
     final gate. The search farms any gap you leave between a
     constraint and a filter. Floor raised to 0.75.
 (5) GATE_BASE 26 px was enormous next to one frame of ball motion, so
     a SLOW hypothesis had a gate in which any nearby clutter looked
     consistent; the beam grew chains crawling at 1.3-2.2 px/frame ->
     10 px, ACC_SCALE 12 -> 6, plus an in-beam MIN_SPEED floor. A ball
     outruns a limb; that is the cheapest real discriminator.
 (6) Joins used a fixed 220 px cap and the gap MIDPOINT. Both wrong: a
     12-frame gap at 30 px/frame is a legitimate ~360 px of travel, and
     the midpoint mis-timed a contact by 0.17 s. Reach is now physical
     (speed x gap) and the time is the INTERSECTION of the two flight
     lines — where the ball actually turned. Joins are also now
     best-physical-successor rather than adjacent-in-sorted-order.

RESULT ON THE REALISTIC SYNTH: 0% -> **18% recall at 100% precision**
(17/95 over four seeds; nothing false in three of four). Real
improvement, still far from the 45.7% decoder bar, and the honest read
is that candidate ASSOCIATION in dense clutter is the binder, not the
join logic (fixing the pairing changed nothing).

STOPPING HERE ON PURPOSE. Six rounds of fixes have been made against a
synthetic whose clutter model I INVENTED — uniform random blobs plus
hand-rolled limb tracks. Optimising that is optimising my own guess.
The user has the video; a 60-90 s clip covering labeled rallies turns
this from blind tuning into measurement, and is the same move that has
resolved every other ambiguity in this thread. Requested rather than
guessed at further.

## 2026-08-20 — VIDEO CLIP IN HAND: the 0% had a ROOT CAUSE, and it was the frame rate. Rally 1 now 60%.

User supplied a 44 s clip (rallies 1-2, 36 labeled contacts). First
time in this thread the pipeline could be measured rather than guessed
at. Clip stays in scratch, never committed (house rule).

**THE SOURCE IS 60 fps, NOT 30.** `ball_track.py` defaulted to
`--fps 30`, and decode_window passes `-r`, so every run RESAMPLED a
60 fps source to 30. Measured cost, replicating the user's own probe at
the 9 verified ball positions in w27:

    adjacent frames at native 60 fps : ball found **8/9**, ranks 1-9,
                                       median **72** candidates/frame
    resampled to 30 fps              : ball found **5/9**, worse ranks,
                                       median **101** candidates/frame

Both directions hurt at once, and both are obvious in hindsight: bigger
inter-frame motion means MORE of the scene differences (clutter up 40%)
and the ball STREAKS instead of staying a compact blob (recall down).
Fixed by defaulting `--fps 0` = native, detected from the ffmpeg banner.

RESULT ON REAL VIDEO, rally 1 (25 labeled contacts, 1357 frames at
60 fps, median 58 candidates/frame):
    segments kept 33, detected 25, **matched 15/25 = 60% recall at
    60% precision**
0% -> 60%, and **above the decoded pose pipeline's 45.7%** on the same
+/-0.5 s metric. Not yet the VLM's 93%. First honest number for the
classical route.

METHOD NOTE FOR THIS SESSION: no ffmpeg binary exists in the remote
environment, so the local harness decodes with cv2 (5.0.0) — same
candidates() and track_all() as production, different frame source.
Anything measured here must be re-run on the user's machine through
the ffmpeg path before it counts as a production number.

OBSERVATION TO CHASE, not yet acted on: every kept segment is pinned at
the 48-frame cap, which at 60 fps is only 0.8 s — MAX_SEG_FRAMES was
sized in FRAMES for a 30 fps assumption and is now clipping real
flights in half. Constants that mean a physical thing (seconds, px/s)
should not be stored in frames. Same bug class as the fps default.

## 2026-08-20 — CLASSICAL PIPELINE ON REAL VIDEO: 67% recall, above the 45.7% decoder. Parameter selection CLOSED.

Full arc of the day, measured on the user's 44 s clip (rallies 1-2,
36 labeled contacts; clip in scratch, never committed):

    user's run, --fps 30 on a 60 fps source ......  0/229 =  0%
    native 60 fps, everything else unchanged ....  24/36 = 67%
    "physically correct" frame->seconds constants  22/36 = 61%
    segment cap restored to 0.85 s .............   23/36 = 64%
    final, principled constants ................   24/36 = **67%**
                                                   precision 44%
Reference on the identical +/-0.5 s metric: decoded pose pipeline
45.7%, VLM on frame grids 93%.

ROOT CAUSE OF THE 0% WAS THE FRAME RATE, nothing subtler. The source is
60 fps and `--fps 30` made decode_window resample. Replicating the
user's own probe at the 9 verified ball positions: native 60 fps finds
the ball **8/9** at ranks 1-9 with 72 candidates/frame; resampled to 30
it finds **5/9** with 101. Both directions hurt at once — doubling the
inter-frame interval means MORE of the scene differs (clutter +40%) and
the ball STREAKS instead of staying a compact blob. Default is now
native, read off the ffmpeg banner.

A COUNTERINTUITIVE RESULT WORTH KEEPING: converting the frame-based
constants to physical units made things WORSE (67 -> 61). The segment
cap was doing REGULARISATION as well as physics — a short cap forces
the beam to commit to short clean chains instead of growing the long
wandering ones that caused the original failure. 1.6 s is the honest
bound on a flight and it is the wrong value. Not every constant means
only what its name says.

PARAMETER SELECTION IS NOW CLOSED, and this is the important
methodological line. The three configurations scored 24/36, 23/36,
22/36. At n=36 the se is 8 pp and the 95% CIs run 45-82% — they are ONE
number. Continuing to pick between them on this clip is fitting noise,
which is the same proxy-optimisation trap already flagged for the
candidate probe. Final constants are set on PHYSICAL grounds (coast and
join windows are occlusion durations; min speed derives from court
scale at ~40 px/ft; the cap carries a documented regularisation note),
not on the 36-contact score. The measurement that can separate
hypotheses is the user running 19 train rallies = 229 contacts, se ~3pp.

TWO PROCESS FAILURES, both mine, both recorded because they nearly
corrupted conclusions:
 (1) A join patch SILENTLY NO-OPPED (index splice matched nothing). I
     measured "no change" and concluded segment pairing was not the
     bottleneck. That inference was drawn from an unchanged program and
     is withdrawn.
 (2) ball_track.py accumulated TWO definitions of contacts_from_tracks;
     Python uses the last. They turned out identical so no number is
     affected, but I was reasoning about code that might not have been
     running. Stopped patching that file by string-splice; regions are
     now rewritten wholesale and verified by grep.
Precision (44%) is the open weakness — 54 detections for 36 contacts.
Expected: segments end at contacts, so every clutter segment that finds
a plausible successor manufactures one. Mutual-best matching is built
but unmeasured.

## 2026-08-20 — THE CLASSICAL TRACKER IS THE ONE CHANNEL THAT DOESN'T COLLAPSE ON FAST SHOTS

Ran the miss profile on the clip (rallies 1-2, 36 contacts), splitting
the tracker's recall by the PACE of the contact it missed:

    fast   12/18 = 67%
    slow   11/14 = 79%
    other   2/4  = 50%
    TOTAL  25/36 = 69%

**A 12 pp fast/slow gap.** Compare every other channel in this thread:
    VLM pace calls ........ 90% slow vs 60% fast = 30 pp
    ball findable by eye .. 94% dink windows vs 56% firefight = 38 pp
    pose .................. actively INVERTED on fast (arm_cmax 0.265
                            on fast tags vs 0.628 on slow)
Everything appearance-based degrades hard on fast; motion differencing
barely notices. The mechanism is clean and it is the OPPOSITE of the
others: **fast motion produces MORE inter-frame displacement, which is
a STRONGER motion-difference signal.** Speed helps this detector and
hurts every other instrument we have.

WHY THIS MATTERS BEYOND THE NUMBER. The standing worry about
ensembling was CORRELATED FAILURE — every channel in this program
degrades on fast, occluded, cluttered moments, so an ensemble might
inherit the failure rather than average it away. This is direct
evidence against that worry for the ball channel specifically: its
error profile runs almost flat where the others fall off a cliff. That
is real orthogonality, and it is the strongest argument yet that
pose + ball is worth ensembling. It also sharpens the earlier
complementarity argument (the ball vanishes behind the body that pose
sees best) with a second, independent axis.

CONSEQUENCE FOR WHAT THE BALL CHANNEL IS FOR: contact recall is the
WRONG metric for judging its data value. The SEGMENTS are the product
— ball position over time. With the court homography already solved at
0.06 ft, positions convert to SHOT SPEED IN MPH, landing location and
direction. Nothing else in this project can produce any of those, and
per the profile above, this is the only channel that still works on
the fast shots where speed is the interesting number.

## 2026-08-19 — EXTERNAL REFERENCE: "Tennis Vision" (note.com/ai_driven, 3D tennis from broadcast)

Read on user pointer. Single 22 s ATP clip, 1080p60: cross-ratio
court fit (0.21 px), Kalman + chi-squared adaptive gating ball
track (90% coverage through 18-frame occlusions), self-trained
3-frame-stack ball detector, contacts from silhouette proximity +
AUDIO spectral flux (28 impacts, ±0.07 s), alternating hit-bounce
physics constraint, 3D height from parabolic fits between contacts.
Relevance map for THIS project: homography we already have (0.06
ft); their physics-constraint decoding is our alternation-prior
decoder independently reinvented (validating); their offline
smoother framing matches the temporal-model plan. Their AUDIO
success does NOT transfer — tennis = sparse loud impacts in quiet
gaps at 60 fps vs our measured pickleball result (pop counts
uncorrelated with shots, r~0-0.2, audio gate retracted); their
self-training-on-own-detections is exactly the auto-label poisoning
trap we measured (42% kitchen-band vs 14%), acknowledged in their
own limitations (per-clip retraining). STEALABLE IF the new-footage
door ever opens (1080p60 uncondensed): their ball recipe (adaptive
gating + motion stacking) is the right restart point for the closed
ball thread. Changes nothing now — every lean of their pipeline
(loud sparse impacts, high-contrast ball, 2 players) is precisely
where pickleball broadcast is hard mode.

## 2026-08-20 — PROFILE: where the classical tracker's time actually goes (and an 11x candidate speedup)

Prompted by "can I run these free/cheap on a GPU?" — profiled instead
of guessing, because my guess was that `track_all` dominated.

    600 frames @ 60fps (10.0 s of video)
      decode              8.24s   13.7 ms/frame   17%
      candidates         20.03s   33.4 ms/frame   41%
      track_ball  x1      0.45s
      track_all  (x45)   20.40s                   42%   <- 46x one pass
      TOTAL              48.67s = 4.9x realtime
    -> ~3.8 h of CPU per match of rally time (~47 min play)

The guess was HALF right. `track_all` is 42% and is algorithmically
wasteful (it re-runs the whole beam search from scratch, up to 45
times, over the same window, extracting one track per pass) — but
`candidates` costs the same 41%, and that is pure implementation.

Fixed the implementation half, since it is the one that cannot change
any result. The 3-frame motion difference was numpy:

    d = np.minimum(np.abs(cur.astype(np.int16) - prev),
                   np.abs(cur.astype(np.int16) - nxt)).max(axis=2)

cv2's saturating uint8 `absdiff` is exactly |a-b| on uint8, so

    ch = cv2.split(cv2.min(cv2.absdiff(cur, prev), cv2.absdiff(cur, nxt)))
    d  = cv2.max(cv2.max(ch[0], ch[1]), ch[2])

is BYTE-IDENTICAL (asserted on random noise in the selftest, where
saturation and ties are actually exercised) at **2.5 vs 28.0 ms/frame,
11x**. End-to-end on the clip: 24/36 = 67% recall, 24/54 = 44%
precision, per-rally 14/28 and 10/26 — every number identical to the
frozen run, so the parameter selection closed on 2026-08-20 is
undisturbed and the pending 19-rally baseline is unaffected.

The `track_all` half is a BEHAVIOUR change (extracting top-K tracks
from one beam pass is not the same search as K sequential passes), so
it waits until after the 19-rally baseline exists. Worth roughly
another 10-20x if it works out.

ANSWER TO THE GPU QUESTION: none of this pipeline is GPU work. Decode,
motion-differencing and beam search are all CPU/memory-bound. The only
GPU steps in the whole program are pose extraction (already free on
Colab T4, `gpu_runbook.md`) and training a ball detector if that ever
happens (Kaggle: 30 free GPU-hours/week). The cost problem here was
algorithmic, and no hardware would have addressed it.

## 2026-08-20 — SELFTEST REPAIR: two synthetic assertions were single-draw coin flips

`ball_track.py --selftest` was failing at HEAD (pre-existing, verified
by stashing) on `dropout broke kinks: [1.33] vs [0.7, 1.3]`. The cause
was not the tracker: both the dropout and the structured-blackout
cases asserted a CERTAINTY (`hit == len(kinks)`, `near` non-empty) on
ONE seeded draw of a stochastic process. Graded over ten draws
instead, which is what the quantity actually is:

    24% dropout:            kinks 13/20 = 65%, spurious 0
    blackout across a kink: single track 0/10, segment joins 4/10
                            (all flagged inferred), kinks 13/20

So the design claim the blackout test exists to defend — a single
track cannot cross an occluded contact, segment joins can — holds
10/10 in ORDERING (0 vs 4) while being false about half the time as a
per-draw certainty. Assertions now cover the invariants (track always
splits, single track never crosses, joins always flagged inferred,
recall floor 60%) and the rates are printed. The 65% synthetic kink
recall under measured-rate dropout sits right next to the 67% measured
on real video, which is a mild coherence check on the synth.

## 2026-08-20 — VLM SCAN COST: the bill is per IMAGE, not per frame (`vlm_pack.py`)

User: "I like the accuracy, don't like the cost." Repriced from the
geometry we actually shipped, and the arithmetic has a lever in it I
had missed.

Images are downscaled to ~1568 px on the long edge before tokenising
(~w*h/750). The shipped 3x3 grid is 1572 px wide — ALREADY AT THE CAP.
So a 4x4, 5x5 or 6x6 grid of the same crop downscales to the same
1568 px and costs the same **2239 tokens** while covering 1.8x, 2.8x or
4x more video. Packing is nearly free in tokens. It is paid for in
PIXELS PER FRAME, which is a different currency:

     grid  cells   span   cell px   imgs/match   top     mid    small
     3x3      9   1.35s  522x357        2089   $44.35  $8.87   $2.96
     4x4     16   2.40s  392x268        1175   $25.57  $5.11   $1.70
     5x5     25   3.75s  313x214         752   $16.88  $3.38   $1.13
     6x6     36   5.40s  261x178         522   $12.16  $2.43   $0.81

(per 47 min of RALLY time, batch pricing; dead time is never tiled
because rally spans come free from the referee logs.) Two independent
levers, 3.6x from packing and 15x from model tier, 55x combined.

MECHANISM PROBE (`vlm_pack.py --video`, rally 1, TRAIN, rendered at
exactly the delivered pixel size so what I looked at is what a model
would see): **the ball channel and the posture channel do NOT degrade
at the same rate.**

  ball     3x3 clear -> 4x4 workable -> 5x5 degraded but present
           -> 6x6 mostly gone (a ~7 px ball at source is ~2 px)
  posture  fully readable at ALL FOUR — swings, lunges, kitchen
           position and paddle attitude are obvious even at 261x178

This is a look-at-it probe on one window, decisive in the negative
only; it is not an accuracy measurement. But the split it shows lines
up with what the user said the data is actually for ("not even 'where
is the ball' data, but 'who's next to hit' and 'fast/slow' data") —
and that is the channel that survives packing. The expensive channel
and the wanted channel are not the same channel.

HYBRID WORTH TESTING: overlay the free classical candidates
(`ball_candidates.py`, ball present at some rank 76%) as drawn markers
before packing. A drawn circle survives a 0.29x downscale when a 2 px
ball does not, which converts "find a 3 px dot" into "pick which
marked dot moves like a ball" — a task the VLM already demonstrated,
having rejected a shoe and flagged an occlusion unprompted.

REGISTERED, before any of it is run (placement recall at +/-0.5s):
top tier 4x4 85-93%, 5x5 70-85%, 6x6 50-70%; hitter SIDE >=90% at
every level through 6x6; small tier at 3x3 55-75%. Predicted dominant
6x6 failure mode is cell INDEXING (naming cell 23 of 36) showing up as
timing error, not as missed contacts — a different failure than
resolution loss, and separable by whether the miss is off-by-a-cell or
absent.

## 2026-08-20 — PACKING LADDER HARNESS + the marker arm's first surprise

`vlm_localize_sample.py` generalised from the hardcoded 3x3 to `--grid
N`, with `--markers` and `--exclude` (draw FRESH of an earlier draw's
ANSWER_KEY, so a new arm never rescores spent windows). Renderer moved
from ffmpeg to cv2: the marker arm has to draw on the frames, and one
renderer keeps the rungs comparable to each other. Cell width is now
LONG_EDGE//N, so every arm is delivered exactly at the downscale cap.

Also REMOVED the realized contacts-per-window print. That is the leak
the 2026-08-19 run had to disclose — a scorer who knows the count
distribution has a prior on how many shots to call.

MARKER ARM, first attempt WRONG and worth recording: drawing the top-12
raw candidates per cell makes the image WORSE. The candidates cluster
on player limb motion and crowd movement — exactly where the eye
already goes — so the ball's marker is buried among a dozen others and
the cell reads as noise. Switched to the TRACKER's surviving segments:
one mark per frame, 32 of 36 cells marked on the probe window, and the
sequence reads as a ball-like path with a few marks visibly on a paddle
or hip (the 44%-precision false positives, and they LOOK like false
positives — which is the point, since the model is being asked to
adjudicate rather than to trust). This is the version to test.

Sizing, from the TRAIN budget (19 rallies, 229 contacts, 276 s of
drawable video incl. pre-pad) at ~55 contacts per arm (se ~7pp):
3x3 36 windows, 4x4 20, 5x5 13, 6x6 9. Marked 6x6 renders in ~15 s.

## 2026-08-20 — marker arm crashed on a STALE ball_track.py (and a dedup)

`--markers` died with `'float' object cannot be interpreted as an
integer` at `range(max_tracks)`. Not a logic bug: the traceback's line
numbers place track_all ~200 lines earlier than this repo has it, so
the local copy is an OLDER ball_track.py whose signature was
`(cand, max_tracks, fps)`. Calling `track_all(cand, fps)` positionally
put fps=60.0 into max_tracks. Fixed by calling with a KEYWORD, which
is correct under either signature.

Worth flagging beyond this crash: a stale ball_track.py also predates
the native-fps detection, which is the single thing that took the
tracker from 0% to 67%. Any run of the 19-rally baseline on that copy
would have been measuring the wrong code.

Separately, `ball_track.py` carried FIVE duplicated top-level defs
(_seg_vel, _join_time, _seg_speed, _ballistic_ok, track_all) from
botched string-splices — Python used the second copy, so nothing was
ever wrong, but the file had 84 lines of code that could silently
diverge on the next edit. Verified line-for-line identical before
removing the first copies; selftest and the clip run are unchanged
(24/36 recall, 24/54 precision, same per-rally splits).

Exclusion verified by reproducing both draws: arm5 opens on rally 20 @
492.96s, arm6 on rally 10 @ 309.39s, zero overlapping windows, 94s of
the 276s train budget spent.

## 2026-08-20 — MARKER ARM: registered before the grids land

The marker arm had no prediction on record and is about to be run, so
it goes in writing first. Three things established while setting it up.

**The comparison is PAIRED, which I had not noticed when specifying it.**
The draw is a deterministic function of (contacts, n, seed, span,
exclusions), and arm6 and arm6m share all five — so they are the SAME
NINE WINDOWS, plain and marked. Verified by reproducing both draws:
identical, and disjoint from arm5's thirteen. The marker effect is
therefore a within-window contrast, not a difference of two ~55-contact
samples, and the noise on it is far smaller than the ~7pp per-arm se
would imply. arm5 (5x5, plain, 13 windows) stays the independent rung.

**Why the hybrid's ceiling is ABOVE the tracker's 67%.** The marker arm
does not need the tracker to FIND the contact, only to carry the ball
THROUGH it — the VLM re-does the hard part, deciding where the path
bends. That is a weaker requirement than the kink detection the 67% was
measured on, so ball-path coverage is the binding number, not contact
recall. On the probe window coverage was 28/36 cells = 78%.

**The dropout is BURSTY, not scattered** — and that is the worse shape.
The two blank runs on the probe window are frames 87-124 and 216-246,
~0.6 s each, where track_all returns no surviving track at all. A
contact inside a blank run has no neighbouring marks to read a
trajectory from, so the marker channel contributes nothing there and
posture has to carry it alone. Scattered dropout at the same rate would
leave every contact with usable context.

`cut_grid` now returns a per-cell 1/0 string of where the tracker put a
mark, stored raw in the answer key (`marked_cells`). Raw rather than a
derived per-contact flag so the mark-to-contact tolerance stays a
scoring-time choice. It makes a miss ATTRIBUTABLE, which is the whole
point: a contact the tracker had and the reader still missed indicts the
reading; one the tracker never had indicts the tracker; one placed with
no mark nearby was carried by posture alone.

REGISTERED (placement recall at +/-0.5 s, top tier):

  1. arm6m lands **70-85%**, i.e. up a full packing rung from plain 6x6
     (50-70%) to roughly plain 5x5. Marked 6x6 at 5x5 accuracy is
     $12.16 vs $16.88 per match — real but modest. The result that
     would matter is arm6m reaching plain 4x4's 85-93% band, making it
     $12 against $26.
  2. FALSIFIER, hybrid dead in this form: **arm6m <= arm6 + 5pp.**
  3. FALSIFIER, and the one I am most exposed on: **arm6m precision
     BELOW arm6 precision.** The notes claim the tracker's false marks
     "LOOK like false positives" and will be adjudicated rather than
     trusted. That is a claim about a model I have not tested. If
     precision falls, the markers are being believed and the arm is
     worse than useless — it launders a 44%-precision tracker through a
     confident reader.
  4. Miss decomposition, **60/40 tracker-bound over reading-bound**,
     because the dropout is bursty. If it comes back reading-bound the
     next move is the prompt or a rung of model tier; if tracker-bound
     it is tracker work, and the tracker is the cheap half.
  5. Cell INDEXING stays the predicted dominant 6x6 failure on both
     arms — off-by-a-cell timing error rather than absent contacts,
     separable at scoring by whether the miss is displaced or missing.

Unchanged and still pending: the 19-rally classical baseline is the
number everything here is measured against, and it must be run on the
current ball_track.py.

## 2026-08-20 — arm6m SCORED: recall 87.5% and it means NOTHING (the metric saturated)

Marked 6x6, nine windows, 32 true contacts, 75 calls locked in
`vision/calls/arm6m_calls.md` before the key.

  recall     28/32 = 87.5%   (registered 70-85% — "beaten")
  precision  28/75 = 37.3%

BOTH NUMBERS ARE AT CHANCE. Placement null = same number of calls per
window, times randomised, 4,000 draws:

    tol      recall     placement null        verdict
    0.500    87.5%   76.8% [65.6%, 87.5%]   inside (exactly on the bound)
    0.375    84.4%   68.5% [56.2%, 81.2%]   marginal
    0.300    75.0%   61.4% [46.9%, 75.0%]   inside
    0.225    59.4%   51.2% [37.5%, 65.6%]   inside
    0.150    53.1%   38.2% [25.0%, 53.1%]   inside
    precision 37.3% vs null 32.8%

One marginal clear out of six tolerances is what chance gives. Same
verdict under the canonical `vlm_loc_score.match`. **arm6m's placement
is not distinguishable from scattering 75 calls at random.**

WHY, and it is not the markers: TOL is 0.5 s on a 0.15 s grid, so one
call covers +/-3.3 cells. Recall is then bought with call COUNT. I made
2.3 calls per true contact; the 3x3 test made 1.3. The null rose with
me. **This is registered falsifier 3 firing** — not yet on
precision-vs-arm6 (that still needs arm6) but on the behaviour it was
written to catch: I called at MARK density rather than shot cadence.
Tightening 91 -> 75 before scoring was not nearly enough.

THE MARKERS WORKED. Tracker coverage 225/324 cells = 69% overall, but
**94% within +/-0.45 s of a true contact** (30/32) — the coverage is
concentrated exactly where it matters. The marker channel delivered;
the reading and the metric did not.

MISS DECOMPOSITION 2/4 tracker-bound, 2/4 reading-bound against a
registered 60/40. n=4. Uninformative; do not score this as a hit.

HITTER SIDE 16/28 = 57%, best of the three possible team splits (the
key gives names, not sides, so that is an upper bound; chance 50%)
against a registered >=90%. But the pairing it is computed on is
near-random, so the side claim is UNTESTABLE in this arm rather than
falsified — when placement is at chance, everything conditioned on
placement is noise.

COUNT SHORTFALL confirmed and worse than the images suggested: 32
contacts realized vs 53 planned (60%), se 7.1% on a 0.8 recall. The
fixed 4.0 s PRE_PAD at a 5.25 s span is the cause, as called before
scoring.

### The 3x3 headline SURVIVES — but it is 16pp, not 40pp

Re-scored `data/vision/vlm_loc_key_20260819.csv` against the same
placement null (calls-per-window held EXACTLY as called, including the
seven correct empties, so the null gets the play/no-play decision free;
only times randomised):

    tol 0.500   92.9%  vs null 77.3% [64.3%, 89.3%]   CLEARS
    tol 0.300   75.0%  vs null 56.8% [39.3%, 71.4%]   CLEARS

So it is real. But "93% against a registered 45-65%" overstated it by
~2.5x, because **a registered band is a PRIOR, not a chance floor.**
The honest sentence is "93% against a 77% placement null". The PR body
and the 2026-08-19 entry should be read with that correction; the
play/no-play 30/30 result is untouched (a real binary decision with a
real null) and so is the 0.15 s median timing error.

### Consequences

1. **TOL = 0.5 s is the wrong metric for a GRID scanner.** It was
   calibrated for a continuous-stream detector where a false call costs
   something. On a 0.15 s grid it makes recall purchasable with call
   volume. Any future grid arm scores against a PLACEMENT NULL, or with
   the call budget pinned to the true count (forced choice), or at
   +/-1 cell. Preferably the null, always.
2. **Registered bands are not nulls.** Every registration in this file
   that compares a rate to a prior band rather than to a chance floor
   is weaker than it reads. This one cost a session to notice.
3. Running arm6 and arm5 as-specified will not answer the packing
   question — the metric saturates before the packing does. Fix the
   scoring rule first, then rescore all three arms under it.

## 2026-08-20 — the arm6m metric was broken; re-scored against a null

arm6m came back 28/32 = **87.5% at +/-0.5s**, above the registered
70-85% band. The registration would have been graded a HIT. It is not
one, and the scoring rule is why.

Re-scored with `vision/score_localization.py` (written after the fact,
selftested, and committed BEFORE the next arm's grids were opened):

    +/-0.5s   recall 87.5%  precision 37.3%   null 80.2% [68.8, 90.6]
              LIFT +7.3%   -> 85.7th percentile, NOT significant
    +/-1 cell recall 53.1%  precision 22.7%   null 42.2% [28.1, 56.2]
              LIFT +10.9%  -> 89.0th percentile, NOT significant

The null places the SAME 75 calls uniformly at random on distinct
cells. At +/-0.5s each call covers +/-3.3 cells of a 36-cell window, so
75 random calls already recover four fifths of the contacts. The
measurement had no way to distinguish reading the grid from spraying
calls at it, and 87.5% was mostly tolerance and call volume.

Both lifts are POSITIVE and neither clears 95% on 32 contacts. Honest
read: not established, direction favourable, underpowered.

WHAT THE RULE CHANGES GOING FORWARD. Under a call-count-indexed null,
over-calling is self-punishing — every extra call raises the null it is
measured against. Optimal play is FEWER, BETTER calls, which is also
the honest way to read a grid. The old free-call regime rewarded the
opposite. Two tolerances are reported because +/-0.5s is the project's
metric of record (the only number comparable to the pose pipeline's
45.7%, the 3x3 VLM's 93% and the tracker's 67%) but is +/-3.3 cells at
this packing, so a sharp +/-1 cell read sits beside it.

MISS DECOMPOSITION (first use of the marked_cells column): of 4 misses,
3 sat where the tracker had NO mark and 1 where it did — tracker-bound,
consistent with the registered 60/40 call, but 4 events is no evidence.

BURNED: arm6m's key is now seen, so those nine windows can never be
re-called under the fixed rule, and the arm6-vs-arm6m pairing is dead —
one half would be scored blind and the other with the answers known.
arm5 and arm6 are untouched and stay valid; a fresh marked draw
(arm6m2, seed 20260821) restores the pairing against arm6.

GENERAL LESSON, and it is the same shape as the 2026-08-15 measurement-
frame bug: a metric with no null and a detector that works are
observationally identical when the tolerance is loose. Build the null
BEFORE the arm, not after it looks good.

## 2026-08-20 — arm5 and arm6m2 scored: RESOLUTION binds, markers do not rescue 6x6

Calls locked at 69b5e1b before either key was requested.

  arm5    5x5 plain,  13 win, 65 calls, 45 contacts
    +/-0.5s   recall 91.1%  prec 63.1%  unif 76.6%  +14.5% @99.8
              shift 88.5%  +2.6% @66.8
    +/-1 cell recall 53.3%  prec 36.9%  unif 38.7%  +14.6% @98.4
              shift 42.0%  **+11.4% @93.0**

  arm6m2  6x6 marked,  9 win, 45 calls, 30 contacts
    +/-0.5s   recall 83.3%  prec 55.6%  unif 65.8%  +17.5% @99.1
              shift 79.1%  +4.2% @68.4
    +/-1 cell recall 26.7%  prec 17.8%  unif 29.0%  **-2.3% @29.9**
              shift 31.4%  -4.8% @19.7

THE HEADLINE. At the loose tolerance both arms clear the uniform null
and neither clears the shift null — i.e. at +/-0.5s the whole result is
call volume plus knowing the sport has a tempo, on both rungs. The
arms only separate at +/-1 cell, and there they separate hard: arm5
carries +11.4% over its own random-phase calls (93rd percentile, the
only positive shift-lift any arm has produced), while marked 6x6 sits
BELOW its null in both directions. Packed to 36 cells the timing
information is gone, and drawing the ball on does not put it back.

MISS DECOMPOSITION FLIPS THE REGISTERED CALL, and the flip is the
mechanism. Registered 60/40 tracker-bound. Actual: of 5 arm6m2 misses,
**5 were MARKED and 0 unmarked** — and 24 of 25 hits were marked too.
The tracker put the ball within a cell of essentially every contact.
So ball-FINDING is solved on this rung and ball-READING is not: the
binding constraint is cell resolution, not the tracker. That also
explains why markers bought nothing — they answer a question that was
not the one being failed.

SCORECARD against the 2026-08-20 registration:
  1. arm6m 70-85%: FAILED under the fixed metric (no localisation
     signal at 6x6 on either the spent arm or the clean re-run).
  2. falsifier arm6m <= arm6 + 5pp: UNRESOLVABLE, my error — arm6 was
     burned by sharing arm6m's windows.
  3. falsifier marked precision < plain: UNRESOLVABLE, same cause.
  4. 60/40 tracker-bound: WRONG, decisively, 5/5 the other way.
  5. cell INDEXING the dominant 6x6 failure: SUPPORTED. 83.3% -> 26.7%
     between the two tolerances is a pure timing collapse; the
     contacts are being seen and mis-placed, not missed.

CONSEQUENCE FOR THE COST TABLE. The $12.16/match 6x6 rung is dead for
placement — it buys a play/no-play detector and a shot COUNT, not
timestamps. 5x5 at $16.88 is the cheapest rung with any localisation
signal, and even that clears only its rhythm null at 93rd percentile
on 45 contacts. 3x3's measured 93% (2026-08-19, n=28) remains the only
arm that was ever convincing, at $44.35.

TWO SAMPLING ARTEFACTS worth fixing before any further arm:
  (a) ~4 of 9 arm6m2 windows are majority DEAD TIME, because PRE_PAD
      is 4.0 s of a 5.25 s span. Production would tile rally spans
      contiguously and not pay this, so measured recall here is on an
      unrepresentative window mix.
  (b) arm6m2 w06 straddles a BROADCAST CUT: its first 13 cells are a
      tight close-up of a pre-serve routine with no court in frame.
      One window in nine. The cells are still 0.15 s apart, so the
      time mapping holds, but no court-frame contact can be read from
      them.

## 2026-08-20 — the 3x3 "93%" ALSO fails the null. Retroactive correction.

The null was built today, so it had never been applied to the result it
most needed to be applied to. Re-scored the 2026-08-19 localization
test from its own locked CALLS (preserved in vlm_loc_score.py) and key:

  3x3, 30 windows, 28 contacts, 37 calls
    +/-0.5s   recall 92.9%  prec 70.3%  unif 78.6%  +14.3% @98.1
              shift 82.5%  +10.4% **@93.6 — short of 95**
    +/-1 cell recall 46.4%  prec 35.1%  unif 35.3%  +11.1% @86.6
              shift 37.3%  +9.2% @79.7

So the project's headline VLM number is NOT ESTABLISHED. Worse, the
specific claim it carried — "double the decoded pipeline's 45.7%" —
evaporates at the sharp tolerance: 46.4% vs 45.7% is the SAME NUMBER.
The doubling was the +/-0.5s tolerance being 3.3 cells wide on a 9-cell
window, not the model seeing more than the pose decoder.

This is the same failure across all four arms now measured (3x3, 5x5,
6x6 marked, 6x6 marked re-run): every one posts 83-93% at +/-0.5s,
every one lands 27-53% at +/-1 cell, and not one clears its own
random-phase null at 95%. The consistent story is that +/-0.5s cannot
discriminate at these packings and never could — it is 3.3 cells wide,
so it measures "did you call roughly the right region", which rhythm
alone answers.

WHAT THIS DOES NOT OVERTURN. The hitter test (side 19/20, four-way
17/20) is a CLASSIFICATION with a flat 25%/50% chance baseline and is
untouched by any of this. Play/no-play 30/30 likewise. The alternation
decoder's shot COUNTS (161/162) are untouched. What is overturned is
specifically TIMESTAMP PLACEMENT, on every instrument tried.

ANY FUTURE PLACEMENT CLAIM must report +/-1 cell against the shift
null. The +/-0.5s metric of record stays for cross-instrument
comparability but must never again be quoted alone.

## 2026-08-20 — 19-RALLY BASELINE IN: 68% / 39% / median error 0.08s

    CONTACT RECALL (+/-0.5s)   155/229 = 68%
    precision                  155/395 = 39%
    median timing error        0.08s
    inferred (inside a coasted gap)  90 of 395

FIRST READ IS A TRAP AND THE NULL CATCHES IT. 68% looks like a pass
against the pose decoder's 45.7%. It is not: 395 detections over 238s
of searchable span is 1.66/s, and each contact carries a 1.0s-wide
acceptance window, so a UNIFORM null placing the same 395 detections at
random recovers **78.5% [74.2, 83.0]**. Observed 67.7% is BELOW its own
null, at the 0th percentile. On the project's metric of record this
detector is worse than sprinkling.

BUT THE MEDIAN TIMING ERROR SAYS THE OPPOSITE, AND IT IS THE MORE
INFORMATIVE NUMBER. 0.08s is HALF A CELL of the VLM grids and a sixth
of the tolerance it is being scored at. A null's matched detections sit
roughly uniform inside +/-0.5s (median error ~0.25s). This detector's
sit at 0.08s. Those are different distributions and +/-0.5s cannot see
the difference — it is the same tolerance-too-wide failure that
inflated every VLM arm, arriving from the other direction.

Tight-tolerance nulls on the same panel: +/-0.15s -> 39.3%
[33.6, 45.0]; +/-0.08s -> 23.3% [18.3, 28.4]. The median error alone
BOUNDS the detector at >= 77 of 155 matches inside 0.08s, i.e. >= 33.8%
recall at +/-0.08s against a 23.3% null. So the tight-tolerance read is
plausibly positive where the loose one is negative. It needs measuring,
not arguing.

WHY BELOW-NULL AT +/-0.5s IS ITSELF A DIAGNOSIS. Uniform placement
spreads; this detector CLUSTERS. Contacts are emitted at segment joins,
segments survive where tracking survives, and tracking dropout was
already measured as BURSTY (0.6s runs with no surviving track at all,
2026-08-20). A clustered set of 395 covers less timeline than a spread
set of 395, which is exactly how you land under a uniform null while
being individually precise. 90 of 395 sitting inside coasted gaps is
the same story. So the binding constraint is COVERAGE, not precision —
the opposite of the VLM arms, where precision was the constraint and
coverage was fine.

ball_track.py now prints a TOLERANCE SWEEP (0.50/0.30/0.15/0.08) with a
uniform null at each, and takes --dump to write detections+truth so
this never needs re-tracking to re-score. Smoke-tested on a synthetic
detector with sd 0.08s jitter: separates +35.9% over null at +/-0.15s,
which is the shape a genuinely precise detector makes.

## 2026-08-20 — side preference: 14/14 stable, and two corrections

User claim: at the top level a player occupies ONE half of her team's
court for the whole match. That is a much stronger identity constraint
than appearance, and it is exactly what would rescue the ANCHOR-FREE
rallies on the coverage branch, where the serve was never televised and
appearance is the only channel.

MEASURED on the local clip (rallies 1-2, `vision/side_preference.py`):
8 instants in rally 1 + 6 in rally 2, both bands, **zero swaps in 14**.
Far pair holds dark-hair-left / white-visor-right in all 14; near pair
is equally stable wherever both are in frame. Two rallies of one game
is not the claim's test -- the tool tiles one mid-rally frame per rally
across a whole match for a human scan -- but nothing here contradicts
it and the within-rally stability is total.

CORRECTION 1, to this session's own claim. I wrote that in same-gender
doubles "both partners match on silhouette, on kit, and on body type --
every channel is gone at once." FALSE on this footage. The far pair
differ by a WHITE VISOR vs no headwear: stable, high-contrast, and
almost certainly a better feature than the torso-lightness signal that
gave the mixed match 71.1%. The near pair differ in hair colour. The
coverage doc's pessimism about partner separation is a MIXED-MATCH
result and should not be generalised to same-gender without measuring.
Also: surnames are printed on the kit back and are PARTLY legible at
720p near-side (most of "JONES" readable at t=45.5) -- jersey OCR may
be nearer than "specced, not built" implies.

CORRECTION 2, to the PR body. It said alternation plus the referee log
gives "exact per-rally turn order". Alternation is indeed EXACT --
measured 0 consecutive-same-hitter violations and 0 2-colouring
contradictions over 19 rallies / 229 contacts, teams resolving to
Jones+Tuionetoa vs Nelson+Wei with no video and no timestamps. But
teams are a MATCH-level fact recovered by POOLING: 5 of 19 rallies
cannot name all four players alone (three are 2-shot rallies; rally 7
runs nine shots and Jones never touches it). A first selftest here
demanded per-rally resolution, failed at 14/19, and looked like an
alternation failure. It was missing data. Pool, then 2-colour.

## 2026-08-29 — PADDLE TRACKING (idea recovered from user memory)

User recollection: "watching the paddle might be better than watching
wrists." Grep of vision/ + the gate docs on MAIN found nothing and this
entry first claimed the idea was never written down.

**CORRECTION, same day (user: "PR #65 has the paddle stuff"):** it IS
in the record — on PR #65's branch
(`claude/alshon-black-patriquin-bright-model-0kcsrg`, the
touch-attribution thread), as `vision/paddle_probe.py`, dated
2026-08-23. Born from the user watching the touch_attribute review
viewer catch a missed serve. The probe is exactly the cheap
pre-detector sanity check sketched below, already built: sample
labeled contacts across all shot types (dinks included — the pitch is
that paddle POSITION may beat wrist SPEED where the wrist barely
moves), crop 130 px around the labeled hitter's already-tracked wrist
at contact, crop the three NON-hitters at the same instant as a
same-frame baseline, emit a base64-embedded HTML contact sheet.
Bounded search around a trusted keypoint — explicitly not a
re-litigation of whole-frame ball tracking. STATUS: BUILT, NEVER RUN —
needs the user's machine (video + ffmpeg + pose_rtm npz). Lesson for
this file: branch-only work is invisible to main greps; anything
STATUS.md-worthy should land a pointer on main when its PR is opened.
Status of the idea: INSTRUMENT EXISTS, unmeasured, still not scheduled
ahead of the labeling program.

What the record already holds that makes it plausible:

- The feature-crudeness hypothesis (2026-08-19, this file ~line 1294):
  the pose channel may fail on features, not information — "wrist
  velocity in a window" is the crudest possible swing-reader. The
  paddle is the actual contact implement, a rigid ~1.4x lever off the
  wrist, so paddle-tip speed is amplified and paddle direction
  reversal IS the stroke, where wrist velocity only correlates with it.
- The 6x6 VLM probe (this file ~line 2258): "kitchen position and
  paddle attitude are obvious even at 261x178" — the paddle is big
  (~30-40 px at 720p vs ~7 px for the ball) and survives packings that
  erase the ball. The ball-findability wall (36% of in-play frames,
  loupe test) measured the BALL; paddle findability at contact is
  UNMEASURED and is almost certainly far higher.
- The pose stack carries 17 COCO keypoints (pose_extract.py) — no
  hand, no implement. Nothing pretrained points at a paddle; a paddle
  channel means a small custom detector, which means clicks.

Cheapest honest probe, if ever picked up (pre-register a bar first,
per house practice): sample frames at ±3 around already-labeled TRAIN
contacts, user clicks paddle-visible yes/no (+ position when visible)
— measures paddle findability at exactly the moments that killed every
other instrument, reusing the existing archive, no new video watching.
If findability clears where the ball's 64% failed, a detector trained
on targeted paddle clicks becomes a temporal-model feature candidate.

GATE NOTE, load-bearing: temporal_gate v2's MAY list enumerates the
feature stack (pose, kitchen, cadence, track candidates); a paddle
channel is not on it. No temporal code exists yet, so adding one is
legal — but it is a user decision to record as a dated pre-code
amendment/supersession note on the gate BEFORE any temporal code is
written, not something to slide in later.

## 2026-08-29 — PADDLE PROBE RUN (user's Mac, eyeball read of the n=24 contact sheet): SANITY GATE PASSED

User ran PR #65's `paddle_probe.py` (default n=24, unmarked mode) and
scored the sheet by eye. Reported verdict, verbatim substance:

- Paddles visible in essentially ALL crops, hitters and non-hitters.
- ALL BUT 2 hitters have the paddle in frame — and both misses are
  WRONG-WRIST CROPS (the probe centers on the more-confident wrist,
  which was the non-paddle hand; the paddle got clipped, it was not
  invisible). The predicted centering limitation, confirmed at ~2/24;
  fix = crop the box spanning BOTH wrists, unbuilt.
- Hitters' paddles are MOSTLY BLURRY at the contact instant.
- The BALL is not visible in the vast majority of contact-instant
  crops.

READINGS, in order of confidence:

1. **The bounded-search premise holds.** A paddle is findable near an
   already-tracked wrist ~always on this footage — the opposite of the
   ball at the same pre-build stage. A small paddle detector is
   FEASIBLE; whether it is USEFUL is untested and is the next question,
   not a conclusion.
2. **Presence is non-discriminative.** All four players hold paddles
   and all are visible — the channel's signal, if any, is POSITION /
   ATTITUDE / BLUR, never mere detection.
3. **Blur cuts both ways.** It degrades precise paddle-tip kinematics
   at exactly the contact instant (same axis every instrument fails
   along), BUT blur magnitude is itself a stroke-speed cue. CAUTION,
   already on the record: motion blur co-predicted the interval
   histogram's gender split (the "never publish from that stream"
   entry), and the pace LABEL is rally-state (produced gap), not
   stroke speed — a drive blurs like a smash, a defensive reach may
   not blur at all. Blur is a legitimate FEATURE candidate; it is not
   validation for anything.
4. **Ball absence at contacts** reconfirms the ball closure's central
   claim ("misses pile at the contacts") from an independent draw —
   and means the paddle channel cannot lean on ball-paddle proximity.

IF PURSUED (not scheduled ahead of labeling): pre-register a bar
BEFORE building the detector — the candidate product question is
whether paddle position/blur features beat wrist velocity on TRAIN
pace classification. Process constraints already on record: any TIMING
claim from a paddle instrument is scored by score_localization.py
against the shift null like every other instrument; and feeding paddle
features into the temporal model T needs a user-recorded pre-code
amendment (temporal_gate.md MAY-list is enumerated).

## 2026-08-29 — USER OBSERVATION ON THE PROBE PANELS: the 4-crop CONTRAST reads the hitter by itself — posture + paddle blur

User, scrolling the sheet (r1@18.81 speed-up panel as the exhibit):
"Even without the names I can tell you who is doing the hitting —
three players are mostly standing up straight, doing nothing, and the
player who is hitting is leaned over weirdly and has a blurry paddle."
(In that exhibit the ball is ALSO sitting on the hitter's paddle —
second time in this session's screenshots the ball shows up precisely
at a contact crop; anecdotal against the "misses pile at contacts"
expectation, worth counting properly if this goes anywhere.)

WHY THIS IS A DESIGN INSIGHT, not just a nice panel: it recasts the
hitter call as a RELATIVE judgment over four SAME-INSTANT crops
(which-of-4, a contrast) rather than an absolute per-crop detection.
Same lesson as blink-compare vs isolated frames — contrast is the
cheap robustness. And the task it attacks is the measured weak link:

- pick_hitter (wrist-speed-within-side) is on record (2026-08-20) as
  "the weakest link in the stack"; the fix proposed there
  (nearest-player-to-ball) NEEDS THE BALL. Posture+blur contrast does
  not.
- The which-partner 50/50 is the ONLY open call in touch share
  (STATUS.md "Buildable now"); VLM does it at ~89% with cost. A local
  4-crop classifier would do it free.
- The labels already manufacture the training set as a side effect:
  every labeled contact = 1 positive crop + 3 same-instant negatives,
  cut around tracked wrists (paddle_probe's own extraction). 323
  contacts ≈ 1.3k crops today, growing ~64/rally labeled. Nothing new
  to click.

WHAT IT DOES NOT TOUCH: timing. Crops are cut AT a contact time, so
this is attribution GIVEN t — the surviving-channel shape
(classification, flat chance baseline), not a resurrection of
placement. Deployment-time t comes from the decoder's candidates,
which are imprecise; robustness of the posture cue to ±0.3 s crop
error is a required arm of any test.

PRE-REGISTERED CAUTION that must be honored if this is ever built: the
2026-08-19 blind-VLM registration predicted hitter calls 15-25pp WORSE
on fast contacts (blur + posture ambiguity in kitchen exchanges — in a
firefight the previous hitter's follow-through and the current
hitter's swing coexist within 0.4 s, so "the one who's leaned over
with a blurry paddle" can be TWO players). Any bar must be split
fast/slow, TRAIN rallies only, frozen before training.

## 2026-08-29 — USER'S "FANCIEST MODEL" SKETCH: per-frame hitter-belief with contacts read off STATE HANDOFFS — and the labeling-economics answer

User's design, paraphrased: a stateful model walks frames; at each
frame it scores all four players on "looks like they're about to hit"
(posture, paddle blur, ball/tracer evidence), carries the belief
forward, and infers a CONTACT when the hitter-belief hands off across
the net ("player 3 now looks like a hitter and player 1 no longer
does — did player 1 strike?"). Follow-through persists past impact,
so the hitter-look is an EPISODE (wind-up → impact → follow-through),
not an instant.

WHY THIS IS THE RIGHT SHAPE, mapped to the record:
- It IS the registered temporal-model brief (temporal_gate.md: joint
  decode of contact times and pace states + non-tempo channels), with
  a richer emission set — per-player visual state — and the
  alternation prior as the transition structure. "Or better" was
  registered; this qualifies.
- Contacts as STATE TRANSITIONS instead of point events is the escape
  from the trap that killed placement: every dead instrument hunted a
  spike at t against a rhythm-proof null; an episode HANDOFF is wide,
  structured, and side-alternating (exact, 0/229). The event is
  derived from the boundary between episodes, not detected at a
  frame.
- The per-frame observables are the existing assets: pose tracks ARE
  a per-frame posture record (17 kpts, native fps, already
  extracted); paddle-blur/attitude crops are feasible per the
  2026-08-29 probe; ball evidence = the classical tracker's segments
  (median 0.08 s precision, bursty coverage — usable as an EMISSION,
  fatal as the sole channel). No VLM in the loop at decode time.

THE LABELING QUESTION — user proposal: code what every player looks
like FRAME BY FRAME to kill click imprecision and measure exactly when
"looks like hitter" starts/stops. ECONOMICS SAY NO AT SCALE, YES AS A
SMALL CALIBRATION SET:
- Per-frame per-player at 30 fps: a 15 s rally = ~450 frames x 4 =
  ~1,800 judgments vs ~16 taps today — ~100x the cost. The gate needs
  71 more train rallies; dense coding would eat the label budget that
  everything else depends on. Even 2 fps sampling is ~8x.
- The INFORMATION wanted is much cheaper than the labels proposed:
  (a) EPISODE BOUNDARIES: 2 extra taps per contact (swing-look start,
  follow-through end) around the existing impact click ≈ 3x per-rally
  cost, gives onset/offset/duration distributions directly;
  (b) a FRAME-EXACT gold subset: on ~5-10 TRAIN rallies, step
  frame-by-frame and mark the exact impact frame — this measures the
  user's own tap jitter (the "imprecision of when I clicked") as a
  distribution, which the model then absorbs as a known label-noise
  term instead of the labels needing to be perfect;
  (c) between labeled impacts, the sequence model LEARNS the latent
  states from pose+crops under weak supervision — dense human states
  would mostly duplicate what the skeleton already records, and add
  human signal only where pose fails (blur/occlusion), which the
  gold subset samples anyway.
- So: keep the 16-tap habit as the corpus; add the two-boundary
  episode taps only if they stay painless; spend one sitting on the
  frame-exact gold subset. Do NOT convert the habit to dense coding.

GATE HYGIENE (unchanged but now load-bearing): new label TYPES
(episode boundaries, frame-exact impacts) and the visual-state
emission channel need a dated pre-code amendment note on
temporal_gate.md BEFORE any temporal-model code exists — legal today,
locked the moment T is written. The truth definition (attack-onset,
bars, panel) does not move.

## 2026-08-29 — STATE-AUDIT PILOT COMMISSIONED (user go) + draft gate amendment

User answered the economics with revealed preference ("easy and
mindless work"), scoped to the gold subset first. Tool built:
`vision/make_state_audit.py` (selftested; boundaries-not-frames
coding, B/I/E/X per player episode, rally-level R = routine start and
D = point dead, both user additions). Protocol addendum in
labeling_protocol.md same date. Scale-up to a "massive set" is
DEFERRED until the pilot returns minutes-per-rally and the episode
stats prove informative — decision by number, not enthusiasm.

DRAFT AMENDMENT for temporal_gate.md (append only on the user's
explicit freeze, and only while no temporal-model code exists):

  "Amendment (2026-08-29, user-approved): T MAY additionally use, from
  TRAIN rallies only: (a) hitter-episode state labels
  (start/impact/end, no-contact flag) and rally-level routine-start /
  point-dead marks per data/vision/state_labels_chicago0725.csv;
  (b) frame-exact impact times from the same file, including as a
  label-noise calibration for the tap-based contact times; (c) visual
  emission channels computed from the video around tracked-wrist crops
  (paddle position/attitude/blur features). Truth definitions, bars,
  panel, burn rules, and the holdout quarantine are unchanged; C and B
  do not gain these inputs (the ceiling and baseline stay as frozen)."

## 2026-08-30 — STATE-AUDIT PILOT, RALLY 1 SCORED (first four-pass rally; data committed as data/vision/state_labels_chicago0725.csv)

Rally 1, all four passes, scored against the frozen contact labels:

- **Coverage 25/25** real contacts matched to a hitting episode with an
  impact on the correct player. Zero extras, zero misses.
- **THE HEADLINE — the old taps were already frame-accurate.**
  Frame-exact I minus old tap: median +0.001 s, sd 0.029 s, max
  |Δ| 0.074 s (n=25). One frame at 30 fps is 0.033 s, so the existing
  323-contact archive's timestamps are ±1-frame quality. The pilot's
  "measure the tap jitter" job is answered on rally 1: label noise is
  NOT a binding problem, and the 0.25x-speed tapping protocol is
  vindicated. (Caveat: nearest-time matching within hitter can't see a
  systematic whole-shot swap, but 25/25 unique assignment leaves no
  room for one here.)
- **Episode shape**: hitter-look onset median 0.53 s before impact
  [0.23, 1.27]; follow-through median 0.32 s [0.13, 0.50]. Episodes
  ~0.85 s wide — a fat, decodable temporal signature vs the point
  events every dead instrument hunted.
- **Hard negatives are PLENTIFUL: 9 no-contact episodes vs 25 real**
  (26% of hitter-looking episodes end with no strike; Tuionetoa alone
  has 6 — the net player gets faked constantly). The old labels
  carried ONE whiff row for this rally; the state pass sees nine
  hitter-looking non-events. This class exists at scale and was
  invisible until now — exactly the confuser a handoff decoder must
  survive, now measured.
- **Rally marks line up with the log story**: routine_start 6.15 s
  before the serve impact (the known ~6 s log lead, seen from video);
  point_dead +1.12 s after the final smash.
- Hands recorded on all 25 (11 fh / 14 bh). C (clearing) unused in
  rally 1 — either none occurred or the toward-ball convention (X)
  absorbed the candidates; watch over the next rallies.
- One mechanical slip: Jones ep8 (impact 28.933) is missing its E —
  the tool leaves it dashed-open; fixable in-place on her pass.

## 2026-08-30 — RALLY 1 EXPLORATORY STRUCTURE (n=1, shape not significance)

- **The handoff is sequential and tight — "exactly one hitter" nearly
  holds.** Next hitter's episode start minus previous impact: median
  +0.27 s, range [+0.10, +0.61], NEGATIVE 0/24 — nobody starts
  hitter-looking before the opponent has struck. In-rally time with
  exactly one player in a hitting episode: 87% (0 players 6%, two 7%).
  For a handoff decoder this is close to a hard constraint: the state
  sequence is a near-deterministic alternating cycle with 0.1-0.6 s
  inter-episode gaps.
- **Wind-up length IS a pace feature**: pre-impact episode length
  median 0.35 s on fast contacts vs 0.77 s on slow. The episode
  boundaries alone carry pace signal before any pixel is consulted.
- **Alternation perfect** (CUCUC…, 0 same-team consecutive) — matches
  the exact-alternation finding on its 26th rally.
- **Backhand share tracks kitchen role**: Wei 6/7 bh, Tuionetoa 3/4 bh
  (the firefight counters) vs Jones 5fh/3bh, Nelson 4fh/2bh.
  Face-valid — kitchen exchanges are backhand-dominant.
- **No-contact episodes are firefight creatures: 9/9 within 1 s of a
  fast contact, most within 0.3 s.** The hard-negative class lives
  exactly where placement is hardest — as suspected, now measured.
- C (clearing) unused; user hypothesis recorded: clearing is a
  space-negotiation behavior expected in mixed/men's play ("they take
  up more space"), not women's doubles. Testable free on future VODs.
- Convention suggestion for next rallies: close X episodes with E too,
  so the negative class gets durations (rally 1's nc episodes are
  start-only).

## 2026-08-30 — THE ORACLE TEST PASSED. Human ball positions recover contact times on this footage.

Rally 1 ball pass complete: 683/685 frames answered
(data/vision/ball_path_r1.csv), scored by the pre-registered
instrument (make_ball_audit.py, bar frozen before the first click).

**PRIMARY: PASS.** Direction-change events on VISIBLE-only positions:
recall 23/25 = 92.0% at ±0.15 s vs circular-shift null median 64%,
95th percentile 76% — observed at the 100th percentile. Bar was
≥80% AND clear the null; both met with room. Matched-event timing:
median |Δ| ≈ 0.02–0.05 s (≈1 frame). Per the frozen consequence:
**human ball positions are a licensed channel on this footage —
train rallies only, pending the temporal_gate amendment.** The
DETECTOR question stays closed; this licenses labels, not a tracker.

**The two misses are both mechanisms, not noise:**
- The SERVE — direction-change is ill-posed at the trajectory's
  start (no incoming leg). Serves are free from rules/logs anyway.
- Jones's dink @25.23 — sits inside the longest occlusion run
  (25.02–25.36 all inferred, behind the near-court player's body:
  the user's own noted case). The gap rule correctly refused to
  claim it; the event surfaced +0.26 s late when the ball re-emerged.
  The closure's central mechanism in miniature — real, and worth ONE
  impact in 25, not the majority it was feared to be.

**Visibility, dense and selection-free (the numbers the 2026-08-19
thread wanted):** V 86%, S 6%, I 6%, N 2% ⇒ **ball in the pixels 92%
of frames** vs the isolated-frame 64% the closure rests on. Isolation
was load-bearing: finding in frame k+1 after frame k is a lookup.
- Occlusions (I) DO pile at contacts — median 0.10 s from the nearest
  impact, 64% within ±0.15 s vs 33% baseline. The closure's mechanism
  was real; contiguity + trajectory anchoring is what defeats it.
- Smears (S) are drive/serve creatures, exactly as the user called
  while labeling: every smear run follows the serve, return, drops or
  the final falling smash; the 19–22 s kitchen firefight (speed-ups,
  smashes, counters) produced ~one smear frame — net exchanges are
  short-travel, low angular blur. Fast-at-net is EASIER to see than
  a baseline drive, not harder.
- User-reported confuser inventory (recorded for any future
  instrument): balls hide behind the near player's torso and paddles;
  white-on-white kills findability — net tape, shoes, sponsor text.
  The one long torso occlusion cost the one real miss.

Cost: one rally ≈ 685 frames of clicking in a single sitting, called
"pretty straightforward" by the labeler. What it buys, beyond the
verdict: a full 2D ball path anchored to frame-exact contacts — with
the 0.06 ft homography this is the raw material for 3D lift
(piecewise-ballistic fits between contacts), i.e. shot depth, net
clearance, and placement WITHOUT any detector. Scale decision (more
ball rallies vs more state rallies) is the user's; state rallies
remain the gate's binding fuel.

## 2026-08-30 — 3D COURT PROTOTYPE (vision/court3d.py): the rally reconstructs, and the free checks mostly pass

User idea ("depth on a 2D screen is unknowable — what would a 3D
court look like?") built same-day from existing assets only: 11
clicked landmarks (court_landmarks_chicago0725.csv) -> DLT camera;
oracle-passed ball path + frame-exact impacts -> piecewise flight
fits (linear-drag ballistics, zero-centered drag prior, bounce splits
at grid-refined detector events, multi-start GN; V clicks w=1, S=0.5,
I excluded).

- **CAMERA: median reprojection 1.0 px** from the user's clicks;
  recovered center (9.5, 97.6, 26.1) ft — dead-center, 26 ft up,
  54 ft behind the near baseline. Textbook broadcast position, found
  from geometry alone.
- **CHECK 1 (serve, double-bounce rule): PASS** — serve bounce at
  (17.0, 36.2), inside the correct (near-right) service box; return
  bounce (5.4, 4.6), deep far court. The rally's mandatory bounce
  pattern reconstructed without being told the rules.
- **Bounce placements are face-valid throughout**: the third-shot
  drop lands AT the far kitchen line (15.2, 15.1); seven dink bounces
  all fall in/near kitchen depths (y 12-28); the final smash-out's
  floor bounce lands at the near baseline (9.0, 43.5) — the rally
  note says "ended on smash out".
- Net clearances mostly 2.7-6.2 ft; 4 of 18 crossings flag LOW —
  fast-exchange segments have few frames per arc, and one crossing
  sits at x=-2 (outside the sideline: either an around-the-post path
  or sideline noise). Not yet trustworthy per-shot.
- **Honest weak spot: contact heights.** 9 of 22 implausible — arc
  ENDS are where monocular depth is worst (selftest: ~1.3 ft median
  3D error mid-arc even for a clean synthetic; ends worse), and
  impact-adjacent frames were trimmed. Needs adjacent-segment
  reconciliation (end of arc k = start of arc k+1 at the paddle)
  before any contact-height claim.
- Precision context (selftest, synthetic): camera to 1 ft, bounce
  location to ~0.3 ft, mid-arc 3D ~1-1.3 ft at this focal/range/1px
  noise. The service box is 10x15 ft — placement analytics are
  comfortably inside the budget; inch-level claims are not available
  from this geometry and never will be.
- Viewer: `python3 vision/court3d.py --viewer` ->
  data/vision/court3d_r1.html (self-contained canvas, drag orbits,
  space plays). Everything runs from CSVs — no video needed.

User's question, answered for the record: the depth information IS
the converging lines — perspective foreshortening of KNOWN geometry
(20x44 ft spacing) fixes the camera; the net verticals fix the last
degrees of freedom; and between contacts GRAVITY supplies the depth a
static single camera cannot (parallax proper would need camera
motion). The same ambiguity that makes corner kicks unreadable to a
human on a flat screen is why "screen-nearest player" misattributes
hitters — and the 3D lift is the principled fix.

## 2026-08-30 — 3D v2: the user's two priors (net-crossing + shared-contact anchors) fix contact heights

Mid-viewer, the user proposed exactly the right constraints: (1)
"ball crosses the net after contact" — every arc must start on the
hitter's side and end on the next hitter's side (sides VOTED from
pass-1 good fits, not assumed); (2) "it doesn't go past player
geometry" — arc k's end and arc k+1's start are the SAME paddle, so
they anchor each other (consensus contact points, two refit sweeps).
Plus a weak containment prior (ball stays near the court volume) that
kills the depth-degenerate "outer space" arcs the first viewer showed.

Effect, before -> after (rally 1):
- contact heights: 9 implausible in [-30, +15] ft -> **4 barely-out
  in [-1.3, 9.6] ft, median 3.9 ft**. The anchors did it.
- degenerate segments: 7 -> 3 (the rms-9.9 drop, one dink arc, the
  final fall-and-roll, which is not ballistic and never will be).
- serve-box PASS unchanged; net clearances now all in sensible bands,
  with LOW flags only at face-plausible spots (two crossings at
  x ~= 0 skimming the tape — around-the-post-shaped, or sideline
  noise; worth eyeballing in the video someday).
Cost: full pipeline now ~4-5 min on this machine (multi-start pass 1
+ two anchored refit sweeps).

## 2026-08-30 — USER SPOTTED IT IN THE VIEWER: 9/24 spans never cross the net — the residual monocular ambiguity, diagnosed

User: "there were a couple areas where it double hit on the same
side?" Measured on their regenerated viewer data: 9 of 24 inter-impact
spans have y-ranges that never cross y=22 (alternation in the LABELS
is exact, so every one is a reconstruction error). They cluster in the
firefight (20.60-22.90, four consecutive) and the smash exchange
(28.93-30.04) — the short fast arcs.

MECHANISM, and it is the textbook one: along a camera ray from
(9.5, 97.6, 26.1), a contact at the kitchen line at paddle height
projects identically to a contact DEEPER and HIGHER. On long floaty
arcs gravity disambiguates; on 0.3-0.5 s volley arcs (10-15 frames,
several at the net) it cannot, so the solver picked "deep+high" —
Jones smashing from y=44-47 at z~9.7 ft, physically absurd for a
kitchen firefight — and the consensus anchors then made consecutive
arcs CONSISTENTLY wrong together. Ends hug y=22.2-22.9 without
formally crossing: the same-side look.

THE FIX IS THE USER'S OWN SECOND PRIOR, now with data: PLAYER
GEOMETRY. Contacts happen at the hitter's paddle; the hitter's feet
are on the floor, where the homography gives EXACT depth. Pose tracks
for rally 1 exist (Drive pickleball-gate/pose/r0001.npz) and the
state labels' track_assign rows name every track. Anchor each arc
endpoint to within reach (~3 ft) of the labeled hitter's (x,y) at the
impact time, z free in [0.5, 8] — that kills the deep+high branch
outright. Unbuilt; next iteration of court3d.py.

## 2026-08-30 — PLAYERS PLACED IN THE 3D COURT: player-geometry anchors eliminate every same-side artifact

Built on the user's ask ("Can we try to place the players in that
space?"): r0001.npz pulled from Drive, ankles -> ground homography
(feet are ON the floor: exact depth, zero ambiguity) -> committed
data/vision/player_positions_r1.csv (4 players, 20 fps, median-
smoothed; extract_player_positions.py, selftested to cm on the plane
inversion). Identities from the user's track_assign clicks. Face-
valid instantly: Jones/Tuionetoa hover at y~31 (behind the near
kitchen line), Nelson/Wei mirror at the far side, Nelson serves from
10 ft behind the far baseline.

Then the anchors: each arc endpoint hinged to within 3 ft (paddle
reach) of the labeled hitter's floor position at the impact time, z
into [0.3, 8.5]. BEFORE -> AFTER:
- net-crossing spans: 15/24 -> **23/23 drawn segments cross** (CHECK 4,
  new permanent check). The "double hit on the same side" artifact the
  user spotted is fully eliminated — the deep+high monocular branch is
  dead once feet pin the depth.
- contact heights: [-30, +15] -> **[-1.1, 4.9] ft, median 2.0 ft**
  (paddle height!), 3 barely-out.
- degenerates: 3 -> 2 (the rms-10 drop segment and the final
  fall-and-roll; both honest).
- serve box PASS unchanged; clearances 2.6-6.0 ft except two LOW at
  face-plausible spots.
The viewer now draws all four players as team-colored stick markers
with names, moving on their pose tracks. Full chain runs from
committed CSVs alone — no video, no npz needed at render time.

The arc of the evening, for the record: the user's corner-kick
intuition -> 11 calibration clicks -> ball path -> "cross the net" +
"player geometry" priors -> a 3D rally where every constraint that
could be checked against the rules of the sport now passes.

## 2026-08-30 — STATE OF THE PROGRAM after the big evening (roadmap consolidation)

EARNED TODAY, in ledger form:
1. Rally 1 fully state-coded; the user's tap timing proven frame-exact
   (sd 0.029 s) — retroactively upgrading the whole 323-contact archive.
2. The hard-negative class measured: 26% of hitter-looking episodes are
   no-contact, all living in firefights. Episode structure validated
   the handoff-decoder design before it exists (87% exactly-one-hitter,
   +0.27 s sequential handoffs, wind-up length carries pace).
3. ORACLE PASSED (pre-registered): human ball positions recover 92% of
   contacts vs null 76%. Ball-in-pixels 92% of frames dense/unbiased —
   the 64% closure number was an isolated-frame artifact. Human ball
   labeling is a licensed train-only channel.
4. 3D COURT: camera from 11 clicks (1.0 px), rally reconstructed with
   drag ballistics + the user's two priors + player-geometry anchors:
   serve in the correct box, 23/23 spans cross the net, contact heights
   median 2.0 ft. Players placed from pose ankles (exact depth). All
   renders from committed CSVs.

ROADMAP (order matters):
- PRIMARY, unchanged: state rallies 2-8, 13, 14 -> pilot analysis ->
  scale decision -> 90 train rallies -> temporal-gate verdict. State
  labels are the binding fuel; nothing tonight changed that.
- Ball passes: optional, one sitting per rally, each compounds into 3D
  (placement maps, net clearance, contact heights). Publishing-grade
  novel stats once a few rallies exist.
- 3D+players analytics now buildable: distance-to-ball at contact,
  3D-nearest hitter attribution (fixes touch_attribute's weakest
  link), poach geometry DIRECT (who took center balls).
- Pending admin: temporal_gate amendment (state labels, ball
  positions, paddle channels as T inputs) drafted, awaiting the
  user's explicit freeze.
- CONSTRAINED BALL TRACKER (user's late-night idea, logged not
  licensed): 3D players + physics + court collapse the per-frame
  search to a small ray-gated region, and visibility is 92%, not 64%
  — the detection problem's SHAPE has changed since the August kills
  (whole-frame, no dynamics, no 3D). The detector question stays
  CLOSED per tonight's frozen registration; reopening needs an
  explicit user call + fresh pre-registration, third of its kind. If
  called: rally 1's dense human path is free ground truth, the old
  tracker's 0.08 s median timing (STATUS's open question) is the
  encouraging prior, and the prize is ball passes without the human —
  3D at scale. A decision for a fresh day, not a tired one.

## 2026-08-30 — DO HITTERS LOOK DIFFERENT IN POSE? Yes — and the fakes look MORE like hitters than the hitters do

Exploratory read on rally 1 (r0001.npz, 60 fps ViTPose; feature =
p90 torso-normalized wrist speed in ±0.25 s; n=1 rally, not a
registered claim):

- Hitter has the FASTEST wrist of the four players at 15/25 impacts
  (60% vs 25% chance); top-2 at 84% (vs 50%). Real signal, clearly
  insufficient alone — the Gate C verdict in miniature.
- Separation: hitter median 3.87 box-heights/s at impact vs 1.48 for
  the other three AT THE SAME INSTANT vs 2.27 for anyone at random
  times. Note the non-hitters are QUIETER than baseline at contact —
  the ready stance is a measurable freeze, which sharpens the
  exactly-one-mover structure the state labels found.
- **THE NUMBER: fake-swing (X) players at their no-contact episodes
  run median 5.82 — FASTER than real hitters (3.87).** The hard
  negatives are not merely similar to the signal; on the raw
  wrist-speed axis they EXCEED it (desperate reaches move the arm
  fastest of anything on court). This is the cleanest one-number
  explanation yet for why every velocity-threshold instrument died:
  the detector's strongest firings are the non-events. And it is why
  the X labels are gold — no threshold separates this; only
  trajectory/context (did the ball actually arrive and leave?) can,
  which is precisely the temporal model's job description.

## 2026-08-30 — user's top-2 question: hypothesis rejected, but the misses exposed the bias, and the fix ties the paid VLM

User asked whether the top-2 wrists at impact are the same side
(partner-lunge story). NO — 40%. But the miss pattern was the real
find: 9 of 10 out-rankings were by the same player (Jones), and the
random-time baselines explain it — the NEAR pair run 2-3x the
normalized wrist speed of the far pair (3.4-3.8 vs 1.1-1.7
box-heights/s): a systematic near/far tracking bias (far wrists are
few pixels; smoothing flattens their motion), NOT a kinetic-player
fact. Second contaminant: a symmetric ±0.25 s window catches the NEXT
hitter's wind-up (handoffs start +0.27 s after impact — the state
labels predicted this collision).

Fixes, measured (rally 1, n=25, exploratory):
- per-player baseline normalization + asymmetric window (-0.25,+0.10):
  hitter top-1 60% -> 72%, top-2 84% -> 92-96%.
- **WHICH-PARTNER given side (side is FREE from exact alternation):
  hitter beats partner on normalized wrist speed 22/25 = 88%
  (chance 50%) — one crude feature, no learning, $0. The VLM does
  this call at ~89% for money (STATUS "touch share ... costs the 3x3
  rung"). The rung may be free.**

Chain worth noting for method: hypothesis -> rejection -> the
rejection's residue diagnosed two instrument biases -> both fixes
measured -> a product-relevant number. All on committed data, no
video. Next label sittings grow n; a learned combo (episodes as
training data) has obvious headroom past 88%.

## 2026-08-30 — feature battery, body part by body part: REACH beats speed, feet are anti-signal again

Same instrument (per-player normalized, asymmetric window, rally 1,
n=25): hitter top-1 among 4 (chance 25%) / which-partner (chance 50%):

    wrist speed    68% /  88%
    elbow speed    76% /  88%
    shoulder rot   76% /  88%
    hip rot        64% /  76%
    hip speed      72% /  80%
    ankle speed    39% /  67%   <- footwork anti-signal REPLICATES
                                   (feature_check's leg AUC 0.464)
    ARM REACH      92% / 100%   <- p90 wrist-to-shoulder distance
    naive mean     84% /  92%   (dilution by weak features)

THE FINDING (candidate, not claim): **extension, not velocity.** The
hitter is the player whose wrist is FAR FROM THEIR SHOULDER — you
reach toward a ball to hit it. Speed is contaminated by fakes
(measured earlier tonight: fakes are FASTER than hits) and recovery
motion; reach at the contact instant is nearly unique to the hitter.
It is also exactly the user's own visual description from the paddle
probe ("leaned over weirdly") — posture, quantified. And it should be
far more robust to the near/far tracking bias than velocity (it is a
STATIC quantity per frame; no differencing of noisy far-court pixels).

MULTIPLICITY CAVEAT, stated before anyone gets excited: 7 features
were tried and reach selected post hoc on n=25; 100% will not
survive as 100%. The honest next step is free: rallies 2+ arrive as
the user labels, and reach gets scored on them UNTOUCHED — name the
feature now, verify on incoming data. If it holds at ~90%+, the
touch-share product's partner call needs NO paddle, NO VLM, NO money.

PADDLE STATUS after this: demoted from "probably needed" to "optional
later channel" — its remaining unique value is fh/bh confirmation,
pace-blur, and contact-instant refinement, not hitter identity.

## 2026-08-30 — ensemble check (the nightcap): errors are COMPLEMENTARY, and naive pooling dilutes

Are the features making the same errors? NO:
- ZERO impacts are missed by all 5 features (n=25).
- **reach shares ZERO errors with wrist and ZERO with elbow** — the
  posture channel and the speed channel fail on disjoint sets. The
  speed features correlate with each other (wrist+elbow share 5 of
  6-8 misses); reach is the odd one out.
- The disjointness is PHYSICAL, readable off the error matrix:
  speed misses SOFT shots (dinks, drops, counters — low wrist
  velocity) which reach nails; reach misses the serve and a smash —
  the OVERHEAD class, where the arm is bent at contact and extension
  compresses — which speed nails. Two channels, two mechanisms, two
  failure modes.
- Naive ensembles all score BELOW reach alone (mean 80%, weighted
  84%, Borda 88% vs reach 92%): averaging a strong feature with
  correlated weak ones dilutes. The headroom is real (complementary
  errors ⇒ a selective/learned combiner could exceed 92%; ceiling on
  this rally is 100%) but it belongs to a TRAINED rule on the growing
  label corpus, not to pooling. Same lesson as the naive-mean row in
  the battery: combination is a learning problem, not an averaging
  problem.
- (Serves are free from logs regardless — reach's serve miss costs
  nothing in deployment.)
n=25, one rally, exploratory throughout; the named candidates (reach
primary, speed as the overhead specialist) get verified untouched on
rallies 2+.

## 2026-08-30 — head channels (the actual last one): nose speed is a partner-call specialist; dgaze's old suspicion confirmed

Same battery, head features (n=25, exploratory):
    nose speed          top1 60%  partner 96%   <- the keeper
    gaze rot (swap-capped) 56% / 68%            <- weak once the
                          ear-swap artifact is capped — CONFIRMS the
                          record's 2026-08-18 suspicion that dgaze's
                          0.680 AUC was partly swap garbage
    head drop           56% / 60%
    face-blur proxy     40% / 80%  (artifact channel, measured honestly)

Nose speed is mediocre at the 4-way (near/far dilution, like all
speed features) but 24/25 on WHICH-PARTNER — the head rides the lunge
and tracks the ball to contact, and within a side the near/far bias
cancels. Second-best partner feature after reach (100%), failing
independently of it in principle — a natural second witness for the
partner call when reach's perfect score regresses on new rallies.

## 2026-08-30 — the recursive ball↔hitter idea has a name (several), and we already run one iteration of it

User's shower thought: a model that uses the ball to identify the
hitter and the hitter to constrain the ball, "over-steering and
correcting". This is a named, theorem-backed family — recording it so
the concept doesn't get lost before the temporal-gate work starts:

- **Alternating optimization / coordinate ascent**: freeze one
  unknown, solve the other, swap, repeat. **EM** is the probabilistic
  version — E-step: given the current ball track, a BELIEF
  distribution over who hit (soft, not a verdict); M-step: refit the
  track weighted by those beliefs. Guarantee: joint likelihood never
  decreases (the spiral converges inward).
- **Data association** (JPDAF / multi-hypothesis tracking): radar
  literature; "which object owns this blob" solved jointly with
  "where is the object". **Factor graphs / belief propagation**
  generalize: ball nodes, hitter nodes, edges for every coupling
  constraint (contact anchors, exact alternation parity, arm reach),
  message-pass to convergence. Modern SLAM is this; our problem is a
  tiny SLAM (ball = robot, hitters = landmarks).
- **We already run one sweep**: court3d's two-pass fit — pass-1 arcs
  independently, then consensus contact anchors + player-geometry
  anchors feed the hitter answer back into pass-2 arc refits. The
  same-side double hits dying (9/24 → 23/23 crossing) was the
  correction step working. The per-frame hitter-belief state-label
  design is the other half of the loop.
- **Failure modes the house rules already guard**: (i) alternation
  converges to A self-consistent story, not necessarily the RIGHT
  one — internal consistency stops being evidence once the halves may
  agree with each other (same lesson as finding 2's shared-error
  split-half fakes); holdout discipline matters MORE for joint
  models. (ii) Hard assignments oscillate; soft beliefs converge —
  the fractional hitter-belief representation is the mathematically
  correct choice, not just a labeling convenience.
- **User follow-up idea, also named**: run TWO chains, one seeded
  ball-first and one hitter-first, and compare. That's multi-start /
  co-training-style agreement checking: agreement of independently
  seeded chains is evidence of a well-identified optimum (not proof —
  both can share a basin); DISAGREEMENT is the operational gold —
  it flags exactly the rallies/spans where the joint model is
  ambiguous, i.e. where the next human label buys the most. Natural
  active-learning loop for the labeling program: label where the
  chains fight.

Status: concept note only. Any implementation is temporal-gate
territory (pre-registration first); the constrained ball tracker
remains logged-not-licensed.

## 2026-08-30 — VERIFICATION RAN EARLY: the pilot rallies already had timestamped labels + Drive pose, so the named candidates got their untouched test today

The "verify on incoming data" step didn't need to wait for new state
labels: rallies 6-10 have manually timestamped, scorebug-verified
contact labels with hitter names (117-train side of the frozen split;
temporal-gate holdout untouched), and the Gate-C pose npz for all
pilot rallies are on Drive. `vision/verify_hitter_features.py` scored
the named candidates on 84 new contacts. Headline:

    POOLED rallies 6-8 (clean, n=27)   top1-of-4   which-partner
      wrist speed                        74%          85%
      ARM REACH                          63%          74%
      nose speed                         48%          78%
    (chance 25% / 50%; rally-1 exploratory was 68/88, 92/100, -/96)

- **The signal is real** — every feature far above chance on untouched
  contacts — but **the post-hoc ranking did NOT replicate**: reach's
  92/100 regressed exactly as the multiplicity caveat predicted, while
  wrist speed held its rally-1 level almost unchanged (88→85 partner).
  On new data the ORIGINAL feature ≥ the post-hoc winner. Textbook
  selection effect, caught by doing the verification. Both remain
  candidates; the learned-combiner headroom argument stands (these are
  fast rallies — 8-10 shots in ~5-8 s — vs rally 1's 26-shot dink
  battle, so some regression is regime, not just selection).
- **Assignment, not features, is now the bottleneck.** With no
  track_assign clicks for 6-10, tracks were named by a serve-geometry
  cascade (serving side = side whose SHALLOWER member is deeper — the
  receiver often stands DEEPER than the server; receiver via the
  cross-court diagonal rule; decisive-motion fallback). Validated 4/4
  against the user's rally-1 clicks. But within-pair naming stays
  fragile when a pair stacks (r9's serve anchors: wrist 12.0 vs 8.2,
  positions 46.8 vs 46.5 ft — nothing decisive), and r9/r10
  as-assigned scores collapse to ~chance while their up-to-relabel
  partner bounds sit at 86%/82% — good features under a (likely)
  swapped pair name, on the two span-anomaly rallies. FIX IS CHEAP:
  4 track_assign clicks per rally in the state-audit tool ends the
  ambiguity; the labeling ask is ~30 seconds per rally.
- **Forking-paths disclosure**: the assignment cascade was iterated
  while new-rally scores were visible (three rounds: deepest-player
  rule → shallower-member side rule → diagonal/motion cascade). The
  feature DEFINITIONS were frozen from the notes and never touched,
  and the relabel bound is naming-immune, but grade today's numbers
  as instrument-limited verification, not gate-grade. A deliberately
  NOT-taken step: r9 flips to 87/90 under a 1.4 motion threshold
  (measured before the threshold was frozen at 1.5) — left flagged
  UNSURE instead of tuned.
- **This is the recursive theme live**: features are now good enough
  to inform naming, naming determines feature scores — the ball↔hitter
  loop's little sibling (identity↔features). Same rule applies:
  resolve it with independent anchors (user clicks), not by letting
  the halves agree with each other.
- **New fact for the r9/r10 span-anomaly thread**: the v4 windows CSV
  and the contact-labels/npz numbering DISAGREE by ~one rally from
  rally 6 onward (rally-6 taps at 146-151 s sit inside v4's rally-5
  window; npz r0006 covers 144.8-161.8 while v4's rally-6 window is
  164.7-181.7; rally 1 aligns exactly, so the shift starts after 1).
  Taps + npz are mutually consistent and label content is internally
  valid (names typed while watching the video); which numbering
  matches the referee log still needs the scorebug check on video.
  This plausibly explains part of the "label spans exceed log
  durations" anomaly.

Roadmap adds (user, this morning): PICKLES Replay pose flourishes —
per-contact badges from the pose data ("full stretch" when reach hits
a high percentile of the hitter's own baseline, "ran a long way" from
floor-track distance since the previous contact; both computable from
already-committed CSVs + the npz-derived features). Not urgent;
pairs naturally with the impact slow-mo moments.

## 2026-08-30 — DEFINITIVE VERIFICATION: user's 20 track-assign clicks land; the candidates hold on all 84 contacts, and the "span-anomaly" rallies score fine

The user ran the track-assign sprint (state-audit tool, rallies 6-10,
~20 clicks + a bonus pair of rally-6 Jones episodes including a
labeled no-contact BACKHAND FAKE — a free hard negative). Clicks are
merged into state_labels_chicago0725.csv; verify_hitter_features now
uses user clicks whenever present (geometry stays as fallback).
Naming is now ground truth on every scored rally:

    ALL 84 new contacts (truth naming)  top1-of-4    which-partner
      ARM REACH                            76%           82%
      wrist speed                          68%           83%
      nose speed                           46%           76%
    per rally (reach): r6 50/62, r7 60/80, r8 78/78,
                       r9 87/90, r10 78/81   (chance 25/50)

- **The candidates verify.** Reach recovers to co-leader once naming
  is right (the interim "wrist > reach" verdict was partly naming
  noise): reach wins top-1 (76 vs 68), the partner call is a tie
  (82/83). Rally 1's 92/100 stays regressed — the multiplicity
  lesson stands — but ~76/82 on 84 untouched contacts is a real,
  usable signal, and serves are free from logs anyway.
- **REGIME SPLIT, now visible across 6 rallies: wrist wins the short
  fast rallies (6-8: 74/85 vs reach 63/74), reach wins the LONG
  rallies (9-10: 82/86 vs 65/82; rally 1, the 26-shot dink battle,
  was reach's 92/100).** Consistent with the error anatomy from the
  rally-1 battery: speed misses soft shots (dinks), reach misses
  overheads/drives. The learned-combiner headroom is not
  hypothetical — the two features have opposite home regimes.
- **Geometry-assigner scorecard vs truth: 18/24 tids** — r1 4/4,
  r6 4/4, r7 4/4, r8 4/4 (incl. the late-track missing-receiver
  reconstruction), r9 2/4 (the serving-pair coin flip it had flagged
  UNSURE), r10 0/4 (both pairs; also flagged). Every miss was
  pre-flagged, and the naming-immune relabel bounds predicted the
  truth partner scores almost exactly (predicted ≤86/≤82 on r9/r10;
  truth 90/81). The instrument-honesty machinery worked.
- **Span-anomaly update: r9/r10 label CONTENT is fine.** Taps, pose
  tracks, and hitter names are mutually consistent and score at the
  top of the table. Whatever the ~24 s duration mismatch is, it lives
  in the windows-CSV numbering/timing, not in the labels — further
  support for the numbering-shift hypothesis (2026-08-30 entry
  above). The video scorebug check is still owed before r9/r10 lose
  their training asterisk.

## 2026-08-30 — BALL TRACKING REOPENING DRAFTED: vision/ball_gate.md (user call; awaiting explicit freeze)

The user called for ball tracking ("draft away"). `vision/ball_gate.md`
is the pre-registration DRAFT: constrained tracker (weak candidate
stage + court3d arc decoding on labeled anchors) vs the anchors-only
nulls, graded once on a SEALED rally-8 ball pass; train = rallies 6-7
ball passes + the spent rally-1 path. Draft bars: PASS = V-frame hit
rate ≥70% AND null+15; KILL = < null+5; MIDDLE = one iteration + one
re-grade on a fresh sealed rally. I/N frames never scored (the
fine-tune-poison lesson). No tracker code until the user says
"freeze it". Labeling economics note from the user, worth recording:
ball passes are the mentally cheap labels ("easy and relatively
mindless... we can do more whenever we need"), hitter/state passes
cost more attention — so future gate designs should spend ball labels
freely and hitter labels carefully.

## 2026-08-30 — ball_gate v2: the research question is "Can we track the ball?" (user), and it subsumes the missile

Design discussion worth keeping: v1 graded the tracker against
anchors-only interpolation WITH contact taps given — the user asked
whether it should instead "fit the heat-seeking missile idea", which
exposed the misalignment: v1 tested the regime where the ball is
least needed (perfect anchors), while the missile's value is the
opposite regime (the ball supplies the timing Gate C killed). The
user then cut to the actual question: "Can we track the ball?" — well
enough to replicate the 3D model, nothing else needed. That is the
STRICT version: the 3D model's skeleton is the arc structure, arcs
are born at contacts, so replication forces turn recovery — the dead
timestamp channel comes along as a byproduct. And with NO taps given
at grade time the anchors-null dissolves; the only null needed is the
circular-shift one the oracle battery already carries. v2 frozen
shape: three checks on one tap-free graded run (V hit rate ≥70% /
25 px; the unchanged make_ball_audit oracle battery on the decoded
turns; court3d replication with impact agreement ≤3 ft), sealed
rally 8, KILL closes ball tracking on this VOD again. Still DRAFT —
awaiting the user's explicit freeze.

## 2026-08-31 — ball passes 6/7/8 delivered; rally 8 SEALED; and the train rallies caught an instrument-power hole before anything burned

The user labeled all three gate rallies in one sitting (~690 answered
frames). Rally 8's pass is committed SEALED per the frozen gate (its
raw CSV transited the assistant's context at upload — unavoidable —
but no analysis touches it before grading). User labeling notes,
recorded: (1) trailing unanswered frames in r7/r8 are DEAD TIME after
the rally, not misses; (2) the V-vs-S smear boundary is "mostly vibes
but consistent" — fine by design: both classes carry positions and
only the scoring tolerance differs (25 vs 40 px), so V/S
misclassification moves frames between tolerance bands, nothing more.

TRAIN-SIDE FINDINGS (rallies 6-7, open):
- **Answer-key bug found and fixed**: the --rally impacts loader
  counted contact=0 WHIFF taps as required turns. A whiff is a
  no-contact swing — the ball flies straight through and physics says
  no turn exists. Both train rallies had one; excluded now (rally 1's
  frozen key untouched).
- **Whiffs excluded, the human path recovers 100% (7/7) of rally 6's
  physical contacts and 7/9 on rally 7** (the misses: the serve —
  which the gate GIVES the tracker as an input — and one +0.21 s
  borderline in a smear-heavy fast exchange).
- **THE REAL FINDING: the frozen turns battery has ZERO POWER on
  7-9-impact rallies.** r6: observed 100%, shift-null 95th = 100%.
  r7: observed 77.8%, null 95th = 77.8%. With ~14-17 events in a
  5-10 s span and ±0.15 s tolerance, a random rotation matches
  everything ≥5% of the time, and the discrete null's 95th lands on
  the maximum. A PERFECT path cannot pass check 2 as frozen on short
  rallies (rally 1 passed because 25 impacts over ~23 s gave the
  null room to fail). Structural, not a label or tracker fact.
- Visibility on the new rallies (secondary, as pre-declared): V 71-81%,
  V+S 93-98%, N ≤2% — the dense-pass findability replicates.

GATE DECISION NEEDED (user's, the gate is frozen): check 2 as
written is unpassable on the sealed rally class. Proposed amendment
(drafted, not applied): score check 2 HUMAN-MATCHED — the tracker's
decoded path must (a) recall at least as many physical contacts as
the human path does on the same rally under the same battery, and
(b) sit at a null percentile >= the human path's on that rally
(r6 human: 94th pct on a null whose p95 is saturated). True to the
gate's stated intent ("the same test without the human") and usable
on any rally length; pooling check 2 across sealed rallies as that
set grows (r9/r10 have 27-30 taps) restores absolute power later.

## 2026-08-31 — BUILD STARTS: candidate stage first numbers (train only)

Clips r6/r7/r8 delivered via Drive (1280x720@60, windows exact to the
frame; clip clock verified against the ball-pass clicks visually —
crosshair ON the ball at slow moments; fast frames carry ±1-frame
browser-vs-decoder seek jitter, tolerated in diagnostics). Rally 8's
clip is NOT touched (seal). `vision/ball_candidates.py` = the first
gated component: classical motion diff (min of ±0.1 s absdiffs,
threshold 18, 4-600 px components, top-40/frame), no labels, no
tuning yet. Candidate recall vs the user's train passes:

    rally 6: V 81.1% (129/159), S 95.7%   (~40 cands/frame)
    rally 7: V 92.1% (174/189), S 100.0%

Smears — the fast ball, the frames the old isolated-frame test said
were hopeless — are caught at 96-100%: motion diff loves fast movers.
Miss anatomy differs by rally: r7 misses are SLOW frames (median 121
px/s vs 348 for hits — the physics-interpolation-friendly case), r6
misses skew to the serve-hold/player-merged region (~400 px/s, blob
merged with the arm). Both are decoder-bridgeable classes in
principle; that's the next component (arc decoding over candidates,
court3d machinery). Sealed-rally guard is in the script (--graded-run
required for rally 8, scoring against the sealed pass refused).

## 2026-08-31 — ball_gate Amendment 1 lands (user-approved): human-matched check 2 + pooled absolute + owner-reopen clause

User: "go with the amendment, but I reserve the right to unkill
things if I feel you were too harsh." Both encoded in ball_gate.md:
check 2 is human-matched per rally (whiffs excluded, Amendment 1a)
with the pooled absolute test becoming binding once the sealed set
reaches 20 physical contacts; and the owner-reopen clause — closures
bind the process, not the owner, who can reopen with an explicit
call + fresh registration (this gate is itself the precedent).
Grading is now well-defined on rally 8. Next build: the arc decoder.

## 2026-08-31 — arc decoder v1 (train iteration log): 0% → 60-67% V in five moves; the remaining gap is named

`vision/ball_decoder.py` = gated component 2: Viterbi over candidate-
pair states (velocity-aware, capped turn costs so real contacts are
affordable, gap bridging to 0.35 s) + piecewise-quadratic arc refit at
its own discovered turns. Train-only iteration, chronologically:

  1. naive (coverage-greedy)      -> 0% V — locked onto the scorebug
     (median 42 px/s, top-left corner; the cheapest smooth "track" in
     the frame is always junk that never moves).
  2. + slow-edge penalty          -> 52-55% V. Ball median ~330 px/s;
     scorebug/crowd/lazy limbs < 150.
  3. + whole-body box tax         -> HARMFUL, disabled with receipts:
     the TRUE ball sits inside a person box 36-41% of its visible
     frames (near players are huge). Taxing bodies taxes the ball.
     (Wrist-radius variant left as a future knob.)
  4. + court-volume prior (court footprint lifted to 16 ft, projected
     through the one-time calibration) — kills bleacher/crowd flights
     on principle, footage-general.
  5. + gate-panel scoring (first contact -> last+0.5 s — I had been
     charging the decoder for the pre-serve second where a ball in
     hand is invisible to motion diff) + serve-pin decode trim (a
     licensed input): r6 57.4%, r7 67.1% V (bar 70), S 74-75%.
  6. + arc refit: r6 59.8 / r7 64.6 — roughly neutral so far; it
     inherits the Viterbi path's over-segmentation.

Turns (human-matched check 2): tracker RECALL 71-100% vs human 78-86%
— at or above human on r7 — but the null PERCENTILE fails (1-43 vs
95-96) because the tracker path still carries ~14-17 events where the
human has 9-11: spurious excursion-turns saturate its own shift null.
Diagnosis from the hit/miss timeline: the path is LOCKED-ON in long
contiguous stretches; losses are (a) one ~1.2 s mid-rally dropout in
r6 spanning contacts, (b) rally-tail segments. Next moves, in order:
prune spurious events (arc-quality-driven segmentation instead of
one detect_events pass), bridge the dropout with ballistic
extrapolation instead of linear, revisit slow-frame candidates.
The sealed rally remains untouched; no readiness claim yet.

## 2026-08-31 (late) — THE MISSILE FLIES: recursive ball<->hitter loop re-creates rally 1 from scratch; first machine PASS of a gate check in program history

User call: "take a stab at the heat-seeking missile approach, maybe
two tracks, re-create a rally from scratch." Target: rally 1 — spent
for the gate, but TRACKER-UNSEEN (candidates/decoder tuned on 6/7
only) and the richest ground truth in the project. User cut the clip.

Built `vision/hitter_chain.py` (the hitter-first track): pose npz
alone — no labels, no track assigns, 4 longest tracks — per-track
wrist-speed + reach excitement, peak-picked into predicted contacts
with wrist anchor positions. Rally 1: 72% contact recall at ±0.15 s,
anchors a median 39 px from the actual ball at contact. Wired into
the decoder as PRICED HINTS (edge discount + cheap turning near an
anchor).

**THE OVER-STEER LESSON, MEASURED**: with position pull 10.0, bad
anchors (r6, 73 px) GLUED the path to wrists — V 60%→7% — while good
anchors (r7, 39 px) helped everything. Anchors carry TIME
information; position pull cut to 2.0 (turn waiver does the work) and
the collapse vanished. The EM doctrine (soft hints, ball evidence can
veto) is now a measured constant, not a slogan.

RALLY 1 FROM SCRATCH (video + pose + court calib; no labels):
  candidates: V 90.8%, S 100.0%
  ball-only:  V 69.3% (bar 70!), S 100%, turns 92% @ null pct 99
  MISSILE:    V 63.5%, turns 96% recall @ pct 100 — BEATS the human
              path's 92%, and CHECK 2 (human-matched) = **PASS**,
              on a long rally where the null has real power
              (median 68, 95th 84 — no saturation excuse)
  path-vs-path: median 10-11 px from the user's hand path over 548
              V frames (~one ball-width); p75 28-38 px
The contact-time channel, dead since Gate C on every instrument
tried, is recovered from scratch by physics + pixels + pose. DEMO
grade, not gate grade: rally 1 is spent, and the hitter chain's r1
score was seen before the missile ran (no tuning on it, but the eyes
touched it). The gate's one graded run remains sealed rally 8,
readiness rule first. Missile-vs-ball-only tradeoff to reconcile
before readiness: missile wins timing, ball-only wins position —
combine (missile turns to segment, ball-only positions within
segments) as the next iteration.

**B/E marks ("movement starts here" / "swing ends here") — user
question, answered with rally-1 measurements**: 34 episodes, 9 fakes.
Wind-up (B→I) median 534 ms, RANGE 233-1267 ms — the feature
windows we guessed (−0.25 s) are wrong for slow wind-ups and the
spread says windows should be shot-dependent; follow-through (I→E)
median 316 ms. Four uses: (1) per-episode feature windows measured,
not guessed; (2) the 9 fakes now have full TIME COURSES (B/E without
contact) — timed hard negatives for the decoder class nothing else
provides; (3) B/E are literally the state transitions the handoff
model trains on; (4) product stat waiting to exist: PREPARATION TIME
(rushed vs set shots) — measurable per player per shot from labels,
and eventually from pose alone.

Addendum (user diagnostic question, same night): pre-serve ball
clicks (the server's bounces/drops) — candidate recall there is
0-69% vs 94-96% in-rally (r6's 0% is partly the closeup camera
angle). Confirmed blind spot: slow small-amplitude motion against
the server's body is exactly what motion diff can't see. No gate
impact (panel starts at first contact; decode trims at serve-0.3s),
but recorded for any future serve-routine analytics — that region
needs an appearance-based candidate mode, not motion.

## 2026-08-31 (overnight, autonomous) — READINESS: rally 7 passes the full battery

Two decoder bugs found and fixed, the check-3 harness built and run
for the first time, and the readiness rule of ball_gate.md is now
SATISFIED on rally 7. The sealed rally-8 run remains unspent — that
one is the owner's to trigger.

**Bug 1 — the `got_close` early-stop was strangling the graph.**
The edge search quit past 3-frame gaps whenever ANY candidate sat
nearby, so the ball's own 4-6-frame candidate gaps could never be
bridged whenever junk was close — which is precisely at contacts
(the ball passes the players). Priced the true-ball chain through
the decoder's own cost model on r6's 1.2 s dropout: it was CHEAPER
than the limb chain the decoder chose (-761 vs -725), i.e. the
decoder never saw it — five true-chain edges, every one of them the
nearest candidate at its destination frame (rank 0), were pruned by
the early-stop. Fix: always search to GAP_MAX, beam SUCC_MAX=8 near
/ SUCC_FAR=4 far. V hit rates: r6 60.7→71.3, r7 66.5→82.3,
r1 69.3→77.7. (Two plausible-sounding fixes measured FIRST and
rejected: wider uniform beam 50.8-67.7, slow-penalty waiver near
anchors 45.9-61.4 — the waiver licenses exactly the limb-hover the
penalty exists to stop, because anchors sit on wrists.)

**Bug 2 — post-rally events flooded the check-2 null.** The decoder
tracked celebrations to the end of the clip; score_events wraps ALL
events into the rally span under the circular shift, so r6 carried
22 out-of-rally events into its null (median 100 → unwinnable).
The rally-window END is as licensed as the serve pin (referee logs
carry the window) — decode now trims at end+0.3 s, mirroring
serve−0.3. Check 2 after both fixes: r7 PASS (recall 100 vs
human 78, pct 99 vs 95), r1 PASS (96 vs 92, pct 100 vs 100, and it
clears the frozen absolute bars outright), r6 near-miss (recall
tied 86=86, pct 92 vs 96 — a few in-rally wobble events remain).

**Check 3 built and run for the first time**
(`vision/ball_replicate.py`): both sides lifted with court3d's
two-pass machinery under one fully-automated anchor policy — the
hitter chain's anchors carry a pose TRACK id, that track's ankle
midpoint through the z=0 homography of the court calibration gives
the hitter floor position (no names, no clicks). Tracked side
segments itself: turns from the arc-refined path (the same stream
check 2 scores — raw-path turns scrambled the claim), each anchor
claims its LARGEST-ANGLE turn within 0.25 s (real contacts turn
101-171°, wobbles hugging the same anchors 42-66°; nearest-in-time
picked the wobble twice), unclaimed turns stay bounce candidates,
and a crossing-demotion loop drops any claimed contact whose
following segment never crosses the net (every real shot crosses —
court3d's own prior). RALLY 7 PASSES: 7/8 impact points matched at
median 1.70 ft (bar 3.0), net crossings 6/6, bounces 3 vs 2 (±1).
Tried and rejected: promoting unclaimed ≥90° turns to contacts
(rescues r6's missile-missed 148.6 contact but swallows r7's real
bounces and flips it to FAIL — a contact bound requires an anchor,
a missed anchor degrades to a longer segment, never a fake contact).

**Where this leaves the seal.** Readiness is formally met (r7 full
battery; r1 concurs on checks 1-2). The honest asterisk: rally 6
still fails check 3 (its 147.5-149.4 stretch keeps mushy geometry
even at V 71.3 — the missile also missed the 148.6 contact
outright), and sealed rally 8 is rally-6-SHAPED (7 contacts in
5.4 s) rather than rally-7-shaped (9 in 7.2 s). Spending the seal
now is licensed; whether to spend it or first widen train coverage
on short fast rallies is an owner call, not an agent one.

## 2026-08-31 (morning) — Paddle probe, the blur channel, and reverse decoding

Three user ideas measured; two productized, one recorded for later.

**Paddle occlusion premise CONFIRMED**: the ball's invisible (I)
frames sit at median 17-32 px from a wrist, 88-100% within 60 px
(smears, by contrast, 75-126 px away) — the ball only disappears AT
the paddle. Paddle-region reasoning targets real occlusion.

**Paddle point = arm extension, no detector needed.** wrist +
0.5*(wrist-elbow) lands 13.5-15 px from the contact ball click
(occluded contacts 12.6-13.7) vs the bare wrist's 18-20. Now dumped
as paddle_x/y NEXT TO wrist_x/y in the anchors CSV. The decoder's
anchor flags deliberately STAY on the wrist: feeding it the paddle
point lost V (r6 71.3→69.7, r1 77.7→75.2 — the 60 px wrist region
also covers the incoming flight), and the wrist∪paddle union merely
tied the wrist. Paddle point is for contact-POSITION consumers
(3D priors, replay), not decoder flags. A color/blob paddle detector
was probed and is NOT needed: motion-blob centroids land 40 px out
(the smear is paddle+forearm, centroid drags to the arm).

**Blur channel (user idea) — weak alone, gold as gap-fill.**
Per-track motion-diff mass near the wrist, z-scored and peak-picked:
standalone recall 24%/57%/22% (r1/r6/r7) vs the pose channel's
72/57/67 — it only fires on hard swings, useless in dink exchanges.
But it is COMPLEMENTARY: on r6 it caught the 148.57 smash the pose
channel missed outright — the missing contact that broke check 3's
segmentation. Productized as `hitter_chain.py --clip --offset`:
blur peaks ≥0.35 s from every pose anchor are appended (gap-fill
only). Measured safe: decoder V/turns identical everywhere, r7's
check-3 PASS byte-identical, r6 missile recall 57→71% and its
148.57 bound recovered. r6's check 3 still FAILS after the fix —
the bottleneck moved from segmentation to fit-evidence quality in
the dropout stretch (only 1-2 of its segments fit at rms<8).

**Reverse decoding (user idea: "trace the point backwards from the
end") — works as a confidence map, not as a second answer.** The
Viterbi is a global optimum, so the time-mirrored decode mostly
reproduces the forward path; where tie-breaks flip is exactly where
evidence is ambiguous. Measured: agreeing frames (≤10 px) have
median error 4-7 px vs clicks; disagreeing frames 32-34 px. Now a
diagnostic flag (`ball_decoder.py --reverse-check`): r6 67%
confirmed, r7 75% — the number ranks the rallies' shakiness
correctly. TRIED AND REVERTED: weighting the 3D fit evidence by
agreement (w=0.25 on disagreeing points) — it nudged r6 up
(2→3 matched impacts) but PERTURBED r7's segment fits enough to
flip its check-3 PASS to FAIL (demotion loop cascades off changed
fits). The confidence map stays out of the graded path until a
formulation survives both rallies; candidate future use: replay
display, and per-segment (not per-point) evidence selection.

State of record after this session: r7 full battery PASS with the
productized anchor pipeline (blur gap-fill on, wrist flags, paddle
column present); readiness stands. r6: V 71.3 PASS, check 2 pct 92
vs 96 near-miss, check 3 FAIL (evidence-bound). Next lever for r6
is decode quality in 147.5-149.4, not more channels.

## 2026-08-31 (midday) — THE SEAL IS SPENT: rally 8 graded, FAIL, DECODER attributed

Owner authorized the graded run. Everything by the book: grade-time
configuration recorded in ball_gate.md BEFORE the run (serve pin
189.2 from the v4 windows row mapped BY TIME — the CSV's t1s turned
out to be a shot-count formula, not a log timestamp, so the
point_dead mark stands in for the log-carried rally end on the same
licensing logic as the serve pin); harness `vision/ball_grade.py`
dry-run on rally 7 in the identical configuration (PASS end-to-end)
before touching the seal; one run.

Result: **FAIL by check 2's null clause** — V 58.1% (between the
bars), turns 85.7% vs the human path's 100% and under its own
shift-null, and **CHECK 3 PASSED** (median impact agreement 2.82 ft,
crossings 3/3, bounces 3v2). The deliverable — the 3D replication —
survived out of sample while path fidelity failed. Pre-stated
forecast (MIDDLE modal, check 3 underdog) wrong in both directions.

Autopsy (same day, arms in the pre-registered order): oracle
substitution V 94.6% / null-beating turns (decoder sound on clean
input); person ablation — the automated 4-longest tracks ARE the
user's four assigned tracks, curated anchors reproduce V 58.1%
exactly (person exonerated); frame checks clean (candidate recall
84.5% at +/-1 frame IS the clock check); candidate stage 84.5%/100%
on the sealed rally (arm 4's "cannot find the ball" unmet).
**Attribution: DECODER, association under clutter** — misses cluster
in 5 runs during the fast exchange WITH the true ball in the
candidate stream (7/7, 12/13), path parked on limb chains 39-161 px
away. The identical failure class as train rally 6, which is rally
8's shape. The next decoder work item was already named before the
grade: selection among junk at contacts in short fast rallies.

Rally 8 is spent -> train data (a third short-rally training case,
exactly the regime the decoder fails in). Re-attempt licensed after
a decoder fix + dated addendum, graded on a NEWLY labeled sealed
rally — one fresh ball pass when the owner is ready.

## 2026-08-31 (afternoon) — first full recursive-missile experiments on spent r8

Rally 8 is train data now; first closed-loop (EM) experiments, all
label-scored on r6/r7/r8. Round-0 = graded config. Feedback = round-0
turns (angle>=60) as extra time anchors at the turn location, plus a
turn-cost hardening factor away from anchors.

| config | r8 V / turns@pct | r6 | r7 |
|---|---|---|---|
| round0 | 58.1 / 85.7@85 | 71.3 / 85.7@88 | 82.3 / 100@99 |
| feedback, hard=1.5 | 60.5 / 85.7@78 | 63.9 / 57.1@51 | 84.2 / 77.8@95 |
| feedback, hard=2.5 | 44.2 / **100@99** | 55.7 / 71.4@78 | 79.1 / 88.9@98 |
| targeted hard (fwd/rev disagree regions only), 2.5 | 54.3 / 57.1@44 | 65.6 / 42.9@49 | 81.6 / 88.9@99 |

Findings, in order of importance:
1. **The coupling has a real dial**: r8 feedback+hard=2.5 produces
   turns 100% at pct 99 — it would pass check 2's human-matched bar
   on the rally that just failed it. But V collapses to 44 and r6
   degrades: no configuration found yet that wins one check without
   paying elsewhere. Over-steer instability, as pre-worried.
2. **Targeted hardening (only in fwd/rev disagree regions) is worse
   everywhere** — contacts and junk CO-LOCATE. The ambiguous regions
   are ambiguous because the ball is passing a player, which is also
   where it turns; hardening there suppresses the real contacts too.
   Reverse-agreement marks WHERE the path is unreliable, not where
   it should be smoother.
3. Candidate future shape: a TWO-REGIME decode — one tuning for the
   contact/timing stream (hard smoothing + turn anchors), one for
   the position stream (production config), consumed by check 2 and
   check 1 respectively, as the 3D stage already conceptually splits
   turns (segmentation) from positions (evidence). Whether scoring
   two streams under one gate is within the registration's spirit is
   an OWNER call — it would go in the DECODER-fix addendum
   explicitly, bars unchanged, before any re-grade.

## 2026-08-31 (afternoon) — PICKLES Replay: measured shot categories + pressure meter

User direction: make the 3D replay as cool as possible, then hand to
Claude Design for styling. Three things shipped to the artifact
(same URL, version "pressure-meter"):

1. **Measured shot categories** replace the hand labels on the shot
   card (`vision/shot_categories.py`, rules frozen in its
   docstring). User taxonomy: dink / speed-up / hand battle. Speeds
   from the court3d path (median 0.05-0.30 s after contact); rally-1
   distribution has a clean gap (dinks 13-21 mph, attacks 25-48);
   FAST >= 24, speed-up needs HOT >= 30 or a fast follow-up (the
   hard-dinking guard — shot 5's lone 28.6 mph push stays a dink,
   matching the user's own label), battle hysteresis absorbs single
   soft blocks. Derived categories reproduce the hand-label
   structure EXACTLY on rally 1, including both labeled speed-ups.

2. **"Who's winning?" pressure bar** — user request, with their
   threshold caveat honored: ONLY speed-up/hand-battle shots move
   it, weighted (mph-20), 2 s exponential decay. Dink rallies read
   "all square"; the first battle reads Utah 79%, the last one a
   contested 51/49 before Wei's miss; at point-dead it flips to
   "Point: Utah" (receipt: side-out on Nelson's serve). The receipt
   line on the plate says explicitly it is heat, not a calibrated
   win probability.

3. Category-colored shot card + timeline ticks (the rally's story
   reads at a glance: yellow dink stretches, red battle clusters)
   and a speed-tinted ball trail (gold -> battle red above ~20 mph).

Verified headless (Playwright + the preinstalled Chromium): all
meter states, categories, and the ending render correctly; no
console errors beyond the sandbox's blocked font fetch.

## 2026-08-31 (evening) — DECODER FIX BUILT: two-regime decode, constants frozen off a 30-config train sweep

Owner approved the two-regime split same day ("I'm good for
two-regime split. Let's go ahead there"); dated addendum in
ball_gate.md. Build: `timing_decode()` in ball_decoder.py — position
stream untouched (checks 1/3), timing stream re-decodes with the
position stream's own >=60-deg turns (forward AND reverse decodes,
union, accumulated) as turn anchors plus TIMING_HARD=2.5 turn-cost
hardening away from anchors (check 2). ball_grade.py wired to score
check 2 on the timing stream; hitter-chain anchors union in at grade
time. Convention fix along the way: score_train's check-2 human bar
was panel-filtered; ball_grade (and the graded run) score the full
label span — score_train now matches (human bars r6 100@94.5, r7
77.8@91.4, r8 100@98.5).

Sweep results (3 scratch grids, ~2.5 h compute, train r6/7/8, NO
pose anchors anywhere):
- Grid 1, hard x angle x rounds (18 configs): r6 and r8 want OPPOSITE
  hardening — r6 collapses soft (71.4@60 at hard=1.5), passes hard
  (100@95-96 at 2.5-4.0); r8 is the mirror (100@96-98 at 1.5, decays
  to 71.4 at 4.0/80). No config 3/3.
- Grid 2: ONE feedback round beats two (grid 1 ran two: r8 85.7 ->
  100@98.2 on dropping the second — over-steer, as pre-worried).
  Hardening LADDERS (soft discovery round seeds anchors for a strict
  round) fix r8 (100@98.7-99.2 PASS) but break r6: the soft round's
  junk turns inherit anchor protection. Fwd/rev-agreement ADAPTIVE
  hardness selection picks wrong on r8 (agreement 0.73 at hard=4.0 vs
  0.56 at 2.5, but 2.5 scores better) — agreement is not a quality
  signal ACROSS hardness values. Refuted.
- Grid 3: INTERSECTION anchors (turn must appear in fwd AND rev)
  are a disaster — r8 14.3@0.8. Fwd/rev disagree exactly where the
  ball passes a player, which is where it turns: the "contacts and
  junk co-locate" geometry, third independent confirmation.

FROZEN: one round, hard=2.5, angle=60, union anchors. Train: r6
100@96.3 PASS, r7 77.8@98.4 PASS, r8 100@98.2 vs bar 100@98.5 —
recall at the human's level on the rally class that failed the
grade; the pct gap is 3 draws of a 1000-draw null, in a config
strictly weaker than grade (no hitter anchors). Dead ends recorded
in the module docstring so they don't get re-tried.

Rally 9's ball pass arrived mid-session (885 frames, 66V/22S/6I/6N)
and is committed UNPEEKED — no tracker run, no human-path score —
until the owner designates seal vs train. It contains the project's
first off-frame lob: 34 consecutive N frames (~1.1 s, ball above the
frame, x moved 330 px while invisible), 3x the decoder's GAP_MAX —
a regime no train rally exhibits and the current decoder cannot
bridge (it would junk-chain or drop half the rally). Recommendation
on record: r9 -> train (it teaches exactly this; also carries the
log-span asterisk), seal r10/17/20. An off-frame-excursion edge
(exit near frame top moving up, ballistic re-entry from the top) is
the named next decoder capability, to build ONLY if r9 lands train.

## 2026-08-31 (late evening) — angle screen closes the sweep: 40 deg, and the train goes 3/3

The rally-1 screen exposed the frozen 60-deg anchor gate's cost on
the long-rally class (timing 84@99.3 vs human 92@99.5 — hardening
taxes real dink turns under 60 deg), which mattered because r10, the
recommended sealed rally, is rally-1-shaped. Final screen, single
round / hard=2.5, panel = r1+r6+r7+r8:

    ang=60 | r1 84.0@99.3  | r6 100@96.3* | r7 77.8@98.4* | r8 100@98.2  | 2/4
    ang=40 | r1 88.0@99.3  | r6 100@96.3* | r7 77.8@98.4* | r8 100@98.5* | 3/4
    ang=30 | r1 84.0@99.3  | r6 100@96.3* | r7 88.9@99.1* | r8 100@98.5* | 3/4

FROZEN FINAL: one round, hard=2.5, TIMING_ANGLE=40. The human-matched
check passes on ALL THREE train rallies with zero pose anchors —
including r8, the shape that failed the grade. r1 dev residual: 22/25
vs the human's 23/25, before hitter-chain anchors (which measured 72%
contact recall on r1 and carried the missile run to 96%). 30 deg is
dominated: it trades r1 back down for an r7 bump. Total sweep: 33
configs, ~3.5 h compute, every refuted mechanism written into the
module docstring. Next: owner designates the sealed rally, readiness
in full grade config (needs the Drive assets), one re-grade.

## 2026-08-31 (night) — READINESS PASSES in grade config; r10 sealed, staged, and waiting on the word

Drive folder now link-shared (owner) — assets pull directly, so the
whole grade config runs from a fresh session. First readiness run on
r7: checks 1/3 PASS (V 82.3, replication 1.70 ft) but check 2 came
back 77.8@86 — the addendum's "hitter anchors union into the timing
stream" conjecture, measured false (77.8@98.4 without the union;
fake-swing anchors mint cheap-turn zones and the bloated event set
eats the null pct). Correction recorded in the gate file: timing
stream = self-feedback anchors ONLY (exactly the swept config);
hitter anchors stay in the position stream, where the same run
proved them (bound claiming, 1.70 ft). Re-run: **GATE VERDICT r7
PASS** — the readiness rule caught a config bug before it could
burn the seal, which is its entire job.

r10 designated SEALED by owner delivery (r9 -> train). Owner
finalized the pass same night ("same clicks but stopped one frame
after the ball bounced out of play", 672 rows, end 318.447 = the
point-dead mark; only the last row was read). Owner also flagged a
mid-rally lob segment they were "mostly sure" they clicked right —
pre-recorded in the addendum for the label-sanity autopsy arm, not
touched. Clip verified (1980 frames @ 60 fps, offset 292.7),
candidates extracted post-readiness (39.7/frame), r0010.npz staged,
graded-run configuration note frozen in ball_gate.md. Bonus train
read: r9's human path passes the frozen battery 82.8% at pct 100
(29 impacts — real null power; the lob-adjacent contacts are hard
for the human too). The graded run fires on the owner's explicit
authorization and not before.

## 2026-09-01 — THE SEAL BROKE TWICE: r10 graded MIDDLE, check 2 beats the human, and the autopsy names the last two defects

Attempt 1 crashed pre-answer-key (no-wrist row in blur_gap_fill —
neutral guard, recorded, seal epistemically intact). Attempt 2, the
one graded run: **MIDDLE**. Check 2 PASS **beating the human path**
— timing stream 73.1@98 vs the human's 65.4@91 on 26 contacts,
beats its own shift-null; the channel dead since Gate C outscored
hand labels out of sample, with zero pose anchors. Check 1 66.2
(between bars). Check 3 FAIL on structure, not space: matched
impacts agreed at 2.73 ft (bar met) but bounces 3 vs 10 and
crossings 10/19 — dink rallies are bounce-rich and the position
stream over-fragments. Pooled absolute newly binding and short
(73.1 < 80) — noted: the HUMAN path also fails the 80 bar here
(65.4; first human pass ever to), the lob class is hard for people.

Autopsy, arms in order, ORACLE ARM VALIDATED FIRST on the r8
reference (my implementation reproduces 93.8 vs the recorded 94.6 —
the swing-proxy lesson applied to the autopsy itself): candidates
exonerated (93.6% recall; the owner's two flagged lob windows fully
covered, 3/3 and 4/4 — their "mostly sure" clicks were right,
again); frame sane; person N/A. The oracle substitution then told
the story: V 50.0% on CLEAN input, and the cause verified directly —
the labels hold exactly one gap over GAP_MAX (307.92-308.32, 24
frames at 60 fps, the flagged lob, ball above the frame) and the
oracle decode stops at its left edge: exactly 50% of the window.
Clean input cannot bridge an off-frame excursion; the graded run
"bridged" it through junk, which is what degraded check 1 and
shredded mid-rally segmentation. ATTRIBUTION: DECODER, two named
defects — off-frame excursion capability (r9's regime, now proven
load-bearing on a graded rally) and bounce-aware segmentation.
The two-regime timing fix survived both defects out of sample.

MIDDLE's one train-only iteration is therefore fully specced:
excursion edges (exit near frame top moving up -> ballistic
re-entry, gaps to ~1.5 s) + bounce segmentation, trained on
r9/r10's lobs and r10's 10 bounces. Then r20 seals when the owner
labels it, and the re-grade decides the channel.

## 2026-09-01 (cont.) — the licensed iteration, run end to end: four keeps, two reverts, and check 3 becomes the last wall

Full-day sequence after the MIDDLE (every step committed with its
numbers; this is the compressed record, the gate file has the
verdict-relevant one):

KEPT (multi-rally evidence): (1) excursion edges + EXC_COVER absence
allowance + (3) looser top-band hull margin — r10 oracle 50->95.8 V,
the flat-price version measured unable to outbid in-hull limb junk
(byte-identical re-run), the blanket court waiver measured breaking
r6 check 2 via corner scorebug junk (restored 100@96 with the
margin); (2) ANCHORS NEVER TOUCH A DECODE — the position stream
anchor-free took r10 check 1 66.2 -> 73.6 PASS and r7 82.3 -> 84.8,
completing what the readiness measurement started on the timing
side; (4) check-3 BOUNDS FROM THE TIMING STREAM — r10 matching
16 -> 20/26 at 2.73 -> 2.15 ft, r7 restored to 6/8 @ 2.76 with 7/7
crossings after the anchor-free position turns had broken it (3/8 @
7.39: the position path's turns were leaning on anchored decoding;
the timing stream is the "when" instrument, so bounds belong to it).

REVERTED with prejudice: strict claiming (paddle gate + alternation
prune) — r10 +2 matches for an r7 collapse; and the two knobs that
DO NOT DISCRIMINATE on r10 truth, recorded so nobody re-probes:
claim radius (true:junk kill ~1:1 at 130/180/240) and turn angle
(distributions overlap; r7's wobble separation does not
generalize). The alternation prune also carries a structural lesson:
exact side alternation only prunes safely over a COMPLETE claim
sequence — with holes, a distant same-side pair usually brackets a
MISSED contact, and pruning kills a true bound (ALT_MAX_S is the
hole-safe form, parked with the machinery).

WHERE IT FROZE: checks 1+2 pass everywhere that matters; check 3
fails on all three rallies, each on a DIFFERENT component — r7
over-generates bounces (5v2: spurious unclaimed timing turns), r10
under-recovers (5v10, crossings 14/21), r6 accuracy (3/7 @ 8.84 on
the 5.6 s pathological short rally). That pattern is the signature
of the two upstream problems, not of any decoder constant: HITTER-
ANCHOR PRECISION (73 anchors / 26 contacts — loose claiming eats
real bounces as bounds, and no local geometry filters junk anchors)
and BOUNCE-VS-JUNK TRIAGE of unclaimed turns (physics candidates:
a bounce reverses vertical image velocity near the floor plane and
sits far from every paddle; junk doesn't). Both are named,
neither is built; knob-turning stopped deliberately at the
overfitting line.

THE FORK (owner's, recorded in the gate file): hold r20 until an
anchor-precision + bounce-triage pass moves check 3 on train — or
re-register a narrower CONTACT-TIMES CHANNEL (checks 1+2) that
licenses touch share / tempo / shot-category products now, leaving
replay + placement (the bounce-dependent family) behind the full
check 3. Checks 1+2 currently carry: touch share, categories,
attack-onset/tempo, rally structure, the temporal-model corpus.
Check 3 carries: the 3D replay at scale, placement/landing stats
(a bounce IS a landing), true mph, contact heights.

## 2026-09-01 — owner eyewitness note: far-court bounces hide against WHITE DRESSES

Measured occlusion at floor-proximity moments (proxy: image-lowest
labeled point per inter-contact segment, +/-2 frames, rallies
1/6/7/9/10): NEAR court 9% carry I/N frames, FAR court 32% — the
end-view camera puts the near players' bodies between the lens and
the far kitchen. Owner adds the mechanism the numbers couldn't see:
the far-court losses are largely CONTRAST, not geometry — the ball
sits in front of the white dresses and the gray-level motion diff
(and the human eye) gets nothing. Implications: (1) the localized
sibling of the ball-closure's "misses pile at contacts" — no
detector fixes it on this footage; fit-demanded bounces (fix #1) is
the correct response because physics needs no pixels at the bounce;
(2) OUTFIT-DEPENDENT sample quirk — this match (women's whites) is
near worst-case; dark-kit matches will measure differently, a
generalization caveat on bounce behavior tuned against Chicago game
1; (3) named future candidate-stage lever if ever needed: hue-channel
differencing (neon ball vs white fabric) where gray-diff is blind —
not built while overall candidate recall runs 84-96%.
