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
