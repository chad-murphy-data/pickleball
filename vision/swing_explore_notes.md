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
