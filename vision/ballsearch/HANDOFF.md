# Ball-search thread — handoff (2026-09-01)

Dated status snapshot + next-thread to-do for the corridor ball-path
tracker. Written because the working thread was running out of context.
The instruments lived in a session scratchpad; they are now committed
HERE (`vision/ballsearch/`) so a fresh thread can run them.

**Direction: `ROADMAP.md` (this directory) — the three phases, what
closes each, and what is deliberately not being chased.**

The narrative record stays in `vision/swing_explore_notes.md` (chapter
"Pose-corridor ball re-search", line ~4506 onward: post-verdict
measurements, spaghetti, emission, soft-DP) and the per-channel ledger
in `vision/STATUS.md`; this file is the operational summary. Where they
disagree on a NUMBER, the notes file is the record.

## 2026-09-03 — r3 / r4 / r2 reads, and TWO SCORER ARTIFACTS

Owner delivered ball_path_r3 (528 rows, 348 V / 154 S / 24 I / 2 N; two
frame gaps at the top of frame = the two lobs that left the picture),
ball_path_r4 (403 rows, 273 V / 106 S / 24 I, no gaps) and ball_path_r2
(291 rows, 171 V / 96 S / 24 I, no gaps). All three are TRAIN.

Reads (incumbent cell, as the scorer counts today):
  r3  tracked+inf r@12 344/502  prec 0.81   decode ceiling 366/502
  r4  tracked+inf r@12 275/379  prec 0.93   decode ceiling 167/379
  r17 (prior)     r@12 253/379  prec 0.94   decode ceiling 244/379
r4 is the strongest read so far: path-first + gap fill reach 108 clicks
the candidate decoder alone cannot (ADDED@12 113).
r2's ball_grade gate returned CHECK 3 FAIL — autopsy still open.

### The S-click question (owner asked: are they really ignore-zones?)

Measured with NO tracker and NO model: leave-one-out local-quadratic
residual, contact-straddling windows dropped (click_diag.py precision).

  V clicks  n=1046  median 1.93 px  p75 3.02  p90 4.25  94% within 6 px
  S clicks  n= 387  median 2.68 px  p75 4.15  p90 7.23  87% within 6 px

S clicks are the ball and they are precise to ~2.7 px. 87% of them
already satisfy R_POS=6. The current rule (R_IGN=22, never a positive)
discards ~500 genuine positives across the train rallies in exactly the
stratum the detector is worst at. RULE UNCHANGED — this is the owner's
call and they have the number now.

### Two SCORING artifacts, both pessimistic, both speed-scaling

1. cdp.score matches a click to `track.get(f) or track.get(f-1) or
   track.get(f+1)` — FIRST available, not nearest, never interpolated.
   When frame f is missing it silently compares against a position up to
   1.5 frames away; at 40 px/frame that is 60 px of bookkeeping error.
   Only fast balls are hit.
2. The click grid and the clip frame grid are out of phase by a fraction
   of a frame. Fit on V clicks ONLY (click_diag.py phase): r3 -0.45,
   r6 -0.25, r7 -0.50, r17 -0.20 frames at 60 fps — every rally negative,
   i.e. click times run 3-8 ms late, consistent with the browser seeking
   to the frame at or before currentTime. The fit never sees an S click
   and still halves S error (r3 5.92 -> 2.66 px, r7 9.90 -> 3.75 px),
   which a per-click artifact could not do.

Rescored with both removed (interpolate, then phase):
  r3   r@8 273 -> 384   r@12 338 -> 388   (of 502)
  r7   r@8 123 -> 169   r@12 157 -> 178   (of 260)
  r17  r@8 236 -> 241   r@12 244 -> 245   (of 379)
r17 barely moves because it was nearly in phase — that, not rally
difficulty, is why r17 looked so much healthier than r3.

DO NOT ship the fitted phase. It is fit against the track and is a
diagnostic that the problem is real, not the number to correct with. A
shipped correction must measure each clip's true cut offset against the
full match video. The incumbent seals (r9 431@0.69, r10 320@0.67) were
scored under the old rule and have NOT been rescored — that is a graded
re-run and needs explicit owner authorization.

### Why whole flights come back empty (owner question, autopsy)

flight_autopsy.py. A flight = clicks between consecutive contacts.
Over r9+r10 (AUTOPSY use of the eval clicks — no knob touched):

                    zero-hit    hit
    clicks             14        27      <- the discriminator
    S share            0.21      0.24    <- no difference at all
    speed (px/f)      10.6      12.1
    duration (s)       0.57      0.90

It is FLIGHT LENGTH, not click type or speed. path-first needs a seed run
of s_min=6 consecutive frames above p_seed and then commits or drops the
whole flight; a 0.3-0.45 s exchange is 18-27 frames at 60 fps, so losing
a handful kills the entire flight at once. That all-or-nothing structure
is what "it detected nothing" looks like from outside.
Caveat: the r9/r10 phase scalars (-0.50, -0.70) were fit for this
autopsy, so they are contaminated for grading — a shipped scorer must not
reuse them.
Biggest single zero-hit category in EVERY rally is the PRE-SERVE segment
(ball stationary in the server's hand, punished by persist + dbody).
Arguably correct behaviour being counted as a miss.

### Owner's speed-vs-appearance hypothesis (partly confirmed)

Label-side only, train rallies (click_diag.py appearance). S clicks run
2.2x faster than V clicks in median pixel displacement (20.8 vs 9.4
px/frame pooled) and the split holds in every rally — "looks like a
streak" really does mean "going fast". The median S patch contains zero
yellow pixels, the median V patch ~12. The SHAPE half is NOT settled: over
a 21x21 patch the ball is a handful of pixels and the court dominates the
statistic, so elongation did not separate (V 1.63 vs S 1.68). Note the
emission model already carries `yellow` as its single largest positive
weight (+1.287 of 14 features); what it has NO feature for is streak
elongation or orientation. That is the open gap.
Off-screen check r3 passed clean: 2 trackpoints across 2.1 s of ball out
of frame, the first rally able to test that failure mode.


### r5 delivered — sitting 3 complete

ball_path_r5: 446 rows, 290 V / 114 S / 23 I / 19 N, no frame gaps
(tool t0 117.279 vs the recorded 117.28). TRAIN. Three N runs — two
singletons and one 17-frame run at frames 191-207, bracketed by y=1 and
y=2, i.e. another lob out of the top of frame. Unlike r3 the owner
marked these N with the row present rather than skipping, which is the
better shape: the reader knows the ball was unplaceable rather than
having to infer it from a frame gap.

### Owner callout: "an S is better than a guess" — CONFIRMED, and bigger

inferred_audit.py. Gap fill v2's inferred frames, crossed against the
click path, over r2/r3/r4/r6/r7/r17:

  414 inferred frames -> V 240 (58%)  S 83 (20%)  I 33  no click 58 (14%)
  78% of the guesses land on a frame the owner could see and click
  32% sit within 0.20 s of a contact (the "near the paddle" half)
  the guesses are poor: median 10-55 px from the click that was there,
  27-63% of them within 12 px

Split on decode@12 — did a candidate exist that path-first declined?

  S clicks, candidate existed   57      S: 57/85  = 67% declined
  S clicks, no candidate        28
  V clicks, candidate existed   92      V: 92/243 = 38% declined
  V clicks, no candidate       151
  overall 149/328 = 45% of the guessed-at clicks HAD a candidate

So the streak stratum is where the tracker most often walks past evidence
already in hand, at nearly twice the V rate. Mechanism, end to end: an S
click yields no positive, so the emission model never learns the streak
appearance; its largest weight is `yellow` (+1.287), the one feature a
white smear lacks; streak candidates therefore score under p_seed;
path-first's seed test declines them; gap fill guesses over the top.
Clicks are ground truth and not a run-time input, so this is NOT "use the
click" — it is that the S stratum carries recoverable evidence the
current rule forbids the model from learning. RULE STILL UNCHANGED.

### r2 gate autopsy (CHECK 3 FAIL, verdict FAIL via c2_nullfail)

  CHECK 1 V: 81.9% (136/166)  [bars PASS>=70, FAIL<40]
  CHECK 1 S: 92.7% (89/96)    <- S is the BEST stratum on this rally
  CHECK 2 turns[tracker]: recall 72.7% at null pct 81 (95th 81.8) - misses
  CHECK 3: 8/10 impacts matched, median 3D 6.13 ft (bar <=3.0);
           bounces tracked 8 vs human 3 (bar +/-1)

Two reads. CHECK 1 puts S ABOVE V here, another data point against "S is
the weak stratum". And r2 has only 11 contacts, so CHECK 2's null is
quantised in 1/11 steps: the 95th percentile is 9/11 and the tracker got
8/11, i.e. clearing the bar needs 10 of 11. That is a power limit on a
short rally as much as an instrument failure — read it next to r3 (17
contacts) and r4 (16) before concluding. The substantive CHECK 3 finding
is the bounce count: 8 claimed vs 3 reconstructed, i.e. the tracker
over-segments the rally into more flights than were played — the same
seed-and-commit machinery that drops short flights whole, failing the
other way.


### r5 read, and the three gate verdicts (r2 FAIL / r3, r4, r5 MIDDLE)

r5 (`train_read_r5.txt`) is the SLOWEST rally in the set and it reads
worst of the four, which is the opposite of the expected ordering:

  404 V/S clicks (290 V / 114 S), decode@12 280/404 = 0.69 ceiling
  path-first    r@12 274  prec@12 0.84  ADDED@12 43
    pf[V]       r@12 209/290   prec 0.92
    pf[S]       r@12  65/114   prec 0.66
  tracked+inf   r@12 299/404 = 0.74  prec 0.79  ADDED@12 65
  inferred      194 frames (the most of any rally), r@12 37, prec 0.50

By prefill contact TYPE the ordering inverts against every prior read:

  fast    48 clicks, 41 pf-hit = 85%
  slow   144 clicks, 75 pf-hit = 52%

That is not a speed effect, it is a PROXIMITY effect. r5 is a dinking
rally: "slow" contacts put the ball low, near bodies and paddles, where
`dbody` and `crowd` — two of the emission model's larger negative
weights — are doing exactly what they were fitted to do on r6/r7, which
had far less kitchen play. The worst single contact is 132.46 (16
clicks, 1 hit). The mirror of the r9/r10 finding: path-first drops
whole flights, and here it drops the flights that live at the kitchen.
So the instrument has TWO failure modes, not one — fast/blurred balls
lose to `yellow`, slow/close balls lose to `dbody`+`crowd` — and they
are not the same rallies. A single global tune cannot be optimal for
both; that is an argument for conditioning, not for another knob turn.

Gate verdicts, all four rallies of sitting 3 (`ball_grade.py`, prefill
contacts, `--prefill-ok`):

  r2 (11 contacts)  FAIL     C1 V 81.9 / S 92.7 = pass
                             C2 recall 72.7 at null pct 81 -> NULLFAIL
                             C3 fail: 8/10, median 6.13 ft, bounces 8 v 3
  r3 (17 contacts)  MIDDLE   C1 V 85.4 / S 87.4 = pass (best C1 of the four)
                             C2 82.4 v human 70.6, both pct 100 = PASS
                             C3 fail: 9/16, median 4.84 ft, bounces 4 v 6
  r4 (16 contacts)  MIDDLE   C1 V 53.4 / S 64.8 = MIDDLE band (bars 70/40)
                             C2 beats own null 95th, human-matched no
                             C3 PASS: 9/16, median 1.76 ft, bounces 5 v 4
  r5 (14 contacts)  MIDDLE   C3 fail on distance ONLY: 8/14 impacts,
                             median 4.60 ft, bounces 6 v 6 (exact),
                             crossings 9/9

THE CHECKS DISAGREE ACROSS RALLIES, AND THE DISAGREEMENT IS THE FINDING.
r3 has the best CHECK 1 and the only clean CHECK 2 human-match, and fails
CHECK 3. r4 has the WEAKEST CHECK 1 of the four (53.4% V, in the middle
band) and is the only CHECK 3 pass, at 1.76 ft. So anchor-level precision
and 3D replication quality are not the same axis here and on this
evidence run opposite: r4's 16 flights are long and well separated, which
is what the ballistic fit wants, while its anchor set is thin (40 anchors
vs r3's 51) on a rally whose clicks are dominated by a few long arcs.
Do not treat any single check as the universal blocker, and do not tune
against CHECK 1 expecting CHECK 3 to follow.

CHECK 2 beats its own null 95th on r3, r4 AND r5 and fails only on r2 —
independent confirmation that r2's FAIL is the 11-contact power limit.

Two things follow. First, r2's FAIL was a POWER artifact and is now
confirmed as such: MIDDLE requires `not c2_nullfail`, so CHECK 2 cleared
its permutation null on r3, r4 and r5 — every rally with 14+ contacts —
and failed only on the 11-contact rally where the null quantises in
1/11 steps and clearing needs 10 of 11. Do not read r2 as the tracker
being worse on r2; read it as an 11-contact rally being unable to
resolve the question. Second, r4 is the first CHECK 3 PASS on any
newly-labeled rally (median impact error 1.76 ft against a <=3.0 bar),
and it is also the strongest train read on record (ADDED@12 113 over a
ceiling of only 167). r4 and r5 bracket the instrument: same tune, same
model, 1.76 ft vs 4.60 ft, and the difference is what the rally is made
of.

All three MIDDLE verdicts route to the same frozen clause — one
train-only iteration, then ONE re-grade on a newly labeled sealed
rally. No seal is consumed by any of the above; r2-r5 are train.


## Constraints in force (owner-set; carry verbatim)

- No graded re-run / seal consumption without explicit owner
  authorization. **r20 is the next seal.** Bars never loosen.
- TRAIN rallies = r6 + r7 only. r9 / r10 owner clicks are
  EVALUATION-only: used for grading and autopsy, never fed into
  production path selection, never used to tune any weight/threshold.
  Any knob is tuned on r6/r7 (cross-fold p-caches `_x`), the selection
  rule is written down BEFORE the numbers, then ONE shot on r9/r10.
- Training discipline (auto-label poisoning lesson): positives ONLY
  from owner V clicks within R_POS=6 px; S clicks ("close to where the
  ball is", hidden) are ignore-zones (R_IGN=22), never positives; never
  train on tracker output or model self-labels.
- Oracle-bounds fits are diagnostic-only. Temporal-gate holdout rows
  (`data/vision/label_split.csv`) are untouchable; holdout burns on use.
- No model identifiers in commits/PRs/pushed artifacts.

## What is built (all in this directory)

| file | role |
|---|---|
| `c3_lab.py` | Stage A cache per rally → `c3_cache_r{r}.pkl` (windowed candidates, decode, timing stream, turns, anchors, floors, human side fit). WINDOWS dict holds serve/end/pose-npz per rally. |
| `claim_lab.py` | `load(rally)`, `paddle_series(npz)`; claim logic labs. |
| `corridor_lab.py` | corridors between consecutive contacts (prod = approach_events + anchors → dedupe → claim_bounds; oracle = hand taps `c["imps"]`); window = chord ± (wx=min(140,40+0.2L), wy=min(170,55+0.3L)); truth loader; `decode_recall`; R_MAIN=12. |
| `corridor_dp.py` | Viterbi chain per corridor over per-frame candidates (K=14, GAP=6): accel + gap + endpoint + body-extremity cost (W_BODY=25, R_BODY=16) − peak bonus + `W_P_SOFT·(1−p)`. |
| `emission.py` | learned per-candidate scorer (hand-rolled Adam logistic, 14 features). `train` (r6↔r7 cross-val + pooled model → `emission_model.json`), `cache <r>` (p-cache from pooled model), `cache-cross` (r6 scored by r7-only model and vice versa → `p_r{6,7}_{mode}_14_x.npz`, fold kp97). |
| `launch_prior.py` / `launch_prior.json` | shot book: 61 launches / 27 bounces from r6/r7 human segments. |
| `spaghetti.py` | trail matcher v3 (DIRECT + BOUNCE families, MC null, mode prior, ABSTAIN=2.0) AND the graded harness: `python3 spaghetti.py <r> --lrn --soft 25` prints the prod/oracle table + per-corridor autopsy rows. Auto-builds `cands_r{r}_{cc,peak}_14.npz` on first use. |
| `softdp.py` | pre-registered W_P_SOFT sweep on r6/r7 cross-fold; frozen rule → W=25. |
| `fusion.py` / `fusion_tune.json` | the L3 three-part model (emission + spaghetti trails + DP as one cost): `tune` (r6/r7 cross-fold grid, frozen rule, writes the verdict), `grade <r>` (refuses r9/r10 without a live verdict or with knob overrides), `selftest` (synthetic corridor; asserts `trail=None` is bit-identical). VERDICT 2026-09-01: **DEAD** — see the section below. `corridor_dp.py` carries the optional `trail=`/band-pool/`return_cost` extension it needs. |
| `geom_lab.py` | to-do #1 diagnostic: per-click strata (nocor / outwin / nocand / cand-hit / cand-miss with pool rank + skipped-vs-wrong), per-corridor excursion + endpoint-error table, geometry-only coverage counterfactuals (kT / cap / K grids). `python3 geom_lab.py <r>`. |
| `geom_fix.py` / `geom_tune.json` / `geom_grade_r9.txt` | the corridor-geometry FIX: knobs cap (wy ceiling), kT (duration term), ep (tapered end pad + END_R), pool (centre vs learned-p). `tune` = 54-cell grid on r6/r7 cross-fold under the frozen rule (writes the verdict), `grade <r>` = one-shot vs incumbent with V/S splits, displaced nulls, strata and per-corridor tables (refuses r9/r10 without a live verdict or with overrides). VERDICT 2026-09-01: **LIVE**, cap=260 kT=40 ep=60 pool=p; **r9 prod 479 @ 0.74 vs 431 @ 0.69**. `corridor_dp.py` carries `POOL_BY_P`. |
| `pathfirst.py` / `pathfirst_gate.md` / `pathfirst_tune.json` / `pathfirst_grade_r{9,10}.txt` | **THE INCUMBENT (adopted 2026-09-01)**: path-first tracker — no contact detector, no corridor box. 3-seed drag-free arc hypotheses over the whole-frame candidate cache (top-4 by learned p per frame, p ≥ P_SEED, not on a body), p-weighted support minus a random-probe baseline, NMS, drag refit + bidirectional growth (R_GROW 10, stop after GAP misses), greedy selection by density, contacts = flight ends. `selftest`, `tune` (12-cell grid r6/r7 cross-fold, frozen rule, writes the verdict), `grade <r>` (one-shot vs the corridor incumbent: V/S splits, displaced + time-shift nulls, strata, oracle-contact recovery, per-flight table; refuses r9/r10 with overrides or a dead verdict). Pre-registration + results addendum in `pathfirst_gate.md`. **r9 537 @ 0.87 vs 431 @ 0.69; r10 422 @ 0.88 vs 325 @ 0.69.** ~3 s per rally. |
| `render_pathfirst.py` | viewer only: draws the path-first track (trail, ball ring, v3 event labels, faint candidate dots) on `r{N}_clip.mp4` → `pathfirst_r{N}.mp4` (960×540 H.264, default half speed). `--reddit` = the share cut (owner ask 2026-09-02): 1280×720, no labels/HUD, ball and trail always red, pose skeletons of the four tracked players → `pathfirst_r{N}_reddit.mp4` (the two share cuts ARE committed, force-added past `*.mp4`, ~9 MB each). Reads no truth, tunes nothing. |
| `rally_stats.py` / `rally_stats_r{9,10}.txt` | PROTOTYPE rally stats off the adopted track + v3 events, three rules written before any rally was looked at (hit = event within 3 ft of a player's paddle proxy, same player twice inside 0.6 s = one; speed-up = first flight after the 3rd hit launched ≥ 38 ft/s and starting at a hit; last shot = last attributed hit + end of the final flight in court ft). Identity is POSITION only (near/far × left/right from the pose tracks). `--grade` (r9/r10, evaluation only) maps names to tracks by majority vote over the owner's labeled contacts. RESULT: hits per player within ±1 of the labels for all 8 player-rallies (r9 10/5/10/5 vs 9/5/10/5; r10 4/9/9/5 vs 5/8/9/4) — re-run 2026-09-02 on the adopted gap-fill flights: r9 unchanged, r10 3/8/9/4 (three exact, far-left −2); last shot right in both; **speed-up WRONG in both** — launch speed off a one-camera 3D fit is depth-dominated and fragments inflate it (r9 named a 71 ft/s fragment at 260.97 vs the labeled first fast shot at 257.84; r10 303.18 vs 300.23). Not a channel; a speed measure the camera can support (image speed at local scale, or hit-to-hit time) is the next thing to try. |
| `rally_3d.py` / `court3d_r{9,10}.html` | orbitable 3D court view (court3d.write_viewer) of the path-first flights — every flight already IS a 3D arc — plus the four players' floor tracks from the pose npz through the z=0 homography and the attributed hits. Prints the free ground-truth check: r9 17 net crossings, 4 under the 34-in tape; r10 10 / 0 (a crossing under the tape = a flight whose depth or height is off, the one-camera weakness made countable). Committed (small HTML). Depth is the weak axis; gaps stay gaps. **PLAYER-BOX RE-LIFT (owner ask 2026-09-02, restoring the pseudo-boundary the court3d pass-2 fits had)**: for display, every flight is refit to its own graded pixels with a hinge keeping the arc inside the four players' floor box (+4 ft) and above the floor (W_BOX 1 px/ft; 4 px/ft made one r9 fit diverge). r9: pixel rms vs the graded track median 0.07 / max 1.8 px, worst excursion 32 ft → 10 ft, path y −29..58 → −5..52, net crossings 14/3; r10: 0.09 / 2.0 px, 21 ft → 4 ft, 11/0. The 2D track is untouched — this is the viewer's lift, not the tracker. |
| `render_court3d.py` / `court3d_r9.mp4` | the 3D viewer as a video (owner ask 2026-09-02, easier to post): reads PATH/IMPACTS/PLAYERS out of the committed viewer HTML, replays them with the viewer's own projection and colours in real time with a slow orbit, 1280x720 @ 30 fps, libx264 crf 18. Three honest cosmetics the page lacks: the path is broken at flight gaps (the page joins consecutive samples with a straight line), the ball is hidden while the track is lost, rings flash at the attributed hits. r9 chosen over r10 (more samples, all four player tracks complete). Committed (force-added past the mp4 ignore, 4.6 MB). Rally 9 is also published as an interactive artifact (same viewer with the same gap fix, real-time play button). Nothing re-fit; viewer only. |
| `handoff.py` / `handoff_gate.md` / `handoff_tune.json` / `handoff_grade_r{9,10}.txt` | Hand-off seeding, pass 2 on top of the frozen path-first pass: short-span seeds (6/8/12 frames) allowed NEAR BODY, only inside a zone next to a pass-1 flight end. Pre-registered, tuned on r6/r7 (16 cells → r_zone 70 / w 18 / p_hand 0.25 / s_min 3, 277 @ 0.810 vs 263 @ 0.807), ONE SHOT on r9/r10 spent 2026-09-02: track r@12 537→563 (r9) and 422→446 (r10) at unchanged precision, nulls clean, BUT the v3 events layer re-run on the new flights lands F1 .636 on r10 vs bar .645 (r9 .758 passes) → **NOT ADOPTED**, incumbent stays. `grade` for r9/r10 is spent. Next gate if reopened = events re-tune on the hand-off track (new pre-registration). |
| `label_picks.md` | the owner's click list for 2–4 more ball-click rallies chosen for FAST exchanges, inside the split rules: r17 (train), r21 (fresh seal), then r4 (train), r20 (the reserved seal). r22–r30 holdout untouched. |
| `click_setup.md` | **THE CLICK PACKAGE (2026-09-02, owner: "don't give me the minimum")** — nine rallies in three sittings with tools built and staging done: r17 / r21 / r18 / r19 (sitting 1), r20 (sitting 2), r3 / r4 / r2 / r5 (sitting 3 — **staged 2026-09-02 off the PREFILL span, no contact pass needed**). Committed: `data/vision/ball_audit_r{2,3,4,5,17,18,19,20,21}.html`, `data/vision/ball_candidates_r{2..5,17..21}.csv.gz` (clips cut at the audit-tool span via `cut_clip.py --stage`, check_clip PASS), c3_lab WINDOWS + corridor_lab CLIPS for 2–5 and 17–21. Pose npz: r17 + r2–r5 extracted here on CPU (gitignored; `pose_meta_r17.json` / `pose_meta_r2to5.json` are the stamps), 18–21 still need a run (CPU here or Colab). What to run when the CSVs land is written there. |
| `train_read.py` / `train_read_r17.txt` | **The first read of a NEW TRAIN rally** (2026-09-02): refuses r9 / r10 / r20 / r21, reads only the committed tune records, scores the incumbent path-first track + gap fill v2 against the V/S clicks with both nulls, then buckets every click by the nearest manual contact (serve / return / slow / fast / middle, S-only split) and prints a per-contact coverage table. `python3 train_read.py 17`. `c3_lab.py` now takes rally args (`python3 c3_lab.py 17`; no args = the original four). |
| `gapfill.py` / `gapfill_gate.md` / `gapfill_tune.json` / `gapfill_grade_r{9,10}.txt` | GAP FILL BY ARC EXTENSION (owner's framing 2026-09-02: the ball behind a paddle/player exists only through inference): for each gap between consecutive path-first flights, extend A's arc forward and B's backward, switch at the frame where they come closest, leave the gap open if they never come within D_MEET px; filled frames tagged `inferred`; NO paddle used (the wrist+forearm proxy picks the switch time worse than the arcs do — autopsy). Pre-registered, tuned on r6/r7 (6 cells → gap_max 0.8 / d_meet 20, 273 @ 0.782 vs 263 @ 0.807), ONE SHOT on r9/r10 spent 2026-09-02: r9 556 @ 0.87 PASS all bars (events F1 .756); r10 443 @ 0.850 FAILS the precision bar by 0.010 (recall, nulls and events F1 .684 all clear) → **NOT ADOPTED**, incumbent stays. Inferred frames alone are right at 12 px ~60% of the time (0.65 / 0.58); events F1 rises on BOTH rallies (the fill gives the seam rule meeting arcs). **v2 SAME DAY, owner go to re-use r9/r10 ("let's not be as sweaty")**: the product is re-stated as tracked frames (bit-identical, asserted) PLUS a TAGGED inferred stratum graded on its own (prec ≥ 0.5, r@12 ≥ 10, own displaced + time-shift nulls ≤ 3) + events F1 ≥ adopted − 0.03. Re-tuned under that rule on r6/r7 → gap_max 0.8 / d_meet 40. r9: inferred 38 @ 0.667, events .764; r10: 41 @ 0.594, events .658 → **ADOPTED AS A TAGGED PRODUCT** (`gapfill.product(ctx)`); the tracker's quoted number stays the tracked half's. Consumers draw inferred frames dashed (rally_3d / court3d viewer / render_court3d) and run events + stats on the filled flights (rally_stats). Clean re-check on r20/r21, bars as written, no re-tune. **v3 / v3b (hit-anchored fill of the open gaps, 2026-09-02) DEAD ON TRAIN, r9/r10 untouched** — see the gate §v3–v3c and `gapfill_explore3.txt`: the open gaps are tracker-tail junk (r7) and a two-contact drive (r6), not something a time-anchored 3D kink can fill. |
| `render_pathfirst.py --bridge` | cosmetic, owner-approved for the demo: gaps ≤ 0.5 s between tracked flights drawn as a DASHED straight segment with a sliding dashed ring, persistent caption "dashed = inferred between tracked flights, not tracked". Never enters the track/events/stats/grades. Output `pathfirst_r{N}_reddit_bridge.mp4`. |
| `learner.py` / `learner_gate.md` / `learner_train.txt` / `emission_gbt.json` / `learner_curve.py` + `.txt` | The better learner (owner go 2026-09-02): gradient-boosted trees on emission.py's 14 features, same labels/discipline, three fixed configs, pre-registered gate. **GATE 1 DEAD**: AUC +0.002 / +0.007 over the logistic (noise on 161–198 positives) but the 97 %-recall tail keeps 0.635 of r6 negatives vs the logistic's 0.319 → no caches, no tune, no shot. Learning curve (diagnostic): logistic FLAT in labels (feature-saturated), trees still CLIMBING and overtaking only at ≥ 75 % of positives → the learner is now LABEL-limited; re-run `train` when r17/r21 land. `pathfirst.py` gained the inert `PF_PXS` p-cache-suffix env hook (unset = unchanged). |
| `events.py` / `events_gate.md` / `events_tune{,_v2,_v3}.json` / `events_grade_{r9,r10,v3_r9,v3_r10}.txt` | EVENTS layer on top of path-first (track untouched, asserted): one event per change of flight. v1 (time-gap pairing) FAILED the recall guard on both r9/r10; v2 (arcs must meet) DEAD on train; **v3 ADOPTED (owner's framing: pair by PHYSICAL distance between arrive and depart points, ≤ 2.5 ft at the local px/ft scale, 0.5 s sanity cap)**: r9 F1 .731 vs RAW .625, r10 .675 vs .617, recall guard and time-shift null cleared on both. `tune --v3`, `grade <r> --v3`. Typing hit/bounce is secondary and unbuilt. |
| `cut_clip.py` | cuts `r{N}_clip.mp4` from the full-match source with the offset and frame count read from the committed candidate CSV (system ffmpeg or the imageio-ffmpeg static binary). |
| `check_clip.py` | verifies an owner-cut clip against the committed `ball_candidates_r{N}.csv.gz` (frame count / size / fps, implied offset, extractor re-run matched at frame shifts −3..+3) before any cache is built from it. |
| `corridor_autopsy.py`, `miss_map.py`, `r10_autopsy.py` | diagnostic-only (truth used to ask WHY). |
| others (`anchor_*`, `hole_*`, `tr_grid*`, `split_lab`, `fit_lab`, …) | earlier-arc labs, kept for provenance; not on the current path. |
| `anchors_grade_r*.csv` | anchor grading outputs (small). |

Scripts hardcode `sys.path.insert(0, "/home/user/pickleball/vision")`
and write caches next to themselves (`SP = Path(__file__).parent`);
`.gitignore` here keeps `*.npz *.pkl *.mp4` out of git.

### Regenerating the environment in a fresh thread

Stage into THIS directory (owner supplies; not in git):
`r6_clip.mp4 r7_clip.mp4 r9_clip.mp4 r10_clip.mp4` (owner-cut 60 fps
clips; offsets in `corridor_lab`/`ball_gate.md`: r10 = 292.7; r6/r7/r9
are in the Drive pose folder, r10 is NOT — cut it with `cut_clip.py`
from the owner's `full_match.mp4.webm` in Drive) and the
pose streams `r0006.npz r0007.npz r0009.npz r0010.npz`
(`vision/pose_extract.py` on Colab per `gpu_runbook.md`). Then:

```bash
cd vision/ballsearch
for r in 6 7 9 10; do python3 c3_lab.py $r; done      # c3_cache_r*.pkl
python3 emission.py train                              # emission_model.json (already committed; re-run only to refit)
for r in 9 10; do python3 emission.py cache $r; done   # p_r{9,10}_{cc,peak}_14.npz
python3 emission.py cache-cross                        # p_r{6,7}_*_14_x.npz
python3 softdp.py                                      # reproduces the W sweep (r6/r7 only)
python3 fusion.py tune                                 # reproduces the fusion grid + DEAD verdict (r6/r7 only)
python3 geom_lab.py 6; python3 geom_lab.py 7             # strata diagnostic (any rally with caches)
python3 geom_fix.py tune                               # reproduces the geometry grid + LIVE verdict (r6/r7 only)
python3 cut_clip.py 10 full_match.mp4.webm             # r10 clip from the full-match source (offset/frames from the committed CSV)
python3 check_clip.py 10 r10_clip.mp4                  # before building r10 caches from a re-cut clip
python3 geom_fix.py grade 10                           # the r10 one-shot (geom_grade_r10.txt)
python3 geom_fix.py grade 9                            # the r9 one-shot (geom_grade_r9.txt); grade 10 once r10 caches exist
python3 pathfirst.py selftest                          # planted-arc self-test (must print OK)
python3 pathfirst.py tune                              # reproduces the 12-cell grid + verdict (r6/r7 only)
python3 pathfirst.py grade 9; python3 pathfirst.py grade 10   # the one-shots (pathfirst_grade_r{9,10}.txt) — already spent; re-running is a RE-GRADE
python3 render_pathfirst.py 9                         # watchable overlay (pathfirst_r9.mp4, gitignored)
python3 render_pathfirst.py 9 --reddit                # share cut (pathfirst_r9_reddit.mp4, committed); ~10 GB RSS — never two at once
python3 rally_stats.py 9 --grade                      # rally stats prototype + evaluation vs labels (rally_stats_r9.txt)
python3 handoff.py tune                                # hand-off pass-2 grid on r6/r7 -> handoff_tune.json (handoff_tune.txt)
python3 handoff.py grade 9                             # SPENT for r9/r10 (handoff_grade_r9.txt); re-running is not a new seal
python3 render_pathfirst.py 9 --reddit --bridge        # demo cut with dashed inferred gaps -> pathfirst_r9_reddit_bridge.mp4
python3 rally_3d.py 9                                 # orbitable 3D court view (court3d_r9.html)
python3 events.py tune --v3                            # events grid (r6/r7); grade 9 --v3 / grade 10 --v3 are spent one-shots
python3 spaghetti.py 9  --lrn --soft 25                # graded r9 (run_in_background; > 2 min)
python3 spaghetti.py 10 --lrn --soft 25                # graded r10
```
Heavy scripts exceed the 120 s foreground limit — background them.
No ffprobe (use cv2), no sklearn, no gh CLI.

## Incumbent (2026-09-01, late): PATH-FIRST — `pathfirst.py`

Adopted under the pre-registration in `pathfirst_gate.md` (frozen
before any number; results addendum there). One shot each, frozen
cell p_seed=0.4 s_min=6 gap=6:

| rally | corridor incumbent (prod) | path-first | nulls disp / tshift |
|---|---|---|---|
| r9 (779 clicks) | 431 @ 0.69 | **537 @ 0.87** (V 421/587, S 116/192) | 0 / 1 |
| r10 (657, re-cut clip) | 325 @ 0.69 | **422 @ 0.88** (V 316/487, S 106/170) | 0 / 0 |

Tune r6+r7 cross-fold: every one of the 12 cells beat the corridor
incumbent (205 @ 0.623); chosen cell 263 @ 0.807. Gain sits in the
out-of-window stratum (r9 82/131 vs 0, r10 74/97 vs 0 — lobs and
bounces the box never contained) plus a small selection gain inside
the box. Oracle-contact recovery from flight ends (secondary): r9
24/29 within 0.10 s (prod detector 19/29), r10 19/26 (15/26) — recall
only, no precision claim (64 / 56 boundaries emitted). Bounce typing
is effectively unbuilt (2 typed on r9, 0 on r10 vs 13 in the human
ledger). The corridor stack below stays committed as the comparison
arm and as the source of the strata; it is no longer production.

## Corridor stack numbers, superseded (dp-ccS+body, W_P_SOFT=25, learned p)

r@12 = truth clicks (V+S) with a track point within 12 px, over ALL
clicks; prec@12 = hits / at-click track points.

| rally | arm | trackpts | r@12 | prec@12 | previous incumbent |
|---|---|---|---|---|---|
| r9 (779 clicks) | prod (34 corridors) | 942 | **431** | 0.69 | 388 |
| r9 | oracle (29) | 885 | 406 | 0.69 | |
| r10 (657 clicks) | prod (28) | 653 | **320** | 0.67 | 282 |
| r10 | oracle (26) | 668 | 309 | 0.61 | |

Spaghetti alone (ccL): r9 prod 101 @ 0.51 (abstained 15/34), oracle
177 @ 0.74; r10 prod 86 @ 0.54 (12/28), oracle 115 @ 0.54. Displaced
nulls 0 everywhere except r10 oracle null0 5/657 (0.07).
Tuning sweep (r6+r7, prod+oracle, cross-fold p): W=0 326/0.472,
3 342/0.505, 6 351/0.533, 12 346/0.556, **25 376/0.633**, 50 368/0.636,
HARD@kp97 336/0.582. Disclosure: a W=12 smoke test on r10 (322/0.66)
ran before the tuning protocol existed; the frozen rule still picked 25.
Emission: cross-rally AUC 0.9042 / 0.9394; pooled kp97 0.0961; fold
kp97 r6 0.0277 / r7 0.1497.

Miss decomposition (why fusion was proposed): r9 prod misses = 197
wrong-emitted + 151 empty frames; r10 156 + 181.

## Corridor geometry fix (to-do #1) — 2026-09-01: LIVE, r9 +48

Diagnostic first (`geom_lab.py`, r6/r7): SELECTION is the largest miss
stratum (38–51 %) with the true candidate already in the pool (97 %+);
OUT-OF-WINDOW second (18–34 %), almost all with a candidate present,
below the chord (bounces) and above (lobs), overshoot p90 120–273 px;
endpoint errors 100–330 px on several corridors. Knobs chosen from
geometry-only counterfactuals, instrument committed before numbers.

Tune (54 cells, rule = max total r@12 s.t. pooled prec ≥ incumbent,
ties fewest knobs / smaller / centre): incumbent 376 @ 0.633 →
**cap=260 kT=40 ep=60 pool=p 458 @ 0.704**. Each knob helps alone and
they add (pool 393, kT 385, cap 414, cap+pool 431, cap+kT+pool 457);
cap 400 ≡ 260, kT 80 ≡ 40, ep 120 ≡ 60.

| rally | arm | incumbent | geom-fix | nulls |
|---|---|---|---|---|
| r9 (779) | prod (34) | 431 @ 0.69 | **479 @ 0.74** (V 368/587, S 111/192) | 0 / 14 |
| r9 | oracle (29) | 406 @ 0.69 | 432 @ 0.73 | 17 / 0 |
| r10 (657, re-cut clip; incumbent re-graded 325 @ 0.69) | prod (28) | 325 @ 0.69 | 323 @ 0.65 (V 227/487, S 96/170) | 5 / 0 |
| r10 (oracle incumbent re-graded 285 @ 0.60) | oracle (26) | 285 @ 0.60 | 310 @ 0.63 | 29 / 0 |

**VERDICT: SPLIT → incumbent stays in production** (rule written
before the r10 shot: adopt only if both clear). r9: +48 / +5 pp on
prod, +18 inside `cand`, +30 from `outwin` (0 → 30), outwin stratum
131 → 86 under the new box, corridor 272.58–274.58 (2.0 s) 9 → 39,
no prod corridor lost a hit. r10: prod −2 / −4 pp, ALL of it one
corridor — 301.02–302.73, the mis-merged double-contact segment
(to-do #4), 25 → 12; every other corridor is ≥ incumbent (+4 +2 +5
+2), outwin 97 → 60, and the item-2 lob/bounce corridors go 0 → 5 and
0 → 2 (reached, not tracked). Oracle +25 on r10 but null0 = 29 there
(highest on record) and the re-cut clip alone moved the oracle
incumbent 309 → 285, so the oracle delta is noise-sized. Full
tables: `geom_grade_r9.txt`, `geom_grade_r10.txt`; narrative in
`vision/swing_explore_notes.md`. Reading: the geometry finding
stands (the box was the miss on lobs/bounces; reaching ≠ tracking),
the one loss is corridor SEGMENTATION not window size, and the next
registration should gate the taller box on corridor quality and add
the pool-only trail — fresh rule, tuned r6/r7, one shot r9/r10.

r10 clip provenance: never in Drive; re-cut 2026-09-01 by
`cut_clip.py` from the owner's `full_match.mp4.webm` (Drive, 1280×720
60 fps, 289,199 frames), `check_clip.py` PASS (0.787 at shift 0 vs
0.567 at ±1; r9 control from the same source 0.767, staged original
1.000). Re-encode noise moves ~20 % of strong candidates > 3 px;
incumbent on the re-cut clip = prod 325 @ 0.69 / oracle 285 @ 0.60
vs 320 @ 0.67 / 309 @ 0.61 recorded on the original.

## Fusion (L3) — built and tuned 2026-09-01: DEAD under the frozen rule

`fusion.py` implements item 1 below exactly as specified (top-M=8
trail proposals per corridor, each conditioning one DP run with a
`W_TRAIL·min(d/R_TRAIL,1)` unary and a chord-window-OR-60-px-trail-band
candidate pool; choice = DP path cost + W_GAP × shot-book penalty;
trail bridge on skipped frames as the `-F` arm; incumbent path on
abstain/no-proposal/no-path). Environment re-verified first: softdp.py
reproduced 376 @ 0.633 at W=25 to the hit, and the tune's INCUMBENT
line is that same number.

Pre-registered grid W_TRAIL {3,6,12,25} × R_TRAIL {8,16,30}, r6+r7 ×
prod+oracle, cross-fold p; rule = max total r@12 (`fus` arm) s.t.
pooled prec@12 ≥ incumbent 0.633, ties smallest W then R, none → dead.

| W_TRAIL | R=8 | R=16 | R=30 |
|---|---|---|---|
| 3 | 379 @ .618 | 381 @ .620 | 381 @ .620 |
| 6 | 366 @ .616 | 373 @ .619 | 372 @ .615 |
| 12 | 283 @ .558 | 304 @ .571 | 324 @ .579 |
| 25 | 258 @ .534 | 278 @ .549 | 293 @ .553 |

No cell clears precision; recall falls monotonically in W. **r9/r10
were NOT run** (`grade 9|10` refuses on a dead verdict). Train-only
autopsy (post-hoc, not a re-tune): spaghetti abstains/no-proposal on
20 of 36 train corridors; a W=0.001 arm (band pool on, cost off) beats
the best cost cell (r7 oracle 99 @ 0.57 vs 93 @ 0.54) — the trail's
content is WHERE TO LOOK, not where the ball is; the bridge is
negative everywhere (trail pixels at skipped frames sit > 12 px off);
M=1 ≡ M=8 within ±1; the displaced-trail confidence is degenerate
(always inf — the displaced band has no candidates). Nulls 0 on every
arm. Reading: item 2's corridor-endpoint/window geometry error is what
the trail inherits — the geometry fix precedes any fusion re-try, and
the only lead worth a FRESH pre-registration afterwards is the
pool-only shape. Full record: swing_explore_notes.md, "Fusion" entry
at the end of the pose-corridor chapter.

## Open questions answered in the last exchange (assessment, unbuilt)

**1. Combining spaghetti + emission + DP (owner: "not an ensemble").**
Fuse as ONE cost function, the DP search inside a trail search:
emission (1−p) + trail distance + smoothness + shot-book prior.
Depths: L1 = trail as a per-candidate DP term
`W_TRAIL·min(d_to_trail/R_TRAIL,1)` where spaghetti doesn't abstain,
plus flagged trail "bridge" points in DP-skipped frames; L2 = EM
alternation (poisoning-shaped feedback risk); L3 = joint objective:
screen shot-book proposals by support, run the top 6–10 trails each
through the DP, choose by DP path cost + shot-book prior, confidence =
best joint cost vs the same DP on a displaced window (truth-free).
Recommendation: build L3 as `fusion.py` (M=1 arm reported alongside),
W_TRAIL/R_TRAIL swept on r6/r7 cross-fold only, frozen rule, one-shot
r9/r10 vs the incumbent, displaced-anchor nulls. BUILT AND TUNED
2026-09-01 on the owner's go — DEAD, see the Fusion section above.

**2. The r10 "candidate deserts" 307.27–308.75 and 308.75–309.53
(should they be excluded as unretrievable-except-by-inference?).**
NO — and the diagnosis changed the question. Measured 2026-09-01:
- 307.27–308.75 (1.48 s): 34 owner clicks, **28 V** / 5 S / 1 N — the
  human saw the ball. It is a lob that leaves the frame top (y→2 at
  307.91, back at y=1 at 308.31: 0.40 s genuinely off-frame). Corridor
  endpoint A=(468,138) sits 83 px from the first click (421,207); the
  window (wx 94 / wy 137 around the chord) covers only 16/33 V+S
  clicks; support 10/90 frames. Both trackers 0/33.
- 308.75–309.53 (0.78 s): 23 clicks, **17 V** / 6 S. Endpoints
  A=(739,113) B=(767,108) → chord 28 px → smallest window (46×63)
  covering 3/23 clicks, while the ball actually drops from y≈190 to
  228, bounces (owner kind: bounce) and rises to 118. Spaghetti read
  the 28-px chord over 0.78 s as a "dink @ 18.7" — the geometry lied
  to it. Both trackers 0/23.
- So these are NOT detection deserts: 2 of 3 causes are CORRIDOR
  GEOMETRY (contact endpoint = nearest paddle to the decode, ~80–100 px
  from where the ball was struck; chord-box window too narrow for a
  lob/bounce excursion). Only the 0.4 s off-frame stretch is
  inference-only. (The lob region 307.85–308.35 was already an
  owner-flagged region in `r10_autopsy.py` ARM 4.)
- Scoring policy: keep every click in the denominator ("bars never
  loosen"); add a per-click STRATUM tag — off-frame / out-of-window /
  in-window-no-candidate / candidate-existed-selection-failed
  (`corridor_autopsy.py` already computes OUT / NOSIG / MERGED / TINY /
  OK) — and report r@12 by stratum. Grade the inference bridge on the
  off-frame stratum separately, with its own displaced null. "Excluded
  as unretrievable" would have hidden a fixable geometry bug.
- Fix candidates (tune on r6/r7 first, one-shot r9/r10): endpoint =
  ball position at contact (snap to the nearest candidate at the
  contact frame, or extend the window along the launch direction)
  instead of the paddle/wrist; window shaped by the shot family (lob →
  tall above the chord; bounce → down to the floor) instead of chord ±
  fixed box — the L3 fusion does this implicitly because the trail
  defines where to look.

**3. Contact-to-contact DURATION as a spaghetti filter ("it's not a
drive if contact 1 is 3 s from contact 2").** Yes, cheap and
legitimate. Today T enters only through the physics: direct-family
speed is fixed by T (`v0_between(pA,pB,T,k)`), bounce needs
ts+T2=T. There is NO explicit P(T | mode) prior, and the restitution
floor mu ≥ 0.05 is the loophole that lets a fast launch "die" at the
bounce and fill a long corridor. Add a soft prior on average speed =
path length / T per mode (or per shot-book neighbour), calibrated from
r6/r7 human segments only (`c["h_segs"]` / `c["hum"]` in the c3 cache)
or from physics alone — never from r9/r10 — and/or tighten the mu
floor; one-shot r9/r10 under the standard protocol. Caveat from item 2:
a chord-derived speed inherits endpoint error (the 28-px chord), so
apply it to path length, softly. It slots into the fusion's shot-book
prior term.

## Next-thread to-do (in order)

0. **Path-first is the incumbent** (section above; adopted 2026-09-01
   under `pathfirst_gate.md`). Its open edges, each a fresh
   pre-registration, tuned on r6/r7, one shot r9/r10, bars never
   loosen: (a) BOUNDARY TYPING — bounce vs contact at flight ends
   (r10 human ledger 13 bounces, path-first typed 0; the flights DO
   break there, the label does not fire — test the z/dt/dxy rule on
   the r6/r7 ledger first); (b) COVERAGE between flights — at-click
   points 616/779 on r9: short or slow flights below S_MIN / P_SEED,
   and the ~0.3 s holes between consecutive flights (e.g. r9
   258.0–258.4); (c) contact PRECISION — 64 boundaries vs 29 oracle
   contacts on r9; a boundary list is not a contact list until the
   fragment breaks are merged. Do not knob-turn pathfirst on r9/r10.
   (c) DONE 2026-09-02 — `events.py` v3 ADOPTED (owner's physical-
   distance pairing; events_gate.md has v1 FAIL / v2 DEAD / v3 PASS in
   order). Remaining misses are lost-track gaps → (b) is next.
   OWNER'S EYES (2026-09-02, overlay watched, see notes): the double
   labels are arrive+depart pairs at every contact; a lob is split at
   its apex and the seam labelled; the down-the-line near→far speedup
   is lost in BOTH rallies (fast + short, below the ~0.25 s seed
   span); one wrong-side-of-net depth read; NO false tracks seen.
   Order: (c) pair-merge + seam-glue first, then (b), then (a).
   2026-09-02 later: share cuts + `rally_stats.py` prototype + `rally_3d.py`
   viewer built (table above). Stats verdict: hits-per-player and
   last-shot work off the current stack; "who sped up first" does NOT
   (3D launch speed is depth-dominated). (b) stays next; then (a) — the
   stats prototype's paddle-distance rule is the natural candidate for
   (a)'s typing. Speed measure: `speed_lab.py` (train only) shows NO
   camera-visible measure separates fast/slow on r6/r7 (notes
   2026-09-02) — down-court speed is along the camera axis; drop it
   until there is a side camera or full-flight coverage.
   (b) SHOT AND SPENT 2026-09-02 as HAND-OFF SEEDING (`handoff.py`,
   `handoff_gate.md`): the door next to a flight end, near-body seeds
   allowed, short spans. Track clears on BOTH rallies (r9 563 @ 0.87,
   r10 446 @ 0.89, nulls 0/2, 0/1) but the v3 events layer on the new
   flights fails its r10 bar (.636 vs .645; r9 .758 passes) → NOT
   ADOPTED, incumbent stays. The mechanism is right and the layer
   above is the cost: five short flights = ten new flight ends the
   seam rule never saw. NEXT on this line, one pre-registration, not
   a knob turn: re-tune events (r6/r7) ON the hand-off track, bars =
   track bars as written + events F1 ≥ adopted − 0.03 on the SAME
   one-shot; both layers adopted together or neither. Needs the
   owner's go (a seal is a seal).
   (b′) GAP FILL BY ARC EXTENSION shot and spent 2026-09-02 on the
   owner's go (`gapfill.py`, `gapfill_gate.md`, table above): r9 PASS,
   r10 fails precision by 0.010 with events F1 UP on both → NOT
   ADOPTED. The inferred frames are ~60% right at 12 px; that is the
   number to design the next registration around: tagged inferred
   stratum with its own bar + displaced null, tracked frames must
   stay bit-identical, events bar as written. DONE same day as v2
   on the owner's go to re-use r9/r10 (disclosed in the gate): PASS
   both → ADOPTED AS A TAGGED PRODUCT, wired into the viewer, video
   and stats with the tag. r20/r21 = the clean re-check, bars as
   written. Next on the coverage line: the long gaps (>0.8 s) and the
   no-meet gaps (arcs never within 40 px — fragment ends) are what is
   left; those are tracker work (hand-off + events joint gate), not
   inference. (b″) HIT-ANCHORED FILL OF THOSE OPEN GAPS (owner's
   "LL hit it, UL hit it, the ball must travel between them" — v3,
   v3b registered in `gapfill_gate.md`, tuned on r6/r7 ONLY, r9/r10
   never loaded): DEAD on train twice (point anchor: paddle proxy
   80–200 px off the ball; region + joint meet refit: unsatisfiable,
   zero fills) and a train-only kink-rule sweep tops out at 7/41
   (`gapfill_explore3.py`). Root cause is in the gate: three of r7's
   four open gaps start on a flight whose LAST frame is 50–110 px off
   the ball (tracker tails on junk — a time-anchored kink cannot fix
   a wrong start), and r6's is a two-contact drive whose 34 streak
   frames want a pixel-space line between contact pixels timed to
   ~1 frame, not a 3D BVP. Line closed; v2 stays. If ever reopened:
   pixel-space fill + frame-accurate contact times, or tail cleanup
   in the tracker — each a fresh registration. Meanwhile: `click_setup.md` (r17
   train / r21 seal first; nine rallies staged) for the owner's clicking, and the
   `--bridge` demo cut exists for the share (cosmetic, captioned).

0-r17. **FIRST CLICK DELIVERY READ — r17, train only (2026-09-02,
   `train_read_r17.txt`).** Owner clicked 403 frames (269 V / 110 S /
   24 I), streak share 27 %. Pose: CPU rtmpose-balanced here (20.7 min;
   `r0017.npz` gitignored — re-extract or take the Colab npz);
   `ball_grade.py` train dry-run → `anchors_grade_r17.csv` (committed;
   that older harness read V 82 % / S 96 % at its own tolerance, CHECK 3
   FAIL, "MIDDLE" — diagnostic only, r17 is train). Incumbent path-first
   at the frozen cell, emission p-cache from the pooled r6/r7 model
   (out-of-fold for r17, so this read is honest): **245 / 379 @ 1.00**
   (V 169/269, S 76/110), nulls 0 / 4; gap fill v2 253 @ 0.94 (inferred
   17/112 @ 0.49, nulls 0/0); 14 flights; ONE wrong point in 379 — every
   other miss is a hole (r9 537/779 @ 0.87, r10 422/657 @ 0.88 for
   scale: same recall band, better precision). WHERE THE HOLES ARE, by
   the owner's contact labels (clicks within ±0.30 s): fast 88/113
   (S-fast 38/49), middle 97/142, return 8/14, serve 7/17, **slow
   45/93 (S-slow 1/17)**. Per contact: the three slow contacts at
   432.12 / 432.82 / 435.95 s cover 5/18, 5/13, 0/18 (the flight list
   has 1-s gaps at 431.97–432.98 and 435.37–436.28 — the tracker never
   seeds a flight there), the serve 7/17; the late fast exchange
   439.25 → 439.75 → 440.25 degrades 12 → 10 → 7 of 18. READING: the
   premise that picked r17 ("the hands battle the tracker keeps losing")
   is NOT what r17 shows — fast contacts are the best-covered bucket and
   fast streaks are 78 % covered already. The losses are dink / reset
   contacts next to a player and the serve, i.e. short slow flights
   the seeding never opens (S_MIN 6 / P_SEED 0.4 / near-body factor),
   the same shape as hand-off's target. CONSEQUENCE for the streak-
   detector question: on this evidence it is not the next instrument —
   the S-slow 1/17 is a proximity hole, not a blur hole. Next machine
   steps, in order, each needing labels or a go: (i) r18 / r19 / r21 /
   r20 land → same chain (pose here on CPU is fine); (ii) once ≥ 2 new
   train rallies exist, `emission.py train` with r17 in fold and
   `cache-cross` extended three-way (it is hard-wired to 6/7 today);
   (iii) a hand-off / seeding re-registration aimed at the slow
   near-body flights, tuned on r6 + r7 + r17, one shot r20/r21 on the
   owner's go. r9 / r10 untouched throughout.

0-r2to5. **SITTING 3 STAGED — r2 / r3 / r4 / r5 click tools + machine
   side (2026-09-02, owner: "set things up for 2, 3, 4 and 5 … fill out
   the training set").** These four have only PREFILL contact rows
   (pop-era types, approximate times), and the click package had them
   gated behind a contact pass. That gate was wrong: the ball tool needs
   a SPAN, not contacts. Frames pulled from the VOD put every prefill
   serve on the right rally (40.54 / 59.43 / 91.16 / 118.78 s; bug
   0-0 / 0-0 / 1-0 / 2-0), so `make_ball_audit.py --prefill-ok` spans
   first prefill − 1.5 s to last prefill + 3.0 s (wider margins than the
   manual-contact 1.0 / 2.0 because the prefill last-contact time is
   soft) → `ball_audit_r{2,3,4,5}.html` = 402 / 626 / 510 / 545 frames
   (t0 39.04 / 57.93 / 89.66 / 117.28). `cut_clip.py --stage` and
   `ball_replicate.py` take the same prefill fallback, clips cut at
   offsets 38.5 / 57.4 / 89.1 / 116.7 s, candidates committed, check_clip
   PASS d = 0 on all four, c3_lab WINDOWS + corridor_lab CLIPS added,
   pose npz via CPU rtmpose-balanced (gitignored; stamp
   `pose_meta_r2to5.json`). `train_read.py` falls back to prefill
   contacts with types folded slow/fast when a rally has no manual
   contacts (prints NOTE — the bucket read is approximate until the
   owner's contact pass). FINDING ON THE WAY: `pin_realignment.md`'s
   "replay of rally 3 at ~91 s" is wrong on video time — 91.3 s is rally
   4's LIVE serve (bug 1-0, near-right court; rally 3 serves near-left at
   59.5 s on 0-0), and the v4 rally windows are ONE RALLY LATE from row 3
   on (row 3 = rally 4 … row 5 = rally 6, matching the manual r6 serve at
   146.34). Addendum written in `pin_realignment.md`; nothing built on
   v4 windows is used for r2–r5. Optional owner contact pass: seek by
   TIME in the contact tool (40.5 / 59.4 / 91.2 / 118.8 s), not by pin.
   Click order r3 (fastest game-1 rally, 626 frames), r4, r2, r5;
   ≈ 2,080 frames total. Chain per rally once a CSV lands = the r17
   chain (`ball_grade.py` dry-run → `c3_lab.py N` → `emission.py cache N`
   → `train_read.py N`).

0a. **LEARNER: label-limited, not code-limited (2026-09-02,
   `learner_gate.md`).** Trees on the 14 features are DEAD at the
   tail bar with 359 positives and still climbing on the learning
   curve; the logistic is flat in labels. Next learner step = the
   owner's clicks (`click_setup.md`: r17 ✅, then r21, r18, r19, r20, and r3/r4/r2/r5 — all nine tools now built), then
   `python3 learner.py train` again under the same gate text; patch-
   appearance model after that. No knob here is worth turning before
   the labels exist.

0b. **SHOT SPEED — owner standing ask (2026-09-02, "don't forget speed
   history").** History: the rally-1 demo speeds (dinks 13–21 mph,
   attacks 25–48 mph, `shot_categories.py`) came from a HAND-labeled
   ball path anchored at HAND-labeled contact times = complete flights
   with known ends. No change removed that; the automatic tracker never
   had it (fragments, ends ≠ contacts, one-camera launch speed is
   depth-dominated — rally_stats named the wrong speed-up twice,
   speed_lab found no separating measure on r6/r7). Ways back, in
   order of honesty: (i) complete flights from the tracker — needs the
   hand-off + events joint gate above to pass; then flight = hit-to-hit
   and the shot_categories speed rule applies as it did on rally 1;
   (ii) DEMO-ONLY hybrid: anchor the tracked flights at the owner's
   labeled contact times (r9/r10 have them) and read speed over the
   full hit-to-hit interval — honest if captioned "contact times
   hand-labeled", never a grade, never fed back; (iii) hit-to-hit TIME
   alone (no 3D) as a proxy — separates on train, failed r10 label
   timings, not spent further. Do not publish a speed off a fragment.

1. ~~Corridor geometry fix (item 2 above).~~ DONE 2026-09-01, both
   shots: r9 CLEARS (479 @ 0.74 vs 431 @ 0.69), r10 does NOT (323 @
   0.65 vs 325 @ 0.69, one mis-merged corridor) → SPLIT, incumbent
   stays (section above). r10 clip re-cut from the owner's WebM
   (`cut_clip.py` + `check_clip.py`). NEXT registration on this line:
   gate the taller box on corridor quality (single-contact / short)
   and pair it with the pool-only trail; tuned r6/r7, fresh rule, one
   shot r9/r10. Do not knob-turn geom_fix on r9/r10.
2. Duration/average-speed prior in spaghetti (item 3).
3. ~~`fusion.py` L3 (item 1), only on the owner's go.~~ DONE 2026-09-01:
   built, tuned, DEAD under its frozen rule; r9/r10 unrun. Re-try only
   AFTER #1, as a pool-only variant under a fresh pre-registration.
   The strata now say selection is ~the whole residual, so the pool
   shape is the right target.
4. Product re-grade on the current stream (bounce ledger,
   eviction/trial, double-contact segment 301.02) — all were measured
   on the OLD stream; re-run before touching any other knob.
   (Items 1–4 are corridor-stack items; with path-first adopted they
   matter only if the corridor arm is ever needed again, e.g. as a
   between-flights fallback under item 0b.)
5. Carried, inactive: MVP-1 split awaits an owner-confirmed fresh
   registration; r20 seal only per adopted gate criteria; September
   obligation (score `model/registered_predictions.md` per its frozen
   method; update the pending entry in `model/receipts.json`).

PR #113 (`claude/vision-model-roadmap-vzu3i1`) carried the corridor
stack; PR #114 (`claude/fusion-model-building-kvl1bt`) carries fusion,
the geometry fix, the r10 re-cut and path-first.
