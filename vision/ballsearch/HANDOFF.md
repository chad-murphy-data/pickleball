# Ball-search thread — handoff (2026-09-01)

Dated status snapshot + next-thread to-do for the corridor ball-path
tracker. Written because the working thread was running out of context.
The instruments lived in a session scratchpad; they are now committed
HERE (`vision/ballsearch/`) so a fresh thread can run them. The
narrative record stays in `vision/swing_explore_notes.md` (chapter
"Pose-corridor ball re-search", line ~4506 onward: post-verdict
measurements, spaghetti, emission, soft-DP) and the per-channel ledger
in `vision/STATUS.md`; this file is the operational summary. Where they
disagree on a NUMBER, the notes file is the record.

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
