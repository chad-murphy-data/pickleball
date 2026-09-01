
## Pose-corridor ball re-search (2026-09-01, owner-approved item #5) — instrument ladder, autopsy, and the "humans aren't the ball" filter

Goal (product): recover the ball observations MISSING between known
contacts — r10's check-3 ledger reads 8 tracked bounces vs the
owner's 13 because the far-court dink observations never made the
stream — without training anything (the killed fine-tune stays
killed; hand-specified appearance and pose-derived geometry are
allowed, learned parameters are not). Truth = the owner's
hand-clicked ball paths (data/vision/ball_path_r{9,10}.csv, 30 fps,
V/S/I/N visibility codes; V+S rows are scoreable). Every instrument
graded with displaced-anchor nulls (corridors shifted ~200 px).
Metric: r@12 = clicks within 12 px, prec@12 = fraction of at-click
track points within 12 px, ADDED@12 = clicks the corridor recovers
that the production decode missed.

**Instrument ladder** (r10 production-contact corridors; decode
baseline 405/657 clicks @12):
- v1 per-frame windows on the A->B interpolant: established the hole
  geometry (252/657 clicks with no decode point within 12 px).
- v2 greedy bidirectional chains (the owner's manual-scan heuristic
  mechanized): arm-hijack — chains seed on the hitter's arm and
  follow it; meet-in-middle agreement too sparse to matter.
- v3 global DP per corridor (corridor_dp.py): Viterbi over per-frame
  candidates, velocity-change (accel) cost, gap penalties, soft
  anchoring to both paddle endpoints. cc candidates, K=14:
  r@12 205, prec 0.36, ADDED 60.  Nulls 0.
- A/B one-sided variants (anchor at paddle A only / B only) as a
  two-sided independence gate for corridors the decode cannot vouch
  for: agreement of two DP passes that share no anchor. First
  product run: ab kept 0 — probe re-diagnosis (corridor_probe_ab.py)
  showed NO bug and only ONE truly unjudged corridor (307.27-308.75 s,
  a candidate desert where even one-sided paths are empty); the other
  11 non-injecting corridors were judged-and-REJECTED on decode
  disagreement. That conflation (unjudged vs rejected) is what the
  --abo arbitration below fixes.

**Decode-hole autopsy** (corridor_autopsy.py; answers the owner's
"how does a ball I can see produce no candidate?"): of r10's 252
holes, ZERO lack local +/-2-frame motion signal — the slow-ball /
far-court-parallax hypothesis is DEAD (a +/-6-frame differencer and a
chroma differencer, tried as counterfactuals, buy +9/252 between
them). The real failure modes: MERGED/DISPLACED 138/252 (55% — ball
signal absorbed into a player-sized blob, median area 1572 px^2 vs
AREA_MAX 600, centroid ~23 px from the click), OUT-OF-WINDOW 28
(11% — fast balls, median 8.7 px/frame, outside the corridor window),
SIGNAL-OK 86 (34% — a candidate existed and selection failed).
Counterfactual candidate emitter (NMS local maxima instead of
connected components): hole coverage 98/252 -> 236/252 (94%) on r10,
72/202 -> 182/202 (90%) on r9 — at the price of ~70 junk peaks per
patch. Lesson bought cheap this time: measure the mechanism before
building the fix — a slow-mode/chroma build would have been wasted.

**Visibility-state taxonomy** (owner-proposed three states, graded
against their own clicks). Fractions: r10 V (plainly visible) 73%,
V+S (visible + hidden-but-inferrable) 98%; r9 V 66%, V+S 88%.
INVERSION: the decoder misses VISIBLE clicks at a higher rate than
hidden ones (r10 42% of V vs 28% of S; r9 28% vs 20%) — 81% of r10's
hole mass is state 1. So the opportunity was never "infer through
occlusion", it is "stop losing the ball you can see" (= the merging
failure above). S runs are short: median 3 clicks (~6 frames, inside
the DP's GAP=6), p90 8, max 15 — occlusion bridging is within reach
of the existing gap mechanic. State 3b (off-frame, N-coded) runs long
only on lobs (r9: runs of 21 and 34 clicks) and is harmless to the
ledger: no bounce happens off-frame, and the corridor re-anchors at
the next contact.

**Scope correction to the old 64% in-play findability finding**
(2026-08-15, ball_visibility.py). The owner's clicks REPLICATE it —
as isolated-frame findability: V fraction 66-73% matches 64%. But a
corridor instrument integrates over time, so its ceiling is V+S =
88-98%, not 64%. The old finding stays retained for exactly what it
killed — training a single-frame detector on labels that cannot be
falsified in the invisible fraction — and licenses nothing about
trajectory-integrating instruments. (Owner directive this session:
do not treat 64% as binding on corridor work; this is the measured
reconciliation.)

**Appearance-peak emitter: built, measured, RETIRED.** NMS peaks of
the motion image ranked by hand-built ball appearance (white tophat
9x9 — bright blob OR thin streak both survive — + yellowness
(R+G)/2-B + motion), cap 600/frame, no training. WORSE than cc blobs
everywhere: r10 prod 147/0.24 (vs cc 205/0.36), with body filter
123/0.21 (vs cc+body 282/0.51). Mechanism: appearance ranking
promotes PERSISTENT bright smooth-moving extremities — shoes,
wristbands — which the accel cost cannot reject because they move
smoothly; appearance and smoothness stopped being independent
filters. The candidate-coverage problem the autopsy found is real,
but this emitter buys coverage with exactly the junk that defeats
the selector.

**"Humans aren't the ball" (owner-specced): the instrument win of
the thread.** Soft DP cost from the pose skeletons already in hand:
candidates near extremity keypoints (elbows/wrists/knees/ankles,
kpc>=0.3, ALL tracks including refs) pay W_BODY=25 * max(0, 1-d/16px).
Soft, not a veto, and interior frames only — the ball legitimately
sits at the hitter's wrist at contact (endpoint frames exempt) and
passes near far-court feet in 2D on kitchen descents. r10 cc+body:
**r@12 282, prec 0.51, ADDED@12 82** (from 205/0.36/60 without);
S-click precision 0.58. Generalizes: r9 cc+body prod r@12 388,
prec 0.55, ADDED@12 32 (oracle 369/0.56) — precision holds on a
second rally with different geometry, nulls ~0.
Null battery: 0 everywhere except one cc+body oracle-null arm at
13/657 (prec 0.18) — the displaced corridor still overlaps the real
flight in places, i.e. the filter is good enough to find the actual
ball from a wrong window; real:null ~ 20:1. Watch it, not alarming.

**Product test — injection into the r10 check-3 fit** (corridor_fit.py
--body --ab --abo --targeted). Three gates arbitrate what enters:
(1) decode-shadow vouching (DP path within 10 px of the existing
decode over >=5 frames — label-free); (2) A/B consensus for corridors
the decode cannot judge; (3) --abo: A/B consensus OVERRIDES a
judged-but-disagreeing sparse decode — licensed by the autopsy
finding that decode points inside merged regions are arm centroids
~23 px off, so two anchor-independent DP passes agreeing within
12 px outrank them. Injection restricted to segments holding a split
candidate (an approach event far from every bound); owner truth used
only to GRADE the injected points, never to select them.

RESULT — the arbitration WORKED and the ledger did NOT move:
27 corridors -> 17 shadow-vouched + 7 consensus (previous best
7 + 0), 41 obs injected into the 6 split-candidate segments,
med-vs-truth 4.5 px (ab-only 3.1 px — the points the old pipeline
threw away on the word of an arm-centroid decode were the MOST
accurate). Fit: matched 18/26, med 2.31 ft (unchanged), **0 splits
accepted**, bounces 7v13 (baseline 8v13 — one segment tipped out of
the plausible set, taking a crossing 16/17 -> 15/16 and a bounce
with it). Bar (tracked >= 12 bounces, med <= 3.0 ft): FAIL. Some
individual impact reads improved (317.37: 0.72 -> 0.37 ft; 304.67:
1.50 -> 0.89), so the injected points are real signal — they just
could not move the fits.

**Truth-fit observability autopsy (corridor_truthfit.py — why 41
accurate points changed nothing).** For each of the 6 candidate
segments, fit_segment run on (a) the resident visited stream and
(b) the owner's clicks alone — the best observations that can ever
exist. Diagnostic use of labels only. The FAIL decomposes cleanly,
and the truth-fit AGREES WITH THE HUMAN LEDGER segment by segment:
- All 6 resident fits are ok=False -> they contribute ZERO bounces,
  which balances the books exactly: the human has ~5 bounces in
  these spans, and the ledger reads 8v13.
- 2/6 are FALSE ALARMS (303.00-304.00, 317.37-318.45): under
  clicks the fit is a clean single arc (1.47 / 1.89 px, 0 bounces)
  — and the human ledger has ARC segments there too. The approach
  events at 303.63 / 317.80 proposed contacts that don't exist;
  the split guards (margin, plausibility) correctly refused them
  even under perfect obs. The machinery's guards WORK.
- 3/6 are WINNABLE with truth-grade obs: 304.67-305.37 fits ok
  with its bounce (+1); 310.22-311.48 splits at 311.03 vs the
  human's contact at 311.07, rms 8.54 -> 1.00 px (+1); 313.95-
  315.22 splits at 314.27 vs the human's 314.32 (+1). Perfect-obs
  ceiling via this lever: 8+3 = 11 — still under the 12 bar.
- 1/6 (301.02-302.73) needs TWO contacts the approach channel
  never proposed (human bounds at 301.32 and 302.34); one split
  cannot fix it and no candidate exists for the other. That is
  the remaining +2 that reaches 13.
WHY the injection failed while clicks succeed: the injection is
ADDITIVE-ONLY — it bridges empty frames but never evicts the
corrupt resident observations (the merged-blob arm centroids the
autopsy measured at ~23 px off), and 41 points cannot outvote
~240 residents. The corruption shows in both directions: seg
303.00 reads rms 7.76 on residents vs 1.47 on truth (junk breaks
a clean flight), seg 304.67 reads 1.74 on residents vs 3.48 on
truth (junk fakes cleanliness and hides a real bounce).

**Corridor-thread verdict.** The INSTRUMENT is the best ball-
position source this footage has ever had — r@12 282-388, prec
0.51-0.55 across two rallies, nulls ~0, injected accuracy 4.5 px,
and a vouching stack (shadow / A-B consensus / consensus-override)
that is fully label-free. The PRODUCT hypothesis — that restoring
missing observations moves the r10 bounce ledger — is FALSIFIED as
posed: the binding failures are corrupt resident observations and
one segment with two undetected contacts, not missing points.
Bar not met; r10 check-3 stays FAIL (18/26, med 2.31 ft, 7v13 in
the injected arm). Next levers, in order of license already earned:
(1) EVICTION, not just bridging — inside a vouched corridor,
resident visited points far from the DP path are the measured arm
centroids; replace them under the same label-free gates, which is
exactly the move the truth-fit says converts 310.22/313.95/304.67
(+3 -> 11). (2) A second contact channel for the 301.32-type miss
(the DP paths themselves kink at contacts — turn-in-corridor as
contact evidence), worth +2 more only if (1) lands. (3) The r9
fit arm stays closed (stream/readout floor on fast drives, oracle
fails the med bar there); r6 stays gated on decode health. None of
this touches MVP-1 (checks 1+2), which is unaffected by the
bounce-ledger research line.
