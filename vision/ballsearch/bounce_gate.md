# Bounce chain on r20/r21 — pre-registration (frozen 2026-09-08, before any r20/r21 number)

Owner's authorization (2026-09-08): "I think we can go ahead and spend
R20 and 21 and see what we get."  r20 and r21 are the last two train
rallies of game 1 that have never been read by any instrument; they have
contact taps and bounce taps (owner's labels) but NO ball clicks, so
there is no human ball path and no track-level check possible.  This is
a one-shot read of the BOUNCE CHAIN (contact bounds -> arc fit -> bounce
typing) graded on the owner's taps only.  r9/r10 already carry one read
of the leaky claimer (tau 0.45; HANDOFF 09-05) and are not re-read.
Rallies 22+ are the temporal-gate holdout and stay untouched.

## What is being decided

Two changes to the bounce chain, adopted TOGETHER or NEITHER:

1. CONTACT BOUNDS from the leak-free claimer (`claimer.py --no-end-feats`,
   model = HistGradientBoosting on 24 features, no t_from_serve /
   t_to_end; trained on ALL SEVEN train rallies r2-r7 + r17 with the
   TAPS key; tau fixed by the train rule) instead of the shipped
   corridor claim, and NO crossing demotion (the claimer already
   drops most of what demotion was for; train shows demotion=none
   keeps more contacts).
2. AT-FEET TYPING (`tap_grade.feet_type`): an ok ARC flight whose fitted
   arc dips to z <= Z ft within the last W s before its end is typed a
   bounce at the first floor crossing.  This targets the fitter's blind
   spot — `fit_segment` only searches (t0+0.12, t1-0.12), so a half-volley
   bounce within ~0.10 s of the next contact is always typed ARC; 13 of
   109 non-terminal bounces in the labels (12 %) are that kind.  The
   (W, Z) cell is chosen by the frozen rule in `tap_grade.tune_feet`
   (max pooled matched-false over train vs no typing; ties -> smaller W,
   then smaller Z; LIVE only if net >= no-typing net + 2; result in
   `tap_feet_tune.json`).  If the train rule says DEAD, the candidate
   arm runs WITHOUT typing and only change 1 is on the line.

## Arms (all fed the same tracked obs, `bound_oracle.predem(c, "raw")`)

- INCUMBENT: shipped corridor claim + shipped crossing demotion
  (`bounce_autopsy.tracked`), no typing.  What HANDOFF calls the shipped
  bounce count.
- CANDIDATE: claimer_ne bounds + demotion none + at-feet rule (if LIVE).
- Diagnostics, reported but not graded: claimer_ne/none without typing;
  incumbent + at-feet rule.  They say WHICH change carried the result.

## Grade (`tap_grade.py grade`, one number set per arm)

Truth = the owner's non-terminal bounce taps (terminal flights excluded:
the machine's end-of-rally cut is not what the owner timed).  A machine
bounce MATCHES a tap within 0.30 s on the same flight (greedy, one per
tap).  A machine bounce is FALSE if it sits on a volley/nobounce flight
or on a bounce flight more than 0.30 s from the tap.  Also read:
landing error (ft, floor homography) for matched bounces, contacts
matched within 0.25 s, junk bounds, intact flights keyed on taps.

## Bars (one shot on r20+r21 POOLED; no re-tune, no knob override, no second run)

Pooled over r20 and r21.  Let net = matched - false.

1. net(candidate) >= net(incumbent) + 2.
2. contacts matched (candidate) >= contacts matched (incumbent) - 1.
3. junk SHARE of bounds (junk / bounds) for the candidate <= the
   incumbent's.  [Amended before the freeze, on train evidence: the
   draft said junk COUNT, which mechanically punishes running without
   demotion (demotion exists to delete bounds).  On train the count
   bar fails by 4 (41 vs 37) while the share is equal (33 % of 126 vs
   34 % of 110).  The worry behind the bar is a candidate that wins
   bounces by spraying bounds; the share answers that, the count does
   not.  Recorded here so the amendment is visible.]
4. intact flights (candidate) >= intact (incumbent).
5. NULL: shift every candidate bounce call by +0.75 s and re-grade;
   the shifted matched count must be <= half the real matched count.
   Fails -> the matches are rhythm, not placement, and the read is void
   regardless of bars 1-4.

All five hold -> ADOPT both changes as the shipped chain (HANDOFF "what
is built" row + STATS).  Bars 2-4 hold but 1 fails -> the leak-free
claimer is a wash on bounces; keep it as a diagnostic, shipped chain
unchanged.  Any of 2-4 fails -> the candidate is REJECTED, no revisit
without new labeled rallies (r18/r19 once poses exist are TRAIN, not a
second shot).  The bars never loosen after the numbers are seen.

## Nulls and leaks

- The claimer never sees r20/r21 (train = r2-r7, r17; the taps key uses
  only train taps).  The at-feet cell is tuned on train only.
- `t_to_end` was a leak (rally end = last contact + 2 s when no
  point_dead label; true for r20 and r21) — that is why the model is
  the `_ne` one.
- Pose npz for r20/r21 are extracted fresh (rtmpose-balanced, native
  fps) and used only for anchors (`ball_grade.make_anchors`) and the
  contact candidates the claimer scores; the owner's taps are never an
  input to any fit on r20/r21.
- The shift null (bar 5) is the only null; there are 19 taps on r20/r21
  so the read is coarse — a +2 net margin is about one flight per rally.

## Discipline

- `tap_grade.check_seal` refuses r>=22 always; r20/r21 need `--seal`
  AND the literal `VERDICT: LIVE` line in this file.  The line is added
  in the freeze commit, after the train evidence below is filled in and
  before any r20/r21 number exists.
- Each r20/r21 fit and grade runs ONCE; caches (`tapgrade_*_r20/21.pkl`,
  `claimer_bounds_ne_r20/21.json`) are the record.
- Results are appended under "Results" and never edited above this line
  afterwards.

## Train evidence (r2-r7 + r17; looked at BEFORE the freeze, allowed)

At-feet rule (`tune-feet --bounds claimer_ne --demotion none`, result in
`tap_feet_tune.json`): **DEAD**.  No typing: 16/35 matched, 15 false,
net +1.  Best cell W 0.15 / Z 1.0: 22/35 matched (at-feet taps 4/5 vs
1/5) but 23 false, net -1.  Every cell in the grid loses net; the rule
finds the half-volleys and pays for them twice over in false calls on
volley flights.  So the CANDIDATE arm on r20/r21 is claimer_ne bounds +
demotion none, NO typing; only change 1 is on the line.

Candidate on train (claimer_ne/none, LORO bounds, no typing):
16/35 matched, 15 false, land 1.6 ft, contacts 73/89, junk 41, intact
19/35.

Incumbent on train (shipped/shipped, same grader): 13/35 matched,
21 false, land 2.0 ft, contacts 65/89, junk 37 of 110 bounds, intact
17/35, ok arc fits 42 (candidate 88).

Train scorecard against the bars, candidate vs incumbent: net +1 vs
-8 (bar 1 pass by 9); contacts 73 vs 65 (bar 2 pass); junk share 0.33
vs 0.34 (bar 3 pass; count would fail); intact 19 vs 17 (bar 4 pass);
shift null +0.75 s: candidate 2/35 vs 16 real, incumbent 5/35 vs 13
real (bar 5 pass both).  Diagnostic incumbent + at-feet: 14/35, 25
false — the rule hurts there too.

Two rallies of train are drawn on for this read's expectation: r6 is
0/3 on both arms (a short rally with no bounce recovered), r4 is 1/4
vs 0/4.  With 19 taps on r20/r21 the bars are coarse; a +2 net margin
is about one flight per rally.

VERDICT: LIVE — the gate is frozen; r20/r21 may be run once.

## Results (appended after the one shot; nothing above this line changes)

