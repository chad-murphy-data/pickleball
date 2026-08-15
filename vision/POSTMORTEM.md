# The vision program: what it was, what it cost, why it's closed

**Status: CLOSED, 2026-08-15. Both re-entry gates measured and shut.**
Read this first. `model/vision_adjudication.md` is the technical record and
is layered with reversals in the order they happened; this is the account
in order, with the numbers.

---

## What we were after

The referee logs are excellent and they stop at the same place every time.
They record who served, who received, who won the rally, and the score.
They never record what happened *inside* a rally — who hit what, where it
landed, how hard. There is no shot-level pro pickleball data anywhere in
this stack, and none for sale that we could reach. Broadcast video is the
only surface where that information exists at all.

So: point a camera pipeline at a match and recover the shots.

Everything below is measured on one matchup — MLP Chicago, 2026-07-25,
a condensed 720p YouTube VOD, 80 minutes of video for 107 minutes of wall
clock. That footnote turns out to be the conclusion, so it is worth
carrying from the start.

---

## What actually worked (and outlives the program)

Three of the four layers were solved, two of them convincingly.

| layer | result |
|---|---|
| rally ↔ video sync | 93%; broadcast edit depth recovered blind (27.0 min vs 26.9 actual) |
| court homography | 0.06 ft median residual, validated out of sample |
| player identity | **99.25%** over 45,689 rallies — from referee logs alone, no camera |
| ball / contacts | the wall |

**Player identity is a keeper and it needs no video at all.** Side-out
doubles is a state machine: partners swap halves exactly when their team
wins a rally while serving, and the receiver is the diagonal opponent.
Walk that over the referee log and its two names become all four players'
court halves at every serve. 45,348 of 45,689 receivers correct across
1,218 MLP 2026 matches. `vision/lineup.py`. This was free the whole time
and is useful independent of any camera work.

The scorebug state reader (`vision/scorebug_read.py`) also survives as a
working instrument, and the sync work proved the flip alignment is
frame-exact — which keeps point-by-point OCR backfill viable for
*unlogged* matches, low value while `getListLogs` covers 2026.

---

## Gate B — swings as a proxy for contacts

**Idea (the user's).** Skip the ball. Players stand still in ready
position and swing when they hit, so find the swings. Pose estimation is
in-domain in a way tennis-ball trackers are not: humans are the
best-covered class in computer vision, and a wrist arc is 40–80 px/frame
where the ball is ~9 px.

**Pre-registered before anything was built:** 75% overall recall, 60% on
the fast stratum, 90% precision; kill below 60% overall; junk-kill if
side-alternation fell under 0.45. Stage 1 chose the operating point
label-free; stage 2 touched the 203 hand labels exactly once.

**First run: 46.7% recall. Kill.**

**Then the verdict was suspended** — the user, while labeling, noticed the
tool's links were landing up to 40 seconds off. The windows telling the
probe *where to look* were wrong for an unknown subset of rallies. This
matters more than it sounds: a 40-second error lands on players walking
between points, which produces exactly the junk signature a failed
detector produces. **A broken measurement frame and a broken detector are
indistinguishable in every label-free diagnostic.** The kill was not
retracted as wrong; it was retracted as *unmeasured*.

**The fix was human.** Machine alignment had failed twice — a cheer-based
join, then a scorebug-grammar chain that mis-anchored in early game 1,
precisely where the labels live. The user hand-marked the serve instant of
all 16 labeled rallies in the audit tool. Marks and labels from the same
video: airtight by construction. Twenty minutes of clicking.

**Re-gate on the sound frame:**

| | broken frame | hand-pinned frame |
|---|---|---|
| precision | 0.721 | **0.871** |
| overall recall | 0.467 | 0.442 |
| fast recall | 0.475 (n=61) | 0.469 (n=81) |
| alternation | 0.460 | 0.448 |

Precision moved 15 points — exactly what un-breaking the spans should do,
and independent proof the frame really was broken. **Recall did not move.**
The suspension was justified and the answer on the deciding axis was
unchanged.

It is a clean kill rather than a near miss. Sweeping every threshold, the
precision/recall curve runs from (74%, 65%) to (44%, 87%); the required
corner (75%, 90%) is nowhere on it. Alternation tops out at 0.448 across
all 34 feasible operating points — under the 0.45 junk line, under the
0.50 two-sided chance rate, far under the 0.608 its own measured recall
implies. The degraded "touch-share only" fallback needed fast recall
40–60% *with* consistency; recall qualified, consistency failed.

**Autopsy.** The audio half was a *premise* failure, not a tuning failure:
per-rally pop counts are uncorrelated with true shot counts at every
threshold (r ≈ 0.0–0.2). The POC's "broadband stripes at shot cadence" was
an eyeball over-read of a spectrogram. The pose half retains count-level
signal (partial r ≈ +0.5–0.6 with shot counts beyond duration), which is
why the channel was worth testing and why it still isn't enough.

---

## Gate A — the ball, via hand labels

The pre-registered plan: hand-label 300–500 speed-stratified frames,
fine-tune a tennis ball tracker, score recall on the fast stratum. Below
0.8 → the wall is physical, dead at any label budget. Above 0.9 → the
entire built stack revives.

**Step 1 answered it, and step 2 never ran.**

The user labeled all 416 sampled frames with a 5× magnifier, blink-compare
against the previous frame, and no time limit — and could not locate the
ball in 48% of them.

Then came the correction that makes the number trustworthy, from the
labeler: *"a decent chunk of the can't-find were between points."* True,
and traceable to a specific defect: windows for rallies 17+ are built as
`t0 = t1 − duration`, and the referee log's duration column includes a
~6 s pre-serve lead (measured on the 16 hand-marked rallies). The 1.2 s
head-pad barely dented it. Binning by seconds-since-window-open separates
dead time from live play, and the curve is exactly the shape that
explanation predicts:

```
 0-3 s   14%  #####
 3-6 s   22%  ########
 6-10 s  58%  #######################
10-16 s  70%  ###########################
16+  s   63%  #########################
```

**In-play human findability: 64.1%, 95% CI [59%, 69%], n = 306.**
Reproduce with `vision/ball_visibility.py`.

The entire interval sits below the 0.8 kill line. Two things make it
generous rather than harsh: the `fast` stratum proxy (division × tempo)
produced no detectable difficulty split (z = −0.62 vs random), so this is
*overall* findability and true fast-shot frames are likely worse; and some
misses are the broadcast framing the ball out, already on record as a bias
that binds even a perfect detector.

**Why human findability closes the program rather than merely describing
it.** Ground truth exists only where a human can see the ball. On 36% of
in-play frames there is nothing to train on and nothing to score against,
so a model's claims there are unfalsifiable — which is precisely the
failure that poisoned the earlier auto-label fine-tune (42% of predictions
piled into the kitchen band vs 14% at baseline). Grant a perfect detector
on every findable frame and ~64% of ball positions come back, with the
misses concentrated where the ball is fast or occluded: at the contacts,
which is the thing the program wanted.

Running the fine-tune would have produced a flattering number on the easy
64% and silence on the rest. That is why it was skipped.

---

## The only door left

This measures **a condensed 720p YouTube VOD**. Higher-resolution or
uncondensed source would need re-measuring; nothing else would. Not more
labels, not more compute, not a better architecture. Different video.

Two structural problems survive even that: broadcasts are condensed and
exist mainly for championship courts, so the sample bias is permanent; and
the freeze-out question stays n = 4 at any tracking quality.

---

## Assets that survive

- `vision/lineup.py` — 99.25% player identity from referee logs, no camera
- `vision/court.py` — homography at 0.06 ft, if video work ever resumes
- `vision/scorebug_read.py` — per-second scorebug state reader (v6)
- `data/vision/shot_labels_chicago0725.csv` — **203 hand-labeled shots**
  across 16 rallies with types and notes; the project's only shot-level
  ground truth and the answer key for any future instrument or any
  hand-coded data we ever buy
- `data/vision/ball_labels_chicago0725.csv` — **416 frames**: 217 verified
  ball positions, 199 verified not-findable. The second half is the part
  that closed the question
- `vision/ball_flipbook.html`, `vision/make_shot_audit.py` — labeling tools
  that work, if a future footage source justifies relabeling

---

## Lessons worth carrying out of here

**1. A measurement-frame bug and a detector failure are observationally
identical.** Every label-free diagnostic — alternation, contact rate,
interval histograms — reads the same whether the detector is blind or the
clock is wrong. This cost two sessions. When windows are machine-derived,
validate them *on the cases being scored*, not on a convenient sample: a
±1 s check covering 10 anchor rallies was generalized to 191 and hid a 40 s
error.

**2. Ask the human where the machine is guessing.** The alignment problem
was attacked three times by machine and solved in twenty minutes of
clicking. The scoreboard's serve-dot grammar — one dot on the side switch,
a second at X-Y-2 — was thirty seconds of domain knowledge that unlocked
the whole scorebug reader after frame-differencing had failed. Both were
cheaper than any algorithm considered.

**3. The labeler's offhand remarks are data.** "This was deceptively hard"
became the measurement that closed Gate A. "A decent chunk were between
points" moved the headline number 12 points and is the reason it can be
trusted. Neither was a request for help.

**4. Never train on labels a machine produced.** Stated twice in this
program's history and confirmed twice. Using a detector to *choose* frames
for a human to label is active learning and is fine; using it to *generate*
labels inherits its bias and amplifies it.

**5. Pre-registration is what makes a negative result publishable.** Both
kills were scored against thresholds written before the tooling existed.
Every number in this document could otherwise have been argued down, and
in a hobby project with no adversary, the only thing stopping that is a
bar set in advance.

**6. Know what a failure costs before you buy it.** Both gates were
designed so failure was bounded and informative. Gate A cost one evening of
clicking and returned "the wall is physical, said with a measurement" —
which is worth considerably more than an open question that keeps
re-attracting hobby hours.
