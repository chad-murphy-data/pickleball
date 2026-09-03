# Ball-search thread — roadmap

Written 2026-09-03, owner's ask: "I'm sort of lost on an overall
roadmap."

`vision/STATUS.md` answers **where are we** (per-channel ledger, every
number next to its null). `HANDOFF.md` answers **what do I run**.
`vision/STATS.md` answers **what comes out the end** — the graded
can-get / can't-get stat list and the ship order.
This file answers **where are we going, and how do we know when to
stop**. If it disagrees with STATUS.md on a number, STATUS.md wins.

Update the phase table when a phase closes. Do not update it to match
a mood.

## Where we are, with the receipt

Same rally, same owner clicks, different instruments, ~two weeks:

| instrument | r9 (779 clicks) | r10 (657 clicks) |
|---|---|---|
| corridor incumbent | 431 r@12 @ 0.69 prec | 325 @ 0.69 |
| + geometry fix | 479 @ 0.74 | 323 @ 0.65 (split → not adopted) |
| **path-first (incumbent)** | **537 @ 0.87** | **422 @ 0.88** |

Recall 55 % → 69 % and 49 % → 64 %, precision 0.69 → 0.88, every gain
booked as a one-shot after a written-down rule, displaced/time-shift
nulls ≤ 2/779. Layers on top: events v3 adopted (F1 .731 / .675 vs raw
.625 / .617, shift-null .30 / .33), gap-fill v2 adopted as a tagged
product, `rally_stats.py` hit counts within ±1 on **8 of 8**
player-rallies, r4's 3D reconstruction a CHECK 3 PASS at 1.76 ft
against a 3.0 ft bar.

The thread is working. That is not the problem.

## Why it stopped feeling like progress

The loop optimizes an **instrument**, and instruments have no finish
line — only a next digit. Nothing from this thread has reached the
website since the 3D replay, and the most recent *finding* was "the
checks disagree across rallies."

Two independent signals say tuning is near its end anyway:

- **r5 exposed a second failure mode** running opposite the first —
  fast balls lose to `yellow`, slow balls near bodies lose to `dbody`
  + `crowd`. One global tune cannot serve both. That argues for
  conditioning, not another knob.
- **`learner_gate.md` came back label-limited, not architecture-
  limited**: logistic is flat from 25 % to 100 % of labels
  (feature-saturated); trees only overtake at ≥ 75 %. The next input
  is clicks, not model code.

So the fix is not to try harder on the instrument. It is to spend what
the instrument already produces.

## The three phases (owner's sequence, 2026-09-03)

> "finish the current coding exercise, take a diversion to get some
> sort of publishable stats on the match we've been coding (derived,
> not from hand coding) then go back to the 3D sim"

| phase | goal | closes when |
|---|---|---|
| **1. Finish the coding** | spend the staged click package | sittings 1–3 delivered and read; seals still unspent |
| **2. Derived match stats** | one publishable stat from the tracker | a graded stat ships to the site with receipts |
| **3. Back to the 3D sim** | the showpiece | r4-class replay for a real rally, embedded |

Phase 2 is the one that converts two weeks of tracker work into
something that exists outside a gate file. Phase 1 feeds it. Phase 3
spends it.

---

## Phase 1 — finish the current coding exercise

Staged and committed in `click_setup.md`; nothing here needs new code.

Delivered: r17 (train), r2, r3, r4, r5 (train, sitting 3 complete).
Outstanding: **r18, r19** (train, ~319 frames combined — nearly free),
and the two seals **r20 / r21**, which stay unclicked-or-unspent until
Phase 2 has a reason to spend them.

Machine-side, before the next batch:

- **Make the emission refit n-way.** `emission.py cache-cross` is
  hard-wired to r6/r7 and structurally cannot absorb r2–r5 or r17.
  Until it does, six clicked train rallies are feeding a model fit on
  two. This is the single highest-value unbuilt thing in the thread
  and it is the direct answer to the learner being label-limited.
- Pose npz for r18–r21 (CPU here or Colab per `gpu_runbook.md`).

- **CHECK 1 on path-first: DONE 2026-09-03, passes on all seven train
  rallies** (pooled 78.1 % V vs a 70 % bar, of-claimed 99.6 % —
  `pf_check1.py`, detail in `HANDOFF.md`). The finding that drove it:
  the graded battery runs the DECODER stack, not the adopted
  incumbent, so the seal is registered against a stack we no longer
  use. The frozen scorer now lives in `vision/gate_checks.py` and both
  stacks call it.
- **CHECK 2 on path-first is BLOCKED on contact taps.** Its truth is
  manual taps at ±0.15 s and r2–r5 carry prefill only; of the clean
  train rallies only r17 clears the 14-contact power floor that r2's
  FAIL established. Owner ask recorded in `HANDOFF.md`: a timing pass
  on r3 / r4 / r5.

**Exit:** n-way emission refit landed and re-read on r6/r7 cross-fold;
r18/r19 clicked; CHECK 2 panel powered; no seal spent.

**Explicitly parked until a later phase asks:** the S-click rule
change (~500 genuine positives currently discarded — owner's call,
they have the number), a streak-elongation emission feature, the
scorer fix (interpolate + per-clip measured cut offset), and any
rescore of the r9/r10 incumbent numbers. Each is a graded re-run or a
rule change and needs an explicit go.

---

## Phase 2 — derived stats on chicago0725

**The hard rule that defines this phase:** the stat must be **derived
by the machine on rallies nobody clicked**. Owner clicks are the
*grader*, never the source. A number that needs a human to click the
ball first is not a stat, it is a transcription.

### The deliverable

**Touch share** — of the balls a team hit, what fraction each partner
hit — plus hits-per-player and last-shot location. STATUS.md already
carries touch share as "buildable now, glue unbuilt," and it is the
direct complement to the coverage model: coverage measures SPACE,
touch share measures BALLS, and the divergence is the "who is carrying
this team" question.

`rally_stats.py` already does the counting channel at ±1 on 8/8
player-rallies. What it is not yet: run at scale, graded at scale, or
attributed to names.

**Not in scope:** "who sped up first" is measured broken — 3D launch
speed is depth-dominated and fragment starts inflate it (wrong twice
of two), and `speed_lab.py` found no camera-visible measure that
separates fast from slow on r6/r7. Do not ship a speed number off this
footage without a side camera.

### Attribution is free

`data/vision/rally_windows_chicago0725_v4.csv` carries all 188 rallies
with `teamA_names` / `teamB_names`, division (125 mixed / 32 mens /
31 womens) and per-rally `server_uuid` + `receiver_uuid`, resolved
from referee logs. `vision/lineup.py` is a solved asset at 99.25 %
over 45,689 rallies. So real names cost nothing.

### Grading is *also* nearly free — the finding that makes this affordable

The obvious worry is that we only hold owner shot labels for r9/r10 —
8 player-rallies, far too thin to publish from. Two independent
validators fix that without a single new click:

1. **Shot 1 and shot 2 are known for every rally.** `server_uuid` and
   `receiver_uuid` are populated on **188 of 188** rallies. That is
   **376 free hitter labels** against which the attribution rule can be
   scored at n=188 instead of n=8, with a flat chance baseline (¼ for
   a four-way call, ½ once the side is known).
2. **Alternation is an internal validator.** Side alternation is EXACT
   in this footage (0 violations / 229 contacts) and the alternation
   decoder got shot counts 161/162. A derived hit sequence that fails
   to alternate sides is detectably wrong, on all 188 rallies, with no
   labels at all.

Both need verifying before they are leaned on — (1) assumes the
tracker's first two events land on the actual serve and return rather
than on a bounce, which is exactly the boundary-typing gap in
`HANDOFF.md` to-do (a). Verify on r6/r7 first, then read at scale.

### The cost, honestly

The chain per rally is clip → pose npz → candidate cache → emission
p-cache → path-first → events → stats. Pose is the bottleneck:
20.7 min/rally on CPU here, so 188 rallies ≈ **65 CPU-hours**. Colab
GPU per `gpu_runbook.md` is the practical route; clips cut in batch
from the owner's `full_match.mp4.webm` with `cut_clip.py`
(`check_clip.py` before trusting any re-cut).

A partial run is a legitimate product — the coverage model published
off 90 of 141 rallies and policed itself into withdrawing three of its
own findings. Pick the rally set by a rule written down first (e.g.
all womens + mens rallies, or every rally over N seconds), not by
which ones came out looking good.

**Exit:** a touch-share table for chicago0725, derived end-to-end,
graded against the 376 free labels and the alternation check, shipped
as a site block or an insight article with its error bars and its
null next to it. If it fails its bars, the phase still closes — the
write-up is a null result, which this project publishes.

---

## Phase 3 — back to the 3D sim

Already partly built: `rally_3d.py` → `court3d_r{N}.html`,
`render_court3d.py` → mp4, the rally-1 replay live in
`web/replay/pickles_replay.html`, r4 rendered and handed to design.

What Phase 2 buys it: r4 passes CHECK 3 at 1.76 ft while r3 and r5
fail at 4.84 and 4.60, and CHECK 3 fails on a *different component*
each rally (bounce over-generation, bounce under-generation, accuracy).
Picking a showpiece rally today means picking the one that happened to
fit. After Phase 2 there is a match-wide reconstruction to choose from
on a stated criterion.

Open before this ships as more than a demo: boundary TYPING (bounce vs
contact — `HANDOFF.md` to-do (a); r10's human ledger has 13 bounces,
path-first typed 0), and per-shot mph, which is blocked by the
standing rule that a speed is never published off a fragment.

**Exit:** a replay of a chosen rally, embedded, with the same
caveats-relaxed framing as the rally-1 page.

---

## Constraints that survive all three phases

Carried verbatim from `HANDOFF.md`; none of this roadmap relaxes any
of them.

- No graded re-run / seal consumption without explicit owner
  authorization. **r20 is the next seal, r21 the second.** Bars never
  loosen.
- TRAIN = r6 + r7 (+ r17, r2–r5 once the refit is n-way). r9 / r10
  owner clicks are EVALUATION-only: grading and autopsy, never
  production path selection, never tuning a weight or threshold. Any
  knob is tuned on r6/r7 cross-fold, the selection rule is written
  down BEFORE the numbers, then ONE shot.
- Positives ONLY from owner V clicks within R_POS=6 px; S clicks are
  ignore-zones (R_IGN=22), never positives; never train on tracker
  output or model self-labels.
- Oracle-bounds fits are diagnostic-only. Temporal-gate holdout rows
  (`data/vision/label_split.csv`) are untouchable; holdout burns on
  use.
- Rallies 22–34 are temporal-gate holdout and stay untouched at every
  level, including Phase 2's match-wide run.

## What this roadmap deliberately does not chase

The pre-registered seal on r20 — a full three-check PASS on a fresh
sealed rally — is the *scientific* finish line and it is not close:
CHECK 3 fails on a different component each rally. That is fine. It
accrues in the background off the same clicking Phase 1 already does,
and Phase 2 does not depend on it. Do not let it gate a publishable
stat, and do not spend a seal to chase one.
