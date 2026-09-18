# Ball-search thread — roadmap

Written 2026-09-03 ("I'm sort of lost on an overall roadmap");
**rewritten the same evening**, because three of its premises changed in
one day. What changed is recorded at the bottom under "What this roadmap
used to say".

`vision/STATUS.md` answers **where are we** (per-channel ledger, every
number next to its null). `HANDOFF.md` answers **what do I run**.
`vision/STATS.md` answers **what comes out the end** — the graded
can-get / can't-get stat list, the ship order, and the house rules.
This file answers **where are we going, and how do we know when to stop**.
If it disagrees with STATUS.md on a number, STATUS.md wins.

Update the phase table when a phase closes. Do not update it to match a
mood.

## Where we are, with the receipt

Same rally, same owner clicks, different instruments, ~two weeks:

| instrument | r9 (779 clicks) | r10 (657 clicks) |
|---|---|---|
| corridor incumbent | 431 r@12 @ 0.69 prec | 325 @ 0.69 |
| + geometry fix | 479 @ 0.74 | 323 @ 0.65 (split → not adopted) |
| **path-first (incumbent)** | **537 @ 0.87** | **422 @ 0.88** |

Recall 55% → 69% and 49% → 64%, precision 0.69 → 0.88, every gain booked
as a one-shot after a written-down rule, displaced/time-shift nulls
≤ 2/779. Layers on top: events v3 adopted (F1 .731 / .675), gap-fill v2
adopted as a tagged product, `rally_stats.py` hit counts within ±1 on
**8 of 8** player-rallies, r4's 3D reconstruction a CHECK 3 PASS at
1.76 ft against a 3.0 ft bar.

And, as of today, the tracker's error is **located**: it is a 90%
instrument mid-flight and a 72% one inside 0.2 s of a contact
(`blackhole.py`). That is not a worse result than we thought — it is the
same result with a map.

## The three things that changed on 2026-09-03

1. **Clicking is no longer the input to the learner.** Both label axes are
   measured flat (`label_curve.py`): pooling 1 → 6 train rallies moves AUC
   +0.008 and the operational metric not at all. The old plan's "the next
   input is clicks, not model code" is dead. Clicks are now for GRADING and
   for PANEL SIZE.
2. **Speed came back.** It was excluded from Phase 2 as measured-broken.
   That verdict was about the 3D launch fit off the ball's own arc (AUC
   0.10, inverted). Speed off player geometry + contact timing scores AUC
   0.829 and is shipped in `rally_stats.py`. Owner call: average speed is
   the speed. See STATS.md's house rule.
3. **A whole stat family stopped needing the ball at the contact.** Player
   positions (exact plane) plus contact times (better than human) place a
   bounce to 5.3 ft with no ball tracking at all — 2× worse than the
   tracked fit and available on every rally today, skipping candidates,
   emission cache, path-first and events entirely.

Together those move the centre of gravity from *tune the tracker* to
*spend what the tracker already produces*, which was Phase 2's whole point.
Phase 2 just got much cheaper.

## The three phases (owner's sequence, unchanged)

> "finish the current coding exercise, take a diversion to get some sort of
> publishable stats on the match we've been coding (derived, not from hand
> coding) then go back to the 3D sim"

| phase | goal | closes when |
|---|---|---|
| **1. Finish the coding** | spend the staged click package | contact taps in, r18/r19 clicked, seals unspent |
| **2. Derived match stats** | one publishable stat from the tracker | a graded stat ships to the site with receipts |
| **3. Back to the 3D sim** | the showpiece | r4-class replay for a real rally, embedded |

---

## Phase 1 — finish the current coding exercise

### Tonight's click, in priority order

**0. The 58 contact taps are already done — do not re-tap them.** This
section previously opened with a 45-minute tap pass on r2-r5. That job did
not exist: every row of `contact_labels_chicago0725.csv` already carries an
owner tap, including all four of those rallies. The `source` column says
HOW a tap was entered (⏎ against the prefilled hitter/type = `prefill`, an
explicit 1-4 key = `manual`), not whether one exists. Three instruments had
filtered `source == "prefill"` out and were running on roughly half their
available panel; fixed 2026-09-03, and folding r2-r5 back in delivered
exactly what the tap pass was forecast to deliver (bounces 36 → 57,
flights 63 → 111, blackhole frames 2,259 → 3,804) for a one-line change.

**1. BALL PATHS on r18 + r19 — ~319 frames combined.** Now the top of the
list. Tools committed (`data/vision/ball_audit_r18.html`, `_r19.html`).
Both already have owner contact taps (3 and 2 shots), so a path turns them
into full LEVEL-C rallies. Grading and coverage, not training — the learner
is saturated. Machine prerequisite: pose npz for r18/r19 (see below).

**2. Nothing else.** r20 and r21 stay unclicked-or-unspent. **r20 is the
next seal, r21 the second**, and no graded run happens on either without an
explicit go. r22-r34 are temporal-gate holdout and stay untouched at every
level.

**Still blocked, and it is not a clicking problem: CHECK 2 on path-first.**
Its truth is contact taps at ±0.15 s, which r2-r5 turn out to have — so the
panel is 4 clean train rallies wide, not 1, and r2's 11 contacts remain
under the power floor its own FAIL established. Run it.

### Machine-side, before the next batch

- **Make the emission refit n-way.** `emission.py cache-cross` is hard-wired
  to r6/r7 and structurally cannot absorb r2–r5 or r17. Still worth doing —
  but for the reason the label curve leaves standing, which is **regime
  coverage, not sample count**: r5's failure is `dbody` + `crowd`
  suppressing kitchen dinks while r3's is `yellow` missing fast streaks,
  and one global weight vector cannot serve both.
- Pose npz for r18–r21 (CPU here, ~21 min each, or Colab per
  `gpu_runbook.md`). r2–r10 and r17 are already extracted.
- **CHECK 1 on path-first: DONE 2026-09-03, passes on all seven train
  rallies** (pooled 78.1% V vs a 70% bar, of-claimed 99.6%; `pf_check1.py`).
- **CHECK 2: unblocked by item 1 above.** Run it the moment the taps land.

**Exit:** contact taps on r2–r5 delivered, CHECK 2 panel powered, r18/r19
clicked, n-way emission refit landed and re-read on r6/r7 cross-fold, no
seal spent.

**Explicitly parked, each needing an explicit go:** the S-click rule change
(~500 genuine positives currently discarded), a streak-elongation emission
feature, the scorer fix (interpolate + per-clip measured cut offset), any
rescore of the r9/r10 incumbent numbers, and the contact-anchored junction
solve (below).

---

## Phase 2 — derived stats on chicago0725

**The hard rule that defines this phase:** the stat must be **derived by the
machine on rallies nobody clicked**. Owner clicks are the *grader*, never
the source. A number that needs a human to click the ball first is not a
stat, it is a transcription.

### The deliverables, in ship order

1. **Touch share** — of the balls a team hit, what fraction each partner
   hit, plus hits-per-player and last-shot location. Direct complement to
   the coverage model: coverage measures SPACE, touch share measures BALLS,
   and the divergence is "who is carrying this team".
2. **Contact map + bounce map** — the non-tracking family. Every shot's
   contact position in court feet (exact plane), and the bounce proxy at
   5.3 ft median / 92% court side, with that error bar printed on the page.
   A bounce map at 5 ft is a real bounce map; a line call at 5 ft is not.
3. **Rally tempo and average speed** — seconds between contacts, average
   mph beside it. Framed as average speed per the STATS.md house rule:
   never next to a radar number, and remembering a shot's average speed is
   partly the receiver's choice.

**Now in scope, reversing the earlier exclusion:** speed. The old line —
"do not ship a speed number off this footage without a side camera" —
applied to the 3D launch fit and still does. It does not apply to
`geom_speed.py`. **Still out of scope:** shot types, in/out calls, spin,
launch speed, anything needing a timestamp from pose.

### Attribution and grading are nearly free

`data/vision/rally_windows_chicago0725_v4.csv` carries all 188 rallies with
`teamA_names` / `teamB_names`, division (125 mixed / 32 mens / 31 womens)
and per-rally `server_uuid` + `receiver_uuid`. `vision/lineup.py` is solved
at 99.25% over 45,689 rallies, so real names cost nothing. Two validators
need no new clicks:

1. **Shots 1 and 2 are known for every rally** — `server_uuid` and
   `receiver_uuid` populated on **188 of 188**, i.e. **376 free hitter
   labels**, scored at n=188 with a flat chance baseline.
2. **Alternation is an internal validator** — exact in this footage (0
   violations / 229 contacts), so a derived sequence that fails to alternate
   is detectably wrong on all 188 rallies with no labels at all.
   `geom_speed.py` already prints this per rally and it already caught
   something: r17's attribution fails it 4/17.

Validator (1) assumes the tracker's first two events land on the actual
serve and return rather than a bounce — verify on r6/r7 before leaning on
it. That is boundary typing again, HANDOFF to-do (a).

### The cost, and the cheaper road

Tracked chain per rally: clip → pose → candidate cache → emission p-cache →
path-first → events → stats. Non-tracking chain: clip → pose → contact
times → stats. **Pose is the shared bottleneck** at 20.7 min/rally on CPU
here, so 188 rallies ≈ 65 CPU-hours; Colab per `gpu_runbook.md` is the
practical route, clips cut in batch with `cut_clip.py` (`check_clip.py`
before trusting any re-cut).

A partial run is a legitimate product — the coverage model published off 90
of 141 rallies and policed itself into withdrawing three of its own
findings. Pick the rally set by a rule written down first (all womens +
mens rallies, or every rally over N seconds), not by which ones came out
looking good.

**Exit:** a touch-share table plus a contact/bounce map for chicago0725,
derived end to end, graded against the 376 free labels and the alternation
check, shipped as a site block or an insight article with error bars and
nulls next to them. If it fails its bars the phase still closes — the
write-up is a null result, which this project publishes.

---

## Phase 3 — back to the 3D sim

Already partly built: `rally_3d.py` → `court3d_r{N}.html`,
`render_court3d.py` → mp4, the rally-1 replay live in
`web/replay/pickles_replay.html`, r4 rendered and handed to design.

What Phase 2 buys it: r4 passes CHECK 3 at 1.76 ft while r3 and r5 fail at
4.84 and 4.60, and CHECK 3 fails on a *different component* each rally.
Picking a showpiece today means picking the one that happened to fit. After
Phase 2 there is a match-wide reconstruction to choose from on a stated
criterion.

Open before this ships as more than a demo: **boundary TYPING** (bounce vs
contact — r10's human ledger has 13 bounces, path-first typed 0). Per-shot
mph is no longer blocked: average speed ships, launch speed never will off
one camera.

### The named next build: the contact-anchored junction solve

Not scheduled, needs a pre-registered bar before any number, and it is the
one piece of tracker work still clearly worth doing. The black hole is
**bracketed** — an arc arrives, an arc departs, one junction is missing, and
the geometry channel now supplies the hitter's paddle position in court
feet. Two arcs + a spatial anchor + a time is over-determined, which is
where inference beats detection. `court3d.fit_segment` already solves the
same shape for a bounce; `gapfill.py` already extends both arcs to their
closest approach. It would also type boundaries for free — a junction at
ground level is a bounce, one at a paddle is a contact — which is exactly
what CHECK 3 keeps failing on.

---

## Constraints that survive all three phases

Carried verbatim from `HANDOFF.md`; nothing in this roadmap relaxes any of
them.

- No graded re-run / seal consumption without explicit owner authorization.
  **r20 is the next seal, r21 the second.** Bars never loosen.
- TRAIN = r6 + r7 (+ r17, r2–r5 once the refit is n-way). r9 / r10 owner
  clicks are EVALUATION-only: grading and autopsy, never production path
  selection, never tuning a weight or threshold. Any knob is tuned on r6/r7
  cross-fold, the selection rule is written down BEFORE the numbers, then
  ONE shot.
- Positives ONLY from owner V clicks within R_POS=6 px; S clicks are
  ignore-zones (R_IGN=22), never positives; never train on tracker output or
  model self-labels.
- Oracle-bounds fits are diagnostic-only. Temporal-gate holdout rows
  (`data/vision/label_split.csv`) are untouchable; holdout burns on use.
- Rallies 22–34 are temporal-gate holdout and stay untouched at every level,
  including Phase 2's match-wide run.

## What this roadmap deliberately does not chase

The pre-registered seal on r20 — a full three-check PASS on a fresh sealed
rally — is the *scientific* finish line and it is not close: CHECK 3 fails
on a different component each rally. That is fine. It accrues in the
background off the same clicking Phase 1 already does, and Phase 2 does not
depend on it. Do not let it gate a publishable stat, and do not spend a seal
to chase one.

## What this roadmap used to say

For the record, so the reversals are visible rather than quietly edited
away. The morning version of this file said:

- *"`learner_gate.md` came back label-limited, not architecture-limited …
  the next input is clicks, not model code."* **Superseded the same day**
  by `label_curve.py`: that read came from the within-rally curve alone,
  and the between-rally curve is flat too.
- *"Not in scope: 'who sped up first' is measured broken … Do not ship a
  speed number off this footage without a side camera."* **Half reversed.**
  True of the 3D launch fit, false of geometry + timing. Average speed is
  now shipped and graded.
- Phase 2 costed only the tracked chain. **The non-tracking family skips
  four of its seven stages**, which is why the contact/bounce map is now a
  ship item rather than a someday.
