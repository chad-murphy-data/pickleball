# What we can measure, what we can't, and what to ship

Written 2026-09-03, owner ask: *"record everything you said about what we
can and can't get ... and think about what we can/want to ship and where we
can start using non-tracking methods (where the ball bounced) for 'good
enough' answers."*

Three files, three jobs:

- `STATUS.md` — **where are we**: the per-channel ledger, every number next
  to its null. It is the authority. If this file disagrees, it wins.
- `ballsearch/ROADMAP.md` — **where are we going**: the phase plan.
- **this file** — **what comes out the end**: the stat list, graded, and the
  order to ship it in.

Every number below is measured and traceable to a script in this repo. Where
something is an estimate rather than a measurement it says so.

## The four input channels

Everything downstream is built from these. Their quality is the whole story.

| channel | how good | where from |
|---|---|---|
| **court geometry** | 0.06 ft median residual | homography, `court3d.py` |
| **player positions** | feet on the z=0 plane, so exact to the above | pose tracks → homography |
| **player identity** | 99.25% over 45,689 rallies | referee logs, `lineup.py` — no camera |
| **contact times** | 73.1% @ ±0.15 s on sealed r10, beating human labels (65.4%) | ball path → timing stream |
| **shot order / sides** | counts 161/162; side alternation 0 violations / 229 | alternation decoder |
| **the ball itself** | 90% mid-flight, 72% within 0.1 s of a contact | `pathfirst.py` + `gapfill.py` |

The pattern that matters: **three of these are solved and one is not.** Where
the ball is at any instant is the only weak channel, and it is weak in a
specific, bounded place — see "the black hole" below.

## The stat list

### Can get, and it's good

| stat | quality | instrument |
|---|---|---|
| hits per player (touch share) | ±1 on 8/8 player-rallies | `rally_stats.py` |
| shot count per rally | 161/162 | alternation decoder |
| who hit it (side) | 95% | pose + alternation |
| who hit it (which of the four) | 85% | pose |
| every contact's court position | feet, exact plane | pose → homography |
| where each player stands, over a match | width share, 90% area, kitchen band, off-court fraction | coverage model, 90 of 141 rallies |
| shot pace fast vs slow | AUC 0.829, null 0.50 | `geom_speed.py` |
| **average shot speed in mph** | 30.3 fast / 17.5 slow, medians | `geom_speed.py`, shipped in `rally_stats.py` |
| rally tempo (time between contacts) | AUC 0.793 on its own | `geom_speed.py` |
| **bounce location without the ball** | 5.3 ft median, court side 92% | `bounce_proxy.py` |
| net crossings | 23/23 on r9; 14–15/21 on r10 | 3D fit |
| bounce ledger from a human ball path | 13/13 exact | `court3d.fit_segment` |

### Can get, with an asterisk

| stat | the asterisk |
|---|---|
| bounce location, no ball | ~5 ft, good for a heat map and for deep-vs-short, not for line calls |
| bounce location, TRACKED | **1.91 ft** where it fires — but it only fires on 13 of 34 (see below) |
| 3D replay | works and looks right (r4 CHECK 3 at 1.76 ft vs a 3.0 bar) but passes on some rallies and not others |
| first speed-up | right player on both eval rallies, right shot on one (see the house rule below) |

### Can't get, and more data won't fix it

| stat | why |
|---|---|
| contact TIMESTAMPS from body pose | 40.7% against an 85 bar; four VLM grid arms all fail their own shifted-call null |
| shot TYPE (dink / drive / roll / drop) | pose magnitude is AUC 0.445 — inverted; fast-vs-slow is the whole resolution |
| launch speed off the paddle | one camera can't see toward-or-away; the 3D launch fit scores AUC 0.10, worse than a coin |
| spin | nothing in the stream carries it |
| in/out calls | needs the ball to a few inches; we have ~5 ft without it and ~2–3 ft with it |
| ball height at contact | depth-degenerate, same reason as launch speed |
| shot selection / decision quality | execution and decision are perfectly confounded in the result |
| anything at all from a better ball detector | closed for a separate reason: on 36% of in-play frames a human can't see the ball, so a detector's claims there are unfalsifiable |

**More clicked rallies do not move any of this.** Both label axes are
measured flat: within-rally (AUC 0.907 at 50 positives vs 0.904 at 198) and
between-rally (k=1 → k=6 moves AUC +0.008 and the operational metric not at
all). 10,000 clicked rallies would be 3.6M clicks, 1,500–3,000 hours, and
would buy nothing for this model class.

## House rule: average speed is the speed (owner call, 2026-09-03)

> *"I think average speed is good enough for speed. When people measure serve
> speed on tennis they don't show frame by frame. And 'close enough' speed is
> fine for pickleball. Not a sport that measures in speed."*

Settled. The project reports **average speed over the flight** — court
distance between consecutive hitters' feet, divided by the time between their
contacts — and does not chase launch speed. Launch speed off a single camera
is depth-dominated and measured inverted (AUC 0.10); it is not a precision
target we are falling short of, it is a measurement this footage cannot make.

Two things that follow, both about labelling rather than accuracy:

- **Say "average", and don't compare it to a radar number.** A tennis serve
  gun reads the ball at launch. Average-over-flight runs lower — drag, plus
  we charge the ball the straight line and it flies an arc. Our 30 mph fast
  shot is not the 40-something a gun would post at the paddle.
- **A shot's average speed is partly the receiver's choice.** The clock stops
  at the next contact, so a put-away that draws a late reply reads slow. This
  is not hypothetical: r10's true first speed-up (300.23 s) measures 23 ft/s
  = 15.7 mph even with perfect labels, because the reply came a full second
  later. Fixable in principle by ending the clock at the bounce or the
  arrival instead — not built.

Shipped 2026-09-03: `rally_stats.py`'s speed-up rule now uses average speed
instead of the 3D launch. Threshold V_FAST = 34.2 ft/s, the midpoint of the
TRAIN-panel medians (fast 44.39 / slow 23.94), fixed on r6/r7/r17 before any
eval read. Evaluation on r9/r10: **first speed-up was wrong on both rallies
under the old rule; it is now the right player on both, and the right shot on
r9** (Emma Nelson at 257.90 vs truth 257.84). r10 names the right player
3.8 s late, for exactly the reason in the second bullet.

## UPDATE 2026-09-04 — the bounce bottleneck moved

Everything above was written before the four commits of 2026-09-04. One of
them changes which problem is the hard one, so read this section against the
tables above.

**Locating a bounce is now solved; finding one is not.** Check 3 gained a
time-matched comparison of the fitted `bounce_xy` against the human ledger
(`1cb9042`, `ac87c99`), so bounce position is reported in court feet rather
than only counted:

| rally | matched | median |
|---|---|---|
| r7 | 3/3 | 2.05 ft |
| r9 | 5/13 | 0.83 ft |
| r10 | 2/13 | 1.60 ft |
| r17 | 3/5 | 1.91 ft |

Pooled n=13, **median 1.91 ft**, 46% within 1 ft, 77% within 3. That beats
the no-ball proxy's 5.3 ft by ~2.8x and beats the tracked *impact* numbers
(2.15-2.98 ft) on every single rally — structurally, not by luck: a bounce
sits on the z=0 plane the homography solves to 0.06 ft and is bracketed by
arcs on both sides, where the tracker is a 90% instrument. A contact is
exactly where the path has its hole.

So the honest table for bounces is now three rows, not one:

| | state |
|---|---|
| locating a bounce we found | **solved**, 1.91 ft |
| finding the bounces | **13 of 34 = 38% recall** |
| not calling non-bounces | of 30 emitted, **6 real / 12 junk / 12 missed contacts** |

Neither live problem is the ball detector. Both are the claim logic
(`a16e2ba`): `bounce_evs = [e for e in turns if e not in claimed]` makes
bounce the residual category, so every anchor miss becomes a fake bounce and
nothing ever asks whether the turn looks like one.

**Two measured fixes exist and neither is shipped.**

1. *The bounce signature.* Image y falls then rises — a sign test, nothing to
   tune. Keeps **6/6** real bounces, kills **10/12** junk markers. The turn
   angle the claim uses today does not separate them at all (real median
   91.5 deg, junk 66.5, ranges overlapping). Measured in `a16e2ba`; the
   predicate lives at `ballsearch/turn_geom.py:42` as `v_shape` and
   `ball_replicate.py` does not call it.
2. *A spatial gate on claiming.* `claim_bounds()` gives an anchor its
   largest-angle turn within 0.25 s with no distance test, which is the
   over-claiming half.

**Path junk is now filtered before any of this** (`ballsearch/path_physics.py`,
graded label-free against the nine owner-clicked paths in `physics_grade.py`):
31.8% of junk points removed for 5.1% of good, junk turns 180 -> 132. Rules
are bounds / teleport / spur / stall / retrace / defected, each with its
measured lift in `ballsearch/path_physics.md`.

**Four things were tested and are null** — recorded so they are not retried:
the court-volume rule ("the ball is never outside the court or behind a
player's feet"), the same rule written player-relative, vertical position
within the body as a delete rule, and occlusion bridging. A per-point trust
ranker does work as a *ranker* (leave-one-rally-out AUC 0.831, every rally
0.75-0.93) but does not fix bridging: pooled AUC is inflated by
stratification, and at a box edge -- exactly where bridging must pick its
anchors -- it is 0.729 on a 63% base. Details and nulls in
`ballsearch/path_physics.md`.

## The black hole, and why it licenses the non-tracking family

Measured 2026-09-03 (`ballsearch/blackhole.py`), the frozen scorer sliced by
distance to the nearest contact:

| | human can't place it | machine claims | machine right |
|---|---|---|---|
| within 0.10 s of a contact | 14% / 11% | 75% / 76% | **72% / 72%** |
| 0.10–0.25 s | 3% / 4% | 80% / 76% | 74% / 73% |
| mid-flight | 1% / 6% | 92% / 90% | **91% / 90%** |

(train / eval.) Mid-flight the tracker is a 90% instrument. It loses ~19
points in a ~0.2 s window at each direction change, and the human loses there
too — the ball really is behind a body. Unrecovered runs are median 6–7
frames, p90 18–22.

So the ball channel is not broadly unreliable. It is reliable in the middle
and blind at the ends. **Every stat that needs the ball only mid-flight is
already fine; every stat that needs it AT the contact should be built from
the players instead.**

## The non-tracking family — the idea worth spending time on

Player positions and contact times are solved. Chain them and a whole class
of shot-location stats falls out with **no ball at all**.

**Bounce location.** A ball that bounces gets struck right after it bounces,
so the receiver's own feet locate the bounce. Graded against the human-fit
bounce ledger (the arm verified exact, 13/13 on r10):

| estimator | median error | depth error | depth bias |
|---|---|---|---|
| receiver's feet | 6.3 ft | 6.1 ft | −2.8 ft |
| **feet + 8.7 ft lead** | **5.3 ft** | **3.8 ft** | **0.0 ft** |
| midpoint of the two hitters (control) | 8.0 ft | 6.9 ft | +5.2 ft |

Eval panel (r9+r10, n=26); the 8.7 ft lead is the median receiver-to-bounce
offset fixed on train and not refit. Categorical: court side 92%, left/right
half 92%, kitchen-vs-deep 73%. Train is 4.0 ft on 31 bounces — the 2.9 ft
first reported was an n=10 panel, and the honest number went UP when the
panel tripled.

Read that against the tracked version: the 3D fit posts 2.15–2.98 ft on
matched impacts. So **no-ball is about 2× worse and costs nothing** — and it
runs on all 188 rallies today, where the tracked version needs a pose
extraction, a candidate cache and a graded path per rally.

That is the whole "good enough" argument. A bounce map at 5 ft is a real
bounce map. A line call at 5 ft is not a line call. Ship the first, never
claim the second.

**Everything else in the family, same two inputs:**

- contact position of every shot, in court feet — free, exact plane
- shot pace and mph — `geom_speed.py`, above
- serve and return depth (how deep the returner takes the ball) — free
- where each player stands when the ball is struck, all four at once
- kitchen arrival time: when does each team get to the line
- rally shape: contact positions over time, which is a rally in one picture

**Be precise about what "non-tracking" means here.** None of these needs to
know where the ball is AT the contact — which is exactly where the tracker
drops to 72% and the human drops too. They do need contact TIMES, and those
still come from the ball path mid-flight, where the tracker is a 90%
instrument and already beats human labels. So the family is not ball-free end
to end; it is free of the ball precisely where the ball is worst. That is the
whole trick, and it is why these stats can be good when line calls can't.

They also skip most of the per-rally machine chain. A tracked stat needs
clip → pose → candidates → emission p-cache → path-first → events → stats.
The non-tracking family needs clip → pose → contact times → stats.

## What to ship, in order

`ROADMAP.md` Phase 2's rule stands: **the stat must be derived by the machine
on rallies nobody clicked.** Clicks are the grader, never the source.

1. **Touch share** — the Phase 2 deliverable already chosen. Who hit how many
   balls, per player, per match. Graded free: `server_uuid` / `receiver_uuid`
   are populated on 188 of 188 rallies (376 free hitter labels), and side
   alternation validates every derived sequence with no labels at all.
   Complement to the coverage model: coverage measures SPACE, touch share
   measures BALLS, and the divergence is "who is carrying this team".
2. **Contact map + bounce map** — the non-tracking family. Every shot's
   contact position, and the bounce proxy at 5 ft. Publishes as a heat map
   per player, or per team, with the error bar stated on the page.
3. **Rally tempo / pace** — median seconds between contacts, and the average
   mph next to it. Framed as average speed, per the house rule above.
4. **The 3D replay** — Phase 3, the showpiece. Already built for r4; needs a
   rally chosen on a stated rule rather than on which one came out well.

Not shipping: shot types, in/out, spin, launch speed, anything that needs a
timestamp from pose.

## Panel sizes — what every claim here currently rests on

Small n is the honest weak point of the two new instruments, and it is the
cheapest thing to fix.

| instrument | panel today | limited by |
|---|---|---|
| `geom_speed.py` | 111 pace-labeled flights (64 train / 47 eval) | rallies with a ball path AND pose |
| `bounce_proxy.py` | 57 human-fit bounces (31 train / 26 eval) | rallies with a ball path AND pose |
| `blackhole.py` | 3,804 owner-judged frames | nothing — this one is fine |
| `rally_stats.py` counts | 8 player-rallies | manual shot labels |
| `label_curve.py` | 663k candidates, 1,888 positives | nothing — saturated |

**Correction, 2026-09-03: r2-r5 were already fully tapped, and the panel
sizes above are the ones that follow.** The first version of this table said
those four rallies carried "only prefill contacts" and costed a 58-contact,
45-minute tap pass to add them. That was a misreading of the CSV on my side:
`source` = `prefill` means the tap was entered with ⏎ against the prefilled
hitter/type, not that the row is an un-timed placeholder. Every one of the
323 rows in `contact_labels_chicago0725.csv` carries an owner tap. Folding
r2-r5 in cost nothing but a one-line fix, and it did what an hour of
clicking was forecast to do: bounces 36 → 57, flights 63 → 111.

What is still genuinely limited: `rally_stats.py`'s per-rally shot counts
(8 player-rallies) and CHECK 2's contact-timing panel, which wants rallies
with taps AND a graded path. The next real click job is a BALL PATH on a
rally that already has taps — r18, r19, r20-r22 — not more contact taps.

## What would change the answer

Only two things, and neither is more clicking:

- **New footage.** Higher resolution or uncondensed source moves the ball
  channel; a second camera angle kills the depth degeneracy outright and
  turns half the "can't" list into "can". Everything in the can't-get table
  that says "one camera" is waiting on this and nothing else.
- **The contact-anchored junction solve** (named, unbuilt, needs a
  pre-registered bar). The black hole is bracketed: an arc in, an arc out,
  one missing junction, and now a paddle position in court feet to anchor it
  to. That is over-determined, which is where inference beats detection. It
  would also type boundaries for free — a junction at ground level is a
  bounce, one at a paddle is a contact — which is the thing CHECK 3 keeps
  failing on.
