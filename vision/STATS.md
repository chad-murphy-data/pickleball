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
| shot pace fast vs slow | AUC 0.805, null 0.50 | `geom_speed.py` |
| **shot speed in mph** | 30.3 fast / 17.5 slow, medians | `geom_speed.py` |
| rally tempo (time between contacts) | AUC 0.793 on its own | `geom_speed.py` |
| **bounce location without the ball** | 5.1 ft median, court side 92% | `bounce_proxy.py` |
| net crossings | 23/23 on r9; 14–15/21 on r10 | 3D fit |
| bounce ledger from a human ball path | 13/13 exact | `court3d.fit_segment` |

### Can get, with an asterisk

| stat | the asterisk |
|---|---|
| ball speed | it's an AVERAGE over the flight, not off the paddle; straight-line, so lobs read slow twice; most of the signal is the clock, not the geometry |
| bounce location | ~5 ft, good for a heat map and for deep-vs-short, not for line calls |
| bounce ledger from the TRACKED path | mechanism is exact; the tracker feeding it is off (8 v 13 on r10) |
| 3D replay | works and looks right (r4 CHECK 3 at 1.76 ft vs a 3.0 bar) but passes on some rallies and not others |
| first speed-up | the concept works off `geom_speed`; the shipped `rally_stats` version uses the broken 3D launch and gets it wrong |

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
| **feet + 8.5 ft lead** | **5.1 ft** | **3.9 ft** | **0.0 ft** |
| midpoint of the two hitters (control) | 8.0 ft | 6.9 ft | +5.2 ft |

Eval panel (r9+r10, n=26); the 8.5 ft lead is the median receiver-to-bounce
offset fixed on train and not refit. Categorical: court side 92%, left/right
half 92%, kitchen-vs-deep 73%. Train (n=10) is tighter at 2.9 ft.

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

None of these needs the ball. All of them need contact times, which is the
channel that already beats human labels.

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
3. **Rally tempo / pace** — median seconds between contacts, and the mph
   version next to it. The honest framing is "pace", not "ball speed off the
   paddle".
4. **The 3D replay** — Phase 3, the showpiece. Already built for r4; needs a
   rally chosen on a stated rule rather than on which one came out well.

Not shipping: shot types, in/out, spin, launch speed, anything that needs a
timestamp from pose.

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
