# Track-level physics filter — what separates ball from junk

Built after the owner audited four labelled videos (`turns_r{7,9,10,17}.mp4`)
and found that most of the bad calls were never the detector: they were path
that no ball could have made — teleports onto a shoe, zigzags behind a
player, hairpins, apex double-turns, and a ball "tracked" before the serve
exists.

Everything below is graded with **no new labels**. The owner's nine clicked
human paths say where the ball actually was, so every tracked point is GOOD
(within 25 px of the human path at the same time) or JUNK. 2,803 good /
1,247 junk over nine rallies.

## Features tested

| feature | GOOD | JUNK | lift | verdict |
|---|---|---|---|---|
| instantaneous speed | — | — | ~1.0× | **null** |
| local parabola residual (±3f) | 7.2% >8px | 24.6% | 1.4× | weak |
| run straightness (pixels) | — | — | 2.0× | weak |
| direction off court long axis | 8.8% >60° | 8.8% | 1.0× | **null** |
| **inside a player's box** | **23.7%** | **69.6%** | **2.9×** | strong |
| **run court displacement <3 ft** | **1.7%** | **18.8%** | **10.9×** | strongest |

Two of these came from the owner during the audit and are the only two that
worked. Both were first stated in a form that is null, and both survive when
moved to the right frame:

- *"it goes to the paddle where it's hidden, goes behind, turns left, turns
  right, spins, and ends up back on path pretty quickly"* → **occlusion**.
  A player box is where the ball is INVISIBLE, not where it is.
- *"any path not toward the court needs justification" / "no one ever hits
  the ball straight to the left"* → **stall**. As an instantaneous direction
  test this is null (lift 1.0×) because junk rides players and players move
  up-court too. Measured as COURT DISPLACEMENT over half a second it is the
  strongest feature found: the ball always goes somewhere, a shoe does not.

## Rules shipped (`path_physics.clean`)

| rule | basis | junk cut | good cut |
|---|---|---|---|
| BOUNDS | no ball before the serve contact or after it dies | ~50 pts/rally | 0 |
| TELEPORT | >2200 px/s (human paths' own p99.9 is 2047) | 108 | 3 |
| STALL | ±0.25 s window covering <4 ft of court | 39 | 4 |
| SPUR | surviving run under 4 frames | 3 | 0 |

Pooled: **11.1% of junk removed for 0.2% of good.** Free, and it deletes the
pre-serve latch outright (confirmed in r7, r9, r10 and r17 — ~50 points per
rally, the class the owner caught in two separate videos), but it is not
yet a big enough bite to move the bounce calls (turns 338 → 314, bounce
recall 40 → 41, impact recall 118 → 117).

## Occlusion bridging — measured, not yet shipped

In-box is the second-strongest feature but deleting in-box points is wrong:
24% of good path is legitimately in front of a player, and contacts happen
at paddles, i.e. inside boxes. The right use is to bridge across.

Straight-line bridging of every box crossing is a wash (median error
20.2 → 23.8 px, p90 135 → 114). The reason is the anchors: of 139
bridgeable spans, only **42% have both endpoints on the ball**. Bridging
between two junk points draws a line between two wrong places.

With oracle anchors (both endpoints known GOOD, n=239 points):

    raw      median  8.2 px   p90 32.5   within 25px 84%
    bridged  median  9.9 px   p90 21.7   within 25px 92%

So even perfect anchoring is a tail fix, not a median fix — when the tracker
is on the ball entering and leaving a box it is usually fine inside too. The
open work is finding trusted anchors for the other 58% of spans, which is
the same problem as identifying junk in the first place.
