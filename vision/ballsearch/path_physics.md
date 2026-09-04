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
| **path retrace >0.70** | **3.7%** | **20.9%** | **5.6×** | strong, high volume |

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
- *"went behind the player, toward the player, behind her again, toward her
  again"* → **retrace**. The deepest of the three, because it is about
  CAUSES rather than positions: a ball reverses only when a paddle, the
  floor or the net reverses it, so four reversals in half a second is four
  causes that do not exist. Measured as the fraction of a window that comes
  back within 2 ft of where the path already was ≥0.15 s earlier. Lower
  lift than stall (5.6× vs 10.9×) but six times the volume, and it is the
  rule that actually moves the calls.

## Rules shipped (`path_physics.clean`)

| rule | basis | junk cut | good cut |
|---|---|---|---|
| BOUNDS | no ball before the serve contact or after it dies | ~50 pts/rally | 0 |
| TELEPORT | >2200 px/s (human paths' own p99.9 is 2047) | 108 | 3 |
| STALL | ±0.25 s window covering <4 ft of court | 39 | 4 |
| RETRACE | window revisits its own path, >0.70 | 262 | 101 |
| SPUR | surviving run under 4 frames | 3 | 0 |

Pooled: **27.3% of junk removed for 3.9% of good**, a 7:1 filter. It deletes
the pre-serve latch outright (confirmed in r7, r9, r10 and r17 — ~50 points
per rally, the class the owner caught in two separate videos).

End to end on the turn calls, which is what the audit was about:

    turns          338 -> 303   (-10%)
    JUNK turns     180 -> 145   (-19%)
    bounce recall   40 -> 41    /59
    impact recall  118 -> 117   /144

A fifth of the over-calling goes away and no real event is lost. BOUNDS +
TELEPORT + STALL are free but small; RETRACE is what moves it.

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
