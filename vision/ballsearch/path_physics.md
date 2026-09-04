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
| DEFECTED | post-teleport run >80% inside player boxes | 57 | 35 |
| SPUR | surviving run under 4 frames | 3 | 0 |

**DEFECTED** exists because the first version of this filter found the
switch and then did nothing with it. r7 t=173.35 is the worked example
(owner: "what happened here?"): the ball is genuinely hit at 173.25, the
tracker sails straight through the contact, then one frame later jumps
88 px — 2,634 px/s against a 2,050 ceiling — reverses, and spends 14 of
the next 19 frames inside a player's box while the real ball goes to the
far corner unwatched. The teleport rule cut the path in the right place
but only dropped runs under 4 frames, so all 19 survived. A teleport says
the tracker changed its mind; which side of the cut is the ball still has
to be decided, and a run that lands in a person is not it.

Pooled: **31.8% of junk removed for 5.1% of good**, a 6:1 filter. It deletes
the pre-serve latch outright (confirmed in r7, r9, r10 and r17 — ~50 points
per rally, the class the owner caught in two separate videos).

End to end on the turn calls, which is what the audit was about:

    turns          338 -> 288   (-15%)
    JUNK turns     180 -> 132   (-27%)
    bounce recall   40 -> 41    /59
    impact recall  118 -> 115   /144

Over a quarter of the over-calling goes away, bounce recall is up one, and
the cost is 3 impacts out of 144. BOUNDS + TELEPORT + STALL are free but
small; RETRACE and DEFECTED are what move it.

DEFECT_BOX was set at 0.80 on the end-to-end sweep, not the point sweep --
the point ratio is flat across the whole range (3.1:1 at 0.5 up to 7.1:1
at "off") while the calls are not: 0.9 misses the r7 excursion entirely,
0.7 costs 7 impacts to remove 6 more junk turns.

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

## "The ball never goes behind players' feet or outside the court" — null

Owner's rule, tested as stated. It does not fire on this footage, and the
two halves fail for two different measured reasons.

**Written as a ray/volume test.** A pixel is not a point, it is a ray from
the camera. So the right form of the rule is: *is there any height at which
this pixel is a ball inside the playable volume* — the court footprint plus
a margin, ground to lob ceiling. That reformulation was supposed to absorb
the lob and ATP exceptions instead of carving them out (a lob is in the
volume, an ATP is in the extended footprint at low height). It does absorb
them. It just has nothing left to delete.

Marginal flags **on top of** the shipped filter, nine human-graded rallies:

    margin  0 ft, ceiling 12 ft   junk  11   good   6      (1.8:1)
    margin  0 ft, ceiling 20 ft   junk  11   good   6
    margin  3 ft, ceiling 15 ft   junk   1   good   0
    margin  6 ft, ceiling 15 ft   junk   0   good   0

The shipped rules run about 7:1. At any honest margin — a pickleball court
has ~6 ft of run-off and ATPs go wider — the rule flags nothing at all, and
the only setting that flags anything is worse than a coin flip. Not shipped.

**Why "outside the court" is empty: the volume fills the frame.** Sampling
every 4th pixel of the 1280×720 frame and asking the same question:

    margin  0 ft, ceiling 12 ft    54.5% of the frame is a legal ball position
    margin  3 ft, ceiling 12 ft    82.0%
    margin  6 ft, ceiling 15 ft    98.6%
    margin 10 ft, ceiling 20 ft   100.0%

There is almost nowhere in this camera's view a ball could not be. And the
junk isn't in the 1.4% that's left, because the junk rides *players*, and
players stand on the court. 99% of junk points are reachable at ground
level inside the lines.

**Why "behind their feet" fires backwards.** A ball at height projects
*deeper* than it is — the camera looks down the court, so height and depth
are the same measurement at z=0. Which means good path is behind the feet
*more* often than junk, not less:

    share of points >3 ft behind the nearest player's feet    good 88%  junk 86%
    share >6 ft                                               good 83%  junk 74%
    share >12 ft                                              good 74%  junk 57%

Lift 0.8–1.0× — the wrong direction at every threshold. The rule would
preferentially delete lobs and drives, which is exactly the exception list
it was going to need.

**"Those have the player feet moving"** — tested separately as its own
feature, on the nearest player's box-bottom speed over ±0.15 s. Null:
p50 123 px/s under good points vs 111 under junk, p90 232 vs 227.

**What survives of the intuition.** The rule is real, it just isn't a
*position* claim on this footage — it's the occlusion claim, already shipped
as `in_player_box` (69.6% of junk vs 23.7% of good). A player box is not
where the ball is, it's where the ball is invisible. The unspent half is
that the ball's height is knowable when you stop treating it as a
constraint on pixels and start treating it as a quantity to solve for
between two known contacts — which is the double-bounce arc fit, not this.
