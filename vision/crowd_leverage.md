# Does the crowd know the score? — No.

*2026-08-10. First derivation off the full-VOD instrumentation (the audio
loudness stream, which completes before the video pass even starts).
Chicago Slice v Utah Black Diamonds, MLP Chicago 2026-07-25, 193 rallies.
Data: `data/vision/chicago0725_loudness.csv` (cheer band per 0.25 s),
`data/vision/chicago0725_cheer_rally_join.json` (the rally↔cheer join).*

## Method, and the two validations that make it trustworthy

Cheer events = excursions of the 2–8 kHz band over a rolling 30 s median
baseline (391 events detected; ~2 per rally is right, since big rallies get
a roar *and* an applause tail). Every cheer was assigned to a rally by
monotone DP alignment of the cheer train against the referee log's 193
rally-end times, allowing for the broadcast's cuts. Two independent checks:

1. **Zero-parameter anchor**: the probe clip pinned video 778.07 s = wall
   18:32:18Z. The alignment put game-1 rallies 29/30's cheers at 746.8 s
   and 774.2 s — right where they must be (cheers peak 2–4 s before the
   referee's button press).
2. **The DP recovered the broadcast's edit depth blind**: accumulated cut
   grew monotonically to 27.0 min; the independently measured condensation
   is 26.9 min (107.2 wall − 80.3 video).

180/193 rallies matched (93%).

## Result 1: the crowd does not track leverage

Every rally carries an exact leverage value (finding 10's scale, computed
from its start score via the serve-aware DP at equal strength; this match
contains four maximum-leverage 0.457 rallies).

| | |
|---|---|
| spearman(leverage, cheer magnitude) | **−0.071** [−0.208, +0.078] |
| …with rally length controlled | −0.071 [−0.206, +0.076] |
| cheer by leverage quartile (Q1→Q4) | 2.69× / 2.65× / 2.55× / 2.62× — flat |

The loudest cheer of the match (7.2× baseline) followed a 13-second rally
at **8-2 in the blowout game** — leverage 0.073, near the bottom of the
scale. Meanwhile the four 0.457 rallies drew cheers at the 70th, 20th, and
11th percentiles of the match — **match point got a golf clap**.

## Result 2: it does not track allegiance either

This is MLP *Chicago*; the Slice are the home team. Cheer after a Chicago
rally win: 2.61×. After a Utah win: 2.65×. Difference −0.04×, permutation
p = 0.75. Top-10 cheers split 4/10 Chicago. The within-side leverage
correlations are null too (−0.004 / −0.123).

## What this means

The crowd's roar is essentially unpredictable from the scoreboard — stakes
and side both null. What is left is the **content of the rally itself**:
spectacle — diving gets, hands battles, ATPs — which is precisely the thing
no scoreboard-derived variable can see and the ball-tracking data can.
Even the crowd is telling us the interesting variable is what happened in
the rally, not what it was worth.

The counterpoint writes the headline: **finding 10 established that the
players track leverage** (clutch is real — Johns banks a full game of win
probability per season from *when* his rally wins land). The crowd does
not. Clutch exists on the court, not in the stands.

## Caveats

One match, one home crowd, n=180. The cheer magnitude is an excursion over
an adaptive baseline, which compresses sustained-noise differences, and the
2–8 kHz band includes commentary excitement. The leverage null's CI
excludes moderate tracking (r > ~0.08) but not tiny effects. Replication is
one `track_full_vod.py` run per VOD — the marginal cost is a laptop-hour.

## Addendum: "are X's matches louder?" (star-power, first look)

Raw levels (not baseline excursions) are directly comparable within one
broadcast — same night, same mics. Across the four games here: **near
flat** (rms spread 1.07×, cheer band 1.10×), and the loudest game by raw
level was the men's 11-3 *blowout*, not the 12-10 thriller. n=4, noise.

The per-player cut within one matchup is **degenerate by design**: 8
players, 2 games each, and co-appearing players (partners AND opponents)
share identical game sets — Goins and Loong are indistinguishable here.
Same identifiability structure as actor/partner effects in the ratings,
same fix: variation across many matchups. The clean star-power design once
more VODs are processed is the **within-broadcast contrast** — X's games
vs the *same night's* other games — which cancels mic gain, venue, and
crowd size, the three confounds that make raw cross-broadcast comparisons
meaningless. One laptop-run per VOD.

## Addendum 2026-08-16: CAVEAT — the join underneath this was later
## shown locally unreliable; nulls downgraded to unverified

The cheer↔rally join used above was subsequently found to land up to
~40 s off for an unknown subset of rallies (user-observed during the
2026-08-14/15 labeling sessions; the v1 swing-gate kill measured on the
same join's windows was retracted for exactly this). The two
validations in the Method section certify the alignment's global SPINE
(anchor + edit depth), not per-rally assignment — and misassigned
cheers attenuate every correlation toward zero, which is the direction
of both of this file's conclusions. "The crowd does not track
leverage/allegiance" was therefore measured with an instrument biased
toward exactly that answer.

Standing status: plausible, NOT quotable. Surfaced by the user's
recollection ("we killed something based on a sync that didn't work") —
this is the one standing conclusion built on the broken join that was
never re-examined after it broke. Cheap re-measurement now exists: the
timestamped contact labels (data/vision/contact_labels_chicago0725.csv,
scorebug-verified serve stamps) give exact video times for their
rallies; recompute the leverage/allegiance correlations on stamped
rallies only, where assignment is certain. Until that is done, treat
both nulls as unverified.
