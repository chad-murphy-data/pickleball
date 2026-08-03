# Clutch, measured against a null that fights back

*Session 2026-08-03. Supersedes `model/clutch.md`, whose central claim — that
clutch is a real, reliable trait owned by the stars — does not survive. Code:
`clutch_leverage.py` (measure), `clutch_mechanical.py` (why the naive measure
is broken), `clutch_null.py` (the null), `clutch_report.py` (the answer),
`clutch_selfcheck.py` (does the machinery return zero on zero),
`clutch_power.py` (would it have seen a real effect). Table:
`data/clutch_leverage.csv`.*

## TL;DR

**We can measure clutch. When we do it properly, nobody has it.**

Not "the effect is small and we're being cautious" — the field of 837 players
is *exactly* as wide as chance (var(z) = 1.018 against a target of 1.000),
all three reliability tests sit on their null floors, and the best player in
the sport is 3.86 sigma up with a multiplicity-corrected p of 0.13.

The reason the previous pass found the opposite is specific and fixable:
**side-out scoring manufactures a leverage/outcome covariance out of nothing,
and the size of that fake effect depends on how good you are.** That is why
the old clutch leaderboard read like a list of the sport's best players — it
was one.

What we can state positively is a ceiling. The pipeline recovers an injected
tau of 0.0100 as 0.0099 and an injected 0 as 0.0010. The real archive gives
0.0031 [0.0005, 0.0043]. So if clutch ability exists in pro pickleball it is
worth **less than about half a percentage point of rally-win probability per
standard deviation of leverage — under half a game per hundred played.**

---

## 1. The measurement

1,464,258 rallies — every logged rally in the warehouse, 2024-01 to 2026-07:
969,470 PPA+MLP doubles and 494,788 PPA singles, across 41,693 games and
20,181 matches. (The previous pass used 162,942, from one spring, serve side
only.)

**Leverage** L_r = |P(win game | win this rally) − P(win game | lose it)|,
exact from the serve-aware side-out DP in `web/sitelib/winprob.py`, evaluated
at eta = 0 so leverage is a property of the *situation* — score plus serve
state — and never of the players in it. Measured k: 0.4383 doubles, 0.5254
singles. The biggest rally on the board swings 0.463 of a game.

**Clutch** is the within-cell covariance of leverage and outcome:

    b_p  =  sum_g Sxy_g  /  sum_g Sxx_g

over that player's (game x channel) cells. Both leverage and outcome are
demeaned *inside each cell*, which makes b_p immune by construction to how
good the player is, how good their team was that day, which games they
played, and how much high-leverage exposure they get. It asks only: among
this player's own rallies in this game, did they win the big ones more than
the small ones? Units: extra probability of winning a rally per +1 sd of
leverage.

Two things the previous pass left on the table:

- **Return rallies are individually attributable.** `pb_rally` carries
  `receiver_uuid` per rally and it alternates within a side, so the man who
  returned the serve is named. Serving and receiving are equally
  "this player struck the ball with a partner on court". Using both doubles
  the exposure.
- **Singles has no partner at all** — 494k rallies where server and receiver
  are each exactly one person.

## 2. Why the naive number is worthless

Side-out scoring: a server keeps serving while winning, so **every service
run is a string of wins terminated by exactly one loss** — and that
terminating loss sits at the run's highest score, usually its highest
leverage. Receiving is the mirror image: you lose a string and then win the
one that ends it. Leverage is computed from the running score, which is the
running sum of past outcomes, so the regressor is endogenous and the
estimator is biased at a true effect of exactly zero.

The bias would be harmless if it were the same for everyone. It is not: a
better server has longer runs, so the single terminating loss is diluted.
**The artifact is a function of skill.**

`clutch_mechanical.py` simulates a league under real side-out rules where
rally outcomes depend only on the server's own rate and there is no
leverage-dependence anywhere. It reproduces the whole thing:

| | real archive | no-clutch simulation |
|---|---|---|
| serve-channel sd across players | 0.0132 | 0.0134 |
| return-channel sd across players | 0.0127 | 0.0120 |
| var(z) | 2.0 – 2.4 | 2.3 – 2.9 |

Under the full real-schedule null, the simulated artifact correlates **+0.654
with v2 rating** and **+0.868 with the player's own serve-win rate**. That is
the old leaderboard.

## 3. The null

`clutch_null.py` replays **every real game with its real players** under the
real side-out rules, 60 times. Each player serves at their own fitted rate
and nobody is clutch. The correction is b_adj = b_obs − mean(b_null).

Two versions, and the difference matters:

- **flat rates** — one archive-wide rate per player. Leaves tau = 0.0090.
- **opponent-aware** — `logit p = a[server] − r[receiver]`, ridge-fitted.
  Leaves **tau = 0.0031**.

The opponent-aware null is the better model of reality (its seasons run
1,508,519 rallies against the real 1,464,258; the flat null overshoots to
1,557,989), and two thirds of what survived the flat null was still artifact
from not pricing who was on the other side of the net.

It works: the correction drives the correlation between the statistic and the
player's own serve rate from +0.322 to −0.088, and with v2 rating from +0.412
to **+0.026**.

**It does not over-correct.** The positive control (§5) recovers injected
clutch at full size through the same null.

## 4. What survives: nothing

| test | observed | null floor | verdict |
|---|---|---|---|
| var(z), 837 players | **1.018** | 1.000 | field is exactly chance-wide |
| tau | 0.0031 [0.0005, 0.0043] | 0.0010 | at the machinery floor |
| serve vs return (disjoint rallies) | −0.037 [−0.142,+0.068] | +0.066 ± 0.033 | null |
| 2024-25 vs 2026 | +0.065 [−0.067,+0.197] | +0.017 ± 0.073 | null |
| doubles vs singles | −0.095 [−0.188,+0.002] | −0.036 ± 0.056 | null |
| max\|z\| across players | 3.86 (Ben Johns) | median 3.30, 95th 4.09 | **p = 0.133** |
| count \|z\| > 1.96 | 44 | median 42 [31, 52] | p = 0.383 |

The null floors are not decoration. Two slices of one *simulated* season are
not quite independent, so a serve-vs-return correlation of +0.066 is what
zero looks like — which is why the flat-null value of +0.091 was never
evidence of anything.

**The one result that looked real, and wasn't.** Under the flat null the
cross-era correlation came in at +0.246 with a calibration slope of 1.00 and
a null floor of −0.028 — three sigma clear, and it survived every check
except the right one. Skill is stable across eras, so *any* residual
skill-linked bias reproduces it exactly. Residualising each era's estimate on
skill (serve rate, serve share, v2 rating, plus squares) before correlating:

    era r, raw                    +0.246
    era r, skill removed          -0.002   [-0.121, +0.117]

The persistence was skill persisting, not clutch persisting. Under the
opponent-aware null the raw version collapses to +0.065 on its own.

## 5. Would we have seen it? Yes.

`clutch_power.py` injects a known clutch coefficient per player and runs the
identical pipeline:

| injected tau | recovered tau | 95% CI | cross-era r | players at p<0.05 |
|---|---|---|---|---|
| 0.0000 | 0.00101 | [0.00000, 0.00300] | −0.005 | 0 |
| 0.0050 | 0.00461 | [0.00352, 0.00566] | +0.017 | 4 |
| 0.0100 | 0.00988 | [0.00881, 0.01096] | +0.259 | 46 |
| 0.0200 | 0.01850 | [0.01686, 0.02028] | +0.536 | 131 |

Unbiased recovery across the range. A real tau of 0.010 would have shown up
as 46 significant players and an unmistakable cross-era correlation. We see
2 and +0.065. **The real archive sits between the zero-injection floor
(0.0010) and the 0.005 injection (0.0046)** — so the honest reading is not
"exactly zero" but "somewhere in 0 to 0.005, indistinguishable from zero, and
certainly nowhere near the 0.013 the naive statistic claimed."

`clutch_selfcheck.py` closes the loop from the other side: run the pipeline on
a season with no clutch in it and it returns tau = 0.0009, var(z) = 1.06, and
all three reliability correlations at zero.

## 6. The best-available ranking, clearly labelled

It does not replicate out of sample. It is here because the question was
asked, not because the numbers mean anything individually.

| | player | rallies | clutch | 95% CI | games/100 |
|---|---|---|---|---|---|
| M1 | Ben Johns | 27,014 | +0.0060 | [+0.0012, +0.0108] | +0.5 |
| M2 | Connor Mogle | 8,215 | +0.0033 | [−0.0022, +0.0088] | +0.3 |
| M3 | Riley Newman | 10,621 | +0.0032 | [−0.0022, +0.0087] | +0.2 |
| W1 | Anna Leigh Waters | 26,400 | +0.0054 | [+0.0004, +0.0104] | +0.3 |
| W2 | Tyra Hurricane Black | 15,893 | +0.0038 | [−0.0015, +0.0090] | +0.2 |
| W3 | Catherine Parenteau | 24,616 | +0.0028 | [−0.0021, +0.0077] | +0.2 |

Johns and Waters are the only two whose intervals clear zero, and they clear
it barely, out of 837 tries — the null expects about 42 such players and we
have 44. Johns is positive in both eras independently (+0.0106 ± 0.0044 in
2024-25, +0.0387 ± 0.0097 in 2026), which is the single most suggestive thing
in this file; but he is also the player we picked *because* he topped the
combined list, and his multiplicity-corrected p is 0.133. Suggestive. Not
established. **Do not publish either name as "clutch".**

The full table is `data/clutch_leverage.csv`, with observed slope, simulated
artifact, corrected value, se, shrunk posterior and CWA per 100 games.

## 7. Honest limits

- **"Big" means big within a game.** Leverage is demeaned inside each
  (player, game) cell, so a decider's 10-10 and a dead rubber's 10-10 count
  alike. Whether players elevate in *big matches* is a different question
  this does not touch.
- **Leverage rises with time in a game**, so "clutch" and "starts slow, plays
  better late" are not separated here. The null reproduces the score path so
  it is not free to fake either, but a genuine within-game warm-up effect
  would land in this statistic as clutch.
- The simulator assigns serve rotation and receiver position semi-randomly.
  Real teams stack, so receiver identity does not follow court geometry
  (no parity rule reproduces it in more than 11% of game-sides). Observed
  within-team shares are close to even (serve 41/59, receive 46/54) and the
  measured effect is insensitive to the assignment rule, but this is the
  least faithful part of the null.
- Everything is rally-outcome level. There is still no shot-level data
  anywhere in this stack.

## 8. Reproduce

```
python model/clutch_leverage.py --fetch      # rally cache from Supabase
python model/clutch_leverage.py              # the UNCORRECTED statistic
python model/clutch_mechanical.py            # why that statistic is broken
python model/clutch_null.py --reps 60 --model --out clutch_null_model
python model/clutch_report.py                # the answer
python model/clutch_selfcheck.py             # zero in -> zero out
python model/clutch_power.py                 # what we could have detected
```
