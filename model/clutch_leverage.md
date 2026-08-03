# Clutch, measured against a null that fights back

*Session 2026-08-03. Supersedes `model/clutch.md`. Code: `clutch_leverage.py`
(measure), `clutch_mechanical.py` (why the naive measure is broken),
`clutch_null.py` (the null), `clutch_report.py` (attributed answer),
`clutch_team.py` (**the doubles answer**), `clutch_selfcheck.py`,
`clutch_power.py`, `clutch_circularity.py` (two objections, tested).
Tables: `data/clutch_leverage.csv`, `data/clutch_team_doubles.csv`.*

## TL;DR

**Clutch is real, and it is about a third of what the naive statistic says.**

Two things had to be fixed to see it, and they pull in opposite directions:

1. **Side-out scoring fakes clutch, and the fake scales with skill.** A
   service run is a string of wins ending in exactly one loss, at the run's
   highest score and so its highest leverage. That manufactures a
   leverage/outcome covariance at a true effect of zero. A simulated league
   where nobody is clutch reproduces the entire observed spread and
   correlates **+0.65 with v2 rating**. This is what `model/clutch.md`
   measured; its named list is retracted.
2. **Crediting a doubles rally to whoever served or returned throws most of
   the signal away.** Four players contest the rally. Injecting *team-level*
   clutch of tau = 0.010 and measuring it through server/receiver attribution
   recovers **0.0042 and finds zero significant players**. Attribute nothing
   instead — the rally belongs to the side — and it comes back.

After both fixes: **tau = 0.0050 [0.0032, 0.0066] in doubles and 0.0069
[0.0043, 0.0088] in singles**, var(z) = 1.56 against a chance value of 1.00,
rising to **2.10 among players with 300+ games** — the excess sharpens as
precision improves, which is what a real effect does and noise does not.

**Ben Johns (z = 5.28) and Anna Leigh Waters (z = 5.00) are genuinely
separated from the field.** They were also top of the retracted list, for the
wrong reason; they survive for the right one.

Clutch does **not** transfer between doubles and singles (r ≈ −0.10). It is
not a portable personality trait.

---

## 1. The measurement

1,464,258 rallies — every logged rally in the warehouse, 2024-01 to 2026-07:
969,470 PPA+MLP doubles, 494,788 PPA singles, 41,693 games, 20,181 matches.
(The previous pass used 162,942, one spring, serve side only.)

**Leverage** L_r = |P(win game | win rally) − P(win game | lose it)|, exact
from the serve-aware side-out DP in `web/sitelib/winprob.py` at eta = 0, so
leverage is a property of the *situation* and never of the players in it.
Measured k: 0.4383 doubles, 0.5254 singles. Biggest rally on the board: 0.463.

**Clutch** is the within-cell covariance of leverage and outcome,
b = sum_g Sxy_g / sum_g Sxx_g, with both leverage and outcome demeaned inside
each cell. That makes it immune by construction to how good the player is,
how good their team was that day, which games they played, and how much
high-leverage exposure they get. Units: extra probability of winning a rally
per +1 sd of leverage. The permutation null has a closed form
(V = Sxx·Syy/(n−1)), verified against shuffling at var(z) = 1.00.

**The cell is the whole game, and the outcome is the SIDE's**
(`clutch_team.py`). Both partners get the same number from a game, so
individuals are identified only by partner rotation across games — the same
channel the v2 rating model uses, and the same reason the house rule says
actor and partner effects are not separable within a pairing. That is what
"the rally is a team outcome" actually implies, not a defect of the
estimator. In singles the side is one person and this reduces to the exact
per-player statistic.

## 2. The artifact

`clutch_mechanical.py`: a league under real side-out rules where outcomes
depend only on the server's own rate and nothing depends on leverage.

| | real archive | no-clutch simulation |
|---|---|---|
| serve-channel sd across players | 0.0132 | 0.0134 |
| return-channel sd across players | 0.0127 | 0.0120 |
| var(z) | 2.0 – 2.4 | 2.3 – 2.9 |

The bias would be harmless if it were equal for everyone. It is not: better
servers have longer runs, so the one terminating loss is diluted. Under the
full null the artifact correlates **+0.87 with the player's own serve-win
rate**. That was the old leaderboard.

## 3. The null

`clutch_null.py` replays **every real game with its real players** under the
real rules, 60 times, nobody clutch. Correction: b_adj = b_obs − mean(b_null).

Ability model matters. Flat archive-wide rates leave tau = 0.0090; an
opponent-aware `logit p = a[server] − r[receiver]` leaves 0.0031 through the
attributed estimator. The opponent-aware null is the better model of reality
(1,508,519 simulated rallies against the real 1,464,258; the flat null
overshoots to 1,557,989). The correction drives the statistic's correlation
with v2 rating from +0.412 to **+0.026** and with own serve rate from +0.322
to −0.088.

## 4. Two objections, tested rather than argued

`clutch_circularity.py`, injected tau = 0.010.

**"The null is circular"** — its abilities are fitted from the same rallies
that contain the clutch, so a clutch player's simulated twin is already
partly clutch and we subtract real signal.

| null abilities | recovered tau | players at p<0.05 |
|---|---|---|
| clean (uncontaminated) | 0.00981 | 43 |
| **refit on the contaminated season** | **0.00864** | 34 |

Real, and quantified: the circle costs about **12% of the effect** and 20% of
the detections. **Every tau in this file is biased low by roughly that much.**
It does not come close to erasing detection.

**"Server/receiver attribution is meaningless in doubles"** — correct, and
worse than it looks:

| injection | arm | recovered tau (injected 0.010) | players at p<0.05 |
|---|---|---|---|
| individual | doubles | 0.00957 | 19 |
| individual | singles | 0.01074 | 13 |
| **team-level** | **doubles** | **0.00415** | **0** |
| team-level | singles | 0.01246 | 27 |

If clutch acts at the team level — the only coherent reading of a four-player
rally — the attributed estimator sees 41% of it and names nobody. This is why
the first version of this analysis concluded there was no clutch. It was
measuring doubles through a lens that could not see it, then pooling that
with singles.

## 5. The result

Team-outcome estimator, opponent-aware null, 30 replicates:

| arm | players | var(z) | tau | 95% CI |
|---|---|---|---|---|
| doubles | 336 | 1.562 | 0.00503 | [0.00321, 0.00662] |
| singles | 135 | 1.556 | 0.00688 | [0.00433, 0.00884] |
| pooled | 419 | 1.176 | 0.00281 | [0.00000, 0.00451] |

Pooling *dilutes*, because doubles clutch and singles clutch do not correlate
(r ≈ −0.10). Whatever this is, it is context-specific.

The excess sharpens with precision, which is the signature that matters:

| doubles players with | n | var(z) | \|z\|>1.96 | expected by chance |
|---|---|---|---|---|
| ≥60 games | 336 | 1.56 | 26 | 17 |
| ≥150 games | 170 | 2.01 | 19 | 8 |
| ≥300 games | 81 | 2.10 | 12 | 4 |
| ≥600 games | 28 | 2.15 | 4 | 1 |

Supporting reliability (each against its own simulated null floor, because
two slices of one season are not quite independent):

| test | observed | null floor |
|---|---|---|
| doubles, 2024-25 vs 2026 | +0.171 [−0.011, +0.325] | +0.037 ± 0.078 |
| singles, serve vs return | +0.207 [+0.073, +0.317] | +0.068 ± 0.054 |
| singles, serve vs return, skill removed | +0.160 [+0.026, +0.283] | +0.075 ± 0.053 |

Positive, consistently, but marginal — this sits near the resolution limit of
the archive for individuals, which is exactly what tau ≈ 0.005 implies.

## 6. Who

`data/clutch_team_doubles.csv`. Doubles, ≥60 games, 7 of 336 clear 95%.

| | player | games | clutch | z |
|---|---|---|---|---|
| 1 | Ben Johns | 1,148 | +0.0126 | 5.28 |
| 2 | Anna Leigh Waters | 1,077 | +0.0125 | 5.00 |
| 3 | Gabriel Tardio | 935 | +0.0074 | 3.08 |
| 4 | Genie Erokhina | 434 | +0.0078 | 2.98 |

Johns and Waters are 5 sigma up in a 336-player field where the null's
typical maximum is around 3.2. That is not a multiplicity accident. Below
them the list thins fast and the names with fewer than ~300 games should be
read as noise.

**Size check.** tau = 0.005 means the spread of true clutch across the field
is half a percentage point of rally-win probability per sd of leverage.
Johns at +0.0126 is worth on the order of **one extra game per hundred**
purely from winning big points rather than small ones at the same total
points. Real, repeatable, and small — nothing like the folklore.

## 7. Honest limits

- **"Big" means big within a game.** Leverage is demeaned inside each cell,
  so a decider's 10-10 and a dead rubber's 10-10 count alike. Whether players
  elevate in big *matches* is a separate question this cannot see, and
  `data/games.csv` carries a `stage` column that would support it — the
  obvious next thread.
- **Uniform clutch is invisible by construction.** If a player's composure
  makes them better on *every* rally rather than differentially on big ones,
  a within-player estimator cannot distinguish that from skill, and neither
  can any other outcome-based method. What is measured here is the *tilt*.
  That is the colloquial meaning ("rises to the occasion"), but it is not the
  only thing the word could mean.
- Leverage rises with time in a game, so "clutch" and "plays better late" are
  not separated.
- All tau values are ~12% low (§4).
- Doubles individuals are identified only by partner rotation; within a fixed
  pairing, partners are indistinguishable.
- The simulator assigns serve rotation and receiver position semi-randomly.
  Real teams stack, so no parity rule reproduces receiver identity in more
  than 11% of game-sides. Observed within-team shares are near even and the
  team estimator does not use attribution at all, but this remains the least
  faithful part of the null.

## 8. Reproduce

```
python model/clutch_leverage.py --fetch      # rally cache from Supabase
python model/clutch_mechanical.py            # why the naive statistic is broken
python model/clutch_null.py --reps 60 --model --out clutch_null_model
python model/clutch_team.py --reps 30        # THE ANSWER (no attribution)
python model/clutch_report.py                # attributed version, for contrast
python model/clutch_circularity.py           # the two objections
python model/clutch_selfcheck.py             # zero in -> zero out
python model/clutch_power.py                 # what could have been detected
```
