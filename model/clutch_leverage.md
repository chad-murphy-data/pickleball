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

**And it is a minority trait, not a small universal one** (§5b). A
spike-and-slab fit gives a likelihood ratio of **72.2 against "nobody is
clutch", versus a null distribution of 1.0 ± 1.3**, with a fitted shape of
about an eighth of doubles players carrying an effect of sd 0.014 and the
rest carrying nothing. Most importantly, the individuals verify out of
sample: **name the top 40 on 2024-25 alone and that fixed roster comes back
at z = 3.77 on 2026 alone**, against a null that centres at 0.29 ± 1.00 and
never once reached the observed value in 30 no-clutch seasons.

**Ben Johns (z = 5.28) and Anna Leigh Waters (z = 5.00) are genuinely
separated from the field.** They were also top of the retracted list, for the
wrong reason; they survive for the right one.

Read tau as the *population* summary of a trait most players do not have —
quoting it as "everyone has 0.005" is the error the rare-trait tests exist to
prevent.

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

## 5b. It is a MINORITY trait, and the individuals verify

τ and var(z) are the wrong instruments if clutch is something a few players
have and nobody else does: 300 zero players dilute 40 real ones, and the
Gaussian empirical-Bayes prior asserts everybody has a little, which shrinks
genuine outliers far too hard. A field where 5% carry +0.03 and 95% carry
exactly 0 has τ = 0.0067 — arithmetically identical, as a summary number, to
everyone carrying a uniform 0.0067. `clutch_rare.py` asks the rare-trait
questions instead.

**Spike and slab.** Fit b_p ~ (1−π)·δ₀ + π·N(μ, σ²) through known noise.
Doubles: π = 0.127, slab sd = 0.0144, **likelihood ratio against π = 0 of
72.2, versus a null distribution of 1.0 ± 1.3**. Singles: LR 18.2 vs 1.1 ±
1.9. Caveat that matters: π and σ trade off (π = 1 with tiny σ is just the
Gaussian solution), so the null fits also return large π and **π must not be
quoted on its own** — the LR is the test. But the fitted shape is a coherent
description: about an eighth of doubles players carry a real effect of sd
0.014, the rest carry nothing.

**Tail counts** — the "13% are left-handed" test, which does not care whether
the bulk of the field is spread out:

| arm | bar | observed | null median | null 95th | p |
|---|---|---|---|---|---|
| doubles | z > 2.0 | 12 | 6 | 11 | 0.033 |
| doubles | z > 2.5 | 6 | 2 | 4 | <0.033 |
| doubles | z > 3.0 | 3 | 0 | 1 | <0.033 |
| singles | z > 2.0 | 6 | 3 | 4 | <0.033 |

And the excess lives where precision is highest, which is the direction that
rules out miscalibration:

| doubles players with | n | \|z\|>2 observed | null median |
|---|---|---|---|
| 60–200 games | 206 | 9 | 7 |
| 200–500 games | 94 | **11** | 4 |
| 500+ games | 36 | **5** | 1 |

**Select then verify** — the only test that establishes individuals. Name the
top K on **2024-25 alone**, then measure that fixed roster on **2026 alone**,
against the identical procedure run on no-clutch seasons (selection plus
regression-to-the-mean has its own signature under the null, which is what
the comparison controls for):

| arm | K | observed z | null mean | null sd | p |
|---|---|---|---|---|---|
| doubles | 5 | **3.31** | 0.02 | 0.97 | <0.033 |
| doubles | 20 | **2.79** | 0.28 | 0.96 | <0.033 |
| doubles | 40 | **3.77** | 0.29 | 1.00 | <0.033 |
| singles | 5 | 1.99 | 0.02 | 0.77 | <0.033 |
| singles | 20 | 1.20 | −0.14 | 0.99 | 0.167 |

(p is floored at 1/30 — no null replicate reached the observed value.)

Players named on the first two seasons deliver in the third. That is the
claim "these particular players are clutch", tested the way it should be.

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
them the list thins fast, and names with fewer than ~200 games should be read
as noise — that bin shows no tail excess over the null at all (§5b).

The tail is **two-sided**: several players sit at z < −3, i.e. reliably worse
on big points than on small ones. That is reassuring for the effect being a
real trait distribution rather than a one-directional bias, but the
lower-volume negative names are not publishable and the house position on
naming chokers still stands.

**Size check.** tau = 0.005 means the spread of true clutch across the field
is half a percentage point of rally-win probability per sd of leverage.
Johns at +0.0126 is worth on the order of **one extra game per hundred**
purely from winning big points rather than small ones at the same total
points. Real, repeatable, and small — nothing like the folklore.

## 6b. The ledger: what happened, as distinct from what will happen

Two different questions hide under one word, and they take different
statistics. **Is this player clutch?** is a forecast — it has to replicate, it
has to be shrunk, and only Johns and Waters survive it. **How much did this
player actually win in big moments?** is a record. It happened. Requiring a
record to replicate is a category error; nobody asks whether an RBI total is
repeatable before printing it. This is the same call CLAUDE.md finding 9 makes
for the MLP MVP award — pure outcome accounting.

`clutch_ledger.py` reports **CWPA — Clutch Win Probability Added**, denominated
in GAMES: +1.0 is one full game's worth of win probability banked purely from
*when* a side's rally wins landed, at the same total rallies won.

Not shrunk (a ledger is not an estimate of a latent thing) but still
baseline-corrected against the no-clutch simulation — subtracting the
side-out artifact is not shrinkage toward the field, it sets where zero is,
and without it this would be a serve-rate leaderboard wearing a clutch hat.

| # | doubles, career (≥200 games) | games | CWPA | per 100 |
|---|---|---|---|---|
| 1 | Ben Johns | 1,148 | **+26.3** | +2.29 |
| 2 | Anna Leigh Waters | 1,077 | **+21.0** | +1.95 |
| 3 | Gabriel Tardio | 935 | +15.4 | +1.65 |
| 4 | Christian Alshon | 930 | +10.6 | +1.14 |
| 5 | Genie Erokhina | 434 | +10.0 | **+2.30** |
| 6 | Tyra Hurricane Black | 890 | +8.7 | +0.98 |
| 7 | Jorja Johnson | 942 | +8.5 | +0.91 |
| 8 | Tyler Loong | 469 | +8.2 | +1.75 |

2026 alone: Johns +12.2 (+4.45/100), Tardio +7.9, Connor Mogle +4.5,
Anna Bright +3.8. Singles career: Christopher Haworth +8.4, Gabriel Joseph
+6.6, John Lucian Goins +6.3 — note Haworth tops singles while sitting near
the bottom of doubles, the doubles↔singles non-transfer in one person.

Per-player noise sd is printed alongside (±4.9 games for a 1,000-game
career), so a reader can see that most of the list is inside its own error
bar. That is the correct way to publish a ledger: the number is what
happened, the error bar says how much of it to read as skill. Only Johns and
Waters clear their own noise by a wide margin — which is why they, and only
they, also survive the trait test.

Gotcha, found the hard way: the null's U must be rescaled to each player's
OWN leverage sum-of-squares before averaging. Waters' games run short (she
wins fast), so her simulated twins accumulated more leverage than she ever
had the chance to, and an unscaled baseline dropped her from 2nd to 14th.

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
python model/clutch_rare.py                  # is it a MINORITY trait? who verifies?
python model/clutch_ledger.py                # CWPA ledger: what happened
python model/clutch_report.py                # attributed version, for contrast
python model/clutch_circularity.py           # the two objections
python model/clutch_selfcheck.py             # zero in -> zero out
python model/clutch_power.py                 # what could have been detected
```
