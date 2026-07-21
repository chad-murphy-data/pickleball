# Clutch in pro pickleball — the whole investigation

*Session 2026-07-20/21. Code: `model/big_points.py` (measure + existence),
`model/clutch_significance.py` (who's confidently clutch), and the
predictive ledger `model/clutch_predictive.py`, `clutch_predictive_subset.py`,
`clutch_residual_test.py`, `clutch_closeness_test.py`. Data:
`data/clutch_players.csv` (full 182-player table). This file is the one
clean story; the scripts are the receipts.*

## TL;DR

**Clutch is real, stable, and descriptive — and has zero game-prediction
value.** Both halves of that are established, not asserted:

- The trait **exists** (the field of players spreads 2.9× wider than
  chance) and is **reliable** for the players who have it (split-half
  r = 0.61 at |z|>1.5, 0.81 at |z|>2.5). The stars own it, and several
  over-deliver even relative to their own elite skill.
- But it **predicts nothing** about who wins. Six independent ways of
  feeding clutch to the model — raw, reliable-only, top-only, bottom-only,
  skill-residualized, and interacted with game-closeness — all come back
  null on the frozen June+ holdout.

The reason both are true: clutch is the *distribution* of a player's
point-wins across leverage, mean-zero within each game by construction, so
it moves wins around *inside* a match without changing the total — and the
total is skill, which the model already has. **Clutch describes *how* the
greats win, not an extra reason they win. Keep it as content; keep it out
of the model.**

---

## 1. How clutch is measured

Every rally in 2,402 completed matches (Jan–May 2026) reconstructed from
the referee logs — **162,942 serving rallies**, players with ≥300 (182
players). For each rally:

1. **Leverage** = |win-prob swing| — how much the game's win probability
   moves depending on whether the serving side wins this rally, computed
   exactly from the serve-aware DP (score *and* serve state). A rally at
   2-1 barely moves it; a rally at 10-10 moves it enormously.
2. **Standardize leverage within the game** (`levz`) — how big this point
   was relative to the others in *that same game*. This is mean-zero
   within every game, which is the whole trick: a uniformly-good player
   earns no clutch credit; only over-performing on the *high-leverage*
   points scores.
3. **Outcome residual** = won − expected (the matchup's serve-win rate
   from the skill gap).
4. **Clutch** = mean(`levz` × residual) over a player's serving rallies —
   the covariance between how big a point was and whether they won it
   above their baseline. **z** = clutch ÷ its permutation-null SD.

Scope note: it's measured on the **server**, so strictly it's *"clutch on
your own serve."* In doubles you can't pin a return rally on one of two
receivers, so we don't.

## 2. Does the trait exist? Yes.

Under pure noise the 182 z-scores would be a standard bell (mean 0,
variance 1). Observed (`clutch_significance.py`):

- **variance = 2.94** — the field is 1.7× wider than chance; **KS p = 0.013**
- tails: **15 players** past z=1.96 (chance: 4.5); **11** past z=2.58 (chance: 0.9)
- mean = 0.01 — no global bias (as it must be; leverage is mean-zero within games)

**Confidently clutch** (Benjamini–Hochberg FDR q=0.05, 7 players; 6 also
clear Bonferroni):

| player | z |
|---|---|
| Anna Leigh Waters | 7.9 |
| Ben Johns | 6.8 |
| Anna Bright | 6.7 |
| Gabriel Tardio | 5.5 |
| Christian Alshon | 4.2 |
| Jorja Johnson | 4.1 |
| JW Johnson | 3.3 |

Loosen to q=0.10 and four more join (Daescu, Funemizu, Rohrabacher,
Staksrud). It's the top of the sport, in roughly the order you'd guess.

**Is it clutch, or just being good?** Clutch correlates ~0.58–0.71 with
overall rating — mostly it *is* skill. Strip skill out (residualize on
rating) and the leftover still has excess variance (1.45× vs 1.0), so a
*separate* clutch-beyond-skill trait does exist in the population — but
only **two individuals** (Ben Johns, Gabriel Tardio) clear significance
for it. So "a clutch gene beyond talent" is real in aggregate, nearly
un-nameable per player.

## 3. Is it reliable? Yes — for the players who have it.

Split-half (does a player's clutch in half their matches hold in the
other half?):

| group | split-half r | same-sign |
|---|---|---|
| all 182 | 0.15 | 55% |
| \|z\| > 1.5 (50 players) | **0.61** | **92%** |
| \|z\| > 2.5 (19 players) | **0.81** | **100%** |

The famous 0.15 is a floor dragged down by ~130 middle players who are
pure noise. For the players with an actual signal it's genuinely stable —
everyone at |z|≥2 agrees in both halves, same sign, similar magnitude.
Clutch is not noise; it's noise *for the middle of the pack* and a real
repeatable trait *at the edges.*

## 4. The residual: clutch relative to your skill class

Because clutch rises with rating, the interesting question per player is
"more or less clutch than your skill predicts" — the residual from the
clutch~rating line (full 182-player fit):

| player | z | rating | residual |
|---|---|---|---|
| Ben Johns | 6.8 | 1.10 | **+4.5** |
| Anna Bright | 6.7 | 1.28 | +3.7 |
| Gabriel Tardio | 5.5 | 0.99 | +3.7 |
| Anna Leigh Waters | 7.9 | 1.78 | +2.6 |
| Christian Alshon | 4.2 | 0.97 | +2.5 |
| Jorja Johnson | 4.1 | 1.16 | +1.6 |

**Johns and Bright over-deliver most; Waters lands closest to trend** —
she's clutch about as much as being the GOAT already predicts, while Johns
is clutch *beyond* even his skill. The residual is a **skill-adjusted**
lens, so two players "below the line" can mean opposite things:

- **Jay Devilliers** (z = 0.0, residual −1.4): plays the big points
  *exactly* at his baseline — a solid pro who's neither elevated nor
  rattled by the moment. Below the line only because his skill predicts a
  small clutch bump he doesn't show. Not a choker (z isn't negative).
- **Jade Kawamoto** (z = +0.6, residual −2.8): genuinely clutch-*positive*
  in absolute terms, but she's *elite* (rating 1.18), and elite players
  usually over-deliver a lot — she does so only a little. Her wins come
  from sustained quality more than big-point heroics. Biggest negative
  residual on the board *despite being positive-clutch.*

(Both have |z|<1.5, i.e. the noisy region — these reads are suggestive,
not reliable individual verdicts. For a |z|>2.5 name they'd be trustworthy.)

## 5. Does it predict games? No — six ways.

All graded on the frozen June+ holdout (n=926), clutch measured on
train-only rallies. Each row adds clutch (some form) as a feature on top
of the skill model and asks whether out-of-sample Brier improves.

| angle tested | script | verdict |
|---|---|---|
| raw clutch, all players | `clutch_predictive.py` | null; overfits to a −0.26 weight, hurts. Mechanism corr(team clutch, win-above-model) = **−0.017** |
| raw clutch, reliable \|z\|≥1.5 | `clutch_predictive_subset.py` | optimizer sets the weight to **exactly 0.000**; Brier unchanged |
| top-only (z≥1.5) / bottom-only | `clutch_predictive_subset.py` | null (top corr −0.042, bottom +0.025) |
| residual (clutch beyond skill) | `clutch_residual_test.py` | apparent −0.0004 Brier, but a **control with zero clutch** (skill-only, same players) matches it exactly → it's a rating recalibration, not clutch |
| clutch × game-closeness | `clutch_closeness_test.py` | holdout toss-ups tease +0.22, but the **1,237 train toss-ups give −0.03** → false positive, fails replication; actually-close games +0.001 |

**Why it keeps coming back zero:** clutch measures the *distribution* of
your point-wins across leverage — mean-zero within a game by construction.
Winning a bigger share of the big points means winning a smaller share of
the small ones, for the **same total points**, hence the **same games**.
Skill sets the total; clutch just rearranges it inside the match. A real,
stable trait can be perfectly game-neutral, and this one is.

The clutch-closeness test is the cleanest cautionary tale in the set: the
holdout's 158 toss-ups showed a tempting +0.22 that would have made a
"clutch wins the close ones!" headline — and the training set's 1,237
toss-ups, with 8× the power, saw nothing. Replication caught the mirage.

## 6. Flavor: the biggest point in pickleball

Between evenly matched teams, the single highest-leverage rally is being
**down 9-10 and receiving on the opponent's second server — a 0.467
(≈47%) swing in win probability.** Nearly half the match on one rally.
Its mirror images (10-9 and 11-10 on your own second server) are just as
big.

## 7. Honest limits

- **Server-only** — "clutch on your own serve," not full-rally clutch.
- The **anti-clutch tail** (players well below zero) is real but is
  lower-profile players on smaller samples; we keep those names private
  (naming chokers is punching down and thinly supported).
- For any **individual in the |z|<1.5 middle**, clutch is within noise —
  the reliability lives at the edges.
- Clutch tracks skill: it correlates r = 0.71 with our overall player
  rating across the top-50 players, so most of "who's clutch" is just
  "who's good" — the residual (§4) is the part worth naming.

## 8. Reproduce

```
python model/big_points.py --clutch          # measure + existence + biggest points
python model/clutch_significance.py           # FDR list + z-distribution test
python model/clutch_predictive.py             # raw clutch: no game value
python model/clutch_predictive_subset.py      # reliable / top / bottom: no game value
python model/clutch_residual_test.py          # residual: recalibration artifact
python model/clutch_closeness_test.py         # close games: false positive
```
