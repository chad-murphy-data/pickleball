# Is clutch forced to be zero-sum? — asked, tested, and the answer is a warning

*Session 2026-08-02. Code: `model/clutch_regimes.py`, output
`data/clutch_regimes.csv` (795 players). **Nothing here is adopted.** The
result is recorded because the failure is instructive, not because the
numbers are usable.*

## The question

Both existing clutch constructions measure a leverage **gradient**:

```
frozen:  mean(levz × resid)  over a player's serving rallies
SRM:     a coefficient on    levz × (side indicator)
```

`levz` is centred within each game, so both are covariances, and the residual
is taken against a baseline fitted on **all** of a player's points — the big
ones included. That pins their average residual near zero. Beating the
baseline on high-leverage points therefore *mechanically* implies falling
short on low-leverage ones, and a uniformly excellent player scores exactly
0.

So: is "wins the big points" doomed to mean "loses the small ones"?

## The conceptual answer

**No — that tradeoff is imposed by the choice of baseline, not by pickleball.**
Estimate the baseline **out of regime** and the constraint vanishes. Give
every player two levels fitted on *disjoint* rallies:

```
logit P(serving side wins) = offset
    + [LOW-leverage rallies ]  (mL_s1 + mL_s2) − (nL_r1 + nL_r2)
    + [HIGH-leverage rallies]  (mH_s1 + mH_s2) − (nH_r1 + nH_r2)

clutch_u = (mH_u − mL_u) + (nH_u − nL_u)
```

`mL` and `mH` never see the same rally, so nothing constrains their
difference. A player is free to be above baseline in both regimes.

## What happened when I ran it

795 players, top-quartile leverage vs bottom-half. The permutation null
(regime label shuffled within game) gives real sd 0.402 vs null 0.153 —
**2.62×**, so there is real structure. And then:

| | corr with v2 skill |
|---|---|
| low-leverage level | **−0.747** |
| high-leverage level | **+0.792** |
| clutch (high − low) | +0.830 |

corr(low, high) across players = **−0.729**.

**This cannot be true.** It says elite players perform *below their own skill
baseline* on ordinary points and make it all back on big ones — Waters at
−0.662 on small points, +1.506 on big. Nobody believes Anna Leigh Waters
loses routine rallies at a below-baseline rate. Taken at face value the model
claims the entire top of the sport is a collection of coasters who only try
when it matters.

## What it actually means

Removing the within-game centring removed the zero-sum artefact and let a
different artefact in through the same door. **The regime label is a function
of the score path, which is a function of the rally outcomes** — the split is
endogenous to the thing being measured. The centred estimators are immune to
this precisely *because* they only ever compare a player to themselves inside
the same game, which is also exactly what makes them zero-sum.

**So the zero-sum property is not a bug to be fixed. It is the price paid for
identification.** That is the real answer to the question, and it is a more
useful one than a new leaderboard would have been.

I attempted a diagnostic to pin the mechanism precisely (is leverage simply a
restatement of score margin?) and **botched it** — two arrays truncated to a
common length whose orderings did not align, because `build()` skips
flat-leverage games and the re-walk did not. Those numbers are not in this
file and should not be quoted from the session log. The mechanism is
identified in principle but not yet demonstrated here.

## What a real fix would need

An **exogenous** definition of "big point" — one not derived from the realised
score path. Candidates, roughly in order of promise:

1. **Pre-committed score states.** Fix the set of "big" states in advance
   (e.g. all rallies at 9-9 or later, or any game point) and compare a
   player's rate there against their rate in *matched* states from
   *different* games. Still endogenous to the current game, but the baseline
   is not.
2. **Instrument the state.** Use the opponent's characteristics or the draw
   to predict arrival at high-leverage states, and use the predicted rather
   than realised regime.
3. **Match-level exogenous stakes.** Elimination vs round-robin, gold-medal
   match vs early round, deciding game of a matchup. These are set by the
   bracket, not by the rally outcomes — genuinely exogenous, though much
   coarser and far fewer observations.

Option 3 is the cleanest identification in the set and is the one I would
build next. It answers a slightly different question — "does this player
raise their level when the *match* matters" rather than "when the *point*
matters" — but it is a question that can actually be answered without the
estimator eating its own tail.

## Status

Not adopted, not published, not a leaderboard. `data/clutch_players.csv`
remains the index of record. The file `data/clutch_regimes.csv` is kept only
so the failure is reproducible.

## Reproduce

```
python model/clutch_regimes.py      # needs SUPABASE_ANON_KEY; reuses the
                                    # clutch_srm rally cache
```
