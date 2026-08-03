# Clutch at 9-9 — the one slice where it isn't null

*Session 2026-08-02. Code: `model/clutch_at_99.py`. This qualifies
`clutch.md` §5, which concluded clutch has "zero game-prediction value."
That verdict stands **as stated** — but it was earned entirely on
start-of-game pricing, and it does not survive conditioning on 9-9.*

## TL;DR

**Conditional on a game reaching 9-9, the clutch differential does carry
independent information about who wins — after controlling for skill.**
It's small, it sits at the edge of what this design can resolve, and about
a third of the raw signal turned out to be a mispriced baseline rather than
clutch. But it is not zero, and it is positive at every grain tested.

| arm (CLEAN subset, n=1,405) | corr | cluster-CI |
|---|---|---|
| clutch differential | +0.071 | [+0.022, +0.119] |
| **skill placebo** (zero clutch content) | **+0.052** | [+0.003, +0.102] |
| **clutch orthogonal to skill** | **+0.056** | [+0.006, +0.104] |
| shuffled-clutch null | — | [−0.052, +0.049] |

## Why start-of-game and 9-9 are different questions

`clutch.md`'s mechanism for why clutch keeps coming back null is exact and
correct: clutch is the covariance between leverage and outcome,
**mean-zero within each game by construction**. Winning a bigger share of
the big points means winning a smaller share of the small ones, for the
same total points, hence the same games.

That argument depends on the whole game being in scope. **Conditioning on
9-9 removes the low-leverage half.** From 9-9 onward every remaining rally
is high-leverage, so the offsetting downside never gets played. A trait
that is mean-zero over a full game can be strictly positive over a
conditional tail of it. This is the one slice where the construction
doesn't apply — and it's the slice nobody had tested.

All six tests in `clutch.md` §5 priced at the start of a game: projected
toss-ups, final margin ≤ 2, raw / reliable / top / bottom / residual /
closeness. None conditioned on arrival at 9-9.

## Method

* **Population** — every doubles game to 11 in the referee-log archive that
  reached 9-9. 2,198 games; 1,818 with ≥1 rated player; **1,405 outside the
  Jan–May 2026 clutch measurement window** (the CLEAN subset — the 9-9
  rallies inside that window fed the clutch estimate itself, so they leak).
  Every headline number above is the clean one.
* **Baseline** — the exact serve-aware DP (`web/sitelib/winprob.py`)
  evaluated *at* the 9-9 state with the true serve state (which side, first
  or second server), eta from month-of-game `v2_trajectories` values through
  the weakest-link `team_eta`, anchored the way the live engine anchors. The
  baseline already knows skill **and** the serve situation.
* **Test** — does the team clutch differential explain `won − predicted`?

> **RETRACTED 2026-08-03 — the "real bug" in this section is not real.** The
> 9-9 miscalibration below is an artefact of applying `calibration.json`'s
> b=0.9 display flattening to in-sample month-of-game trajectory values.
> Removing it puts the recalibration slope at 0.92–1.09 across every tied
> state. The DP is correctly calibrated; see CLAUDE.md finding #11. The
> skill-placebo result below still stands as a reason the clutch estimate was
> confounded — but the confound is the calibration layer, not the DP.

## The placebo found a real bug (not in clutch — in the baseline)

The skill placebo — swap clutch for the skill differential, zero clutch
content — came back **+0.052, CI excluding zero**. If the 9-9 baseline were
correctly calibrated this should be flat.

**The serve-aware DP underprices the better team at 9-9.** Observed 0.572
vs predicted 0.559 on the clean subset, and skill predicts the residual.
This is a live finding in its own right: `site/live.html` and
`replay_winprob.py` are showing 9-9 numbers that are slightly too flat.
It's consistent with the smoke calibration already noted in CLAUDE.md (top
decile ~88% observed vs ~95% predicted). **Follow-up: refit in-game
calibration; the DP's endgame states need their own anchor.**

That is exactly why the placebo arm is not optional. Roughly a third of the
raw +0.071 was this artifact. Without the control the honest number would
have been overstated by half.

## What survives

Clutch orthogonalized to skill: **+0.056, CI [+0.006, +0.104]**, outside the
shuffled-clutch null of [−0.052, +0.049]. The margin is narrow.

The more convincing evidence is **coherence across independent grains**,
all after skill control:

| grain | n | effect |
|---|---|---|
| game, team-sum clutch, orthogonal to skill | 1,405 | corr +0.056 |
| game, **only the server at 9-9** | 982 | corr +0.058, logit +4.03 |
| rally, every rally at ≥9-9, server's clutch | 6,217 | logit **+1.56**, +1.1 pp |

The rally arm is the right grain — clutch is *defined* on the server, and at
9-9 we know exactly who is holding the ball. Note the raw rally logit is
+3.70 and drops to **+1.56 once skill is controlled**: more than half of
that arm was skill riding along. The controlled +1.56 is the honest number.

## Attenuation: the raw slopes are FLOORS

Clutch is measured with error, and regressing on a noisy predictor biases
the slope toward zero. From the z-distribution (observed variance 0.00091,
sampling-error variance 0.00046), **reliability = 0.50** — so every
coefficient above is roughly **half** its true size.

Disattenuated, the game-level effect is ~**+6.8 pp per sd** of clutch
differential rather than +3.4, and the rally effect ~+2.2 pp rather than
+1.1. The raw numbers are the conservative end of the range, not the
estimate.

## Power: what this design could even see

Injecting known effects into synthetic outcomes and re-fitting (n=1,405):

| injected effect | recovered with 95% confidence |
|---|---|
| +1.2 pp per sd | 12% |
| +2.3 pp per sd | 37% |
| **+3.5 pp per sd** | **68%** |
| +4.7 pp per sd | 90% |
| +7.0 pp per sd | 100% |

The observed effect sits right at the **68% recovery** point. So this
design is modestly powered: it can see an effect this size most of the
time, but not reliably. **A replication that came back flat would not
falsify this** — it would be within the expected miss rate. Treat the
magnitude as uncertain and the sign as the finding.

## Honest verdict

Not null. Not established. **Positive, small, coherent across four grains,
and at the edge of resolution.**

The right characterization: at 9-9 a clutch edge is worth on the order of
**a couple of percentage points** of win probability — real, but an order
of magnitude smaller than the skill gap it travels with, and swamped by
noise in any single match. It is nowhere near strong enough to move a
pre-match number, which is consistent with `clutch.md` §5 being right about
everything it actually tested.

**Do not put this on the site as a headline.** Two things should happen
first: (1) the 9-9 baseline miscalibration gets fixed, since the clutch
estimate is measured against it, and (2) it replicates on the next season's
games, out of window.

## Scope limits

* Doubles to 11, side-out scoring, only. **This says nothing about
  DreamBreakers** — those are rally-scored singles to 21, where the leverage
  curve is a different shape entirely and 9-9 is early-middle rather than
  the highest-leverage score in the sport. DB prediction goes through mean
  roster singles value (CLAUDE.md finding #6), not clutch.
* Server-only, inheriting `clutch.md`'s limit: this is "clutch on your own
  serve." Return-side clutch is unidentifiable in doubles.
* Individual players in the |z| < 1.5 middle remain noise. A team-level
  differential is not a licence to read a single player's number.

## Reproduce

```
python model/clutch_at_99.py          # needs SUPABASE_ANON_KEY
```

Pulls rally states at ≥9-9 from `pb_rally`, caches to
`model/_clutch99_cache_v2.json` (gitignored, regenerable).
