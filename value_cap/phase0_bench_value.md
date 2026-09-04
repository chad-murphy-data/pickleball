# Phase 0 — does the bench have real value?

Status: 2026-09-02, first pass. This is a scoping document, not a verdict.
Nothing below is closed.

## Working rule for this whole project

**Don't program a dominant strategy deliberately.** One may fall out of
the math once real rules and real parameters go in, and that's a fine,
publishable outcome. But no assumption in this pipeline should be
chosen because it produces (or avoids) a particular answer about which
roster archetype wins. Where a real number isn't confirmed yet, treat
it as a parameter to sweep, not a constant to pick.

## The question the project can't skip

MLP rosters are 6 (3 men, 3 women) but a tie only starts 4. If the 5th
and 6th players are pure injury insurance with zero realistic
playing-time probability, their win contribution is flat at replacement
level and no price curve can create value where none exists. If that's
true, "4 good, 2 floor" isn't one build among several, it's the
dominant strategy, and the honest product is the writeup of that fact,
not a value cap with a curve pretending otherwise.

Whether that's actually true is not decided in this document. It
depends on facts we don't have yet (the real rules) and a parameter we
haven't measured yet (how often a bench player realistically sees the
court for reasons short of a formal rule). Phase 0's job is to lay out
what's confirmed, what's assumed, and what's still open, so nothing
downstream mistakes an assumption for a finding.

## Rules check: three still open, one confirmed

Four things would give the bench real, non-injury value. Three remain
unconfirmed against the current MLP rulebook, and nothing in this repo
(CLAUDE.md, IDEAS.md, ROADMAP.md, HANDOFF.md) had prior research on any
of them before this document:

- Mandatory rotation or minimum games played -- still open
- Substitution rules mid-tie -- still open
- Fatigue provisions across a multi-day event (does a team play
  back-to-back ties in a way that plausibly forces a bench player onto
  the court) -- still open
- DreamBreaker eligibility restrictions -- **confirmed (2026-09-02,
  Chad's direct knowledge of the season): a team picks its own 4-player
  DreamBreaker lineup. It is not constrained to the four players who
  started that tie's regular games.** See the section below -- this
  isn't a minor rules footnote, it's a live, already-adopted strategy.

The first three stay open pending Chad's confirmation against the
actual rulebook. The modeling below is a way of making progress
*despite* not having those three yet, not a substitute for them.

## What actually got crunched: the talent distribution is a data question, not a rules question

Whether depth matters doesn't only hinge on rules. It also hinges on
how uneven the talent pool is. If the gap from your best available
player to your 10th-best is tiny, who's on the bench barely matters
regardless of rotation rules. If the gap is enormous, losing access to
your best player for any reason (rule-mandated rest or not) is
expensive. That part is answerable straight from PICKLES today.

Pulled `data/v2_players.csv` (v2 per-point logit values, players with
20+ games, current-form value), split by gender.

**Men, top 10:**

| rank | player | value | gap to next |
|---|---|---|---|
| 1 | Ben Johns | +1.114 | |
| 2 | JW Johnson | +1.085 | -0.028 |
| 3 | Hayden Patriquin | +1.080 | -0.005 |
| 4 | Gabriel Tardio | +1.071 | -0.010 |
| 5 | Christian Alshon | +1.014 | -0.056 |
| 6 | Andrei Daescu | +0.975 | -0.039 |
| 7 | Eric Oncins | +0.948 | -0.027 |
| 8 | Jay Devilliers | +0.921 | -0.027 |
| 9 | Federico Staksrud | +0.905 | -0.016 |
| 10 | Riley Newman | +0.905 | -0.000 |

Flat. #1 to #10 spans 0.209 total, no single gap bigger than 0.056, no
tier break anywhere in the top ten.

**Women, top 10:**

| rank | player | value | gap to next |
|---|---|---|---|
| 1 | Anna Leigh Waters | +1.799 | |
| 2 | Anna Bright | +1.327 | -0.473 |
| 3 | Jorja Johnson | +1.173 | -0.154 |
| 4 | Jade Kawamoto | +1.156 | -0.017 |
| 5 | Parris Todd | +1.139 | -0.017 |
| 6 | Tina Pisnik | +1.125 | -0.014 |
| 7 | Rachel Rohrabacher | +1.117 | -0.008 |
| 8 | Jackie Kawamoto | +1.088 | -0.029 |
| 9 | Tyra Hurricane Black | +1.088 | -0.001 |
| 10 | Kate Fahey | +1.081 | -0.007 |

Not flat. Two real breaks. Waters clears the field by 0.473, bigger
than the entire spread from #2 down to #10 combined (0.246). Bright
then clears #3 by 0.154, about 3x the typical rank-to-rank step in the
pack below her. From #3 (Jorja Johnson) to #10 (Kate Fahey) it's flat
again, 0.092 total over 7 ranks, same texture as the men's side.

Read as tiers: Waters alone, Bright alone, then genuine parity from
roughly #3 through #10.

This is a measurement, not a conclusion about strategy. It says the
women's replacement-level conversation and the men's replacement-level
conversation are different conversations, because the shape of the
talent pool differs, not because of any assumption we chose. It does
not by itself say concentration risk is priced correctly anywhere, and
it doesn't settle whether "4 good, 2 floor" is correct on the flatter
men's side either — that still depends on the unconfirmed rules above
and on match-level variance a ratings table alone can't show.

## A third bench category the original framing missed: the DreamBreaker specialist

Phase 0 started from a binary: the bench is either "plays for real
reasons" or "pure injury insurance, worth nothing." The DreamBreaker
eligibility confirmation above breaks that binary. Since a team can
name any 4 of its 6 rostered players to the DreamBreaker regardless of
who started the tie's other four games, a player can be **rostered
specifically and only for DreamBreaker duty** -- not hurt, not resting,
not "next man up," just never in the regular-discipline lineup at all,
by design.

This is not hypothetical. Chad's read of the current 2026 season: this
is already the meta among several of the league's stronger teams --
Brooklyn Pickleball Team with Christopher Haworth, the New Jersey 5's
with Federico Staksrud, Dallas Flash with Hunter Johnson. Adoption
skewing toward the top of the league is itself worth carrying forward:
it suggests DB specialization might be a strategy you can only afford
once your other five roster spots are already strong enough to be
tie-competitive without the sixth doing double duty -- a
roster-construction complementarity, not a strategy open to any team
regardless of the rest of its roster. Not established, just noted, the
same way finding 1's per-division γ split got noted before it was
tested.

Real numbers, pulled 2026-09-02. Doubles (v2, `data/v2_players.csv`):
Christopher Haworth ranks **66th of 638** tracked men, value +0.619 --
solid, unremarkable. Singles (suite, `data/singles_players.csv`,
fitted tier): he ranks **1st of 610** tracked men, value +2.068 -- the
single highest-rated men's singles value in the dataset. That's as
clean a specialist profile as the data could hand us, and it lines up
exactly with how Brooklyn is actually using him.

DreamBreakers aren't a rare event either: 71 of 286 MLP ties in 2026
(`data/dreambreakers.csv` vs `data/mlp_matchups_2026.csv`) reached one
-- **24.8%**, roughly one tie in four. A real, mild, singles-driven
edge in a quarter of all ties (`model/db_model.md`: stronger-singles
roster wins 60.4% of them) is not a marginal thing to leave out of a
value model.

### First-cut Phase 1 model already built and checked against this

`value_cap/phase1_value_model.py` (2026-09-02) is a rough first pass at
the actual Phase 1 value model -- see `phase1_first_cut.md` for the
results. It draws the regular-discipline lineup and the DreamBreaker
lineup independently from the same 6-player roster (top 2W+2M by
doubles value for the four regular games, top 2W+2M by singles value
for the DB), which is what lets a specialist's value show up on its
own without any special-casing for "is this a DB-only player."

Without being told anything about how real teams are actually using
their rosters, the model's DB-channel value (V_db) independently ranks
Haworth **1st of 1,033** tracked players, with Staksrud and Hunter
Johnson both in the top 5 -- the same three teams named above. That's
a real, unprompted validation: the mechanism recovers a strategy real
GMs already found, rather than needing to be told the strategy exists.
Treat this as a smell-test pass, not a finished valuation -- the
model's list of open assumptions is as long as Phase 0's.

## Also carried forward: injury/absence as a draw, not a rate

The original bench-value framing (`bench_win_contribution = P(plays) x
V_bench`, below) asks Phase 1 to pick a playing-time probability, which
risks exactly the circularity the working rule warns against: pick a
high number and depth looks valuable, pick a low one and it doesn't.
A cleaner mechanism, for Phase 2 or 3 once real rosters and a real
season schedule are in play: simulate discrete absences directly. Draw
which ties a given starter misses (injury, rest, a scheduling
conflict), swap in the real bench player for those ties, and compare
the season's realized expected ties won against the same schedule with
an emergency replacement-level fill-in instead. Bench value falls out
of the comparison instead of being assumed into it.

This moves the free parameter from an abstract "how often does the
bench realistically play" guess to a per-player **absence rate** --
still a number that has to come from somewhere, but a much more
grounded one: measurable, at least approximately, from actual
withdrawals and lineup changes already in the historical data, or
swept the same way Phase 1's P(plays) is swept if it can't be measured
well enough. Same working rule, moved to a better-grounded parameter.
It also naturally produces a variance story, not just a mean one -- a
roster that leans on one outlier player (see the Waters/Bright tier
read above) will show a fatter bad-draw tail across simulated seasons
than a balanced roster with a similar average, even if their expected
V looks similar on paper. That is the concentration-risk argument from
earlier in this project made concrete instead of asserted.

## What Phase 1 should carry forward: a swept parameter, not a picked one

The honest state after Phase 0: we can't yet say whether the bench has
real value, because the input that would settle it (real rotation or
substitution rules) is unconfirmed. Rather than block on that, or guess
a number that produces a preferred narrative, Phase 1 should carry
bench playing time as an explicit swept parameter:

```
bench_win_contribution = P(plays) x (V_bench - V_replacement)
```

`P(plays)` has two components that must stay analytically separate so
neither gets mistaken for the other:

1. **Confirmed mechanical requirement.** Whatever the real MLP rules
   turn out to require. Zero until Chad confirms otherwise, at which
   point this document gets updated with the real number, not a guess.
2. **An assumed realistic-use rate** (fatigue, injury, a bad-matchup
   swap) — a genuine unknown, not a rule. Phase 1 should sweep this
   across a range (say 0% to 25%) rather than fix it at one number, and
   report whether the roster-archetype ranking is stable or flips
   across that range. If "4 good, 2 floor" wins at every value in the
   swept range, that's a real finding. If it only wins below some
   threshold, that threshold is the finding, and it's more useful than
   a single verdict either way.

One measurable-not-assumed option worth flagging for whoever picks this
up next: `web/make_forecast.py`'s `mlp_rosters()` plus `data/games.csv`
can reconstruct, for actual 2026 rosters, how often each team's 5th and
6th most-used player actually appeared on court historically. That
turns the realistic-use rate from an assumption into a measurement, at
least for the rate the league has revealed so far (rules aside). Not
run this session; flagging it as the better version of the sweep above
once someone has time for the roster join.

## Planned sensitivity check, once Phase 1/2 exist: run it with and without Waters

Given the tier structure above, one cheap falsification test once the
value model and price curve exist: run the full pipeline once with
Anna Leigh Waters in the pool and once with her removed.

- If the #2 tier (Bright) simply steps up and inherits a similar-sized
  premium once Waters is gone, that suggests the curve mechanically
  manufactures an outlier premium regardless of who is actually in the
  pool.
- If the premium compresses instead, that's evidence the gap reflects a
  real, non-reproducible outlier rather than a fitting artifact.

Either result is worth publishing as-is. This is a sensitivity check on
the method, not a referendum on a player, and should be written up that
way regardless of which way it lands.

## Open items only Chad can close

- Confirm the three still-open rules items (rotation, mid-tie
  substitution, multi-day fatigue) against the current MLP rulebook for
  next season. DreamBreaker eligibility is now confirmed, see above.
- Confirm team count for next season (sets replacement level, N =
  teams x 3, per gender; the first-cut model used 20, the real 2026
  franchise count excluding All-Star/Team-country entries).
- Decide whether the realistic-use rate should be swept (recommended
  for the first pass) or set once the historical roster-appearance data
  is pulled.
- Decide whether the DB-specialist complementarity hunch (only viable
  once the rest of the roster is strong enough) is worth testing before
  Phase 2, or left as a noted pattern for now.

## Data used here

`data/v2_players.csv` (PICKLES v2, per-point logit scale), filtered to
20+ games. No new scraping; this is a read of data already in the repo.
