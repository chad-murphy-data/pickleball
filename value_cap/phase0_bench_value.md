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

## Rules check: still open, needs Chad

Four things would give the bench real, non-injury value. None are
confirmed against the current MLP rulebook, and nothing in this repo
(CLAUDE.md, IDEAS.md, ROADMAP.md, HANDOFF.md) has prior research on any
of them:

- Mandatory rotation or minimum games played
- Substitution rules mid-tie
- Fatigue provisions across a multi-day event (does a team play
  back-to-back ties in a way that plausibly forces a bench player onto
  the court)
- DreamBreaker eligibility restrictions (is any rostered player excluded
  from being the DB pair, which would make depth mandatory there
  specifically)

These stay open pending Chad's confirmation against the actual
rulebook. The modeling below is a way of making progress *despite* not
having that answer yet, not a substitute for it.

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

- Confirm the four rules items above against the current MLP rulebook
  for next season.
- Confirm team count for next season (sets replacement level, N =
  teams x 3, per gender).
- Decide whether the realistic-use rate should be swept (recommended
  for the first pass) or set once the historical roster-appearance data
  is pulled.

## Data used here

`data/v2_players.csv` (PICKLES v2, per-point logit scale), filtered to
20+ games. No new scraping; this is a read of data already in the repo.
