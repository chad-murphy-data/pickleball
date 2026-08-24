# Is Haworth's 2026 DreamBreaker record bad luck or a real weak spot? (first pass, 2026-08-24)

Started late-night off a vibe check ("his DBs have not been great"). This
is a first pass, not a frozen finding — and the first version of it
contained a real error, corrected below. Reproduce with
`model/haworth_db_2026.py`.

## CORRECTION (same session)

The original version of this analysis claimed MLP's DreamBreaker format
changed mid-season — rotating 4-player quartet through 6/28, then a
single winner-take-all 1v1 champion from 7/23 on — based on the referee
logs for the 7/23 and 7/26 matches showing only two player uuids
(Haworth + his opponent) for the entire match, with an exact score match.

**That claim was wrong.** Caught by the user: "it was always rotating, we
just calculated it wrong." Checked directly against the matchup API's own
tie-breaker configuration fields:

```
tieBreakerTeamRotation = TEAM_ROTATION_COMBINED_SCORE_EQUALS_POINTS
tieBreakerRotationCombinedPoints = 4
```

**Identical** on the 6/28 match (where rotation is independently confirmed
— explicit `log_type=32` substitution log rows name all 4 players per
side, matching `data/db_rosters.csv`'s Brooklyn roster) and on the 7/23 /
7/26 matches. The rotate-every-4-combined-points rule was never turned
off.

What actually changed: MLP's referee-logging schema, sometime around
7/23, **stopped emitting `log_type=32` substitution events at all** (0 of
them in either newer match's full log, checked exhaustively — every uuid
appearing anywhere in the raw log JSON for the 7/23 match is either
Haworth, his opponent, or a team/referee/group id; no other player ever
appears). Without those events, `server_uuid`/`receiver_uuid` on the
`log_type=14` POINT rows apparently just stay pinned to the **opening**
pair for the whole match, rather than tracking real substitutions. The
log still validates cleanly against the official final score (side-out
alternation looks completely normal) — it's *self-consistent*, just not
*true*. There is currently no way to tell, from this log stream, whether
Haworth played the entire 7/23 and 7/26 DBs or rotated out after the
first 4 points like the format says he should have.

**Consequence**: the 62 rallies drawn from the 7/23 and 7/26 matches are
not usable for player attribution and have been dropped. That leaves only
the 6/28 match — 12 rallies, one opponent — which collapses the sample
from "thin but suggestive" to "not enough to say anything." The whole
headline of the original version (delta −0.32 logit, ~92% posterior
probability of underperformance) doesn't survive this correction.

## What's left

Brooklyn played **4 DreamBreakers in 2026** (record: 2-2), but only one —
**6/28 vs Dallas Flash (JW Johnson), W 23-21** — has referee logs that
reliably attribute rallies to Haworth specifically (old logging schema,
substitution events present, reconstruction validated exactly against the
official score).

| date | opponent | model p(win) | actual |
|---|---|---|---|
| 06-28 | JW Johnson (+1.37 singles) | 0.553 | 8-4 (.667) |

Fitting the same Haworth-specific-offset model as before, now on just
these 12 rallies: **delta = +0.48 ± 0.61 logit** (z = 0.79, p = 0.43, 95%
CI [−0.72, +1.68]). That's not evidence he's *good* at DBs either — it's
a coin flip's worth of data. The honest statement is: **we currently have
no usable signal on Haworth's 2026 DreamBreaker performance beyond one
match against one opponent, which he won**.

## Why this matters beyond just Haworth

This bug would silently corrupt **any** future player-level DB query that
uses referee logs from July 2026 onward, not just this one. The rotation
rule is unchanged; the *data* is unreliable for that period specifically
because of the missing substitution events, not because the game changed.
Anyone re-running `model/db_impute.py`-style reconstruction should check
for `log_type=32` presence before trusting `server_uuid`/`receiver_uuid`
attribution — its absence isn't "no rotation happened," it's "we can't
see who's on court."

## What would actually resolve the original question

- **A different data source for player identity in the new-schema
  matches** — broadcast footage, box-score-by-player if pickleball.com
  publishes one elsewhere, or a support request to MLP/pickleball.com
  about the logging gap.
- **The 7/09 loss remains completely unattributed** (no digital log at
  all, old or new schema).
- If/when the logging gap gets fixed (or backfilled), rerun this with the
  full season. Until then, this is a one-match, one-opponent data point —
  db_model.md's existing caution that "player-level DB effects are
  hopeless at this sample size" applies in full force here.

## Bottom line

**Retracted**: the earlier "Haworth is probably underperforming his
DreamBreaker expectation" read. The data behind it was corrupted by a
referee-logging gap that made two matches look like clean 1v1 data when
they weren't. What's left (one match, 12 rallies, an 8-4 win) is not
enough to conclude anything about whether Haworth's 2026 DreamBreaker
play is good, bad, or average. This needs either a different data source
for the two unattributable matches or more games before it's answerable.
