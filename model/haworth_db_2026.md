# Is Haworth's 2026 DreamBreaker record bad luck or a real weak spot? (first pass, 2026-08-24)

Started late-night off a vibe check ("his DBs have not been great"). This
is a first pass, not a frozen finding. Reproduce with
`model/haworth_db_2026.py`. **This took three tries to get the data story
right in the same session — read the "how this got here" section before
touching it again.**

## How this got here (three passes, worth reading before extending)

DreamBreakers officially rotate 4 players per team every 4 combined
points — confirmed via the matchup record's own config
(`tieBreakerTeamRotation = TEAM_ROTATION_COMBINED_SCORE_EQUALS_POINTS`,
`tieBreakerRotationCombinedPoints = 4`), identical across every DB checked
this season. Haworth's 6/28 DB shows that rotation actually happening —
explicit `log_type=32` substitution events name all 4 players per side,
reconstruction validates exactly against the official score. His 7/23 and
7/26 DBs show **zero** `log_type=32` events and only 2 distinct player
uuids each (Haworth + one opponent, the whole way), also matching the
official score exactly.

1. **Pass 1**: read that as a mid-season format change — rotating quartet
   → single 1v1 champion. **Wrong.** The rotation config is identical
   across all three matches; there's no format change to point to.
2. **Pass 2**: read the missing `log_type=32` rows as a referee-logging
   schema regression starting ~7/23, and threw out both matches as
   unattributable. **Also wrong.** Checked ~20 other post-7/23
   DreamBreakers, including one worked by the *same referee* (`6556e1db`)
   who officiated Brooklyn's 7/26 match — every one of them shows full
   8-player rotation with proper substitution logging. The logging
   mechanism plainly still works in this window, for that referee, just
   not in Brooklyn's two games.
3. **Pass 3 (current, correct)**: the rotation *rule* is real and
   constant; *enforcement* is inconsistent match-to-match. The same
   referee who fully rotated a different DB on 7/24 simply let Brooklyn's
   7/26 DB run as a straight 1v1 the entire way. No substitution
   happened, so there's nothing to log — the data was accurate all along.
   Some referees enforce the rotation rule, some don't bother on a given
   night, and nothing in the matchup record predicts which.

**Net**: 6/28, 7/23, and 7/26 are all legitimate single-opponent, full-DB
rally records for Haworth. Only 7/09 (no digital log at all, old or new
schema) stays unattributed.

## Data

Brooklyn played **4 DreamBreakers in 2026** (record: 2-2). Haworth
started all 4 (confirmed via the matchup record's
`teamOnePlayerOneUuid`, populated independent of referee logs):

| date | opponent | Brooklyn result | Haworth's rally record |
|---|---|---|---|
| 06-28 | JW Johnson | W 23-21 | 8-4 (of 12, rotated out for the rest) |
| 07-09 | John Lucian Goins | L 21-23 | **unknown — no digital log** |
| 07-23 | James Delgado | W 21-14 | 21-14 (played the entire DB solo) |
| 07-26 | JW Johnson (rematch) | L 10-21 | 10-21 (played the entire DB solo) |

78 rallies total, 2 distinct opponents (JW Johnson twice, James Delgado
once).

## Expectation

House DB rally model (`model/db_model.md` v2): per-rally
`P(win) = sigmoid(K · (singles_Haworth − singles_opponent))`, `K = 0.42`
(league-wide fit, held fixed — refitting on Haworth's own rallies would be
circular). Haworth's singles value is **+1.87** on 290 ranked games,
clearly the strongest singles player in every one of these matchups
(JW Johnson +1.37, James Delgado +0.81) — the "tough competition" framing
undersells it a little; the model favors him 55–61% per rally in all
three.

| date | opponent | model p(win) | actual |
|---|---|---|---|
| 06-28 | JW Johnson (+1.37) | 0.553 | 8-4 (.667) |
| 07-23 | James Delgado (+0.81) | 0.610 | 21-14 (.600) |
| 07-26 | JW Johnson (+1.37) | 0.553 | 10-21 (.323) |

Combined observed: **39/78 (.500)**. Combined model-implied baseline:
**45.1/78 (.578)**. The shortfall is almost entirely the 7/26 rematch —
he beat Johnson 8-4 a month earlier, then lost to the same guy 10-21.

## Is that signal or noise?

1-parameter logistic MLE for a Haworth-specific offset on top of the
fixed-K baseline:

- **delta = −0.317 ± 0.227 logit** (z = −1.40, two-sided p = 0.16)
- 95% CI: **[−0.76, +0.13]** — includes zero
- Flat-prior posterior **P(true offset < 0 | data) ≈ 0.92**
- Scale: −0.317 logit ≈ **−7.9 percentage points** of rally win
  probability at an even matchup — real-sized if it's real

**Read**: more likely than not he's underperforming his singles-implied
DB expectation, but the 95% interval still touches zero and the two-sided
p-value doesn't clear conventional significance. Lean yes, not proven —
and `db_model.md` already warned that player-level DB effects are close
to hopeless league-wide at this kind of sample size, let alone for one
player with 3 matches and 2 opponents.

## What's still missing / what would resolve it further

- **7/09 (the other loss) is unattributed** — no digital log at all. If
  Haworth played heavy minutes there and it went badly too, the real
  shortfall is worse than measured; if he rotated out early, less so.
- **Only 2 distinct opponents** (Johnson ×2, Delgado ×1) — the negative
  read is disproportionately one bad rematch vs Johnson. A specific
  Haworth–Johnson matchup problem reads very differently than a general
  DB weakness, and n=2 opponents can't separate those.
- **Referee-enforcement inconsistency is itself worth knowing** for any
  future DB work: don't assume a missing `log_type=32` means missing
  data. Check whether the rally count implies real rotation (few points
  attributed to any one player, like 6/28) or a straight 1v1 (rally count
  matches the full official score, like 7/23/7/26) before deciding
  whether a match is usable.

## Bottom line

Slight lean toward "he's actually been a bit below where his singles
rating says he should be" (delta −0.32 logit, ~92% posterior probability
of a true deficit), but the sample (78 rallies, 3 matches, 2 opponents)
is too thin to call it proven, and it's driven mostly by one bad rematch
against JW Johnson. Not "just noise, don't worry about it" and not "yes,
he has a DreamBreaker problem" either — genuinely 60/40 pending more data,
starting with whatever can be recovered about the unlogged 7/09 match.
