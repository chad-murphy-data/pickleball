# Is Haworth's 2026 DreamBreaker record bad luck or a real weak spot? (first pass, 2026-08-24)

Started late-night off a vibe check ("his DBs have not been great"). This
is a first pass, not a frozen finding. Reproduce with
`model/haworth_db_2026.py`. **This took four tries to get the data story
right in the same session — read "how this got here" before extending
it, so the same mistakes don't get repeated.**

## How this got here (four passes)

DreamBreaker rotation (4 players per side, swap every 4 combined points)
is **mandatory** — there is no team-discretion exception. Haworth's 6/28
DB shows it properly logged: explicit `log_type=32` substitution events
name all 4 players per side, reconstruction validates exactly against the
official score. His 7/23 and 7/26 DBs show **zero** `log_type=32` events
and only 2 distinct player uuids each (Haworth + one opponent, the whole
way), and those logs *also* validate exactly against the official final
score.

1. **Pass 1**: read the two-player logs as a mid-season format change to
   single-champion 1v1. Wrong — no such format exists.
2. **Pass 2**: read the missing substitution events as a referee-logging
   schema regression, excluded both matches. Right call, wrong mechanism.
3. **Pass 3**: found other post-7/23 DreamBreakers — including one worked
   by the *same referee* who officiated Brooklyn's 7/26 game — with full
   8-player rotation logged correctly, and concluded from that that
   Brooklyn's two games genuinely didn't rotate (team discretion). Wrong:
   rotation is mandatory, so a clean-looking two-player log still can't
   be real, no matter how well it validates against the final score.
4. **Pass 4 (current, correct)**: rotation happened in all three matches
   — it has to, it isn't optional — but the referee logs for 7/23 and
   7/26 simply **failed to capture it**. A log can be internally
   consistent (correct running score, normal-looking side-out
   alternation) and still be wrong about *who* was on court; score
   validation only proves the point total is right, not that
   `server_uuid`/`receiver_uuid` tracked real substitutions. There is
   currently no reliable way to recover true on-court identity for these
   two matches from this log stream.

**Net**: only 6/28 (12 rallies, one opponent) is trustworthy player-level
data. 7/23, 7/26, and 7/09 (no digital log at all) are all excluded.

## What's left

Brooklyn played **4 DreamBreakers in 2026** (record: 2-2). Haworth
started all 4 (confirmed via the matchup record's `teamOnePlayerOneUuid`,
populated independent of referee logs):

| date | opponent | Brooklyn result | Haworth attribution |
|---|---|---|---|
| 06-28 | JW Johnson | W 23-21 | **trustworthy** — 8-4 of 12 rallies, SUB-log verified |
| 07-09 | John Lucian Goins | L 21-23 | unusable — no digital log at all |
| 07-23 | James Delgado | W 21-14 | unusable — log exists but has no substitution events |
| 07-26 | JW Johnson (rematch) | L 10-21 | unusable — same issue |

He started every one of them; only one is verifiable.

## Expectation vs. actual (the one trustworthy data point)

House DB rally model (`model/db_model.md` v2):
`P(win) = sigmoid(K · (singles_Haworth − singles_opponent))`, `K = 0.42`
(league-wide fit, held fixed). Haworth +1.87 singles vs JW Johnson +1.37 →
model favors him 55.3% per rally. Actual: **8-4 (66.7%)**.

Fitting a Haworth-specific offset on top of the fixed-K baseline (n=12,
one opponent): **delta = +0.48 ± 0.61 logit** (z = 0.79, p = 0.43, 95% CI
[−0.72, +1.68]). That's not evidence he's *good* at DBs — it's a coin
flip's worth of data. The honest statement is: **we currently have no
usable signal on Haworth's 2026 DreamBreaker performance beyond one match
against one opponent, which he won**.

## Why this matters beyond just Haworth

This logging failure would silently corrupt **any** player-level DB query
that trusts a log without checking for substitution events first —
`server_uuid`/`receiver_uuid` on a POINT row is not proof of who was
actually on court just because the score adds up correctly. Anyone
extending `model/db_impute.py`-style reconstruction should require
`log_type=32` presence (or some other independent confirmation) before
trusting player attribution, and should not treat "the final score
matches" as sufficient validation for *who played*, only for *what the
score was*.

## What would actually resolve the original question

- **A different data source for player identity in the unlogged/
  under-logged matches** — broadcast footage, a box-score-by-player if
  pickleball.com publishes one elsewhere, or a support request to
  MLP/pickleball.com about the referee-logging failure.
- **All three excluded matches remain open** — 7/09 has no log at all;
  7/23 and 7/26 have logs that are simply wrong about participant
  identity. None of them can be fixed from data already on hand.
- If a fix or backfill ever becomes available, rerun this with the full
  season. Until then, this is a one-match, one-opponent data point —
  `db_model.md`'s existing caution that "player-level DB effects are
  hopeless at this sample size" applies in full force.

## Bottom line

Not enough trustworthy data to conclude anything about whether Haworth's
2026 DreamBreaker play is good, bad, or average. The one verified match
(6/28, 8-4 vs JW Johnson) beat his singles-model expectation, but n=12
against one opponent proves nothing either way. This needs either a
different data source for the three unattributable matches or more games
before it's answerable — and any future attempt should distrust a
DreamBreaker log's player attribution unless substitution events are
actually present in it.
