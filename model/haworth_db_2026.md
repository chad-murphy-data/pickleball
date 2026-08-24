# Is Haworth's 2026 DreamBreaker record bad luck or a real weak spot? (first pass, 2026-08-24)

Started late-night off a vibe check ("his DBs have not been great") — this
is a first pass, not a frozen finding. Reproduce with
`model/haworth_db_2026.py`.

## Setup

Brooklyn played **4 DreamBreakers in 2026** (found by joining
`data/dreambreakers.csv` on matchup_id against `data/mlp_matchups_2026.csv`):

| date | opponent | Brooklyn score | result |
|---|---|---|---|
| 2026-06-28 | Dallas Flash | 23-21 | W |
| 2026-07-09 | Chicago Slice | 21-23 | L |
| 2026-07-23 | Miami Pickleball Club | 21-14 | W |
| 2026-07-26 | Dallas Flash | 10-21 | L |

Team record: 2-2. Nothing damning at the team level — this is really about
whether Haworth specifically is dragging on his slice of it.

**Player-level attribution required reconstructing referee logs**, since
DreamBreakers never carry player IDs in the official score record. Pulled
fresh (network) for this session:

- **6/28**: reused the existing cached reconstruction
  (`data/db_rallies.csv`, built by `model/db_impute.py`). This DB used the
  **old rotating-lineup format** — 4 players per team rotate on court, subs
  logged as separate `log_type=32` rows. Haworth's slice was 12 of that
  match's ~42-44 points.
- **7/23 and 7/26**: fetched live via `PBClient.match_logs` — these use a
  **different, newer log schema** with no `log_type=32` rows at all.
  Instead `log_type=14` POINT rows carry `server_uuid`/`receiver_uuid`
  inline, and — this is the interesting bit — **both matches are a single
  continuous 1v1 the entire way, no rotation**. Haworth's opponent never
  changes, and his personal rally count matches the *entire* official
  score (35 rallies on 7/23 = the full 21-14; 31 on 7/26 = the full
  21-10). That's consistent with a **rule change mid-season**: DreamBreaker
  moved from "rotating quartet" to "one designated champion per team plays
  the whole thing." Worth confirming against the rulebook/broadcast, but
  the log format alone (no more per-point substitution to track) is a
  pretty strong tell. **This is a new finding, not previously in
  CLAUDE.md** — flagging it here since it changes what a DB even *is*
  strategically (one player now carries 100% of it, not ~25%).
- **7/09** (the other loss): **no digital referee log at all** — excluded.
  So this analysis covers 3 of Haworth's 4 DBs, all we can currently
  attribute at the player level.

Total: **78 rallies**, 2 opponents (JW Johnson twice, James Delgado once).

## Expectation

Used the house DB rally model (`model/db_model.md` v2): per-rally
`P(win) = sigmoid(K · (singles_Haworth − singles_opponent))`, `K = 0.42`
(the league-wide fit, held fixed — refitting K on Haworth's own 78 rallies
would be circular). Haworth's singles value is **+1.87** on 290 ranked
games — clearly the strongest singles player in any of these three
matchups (opponents: JW Johnson +1.37, James Delgado +0.81). So yes, the
premise "he's playing tough competition" undersells it a little — by the
singles model he should be *favored* in all three, 55-61% per rally.

| date | opponent | model p(win) | actual |
|---|---|---|---|
| 06-28 | JW Johnson (+1.37) | 0.553 | 8-4 (.667) |
| 07-23 | James Delgado (+0.81) | 0.610 | 21-14 (.600) |
| 07-26 | JW Johnson (+1.37) | 0.553 | 10-21 (.323) |

Combined observed: **39-39 (.500)**. Combined model-implied baseline:
**45.1/78 (.578)**. That's the shortfall driving the "not great" feeling —
and it's almost entirely the 7/26 rematch: he beat Johnson 8-4 a month
earlier, then lost to the same guy 10-21.

## Is that signal or noise?

Fit a single Haworth-specific offset `delta` on top of the fixed-`K`
baseline (1-parameter logistic MLE, same shape as the clutch/gap-exploit
player estimators elsewhere in this repo):

- **delta = −0.317 ± 0.227 logit** (z = −1.40, two-sided p = 0.16)
- 95% CI: **[−0.76, +0.13]** — includes zero
- Flat-prior posterior **P(true offset < 0 | data) ≈ 0.92**
- Scale: −0.317 logit ≈ **−7.9 percentage points** of rally win probability
  at an even matchup — a real-sized effect *if* it's real

**Read**: more likely than not he's underperforming his singles-implied DB
expectation, but the 95% interval still touches zero and the two-sided
p-value doesn't clear conventional significance. This is NOT a confident
finding — it's "lean yes, can't rule out chance," and `db_model.md`
already warned that player-level DB effects are close to hopeless at this
sample size league-wide, let alone for one player with 3 matches and 2
opponents.

## What would actually resolve this

- **The 7/09 loss is unattributed** — if Haworth played it and it went
  badly too, the real shortfall is worse than what's measured here; if he
  sat it out, irrelevant. Worth checking whether that match ever gets a
  digital log (older matches sometimes backfill) or whether box-score-only
  reconstruction is possible.
- **Only 2 distinct opponents** (Johnson ×2, Delgado ×1) — the negative
  read is disproportionately one bad rematch vs Johnson. A specific
  Haworth–Johnson matchup problem (scouting, style) reads very differently
  than a general Haworth DB weakness. Can't separate those with n=2
  opponents.
- **The format change matters going forward**: if DBs really are now
  winner-take-all singles instead of a rotating quartet, every future
  Brooklyn DB rides on whoever they send out (presumably Haworth, per his
  `data/db_rosters.csv` M1 slot) — so this question gets *much* easier to
  answer with a few more weeks of games, and matters more (100% of the DB
  now, not ~25%).
- Extending `model/db_impute.py` / `scraper/harvest_logs.py` to natively
  handle the new no-SUB-log schema (it's actually simpler — no
  reconstruction needed, just read `log_type=14` rows directly) would
  make this a five-minute query instead of a bespoke script next time.
  Left as a follow-up rather than done here since it touches shared
  scraper code and this was meant to be a quick look.

## Bottom line

Slight lean toward "he's actually been a bit below where his singles
rating says he should be" (delta −0.32 logit, ~92% posterior probability
of a true deficit), but the sample (78 rallies, 3 matches, 2 opponents) is
too thin to call it proven, and it's driven mostly by one bad rematch
against JW Johnson. Not "just noise, don't worry about it" and not "yes,
he has a DreamBreaker problem" either — genuinely 60/40 pending more data.
