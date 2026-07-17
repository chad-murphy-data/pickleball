# Pre-registration: momentum in pro pickleball rallies

**Frozen 2026-07-17 (Pacific), BEFORE any sequential statistic was computed.**
Committed together with the frozen analysis code (`model/momentum_test.py`);
the analysis runs ONCE on the sample defined below, and every pre-registered
number is reported regardless of outcome. Grading lives in
`model/receipts.json` (entry `2026-momentum-prereg`).

## Disclosure of what has already been observed

Before this registration we computed, from the same referee-log corpus:
aggregate serve-rally win rates (k ≈ 0.444 MLP doubles / 0.449 PPA doubles /
0.537 PPA singles), per-match rally/point tallies, score validation, and a
win-probability calibration-by-decile table. **No sequential statistic —
no autocorrelation, streak, timeout, or pressure quantity — has been
computed on any of this data.** k enters the analysis only as a nuisance
baseline.

## Question

After conditioning on the mechanical structure of side-out scoring (serve
state) and team strength, do rally outcomes show momentum — dependence on
recent rally history — and do timeouts or game-point pressure move the
serve-rally win probability?

## Sample (frozen)

- Corpus: `raw/match_logs` referee logs (open-BFF `getListLogs` cache).
- Population: rows of `data/match_rally_summary.csv` as committed in
  `0799d6c` with `discipline == doubles` AND `score_check == "ok"` AND all
  four players of the match present in `data/v2_players.csv`.
- Sample: `random.Random(20260717).sample(sorted(population), 2500)` —
  2,500 matches, expected ≈ 120k resolved rallies.
- No other data may be added; matches whose log fails to parse are dropped
  and counted in the report.

## Rally construction (identical to the validated harvest tally)

A resolved rally is a `log_type 12` row paired with the NEXT
`14`/`16`/`23` row in `log_index` order within the same match. `y_srv = 1`
iff the outcome row is a type-14 with positive reconciled delta; type-14
correction rows (negative/zero delta) resolve no rally and reset the
possession counter. Per rally we record: game number; serving side and
server number (from `server_uuid` + the score string); server and receiver
scores (from the type-12 start string, server perspective); `y_A` = rally
won by team one. Derived regressors:

- `eta_m`: v2 `team_eta` for the match's pairings (team-one perspective);
  `eta_srv` = `eta_m` signed to the serving side.
- `margin` = server score − receiver score, clipped to [−7, +7].
- `srv2` = 1 if second server.
- `poss1`, `poss2p`: 1 or ≥2 points already won by this server in the
  current service possession (0 = first rally of the possession; counter
  resets on side-out, second-server pass, correction, or game boundary).
- `trail5c` = (share of the previous 5 resolved rallies in this game won
  by team one) − 0.5; requires a full window of 5, else the rally is
  excluded from M2 only.
- `lag1_A` = team one won the previous resolved rally (M2 secondary; note:
  mechanically collinear with state A1 (≡1) and B1 (≡0) — identified from
  the A2/B2 cells; this is understood and intended).
- `to_recv`, `to_srv`: a type-18/35 timeout row occurs after the previous
  rally's outcome row and before this rally's type-12 row, attributed via
  the log's own team→side map to the current receiving / serving side.
- `gp_srv`: server score ≥ T−1 AND lead ≥ 1 (winning this rally wins the
  game, win-by-2 respected); `gp_recv`: the mirror for the receiving side.
  T ∈ {11, 15} from `games.csv scoring_format`.

## Models and estimands (all frozen)

Logistic regressions, MLE via IRLS; **cluster-robust (CR0) standard errors
clustered by match**; z-tests. Effects reported as **average partial
effects (APE)** on the probability scale, with 99% CIs.

| id | model | estimand (primary in bold) |
|---|---|---|
| M1 | `y_srv ~ 1 + srv2 + eta_srv + margin + poss1 + poss2p` | **φ_poss = APE(poss2p)** — server heating up within a possession |
| M2 | `y_A ~ 1 + state(A2,B1,B2) + eta_m + trail5c` | **φ5 = APE(trail5c per +0.2)**, i.e. one extra win in the last five; secondary φ1 = APE(lag1_A) from the same spec with lag1_A |
| M3 | M1 + `to_recv + to_srv` | **τ_recv = APE(to_recv)**; secondary τ_srv |
| M4 | M1 + `gp_srv + gp_recv` | **β_gp = APE(gp_srv)**; secondary β_gp_recv |

## Decision criteria (frozen)

Four primary tests, each at **α = 0.01** (two-sided), PLUS a practical
threshold — both must hold to claim an effect:

- H1 (possession momentum): effect iff p < .01 AND |φ_poss| ≥ 1.0 pp.
- H2 (cross-possession momentum): effect iff p < .01 AND |φ5| ≥ 1.0 pp.
- H3 (timeout): effect iff p < .01 AND |τ_recv| ≥ 2.0 pp.
- H4 (game-point pressure): effect iff p < .01 AND |β_gp| ≥ 2.0 pp.

If the 99% CI lies entirely inside ±threshold → "**no meaningful effect
(bounded)**". If neither criterion resolves → "inconclusive". Secondary
estimands are reported with CIs, never promoted to headline claims.

## Registered predictions (graded in receipts.json)

| hypothesis | prediction | prob |
|---|---|---|
| H1 | no meaningful possession momentum (bounded or non-sig) | 0.85 |
| H2 | no meaningful cross-possession momentum | 0.85 |
| H3 | no meaningful timeout effect | 0.80 |
| H4 | no meaningful game-point effect | 0.70 |
| H4 direction if an effect exists | negative (servers convert WORSE on game point) | 0.20 of the 0.30 |

## Validation before unblinding

`model/momentum_test.py --selftest` simulates iid side-out games (known k,
eta heterogeneity, random timeouts) and must recover all four APEs
statistically indistinguishable from zero before the real run. Synthetic
only — no real logs are touched by the selftest.

## What gets reported no matter what

All four primary APEs + 99% CIs + p-values, both secondaries, sample
counts (matches parsed/dropped, rallies, timeout events, game-point
rallies), and the honest caveats: logged matches are not a random sample
of pro pickleball (digital-refereeing coverage is event-dependent), and a
significant τ_recv admits a selection interpretation (timeouts are called
at non-random moments) — under the iid null the test is still exact.
