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

---

# ADDENDUM H3b — windowed timeout effect (frozen 2026-07-17, later the same day)

Registered AFTER the main results were unblinded, BEFORE any **windowed**
post-timeout statistic was computed (none has been). Motivated by the
user's hypothesis that a timeout's value is a strategic adjustment that
expresses over several rallies, not the single next rally H3 tested.
Explicitly NOT the naive before-vs-after comparison: conditioning on the
timeout's trigger (a bad run) makes "after minus before" positive under a
pure null by regression to the mean — the self-test demonstrates this
numerically. The comparison is treated vs UNTREATED MOMENTS OF THE SAME
KIND.

**Sample:** the same frozen 2,500-match sample and rally construction.

**Moments:** every resolved rally t with ≥10 resolved rallies before it
in the same game (full pre-window) and ≥10 from t onward (full post-
window). Perspective P = the RECEIVING side of rally t. Moments where the
serving side called a timeout are excluded from both arms.

**Treatment:** `to_recv = 1` at rally t (receiving side called timeout in
the preceding break). **Control:** no timeout in the preceding break.

**Outcome:** Y = point differential from P's perspective over rallies
t..t+9 (server scores +1 to its side on a rally win; side-outs score 0).

**Model:** OLS, cluster-robust (CR0) by match:
`Y ~ 1 + treat + prew10 + margin + srv2 + eta_recv`, where `prew10` is
P's point differential over rallies t−10..t−1, `margin` is P's score
minus opponent's (clipped ±7), `srv2` the opponent's server number,
`eta_recv` P's v2 eta. **θ = coefficient on `treat`.**

**Secondary:** same with 5-rally windows (pre and post).

**Criteria (frozen):** effect iff p < .01 AND |θ| ≥ 0.5 points per
10-rally window. 99% CI inside ±0.5 → "no meaningful effect (bounded)".
Otherwise inconclusive.

**Registered prediction:** no meaningful effect, **prob 0.75** (if an
effect exists: positive/helps the caller 0.15, negative 0.10).

**Also reported, descriptive only:** the trigger profile — the
distribution of `prew10` at treatment vs control moments.

**Validation before unblinding:** `model/momentum_h3b.py --selftest`
simulates iid play with ENDOGENOUS timeout calling (hazard increasing in
the caller's recent deficit, zero true effect) and must show (a) the
naive after-minus-before difference is strongly positive (the trap), and
(b) θ statistically indistinguishable from zero.

---

# ADDENDUM H3c — the steelman: rallies, not points (frozen 2026-07-17)

Registered after H3b's bounded null, BEFORE any rally-share post-timeout
statistic was computed. Rationale (user's steelman): in side-out scoring
the receiving side's first win is killing the server's possession —
rallies won — which a point differential cannot see until converted.
H3 tested rallies at window 1; H3b points at window 10; this tests the
strongest cell, rally share at window 10.

**Discipline:** this is the THIRD timeout test on the same sample. It is
hereby the FINAL one on this sample: no further timeout estimands will be
computed on these 2,500 matches. If any registered cell below crosses its
bar, the claim is made ONLY after replication on a freshly drawn,
disjoint 2,500-match sample from the untouched remainder of the eligible
pool (seed to be drawn as `random.Random(20260718)`).

**Moments, treatment, controls, adjustment:** identical to H3b (same
sample, same moment construction, same regression controls with the
pre-window measured in the SAME units as each outcome).

**Primary:** Y = receiver's share of the next 10 rallies won (a rally is
won by a side if it wins serving, or its opponent loses serving).
θ_share = adjusted treatment coefficient. Effect iff p < .01 AND
|θ_share| ≥ 2.5pp. 99% CI inside ±2.5pp → bounded null.

**Secondary steelman cells (reported, never promoted to headline):**
- S1: 3-rally share — "right out of the huddle".
- S2: possession-killer — P(the interrupted server possession yields no
  further point), same controls.
- S3: severe triggers only — primary model restricted to moments with
  point pre-window ≤ −4 (the timeout at its most desperate/most needed).

**Registered predictions:** primary null 0.75; P(at least one of the
four cells nominally crosses p<.01) ≈ 0.15 under honest multiplicity —
a lone secondary ping without primary support will be reported as noise
pending replication, not as an effect.

**Validation:** `model/momentum_h3c.py --selftest` — same endogenous
zero-effect simulation; the rally-share estimator must recover θ ≈ 0
while the naive rally-share before/after shows the selection artifact.

---

# ADDENDUM H3d — fresh-sample confirmation of the 3-rally hint (frozen 2026-07-17)

H3c's lone secondary lean (3-rally share +0.86pp, p=.045 two-sided — one
nominal ping across ~10 timeout estimands, i.e. the expected yield of
pure noise) is treated per the registered language as noise. This
addendum gives it the one clean shot science allows: **hypothesis
generated on sample A, tested once, confirmatorily, on disjoint data.**
Frozen before any statistic touches the replication pool.

**Sample:** ALL 9,426 eligible matches NOT in the seed-20260717 sample
(the full untouched remainder — superseding the earlier sketch of a
fresh 2,500, decided before unblinding for power).

**Design:** moments, treatment, controls, and construction identical to
H3c. **A single estimand**: θ3 = adjusted treatment effect on the
receiver's share of the next 3 rallies. Nothing else is computed on this
pool.

**Test:** one-sided (direction fixed positive by sample A), α = .01.
Power, registered up front: se ≈ 0.22pp (scaling sample A's 0.43pp by
√(2500/9426)); one-sided .01 needs θ̂ ≥ ~0.51pp; power ≈ 94% if the true
effect is sample A's +0.86pp. A decisive test either way.

**Claim ladder (frozen):**
- Significant → "a REAL but TINY short-window effect: the huddle buys
  under one percentage point of the next three rallies — far below the
  2.5pp meaningful bar. Timeouts still don't 'work'; they're just not
  literally nothing."
- Not significant → the 3-rally hint is retired as noise; the timeout
  program is closed everywhere.
- Either way, the 99% CI is reported, with bounded-null language if it
  sits inside ±2.5pp.

**Registered prediction:** fails to replicate (null), **prob 0.80**.

**Validation:** `model/momentum_h3d.py --selftest` — endogenous
zero-effect simulation fit on the 3-rally share must recover θ3 ≈ 0.
