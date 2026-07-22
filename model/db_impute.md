# DreamBreaker correction for non-ranked singles players (v2, 2026-07-22)

Players without a real PICKLE singles record get a singles value imputed
from doubles (`singles ≈ 0.28 + 1.14·doubles`, r = 0.74, n = 543).
db_model.md flagged the selection bias: they don't play singles *because*
they're worse at it. DreamBreakers **are** singles points, so the bias is
measurable: pool every non-ranked player's DB rallies and compare their
actual rally wins to what the imputation predicts, controlling for
opposition strength and their own doubles rating (both enter through the
value gap in the rally logistic).

## v2 parser — explicit, validated reconstruction (`model/db_impute.py`)

v1 inferred winners from score strings and validated only 37/94 matches;
its estimate (0.20) was attenuated by reconstruction noise. v2 uses what
the referee logs actually contain:

- **type-14 POINT rows**: `point_log.team_uuid` = the scoring *franchise*
  uuid; start/end give a correction-aware delta (`harvest_logs._point_delta`
  handles rewinds, phantom double-entries, garbled strings — all observed).
  The payload's per-team cumulative `end_score` is cross-checked against the
  reconstructed running total: `end == total+1` → point; `end ≤ total` →
  duplicate entry, dropped; `end > total+1` → unlogged score gap, credited
  to the team without rally attribution.
- **type-32 substitution rows**: every 4-point rotation is announced
  explicitly (`player_in_uuid`, `player_out_uuid`, `team_uuid`), so the
  on-court singles matchup for every rally is **explicit**, not inferred.
  (Quirk: a segment's 4th POINT row is logged *after* the two sub rows, so
  on-court players are snapshotted at the rally row.) Each team's announced
  rotation **order** falls out for free (`data/db_orders.csv`).

**Validation: 88 of 94 logged DreamBreakers reconstruct the official final
score EXACTLY** (teamOne/teamTwo-oriented via the matchup record). The 6
others also match on totals but carry attribution-risk flags (server not in
the tracked on-court pair, sub-sequence mismatch) and are **excluded** from
the fit out of caution. 9 of 103 DBs have no digital referee log.
Rally-level dataset: `data/db_rallies.csv` (3,213 rallies).

## Empirical bonus: how teams actually order players today

From the 176 announced team-orders: **slot 1 is a man 176/176 times
(100%)**; men fill **96%** of slots 1–2. Anna Bright's premise ("teams put
their men first") is not a tendency — it is the entire league's revealed
strategy. Corollary: only 2.7% of DB rallies are cross-gender (88/3,213).

## Fit

Same-gender rallies only (cross-gender gaps aren't identified — house
rule). "Ranked" = ≥ 10 pro singles games. Logistic on rally outcomes:

```
P(i wins rally) = sigmoid(b0 + b1·gap + b2·impdiff)
gap     = v_i − v_j   (singles value; non-ranked use 0.28 + 1.14·doubles)
impdiff = imp_i − imp_j  (non-ranked indicator)
```

| quantity | estimate |
|---|---|
| rallies in fit | 3,125 (640 imputed-vs-other) |
| empirical rally k (b1) | **0.502 ± 0.077** (team-level fit was 0.42 — consistent) |
| imputed penalty (b2) | **−0.177 ± 0.081** (z = −2.18) |
| **correction (value shrink −b2/b1)** | **0.35** |
| cluster bootstrap (2,000×, by match) | 95% CI [+0.00, +0.74], P(shrink ≤ 0) = **2.5%** |

Sensitivity to the "ranked" threshold: ≥1 game → 0.41 [−0.00, +0.85];
≥30 games → 0.44 [+0.13, +0.84]. Same direction and magnitude throughout;
the effect is *not* an artifact of where the ranked line is drawn.

## Applied

Correction = **0.35**, subtracted from every non-ranked player's imputed
singles value: `SINGLES_IMPUTE` intercept 0.28 → **−0.07**
(web/make_forecast.py, model/db_order_sim.py, model/db_scenarios.py).
E.g. Jade Kawamoto 1.60 → 1.25; Columbus Sliders (3 of 4 imputed) drop most.

## Honest limits

- Borderline-significant at the primary threshold (CI touches 0); the ≥30
  sensitivity clears zero. Sign is consistent everywhere; magnitude has a
  wide CI. Refit as DreamBreakers accrue.
- 640 imputed-vs-other rallies carry the identification.
- iid-per-rally, serve-blind (serve effects cancel between groups — both
  serve about half of rallies).
- Imputation-eligible non-ranked players enter via doubles value; players
  with 1–9 singles games are treated as non-ranked (their nominal singles
  fit is prior-dominated).
