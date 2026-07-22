# DreamBreaker imputation shrink — do never-singles players underperform? (2026-07-22)

Never-play-singles players get a singles value imputed from doubles via
`make_forecast.SINGLES_IMPUTE` (`singles ≈ 0.28 + 1.14·doubles`, r = 0.74).
`db_model.md` flagged the obvious selection bias: these players don't play
singles *because* they're worse at it, so the imputation runs high. This
measures the bias directly — **DreamBreakers are singles points** — by
pooling every non-singles player's DB rallies and comparing their actual
rally wins to what the imputation predicts.

## Data & method (`model/db_impute.py`)

- DreamBreakers are **not** in the Supabase rally warehouse (filtered as
  tie-breakers, verified: 0 rows), so fetch the referee logs directly
  (`getListLogs`, cached in `raw/match_logs`). 94 of 103 DBs have rally logs
  → 3,522 rallies.
- DB logs are quirky, and getting this wrong silently corrupts everything:
  the score string is the **team** total (server-team first, `a+b` = rally
  index), and **`receiver_uuid` is unreliable** (it sticks to one player
  across segments). So the winner of each rally is reconstructed from the
  **server rotation + team scores**, and each matchup is read off the two
  distinct servers in its 4-rally rotation segment.
- Restrict to **same-gender rallies** — cross-gender value gaps aren't
  identified (house rule).
- Compare imputed vs real-singles players; fit a logistic for the shrink.

**Internal validation:** the recovered rally-level coefficient is
**k = 0.458**, essentially matching the independent *team-level* DB fit
(0.42, db_model.md). That agreement is the evidence the reconstruction is
sound — an earlier buggy version gave k ≈ 0, which is how the bug was caught.

## Result

| group | rallies | actual win% | expected (from rating) | residual |
|---|--:|--:|--:|--:|
| real singles | 5,218 | 50.1% | 49.8% | +0.3 pp |
| **imputed** | 840 | 49.5% | 51.2% | **−1.7 pp** |

Logistic fit controlling for the value gap:

```
P(win) = sigmoid(-0.075 + 0.458·gap − 0.091·impdiff)
```

The imputed-player penalty is **−0.091 logit ± 0.079** → imputed players
play like their singles value is **~0.20 lower** than the formula gives them.

## Honest limits

- **Not significant**: z = −1.15, p ≈ 0.25, rough 95% CI on the shrink
  ≈ [−0.14, +0.54]. The *sign* matches the prior hunch; the *size* is
  uncertain. Only 666 rallies pit an imputed player against a real one.
- Reconstruction is imperfect (37/94 DBs reproduce the final score exactly,
  most others within 1–3 points); the noise attenuates toward zero, so the
  true effect could be a touch larger.
- iid-per-rally, serve-blind (cancels between groups since both serve ~half).

## Applied

`SINGLES_IMPUTE` intercept **0.28 → 0.08** (subtract 0.20) in
`web/make_forecast.py` and `model/db_order_sim.py`. A *rough, directional*
correction — a better default than the un-shrunk imputation for never-singles
players, explicitly not a certified constant. Refit as DBs accrue. Effect,
e.g.: Jade Kawamoto 1.60 → 1.40, Columbus Sliders (3-of-4 imputed) drops most.
