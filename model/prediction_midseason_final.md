# Pre-match prediction: MLP Mid-Season Tournament Gold Final
## New Jersey 5s vs St. Louis Shock — committed 2026-07-12, before first serve

Model: main 2026 SRM values + dyad chemistry; game noise SD 4.70;
DreamBreaker treated as 50/50 (singles is outside the doubles model).

| game | NJ pair | STL pair | pred margin | P(NJ win) | modal score |
|:--|:--|:--|--:|--:|:--|
| WD   | Waters/Johnson | Bright/Fahey     | +2.40 | 69.5% | NJ 11-6 |
| MD   | Khlif/Howells  | Tardio/Patriquin | −3.81 | 20.9% | STL 11-5 |
| MXD1 | Waters/Khlif   | Bright/Patriquin | +0.34 | 52.9% | NJ 11-7 |
| MXD2 | Johnson/Howells| Fahey/Tardio     | −1.35 | 38.7% | STL 11-7 |

P(2-2 → DreamBreaker) = 40.2%.
**Overall: St. Louis Shock 57.0% — New Jersey 5s 43.0%.**

Known biases, stated in advance: holdout calibration says these probabilities
are underconfident (favorites land ~8-10 points above stated in the 65-80%
band), so STL's true edge is likely somewhat larger; the player-tour effects
fit also has Tardio (+0.28) and Fahey (+0.24) as MLP-positive, same direction.
Offsetting wildcard: a DreamBreaker is 40% likely and Waters — the sport's
best singles player — plays for NJ, so the 50/50 DB assumption plausibly
understates NJ overall.

## Addendum: v2 forecast (still pre-match, 2026-07-12 later same day)

Validated v2 model (77.4% holdout), full posterior simulation:
WD NJ 88% (11-5/11-6) | MD STL 92% (11-5/11-4) | MXD1 NJ 54% (11-9) |
MXD2 STL 75% (11-7). Paths: NJ 3-1/4-0 16.0%, DreamBreaker 46.5%,
STL 3-1/4-0 37.5%. Overall: STL 60.7% at DB=50/50; STL 53.8% if
Waters' singles makes the DB 65% NJ. The match hinges on the DB.
