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

---

## RESULT — graded 2026-07-16

**St. Louis Shock def. New Jersey 5s, 3–0** (matchup `c81177b9`, played 2026-07-12).
STL swept the three contested lines; MXD2 and the DreamBreaker were never played —
2026 MLP skips the dead 4th game once a side reaches 3–0 (`matchCompletedType` 6 then
14 in the feed).

| line | matchup | P(NJ): v2 / v1 | actual | winner | call |
|:--|:--|--:|:--|:--|:--|
| WDG  | Waters/Johnson vs Bright/Fahey    | 88% / 69.5% | STL 11–6 | STL | ❌ **MISS** |
| MDG  | Khlif/Howells vs Tardio/Patriquin |  8% / 20.9% | STL 11–3 | STL | ✅ HIT |
| MXD1 | Waters/Khlif vs Bright/Patriquin  | 54% / 52.9% | STL 11–8 | STL | ❌ miss (coin-flip) |
| MXD2 | Johnson/Howells vs Fahey/Tardio   | 25% / 38.7% | not played | — | n/a |
| DB   | —                                 |  —  |  —  | — | n/a |

**Overall call: ✅ HIT.** Both models favored St. Louis (v1 57.0%, v2 60.7%); STL won.
Match-level Brier: **v2 0.154, v1 0.185** — both beat the 0.25 coin-flip baseline, and the
validated v2 scored better on the call that matters most.

**Right answer, wrong path.** The forecast expected STL to win *through* MD + MXD2 and a
coin-flip DreamBreaker it named "the hinge" (P(DB)=46.5%). Instead STL closed it out in
regulation by taking the two Anna Leigh Waters lines the model leaned NJ on. "Waters lost
twice" is confirmed: WD 6–11 and MXD1 8–11.

**Calibration, honestly:**
- MDG was a clean, confident HIT (v2 92% STL → 11–3).
- The **WDG call was the big miss** — v2 put NJ at 88% (Waters/Johnson), STL won 11–6.
  On the three played lines that lone error dominates a poor per-line Brier
  (v2 0.358, v1 0.269; n=3, read gently). The *less*-confident v1 scored better line-by-line
  precisely because it hedged WD (69.5% vs 88%).
- Net: STL's aggregate edge was, if anything, *understated* — a 3–0 sweep beats the modal
  DreamBreaker path, consistent with the pre-registered "favorites underconfident" caveat —
  yet one high-confidence line went hard the wrong way. Both lessons logged to
  `model/receipts.md` (entry 1).
