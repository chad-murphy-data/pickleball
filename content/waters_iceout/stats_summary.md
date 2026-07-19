# The Waters/Johnson ice-out — stats backbone

*Everything the model can say about the Mid-Season upset, with receipts.
Numbers from `model/iceout_waters.py` + this session's analysis. Source of
truth for the Reddit / Threads drafts in this folder.*

## The event

- **MLP Mid-Season (Edward Jones Mid-Season Tournament, Grand Rapids), 2026-07-12.** Women's-doubles game of the final.
- **Anna Leigh Waters / Jorja Johnson lost 6-11 to Anna Bright / Kate Fahey.**
- The model (v2: every MLP + PPA pro game, 77% winner accuracy, beats DUPR at 65%) made **Waters/Johnson 88% favorites**. Graded a **MISS** on the site's public receipts — the headline miss of the weekend.
- The eye-test story: Bright/Fahey "iced out" Waters — funneled the ball to Johnson, kept it away from the GOAT.

## The pre-match number, honestly

| quantity | value |
|---|---|
| Model win prob, Waters/Johnson | **88%** (site receipt) |
| Honest loss probability | **~12%, i.e. ~1 in 8** |
| Player values (per-point logit) | Waters 1.80 · Johnson 1.17 · Bright 1.33 · Fahey 1.08 |

Note (receipts honesty): we tested whether to widen this with the model's per-match noise term and the holdout said **no** — it makes calibration worse (Brier 0.1668 vs 0.1658). Games predicted at 90% actually won 91.7%. So ~12% is the validated loss probability, not the ~17% a naive posterior suggests.

## Why 88% and not higher — the weakest-link structure

- Pro doubles teams are **~59% their weaker player, ~41% their stronger** (the fitted weakest-link coefficient, γ = −0.18). A superstar can't carry a team the way a rating gap suggests, because opponents choose who to play — and they play the weaker one.
- So a star's value is **capped by how much of the court they can cover.** Waters/Johnson are "only" 88% because, in a normal game, opponents already tilt roughly **59% of the load onto Johnson** (the weaker side); Waters covers ~41%. That tilt is *already priced in* to the 88%.

## The court-coverage dial (the core of the story)

Model team value = 2 × [ (1−w)·Johnson + w·Waters ], where **w = Waters's share of the court**:

| Waters's share of the court | model win prob |
|---|---|
| 0% — **fully iced out** | **48%** |
| 41% — a normal game (the actual 88%) | 88% |
| 50% — equal coverage | 93% |
| 100% — Waters takes everything | ~100% |

- **Freeze Waters completely and the team plays like two Jorja Johnsons.** Two Johnsons (2 × 1.17) is worth almost exactly what Bright/Fahey's real pairing is worth (2.35 vs 2.35).
- **Sanity check:** literally simulate *two Jorja Johnsons* vs Bright/Fahey → **48%.** Same answer as the freeze, as it must be — freezing the star = replacing her with a second copy of the partner.
- **So the ice-out alone — no bad luck, no misrating — turns 88% into a coin flip.** Freezing the star doesn't just subtract her shots; the weakest-link math drags the whole team's level down toward the partner.

## What it would take to make Bright/Fahey actual favorites

The freeze alone gets to ~50/50. To push past it you need Johnson's on-court level to dip *below* Fahey's — either a genuine skill gap we underrated, or the human part: Johnson pressing under a 20-ball diet, or Waters forcing too much on her rare touches. Both dials together (freeze + a modest dip) comfortably make Bright/Fahey favorites; **neither does it alone.**

## The honest limits (non-negotiable for the posts)

- **We cannot prove the ice-out happened.** The referee logs show Bright/Fahey served *about evenly* to both (Johnson 10, Waters 9) — serve target wasn't the weapon (rotation constrains it). The freeze lives in **mid-rally ball placement**, which **no data source in the sport tracks.** Ceiling on proof = broadcast vision, which we don't have.
- **An 88% favorite still loses ~1 in 8.** One game cannot separate "repeatable exploit" from "the Tuesday the model budgets for."
- **The 6-11 scoreline was moderately lopsided, not freakish:** conditional on losing, the modal loss is 11-9; ~2/3 of losses are 11-8-or-closer; 6-11 sits around the 80th percentile of loss severity.

## The one-line takeaway

**Icing out the GOAT is a structurally sound strategy — not because it tires her out, but because pro teams are mostly their weaker player, so neutralizing the star turns her team into two of her partner.** Whether Bright/Fahey truly executed it or just had a great night, one game can't say. But the *ceiling* of the tactic is exactly this: 88% → a coin flip.

## Numbers appendix (for captions)

- 88% → lost 6-11 · honest loss ~1 in 8
- teams = 59% weaker player / 41% stronger
- fully iced → 48% · two Johnsons → 48% · equal coverage → 93%
- served-to split in the actual game: Johnson 10 / Waters 9 (≈ even)
- model: 77% accuracy vs DUPR 65%; every MLP + PPA pro game since 2024
