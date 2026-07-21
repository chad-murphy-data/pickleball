# The Waters/Johnson ice-out — full analysis dossier

*Everything we did and every number we have on the one game:
Anna Leigh Waters / Jorja Johnson **lost 6-11** to Anna Bright / Kate Fahey,
MLP Mid-Season women's-doubles final, 2026-07-12. Reference for building a
story. Numbers reproduce from `model/iceout_waters.py`; deeper method notes
in `model/spec_shootout.md`.*

---

## 1. The seven things we actually did (the "analyses" menu)

Each is a distinct lens; a story can use any subset.

| # | Question we asked | Method | One-line finding |
|---|---|---|---|
| 1 | Can we "weight Johnson more" to move the odds? | Swept the weak-link dial γ | Yes — γ is literally that dial; but symmetric γ can't make BF favorites (Johnson > Fahey) |
| 2 | Did they actually target a player? | Pulled the **referee log** for the game | Serve target was ~even (Johnson 10, Waters 9); the freeze is mid-rally, **invisible to any data** |
| 3 | How surprised should the model have been? | Posterior decomposition into 4 sources | An outright loss was ~1-in-8; **no single dial** makes it un-surprising |
| 4 | How likely was a *lopsided* (11-7-or-worse) loss? | Exact race-score distribution | ~1-in-13; and close losses dominate — 6-11 was ~80th-pctile severity |
| 5 | Is our "honest" loss number right? | **Holdout calibration test** | ~12% is validated; the 17% we'd floated over-disperses (a self-correction) |
| 6 | What does the ice-out *do* mechanically? | Asymmetric freeze + court-coverage curve | The freeze **alone** turns 88% → a coin flip, no luck required |
| 7 | Package it | stats summary, Reddit + Threads drafts, infographic | In `content/waters_iceout/` |

---

## 2. The game (facts)

- **When/where:** 2026-07-12, Edward Jones Mid-Season Tournament (Grand Rapids), MLP, women's doubles game of the final.
- **Result:** Waters/Johnson **6-11** Bright/Fahey. (Part of the losing side's 0-3 final sweep; Bright/Fahey's team won.)
- **The receipt:** the model made Waters/Johnson **88% favorites** (site's public number) — graded a **MISS**, the headline miss of the weekend.
- **The narrative in the wild:** Bright/Fahey "iced out" Waters — funneled the ball to Johnson, kept it from the GOAT.

---

## 3. The ratings (the raw material)

Per-point logit values (v2 model), current:

| player | value | ± sd | role |
|---|---|---|---|
| Anna Leigh Waters | **1.80** | 0.10 | the star |
| Jorja Johnson | 1.17 | 0.09 | the partner (weaker link) |
| Anna Bright | 1.33 | 0.09 | opponent |
| Kate Fahey | 1.08 | 0.08 | opponent (their weaker link) |

- Waters/Johnson skill **gap = 0.63 logit** (large — a 1.7σ pairing). Bright/Fahey gap = 0.25.
- Model scalars used throughout: weak-link **γ = −0.18** (published −0.17), per-match noise **sd_m = 0.35**, dyad chemistry: Waters/Johnson +0.024, Bright/Fahey −0.014.
- Team values (γ applied): **Waters/Johnson ≈ 2.86**, Bright/Fahey ≈ 2.35, **two Johnsons = 2.35**.

---

## 4. Pre-match probability (the "how surprised" numbers)

- Raw model: **p_point = 0.63**, **P(win) ≈ 90%** (plug-in) → **88%** calibrated/published.
- **Honest loss probability ≈ 12% (about 1 in 8).**
- Most likely single outcome: a Waters/Johnson win, **modal score 11-5**.
- P(the exact 6-11) = **0.9%**.

**Decomposition of the loss probability** (add uncertainty sources one at a time):

| model of the world | P(loss) |
|---|---|
| plug-in (skills known exactly) | 10.0% |
| + skill-estimation error | 11.9% ← **the honest number** |
| + γ uncertainty | 12.1% |
| + match shock (sd_m) | 17.6% ← *over-disperses, rejected (see §8)* |

**Takeaway for a story:** an 88% favorite losing one game to 11 is a ~1-in-8 event. The model "knew" it was possible; it wasn't a model failure.

---

## 5. The freeze mechanism (the heart of the story)

**5a. The weakest-link fact.** Pro doubles teams are **~59% their weaker player, 41% their stronger** (γ = −0.18). Opponents choose who to hit, and they hit the weak link — so a star is capped by how much court she can cover.

**5b. The court-coverage curve** — win prob as Waters's share of the court (w) changes:

| Waters's share of the court | model win prob |
|---|---|
| 0% — **fully iced out** | **48%** |
| 41% — realized normal game | 88% |
| 50% — equal coverage | 93% |
| 100% — Waters takes everything | ~100% |

- **The marginal value of one unit of coverage = the skill gap (0.63).** Big gap → coverage is decisive. (Two equal players → coverage wouldn't matter.)
- In a *normal* game opponents already tilt ~59% of the load onto Johnson — that tilt is **baked into the 88%.**

**5c. The asymmetric freeze** (freeze Waters; Bright/Fahey play their normal game) — this is the cleanest number:

| skill shift k (WJ down / BF up, σ) | P(Waters/Johnson win) |
|---|---|
| **0 — freeze only, skills as rated** | **52% (a coin flip!)** |
| 0.5 | 36% |
| 1.0 | 22% |
| 1.5 | 12% |

- **The freeze alone — no bad luck, no misrating — turns 88% into a coin flip**, because 2×Johnson (2.35) ≈ Bright/Fahey's real value (2.35).
- **"Two Jorja Johnsons"** — literally simulating two Johnsons vs Bright/Fahey = **48%.** Same as the freeze (freezing the star = swapping her for a second copy of the partner).
- To make Bright/Fahey *actual favorites* you need the freeze **plus** Johnson's on-court level dipping below Fahey's (skill gap or the human "pressing under a 20-ball diet"). Neither dial does it alone.
- If the freeze were only *partial* (Waters de-weighted, not erased), you're at ~79% and need a real skill miss (k≈1) to reach 50/50 — so **the story hinges on how total the ice-out was.**

---

## 6. The referee-log keyhole (what we could actually see)

We pulled the game's full referee log (it carries server + receiver UUIDs):

- **Bright/Fahey served ~evenly:** to Johnson **10** rallies, to Waters **9.** Serve target was *not* the weapon — rotation constrains who you serve to.
- On those return rallies, Waters/Johnson won 30% when Johnson returned vs 56% when Waters returned — **but n = 19, statistically meaningless.**
- **The real ice-out weapon is mid-rally ball placement (the 3rd/5th/7th ball), and no data source in the sport tracks it.** So the model can show the strategy is *sufficient*; it cannot confirm it *happened*. (Ceiling on proof = broadcast vision, which we don't have.)

---

## 7. The scoreline (was 6-11 a fluke margin?)

Conditional on losing, the distribution of Waters/Johnson's point total:

| loss score | share of all losses |
|---|---|
| 11-9 | 26% (modal loss) |
| 11-8 | 19% |
| deuce (10-12, 11-13…) | 22% |
| 11-7 | 14% |
| **11-6 (actual)** | **9%** |
| 11-5 or worse | 10% |

- **~67% of losses are 11-8 or closer.** Close losses dominate (a per-point favorite that loses usually only just lost).
- 6-11 sat at the **~80th percentile of loss severity** — a moderately lopsided loss, not a typical one, not a freak either.
- P(11-7 or worse) = 3.2% (plug-in) / 4.2% (with uncertainty).

---

## 8. The method receipt (a story beat in itself)

We nearly reported the loss as ~17% (adding the match-noise term). Testing killed it:

- **Holdout calibration test** (926 games): plug-in Brier 0.1666, **+ value uncertainty 0.1658 (best)**, + value + match shock **0.1668 (worse)**, + match shock only 0.1665.
- **calibration.json:** games predicted at 90% actually won **91.7%** (n=230) → a 90% favorite loses ~1 in 11, not 1 in 6.
- **Conclusion:** ~12% is the honest number; adding the match shock double-counts noise and over-disperses. *The out-of-sample holdout vetoed a plausible-sounding correction.*

This is a nice "we hold ourselves to receipts, even against our own first instinct" angle.

---

## 9. Candidate story angles (pick the spine)

1. **"How to beat the GOAT: ice her out."** *(current infographic/Reddit spine.)* Weakest-link → coverage curve → "two Johnsons is a coin flip." The freeze alone flips 88% → 50/50. Strongest, most shareable.
2. **"The model's honest miss."** Receipts culture: we posted 88%, we were wrong, here's us interrogating our own miss instead of hiding it. Trust-building.
3. **"What the numbers can't see."** The serve-log keyhole: we can prove the strategy is *sufficient* but not that it *happened*, because ball-placement is untracked. The honest edge of analytics.
4. **"Skill or luck?"** The decomposition: it would take a perfect storm (heavy freeze + multi-σ skill miss) to make it "expected"; most likely it was a coin-flip-level shock the model already budgeted for.
5. **"6-11 wasn't a fluke."** Loss-severity angle — close losses dominate, and this was an 80th-percentile beating, consistent with a real tactical edge on the night.
6. **"Two Jorja Johnsons."** The single most vivid hook: freezing Waters mathematically turns her team into two of her partner. Could anchor a whole piece.

**The honest limits every version must keep:** (a) can't prove the freeze happened (no ball-placement data); (b) 88% favorites still lose ~1 in 8; (c) one game can't separate a repeatable exploit from variance. The caveat *is* the credibility — especially on Reddit.

---

## 10. Where it all lives

- **`model/iceout_waters.py`** — reproduces §4–§8 (`--json` dumps every number to `iceout_waters_summary.json`). Reusable: edit MATCHUP/OBSERVED to point at any game.
- **`model/iceout_waters.md`** — the method writeup + the §8 self-correction.
- **`content/waters_iceout/`** — `stats_summary.md`, `reddit_post.md`, `threads_post.md`, `iceout_infographic.html` (+ this dossier).
- Weakest-link / γ background: **`model/spec_shootout.md`** and `model/weakest_link.md`.
