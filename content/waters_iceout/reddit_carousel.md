# Ice-out — Reddit post / carousel (the simple version)

*Drafted 2026-08-28, revised same day with author notes. All numbers
verified against current v2 values (`data/v2_players.csv`), γ = −0.1829,
race-to-11 DP + site calibration; repro at the bottom. The dossier/
explainer in this folder stay as the long version. This is the one that
ships.*

## The numbers (all verified today)

| Fact | Number |
|---|---|
| Current women's ranks (v2) | ALW **#1**, Bright **#2**, Jorja **#3**, Fahey **#10** |
| The gap ladder (per-point logit) | ALW→Bright **0.47** ≫ Bright→Jorja **0.15** > Jorja→#10 Fahey **0.09** |
| Same ladder, display scale (exp. margin vs avg pair) | +8.5 / +7.3 / +6.8 / +6.4 |
| Bright's edge over #3 | bigger than the whole gap from #3 down to #10 |
| Partner weighting (best fit over ~37,000 pro games) | **59% weaker / 41% stronger** |
| ALW+JJ vs AB+KF, normal strategy | **86%** ALW/JJ (modal score 11-6) |
| Mid-Season final, frozen pre-match receipt | **88%** ALW/JJ → **lost 6-11** (graded MISS) |
| Full ice-out (team = "two Jorja Johnsons") | 2.35 vs 2.36 team value → **50.1%. A coin flip.** |
| Give ALW back just 10% of the action | 61% ALW/JJ |
| Give her back 20% | 70% |
| Stacked: ice-out + Jorja playing like #5 / #10 / #13 | Bright/Fahey **56% / 65% / 78%** |
| Lever 2 *alone* (normal shot mix) | Jorja must play like the **~#28 woman** just to reach even |
| 2026 head-to-head | ALW/JJ **3-1** (11-4, 11-3, **6-11**, 11-5) |
| **The final: NJ vs STL, traditional lineups** | **NJ 55%** per matchup — NJ 3+ games 26%, **2-2 DreamBreaker 47%**, STL 3+ games 27% |
| If the final is a best-of-3 series | NJ **58%** (independent matchups; correlation pulls it slightly back toward 55) |
| Game-by-game (NJ side) | WD **86%**, MD **16%**, MXD1 **66%**, MXD2 **31%** |
| DreamBreaker (mean roster singles, k=0.42) | **NJ 62%** (roster means 1.93 vs 1.71) |
| Final if STL re-runs the WD ice-out (ALW → 10%) | **STL 55%** — the tactic flips the matchup favorite |
| Final if NJ flips its mixed pairs to hunt the DB | NJ 55.7% vs 55.2% — **equity-neutral** (P(DB) 47%→50%) |
| The one NJ–STL DreamBreaker this season (5/25 Dallas) | **STL won 21-15** (NJ played Howells, not Staksrud); STL leads series 2-1 |
| The Staksrud acquisition | NJ bought him from Orlando in late July (last Squeeze games 7/19, first 5s games 7/30) — right after the DB loss + Mid-Season final loss to STL |
| Singles, if it goes to a DreamBreaker | ALW **#1 woman in >99% of posterior draws**; Staksrud **45% to be #1 man** (Haworth 48%, Duong 5%) |

## The narrative arc (~10 slides)

**1 — Hook.**
**"My model said 88%. They lost 11-6."**
On July 12 my model made Anna Leigh Waters & Jorja Johnson 88% favorites
over Anna Bright & Kate Fahey in the Mid-Season final. Dave Fleming
called me out on Threads for not pricing in the strategy. Was this an
upset? Or did Anna Bright and Kate Fahey do something to change the math?

**2 — The four players.**
**"On paper, this isn't close."**
Current model ranks: Waters #1, Bright #2, Jorja Johnson #3, Kate Fahey
#10. Waters is #1 *by a mile* — her lead over Bright is about twice the
entire gap from #2 down to #10. But give Anna Bright her due: she's a
*clear* #2. Her edge over Jorja is bigger than the whole gap from Jorja
down to #10. The ladder isn't evenly spaced, and that matters later.

**3 — How you price a doubles team.**
**"Your team is 59% your weaker player."**
The model rates every player from ~37,000 pro doubles games. Then the
question: how do two ratings combine into a team? Not the sum, not the
average — sweep every weighting from 50/50 to 100/0 and score each one on
how well it predicts actual point-by-point results. The fit peaks at
**59% weaker / 41% stronger**, decisively better than a pure sum.
Equivalent form: team = sum of ratings − 0.18 × the gap between partners.
The reason is strategic, not mystical: your opponents choose who to hit
at, so the gap between you and your partner is itself a target.

**4 — The strategy.**
**"Two Jorja Johnsons."**
At the Mid-Season final, Bright and Fahey took the weakest-link rule to
its logical extreme: they iced Waters out almost completely. Every serve,
every third shot, every speed-up — at Jorja. In the stylized limit, the
59/41 weighting becomes 100/0, and the team on the other side of the net
is no longer Waters + Johnson. It's two Jorja Johnsons.

**5 — The math.**
**"86% → 50.1%. A literal coin flip."**
Two Jorja Johnsons: team value 2.35. Bright + Fahey: 2.36. Why does
deleting one player flip 86% to even? Because of where the gaps live:
**#1 ≫≫ #2 ≫ #3 > #10**. The drop from Waters to Bright (0.47) is three
times the drop from Bright to Jorja (0.15), which is itself bigger than
Jorja down to Fahey (0.09). The ice-out doesn't swap one star for
another — it deletes the single biggest gap in women's pickleball, and
everything left over is bunched together. Jorja is a monster, and it's
*still* only a coin flip. That's how good the tactic is on paper — and
how good you have to be to run it.

**6 — The knife edge.**
**"Let her touch 10% of the balls and you've already lost the edge."**
Give Waters back just 10% of the action: 61/39. Give her 20%: 70/30. The
lever only works near-total — which is why the execution looked so
extreme. Open courts conceded, ugly angles taken, anything to keep the
ball away from the best player alive.

**7 — The other lever.**
**"The ice-out buys you a coin flip. Then you go get more."**
Lever 1 changes *who plays*. Lever 2 makes someone *play below their
rating* — lobs, awkward resets, body pressure. Once the ice-out holds,
every notch you shave off Jorja converts directly into equity: if she's
forced to play like the #5 woman, Bright/Fahey are 56%. Like the #10:
65%. Like the #13: 78%. (For contrast, lever 2 *alone* is hopeless — at
a normal shot mix you'd have to drag Jorja all the way to ~#28 in the
world just to reach even.) The plan isn't ice-out OR make-her-worse.
It's ice-out, *then* make-her-worse.

**8 — The counter.**
**"A longer point is a leaky point."**
The Orlando rematch: 11-5, Waters/Johnson — and Bright/Fahey were still
plenty icy. What changed? Jorja played closer to her best, points ran
longer, and every extra shot in a rally is another chance for the ball to
find Waters' paddle. The ice-out has to survive the whole point, every
point; Jorja at her best makes the points long. 2026 head-to-head:
Waters/Johnson 3-1. The 6-11 is the one where everything held at once.

**9 — So what does a prediction mean?**
**"This is why they play the matches on a court and not a spreadsheet."**
My 88% was the win rate under typical strategy — an average over
everything that usually happens across 37,000 games. A specific game plan
against a specific team on a specific day has its own true number, and
nobody — not me, not the players — can compute it exactly. Precise
numbers are tough. But "hard to calculate" is not "doesn't exist": the
ice-out demonstrably moves this matchup from ~86% toward a coin flip, and
execution decides the rest. The number is real. The court is where you
find out what it was.

**10 — The final: St. Louis vs New Jersey.**
**"Half of this match funnels into a DreamBreaker."**
The final is a virtual coin flip, with New Jersey a **slight** favorite
at 55% over St. Louis at 45%. Not much separates these teams — the
structure is almost perfectly mirrored: NJ heavily favored in Waters'
two games (86% in the WD rematch, 66% in mixed), STL heavily favored in
the other two (84% in men's, 69% in the Fahey/Tardio mixed). Chalk on
both sides lands at 2-2 — a 47% chance — and in regulation the model
actually leans *St. Louis* 27-26. New Jersey's entire edge lives in the
DreamBreaker, where their singles stars get it 62%. Which means each
team is holding a lever:
**St. Louis' lever is the big one: the ice-out.** Run it back
successfully in the women's game and the whole matchup flips — St.
Louis goes from 45% to 55%, a ten-point swing in title odds from one
tactical choice in one game. That's the tactic that won them the
Mid-Season title, priced for this final.
**New Jersey's lever is the sweaty one: swap the mixed lineups.** Feed
Jorja/Staksrud to Bright/Patriquin, unleash Waters/Khlif on
Fahey/Tardio, and push even more of the match into the DreamBreaker —
where you're capitalizing on the best women's singles player in the
world and on Federico Staksrud, who my model gives a **45% chance of
being the best men's singles player alive** (a photo finish with Chris
Haworth at 48% — we're Bayesian here, not frequentist). The model
scores the swap equity-neutral (55.7% vs 55.2%): an optimization for
true believers only.
And one receipt that makes the sweaty lever look a lot less crazy:
these teams already played a DreamBreaker this season — May 25 in
Dallas — and **St. Louis won it 21-15**. New Jersey's lineup that night
had Will Howells, not Staksrud. Then St. Louis beat them again for the
Mid-Season title, and within weeks New Jersey went out and acquired
Staksrud from the Orlando Squeeze (his last Squeeze games are July 19;
his first 5s games are July 30). A 45%-to-be-the-best singles player on
the planet, bought mid-season, right after two losses to this exact
opponent — one of them in a DreamBreaker. That's not a coincidence.
That's a roster move aimed at this exact scenario. St. Louis leads the
season series 2-1; Staksrud is New Jersey's answer.

## Honesty footnotes (keep these in the post, short)

- **100/0 is stylized.** The rally logs can't verify how total the
  ice-out actually was — serve targets are rotation-constrained, so
  "they hit everything at Jorja" is broadcast-eye, not measured.
- **86% today vs 88% then**: the receipt was priced on values frozen
  before the match; 86% is the same matchup at current values. Both are
  calibrated (no displayed probability ever reaches 0% or 100% —
  about 1% of ≥99% favorites lose).
- The "coin flip" and the stacked-lever numbers are the model's own
  weakest-link structure evaluated at extreme settings — what-ifs inside
  the model, not fitted results.

## Repro

```
values (data/v2_players.csv, per-point logit): ALW 1.799, Bright 1.327,
  Jorja 1.173, Fahey 1.081;  γ = −0.1829
gap ladder: ALW→AB 0.473, AB→JJ 0.154, JJ→KF 0.092
team = v1 + v2 + γ|v1−v2|  ⇔  2·(0.41·stronger + 0.59·weaker)
ALW/JJ 2.858 vs AB/KF 2.363 → race-to-11 DP → calibrate → 0.861
ice-out: 2·Jorja = 2.346 vs 2.363 → 0.501
share sweep: p(s) from team = 2·(s·ALW + (1−s)·JJ)
stacked lever 2: p from 2·v vs 2.363 at v = 1.139 (#5 Todd) → .443,
  1.081 (#10 Fahey-level) → .347, 0.992 (#13 Humberg) → .219
lever 2 alone: Δv = (2.858−2.363)/(1−γ) = 0.42 → Jorja at 0.75 ≈ #28
H2H from data/games.csv
singles: data/singles_players.csv — ALW 2.514±.051 (#1 W in >99% of
  50k posterior draws over the top-20 field, independent normals);
  Staksrud 2.067±.034 vs Haworth 2.068±.041 → P(#1 M) .45/.48
  (Duong .05); Khlif 1.69, Jorja 1.46
FINAL (NJ = ALW/JJ/Staksrud/Khlif vs STL = AB/KF/Patriquin/Tardio),
  make_forecast pipeline exactly (calibrated game probs → 4-game tree,
  p_win = p40 + p31 + p22·p_db; DB = race-to-21 at k·mean-singles-gap,
  k = 0.42, gap +0.217 → NJ .618):
  A traditional  WD .86 MD .16 MX1 .66 MX2 .31 → NJ 3+ .258 / 2-2 .475 /
    STL 3+ .267, overall NJ .552
  B WD ice-out (ALW share 10%): WD .61 → overall NJ .446
  C flip mixed (Jorja+Fede sac vs Bright/Patriquin .25; ALW+Khlif
    vs Fahey/Tardio .72): 2-2 rises to .504, overall NJ .557 (~neutral;
    game-prob sums nearly equal: .66+.31 ≈ .25+.72)
  B+C both: NJ .447
```
