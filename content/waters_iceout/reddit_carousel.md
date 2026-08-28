# Ice-out — Reddit post / carousel (the simple version)

*Drafted 2026-08-28 against current v2 values (`data/v2_players.csv`),
γ = −0.1829, race-to-11 DP + site calibration. Numbers verified; repro at
the bottom. This supersedes nothing — the dossier/explainer in this folder
stay as the long version. This is the one that ships.*

## The numbers (all verified today)

| Fact | Number |
|---|---|
| Current women's ranks (v2) | ALW **#1**, Bright **#2**, Jorja **#3**, Fahey **#10** |
| ALW's lead over #2 Bright | ~2× the entire gap from Bright down to #10 Fahey |
| Partner weighting (optimal over ~37,000 pro games) | **59% weaker / 41% stronger** |
| ALW+JJ vs AB+KF, normal strategy | **86%** ALW/JJ (modal score 11-6) |
| Mid-Season final, frozen pre-match receipt | **88%** ALW/JJ → **lost 6-11** (graded MISS) |
| Full ice-out (team = "two Jorja Johnsons") | 2.35 vs 2.36 team value → **50.1%. A coin flip.** |
| Give ALW back just 10% of the action | 61% ALW/JJ |
| Give her back 20% | 70% |
| Lever 2 equivalent (normal shot mix) | Jorja must play like the **~#28 woman** to make it even |
| 2026 head-to-head | ALW/JJ **3-1** (11-4, 11-3, **6-11**, 11-5) |
| This weekend (MLP Finals NYC) | STL 96% / NJ 82% to advance → rematch is the likely final |

## The narrative arc (~10 slides)

**1 — Hook.**
On July 12 my model made Anna Leigh Waters & Jorja Johnson 88% favorites
over Anna Bright & Kate Fahey. They lost 11-6. Dave Fleming called me out
on Threads for not pricing in the strategy. He had a point — and it's a
more interesting point than either of us made at the time.

**2 — The four players.**
Current model ranks: Waters #1 (by a mile — her lead over #2 is about
twice the entire gap from #2 down to #10), Bright a clear #2, Jorja
Johnson #3, and Kate Fahey has climbed into the top 10. This is the best
women's doubles rivalry the sport has, and it's still not close on paper.

**3 — How the model prices a doubles team.**
Not the sum, not the average. Sweeping ~37,000 pro games, the best fit
weights the *weaker* partner 59% and the stronger 41%. Why? Because your
opponents get to choose who to hit at. That's the weakest-link rule, and
it's the whole story of what happened next.

**4 — The strategy.**
At the Mid-Season final, Bright and Fahey took the weakest-link rule to
its logical extreme: they iced Waters out almost completely. Every serve,
every third shot, every speed-up — at Jorja. In the stylized limit, the
59/41 weighting becomes 100/0, and the team on the other side of the net
is no longer Waters + Johnson. It's two Jorja Johnsons.

**5 — The math.**
Two Jorja Johnsons: team value 2.35. Bright + Fahey: 2.36. The 86%
favorite becomes a **coin flip** — actually a hair *under* 50%. Sit with
that: Jorja is the #3 women's doubles player in the world, and duplicating
her still only draws even, because Bright is a clear #2 and Fahey is
top-10. That's how good the ice-out is on paper — and how good you have
to be to run it.

**6 — The knife edge.**
The lever only works if it's near-total. Give Waters back 10% of the
action and it's 61/39. Give her 20% and it's 70/30. That's why the
execution looked so extreme — open courts conceded, awkward angles taken —
anything to keep the ball away from the best player alive.

**7 — The other lever.**
Icing someone out changes *who plays*. The second lever is making someone
*play worse than their rating* — which is what Bright tried in the Orlando
rematch, lobbing Jorja relentlessly. The math on that lever is brutal: at
a normal shot mix, Jorja has to be dragged from #3 down to roughly the
#28 woman in the world just to make the match even. It didn't happen:
11-5, Waters/Johnson.

**8 — The catch: levers cost.**
The levers interact, and both charge rent. Hitting everything at one
player makes you predictable and forces low-percentage balls (lever 2
recoiling on yourself). Overloading Jorja's side warps your own court
coverage. The 2026 ledger says the price is usually too high: head-to-head
it's Waters/Johnson 3-1, and the ice-out game is the 1.

**9 — So what does a prediction mean?**
Fleming's real question. My 88% was the win rate under *typical* strategy,
averaged over everything in 37,000 games — and 88% favorites lose one time
in eight with nothing weird happening at all. A great game plan isn't
outside the model; it's one of the ways the 12% happens. Strategy can
genuinely move the number — the ice-out provably moves this matchup from
86% to 50% *if executed perfectly and if the counter never lands*. On
average, though? On average Waters and Johnson are virtually unbeatable.

**10 — This weekend.**
MLP Finals, New York. Model has St. Louis (Bright/Fahey) 96% and New
Jersey (Waters/Johnson) 82% to win their Friday matchups — the rematch is
the likely final. Normal-strategy price: 86% Waters/Johnson, modal score
11-6. Bright and Fahey's job is to make it a coin flip again. Either way,
the prediction is frozen before first serve and graded after — receipts
at [site link].

## Honesty footnotes (keep these in the post, short)

- **100/0 is stylized.** The rally logs can't verify how total the
  ice-out actually was — serve targets are rotation-constrained, so
  "they hit everything at Jorja" is broadcast-eye, not measured.
- **86% today vs 88% then**: the receipt was priced on values frozen
  before the match; 86% is the same matchup at current values. Both are
  calibrated (no displayed probability ever reaches 0% or 100% —
  about 1% of ≥99% favorites lose).
- The "coin flip" is the model's own weakest-link structure evaluated at
  an extreme weighting — a what-if inside the model, not a fitted result.

## Repro

```
values (data/v2_players.csv, per-point logit): ALW 1.799, Bright 1.327,
  Jorja 1.173, Fahey 1.081;  γ = −0.1829
team = v1 + v2 + γ|v1−v2|  ⇔  2·(0.41·stronger + 0.59·weaker)
ALW/JJ 2.858 vs AB/KF 2.363 → race-to-11 DP → calibrate → 0.861
ice-out: 2·Jorja = 2.346 vs 2.363 → 0.501
share sweep: p(s) from team = 2·(s·ALW + (1−s)·JJ)
lever 2: Δv = (2.858−2.363)/(1−γ) = 0.42 → Jorja at 0.75 ≈ #28 woman
H2H from data/games.csv; weekend prices from data/forecasts.json
  (generated 2026-08-27, best-lineup tier)
```
