# Reddit draft — r/pickleball

**Title:** I built a model of every pro pickleball game. Here's the actual math on Anna Leigh getting "iced out" at Mid-Season.

---

If you watched the MLP Mid-Season final, you saw Anna Leigh Waters and Jorja Johnson lose 6-11 to Anna Bright and Kate Fahey, and you probably heard the same take I did: Bright/Fahey **iced out** Waters — funneled everything to Johnson and kept the ball away from the GOAT.

I run a Bayesian model of every MLP + PPA pro doubles game since 2024 (77% winner accuracy out of sample; DUPR is 65% on the same games). It had Waters/Johnson as **88% favorites** in that game. So it was the model's headline miss of the weekend — which makes it the perfect thing to interrogate. Does the "ice-out" hold up mathematically, or is that just a story we tell after a favorite loses?

**First, the thing that makes the ice-out work: pro doubles teams are ~59% their *weaker* player.**

That's the single most robust finding in the model. Fit across ~35,000 games, a team's strength is roughly 59% its weaker player and 41% its stronger one — because opponents choose who to hit to, and they hit to the weak link. It means a superstar can't carry a team as far as a raw rating gap suggests. **A star's value is capped by how much of the court they can cover.**

**So here's the dial.** I can model "what share of the court does Waters cover?" and read off the win probability:

- **Waters covers ~41% (a normal game):** 88% — this is the real number. Opponents already tilt most of the load onto Johnson; that's *baked in* to the 88%.
- **Waters covers 50% (equal):** 93%
- **Waters covers 0% (fully iced out):** **48%**

Freeze Waters completely and the team plays like **two Jorja Johnsons.** And two Johnsons, by the model, are worth almost exactly what Bright/Fahey are worth. Sanity check — I literally simulated two Jorja Johnsons vs Bright/Fahey: **48%.** Same answer, as it has to be, because freezing the star is mathematically the same as swapping her for a second copy of her partner.

**The punchline: the ice-out *alone* — no bad luck, no "the model overrated them" — turns 88% into a coin flip.** Neutralizing the star doesn't just remove her shots; the weakest-link structure drags the whole team's level down toward the partner. That's why it's a real strategy and not just cope.

**Now the part where I keep myself honest, because this is one game:**

1. **I can't prove they actually did it.** The referee logs show Bright/Fahey served *about evenly* to both (Johnson 10 times, Waters 9) — serve target isn't the weapon, because rotation constrains who you serve. The ice-out lives in *mid-rally* ball placement, and no data source in this sport tracks where the 3rd/5th/7th ball goes. So the model can show the strategy is **sufficient**; it can't confirm it **happened**.
2. **An 88% favorite still loses about 1 in 8.** The freeze gets you to ~50/50 at most, and they still had to win the coin flip. This was a coin-flip-level shock, not a miracle.
3. **6-11 wasn't even a freak scoreline.** Conditional on losing, the most likely loss is 11-9, and about two-thirds of losses are 11-8 or closer. 6-11 was on the lopsided side (~80th percentile) but not bizarre.

**TL;DR:** Icing the GOAT is structurally sound — not because it wears her down, but because pro teams are mostly their weaker player, so freezing the star turns her team into two of her partner. That alone flips an 88% game to a coin flip. Whether Bright/Fahey truly pulled it off or just had a great night, one game can't tell you — but the ceiling of the tactic is exactly what we saw.

Happy to run other matchups if people want — drop a pairing and I'll post what the model says. (More on how it's built + a live win-prob board in the comments.)
