# Content handoff for Threads post design system
## Pro pickleball analytics project — everything postable

Context for the designer: this repo scrapes every pro pickleball doubles game
(MLP + PPA, 2024–2026, ~36,000 games), fits a Bayesian rating model, and
validates it by prediction (77% winner accuracy on unseen games vs 65% for
DUPR, the sport's official rating). The brand is **receipts and honesty**:
every prediction is committed with a timestamp before the match, every number
carries uncertainty, and we say "we can't know that" out loud. The design
system should make error bars and probabilities feel like a flex, not a hedge.

---

## A. Recurring post types (design templates for these)

**A1. Pre-match forecast card** — the flagship.
Per game: two pairs, win probability, most-likely scores. Plus an outcome
tree (3-0 / 3-1 / DreamBreaker / 3-1 / 3-0 with probabilities) and one
overall number. Live example (tonight's MLP Mid-Season Gold final):
WD Waters/Johnson 88% over Bright/Fahey (11-5) · MD Tardio/Patriquin 92%
(11-5) · MXD1 Bright/Patriquin vs Waters/Khlif 46/54 coin flip (11-9) ·
MXD2 Fahey/Tardio 75% (11-7) → DreamBreaker 46%, STL wins 61/39.
Renewable: every MLP matchup and PPA championship Sunday.

**A2. Receipts scorecard (post-match).** Forecast vs actual, side by side,
with a running "model record" tracker (currently 77.4% on 884 held-out
games). Design needs a win-state and a lose-state that are equally
shareable — taking the L publicly is part of the brand.

**A3. Current-form power rankings.** Top 10 per gender from the dynamic
model (values + error bars + games played), with movement vs last month.
Data: `data/v2_players.csv`, refreshed by one command. Design constraint:
men's and women's lists must be visually SEPARATE (cross-gender ranking is
statistically meaningless in this data — house rule, never break it).

**A4. Player trajectory card.** Monthly skill curve Jan 2024 → now with
uncertainty band. Launch set: Tardio (smooth rise to #1), Johns (flat line
while the field rose — "he never declined"), Waters (still steepening),
Parenteau (genuine decline), Patriquin (steady climb), Black (one wild 2025
peak). Data: `data/v2_trajectories.csv` (every regular, every month).

**A5. Hypothetical matchup simulator.** "Who wins: X/Y vs W/Z?" — any four
players, win prob + likely score, computable on demand. Includes the
weakest-link penalty, so "superstar + passenger vs two solids" posts write
themselves.

**A6. Upset log.** Weekly: games the model priced under ~15% that hit,
with the pre-game number. Pairs with A2 for the honesty brand.

**A7. Chemistry check.** A pair's actual results vs sum-of-parts
expectation, with the honest error bar. Almost always lands on "it's the
players, not the pairing" — that IS the content.

**A8. DUPR divergence.** Where the official rating and the model disagree
most. Evergreen ammunition: Tardio ranked ~#30 by DUPR while #1 here.

## B. One-off finding posts (evergreen, each is one strong visual)

1. **The Waters chasm.** Her lead over the #2 woman (1.65 pts/game) equals
   the ENTIRE spread of the men's top 25. One-axis dot plot, two brackets.
2. **Chemistry is (mostly) a myth.** Player quality matters ~5× more than
   pair fit; no pair in the sport has statistically proven chemistry; you'd
   need ~1,000 games together to prove a typical effect (max on record: 138).
3. **Pickleball is a weakest-link game.** Every point of skill gap between
   partners costs ~half a point of team strength. Tagline: "choose your
   equal." (Also: the superstar Waters+Bright pairing performs BELOW the
   sum of its parts — 3rd percentile.)
4. **In mixed, targeting is gender-blind.** The data rejects "attack the
   woman" in favor of "attack the weaker player, whoever that is."
5. **Ben Johns never declined — everyone else arrived.** Flat absolute
   curve, rising field. The most counterintuitive trajectory result.
6. **The men's #1 is a five-way statistical tie** (Tardio, Johns, Patriquin,
   JW Johnson, Alshon within error bars). Great overlapping-intervals visual.
7. **Model vs DUPR: 77% vs 65%** winner accuracy on identical unseen games —
   while DUPR got to keep updating and the model was frozen. Credentials post.
8. **DUPR's greatest hits (data-quality horror):** a pro's rating collapsing
   6.13 → 3.50 mid-season; Tardio's rating FALLING through his breakout;
   the entire tour dropping ~0.5 overnight in a recalibration.
9. **No, the pros don't sandbag MLP.** Skill gaps convert to points at the
   same rate in both tours; if anything stars tick UP in MLP.
10. **New-pairing honeymoon (asterisked).** New partnerships overperform
    their first ~6 games together. Fun, but carries a data caveat — post as
    a question, not a claim.
11. **The Kawamoto twins problem.** Why player IDs beat names: three
    Kawamotos, two of them twins who partner each other, and the official
    rating system apparently lost one mid-season. Data-nerd catnip.
12. **How it works, thread version.** The EXPLAINER.md content: one equation,
    36,000 games, skepticism built in, then the receipts.
13. **September accountability post (scheduled).** Preregistered chemistry
    predictions (committed July 12) get publicly scored against the second
    half of the season — win or lose.

## C. Data assets the design system can rely on

| asset | contents | refresh |
|:--|:--|:--|
| `data/v2_players.csv` | current-form value ± sd, games, per player | one command |
| `data/v2_trajectories.csv` | monthly curve for every regular since 2024 | same |
| `data/v2_dyads.csv` | pair chemistry (small, honest) | same |
| `data/yearly_values.csv` | season-by-season values + gender ranks | same |
| `data/games.csv` | every game, score, players, date (36k rows) | same |
| `data/platform_ratings.csv` | DUPR snapshots for 1,142 players | same |
| forecast machinery | any matchup → win probs, score distributions, outcome trees | on demand |

## D. Voice & non-negotiables

- **Uncertainty is the brand.** Every number that has an error bar shows
  it. "±" should be a visual signature, not fine print.
- **Receipts culture.** Predictions are timestamped before matches;
  post-match posts show the pre-match number, hit or miss.
- **Never**: cross-gender rankings as fact; individual-pair chemistry as
  proven; probabilities without their base ("of games it had never seen").
- Scale cheat-sheet for consistent copy: values are points-per-game vs an
  average pro (median regular ≈ +2, star ≈ +5, Waters ≈ +7.7); win probs
  come from margins via ~4.7 points of per-game luck.
- Attribution: "based on public results data" — this is unofficial fan
  analytics, not tour-affiliated content.

---

## E. Match-card field menu (evergreen, any match)

Tier 1 headline: overall win prob; most-likely outcome; full outcome tree
(3-0/3-1/DB/1-3/0-3 for MLP, 2-0/2-1 for PPA Bo3); modal game scores;
prediction-locked timestamp.

Tier 2 per game: win prob; score distribution; HINGE INDEX (how much the
overall flips if this game flips — the card's signature stat); P(deciding
game); upset-alert badge (<65% favorite); expected total points.

Tier 3 per pairing: record together + avg score; combined value; internal
balance gap + weakest-link penalty in points; chemistry ± sd + percentile;
new-pairing badge; exact-pairing head-to-head with scores.

Tier 4 per player: season W-L%; avg score for/against; value ± sd + gender
rank; 3-month form arrow; deciding-game record; overtime record (games past
11); blowout rate (wins allowing <=5); DUPR + divergence flag.

Tier 5 context (MLP): season series; rematch/history flags; prior meetings
of the same four pairings; DreamBreaker sensitivity line; best player on
court; star-collision callout.

Tier 6 trust strip: model record (77% on 884 unseen games vs DUPR 65%);
uncertainty shown wherever it exists; "public results data, unofficial."

All fields regenerate from games.csv + v2_players.csv + v2_draws.npz +
the matchup cache. Example values for every field exist in the 2026-07-12
Gold final worked example (session log / prediction_midseason_final.md).
