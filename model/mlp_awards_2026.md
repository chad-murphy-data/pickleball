# MLP 2026 season awards — metrics, winners, receipts

*Written 2026-08-01, covering the MLP 2026 season through July 30 (1,111
doubles games; DreamBreakers and forfeits excluded). Reproduce every table
with `python model/mlp_awards.py` (stdlib-only, reads committed `data/`
CSVs). Awards are given separately to a man and a woman — cross-gender
values share a scale only by prior convention (house rule), so we never
rank across.*

## The winners

| Award | Metric | Women | Men |
|---|---|---|---|
| **Most Improved** | 2025 → 2026 season value delta | **Elsie Hendershot** (+2.0 pts/gm) | **Tama Shimabukuro** (+1.6 pts/gm) |
| **MVP** | Matchup Win Probability Added | **Anna Leigh Waters** (+7.12) | **Ben Johns** (+5.62, 13 clinchers) |
| **Under the Radar** | Games Won Above Expectation | **Sofia Sewing** (+4.6) | **Anouar Braham** (+4.0) |

---

## 1. Most Improved — season-over-season value delta

**Definition.** Change in v1 season value (expected points per game vs an
average pairing) from the 2025 fit to the 2026 fit, among MLP-2026-active
players with ≥20 games in both seasons.

| Player | 2025 | 2026 | Δ pts/gm | Now ranked |
|---|---|---|---|---|
| Elsie Hendershot (F) | −0.67 | +1.36 | **+2.03** | #74 F |
| Tama Shimabukuro (M) | +2.08 | +3.65 | **+1.57** | #20 M |
| Paula Rives (F) | +0.78 | +2.02 | +1.24 | #56 F |
| Jack Sock (M) | +3.38 | +4.28 | +0.90 | #12 M |
| Donald Young (M) | +2.56 | +3.30 | +0.74 | #30 M |
| Armaan Bhatia (M) | +3.19 | +3.92 | +0.73 | #15 M |
| Keilly Ulery (F) | −0.34 | +0.38 | +0.72 | #96 F |
| Estee Widdershoven (F) | +2.66 | +3.37 | +0.71 | #22 F |

Hendershot is the cleanest Most Improved story in the dataset: the only
player who moved from below-average to solidly positive, and she corroborates
on every other cut we tried — top in-season v2 trajectory riser among MLP
players, #2 per-game in points-over-expectation. Shimabukuro's jump carried
him from fringe to top-20 among men.

**Why not "improved during the season"?** We looked. The v2 monthly random
walk moves slowly by design: the biggest Apr→Jul mover (Alexa Schull) gained
+0.06 logit while the posterior uncertainty on any single player's in-season
delta is ~0.12 — i.e. the whole leaderboard is inside the noise. The
in-season list is also dominated by stars peaking during title runs (Tardio,
Fahey, Patriquin, Johns), which is a different award. Year-over-year is where
genuine improvement separates from noise.

## 2. MVP — Matchup Win Probability Added (WPA)

**Setup.** MLP matchups are races to 3 game wins out of at most 4 doubles
games (WD, MD, MX1, MX2), with the dead 4th game skipped at 3-0 and a 2-2 tie
sent to a DreamBreaker. `data/games.csv` stores MLP games as independent
single-game matches with no matchup id, so matchups are reconstructed:
group games by (event, date, stage), split each group into matchups by
roster overlap (union-find on partnerships — the two 4-player rosters fall
out as connected components), and order games by `match_times.csv` start
times. This yields 241 clean matchups; 70 odd groupings are dropped
(single-game strays from bracket-day stage labels, plus matchups whose
mid-matchup substitutions defeat the two-roster split).

**Definition.** With future games treated as coin flips, the matchup win
probability at game score (a, b) satisfies P(a,b) = ½P(a+1,b) + ½P(a,b+1),
with P(3,·)=1, P(·,3)=0, P(2,2)=½ (DreamBreaker). Each game's WPA is the
change in P from its result: ±0.1875 for the openers, ±0.25 at 1-1 or in a
live game 4, ±0.125 for a clincher at 2-0. **Both players on court bank the
full swing** (a doubles score can't apportion credit within the pair). A
player's season WPA is the sum over their games. No model ratings enter
anywhere — WPA is pure outcome accounting from a 50/50 intercept.

**Accounting identity** (the property that makes it legible): because the
swings within a matchup telescope from 0.5 to 1 or 0, and each game credits
two players, a roster's summed WPA equals its net record in
matchups-decided-in-games. Verified on the NJ 5s: Waters +7.12, J. Johnson
+4.62, Khlif +3.19, Howells +0.75, Staksrud +0.31 → **+16.00 = 18 − 2**
(their 4 DreamBreaker matchups sit at exactly 50% after the doubles games
and net to zero; the DB's remaining ±0.5 is deliberately uncredited here —
it's rally singles and could be apportioned from `db_rallies.csv` if wanted).

**Leaderboard.**

| Player | WPA | W-L | Clinchers | Elim. saves | Avg leverage |
|---|---|---|---|---|---|
| Anna Leigh Waters (F) | **+7.12** | 44-4 | 11 | 0 | 0.177 |
| Ben Johns (M) | **+5.62** | 39-10 | 13 | 1 | 0.189 |
| Gabriel Tardio (M) | +5.62 | 41-5 | 10 | 1 | 0.158 |
| Hayden Patriquin (M) | +5.50 | 43-8 | 13 | 0 | 0.162 |
| Kate Fahey (F) | +5.25 | 39-6 | 10 | 1 | 0.153 |
| Anna Bright (F) | +5.12 | 40-8 | 13 | 0 | 0.164 |
| Jorja Johnson (F) | +4.62 | 38-7 | 7 | 0 | 0.167 |
| Christian Alshon (M) | +4.44 | 30-7 | 7 | 1 | 0.188 |
| Eric Oncins (M) | +3.94 | 40-21 | 7 | 4 | 0.188 |
| Danni-Elle Townsend (F) | +3.25 | 39-21 | 8 | 4 | 0.181 |

Waters is the runaway women's MVP: her games alone account for 44% of her
team's net matchup progress. The men's race is a dead heat at +5.62; we give
it to **Johns on the clincher tiebreak** (13 matchup-sealing wins to
Tardio's 10, most in the league alongside Patriquin, Bright, and Jade
Kawamoto). Tardio's counter-case is the better record (41-5) — a reasonable
voter could flip it. Trivia the metric surfaces: **Augustus Ge leads the
league with 6 elimination saves** (wins with his team facing matchup point).

**Is the leverage weighting fair?** The natural worry — that late-slot
players get systematically juiced or starved — turns out to be a non-issue.
Under the 2026 dead-game rule, game 4 only happens when live (always ±0.25),
so average leverage by slot is nearly flat: 0.188 / 0.188 / 0.180 / 0.171
(the tail-off is ordering noise on days without clean timestamps). What
actually varies is *team quality*: players on dominant teams face less
leverage because they're often up 2-0 (clincher worth only ±0.125) and their
game 4s get skipped. Kate Fahey has the lowest average leverage of any
regular (0.153) and still ranks 5th — she earned it through the blowout
handicap, not from it. If WPA is mildly unfair to anyone it's juggernauts,
and defensibly so: less rides on each game of a sweep.

## 3. Under the Radar — Games Won Above Expectation (GWAE)

**Definition.** For every game, the model's win probability is priced from
month-of-game v2 values (team strength = sum of values + γ·|gap| with
γ = −0.18, per-point p = σ(Δstrength), exact race-to-11 with the win-by-2
branch lumped). GWAE = Σ (won − p̂) over a player's season, min 25 games.
Where WPA lets favorites shine, GWAE prices skill *in* — the model already
expects stars to win, so this board finds the players who beat their own
rating. It is structurally a surprise metric, which is exactly what an
under-the-radar award should be.

| Player | GWAE | W-L | Expected wins |
|---|---|---|---|
| Sofia Sewing (F) | **+4.6** | 38-16 | 33.4 |
| Anouar Braham (M) | **+4.0** | 12-23 | 8.0 |
| Mya Bui (F) | +3.6 | 14-25 | 10.4 |
| Kiora Kunimoto (F) | +3.2 | 9-17 | 5.8 |
| Armaan Bhatia (M) | +3.1 | 23-20 | 19.9 |
| Paula Rives (F) | +3.1 | 8-19 | 4.9 |
| Tyson McGuffin (M) | +3.0 | 20-21 | 17.0 |
| Alexa Schull (F) | +3.0 | 14-18 | 11.0 |

Sewing is the strongest version of the story: she beat expectation by 4.6
wins *while playing a full schedule with a winning record* — not a
small-sample bounce. Braham is the purest one: a $10k-minimum draft pick
(California, traded mid-season to Miami) whose 12-23 record hides that 11 of
his 35 games were within two points, with wins over Staksrud/Y. Johnson,
Alshon/Haworth, and Martinez Vich/Devilliers. The model expected 8 wins from
his schedule; he took 12. (James Delgado posted +4.0 in only 21 games and
misses the 25-game floor.)

---

## Alternatives considered and set aside

- **Total points won** (Townsend led, 682): a volume/durability stat — it
  mostly measures whose team played the most games.
- **Points over expectation** (Braham led): same surprise structure as GWAE
  at point grain; redundant with it and noisier per unit.
- **Points above replacement** — (value − 20th-pctile MLP value) mapped to
  expected margin × games / 2 (Waters led, ~226 pts): a fine skill-×-usage
  metric, but it ignores *when* you won and its ratings are career-informed
  rather than MLP-2026-earned. WPA answers the MVP question better.
- **In-season trajectory delta**: inside the noise (see Most Improved).

## Caveats

- Men and women are never compared on one scale; every award is split.
- GWAE expectations use v2 values that are career-informed (all tours), not
  MLP-only; that's a feature for pricing but means "expectation" includes
  PPA evidence.
- WPA covers the 241 cleanly reconstructed matchups (~87% of season games);
  dropped groupings are roster-substitution edge cases, not selection on
  outcomes.
- DreamBreaker games are excluded throughout (house rule: rally-to-21
  singles, never enter doubles models). In WPA they net to zero by
  construction; a DB-inclusive variant could apportion the remaining ±0.5
  from `db_rallies.csv`.
