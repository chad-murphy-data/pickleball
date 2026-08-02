# MLP 2026 season awards — metrics, winners, receipts

*Written 2026-08-01, covering the MLP 2026 season through July 30 (1,111
doubles games across 286 matchups; DreamBreakers and forfeits excluded).
Reproduce every number with `python model/mlp_awards.py` (stdlib-only, reads
committed `data/` CSVs; the matchup table `data/mlp_matchups_2026.csv` is
committed too and regenerates via `scraper/mlp_matchups.py`). Awards are
given separately to a man and a woman — cross-gender values share a scale
only by prior convention (house rule), so we never rank across.*

## The winners

| Award | Metric | Women | Men |
|---|---|---|---|
| **Most Improved** | MLP-only improvement, 2025 → 2026, two independent measures | **Kate Fahey** (+4.1pp rally share) | **Christian Alshon** (+5.3pp rally share) |
| **MVP** | Matchup Win Probability Added | **Anna Leigh Waters** (+8.31) | **Hayden Patriquin** (+6.06, 16 clinchers) |
| **Under the Radar** | Games Won Above Expectation | **Sofia Sewing** (+4.6) | **Anouar Braham** (+4.0) |

---

## 1. Most Improved — MLP-only improvement, full-timers, two measures

**Definition.** Improvement *in MLP play specifically*, 2025 → 2026, among
**full-time MLP players: ≥ 25 MLP games in each season** (the games-played
distribution is cleanly bimodal in both years — subs and part-timers below
~20 games, the full-time mass from ~30 up — so 25 sits in the valley).
Because any single number can be argued with, the award requires agreement
between **two independent measures**:

- **(A) Rally point share** — of every serve and return rally the player
  contested in MLP that season (from the referee logs; ≥ 500 rallies each
  year), the share their side won. Transparent, no model, huge samples;
  not opponent-adjusted.
- **(B) MLP-only margin fit** — a ridge regression on nothing but that
  season's MLP game margins, giving each player a points-per-game effect.
  Opponent- and partner-adjusted; noisier at ~50 games a season. Deltas are
  anchored so the returning full-time cohort nets to zero, which removes
  pool drift.

| Women | Rally share Δ | Fit Δ (pts/gm) | Men | Rally share Δ | Fit Δ (pts/gm) |
|---|---|---|---|---|---|
| **Kate Fahey** | **+4.1pp** (.555→.597) | +1.2 (#4) | **Christian Alshon** | **+5.3pp** (.530→.583) | **+1.7 (#1)** |
| Cailyn Campbell | +3.6pp | +0.8 (#5) | Riley Newman | +5.1pp | — |
| Liz Truluck | +3.1pp | +1.5 (#2) | Max Freeman | +3.6pp | — |
| Parris Todd | +2.8pp | — | Jay Devilliers | +3.1pp | — |
| Lacy Schneemann | +2.3pp | +1.4 (#3) | Jaume Martinez Vich | +3.1pp | — |

(47 players qualify. "—" = outside that measure's top five. Fit top fives:
women Erokhina +1.8, Truluck +1.5, Schneemann +1.4, Fahey +1.2, Campbell
+0.8; men Alshon +1.7, Frazier +1.1, Khlif +1.1, Sock +1.1, Oncins +0.9.)

**Alshon is the men's winner on both boards independently** — the biggest
rally-share gain in the league (+5.3 points per hundred rallies, ~2.8
standard errors) *and* the biggest opponent-adjusted margin gain. **Fahey
wins the women** with the largest rally-share gain of any full-timer and a
top-five fit delta — and she was already a star, which makes it better: she
got better at every rally she played on the way to St. Louis's 24-1 season.
Truluck is the honorable mention (top-3 on both boards, below-average →
average-plus); Erokhina tops the fit but the rally board doesn't
corroborate her, so she doesn't get the convergence call.

**Why MLP-only?** The obvious construction — delta in all-tour season value
— crowns Jack Sock (+0.90 pts/game) and Tina Pisnik (+0.46). Real gains,
but partly earned in PPA play: on MLP-only evidence Sock drops to #4 among
men and off the rally top five entirely, while Alshon is #1 on both MLP
boards. For an MLP award, MLP evidence decides.

**Where did the breakout names go?** An earlier draft of this award (all
pro games counted, no MLP floor) crowned players like Tama Shimabukuro
(0 MLP games in 2025 → 58 in 2026) and Elsie Hendershot (0 → 18). Their
improvement is real, but they didn't *return* to MLP better — they broke
into it, several with barely any 2026 MLP presence at all. That's a
Breakout award, not Most Improved, so the full-time filter screens them out.

**Why not "improved during the season"?** We looked. The v2 monthly random
walk moves slowly by design: the biggest Apr→Jul mover (Alexa Schull) gained
+0.06 logit while the posterior uncertainty on any single player's in-season
delta is larger than that (monthly value sd ≈ 0.11, so ≈ 0.1–0.15 on a
three-month delta) — the whole leaderboard is inside the noise. The
in-season list is also dominated by stars peaking during title runs, which
is a different award. Year-over-year is where genuine improvement separates
from noise.

## 2. MVP — Matchup Win Probability Added (WPA)

**Setup.** MLP matchups are races to 3 game wins across 4 doubles games
(WD, MD, MX1, MX2), with a 2-2 tie sent to a DreamBreaker. In 2026 round
robin the dead 4th game at 3-0 is generally still played; in bracket play
it's recorded as a walkover and skipped.
`data/games.csv` stores MLP games as independent single-game matches with
no matchup id, so the matchup structure — which games belong together, in
what order — comes from MLP's own records via the open BFF
(`scraper/mlp_matchups.py` → `data/mlp_matchups_2026.csv`): **286 completed
matchups covering all 1,111 games**. Walkover/dead games (33 slots) advance
the matchup score with nobody credited.

**Definition.** With future games treated as coin flips, the matchup win
probability at game score (a, b) satisfies P(a,b) = ½P(a+1,b) + ½P(a,b+1),
with P(3,·)=1, P(·,3)=0, P(2,2)=½ (DreamBreaker). Each game's WPA is the
change in P from its result: ±0.1875 for the openers, ±0.25 at 1-1 or in a
live game 4, ±0.125 for either side of a game played at 2-0, and 0 for a
dead game at 3-0. **Both players on court bank the full swing** (a doubles
score can't apportion credit within the pair). A player's season WPA is the
sum over their games. No model ratings enter anywhere — WPA is pure outcome
accounting from a 50/50 intercept.

**Accounting identity** (the property that makes it legible): swings within
a matchup telescope from 0.5 to 1 or 0, and each game credits two players,
so a roster's summed within-team WPA equals its net record in
matchups-decided-in-games (DreamBreaker matchups sit at exactly 50% after
the doubles games and net to zero; walkover swings go uncredited, which can
shift a team's sum slightly). Verified on the New Jersey 5s: Waters +8.31,
J. Johnson +5.31, Khlif +4.00, Howells +1.19, Staksrud +0.19 (his NJ share;
he also logged games elsewhere) → **+19.00 = 21 − 2**, with their 4
DreamBreaker matchups netting to zero.

**Leaderboard.**

| Player | WPA | W-L | Clinchers | Elim. saves | Avg leverage |
|---|---|---|---|---|---|
| Anna Leigh Waters (F) | **+8.31** | 50-4 | 14 | 0 | 0.179 |
| Anna Bright (F) | +6.06 | 44-8 | 16 | 0 | 0.167 |
| Hayden Patriquin (M) | **+6.06** | 45-8 | 16 | 0 | 0.164 |
| Ben Johns (M) | +5.75 | 43-12 | 16 | 2 | 0.191 |
| Kate Fahey (F) | +5.56 | 42-6 | 8 | 1 | 0.150 |
| Gabriel Tardio (M) | +5.56 | 42-5 | 8 | 1 | 0.153 |
| Jorja Johnson (F) | +5.31 | 42-7 | 7 | 0 | 0.167 |
| Christian Alshon (M) | +5.06 | 35-10 | 7 | 1 | 0.190 |
| Eric Oncins (M) | +4.12 | 41-21 | 7 | 4 | 0.188 |
| Danni-Elle Townsend (F) | +4.12 | 47-26 | 10 | 5 | 0.183 |
| Sofia Sewing (F) | +4.12 | 38-16 | 7 | 6 | 0.194 |
| Noe Khlif (M) | +4.00 | 38-15 | 14 | 0 | 0.179 |

Waters is the runaway women's MVP: her games alone account for 44% of her
team's net matchup progress, and no other woman is within two swings of
her. The men's award goes to **Patriquin outright** (+6.06): highest WPA,
best blend of record (45-8) and matchup-sealing wins — his 16 clinchers
share the league lead with Johns, Bright, and Jade Kawamoto. Johns (+5.75)
has the counter-case: two elimination saves and the highest average
leverage faced among the leaders (0.191). Tardio's +5.56 rides the best
record (42-5). Trivia the metric surfaces: **Augustus Ge and Etta
Tuionetoa share the league lead with 7 elimination saves** (wins with
their team facing matchup point).

**Is the leverage weighting fair?** The natural worry — that late-slot
players get systematically juiced or starved — turns out to be a non-issue.
Average leverage by slot is nearly flat: 0.188 / 0.188 / 0.176 / 0.167.
The mild game-3/4 tail-off is real but mechanical: at 2-0 a game is worth
only ±0.125, and round-robin dead game 4s carry zero. What actually varies
is *team quality*: players on dominant teams face less leverage because
they're often up 2-0 and their game 4s are dead. Kate Fahey has the
second-lowest average leverage of any 30-game regular (0.150, behind only
Michael Loyd's 0.138) and still ranks 5th — she earned it through the
blowout handicap, not from it. If WPA is mildly unfair to anyone it's
juggernauts, and defensibly so: less rides on each game of a sweep.

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
small-sample bounce (and she's tied for second in elimination saves, with
6). Braham is the purest one: bought by the California Black Bears with the
last starter pick of the draft at the $10k minimum bid, traded mid-season
to the Miami Pickleball Club, and his 12-23 record hides that 11 of his 35
games were within two points, with wins over Staksrud/Y. Johnson,
Alshon/Haworth, and Martinez Vich/J. Devilliers. The model expected 8 wins
from his schedule; he took 12. (James Delgado posted +4.0 in only 21 games
and misses the 25-game floor.)

---

## Alternatives considered and set aside

- **Total points won** (Townsend led, 682): a volume/durability stat — it
  mostly measures whose team played the most games.
- **Points over expectation** (Braham led, +64): same surprise structure as
  GWAE at point grain; redundant with it and noisier per unit.
- **Points above replacement** — (value − 20th-pctile value among MLP-2026
  players) × games / 2: a fine skill-×-usage metric, but it ignores *when*
  you won and its ratings are fit on the full season rather than earned
  game by game. Waters leads it under either skill input — all-tour v1
  values (~185) or an MLP-games-only fit (~159). The men's board is why
  the input matters: all-tour ratings crown JW Johnson (PPA-informed);
  MLP-only ratings say Johns by a hair over Patriquin — and neither
  version knows Patriquin converted his games into more matchup wins.
  WPA answers the MVP question better.
- **In-season trajectory delta**: inside the noise (see Most Improved).

## Caveats

- Men and women are never compared on one scale; every award is split.
- Matchup structure comes from MLP's own records (fetched once from the
  same open BFF that feeds the rest of the project and committed as
  `data/mlp_matchups_2026.csv`); all 1,111 games are covered. Walkover and
  dead games advance the matchup score with nobody credited.
- GWAE expectations use v2 values that are career-informed (all tours), not
  MLP-only; that's a feature for pricing but means "expectation" includes
  PPA evidence.
- The season includes MLP Orlando's exhibition-format matchups (opponents
  like Team Canada and the College All-Stars); they're MLP-sanctioned pro
  doubles and count everywhere here.
- DreamBreaker games are excluded throughout (house rule: rally-to-21
  singles, never enter doubles models). In WPA they net to zero by
  construction; a DB-inclusive variant could apportion the remaining ±0.5
  from `db_rallies.csv`.
- Most Improved's rally shares come from the committed referee-log
  aggregate (`data/player_serve_rallies.csv`); its margin fit uses MLP
  games only and is pool-drift-anchored to the returning cohort. Both are
  rates, so the partial 2026 season (through July 30) doesn't bias them.
