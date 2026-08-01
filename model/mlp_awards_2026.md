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
| **Most Improved** | 2025 → 2026 value delta, full-time MLP players | **Tina Pisnik** (+0.5 pts/gm) | **Jack Sock** (+0.9 pts/gm) |
| **MVP** | Matchup Win Probability Added | **Anna Leigh Waters** (+8.31) | **Hayden Patriquin** (+6.06, 16 clinchers) |
| **Under the Radar** | Games Won Above Expectation | **Sofia Sewing** (+4.6) | **Anouar Braham** (+4.0) |

---

## 1. Most Improved — season-over-season value delta, full-timers only

**Definition.** Change in v1 season value (expected points per game vs an
average pairing, fit per season) from the 2025 fit to the 2026 fit, among
**full-time MLP players: ≥ 25 MLP games in each season**. The games-played
distribution is cleanly bimodal in both years — subs and part-timers below
~20 games, the full-time mass from ~30 up — so 25 sits in the valley, and
the boards are identical at any floor from 25 to 30.

| Women | 2025 | 2026 | Δ | MLP games | Men | 2025 | 2026 | Δ | MLP games |
|---|---|---|---|---|---|---|---|---|---|
| **Tina Pisnik** | +4.56 | +5.02 | **+0.46** | 71→56 | **Jack Sock** | +3.38 | +4.28 | **+0.90** | 56→43 |
| Cailyn Campbell | +1.86 | +2.23 | +0.37 | 33→60 | Christian Alshon | +4.92 | +5.27 | +0.35 | 52→45 |
| Mya Bui | +1.82 | +1.91 | +0.09 | 65→39 | Max Freeman | +3.17 | +3.41 | +0.24 | 60→52 |
| Lacy Schneemann | +3.73 | +3.72 | −0.00 | 67→32 | Hayden Patriquin | +5.24 | +5.42 | +0.18 | 74→53 |
| Anna Bright | +6.17 | +6.09 | −0.08 | 74→52 | Gabriel Tardio | +5.67 | +5.55 | −0.13 | 76→47 |

(24 women and 23 men qualify. "2026 rank" below = that season's v1 fit,
which can differ from the site's v2-based rankings page: Pisnik is #7F,
Sock #12M.)

Sock is the story: a second-season leap of +0.90 points per game — twice
anyone else's gain among full-time men — carrying him to #12 in the 2026
men's fit. Pisnik's +0.46 wins a thinner women's race; the honest reading
of both boards is that **among players who ran the full MLP gauntlet twice,
almost nobody improved** — the field mostly held or slipped, which is what
a maturing league looks like. Values are pool-relative per season, so these
deltas measure movement against the field, not against a fixed yardstick.

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
- **Points above replacement** — (2026 v1 value − 20th-pctile value among
  MLP-2026 players, +0.87) × games / 2 (Waters led, ~185 pts): a fine
  skill-×-usage metric, but it ignores *when* you won and its ratings are
  fit on the full season rather than earned game by game. WPA answers the
  MVP question better.
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
- Player ranks quoted with Most Improved are the per-season v1 fits
  (`data/yearly_values.csv`), not the site's v2 rankings page.
