# DreamBreaker match order — how much does the slot matter? (2026-07-22)

Prompted by Anna Bright's essay *"Women don't matter enough at MLP"*
(brighterpickleball.beehiiv.com). Her structural claim: the MLP
DreamBreaker rotates **four singles matchups, four points at a time**, in
the announced order, racing to 21 (win by 2). Because the game ends partway
through a rotation cycle, players slotted **early play more rallies**. Teams
therefore stack their strongest players first — league-wide, the men — so
women "see the court less" and the format undervalues them.

Question underneath the argument: holding a roster fixed, **how much can the
order you slot your four players into swing the result?**

## Why same-gender (and why that's the honest choice)

Cross-gender singles values are **not identified** in this project — the M/W
offset is a prior convention, never a fact (CLAUDE.md house rules). A mixed
roster's per-matchup "gaps" would be dominated by that arbitrary offset, so
any order effect we measured would be an artifact of the convention. Within
one gender every gap is a real, identified skill gap. And the order
**mechanism is gender-blind** (early slot = more volume), so the magnitude
we measure transfers directly to Anna's cross-gender case.

## Engine

`model/db_order_sim.py` — exact DP over `(a, b, pos, j)`: scores, current
rotation slot `pos ∈ 0..3`, rallies played in that slot `j ∈ 0..3`. Rotate
every 4 points, race to 21 win-by-2. Per-rally win prob for the matchup on
court = `sigmoid(K_DB · (v_a − v_b))`, `K_DB = 0.42` on the fit_singles
scale — the **same** rally coefficient the forecast uses (db_model.md,
make_forecast.py). Rallies are iid within a matchup (the repo's DB model is
serve-blind; k was fit at rally level). No simulation noise.

Validation: with one p in all four slots the DP reproduces the repo's
existing `winprob.rally_race_p` to 1e-6 — this is a strict generalization of
the DB model already in the codebase. The **only** thing the rotation adds
is that the four matchups take turns.

Note the current forecast prices the whole DreamBreaker with a **single**
per-rally p from the *mean* roster gap (`make_forecast.py:76`), so it
assigns **zero** value to order. Everything below is invisible to the site
today.

## 1. The mechanism: early slots play more rallies

Between evenly matched teams (every matchup 50/50):

| slot | E[rallies] | share |
|------|-----------:|------:|
| 1    |      11.21 | 30.1% |
| 2    |       9.71 | 26.1% |
| 3    |       8.37 | 22.5% |
| 4    |       7.94 | 21.3% |

**Slot 1 plays 1.41× as many rallies as slot 4** (11.2 vs 7.9). That's the
entire lever — earlier is busier, regardless of who's in the slot. Slots 1–2
are nearly a full extra cycle busier than 3–4 because the average game (≈37
rallies) ends early in the *third* cycle, which slots 1–2 reach and 3–4
usually don't.

## 2. Same edge, different slot

Team A has one player who is +Δ better than his opposite; the other three
matchups are dead even. Where the edge sits changes the win probability by
placement alone:

| edge Δ | slot 1 | slot 2 | slot 3 | slot 4 | best−worst |
|-------:|-------:|-------:|-------:|-------:|-----------:|
| +0.25  | 54.0%  | 53.9%  | 53.1%  | 52.8%  | 1.2 pp |
| +0.50  | 57.9%  | 57.8%  | 56.2%  | 55.5%  | 2.4 pp |
| +1.00  | 65.4%  | 65.3%  | 62.2%  | 60.8%  | 4.6 pp |
| +1.50  | 72.1%  | 72.0%  | 67.6%  | 65.7%  | 6.4 pp |

Identical skill converts to more win probability in the busy slots. Your
biggest edge belongs in slot 1.

## 3. Real roster: the full 24-ordering spread

Team A: Waters 2.27, Fahey 1.80, Bright 1.49, Rane 0.89.
Team B: Todd 1.69, Jansen 1.51, Christian 1.47, Parenteau 1.44
(B fixed strongest-first).

- **Best A ordering** (Waters > Fahey > Bright > Rane): **58.1%**
- **Worst A ordering** (Bright > Rane > Fahey > Waters): **50.4%**
- **Match order alone swings Team A by 7.8 pp.** Strongest-first beats
  weakest-first by +7.7 pp.

For scale: that 7.8 pp is *larger than the entire roster-strength edge* the
forecast currently prices for most DreamBreakers (a few pp). Order is a
first-order effect the model ignores.

## 4. The rule is "relative-strongest first," and the meta is degenerate

The optimal ordering is **not** by your own absolute strength — it's by your
**edge** (your win probability in each matchup): put the matchup you're most
likely to win in the busiest slot, descending. This is a rearrangement
inequality (the busiest slot has the highest marginal value of +dp), and it
is optimal in 2000/2000 random matchup sets. The two rules coincide only
when your player ranking matches the gap ranking — e.g. your strongest
player is a 45% underdog while a weaker teammate has a 60% edge, and
edge-first (58.0%) beats absolute-first (55.6%).

For *this* roster the two happen to coincide (A's strongest players also
hold its biggest gaps), so both teams sorting strongest→weakest is a
**Nash equilibrium** — neither side can profitably reorder (A's best
deviation +0.00 pp, B's +0.03 pp). A's control over its own win prob is the
same 7.8 pp spread whether B is fixed or best-responding.

There is no interesting strategy here today: everyone stacks strong-first,
high-vs-high. This is exactly what Anna's proposed fix targets — letting one
team set matchups while the other sets position would break the degenerate
equilibrium and create a real ("unsolved") meta, with an incentive to
feature more players competitively rather than bury them.

## 5. Anna's case, same-gender proxy

Two strong (Fahey 1.80) + two weak (Rane 0.89) vs four average (1.35), so
A's per-slot edge is +0.46 for the strong pair and −0.46 for the weak pair:

| A's order | win prob |
|-----------|---------:|
| strong first (slots 1,2) | 53.8% |
| interleaved (S,W,S,W)    | 50.7% |
| weak first (slots 1,2)   | 46.2% |

Playing the strong pair first is worth **+7.5 pp** over playing them last.
That gap *is* the incentive Anna describes: the busy early slots go to your
strongest players; league-wide those are the men, so the women inherit the
low-volume tail — structurally "mattering less," with no cross-gender value
claim required to see it.

## 6. Real teams: St. Louis Shock vs New Jersey 5s

The graded Gold-final rosters (prediction_midseason_final.md; NJ DB four
from db_model.md), a real 2W+2M-per-side DreamBreaker:

- **St. Louis Shock**: Fahey 1.80, Bright 1.49 (W); Patriquin 1.24, Tardio 1.22 (M)
- **New Jersey 5s**: Waters 2.27, Johnson 1.20 (W); Khlif 1.48, Howells 1.32 (M)

St. Louis is the DB underdog (Waters plus a stronger men's pair). Every
number below is P(St. Louis wins); per-rally p = sigmoid(0.42·gap).

**The pairings (who faces whom), holding a neutral W,M,W,M order.** The
women pairing is the real lever; the men pairing barely moves it:

| women pairing | men pairing | St. Louis |
|---|---|---:|
| Fahey–Waters, Bright–Johnson | Patriquin–Khlif, Tardio–Howells | 41.3% |
| Fahey–Waters, Bright–Johnson | Patriquin–Howells, Tardio–Khlif | 42.1% |
| **Fahey–Johnson, Bright–Waters** | Patriquin–Khlif, Tardio–Howells | 44.9% |
| **Fahey–Johnson, Bright–Waters** | Patriquin–Howells, Tardio–Khlif | **45.7%** |

Pairing choice alone is worth **4.4 pp**. The two women pairings have the
*same* summed rally prob (98.1% either way), yet "Fahey draws Johnson" is
~3.6 pp better for St. Louis: as the underdog it wants its one winnable
women's matchup concentrated and busy, not averaged away.

**Adding slot order.** Across all 96 valid same-gender configurations St.
Louis spans **39.2%–46.4% (7.3 pp)**. The best case buries Bright–Waters
(its worst matchup) in slot 4 and features Fahey–Johnson in slot 1; the
worst case does the reverse.

**But the extremes need the opponent's cooperation** — NJ would never put
Waters in slot 4. When both order well, the outcome turns on **who
announces their order first**, because the gender-interleave pattern must
match, so the announcer sets it and constrains the responder:

- St. Louis announces first (sets the interleave): **43.0%**
- New Jersey announces first: **40.7%**
- The "button" (announcing first) is worth **2.3 pp** here.

NJ's optimal opener is simply edge-first on its players — **Waters, Khlif,
Howells, Johnson** — which forces St. Louis to split its two women into
slots 1 and 4. So the *realistic* band is **~41–43%**, not the full 39–46%
envelope — and today's order-blind forecast (42.8%) sits at the top of it,
because it implicitly assumes placement never costs the underdog anything.

This is the concrete, same-gender version of Anna's mechanism-and-fix: the
lever is real (a few pp), it's biggest on the woman-vs-woman pairing, and
**who announces first is itself worth ~2 pp** — exactly why she wants the
announcement order changed.

### A tempting wrong move: "both women first"

NJ's intuitive play is "my women are relatively stronger, play them both
first." But NJ's women are a **barbell**, not a strong pair: Waters is the
board's top weapon (+0.47 even against Fahey, St. Louis's best), while
**Johnson is NJ's weak link** — an underdog to both St. Louis women. As
*pairs*, NJ's edge actually lives with the men (net +0.34) over the women
(net +0.18).

| NJ opener | NJ win prob |
|---|---:|
| both women first (W,W,M,M) | 57.0% |
| **split — Waters 1st, Johnson last (W,M,M,W)** | **59.3%** |

Clustering both women costs NJ **2.3 pp**, because it drags Johnson (a ~47%
matchup) into busy slot 2. The right unit is the **player's edge, not the
gender pair**: edge-sort the four (Waters, Khlif, Howells, Johnson) — lead
Waters, bury Johnson in the dead slot. "Both women first" only wins when
*both* women are your relatively stronger picks; when one is a barbell end,
split them. (You also can't *force* the +0.78 Waters-vs-Bright: lead Waters
and St. Louis counters with Fahey, so the realized Waters edge is +0.47.)

## 7. Anna's fix in the concrete: one team pairs, the other orders

Split the two decisions — St. Louis fixes the four matchups, New Jersey then
slots them into the rotation. It's a Stackelberg game: NJ, moving second,
puts St. Louis's *worst* matchup in the busy slot 1, so St. Louis picks the
pairing with the best worst-case.

| St. Louis's pairing | NJ orders → | St. Louis |
|---|---|---:|
| **Fahey–Waters + Bright–Johnson** (balanced) | worst matchup 45% into slot 1 | **40.7%** |
| Fahey–Johnson + Bright–Waters (lopsided) | worst matchup **42%** into slot 1 | 39.2% |

Two results worth pulling out:

1. **Setting position is the stronger half of the split.** Handing NJ the
   order pins St. Louis at **40.7%** — the floor of the realistic band (vs
   42.8% order-blind, 43.0% if St. Louis set the order itself). Whoever sets
   *position* holds the ~2 pp of leverage; the pairing-only role is the weak
   hand.

2. **The pairing logic flips.** With order power St. Louis wanted the
   *lopsided* women pairing (Fahey–Johnson 56%, to feature early). Against an
   adversarial orderer it wants the *balanced* one — pair Fahey against
   Waters (45%, best resistance) and Bright against Johnson (53%, a matchup
   it wins), so no matchup drops to the 42% that NJ would exploit. Pairing
   becomes a **maximin (defend your weakest)** problem, not a **maximization
   (feature your strongest)** one. That defensive choice is worth ~1.5 pp;
   the men's pairing is a wash.

That role-flip — same team, opposite pairing instinct depending on who holds
the order — is precisely the "unsolved meta" Anna's proposal is after.

## 8. League-wide: does optimal play feature women in the top 2 slots?

Anna's hypothesis, tested across the field. Rosters for 11 MLP teams
reconstructed from projected lineups (forecasts.json — WD pair = a team's
two women, MD pair = its two men; two players imputed from doubles). This is
**cross-gender safe**: optimal ordering is edge-first, and every edge is a
*within-gender* gap on the same identified scale, so "is my women's edge
bigger than my men's edge?" is a legitimate comparison even though "is a
woman stronger than a man?" is not.

Over all 110 team-vs-team orderings, counting women in the two top slots
under edge-first optimal play (rank-matched pairing):

| women in top-2 | share |
|---|---:|
| 0 (both men) | 32% |
| 1 | 36% |
| 2 (both women) | 32% |

**mean = 1.00 women in the top 2.** Now the comparison Anna cares about:

- **Status quo she describes** (teams stack their strongest = men first):
  **0 women** in the top 2, always.
- **Optimal edge-first play: 1.00 on average — and both top slots are women
  in 32% of matchups.**

So optimizing instead of reflexively stacking men features women *far* more.
The punchline is that the optimal rule is **gender-blind**: it just fronts
your biggest edges, and those are women's matchups about half the time. It
lands at ~1.0 (not higher) because it tracks spread, and in *this* singles
cohort men's spread (sd 0.48) slightly exceeds women's (sd 0.33) — the
reverse of the doubles finding (women's spread 1.5×), so no strong tilt
either way. Favorites feature 1.09 women in the top 2, underdogs 0.91 —
stronger teams lean marginally more on their women.

**The concrete case (the "LA vs Dallas" you asked for).** Dallas Flash
stacked strong men (JW Johnson 1.37, Ge 1.00) with a weak women's pair
(Truong 0.70, Townsend 0.57) — exactly the men-first construction Anna
critiques. Their opponents' optimal answer is *women first*:

| slot | California Black Bears vs Dallas | win prob |
|---|---|---:|
| **1** | **[W] Dennehy v Truong** | 57% |
| **2** | **[W] Weil v Townsend** | 53% |
| 3 | [M] Tellez v J. Johnson | 48% |
| 4 | [M] Wild v Ge | 44% |

Optimal play buries California's *men* (underdogs to Dallas's strong men) in
the low-volume slots and features its women up top. Dallas's men-first
construction is self-defeating: it hands every opponent "women first" as the
best response. So yes — Anna's hypothesis holds under optimal play, and it
holds *hardest* precisely against the teams that stack men.

## 9. The full MLP split-role sim (team 1 pairs, team 2 orders)

Anna's proposed rule, run over all 20 current MLP teams (authoritative
rosters pulled 2026-07-22; shrunk imputation from db_impute.md applied).
Every ordered pair: team 1 fixes the four matchups, team 2 then slots them
adversarially. Team 1 picks the pairing with the best worst-case (maximin).

**Position is the stronger half — league-wide.** Holding the *order* is
worth **+5.0 pp** on average over holding the *matchups*. Whoever sets
position has the leverage; the matchup-picker plays defense.

**Women in the top-2 slots, decided by the adversarial order-picker:**

| women in top-2 | share |
|---|---:|
| 0 | 31% |
| 1 | 39% |
| 2 | 31% |
| **mean** | **1.00** |

Neutral on average — but that's the whole point versus the status quo. If
teams currently stack men first (Anna's premise), women are ~0 of the top-2;
under the split-role rule they average **1.0**. The rule roughly doubles
women's presence in the busy slots, and **whoever holds position imposes
their gender preference**:

| team 1 picks matchups vs | team 1 win% | order-picker's slots | women top-2 |
|---|--:|---|:--:|
| LA Mad Drops vs Dallas | 73.3% | Dallas orders **MMWW** | 0 |
| Dallas vs LA Mad Drops | 22.7% | LA orders **WWMM** | 2 |
| New Jersey vs St. Louis | 55.2% | St. Louis orders WMMW | 1 |
| St. Louis vs New Jersey | 40.7% | New Jersey orders WMMW | 1 |

Same two teams, opposite outcome depending on who holds the order: LA (whose
women are its biggest edge) orders **women-first** to feature them; Dallas
(weak women) orders **men-first** to bury LA's women's blowouts in the dead
slots. The St. Louis/New Jersey line (40.7% when St. Louis picks and New
Jersey orders) reproduces the section-7 two-team result exactly — a
consistency check on the full sim.

Bottom line for Anna's essay: her fix is *not* a guaranteed win for women,
but it moves them from buried (~0) to neutral (~1.0) in the busy slots, and
it makes featuring-your-women the *optimal* play for every team whose women
are its edge. Reproduce: `python model/db_order_sim.py --only 11`.

## Honest limits

- Rallies are iid within a matchup (repo DB model is serve-blind). Real DB
  rallies have side-out/serve streakiness; adding it would only sharpen the
  volume asymmetry, not remove it.
- K_DB = 0.42 has a wide CI (0.20–0.65, n=101 DBs). Bigger K widens every
  number here; smaller shrinks them. The *ordering* of the conclusions is
  K-independent.
- All values are singles form; DB performance is correlated-but-not-equal to
  pro singles (k ≈ half doubles strength). Numbers are illustrative
  magnitudes, not graded predictions.
- Same-gender by design (see top). The mechanism, not the specific pp, is
  the transferable result.

## Reproduce

```bash
python model/db_order_sim.py          # full battery (experiments 1-11)
python model/db_order_sim.py --only 6 # St. Louis Shock vs New Jersey 5s
python model/db_order_sim.py --only 8 # NJ opener: split the barbell
python model/db_order_sim.py --only 9 # Anna's fix: StL pairs, NJ orders
python model/db_order_sim.py --only 10 # league-wide: women in top-2 slots
```
