# When would Team 1 ever choose cross-gender matchups? (2026-07-23)

**The question** (from Anna Bright's proposal, where Team 1 sets the four
DreamBreaker singles matchups and Team 2 sets the slot order): under what
conditions would a rational Team 1 ever pair a woman against a man?

**The answer in one line:** essentially never on anything resembling a real
roster — and when it finally becomes rational, it's the least watchable
version imaginable: a sacrifice, not a marquee matchup.

Everything here is computed on **hypothetical players** with an exact
full-game solve (`model/db_crossgender_conditions.py`). No real-player
cross-gender claim is made anywhere — the model's cross-gender house rule
is not violated because gender enters only as a label on made-up strength
values.

---

## 0. The reframe that makes this tractable

The model never had a gender question. It has a **matching question**:

> Is gap-minimizing (your #1 on their #1, ..., your #4 on their #4)
> always Team 1's best play when Team 2 controls the schedule?

Gender is stapled on afterward. Under the assumption that every man on a
DreamBreaker four is a better singles player than every woman on it, both
rosters sort M, M, F, F — so rank-matching down the ladder *automatically*
produces four same-gender matchups, and "choose cross-gender" and "deviate
from gap-minimizing" become the same event.

**That assumption is the league's, not ours.** In 176 real logged
DreamBreaker team orders, slot 1 was a man 176/176 times; men fill 96% of
slots 1–2. Teams, with matches on the line and full knowledge of their own
players, unanimously order as if their men outrank their women at singles.
We read the hierarchy off the league's revealed behavior. And the
assumption is self-correcting under BOTH structures: if a woman is
genuinely her team's #2 singles player, her coach should slot her second
under the current rules (she then faces whatever the other team put
second, usually a man) and rank-matching does the same job under Anna's
proposed rules. The competitive woman-vs-man matchup schedules itself the
moment a coach believes in it — no rule change required. The referee logs
agree: 88 of 3,189 logged DreamBreaker rallies (2.8%, in 7 of 88 matches)
were already man-vs-woman under the current format (ALW vs Newman et al.).

## 1. Setup

- Rally model: P(win rally) = sigmoid(k·(v1−v2)), k = 0.510 (the
  coefficient fitted on 3,101 same-gender DB rallies; here applied to
  hypothetical values).
- Game value: the exact backward DP from `model/db_scenarios.py` — full
  race to 21, win by 2, the freeze, winner-serves-next, 4-point rotation
  cycling until the game ends, averaged over the serve coin toss. Every
  number below is a win probability from that DP. **No expected-points
  shortcuts**: sequence, truncation, and the race nonlinearity are fully
  modelled.
- Team 2 plays its true optimal order: min over all 24 slot permutations.
  (Robustness vs the realistic edge-sort heuristic checked in §4.)
- Team 1 compares its best same-gender matching against its best
  cross-gender matching, both evaluated at Team 2's best response.

## 2. Why the swap almost never helps (the mechanism)

Swapping your man and woman onto cross-gender matchups does exactly one
thing: it moves BOTH affected matchups' strength gaps by the same amount —
the gap between your own man and woman (the "house gap"). If they're
equal, the swap is a strict no-op (pure theater). If he's better, your
losing matchup loses harder and your winning matchup wins harder. The
swap never moves your average strength; it only **stretches your matchups
further apart**.

And stretch is precisely what the order-picker weaponizes: your deepened
hole gets scheduled early (slot 1 plays ~11 of a typical 37 points; slot
4 plays ~8), where it buries you while the game is alive; your fattened
blowout gets scheduled late, where the game often ends before its third
pass, and where surplus points past 21 are worthless. When your opponent
controls the schedule, extreme matchups are their weapon, not yours.

### Worked example (illustrative logits, real names for readability only)

St. Louis Shock (T1) vs Carolina Hogs (T2), values chosen to fit the
scenario — French is pretended to be a monster (2.0) over Tardio (1.0),
Bright (0.5) crushes Hatton (−0.5), middles dead even (Hayden = Young 0.8,
Fahey = Conard −0.8). g_M = 1, g_W = 1, house gap D = 0.5.

Same-gender, Hogs order adversarially — **Shock win 42.8%**:

| slot | matchup | Shock rally p | E[points] Shock–Hogs |
|---|---|---:|---:|
| 1 | Hayden v Young | .500 | 5.7–5.7 |
| 2 | Tardio v French | .375 | 3.7–6.3 |
| 3 | Fahey v Conard | .500 | 4.3–4.3 |
| 4 | Bright v Hatton | .625 | 5.0–3.0 |

Cross-gender top swap (Bright takes French, Tardio takes Hatton) —
**Shock win 39.6%**:

| slot | matchup | Shock rally p | E[points] Shock–Hogs |
|---|---|---:|---:|
| 1 | Hayden v Young | .500 | 5.7–5.7 |
| 2 | Bright v French | .318 | 3.1–6.9 |
| 3 | Fahey v Conard | .500 | 4.3–4.3 |
| 4 | Tardio v Hatton | .682 | 5.5–2.5 |

The rally probs are perfectly mirrored ({.375, .625} → {.318, .682}, both
average .500) and expected points barely move — yet the swap costs 3.3 pp
of win probability. That gap is pure sequencing: the deeper early hole
ends more games before the fattened slot-4 surplus can matter.

## 3. Stage 1 — the exact frontier (neutral middles)

Second men's and second women's matchups pinned at p = 0.5; three knobs
in logits: g_M (T2 best man over T1 best man), g_W (T1 best woman over T2
best woman), D (T1's house gap, best man over best woman). T1's candidate
strategies: same-gender rank-match vs the top-pair cross swap.

**Cross-gender is never optimal anywhere in the realistic box**
(g_M, g_W ∈ [0, 2] — for scale, the entire real MLP men's singles field
spans ~1.1 logits). The exact critical men's deficit g_M* (bisection,
tol 1e-3):

| g_W \ D | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 2.81 | 2.70 | 2.50 | 2.34 | 2.20 |
| 0.50 | 2.89 | 2.80 | 2.65 | 2.53 | 2.43 |
| 1.00 | 3.06 | 2.99 | 2.89 | 2.80 | 2.74 |
| 2.00 | 3.65 | 3.62 | 3.57 | 3.53 | 3.50 |

Reading it:

- **The bar is 2.2–3.6 logits of same-gender men's deficit** — your best
  man winning only ~14–25% of rallies against theirs — i.e. a mismatch
  2–3× the spread of the entire real men's field. In the Shock–Hogs cast
  above, French would need to be ~3.0 logits above Tardio before the swap
  breaks even.
- **A bigger women's edge RAISES the bar** (rows increase). If your women
  already crush theirs, the swap's gain end is saturated too — you pay
  the stretch and collect nothing. The "our women are dominant, feature
  them on the men" intuition is exactly backwards.
- **A bigger house gap LOWERS the bar** (columns decrease) — the opposite
  of the "your woman must be close to the man" intuition. The gulf is
  what the swap cashes on the smash side; the marquee side is in the flat
  tail either way.

### When it finally flips, what the play IS

At the frontier the marquee pair is not close (the woman wins ~14–25% of
rallies). The rational cross-gender play is a **sacrifice**: your man is
already so hopeless same-gender that substituting your woman costs almost
nothing (the sigmoid tail is flat — a lost matchup can't get much more
lost), while your man moving onto their weak woman cashes the full house
gap on the steep middle of the curve. Throw the lost matchup, cash the
gulf. Both ends must cooperate: the men's matchup must be annihilation
(cheap to donate) AND the women's matchup must still be live (real
upgrade available). Near-parity in one gender, catastrophe in the other,
simultaneously.

## 4. Stage 2 — assumption relaxed (random 8-player configs)

1,500 seeded random configurations, men ~ U[0,2], women uniform in a band
up to 2.5 logits below the weakest man (so gulfs far beyond anything
real are allowed), full solve: T1 max over all 24 matchings, T2 exact
adversarial order.

- **Cross-gender optimal in 285/1500 (19.0%)** of these deliberately
  extreme configurations; median gain where it happens is small
  (+1.1 pp; max +9.5 pp).
- All 285 survive when T2 uses the realistic edge-sort heuristic instead
  of the true adversarial order (285/285) — the conclusion is not an
  artifact of assuming a perfect opponent.
- Flagged-config medians: g_M = 0.22, g_W = −0.25, **D = 1.99** — the
  flag is driven by giant house gaps, not by the stage-1 top-swap logic;
  the full matching space finds cheaper asymmetric sacrifices (see
  pattern table below).

### What the winning cross-matchings actually are

Re-solving the flagged configs and classifying the optimal matching
(T1 role v T2 role, players ranked within team):

| count | winning matching | structure |
|---:|---|---|
| 248 | M1vM2 \| M2vF1 \| F1vF2 \| **F2vM1** | full one-rung rotation |
| 18 | M1vM1 \| M2vF1 \| F1vF2 \| **F2vM2** | partial rotation (keep the top men's matchup) |
| 12 | M1vM2 \| M2vF1 \| **F1vM1** \| F2vF2 | rotation variant (F1 absorbs) |
| 7 | M1vM1 \| M2vF1 \| **F1vM2** \| F2vF2 | partial variant |

**87% of the time it's the same play: feed your WORST player to their
BEST man, and shift everyone else down one rung of their ladder.** Your
best man takes their #2 man, your weak man takes their best woman, your
best woman takes their weak woman. This is the Tian Ji horse-race
strategy, verbatim (Sun Bin, 4th century BC: race your worst horse
against their best, your best against their middle, your middle against
their worst) — rediscovered by exhaustive search over a pickleball
tiebreaker.

Example flagged config: T1 = one stud and a cliff (1.93, 0.03, −0.88,
−1.79) vs a balanced T2 (1.64, 1.41, −0.11, −1.60). Rank-matching gives
T1 one narrow win and three losses → 15.3%. The rotation gives three
favored matchups plus one total sacrifice (F2, 3.4 logits under their
M1) → 17.5%. It's an underdog's variance play, and the gain is small
(median +1.1 pp across all flagged configs).

Note what the rotation's cross-gender matchups look like on court:
their best man annihilating your weakest woman, and your weak man
against their best woman. The marquee woman-vs-man showcase appears in
**zero** of the 285 optimal cross-matchings.

## 5. Bottom line

1. Team 1's optimal strategy is gap-minimizing rank-matching, which under
   the league's own revealed ordering produces four same-gender matchups
   automatically. (Consistent with the real-roster S2 result: deliberate
   unbalancing is worse in 379/380 matchups.)
2. Cross-gender pairings become rational only under cartoon conditions —
   a same-gender mismatch far wider than the real league's entire spread
   — and the rational version is a designated-loser sacrifice, not the
   competitive woman-vs-man showcase.
3. The showcase matchup can't be engineered by this rule change anyway:
   whenever a woman is genuinely good enough for it to be competitive,
   skill-ordering already schedules it under BOTH the current and the
   proposed structure. The logs show it already happens (88 rallies, 7 of
   88 DBs).
4. Unmodelled and acknowledged: crowd/camera effects ("ALW loves the
   smoke"), matchup-specific tactics, and any true M/W offset — the model
   is gender-blind by construction, which is exactly what makes the
   matching argument clean.

## 5b. The spectacle exception (why a team might still do it)

Everything above prices cross-gender as a *win-maximizing* choice, and
finds it almost never is. But a team optimizing for *entertainment* is
playing a different game, and might rationally pay a small win-probability
tax to stage a woman-vs-man showcase.

The tax is cheapest for a heavy favorite. The win-probability curve is
flat near the extremes: if you're already ~70% (let alone ~90%) to win
the DreamBreaker, giving up a couple points of edge barely moves your
win probability, because you were going to win anyway. In an even game
the same couple of points can flip the result outright. So fireworks are
"affordable" mostly in games that are already decided — which is a little
melancholy: the spectacle is free precisely when the competitive stakes
are lowest, and unaffordable in the title games where everyone is
watching.

There is one configuration where the tax is genuinely small *and* the
matchup is worth watching: you judge your best woman to be nearly as
strong as your second-best man, and the opponent's second man is weak.
Then moving her onto a man costs you little (she was already close to
that strength tier) and produces a real contest rather than a blowout.
That is a narrow, judgment-based, entertainment-driven call — **not** a
claim that any current roster satisfies it (this analysis asserts no
cross-gender player comparison, and rosters churn week to week). It's a
"why a coach might still do it" coda, not a prediction.

## 5c. Within-team, not league-wide — and why the sacrifice softens

The "every woman weaker than every man" premise is only what the data
reveals **within a team** (176/176 men-first orders). It is NOT a
league-wide fact. A superteam's woman can outrank a budget team's man:
Anna Bright is (plausibly) not a stronger singles player than her own
STL men, but she may well be stronger than the cheapest man on a
budget roster. The men-first hierarchy is real inside each locker room
and false across the standings.

This matters because §4's "designated-loser sacrifice" characterization
is partly an artifact of the stage-2 draw, which enforced the *stronger*
league-wide ordering (every woman below the weakest man on **either**
team). Relax it to the within-team-only version (women below their own
team's weakest man, cross-team overlap allowed) and the picture softens
where teams are unequal. A quick sweep of 20,000 random asymmetric pairs
under that relaxed draw (illustrative — depends on the draw, but the
direction is robust):

- T1's best woman outranks T2's **weakest** man ~31% of the time, and
  T2's **best** man ~22% of the time.
- A genuinely competitive woman-vs-man matchup (rally p within .45–.55)
  exists in ~42% of random pairings.
- Among pairs where T1 is the clearly stronger team, T1's star woman is
  competitive-or-favored against at least one opposing man **~99%** of
  the time.

So in a superteam-vs-budget DreamBreaker the cross-gender showcase is not
a sacrifice at all — it's a matchup the star woman is favored in, and
(per §5b) it's cheap precisely because the superteam wins anyway. That is
the strongest real form of Anna's entertainment case: not "engineer a
woman into a losing spot for drama," but "when you outclass the
opponent, your best woman genuinely outclasses some of their men, so
featuring her is both watchable and nearly free." The win-maximizing
conclusion (§5.1–5.2) is unchanged — rank-matching still wins — but the
entertainment tradeoff is far more palatable in lopsided matchups than
the same-budget sacrifice math implied.

## 6. Reproduce

```bash
python model/db_crossgender_conditions.py --step 0.1 --draws 1500
python model/db_crossteam_overlap.py        # §5c overlap stats (instant)
```

Stage 1 ~40 s; stage 2 ~20 min (exact 24-matching × 24-order solve per
draw). Sanity checks (all-neutral → 50%, degenerate cases, monotonicity,
team-swap symmetry) assert at startup. The overlap script is pure value
comparisons (no DP) and runs instantly.
