# Does partnership survivorship hide chemistry? (2026-08-16)

Reproduce: `python model/chem_survival.py` (~5 min, stdlib only) →
printed battery + `model/chem_survival_summary.json`.

## The question

Finding 2 says pair chemistry is small (sd_d = 0.053 logit ≈ 0.013
point-share ≈ 0.25 pts/game) and no single pair is certifiable. The
user's hypothesis (2026-08-16): maybe that's *censoring* — **bad
partnerships don't last**, so the negative tail never accumulates games,
the fitted spread is squeezed, and "we can't find chemistry" really
means "the market deleted the evidence." The proposed instrument was a
sweep of regular partnerships against a generic "other pairing" bucket.
analysis.md already carried the seed of this: the unshrunk fixed-effects
check found survivor dyads' t-stats average +0.44 with the note "hints
at survivorship (pairs that keep playing together are pairs it's
working for)" — never adjudicated.

## Verdict, in one block

**The selection mechanism is real and strong — but it is not hiding
chemistry.** Pairs ARE dropped on results (continuation odds rise
steeply with performance-so-far, +13pp top-vs-bottom tercile). Yet every
forward-clean read of what selection *retains* comes back empty:
survivors are **not** enriched for positive pair effects (the forward
tenure curve is flat-to-negative where the survivorship story demands
it rise), and the calibrated year-scale persistence moment puts
persistent chemistry **at or below finding 2's already-small value**,
excluding 1.5× finding-2 and above. What the dissolution market is
actually selecting on is **luck and transient pair form**, not a
persistent pair-specific trait — consistent with a world where
persistent chemistry barely exists to be selected on. Finding 2 stands,
now with the censoring objection tested rather than assumed away. Two
genuine side-discoveries: the naive "survivors look good" statistic
(analysis.md's +0.44) is manufactured by stopping-rule mechanics plus a
value-tracking artifact, and **finding 5's newness bump (beta_new > 0)
is largely a conditioning artifact** — unconditionally, first-time
pairings *underperform*.

Scale note used throughout: cleaned point-share units; ×4 ≈ per-point
logit; ×16.9 ≈ points/game. Finding-2's sd = 0.0132 share, i.e.
τ² = 175×10⁻⁶ share².

## What the archive contains (arm 1 — the censoring raw material)

9,290 distinct pairings ever tried (37,769 clean games, 2024–2026):

| final tenure | dyads | % of dyads | % of game-sides |
|---:|---:|---:|---:|
| 1–2 games | 2,085 | 22.4% | 5.3% |
| 3–5 | 3,339 | 35.9% | 17.5% |
| 6–14 | 2,871 | 30.9% | 32.5% |
| 15–29 | 685 | 7.4% | 18.0% |
| 30–59 | 212 | 2.3% | 11.0% |
| 60+ | 98 | 1.1% | 15.6% |

58% of pairings die within 5 games (4,234 of them conclusively
dissolved, not just recent). So the "tried and dropped" mass the
hypothesis needs is *huge* — the negative tail had every chance to
leave a mark in these games. The one thing no analysis can see is pairs
filtered out on the practice court before any official game (mechanism
M2); what this battery measures is everything the archive was allowed
to record.

## Selection is real (arm 3 — the dissolution hazard)

Discrete-time hazard at event granularity, outcome = "this pair plays
another event together" (blocks within 180 days of the archive end are
censored), predictor = the pair's cumulative mean cleaned residual:

- All pairs: β = **+2.06 ± 0.30** per share unit (PPA); MLP interaction
  +2.53 ± 1.28.
- Identifiable pairs (both players ≥60 career games and ≥20 out-of-dyad
  games — the population every quantitative claim below uses):
  β = **+2.91 ± 0.42**.
- Raw, within (tour × block-index) strata: PPA continuation .509 →
  .584 → .635 across residual terciles; MLP .647 → .733 → .807.
- MLP's much higher baseline continuation is the roster constraint
  doing exactly what it should (pairs can't dissolve mid-season).

So the premise of the hypothesis is confirmed and quantified: winning
pairs are kept, losing pairs are cut, PPA more freely than MLP.

## But survivors are not enriched (arm 2 — the forward tenure curve)

The discriminating instrument: mean cleaned residual of a pair's *next*
game as a function of how many games they've already played together.
Continuation decisions can only condition on past games, so under
no-chemistry the curve is flat at zero **no matter how aggressive
selection is** — and if selection were retaining real chemistry,
survivors would sit above zero at high tenure. Buckets 0 and 1–5 carry
finding 5's newness window; the pre-specified enrichment contrast is
≥12 games vs the 6–11 reference. Cluster-robust (multi-membership by
dyad) SEs; ×10⁻³ share:

Primary panel (all four players dynamic, 20,305 games):

| prior games together | 0 | 1–5 | 6–11 | 12–19 | 20–39 | 40+ |
|---|---|---|---|---|---|---|
| observed | −8.2±3.7 | −4.6±2.4 | ref | −6.8±3.0 | +0.3±2.8 | −2.1±3.6 |

**Enrichment (≥12 vs 6–11): −2.70 ± 2.54 ×10⁻³ (z = −1.06).** Not
only is there no rise — the point estimate is negative. The all-games
panel says the same (−3.64 ± 2.22, z = −1.64), plus a much deeper
bucket-0 hole (−22.0 ± 2.6) that is mostly thin-player rating
shrinkage, which is why the dynamic panel is primary.

Two structural notes on this curve, both load-bearing:

1. **The 12–19 dip with recovery by 20–39/40+ is the signature of
   value tracking, not of chemistry.** v2's monthly walk chases form;
   selection keeps pairs that ran lucky; both partners' values
   over-credit that luck; the pair's next games underperform the
   inflated prediction — worst just past the selection gates, healing
   as the walk re-converges. This bias is *negative*, so it cannot be
   hiding a positive enrichment signal at the tracking-converged 40+
   bucket, which reads −2.1 ± 3.6: flat.
2. The within-dyad shuffle decomposition (composition vs order) shows
   first games sitting **+14×10⁻³ above their own pair's mean** — which
   looks like finding 5's beta_new, and is exactly what a stopping rule
   manufactures with zero real effect (a pair's game 2 exists *because*
   game 1 went well). Unconditionally — the forward-clean read — first
   games of never-before-paired dynamic players run **−8.2 ± 3.7 below
   settled pairs**. So the honest answer to the user's literal "regular
   partnership vs generic other pairing" sweep: **scratch pairings of
   established players cost ≈ 0.8pp of point share (≈ 0.14 pts/game)
   relative to settled pairs — familiarity's absence hurts a little —
   while v2's beta_new (+0.088 logit, conditioned on the pair
   eventually reaching ≥15 games) is largely selection shape, resolving
   finding 5's "window-edge caveat" in the deflationary direction.**

## How big can persistent chemistry be? (arm 4 — calibrated moments)

Estimator: each identifiable pair's product of half-means,
E[m_h1·m_h2] = persistent τ² with **no noise-variance model at all**
(game/match/event shocks never straddle halves). Getting this estimator
honest consumed most of the session — three confident wrong versions
are kept in the output as receipts (see "Estimator lessons" below).
The primary is the **era-crossfit product**: 2024-25 × 2026 halves,
each era cleaned with the *other* era's nuisance fit (leave-own-dyad-out
offsets), so (a) cleaning errors can't straddle the product, (b) the
second factor is entirely post-selection — the stopping rule cannot
bias it — and (c) at year lag the tracking transient has healed.
406 era-spanning identifiable pairs; ×10⁻⁶ share²; finding-2 = 175:

- **ALL: dyad-weighted +18 [−621, +654]; games-weighted −153
  [−578, +255].**
- Receipts bracketing it: random event-halves −273 [−460, −84]
  (tracking bias, an impossible negative "variance", left visible on
  purpose); single-fit halves +273 [+120, +424] (shared offset-error
  bias, equally visible).
- Classes: nothing coherent. 60+ pairs +186 [−159, +581]; 30–59
  −1,229 [−3,008, +201]; divisions all straddle zero. The MLP
  roster-forced arm dies at era scale (n=3 — rosters redraft yearly)
  and is too noisy within season (n≈80, CI ±1,300) to distinguish
  forced from market populations: honest answer, **underpowered**, not
  "no difference".
- One oddity worth a thread note: the 12 pairs with 3–5 career games
  *spanning both eras* — occasional re-uniting pairs — show +6,823
  [+2,137, +11,894]. Tiny n, but these "reunion pairs" are exactly the
  ones a chemistry-aware market would re-book. Not a claim; a lead.

## Calibration of that bound (arms 5–6)

Lifecycle simulation: one seat per real identifiable pair (its tour and
its real remaining-calendar cap), block structures resampled from real
event blocks, measured noise components (game 29.6×10⁻³, match
1.5×10⁻³, event ≈0), continuation by the *fitted* hazard, scored by the
same sequential-half product. Chemistry grid in multiples of finding-2:

| true τ | sim games-wt τ̂² | sim enrichment ×10⁻³ |
|---|---|---|
| 0 | −15 [−133, +81] | 0 (exact) |
| 0.5× | +28 | +0.17 |
| 1.0× | +155 [+71, +286] | +0.65 |
| 1.5× | +374 | +1.76 |
| 2.0× | +688 [+525, +830] | +3.18 |
| 3.0× | +1,607 | +6.49 |

Injection through the full real pipeline measures the estimator's
transfer: era-moment recovery **λ = 0.89 games-wt / 0.93 dyad-wt**
(no-recleaning probe 1.01 — the crossfit-LODO estimator is essentially
calibrated; the residual ~10% is chained see-saw absorption, corrected
for), curve recovery λ ≈ 0.61. Selection-at-fitted-strength on its own
(τ=0 row) produces essentially nothing — **the stopping-rule bias that
fakes the backward-looking +0.44 does not fake the forward reads**,
which is the entire reason the battery is built on them.

Reconciliation (full numbers in `chem_survival_summary.json`): the era
moment **excludes τ ≥ 1.5× finding-2** (predicted games-wt +335×10⁻⁶
at 1.5×, +607 at 2×, vs observed CI upper +255) and puts the implied
point estimate at **0–0.63× finding-2** depending on weighting; 1.0×
finding-2 itself is comfortably allowed (predicted +141). The curve
independently excludes 3× (z = 2.62) with 2× borderline (z = 1.83). A
true chemistry spread of 2× finding-2 (≈ 0.45 pts/game) under the
measured selection would have produced a games-weighted era product
≈ +600×10⁻⁶ and a visible tenure rise; observed are −153 [−578, +255]
and −2.7 ± 2.5×10⁻³.

## Cross-season corroboration (arm 7)

v1's per-season dyad estimates (independently fitted seasons, so value
errors don't straddle the join), pairs with ≥10 games in consecutive
seasons: r = +0.008 (2024→25, n=102), +0.130 (2025→26, n=88), against
shrinkage-attenuation ceilings of ≈ 0.155/0.160 (the season fits are
noise-dominated, so even fully-real chemistry couldn't push r much past
0.16). Consistent with finding-2-sized-or-smaller; uninformative
beyond that — kept as corroboration, not evidence.

## Estimator lessons (the expensive part; do not "simplify" these away)

Each of these produced a *confident, wrong* number before being caught.
All three live in the printed output as permanent receipts.

1. **Offsets must be stripped** (rating error nests inside every dyad
   of a player) — but naive per-player offsets **chase** the player's
   dyad effects (the see-saw as an estimator bug): injected chemistry
   came back λ ≈ 2–4. Fix: leave-own-dyad-out (LODO) offsets.
2. **A single fit's offset noise is shared by both halves of a
   product** (+Var(error) bias; inflated the thin-pair class ~20×).
   Fix: cross-fitting — clean each half with the other half's fit.
3. **v2's monthly walk tracks form**, so selection-luck gets absorbed
   into values and the pair's later games underperform: short-lag
   products go *negative* (−273×10⁻⁶ "variance") and the tenure curve
   dips transiently. Injections cannot see this bias (they bypass the
   tracker) — it must be dodged by design: era-lag products and the
   tracking-converged 40+ bucket. Per-(player, quarter) offsets would
   absorb tracking directly but tripled the cleaned variance after
   LODO exclusion (cells too thin) — tried and rejected.
4. Thin players' ridge-dominated offsets leak rating error into small
   classes → the identifiable-pairs filter (both dynamic, ≥20
   out-of-dyad games) is the house identifiability rule as a
   population definition.

## What this does and does not settle

**Settled**: outcome-based dissolution (M1) is strong, measured, and
not hiding chemistry — survivors are not an enriched-chemistry
population; persistent pair chemistry including the doomed pairs is
bounded at < 1.5× finding-2 and pointed at ≤ 0.6×; the naive survivor
statistic is stopping-rule + tracking artifact; unconditional newness
is mildly negative and beta_new's positivity is conditioning shape.

**Not settled**: practice-court filtering (M2) of pairs that never got
an official game is invisible to any archive analysis — if teams
psychically screen out catastrophic pairings before game 1, no game
data can measure what was screened; the MLP forced-pair arm, the right
quasi-experiment for this, is underpowered at current sample sizes
(revisit after 2–3 more MLP seasons ≈ n×3); "chemistry" that is
transient pair *form* rather than a persistent trait is deliberately
outside this estimand (the short-lag receipts suggest something lives
there, entangled with tracking); and the n=12 reunion-pairs cell.

**Production implications**: none required — v2's pooled sd_d already
prices pair identity about right, and nothing here beats it out of its
holdout gate. Two candidate cleanups for a future gated refit: the
beta_new flag's future-conditioning (replace "first 6 games of
eventually-≥15 dyads" with an unconditional first-games flag, expected
to flip the sign), and noting that sd_d partly absorbs within-event
pair form rather than career chemistry.
