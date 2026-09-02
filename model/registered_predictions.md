# Preregistered predictions — frozen 2026-07-12

Written mid-season (data through 2026-07-11) so the remaining 2026 games are a
clean out-of-sample test. Score these in September against games played after
2026-07-12 only. These were the most extreme within-player dyad contrasts in
the data; if they were forking-paths artifacts, they should wash out.

1. **Waters + Bright (womens, PPA)** will underperform the sum-of-parts
   benchmark in their remaining 2026 womens games (predicted chemistry < 0;
   point estimate ≈ −1.7 unshrunk, −0.15 shrunk).
2. **Alshon + Patriquin (mens, PPA)** will underperform sum-of-parts in
   remaining 2026 mens games (same magnitudes).
3. **Bright + Patriquin (mixed)** chemistry will remain small: |chem| < 1
   point/game unshrunk on remaining games (i.e., their edge stays explained
   by individual values).
4. **League-wide**: the dyad-chemistry scale (sd_d) refit on H2-only games
   will stay under 1.0 point/game.

Scoring method: fixed-effects regression (model/fixed_effects_dyads.py) on
games dated > 2026-07-12 only, using full-season player FEs; compare each
registered dyad's H2-only estimate sign/magnitude to the above.

---

## Scored 2026-09-02

Method as registered: full 2026-season player fixed effects + tour intercepts
(11,223 games), dyad coefficient estimated by FWL on the **2,326 games dated
after 2026-07-12 only**, cluster-robust (CR1) SEs by match.

| # | prediction | H2 result | grade |
|---|---|---|---|
| 1 | Waters + Bright underperform sum-of-parts | **0 games together** | VOID |
| 2 | Alshon + Patriquin underperform sum-of-parts | **0 games together** | VOID |
| 3 | Bright + Patriquin \|chem\| < 1 pt/game | +0.05 ± 2.17 (14 g) | **HIT** |
| 4 | league-wide sd_d < 1.0 pt/game on H2 | 0.340 ± 0.248 (~95% upper 0.835) | **HIT** |

**Predictions 1 and 2 are VOID, not missed.** Neither pair played a single game
together after the freeze date. This is not missing data: all four players were
active in H2 (Waters 25 games, Bright 26, Alshon 33, Patriquin 26) — they simply
never appeared as those pairs again. Their frozen full-season estimates
(Waters+Bright −2.17 ± 0.85 on 98 games; Alshon+Patriquin −1.61 ± 0.86 on 95)
are unchanged from what was registered, but the out-of-sample test they were
written for cannot be run.

Note the direction: the two most extreme *negative* chemistry contrasts in the
data both dissolved. That is exactly what finding 2's survivorship result
describes (pairs are dropped on results, continuation β ≈ +2.9/share) and it is
n = 2 — consistent with the established selection effect, not new evidence for
it, and certainly not evidence the chemistry estimates were right.

**Lesson for the next preregistration:** a prediction about a *pair* silently
depends on the pair continuing to exist. Roster churn voided half this slate.
Future registrations should carry a minimum-games condition and a named
fallback (e.g. a league-level quantity, like #4, which was immune).
