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
