# Weakest-link structure in pro doubles (gamma model, 2026)

Motivated by the question "can we split actor/partner by making the higher-
ranked player the actor?" — the identifiable one-parameter version:

    team strength = v_i + v_j + gamma * |v_i - v_j|  (+ chemistry)

**gamma = -0.470 ± 0.101** (z = -4.6, 0 divergences). Pro pickleball doubles
is a WEAKEST-LINK game: every point of value gap between partners costs the
team ~0.47 points of effective strength. A +7.7 star with a +3.0 partner
plays like a balanced +4.2/+4.2 team, not like their +10.7 sum.

Consequences:
- Player values re-sort modestly: players who habitually carry weaker
  partners were being under-credited (Jorja Johnson +0.47, Kate Fahey +0.18);
  players in balanced elite pairings give some back (Tardio -0.57).
- Matchup prediction shifts most for unbalanced lineups: in the Mid-Season
  Gold final, MXD1 (Waters/Khlif gap 3.7 vs Bright/Patriquin gap 0.8) flips
  from 52.9% NJ to 47.1% NJ; overall NJ 43.0% -> 39.8%.
- Roster-construction implication: balanced pairs beat star-and-passenger
  pairs at equal total value.

Caveat: gamma conflates true weakest-link dynamics (targeting the weaker
player) with scoreboard truncation (blowout margins cap at ~11, which also
penalizes unbalanced pairings' observed margins). Separating those requires
the race-to-11 likelihood upgrade. The safe claim is about observed margins,
not mechanism.

## Mixed-doubles horse race: gender-blind vs gender-role (2026)

Which defines the "weak link" in mixed — the lower-valued player, or the
woman by role? Both terms fit jointly (delta is immune to the gender-offset
flat direction; gamma_mix is not — gamma_same is the clean estimate):

    gamma_same (gendered games) = -0.470 ± 0.114   z = -4.1
    gamma_mix  (mixed, |gap|)   = -0.375 ± 0.130   z = -2.9
    delta      (v_man - v_woman)= +0.132 ± 0.073   z = +1.8

Gender-blind weakest link wins: the |gap| term carries the effect in mixed
nearly as strongly as in gendered play. The gender-role term is weak and,
if anything, points OPPOSITE to the "target the woman" doctrine (slightly
more weight on the man's value — the "man covers more court" direction —
but at 1.8 sigma this is a lean, not a finding). Teams are punished for
imbalance per se, whichever gender the weaker player is.
