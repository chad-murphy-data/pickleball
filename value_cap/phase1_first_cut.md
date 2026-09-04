# Phase 1 — first cut: expected-ties-won value (V) per player

Status: 2026-09-02, rough first pass. Run with `python
value_cap/phase1_value_model.py`; full methodology and every stated
assumption is in that script's docstring, not repeated here. This
document is the results and the read, not the recipe.

Nothing here is a finished valuation. It exists to check whether the
mechanism (dyad-aware, role-aware, reusing the production PICKLES race
engine) produces sensible output before anyone spends time tuning it --
a smell test, per the project's working rule: nothing was picked to
make any strategy look good or bad.

## What it does, in one paragraph

For every tracked player, build a 6-player reference roster: the player
plus five replacement-level fillers (an actual player near the
60th-best-by-doubles-value mark per gender, standing in for "any team
could roster this instead"). Draw the regular-discipline lineup
(WD/MD/MXD1/MXD2) as that roster's top 2W+2M by **doubles** value, and
the DreamBreaker lineup as its top 2W+2M by **singles** value --
independently, per the confirmed 2026 rule that a team's DB four isn't
tied to who started the other games. Simulate the tie against a mirror
opponent made entirely of replacement-level players (which plays itself
to an exact 50/50 by construction) using the same race-to-11 DP,
weakest-link gamma, and DreamBreaker singles-gap model
(`web/sitelib/race.py`, `model/db_model.md`) that prices real matchups
on the live site. V = P(win) - 0.5. V_regular and V_db split that
number by re-running with the player excluded from the DB-lineup pool.

## Top 10 by V_total

**Men:**

| player | V | (regular / db) | doubles | singles |
|---|---|---|---|---|
| Ben Johns | +0.269 | +0.219 / +0.050 | +1.114 | +1.910 |
| JW Johnson | +0.245 | +0.208 / +0.037 | +1.085 | +1.599 |
| Hayden Patriquin | +0.240 | +0.207 / +0.034 | +1.080 | +1.529 |
| Christian Alshon | +0.234 | +0.180 / +0.053 | +1.014 | +1.939 |
| Gabriel Tardio | +0.233 | +0.203 / +0.030 | +1.071 | +1.451 |
| Federico Staksrud | +0.194 | +0.132 / +0.061 | +0.905 | +2.066 |
| Andrei Daescu | +0.190 | +0.164 / +0.027 | +0.975 | +1.347 |
| Eric Oncins | +0.185 | +0.152 / +0.034 | +0.948 | +1.489 |
| Jay Devilliers | +0.178 | +0.139 / +0.039 | +0.921 | +1.582 |
| Jack Sock | +0.149 | +0.100 / +0.049 | +0.837 | +1.784 |

**Women:**

| player | V | (regular / db) | doubles | singles |
|---|---|---|---|---|
| Anna Leigh Waters | +0.422 | +0.360 / +0.063 | +1.799 | +2.510 |
| Anna Bright | +0.342 | +0.295 / +0.047 | +1.327 | +1.829 |
| Parris Todd | +0.296 | +0.241 / +0.056 | +1.139 | +1.927 |
| Jorja Johnson | +0.288 | +0.252 / +0.035 | +1.173 | +1.453 |
| Kate Fahey | +0.282 | +0.221 / +0.061 | +1.081 | +2.033 |
| Jade Kawamoto | +0.271 | +0.246 / +0.025 | +1.156 | +1.220 |
| Rachel Rohrabacher | +0.263 | +0.234 / +0.030 | +1.117 | +1.309 |
| Tina Pisnik | +0.258 | +0.237 / +0.021 | +1.125 | +1.120 |
| Jackie Kawamoto | +0.249 | +0.222 / +0.027 | +1.088 | +1.230 |
| Tyra Hurricane Black | +0.245 | +0.223 / +0.022 | +1.088 | +1.118 |

Waters clearing the field here by nearly 25% over Bright is the same
tier structure Phase 0 already found in raw doubles value, carrying
through cleanly once win probability and the DB channel are added.
Parris Todd (#3) passing Jorja Johnson (#4) despite lower doubles value
is new information this model adds on top of Phase 0's ranking: her
much higher singles value (+1.927 vs +1.453) buys enough DB-channel
value to flip the order. Worth sitting with, not a bug.

## Top 10 by V_db (DB-channel value, isolated)

| player | V_db | V_total | doubles | singles |
|---|---|---|---|---|
| **Christopher Haworth** | **+0.064** | +0.064 | +0.619 | +2.068 |
| Anna Leigh Waters | +0.063 | +0.422 | +1.799 | +2.510 |
| Federico Staksrud | +0.061 | +0.194 | +0.905 | +2.066 |
| Kate Fahey | +0.061 | +0.282 | +1.081 | +2.033 |
| Hunter Johnson | +0.059 | +0.102 | +0.720 | +1.975 |
| Quang Duong | +0.058 | +0.135 | +0.796 | +1.953 |
| Brooke Buckner | +0.056 | +0.115 | +0.726 | +1.751 |
| Parris Todd | +0.056 | +0.296 | +1.139 | +1.927 |
| Lea Jansen | +0.055 | +0.101 | +0.701 | +1.733 |
| Christian Alshon | +0.053 | +0.234 | +1.014 | +1.939 |

**This is the headline result.** Haworth's doubles value (+0.619) sits
almost exactly on the replacement line (+0.637 for the 60th-ranked
man), so his V_regular computes to ~0.0000 -- the model independently
concludes he doesn't belong in a regular-discipline lineup, without
being told that Brooklyn already reached the same conclusion. His
entire value runs through the DB channel, and it's the single highest
DB-channel value of any of the 1,033 tracked players. Federico
Staksrud and Hunter Johnson -- the other two specialists named this
session -- land 3rd and 5th. Three real GM decisions, recovered blind.

That doesn't mean the number (+0.064, in tie-win-probability terms) is
right. It means the mechanism -- letting the DB lineup draw from the
same roster independently of the regular lineup -- is doing what it's
supposed to, which is the thing worth checking before trusting any
number it produces.

## What this first cut does NOT establish

- **Not a finished replacement level.** N = 20 teams (excluding
  All-Star/Team-country entries) is a read of the current season, not
  a confirmed number for next season -- same open item as Phase 0.
- **No injury/absence draws.** V assumes the player always plays
  whichever role helps most, every tie. The Monte Carlo layer described
  in `phase0_bench_value.md` isn't built.
- **No real roster construction.** Every player is evaluated in
  isolation against a synthetic replacement-level roster, not against
  the real teammates they'd actually be rostered with. Phase 2's
  archetype tests (A/B/C rosters) still need to happen.
- **The "5 identical replacement players" simplification.** The
  reference roster's five filler slots are literally the same one or
  two real players repeated, not five distinct replacement-tier
  individuals. Fine for isolating one player's marginal value, wrong
  if read as "what a real bottom-of-the-pool roster looks like."
- **Combined uncertainty is a rough approximation** (quadrature sum of
  the four contributing players' own value_now_sd, ignoring the
  weakest-link term and cross-player covariance).

None of these are hidden -- they're in the script's docstring and
repeated here on purpose. Next real step is Phase 2: pick real
archetype rosters (per the original brief: two-stars-four-floor,
four-good-two-floor, four-good-two-DB-specialists) and see what this
engine says about how they'd actually fare against each other.
