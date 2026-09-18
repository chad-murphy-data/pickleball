"""value_cap/pool.py -- the priced pool: which 120 players a 20-team league
rosters, and what each is worth for pricing.

    from pool import load_pool
    pool = load_pool("phi")     # gender -> [(player_id, name, value)], 60 each

Two bases, deliberately kept apart:

  "total" -- HANDOFF-era convention: top 60 per gender by Phase 1's V_total
             (player_value.csv). Exists so the 2026-09-04 morning numbers
             (indifference pairs A/B) stay reproducible.
  "phi"   -- the SELF-CONSISTENT pool from shapley_value.py: phi is a
             context-averaged value whose context IS the pool, so the pool
             is iterated until "top 60 by phi" reproduces itself
             (player_value_shapley.csv, in_pool == 1). This is the pricing
             basis phase2_pricing.py uses by default.

Ranks quoted anywhere in value_cap/ ("#3M", "#60 = replacement") are
DOUBLES ranks (data/v2_players.csv order within gender), which is a third
ordering; replacement level = doubles #60 per phase1_value_model.py.
"""
from __future__ import annotations

import csv

from phase1_value_model import N_TEAMS, ROOT

POOL_SIZE = N_TEAMS * 3
VALUE_DIR = ROOT / "value_cap"


def load_pool(basis="phi"):
    pool = {"M": [], "F": []}
    if basis == "total":
        for r in csv.DictReader((VALUE_DIR / "player_value.csv").open()):
            pool[r["gender"]].append((r["player_id"], r["full_name"], float(r["V_total"])))
        for g in pool:
            pool[g].sort(key=lambda t: -t[2])
            pool[g] = pool[g][:POOL_SIZE]
        return pool
    if basis == "phi":
        for r in csv.DictReader((VALUE_DIR / "player_value_shapley.csv").open()):
            if r["in_pool"] == "1":
                pool[r["gender"]].append((r["player_id"], r["full_name"], float(r["phi"])))
        for g in pool:
            pool[g].sort(key=lambda t: -t[2])
            assert len(pool[g]) == POOL_SIZE, (g, len(pool[g]))
        return pool
    raise ValueError(basis)
