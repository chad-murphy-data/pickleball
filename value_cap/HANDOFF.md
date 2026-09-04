# MLP Value Cap — HANDOFF (2026-09-04)

Dated status snapshot + next-thread to-do. Read this first; `phase0_bench_value.md`,
`phase1_first_cut.md`, and `phase2_notes.md` are the layered technical record
underneath it if you need the full derivation of something below.

## Where we are

Phase 0 (does the bench have value) and Phase 1 (the value model, `V` per
player) are built and documented elsewhere. Phase 2 (turning `V` into
dollars) is mid-flight, and it changed shape from the original plan: rather
than fitting to 2-3 hand-picked archetype rosters, it turned out to be more
productive to **search for real indifference points** (pairs of totally
different rosters that the model says are equally competitive) and use
those as the fit targets. Nobody picked these pairs to make a point — they
fell out of rank sweeps — which is better-provenance data than hand-picked
examples would have been.

Everything below reuses the production simulation exactly as built in
Phase 1: a real 4-game tie (WD/MD/MXD1/MXD2) plus DreamBreaker,
`web/sitelib/race.py`'s race-to-11 DP + weakest-link gamma for the regular
games, the DB singles-gap model (`model/db_model.md`) for the tiebreaker.
Nothing here is a shortcut or a proxy calculation.

## Session findings (2026-09-02 through 09-04)

1. **Two empirically-discovered indifference pairs**, both found by
   sweeping contiguous rank blocks against a fixed "1 anchor + 2
   near-useless bench players" roster until the win probability crossed
   50%:
   - **Pair A**: Patriquin(M,#3) + Loong(#79M) + Heerboth(#80M) / Bright(F,#2)
     + Stewart(#79F) + Cederquist(#80F) ties Munro(#17M) + Duong(#18M) +
     Lange(replacement,#60M) / Parenteau(#17F) + David(#18F) +
     Frantova(replacement,#60F) at 49.8%/50.2%. Price-matching this pair
     (at a $30k floor) needs **alpha ≈ 0.88**.
   - **Pair B**: same bench shape, anchor swapped to Newman(M,#10) +
     Fahey(F,#10); balances against ranks 24-27 + replacement at ~50%.
     Needs **alpha ≈ 0.76**.
   - **Floor turns out to be irrelevant to which alpha reconciles a
     pair** — proven algebraically, not just observed: floor contributes
     an identical dollar amount to both sides of any same-size-roster
     comparison, so it cancels out of the difference completely. Only
     alpha decides which archetype wins; floor only sets the overall
     price spread.
   - Two pairs implying two different alphas (0.88 vs 0.76) is not a
     failure. A single power law was never going to make every pair of
     equally-good rosters cost exactly the same — and if it did, there'd
     be no GM skill left, which the original brief explicitly didn't
     want. Working range for now: **alpha ≈ 0.76-0.88**, sub-linear
     (a "star discount" relative to proportional, not a premium).

2. **The Waters problem, and a correction to how we tested it.**
   - Even with the worst plausible supporting cast, a Waters-anchored
     roster needs the opposition to reach all the way to **rank ~8-9 in
     both genders** (essentially the rest of the sport's real elite
     tier) just to tie — a much bigger ask than pair A or B.
   - First test (methodologically flawed): compared her fixed roster
     against a FIXED alternative roster. Breakeven alpha = 1.16, and at
     that alpha both rosters cost $1,116,007 — over the $1M cap. Read at
     the time as "structural, can't be fixed under this cap."
   - **Corrected test**: once the alternative roster is allowed to
     actually reoptimize its purchases at the real resulting prices
     (what a rational GM would do instead of sitting on a fixed
     lineup), the crossover drops to **alpha ≈ 0.96-0.98** — the same
     neighborhood as pairs A and B. The "structural, unfixable"
     conclusion was an artifact of testing against a roster that
     couldn't respond to prices. No separate mechanism is needed for
     Waters specifically; the same sub-linear alpha that handles A and B
     handles her too, once the comparison is done properly.

3. **Chad's tiered-pricing idea** (Waters $850k, Bright $700k, Jorja down
   to Kate Fahey/Danni-Elle Townsend $500k→$400k, presumably a smoother
   curve below rank ~10) is a genuinely different design from a single
   global alpha, not a restatement of finding 2 — it hand-sets honest,
   generous prices for the confirmed outliers and lets a flatter curve
   run underneath for the broad market, rather than asking one exponent
   to do both jobs. **Partially tested, hit a real accounting bug, not
   yet properly redone**: the top-tier hand-set dollars need to be
   carved out of the shared $10M pool before running the "everyone else"
   formula on the honest remainder, or the rest of the market gets an
   accidental windfall (which is what happened on the first attempt —
   the "cheap" depth priced absurdly high because the full $10M was
   still being split across only the remaining 58 women). This is the
   single most valuable unfinished thread — see Next Step 2 below.

4. **The $500k minimum team spend rule** (Chad's recollection this
   session — needs the same official confirmation as Phase 0's other
   rules items). Whether it does real work depends entirely on alpha, at
   the $30k floor used throughout:
   - alpha=0.5: cheapest possible legal roster already $679k — rule is
     redundant.
   - alpha=1.0: cheapest legal roster drops to $458k — **rule binds**.
   - alpha=1.5/2.0/3.0: $323k / $248k / $193k — binds harder.
   This is a real, independently-confirmable rule that rules out the
   "field 6 scrubs" degenerate strategy Phase 0 worried about, at least
   at linear-or-higher alpha.

5. **Chad's floor/ceiling decoupling idea**: keep the floor very low
   (e.g. $5,000) and let the $500k minimum-spend rule (finding 4) do the
   work of preventing scrub-stacking, instead of a high floor. Tested:
   - Mechanism works as intended: at floor=$5,000 the rule is redundant
     at alpha=0.5 ($620k cheapest roster) but binds hard from alpha=0.85
     up ($426k → $199k across 0.85-1.5).
   - **Does not solve the Waters problem as much as hoped.** Raises the
     hard per-player ceiling from $850k to $975k, but the Waters
     crossover only moves from alpha≈0.97 to alpha≈0.90, because a lower
     floor also raises everyone else's SHARE-based price (more of the
     $10M pool flows through the merit formula instead of flat floor
     payments), which partly offsets the extra ceiling room.
   - Takeaway: **floor/min-spend and alpha are answering two different
     questions** — floor prevents scrub-stacking, alpha shapes
     competitive balance among real strategies — and shouldn't be tuned
     as if either one fixes both.

6. **Open, unresolved as of this session's end**: the $10M-per-gender
   subpool itself (used everywhere above) is a 50/50 split of a
   "20 teams × $1M cap = $20M league total" — exactly as unexamined an
   assumption as floor and alpha have been all along. Given the real
   value asymmetry between the men's market (flat, no real cliff) and
   the women's market (Waters, then Bright, then a flat pack), an even
   split isn't obviously the right one.

## Next steps

1. **Test the gender-split assumption (finding 6).** Parametrize the
   50/50 split (e.g. try 55/45, 60/40 favoring women, since that's where
   the outlier value actually sits) and rerun the three core checks —
   pair A, pair B, and the Waters-vs-reoptimized-alternative crossover —
   to see whether a single alpha reconciles more cleanly once the split
   isn't forced to be even. This is the live thread Chad and I were
   mid-conversation on when this handoff was written.
2. **Properly retest Chad's tiered-pricing idea (finding 3)** with the
   pool accounting fixed: carve Waters ($850k) and Bright ($700k) out of
   the $10M women's pool first, run the flat curve (alpha ~0.85) on the
   honest $8.45M remainder across the other 58 women, then check whether
   a Bright-anchored $1M roster with REAL optimized depth (not the
   floor-only filler from the first, buggy attempt) is competitive with
   a maxed-out Waters roster.
3. **Confirm the $500k minimum-spend rule officially** — same status as
   Phase 0's still-open rotation/substitution/fatigue items. Currently
   resting on Chad's recollection ("supposedly").

## Reusable code

- `value_cap/phase1_value_model.py` — the value model. `load_doubles`,
  `load_singles`, and `tie_win_prob` are what every ad hoc search this
  session was built on.
- `value_cap/phase2_price_model.py` — pricing + search infrastructure:
  `load_pool`, `prices_for_alpha`, `all_triples`, `optimize`,
  `best_frontier`/`lookup`.
- Everything else this session (the rank-sweep searches that found pairs
  A and B, the Waters ceiling tests, the floor-sensitivity checks) was
  one-off `python3 -c "..."` snippets built on those two modules and was
  **not saved as standalone scripts**. If this thread continues, the
  rank-sweep-for-an-indifference-point pattern used repeatedly tonight is
  worth promoting to a real reusable function (something like
  `find_crossover(anchor_roster, sweep_spec) -> rank, alpha`) rather than
  re-deriving the same boilerplate searches again next time.
