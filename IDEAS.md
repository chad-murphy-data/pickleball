# IDEAS.md — content & analysis backlog

Running list of Reddit/Threads post ideas and the analyses behind them.
Newest ideas at the bottom of "To develop." Status tags:

- **READY** — analysis done AND a draft/bundle exists; just needs to be posted.
- **DRAFTABLE** — analysis is done and validated; needs the write-up.
- **TO DEVELOP** — an angle/idea; analysis not yet built (or only partly).

See also: `CLAUDE.md` "Open threads", `ROADMAP.md`, and open PRs.

---

## READY (analysis + draft both done)

- **Clutch — "the best players don't give it away in crunch time"**
  Dave Fleming's cliché checked against 163k rallies. Leaderboard is the
  sport's stars; honest twist = there's no separate "clutch gene," it's
  ~0.58 correlated with raw skill. Bundle: `content/clutch/` (reddit_post.md,
  threads_post.md, plots). Bonus standalone stat lives here too (see below).

- **Waters "ice-out" — STL Shock vs NJ 5s, targeting Jorja Johnson**
  The Mid-Season final headline miss (Waters/Johnson were 88% favorites and
  lost). Weakest-link math shows freezing Waters turns the team into "two
  Jorja Johnsons" → an 88% game becomes a coin flip. Honest caveat: the logs
  can't prove the ice-out *happened* (serve target is rotation-constrained).
  Bundle: `content/waters_iceout/` (reddit, threads, dossier, explainer,
  infographic).

## DRAFTABLE (analysis validated; needs write-up / merge)

- **DreamBreaker match order — Anna Bright's "women don't matter enough" fix**
  Open PR #29 (`claude/dreambreaker-match-order-sim-0td5m8`). Slot 1 plays
  ~1.4× the rallies of slot 4; order alone swings a roster ~7.8pp; the
  current men-first meta is a self-defeating Nash equilibrium. Anna's
  proposed rule (one team sets matchups, the other sets order) breaks it.
  Marquee validated example: **NJ 5s vs Brooklyn** — batting Waters last
  makes NJ an *underdog* (48.4%), current meta a coin flip (51.4%),
  Waters-first a favorite (58.4%); quadruple-checked (clean-room DP + 20M
  Monte Carlo + k/serve sensitivity + real referee logs). Freeze-aware
  nugget: your biggest edge belongs in **slot 2** (the "clinch slot" — the
  average DB ends on point ~37, in slot 2's third pass; final point lands in
  slot 2 in 41% of 88 logged DBs) — worth +0.4pp over biggest-first, +3.1pp
  over men-first. Under the rule, a woman leads off 52% of the time (vs
  0/176 in the real logged record). Source: `model/db_order.md`,
  `db_scenarios.py`, `db_post_notes.md`. TODO: merge PR, promote notes into a
  `content/db_order/` bundle, render the viz.

- **Serve & return — who beats the field on return**
  Per-player serve% and return-over-expectation from the Supabase rally
  warehouse. Waters/Johns/Bright lead on serve; the return over-performers
  are surprising non-marquee names (Thomas Yu, etc.). Engine done
  (`model/serve_return_report.py`, runs off `pb_*` tables / committed CSV);
  no draft or `content/` bundle yet.

## Standalone stats / smaller hooks

- **The single biggest point in pickleball** — down 9-10 receiving on the
  opponent's 2nd server = a 47% win-prob swing. Currently ridealong in the
  clutch post; strong enough to stand alone. Source: `model/big_points*`.

## TO DEVELOP (idea only; analysis not built, or needs reconciling)

- **Cross-gender exhibition value** — a single 2W-vs-2M game carries ~se 0.24
  logit of DIRECT M/W offset info; a weekend of them beats 14k mixed games.
  Can't be published as a ranking (offset is a prior convention). Source:
  `model/cross_gender.md`, CLAUDE.md finding #8.

- **Do women decide mixed? (Samin Odhwani angle)** — added 2026-07-24
  > *"And there's a genuinely useful second layer: MLP's chief strategy
  > officer Samin Odhwani said there's no other sport where men and women
  > compete on court together with an equal opportunity to impact the game —
  > and, more pointedly, that early analysis suggests the female player
  > outweighs the male in terms of winning, and that the reflexive strategy
  > of targeting the woman is statistically the wrong one."* — Chat

  Chad's take: **our evidence points to this being right AND wrong.** Yes,
  the female player outweighs the male in terms of winning — but simply
  because they're targeted more.

  Why it's a great post: it's a real, named, counterintuitive claim from
  the league's own strategist that we can actually adjudicate.

  What our evidence already says (to reconcile before drafting):
  - **Weakest-link is the master key.** A team = ~59% its *weaker* player,
    because opponents choose who to hit to and hit the weak link
    (`model/weakest_link.md`, CLAUDE.md finding #1). In mixed the woman is
    usually the lower-rated player, so she absorbs more balls → more
    apparent impact on the result. That's the mechanism behind "the woman
    outweighs the man": it's a targeting artifact, not a claim that she's
    the better player.
  - **But our fitted gender-role term leans the *opposite* way from "target
    the woman."** The |gap| weakest-link effect is essentially gender-blind;
    the residual gender-role coefficient, at ~1.8σ, leans slightly toward
    *more* weight on the man ("man covers more court"). So "targeting the
    woman is statistically wrong" is NOT clearly supported by our current
    fit — teams are punished for *imbalance per se*, whichever gender is
    weaker. Reconcile Odhwani's "early analysis" against ours.

  The honest thesis to test/write: *the female player does swing mixed
  outcomes more — but as the targeted weak link, not as an underrated
  weapon; "stop targeting the woman" only helps insofar as she isn't
  actually your opponents' weak link.* Needs a dedicated cut (mixed-only,
  targeting proxy from rally logs: who receives the 3rd/5th ball — though
  no shot-level data exists, so it may cap at the weakest-link inference).

## Bookmarks (angles to remember; analysis mostly exists)

- **Is Ben Johns phoning it in during MLP?** (Answer looks like "no.")
  Evidence already in hand: no MLP sandbagging — skill transfers across
  tours (sd_w ≈ 0.13), slope test + player-tour effects both null
  (CLAUDE.md finding #3); and Johns never declined in absolute terms —
  the field rose (finding #4, dynamic model). Post = myth-check framing;
  mostly assembly, not new analysis.

- **How did Bright/Fahey ice out ALW at the Mid-Season tournament?**
  (Not the final title — memory hook.) The math exists:
  `content/waters_iceout/` bundle + `model/iceout_waters.py` — weakest-link
  coverage dial (41% → 0% Waters coverage takes 88% to 48%), "two Jorja
  Johnsons" equivalence, serve-target check from the referee logs
  (Johnson 10 / Waters 9 — the freeze lives mid-rally, unprovable from
  our data). READY bundle; this bookmark is about picking the right
  title/angle when posting.

## Data hygiene / repo TODOs

- **DreamBreaker rally logs undercount points** (found 2026-07-25, DB-post
  verification audit). In the PR #29 branch's `data/db_rallies.csv`,
  21 of 88 matches have fewer rally rows than the true final score
  (t1+t2 from `data/dreambreakers.csv`): 13 short by 1, 6 by 2, 2 by 3 —
  never over. Likely dropped correction/rewind rallies in the log parser.
  Consequence: any stat built on rally-ROW counts inherits the error —
  e.g. the final-point-slot distribution is 34/41/12.5/12.5 on row counts
  but **32/44/11/13 (slots 1+2 = 76.1%) on the trustworthy true-score
  basis**; median length is 36.5 vs the correct 37. Rule going forward:
  derive point counts from final scores, use rally rows only for
  within-game sequence. Root fix: re-derive from raw logs with correction
  handling (mirror `H.rally_events`' approach) and regenerate the CSV.
