# LLM owners — a cheap probe (2026-09-06, in-session measurement)

**Question.** Every auction room in this thread hands the owners rules WE
wrote (personas, learning channels, our own list as a prior). Does anything
survive when the owners are twenty independent language-model agents who
never saw a value, a price, or a name?

**Design.** `llm_owners.py --packet` writes `packet.md`: 183 players from
the 2026 MLP board as anonymous ids (W01-W88, M01-M95, shuffled within
gender), records only — 2026 same-gender doubles W-L and points%, mixed
W-L and points%, 2026 and career PPA singles W-L, MLP matchups played.
Twenty agents (one per seat: 3 each of star-and-scrubs, two-stars,
four-starters, balanced-six, singles-minded, risk-averse, plus 2 with no
persona) each read ONLY the packet plus the league rules and wrote one
sheet: a maximum price for every id. Agents were told not to guess
identities, not to read anything else, not to run anything. Sheets are in
`sheets/`; `key.json` maps ids back to players (the agents never had it).
The deterministic engine then runs the twenty sheets through 200
rotation-nomination second-price rooms (random sale order; an owner with an
open slot and no positive ceiling left takes the room-consensus best
remaining player at the floor) and scores every roster on the tie model
(`draft_sim.season`). One sheet (`four_starters_1`) wrote thousands
instead of dollars and is read as such.

Outputs: `analyze.txt`, `rooms.txt`. Everything below is copied from them.

## What the sheets say

- **Not twenty copies of one owner, and persona barely moves the ranking.**
  Within-gender Spearman between pairs of sheets: same-persona median 0.74
  (W) / 0.71 (M), different-persona 0.71 / 0.67. Persona shows up in the
  price SHAPE (how much of the $1M goes to the top name), not in who is
  ranked where.
- **They mostly sorted the win% column.** Per-owner Spearman vs raw 2026
  same-gender win%: 0.79 (W) / 0.60 (M). Vs the shipped price list, which
  they never saw: 0.51 / 0.46 per owner, 0.56 / 0.49 for the consensus
  (mean ceiling). Lower against the list than against the raw column, so no
  sign of contamination — and no sign the agents see anything the raw
  record does not.
- **Anna Leigh Waters (W36) is the top woman on 19 of 20 sheets** (one
  balanced-six owner priced her at 0 and led with Catherine Parenteau).
  Ceilings by persona (k$): star-and-scrubs 800 / 745 / 700; two-stars
  420 / 380 / 420; everyone else 160-400, median 250. Ben Johns is the top
  man on 12 of 20 sheets (Staksrud 4, Tardio 3; consensus $194k vs
  Tardio $161k).
- **Consensus top of the board matches the list's top in rank, not in
  price.** Women: Waters, Bright, Fahey, Jorja Johnson, Parenteau, Todd,
  Black, Pisnik (list #1, #2, #5, #4, #15, #3, #10, #8). Men: Johns, Tardio,
  Alshon, Patriquin, Staksrud, JW Johnson, Daescu, Ge (list #1, #4, #5, #3,
  #7, #2, #6, #30). Ge and Parenteau are the two win%-driven outliers.
  Consensus mean ceilings run at roughly a third of list prices for the
  top ten, because most sheets spread $1M over six targets.

## What the rooms do

Same twenty sheets in every room, so the 200 seeds vary sale order and
season luck only; prices barely move.

- **Waters sells for $750k in every room** (list $769k; the hand-coded
  no-sheet room's default mix gave $570k). That is the second-highest
  star-and-scrubs ceiling plus the increment: her price is set by whether
  two star-hunters are in the room, exactly the mechanism the hand-coded
  room showed. Take the three star-and-scrubs seats out and the next
  ceiling is $420k.
- **Paying it leaves a non-favourite.** Her team: 61% win, 0.6% title,
  best team in 0 of 200 rooms — five floor teammates. The best team is the
  one full-sheet four-starters owner (Bright + Parenteau + Patriquin +
  JW Johnson + two $30-125k men) at 92%.
- **Frozen sheets make a very unequal league.** Parity spread 82 pts,
  $462k of every $1M unspent, stars at 37-50% of list (Bright $305k, Johns
  $405k), depth at 82-117% of list. Second price with two or three bidders
  per star is a low price; owners who wrote six targets and lost them all
  filled at the floor. Persona table: balanced-six 73% / 14% title,
  singles-minded 61%, risk-averse 49%, four-starters 49% / 16% (one seat
  carries it), no-persona 45%, two-stars 37%, star-and-scrubs 34%.

## What this does and does not establish

Independent of anything we typed: (1) twenty blind owners reproduce the
list's ordering at the top of both genders and put Waters first on
19 of 20 sheets; (2) her price is a room-composition question — the maximum
only when two star-hunters meet on her, ~$420k otherwise; (3) the owner who
pays the maximum is not the favourite. Those three were already the
robust (tier-2) reads from the hand-coded rooms; this is a second,
differently-built witness for them, not a proof.

Not established: any PRICE level. Agents price by dividing $1M over a
target roster, so consensus ceilings sit at a third of list and the room
prices at a half — that is the sheet format and second-price arithmetic,
not a valuation of the players. The rooms also freeze each owner's sheet
(no in-room adaptation), which is why the league is so unequal; a real
owner who lost six targets would re-price the next tier.

Caveats: one model family, one prompt, one packet, twenty sheets; the
floor-fill ordering is ours; in-session numbers, not a frozen result.

# The named arm (same day, user request: "JUST give them names")

`--packet-names` writes `packet_names.md`: the same 183 players as names
only, alphabetical within gender, no records. Same twenty seats, same
prompt otherwise; agents were told to use whatever they already know.
Sheets in `sheets_names/`, key `key_names.json`; outputs
`analyze_names.txt`, `rooms_names.txt` (run with `--key key_names.json`).

## What names alone produce

- **Reputation, not form.** Per-owner Spearman vs the shipped list 0.19
  (W) / 0.23 (M), vs raw 2026 win% 0.17 / 0.17 — both far below the blind
  arm's 0.5 and 0.7-0.8. Named-vs-blind consensus agree at rho 0.11 (W) /
  0.31 (M), 4 of 8 top-8 names in common per gender.
- **The top two per gender survive; the rest is name recognition.**
  Consensus women: Waters, Bright, then Genie Bouchard (list #57, $55k),
  Jessie Irvine (#37), Fahey, Genie Erokhina (#56), Parenteau,
  Rohrabacher. Men: Johns, Tardio, Patriquin, JW Johnson, then Zane
  Navratil (list #85, the $30k floor), Christopher Haworth (#32),
  Tyson McGuffin (#50), Hunter Johnson. Bouchard, Navratil and McGuffin
  are names the agents know from outside pickleball or from years ago.
- **Owners disagree more without data.** Same-persona pairs 0.50 (W) /
  0.61 (M), different-persona 0.51 / 0.59, vs 0.7 in the blind arm.
- **Waters is the top woman on 15 of 20 sheets** (blind arm 19): Bright
  led on 3, Fahey and Rohrabacher on one each; one balanced-six owner
  priced her at 0. Median ceiling
  $410k (blind $250k); star-and-scrubs 850 / 800 / 820; the two
  no-persona owners wrote $800k for her AND $800-900k for Johns.

## What the named rooms do

- **Waters $825k in every room** (blind $750k, list $769k), again to a
  star-and-scrubs seat, again the second-highest star-hunter ceiling.
  Johns $805k (list $474k), Bouchard $355k (list $55k), Navratil $275k
  (list $30k), Haworth $305k, McGuffin $235k.
- **Her team is 81% / 23% title and the best team in 14% of rooms** —
  much stronger than the blind arm's 61% / 0.6%. The reason is the fill:
  in a room that bids on names, the players nobody recognises go unsold,
  and the floor fill (room consensus, then shrunk 2026 win%) hands the
  star buyer real 2026 performers at $30k. A reputation room leaves
  current value on the floor for whoever is still buying.
- **Prices vs list by band:** #1-5 at 74%, #6-15 at 54%, #16-30 at 24%,
  #31-60 at 30%, unpriced players at 167%. The middle of the real list
  is where names fail: the room pays for the top of the marquee and for
  names it remembers, and nothing for the 2026 regulars in between.
- Persona table: no-persona 65% / 11% (they wrote the biggest numbers on
  the biggest names), two-stars 57%, star-and-scrubs 55% / 8%,
  risk-averse 53%, four-starters 52% / 11%, balanced-six 39%,
  singles-minded 34% (they paid $305k for Haworth's name). Spread 76
  pts, $427k per team unspent.

## Read across the two arms

Records without names reproduce the list's ordering at rho ~0.5 and put
the price on Waters only through the star-hunter persona. Names without
records reproduce only the top two per gender and spend the rest on
memory. The list is the thing neither arm has: current form for all 183,
on one scale. Both arms agree on the one robust mechanism — her price is
the second-highest star-hunter's number, and the buyer's team strength
is set by what the rest of the room leaves on the floor, not by her.

## The named arm (same day; `packet_names.md`, `sheets_names/`, `analyze_names.txt`, `rooms_names.txt`)

Same twenty seats and prompt, but the packet is nothing except the 183
names, alphabetical by surname. No records. Whatever the agents bring is
outside knowledge, and it is stale knowledge: the model behind the agents
stopped learning about pickleball well before the 2026 season.

- **Reputation replaces the record.** Consensus top women: Waters,
  Bright, Genie Bouchard (list #57), Jessie Irvine (#37), Fahey, Genie
  Erokhina (#56), Parenteau, Rohrabacher. Men: Johns, Tardio, Patriquin,
  JW Johnson, Zane Navratil (#85), Christopher Haworth (#32), Tyson
  McGuffin (#50), Hunter Johnson. Spearman vs the list falls from ~0.5
  (blind) to 0.21 / 0.17, and vs raw 2026 win% to 0.17 / 0.17. The two
  arms' consensus rankings agree at only 0.11 (W) / 0.31 (M), 4 of 8 top
  names in common per gender.
- **Owners agree less with each other** (same-persona 0.50 / 0.61,
  different-persona 0.51 / 0.59 vs ~0.7 blind): a name is a weaker
  shared signal than a win% column.
- **Ceilings are higher and dense.** 18 of 20 priced all 183 (blind: 8);
  top-6 sums run $1.0-4.2M. Waters' ceilings: star-and-scrubs 850 / 800 /
  820, the two no-persona owners 800 / 800, median 410k (blind 250k).
  Johns is priced up to $900k.
- **Rooms.** Waters $825k in every room (the $850k first-buy max minus an
  increment; list $769k), Johns $805k [555-805], Bright $455k, then
  Bouchard $355k, Haworth $305k, Irvine $280k, Navratil $275k, McGuffin
  $235k. List #16-30 sell at 24% of list, #31-60 at 30%: Todd, Jorja
  Johnson, Alshon, Staksrud go for less than names that the 2026 record
  puts on the replacement line. **Her team is 81% / 23% title** (blind
  61% / 0.6%): not because she got cheaper, but because the field spent
  its money on the wrong players and the real #3-#30 were left for the
  floor fills. Best team is still someone else's in 86% of rooms (a
  no-persona owner at 90.6%, on Johns + the underpriced middle).

What that adds: the blind and named arms disagree with each other far
more than either disagrees with the list, and the named arm is the one
that is wrong on the record — so "owners who go by name" is a concrete,
testable way MLP's real prices could depart from the list (old names
over new results). It is also the arm where paying the max for Waters is
NOT the worst way to own her, because the rest of the room is worse.

# Running it again with your own prompt and personas

1. Write a personas file (see `personas_v1.json`, the v1 seats above):
   `{"personas": {name: text}, "seats": [name, ...], "template": optional}`.
   The template, if given, is the whole agent prompt with `{persona_text}`,
   `{persona}`, `{packet_path}`, `{packet_note}`, `{sheet_path}`,
   `{n_players}`, `{id_range}` placeholders; the default is in
   `llm_owners.py` (`PROMPT_TEMPLATE`).
2. `python value_cap/llm_owners.py --prompts <file> --arm blind|names --out <dir>`
   writes one `<seat>.prompt.txt` per seat; a session launches one agent
   per prompt verbatim and the agents write `<seat>.json` into the same
   directory.
3. `--analyze <dir>` (add `--key key_names.json` for the names arm, and
   `--compare <other dir>` for arm-vs-arm), then `--rooms <dir>`.

Prompt lessons from v1, so the next prompt does not repeat them:
- Say a ceiling is a maximum, not a spending plan. Half the v1 owners
  wrote six numbers that add to exactly $1M (a plan) and zeros elsewhere;
  the other half wrote maxima summing to $2-4M. Those are different
  objects and the room treats them the same.
- Sparse sheets go silent once their targets are gone, and the engine
  fills them at the floor by room consensus. Ask for a number on every
  player they would take at any price, or accept that the fill is ours.
- Units: one owner wrote thousands. The prompt now gives a dollar example.
- Names arm: say what to do with a name they do not know (v1 said "price
  accordingly"; most wrote 0, some wrote $30-50k).
- Personas written as roster SHAPES ("40% each on two stars") produce the
  shape by construction. Personas written as BELIEFS about what wins
  leave the shape to the owner, which is the more informative run.
