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
- **Anna Leigh Waters (W36) is every one of the twenty owners' top woman.**
  Ceilings by persona (k$): star-and-scrubs 800 / 745 / 700; two-stars
  420 / 380 / 420; everyone else 160-400, median 250. Ben Johns is the top
  man on 17 of 20 sheets (consensus $194k vs Tardio $161k).
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
list's ordering at the top of both genders and put Waters first
unanimously; (2) her price is a room-composition question — the maximum
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

Caveats: one model family, one prompt, one packet, twenty sheets, blind
arm only (a named arm — real names, outside knowledge allowed — was not
run); the floor-fill ordering is ours; in-session numbers, not a frozen
result. Cheap enough to re-run with more seats, a second prompt, or names.
