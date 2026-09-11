# What moves the Waters team, other than her price? (2026-09-04, late)

User question: is there any persona or dial that makes the competition
between the Waters team and the Bright / Johns teams closer, or is the only
lever "Waters at $900k and everyone else cheap"? Numbers from
`dials_probe.py` (one perfect-information snake draft on the mlp2026 board
with the shipped tag list; the drafted field is held fixed unless stated).

## The three teams as drafted

| team | slot | tie win% vs field | spend | roster |
|---|---|---|---|---|
| Waters | 1 | 66.1% | $999k | Waters $769k + Joseph, Dunlap, Hewett, Tereschenko, Truong ($30-88k) |
| Bright | 2 | 51.5% | $1,000k | Bright $613k + Howells $154k, Irvine, Yu, Blatt, Yang |
| Johns | 3 | 52.4% | $991k | Johns $474k + Tuionetoa $187k, Acevedo $187k, Rane, Crum, Fujiwara |

Head to head: Waters' team beats Bright's 65.9%, Johns' 63.9%. Per game vs
the field her team runs WD 88%, MD 26%, MXD1 77%, MXD2 38%, DreamBreaker
59%: she wins her two, the cheap men lose theirs, and the tie is decided at
2-2 more often than for any other roster.

## Price-side: why there is only one dial

Her value exceeds what any team may pay. The tag already charges the most
the cap allows (cap minus the cheapest legal completion), so every
price-side knob -- floor, cap level, pool size, alpha, how the tag gap is
redistributed -- collapses into one number: how much of her worth she is
charged for. Her roster is always "Waters + whatever is left", and the only
way to weaken it by pricing is to charge more than the cap allows and let
her team fill below the floor (`phase2_pricing.md` §8: reaches the low 50s;
rejected). So yes: on the price side, the only lever is the one the user
named.

## Non-price dials, ranked by leverage

| dial | Waters team | Bright team | Johns team | note |
|---|---|---|---|---|
| as drafted | 66.1% | 51.5% | 52.4% | gap 14-15 pts |
| **playing time / availability**: she plays 90% / 80% / 75% / 67% of ties | 60.7% / 55.3% / 52.6% / 48.2% | -- | -- | without her the team is **11.9%** (Bright's without her 18.8%, Johns' without him 25.7%) |
| **DreamBreaker as a coin flip** | 61.8% | 53.2% | 54.5% | gap 14.6 -> 8.6 pts; her design lives at 2-2 |
| **a rival built to beat HER roster** (Johns + Acevedo + Shimabukuro, cheap women) | -- | -- | -- | beats her team 41.9% (drafted Bright/Johns teams: 34-36%) by conceding WD 3% / MXD1 7% and taking MD 91% / MXD2 82%; costs nothing vs the field (51.4%) |
| **split caps** ($500k women / $500k men, two $10M pools) | **82.4%** vs reference (joint 68.2%) | over cap ($550k) | over cap ($559k) | her curve price in the women's pool is $778k vs a $500k cap; tagged at $408k she gets a full $500k men's side for free; Bright and Johns need tags too |

Measured 2026 playing time (share of the franchise's matchups its best
player appeared in): Johns, Bright, Fahey, Waters, Jorja Johnson, Pisnik,
Sewing all 100%; median franchise 92%, mean 76% (the Black Bears, Hogs and
Miami used 11-12 players). Stars on contending teams do not rotate today.
A rotation rule is therefore the biggest available lever and the one Phase
0 lists as unconfirmed: "every rostered player starts at least a third of
ties" puts her team at ~48% against a fixed field (and hits her harder
than Bright or Johns, whose teams keep 19-26% without them).

## Season format: how many teams can win

Same drafted rosters, different seasons (4,000 seasons each):

| format | Waters team title | runner-up favourite | teams >= 10% | teams >= 5% | effective contenders |
|---|---|---|---|---|---|
| double round robin + top 4 (the sim's default) | 37% | 6% | 1 | 5 | 6.1 |
| single round robin + top 4 | 31% | 6% | 1 | 5 | 8.0 |
| single round robin + top 8 | 25% | 6% | 1 | 8 | 10.5 |
| double round robin, table only | 55% | 5% | 1 | 2 | 3.2 |
| 16-team bracket, random seeds | 14% | 6% | 1 | 7 | 16.6 |
| 3 events (4-tie pool + 8-team bracket), most event titles | 22% | 6% | 1 | 6 | 12.3 |

Variance spreads her title odds over the PACK, evenly: the runner-up never
gets past 6% and the count of teams at 10%+ stays at one, whatever the
format. The English Premier League shape (a favourite and two or three
challengers) needs INEQUALITY in the pack -- slots 2 and 3 getting real
edges -- and a fair price list plus a snake draft is exactly what removes
it. Parity in the middle and a chase at the top are in tension.

Same read from the persona grid (`personas.md`, title-odds concentration):
favourite 33-37%, runner-up 7-9%, one team at 10%+ in every cell where she
is drafted; only the leagues that leave her undrafted (twenty cheapskates,
twenty bargain hunters at $250k) produce two or more 10%+ teams.

## Reads

- On the price side there is one dial and it is the one the user named.
  The cap, not the price rule, creates the lottery prize: she is worth more
  than a team is allowed to pay.
- The levers that work are RULES about how the game is played, not prices:
  playing-time / rotation (biggest; also the one MLP may already have --
  confirm), the DreamBreaker (a coin-flip DB cuts the gap over Bright and
  Johns almost in half), and who her opponents are (a targeted rival gets
  to 42/58 at no cost to its own season).
- Splitting the pools makes it worse, not better: her share of the women's
  pool is 7.8% against a 5% team share, so the tag discount grows, and the
  cap no longer makes her crowd out her own men's side.
- Format changes the title lottery, not the chase: more variance = more
  teams at 5%, never a second team at 10%.

Reproduce: `python value_cap/dials_probe.py` (~2 min).
