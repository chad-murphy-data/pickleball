# Pool size x floor sweep -- does a bigger priced pool or a different floor blunt the #1 pick?

Every cell: phi rebuilt for a pool of P per gender (self-consistent, replacement = doubles #P), Waters franchise-tagged at alpha 1 on one joint $20M pool with the given floor, then a 20-team snake draft on the real-2026-fill-in board (`draft_sim.set_board('mlp2026', pool)`): one perfect-information draft and 10 drafts at 10% owner error, 200 seasons each. Built by `pool_floor_sweep.py`.

## Waters' share of the pool's value (the thing pool size moves)

| P per gender | positive-phi players | Waters share | 1/20 = fair team share | #1-#6 women phi | #1-#6 men phi |
|---|---|---|---|---|---|
| 60 | 120 | 5.32% | 5.00% | 0.437 0.290 0.228 0.221 0.210 0.205 | 0.221 0.199 0.194 0.188 0.184 0.146 |
| 80 | 140 | 4.20% | 5.00% | 0.448 0.315 0.256 0.250 0.241 0.234 | 0.222 0.200 0.196 0.190 0.188 0.150 |
| 100 | 166 | 3.25% | 5.00% | 0.443 0.322 0.269 0.262 0.254 0.246 | 0.235 0.215 0.210 0.205 0.204 0.169 |

## The grid

| P | floor | Waters price | Bright | Johns | slot-1 win% (perfect) | slot-1 win% / title% (10% error) | Bright team | Johns team | slots 2-20 mean | parity spread | mean spend | floor players taken | top-30 undrafted (10% error) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | $10k | $857k | $679k | $519k | 66.1% | **65.6%** / 36% | 51.8% | 50.5% | 49.2% | 4.3 pts | $972k | 15 | none |
| 60 | $20k | $813k | $646k | $497k | 66.1% | **65.6%** / 36% | 51.8% | 50.5% | 49.2% | 4.3 pts | $974k | 15 | none |
| 60 | $30k | $769k | $613k | $474k | 66.1% | **65.6%** / 36% | 51.8% | 50.5% | 49.2% | 4.3 pts | $976k | 15 | none |
| 60 | $50k | $681k | $548k | $429k | 66.1% | **65.6%** / 36% | 51.8% | 50.5% | 49.2% | 4.3 pts | $979k | 15 | none |
| 60 | $75k | $571k | $466k | $373k | 66.1% | **65.6%** / 36% | 51.8% | 50.5% | 49.2% | 4.3 pts | $984k | 15 | none |
| 80 | $10k | $950k | $549k | $390k | 57.9% | **57.3%** / 13% | 61.0% | 51.9% | 49.5% | 5.4 pts | $911k | 6 | none |
| 80 | $20k | $900k | $511k | $366k | 57.9% | **57.3%** / 13% | 61.9% | 52.7% | 49.4% | 5.5 pts | $899k | 6 | none |
| 80 | $30k | $850k | $474k | $343k | 57.9% | **57.2%** / 14% | 63.4% | 53.0% | 49.3% | 5.8 pts | $887k | 6 | none |
| 80 | $50k | $750k | $399k | $296k | 57.8% | **57.2%** / 13% | 65.6% | 52.5% | 49.2% | 6.1 pts | $858k | 6 | none |
| 80 | $75k | $625k | $305k | $237k | 57.8% | **57.2%** / 13% | 65.7% | 52.6% | 49.2% | 6.1 pts | $823k | 6 | none |
| 100 | $10k | $950k | $427k | $314k | 57.3% | **57.6%** / 12% | 64.3% | 49.6% | 49.2% | 5.8 pts | $852k | 4 | none |
| 100 | $20k | $900k | $389k | $290k | 57.3% | **57.6%** / 12% | 64.5% | 49.4% | 49.2% | 5.8 pts | $824k | 4 | none |
| 100 | $30k | $850k | $352k | $265k | 57.3% | **57.6%** / 12% | 64.5% | 49.4% | 49.2% | 5.8 pts | $797k | 4 | none |
| 100 | $50k | $750k | $277k | $216k | 57.3% | **57.6%** / 12% | 64.5% | 49.4% | 49.2% | 5.8 pts | $741k | 4 | none |
| 100 | $75k | $625k | $184k | $154k | 57.3% | **57.6%** / 12% | 64.5% | 49.4% | 49.2% | 5.8 pts | $672k | 4 | none |

Reading guide: 'floor players taken' counts drafted players priced AT the floor (fill-ins and any priced player whose curve price rounds to the floor). Undrafted = priced players inside their gender's top 30 by phi left on the board in at least one draft (info, not a test). Parity = 50% / 5%.

## What this says

- **The floor is cosmetic at P=60.** Every floor from $10k to $75k gives the identical draft: same slot-1 team at 66% / 36% title, same spread, same 15 floor players taken. Prices scale together and the tag re-pins Waters at cap minus five cheapest completions, so nothing an owner compares ever changes.
- **A bigger priced pool lowers Waters' team to ~57% / 12-14% title, but for the wrong reason.** At P=80/100 the $20M is spread over 20-46 more players per gender who never get drafted, so every drafted dollar buys more; spend falls to $670-910k of the $1M cap and the prize moves from Waters (pinned at the tag, now ABOVE her curve price) to Bright's team (61-66%). The parity spread widens (4.3 -> 5.4-6.1), which is the opposite of what a cap is for.
- **The cleaner version makes Waters STRONGER.** Probe (not in the grid): price only the top 60/gender but measure value against replacement #60 / #80 / #100 at floor $30k. Waters' tagged price goes $769k -> $744k -> $632k, and her team goes 66% -> 66-68% -> 72-73% (title 35% -> 45-50%), Bright's team 52 -> 58 -> 61%, spread 4.3 -> 5.7 -> 7.4. A deeper replacement line makes the whole top cheaper relative to the cap, and she is the top of the top.
- Net: neither pool size nor the floor is a lever on the #1 pick. The only thing that has moved her team is the tag accounting rule (charge her full curve price, lower the floor for the tagged team), and that only reaches the low 50s.
