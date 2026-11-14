# Template rosters, played as a season

Each price list's template rosters (from `draft_strategies*.md`, reference-only rosters excluded) play a double round robin (50,000 seasons, seed 1), standings ties broken by coin flip, top 4 to a 1v4 / 2v3 / final playoff. Tie probabilities come from the real tie model. NOTE: these rosters were built without exclusivity and share players, so this is blueprint-vs-blueprint, not a league. Built by `simulate_templates.py`.

## draft_strategies_tag.md

Prices: alpha = 1.0, one joint $20M pool, $30k floor, **Anna Leigh Waters franchise-tagged at $769,149** (most a team can pay and still field a legal roster; the surplus is spread over the rest of the pool) (`phase2_pricing.py`). Any 6 from the priced pool (3M+3W), four start (top 2 per gender by doubles value), DreamBreaker foursome picked separately by singles value. Each roster is the best legal roster for its strategy against the other strategies' rosters after 2 best-response rounds (k=20 candidate triples per gender per budget split, $10,000 budget grid). No exclusivity between teams.

League = 8 template rosters, 14 ties each per season.

| roster | expected wins | mean wins | P(1st in table) | P(top 4) | **P(title)** | P(last) | spend |
|---|---|---|---|---|---|---|---|
| Balanced four | 7.41 | 7.41 | 16.5% | 58.4% | **15.7%** | 8.4% | $996k |
| DreamBreaker specialist | 7.37 | 7.35 | 15.8% | 57.2% | **15.5%** | 8.8% | $997k |
| Quant (no constraint) | 7.28 | 7.27 | 15.0% | 55.5% | **14.7%** | 9.6% | $999k |
| Men first | 7.15 | 7.17 | 13.9% | 53.3% | **13.8%** | 10.6% | $1,000k |
| Women first | 7.13 | 7.15 | 13.4% | 53.0% | **13.3%** | 10.4% | $999k |
| Superstar: Waters | 7.00 | 7.01 | 11.9% | 50.2% | **12.3%** | 11.9% | $1,000k |
| Two anchors | 6.40 | 6.39 | 7.0% | 37.8% | **7.8%** | 19.1% | $997k |
| Deep six | 6.25 | 6.24 | 6.4% | 34.6% | **6.9%** | 21.2% | $965k |

Parity yardstick: 8 equal teams would each take the title 12.5% of the time and average 7.0 wins.

Finish-position distribution (P of finishing 1st .. last):

| roster | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Balanced four | 17 | 15 | 14 | 13 | 12 | 11 | 10 | 8 |
| DreamBreaker specialist | 16 | 15 | 14 | 13 | 12 | 11 | 10 | 9 |
| Quant (no constraint) | 15 | 14 | 14 | 13 | 12 | 12 | 11 | 10 |
| Men first | 14 | 14 | 13 | 13 | 12 | 12 | 12 | 11 |
| Women first | 13 | 13 | 13 | 13 | 13 | 12 | 12 | 10 |
| Superstar: Waters | 12 | 13 | 13 | 13 | 13 | 13 | 12 | 12 |
| Two anchors | 7 | 9 | 10 | 12 | 13 | 14 | 16 | 19 |
| Deep six | 6 | 8 | 9 | 11 | 12 | 15 | 17 | 21 |

Shared players (why this is not a league):

- Katerina Stewart ($65k): 3 rosters -- Balanced four, Quant (no constraint), Women first
- Grayson Goldin ($44k): 3 rosters -- Men first, Superstar: Waters, Two anchors
- Ewa Radzikowska ($176k): 2 rosters -- Balanced four, Deep six
- Thomas Wilson ($209k): 2 rosters -- Balanced four, Quant (no constraint)
- Jack Sock ($226k): 2 rosters -- Balanced four, DreamBreaker specialist
- Zane Ford ($53k): 2 rosters -- Balanced four, Quant (no constraint)
- Etta Tuionetoa ($187k): 2 rosters -- DreamBreaker specialist, Deep six
- Nicolas Acevedo ($187k): 2 rosters -- DreamBreaker specialist, Deep six
- Genie Bouchard ($55k): 2 rosters -- DreamBreaker specialist, Men first
- Christopher Haworth ($95k): 2 rosters -- DreamBreaker specialist, Deep six
- Cailyn Campbell ($46k): 2 rosters -- Superstar: Waters, Two anchors

## draft_strategies.md

Prices: alpha = 0.845, one joint $20M pool, $30k floor (`phase2_pricing.py`). Any 6 from the priced pool (3M+3W), four start (top 2 per gender by doubles value), DreamBreaker foursome picked separately by singles value. Each roster is the best legal roster for its strategy against the other strategies' rosters after 2 best-response rounds (k=20 candidate triples per gender per budget split, $10,000 budget grid). No exclusivity between teams.

League = 8 template rosters, 14 ties each per season.

| roster | expected wins | mean wins | P(1st in table) | P(top 4) | **P(title)** | P(last) | spend |
|---|---|---|---|---|---|---|---|
| Superstar: Waters | 7.40 | 7.42 | 16.6% | 58.9% | **16.2%** | 8.3% | $998k |
| Quant (no constraint) | 7.23 | 7.23 | 14.5% | 54.8% | **14.3%** | 9.8% | $997k |
| Women first | 7.23 | 7.23 | 14.7% | 54.8% | **14.2%** | 10.0% | $997k |
| DreamBreaker specialist | 7.09 | 7.08 | 13.1% | 51.5% | **13.0%** | 11.1% | $996k |
| Balanced four | 7.08 | 7.08 | 12.8% | 51.3% | **13.0%** | 11.0% | $999k |
| Two anchors | 7.01 | 7.01 | 12.2% | 49.8% | **12.1%** | 12.0% | $999k |
| Men first | 6.94 | 6.93 | 11.1% | 48.6% | **11.7%** | 12.7% | $996k |
| Deep six | 6.02 | 6.02 | 5.0% | 30.3% | **5.5%** | 25.0% | $991k |

Parity yardstick: 8 equal teams would each take the title 12.5% of the time and average 7.0 wins.

Finish-position distribution (P of finishing 1st .. last):

| roster | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Superstar: Waters | 17 | 15 | 14 | 13 | 12 | 11 | 10 | 8 |
| Quant (no constraint) | 14 | 14 | 14 | 13 | 12 | 12 | 11 | 10 |
| Women first | 15 | 14 | 13 | 13 | 12 | 12 | 11 | 10 |
| DreamBreaker specialist | 13 | 13 | 13 | 13 | 13 | 12 | 12 | 11 |
| Balanced four | 13 | 13 | 13 | 13 | 13 | 13 | 12 | 11 |
| Two anchors | 12 | 13 | 12 | 12 | 13 | 13 | 13 | 12 |
| Men first | 11 | 12 | 13 | 12 | 13 | 13 | 13 | 13 |
| Deep six | 5 | 7 | 8 | 10 | 12 | 15 | 18 | 25 |

Shared players (why this is not a league):

- Grayson Goldin ($51k): 5 rosters -- Superstar: Waters, Quant (no constraint), Women first, DreamBreaker specialist, Balanced four
- Anna Bright ($520k): 3 rosters -- Quant (no constraint), Women first, DreamBreaker specialist
- Genie Bouchard ($65k): 3 rosters -- Quant (no constraint), Women first, Men first
- Alix Truong ($142k): 2 rosters -- Quant (no constraint), Women first
- Hunter Johnson ($147k): 2 rosters -- Quant (no constraint), Women first
- Harsh Mehta ($72k): 2 rosters -- Quant (no constraint), Women first
- Meghan Dizon ($177k): 2 rosters -- DreamBreaker specialist, Balanced four
- Etienne Blaszkewycz ($67k): 2 rosters -- DreamBreaker specialist, Two anchors
- Nicolas Acevedo ($192k): 2 rosters -- Balanced four, Deep six
- Christine Maddox ($60k): 2 rosters -- Balanced four, Two anchors
