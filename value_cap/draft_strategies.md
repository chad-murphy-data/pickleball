# Draft strategies under the $1M cap -- hypothetical rosters

Prices: alpha = 0.845, one joint $20M pool, $30k floor (`phase2_pricing.py`). Any 6 from the priced pool (3M+3W), four start (top 2 per gender by doubles value), DreamBreaker foursome picked separately by singles value. Each roster is the best legal roster for its strategy against the other strategies' rosters after 2 best-response rounds (k=20 candidate triples per gender per budget split, $10,000 budget grid). No exclusivity between teams.

## Rosters

### Superstar: Waters  --  must roster Anna Leigh Waters, fill around her

- **vs field 52.9%** | vs replacement 96.2% | spend $997,759 (84% on women)
- Starters: Anna Leigh Waters ($725k), Callie Smith ($59k), Martin Emmrich ($59k), Grayson Goldin ($51k)
- Bench: Cailyn Campbell ($53k), Hoang Nam Ly ($51k)
- DreamBreaker four: Anna Leigh Waters, Cailyn Campbell, Grayson Goldin, Hoang Nam Ly  (bench in the DB: Cailyn Campbell, Hoang Nam Ly)

### Quant (no constraint)  --  whatever the optimizer likes best at these prices

- **vs field 51.6%** | vs replacement 97.1% | spend $997,004 (73% on women)
- Starters: Anna Bright ($520k), Alix Truong ($142k), Hunter Johnson ($147k), Harsh Mehta ($72k)
- Bench: Genie Bouchard ($65k), Grayson Goldin ($51k)
- DreamBreaker four: Anna Bright, Genie Bouchard, Hunter Johnson, Grayson Goldin  (bench in the DB: Genie Bouchard, Grayson Goldin)

### Women first  --  >= 65% of spend on the women

- **vs field 51.6%** | vs replacement 97.1% | spend $997,004 (73% on women)
- Starters: Anna Bright ($520k), Alix Truong ($142k), Hunter Johnson ($147k), Harsh Mehta ($72k)
- Bench: Genie Bouchard ($65k), Grayson Goldin ($51k)
- DreamBreaker four: Anna Bright, Genie Bouchard, Hunter Johnson, Grayson Goldin  (bench in the DB: Genie Bouchard, Grayson Goldin)

### DreamBreaker specialist  --  a singles-elite, doubles-middling player on the bench (either gender)

- **vs field 50.7%** | vs replacement 96.9% | spend $995,840 (80% on women)
- Starters: Anna Bright ($520k), Meghan Dizon ($177k), Etienne Blaszkewycz ($67k), Pablo Tellez ($82k)
- Bench: Kaitlyn Christian ($99k), Grayson Goldin ($51k)
- DreamBreaker four: Anna Bright, Kaitlyn Christian, Grayson Goldin, Pablo Tellez  (bench in the DB: Kaitlyn Christian, Grayson Goldin)

### Balanced four  --  no starter over 28% of cap, bench at <= $80k each

- **vs field 50.6%** | vs replacement 97.1% | spend $998,749 (48% on women)
- Starters: Bobbi Oshiro ($243k), Meghan Dizon ($177k), Jay Devilliers ($276k), Nicolas Acevedo ($192k)
- Bench: Christine Maddox ($60k), Grayson Goldin ($51k)
- DreamBreaker four: Bobbi Oshiro, Christine Maddox, Jay Devilliers, Grayson Goldin  (bench in the DB: Christine Maddox, Grayson Goldin)

### Two anchors  --  a man AND a woman each costing >= 35% of cap, fill the rest

- **vs field 50.0%** | vs replacement 96.6% | spend $998,844 (49% on women)
- Starters: Tyra Hurricane Black ($353k), Allie Reichert ($77k), JW Johnson ($387k), Etienne Blaszkewycz ($67k)
- Bench: Christine Maddox ($60k), Matthew Barlow ($55k)
- DreamBreaker four: Christine Maddox, Tyra Hurricane Black, Matthew Barlow, JW Johnson  (bench in the DB: Christine Maddox, Matthew Barlow)

### Men first  --  >= 55% of spend on the men

- **vs field 49.6%** | vs replacement 96.7% | spend $995,807 (42% on women)
- Starters: Vivian Glozman ($262k), Maggie Brascia ($93k), Ben Johns ($420k), George Wall ($94k)
- Bench: Genie Bouchard ($65k), Zane Ford ($62k)
- DreamBreaker four: Genie Bouchard, Vivian Glozman, Ben Johns, Zane Ford  (bench in the DB: Genie Bouchard, Zane Ford)

### Deep six  --  nobody over 20% of cap; six real contributors

- **vs field 43.0%** | vs replacement 96.2% | spend $991,436 (51% on women)
- Starters: Etta Tuionetoa ($192k), Ewa Radzikowska ($183k), Nicolas Acevedo ($192k), Robert Slutsky ($190k)
- Bench: Lea Jansen ($128k), Christopher Haworth ($107k)
- DreamBreaker four: Lea Jansen, Etta Tuionetoa, Christopher Haworth, Robert Slutsky  (bench in the DB: Lea Jansen, Christopher Haworth)

### Cheapskate ($500k)  --  best roster on half the cap (the min-spend floor); reference only, not in the field

- **vs field 17.9%** | vs replacement 81.2% | spend $499,610 (52% on women)
- Starters: Allie Reichert ($77k), Judit Castillo ($118k), Oscar Serra ($104k), Hien Truong ($85k)
- Bench: Genie Bouchard ($65k), Hoang Nam Ly ($51k)
- DreamBreaker four: Judit Castillo, Genie Bouchard, Hoang Nam Ly, Hien Truong  (bench in the DB: Genie Bouchard, Hoang Nam Ly)

## Head to head: P(row beats column), one tie

("mean" and "vs field" exclude the $500k reference roster.)

| | Superstar | Quant | Women first | DreamBreaker specialist | Balanced four | Two anchors | Men first | Deep six | Cheapskate | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| **Superstar** | -- | 51 | 51 | 51 | 54 | 52 | 52 | 59 | 79 | 52.9 |
| **Quant** | 49 | -- | 50 | 51 | 51 | 51 | 52 | 58 | 84 | 51.6 |
| **Women first** | 49 | 50 | -- | 51 | 51 | 51 | 52 | 58 | 84 | 51.6 |
| **DreamBreaker specialist** | 49 | 49 | 49 | -- | 49 | 50 | 50 | 57 | 83 | 50.7 |
| **Balanced four** | 46 | 49 | 49 | 51 | -- | 51 | 51 | 56 | 84 | 50.6 |
| **Two anchors** | 48 | 49 | 49 | 50 | 49 | -- | 51 | 55 | 82 | 50.0 |
| **Men first** | 48 | 48 | 48 | 50 | 49 | 49 | -- | 55 | 82 | 49.6 |
| **Deep six** | 41 | 42 | 42 | 43 | 44 | 45 | 45 | -- | 80 | 43.0 |
| **Cheapskate** | 21 | 16 | 16 | 17 | 16 | 18 | 18 | 20 | -- | 17.9 |
