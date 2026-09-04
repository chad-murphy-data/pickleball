# Draft strategies under the $1M cap -- hypothetical rosters

Prices: alpha = 1.0, one joint $20M pool, $30k floor, **Anna Leigh Waters franchise-tagged at $769,149** (most a team can pay and still field a legal roster; the surplus is spread over the rest of the pool) (`phase2_pricing.py`). Any 6 from the priced pool (3M+3W), four start (top 2 per gender by doubles value), DreamBreaker foursome picked separately by singles value. Each roster is the best legal roster for its strategy against the other strategies' rosters after 2 best-response rounds (k=20 candidate triples per gender per budget split, $10,000 budget grid). No exclusivity between teams.

## Rosters

### Balanced four  --  no starter over 28% of cap, bench at <= $80k each

- **vs field 53.0%** | vs replacement 97.2% | spend $995,560 (51% on women)
- Starters: Catherine Parenteau ($267k), Ewa Radzikowska ($176k), Thomas Wilson ($209k), Jack Sock ($226k)
- Bench: Katerina Stewart ($65k), Zane Ford ($53k)
- DreamBreaker four: Catherine Parenteau, Katerina Stewart, Jack Sock, Zane Ford  (bench in the DB: Katerina Stewart, Zane Ford)

### DreamBreaker specialist  --  a singles-elite, doubles-middling player on the bench (either gender)

- **vs field 52.7%** | vs replacement 97.1% | spend $997,332 (49% on women)
- Starters: Bobbi Oshiro ($247k), Etta Tuionetoa ($187k), Nicolas Acevedo ($187k), Jack Sock ($226k)
- Bench: Genie Bouchard ($55k), Christopher Haworth ($95k)
- DreamBreaker four: Genie Bouchard, Bobbi Oshiro, Christopher Haworth, Jack Sock  (bench in the DB: Genie Bouchard, Christopher Haworth)

### Quant (no constraint)  --  whatever the optimizer likes best at these prices

- **vs field 52.0%** | vs replacement 97.0% | spend $999,447 (56% on women)
- Starters: Vivian Glozman ($271k), Lacy Schneemann ($225k), Thomas Wilson ($209k), Noe Khlif ($178k)
- Bench: Katerina Stewart ($65k), Zane Ford ($53k)
- DreamBreaker four: Katerina Stewart, Vivian Glozman, Noe Khlif, Zane Ford  (bench in the DB: Katerina Stewart, Zane Ford)

### Men first  --  >= 55% of spend on the men

- **vs field 51.1%** | vs replacement 97.3% | spend $999,829 (39% on women)
- Starters: Sahra Dennehy ($203k), Alix Truong ($132k), Federico Staksrud ($315k), Riley Newman ($251k)
- Bench: Genie Bouchard ($55k), Grayson Goldin ($44k)
- DreamBreaker four: Sahra Dennehy, Genie Bouchard, Federico Staksrud, Grayson Goldin  (bench in the DB: Genie Bouchard, Grayson Goldin)

### Women first  --  >= 65% of spend on the women

- **vs field 50.9%** | vs replacement 96.9% | spend $999,145 (74% on women)
- Starters: Danni-Elle Townsend ($325k), Sofia Sewing ($350k), Armaan Bhatia ($130k), George Wall ($83k)
- Bench: Katerina Stewart ($65k), Matthew Barlow ($48k)
- DreamBreaker four: Katerina Stewart, Sofia Sewing, Matthew Barlow, Armaan Bhatia  (bench in the DB: Katerina Stewart, Matthew Barlow)

### Superstar: Waters  --  must roster Anna Leigh Waters, fill around her

- **vs field 50.0%** | vs replacement 95.7% | spend $1,000,000 (87% on women)
- Starters: Anna Leigh Waters ($769k), Callie Smith ($50k), Mohaned Alhouni ($47k), Grayson Goldin ($44k)
- Bench: Cailyn Campbell ($46k), Hoang Nam Ly ($44k)
- DreamBreaker four: Anna Leigh Waters, Cailyn Campbell, Grayson Goldin, Hoang Nam Ly  (bench in the DB: Cailyn Campbell, Hoang Nam Ly)

### Two anchors  --  a man AND a woman each costing >= 35% of cap, fill the rest

- **vs field 45.7%** | vs replacement 95.9% | spend $996,789 (50% on women)
- Starters: Tyra Hurricane Black ($386k), Allie Reichert ($66k), Christian Alshon ($400k), Juan Benitez ($55k)
- Bench: Cailyn Campbell ($46k), Grayson Goldin ($44k)
- DreamBreaker four: Cailyn Campbell, Tyra Hurricane Black, Christian Alshon, Grayson Goldin  (bench in the DB: Cailyn Campbell, Grayson Goldin)

### Deep six  --  nobody over 20% of cap; six real contributors

- **vs field 44.6%** | vs replacement 96.2% | spend $964,811 (52% on women)
- Starters: Etta Tuionetoa ($187k), Ewa Radzikowska ($176k), Nicolas Acevedo ($187k), Robert Slutsky ($185k)
- Bench: Brooke Buckner ($134k), Christopher Haworth ($95k)
- DreamBreaker four: Brooke Buckner, Etta Tuionetoa, Christopher Haworth, Robert Slutsky  (bench in the DB: Brooke Buckner, Christopher Haworth)

### Cheapskate ($500k)  --  best roster on half the cap (the min-spend floor); reference only, not in the field

- **vs field 22.0%** | vs replacement 83.8% | spend $499,559 (56% on women)
- Starters: Jillian Braverman ($149k), Allie Reichert ($66k), George Wall ($83k), Thomas Yu ($84k)
- Bench: Katerina Stewart ($65k), Zane Ford ($53k)
- DreamBreaker four: Katerina Stewart, Jillian Braverman, Zane Ford, Thomas Yu  (bench in the DB: Katerina Stewart, Zane Ford)

## Head to head: P(row beats column), one tie

("mean" and "vs field" exclude the $500k reference roster.)

| | Balanced four | DreamBreaker specialist | Quant | Men first | Women first | Superstar | Two anchors | Deep six | Cheapskate | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| **Balanced four** | -- | 51 | 51 | 52 | 51 | 52 | 57 | 57 | 80 | 53.0 |
| **DreamBreaker specialist** | 49 | -- | 50 | 51 | 52 | 53 | 56 | 56 | 80 | 52.7 |
| **Quant** | 49 | 50 | -- | 51 | 50 | 52 | 56 | 56 | 80 | 52.0 |
| **Men first** | 48 | 49 | 49 | -- | 50 | 51 | 55 | 55 | 80 | 51.1 |
| **Women first** | 49 | 48 | 50 | 50 | -- | 50 | 54 | 56 | 79 | 50.9 |
| **Superstar** | 48 | 47 | 48 | 49 | 50 | -- | 52 | 56 | 74 | 50.0 |
| **Two anchors** | 43 | 44 | 44 | 45 | 46 | 48 | -- | 51 | 75 | 45.7 |
| **Deep six** | 43 | 44 | 44 | 45 | 44 | 44 | 49 | -- | 76 | 44.6 |
| **Cheapskate** | 20 | 20 | 20 | 20 | 21 | 26 | 25 | 24 | -- | 22.0 |
