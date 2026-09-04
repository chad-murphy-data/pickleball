# Draft strategies under the $1M cap -- hypothetical rosters

Prices: alpha = 1.0, one joint $20M pool, $30k floor, **Anna Leigh Waters franchise-tagged at $769,149** (most a team can pay and still field a legal roster; the surplus is spread over the rest of the pool) (`phase2_pricing.py`). Any 6 from the priced pool (3M+3W), four start (top 2 per gender by doubles value), DreamBreaker foursome picked separately by singles value. Each roster is the best legal roster for its strategy against the other strategies' rosters after 2 best-response rounds (k=20 candidate triples per gender per budget split, $10,000 budget grid). No exclusivity between teams.

## Rosters

### Balanced four  --  no starter over 28% of cap, bench at <= $80k each

- **vs field 52.4%** | vs replacement 97.2% | spend $995,560 (51% on women)
- Starters: Catherine Parenteau ($267k), Ewa Radzikowska ($176k), Thomas Wilson ($209k), Jack Sock ($226k)
- Bench: Katerina Stewart ($65k), Zane Ford ($53k)
- DreamBreaker four: Catherine Parenteau, Katerina Stewart, Jack Sock, Zane Ford  (bench in the DB: Katerina Stewart, Zane Ford)

### DreamBreaker specialist  --  a singles-elite, doubles-middling player on the bench (either gender)

- **vs field 51.7%** | vs replacement 97.1% | spend $997,332 (49% on women)
- Starters: Bobbi Oshiro ($247k), Etta Tuionetoa ($187k), Nicolas Acevedo ($187k), Jack Sock ($226k)
- Bench: Genie Bouchard ($55k), Christopher Haworth ($95k)
- DreamBreaker four: Genie Bouchard, Bobbi Oshiro, Christopher Haworth, Jack Sock  (bench in the DB: Genie Bouchard, Christopher Haworth)

### Quant (no constraint)  --  whatever the optimizer likes best at these prices

- **vs field 51.3%** | vs replacement 97.3% | spend $999,507 (56% on women)
- Starters: Vivian Glozman ($271k), Lacy Schneemann ($225k), Nicolas Acevedo ($187k), Connor Garnett ($202k)
- Bench: Katerina Stewart ($65k), Oliver Frank ($51k)
- DreamBreaker four: Katerina Stewart, Vivian Glozman, Connor Garnett, Oliver Frank  (bench in the DB: Katerina Stewart, Oliver Frank)

### Men first  --  >= 55% of spend on the men

- **vs field 50.5%** | vs replacement 96.8% | spend $999,709 (39% on women)
- Starters: Ewa Radzikowska ($176k), Jillian Braverman ($149k), Federico Staksrud ($315k), Riley Newman ($251k)
- Bench: Mary Brascia ($65k), Grayson Goldin ($44k)
- DreamBreaker four: Mary Brascia, Jillian Braverman, Federico Staksrud, Grayson Goldin  (bench in the DB: Mary Brascia, Grayson Goldin)

### Women first  --  >= 65% of spend on the women

- **vs field 50.2%** | vs replacement 96.7% | spend $999,326 (66% on women)
- Starters: Danni-Elle Townsend ($325k), Vivian Glozman ($271k), Phuc Huynh ($210k), George Wall ($83k)
- Bench: Seone Mendez ($64k), Matthew Barlow ($48k)
- DreamBreaker four: Seone Mendez, Vivian Glozman, Phuc Huynh, Matthew Barlow  (bench in the DB: Seone Mendez, Matthew Barlow)

### Deep six  --  nobody over 20% of cap; six real contributors

- **vs field 43.8%** | vs replacement 96.2% | spend $964,811 (52% on women)
- Starters: Etta Tuionetoa ($187k), Ewa Radzikowska ($176k), Nicolas Acevedo ($187k), Robert Slutsky ($185k)
- Bench: Brooke Buckner ($134k), Christopher Haworth ($95k)
- DreamBreaker four: Brooke Buckner, Etta Tuionetoa, Christopher Haworth, Robert Slutsky  (bench in the DB: Brooke Buckner, Christopher Haworth)

### Cheapskate ($500k)  --  best roster on half the cap (the min-spend floor); reference only, not in the field

- **vs field 21.0%** | vs replacement 83.8% | spend $499,559 (56% on women)
- Starters: Jillian Braverman ($149k), Allie Reichert ($66k), George Wall ($83k), Thomas Yu ($84k)
- Bench: Katerina Stewart ($65k), Zane Ford ($53k)
- DreamBreaker four: Katerina Stewart, Jillian Braverman, Zane Ford, Thomas Yu  (bench in the DB: Katerina Stewart, Zane Ford)

## Head to head: P(row beats column), one tie

("mean" and "vs field" exclude the $500k reference roster.)

| | Balanced four | DreamBreaker specialist | Quant | Men first | Women first | Deep six | Cheapskate | mean |
|---|---|---|---|---|---|---|---|---|
| **Balanced four** | -- | 51 | 51 | 52 | 52 | 57 | 80 | 52.4 |
| **DreamBreaker specialist** | 49 | -- | 50 | 51 | 52 | 56 | 80 | 51.7 |
| **Quant** | 49 | 50 | -- | 50 | 51 | 56 | 80 | 51.3 |
| **Men first** | 48 | 49 | 50 | -- | 50 | 55 | 79 | 50.5 |
| **Women first** | 48 | 48 | 49 | 50 | -- | 55 | 78 | 50.2 |
| **Deep six** | 43 | 44 | 44 | 45 | 45 | -- | 76 | 43.8 |
| **Cheapskate** | 20 | 20 | 20 | 21 | 22 | 24 | -- | 21.0 |
