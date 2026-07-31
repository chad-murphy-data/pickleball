# B4 (1-2) — leave-one-event-out fragility of every tail-bin statistic AND of the headline nulls

Every statistic below is recomputed from the committed data with the SAME joins/labels/bins the published version used (heuristic event_geo labels, match-hour wind where the published test used it). 'Δ estimate' is the change when that one event is deleted. Cluster/CI work lives in the spec-curve script; this script is about INFLUENCE.


## 1a. Design B — point-share swing z², windy 14+ tail bin (published 1.95, n=111)

### Design B mean z², OUTDOOR windy 14+

full = **+1.9494**   n = 111   events = 15   LOEO range [+1.8437, +2.0278]   biggest single event = 18% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | 2024 Lapiplasty Pickleball World Championships (2024-11-04, Brookhaven Country Club) [UNAUDITED — heuristic label] | 16 | -0.1056 | +1.8437 |
| 2 | PPA Tour: 2026 Carvana Mesa Cup (2026-02-16, Arizona Athletic Grounds) [audit: outdoor/medium] | 11 | +0.0784 | +2.0278 |
| 3 | PPA Tour: Veolia Texas Open (2026-03-09, The Courts McKinney Pickleball & Tennis Center) [audit: outdoor/high] | 20 | -0.0684 | +1.8810 |

*Audit composition of the Design B windy-14+ bin: unaudited 60 (54%), outdoor-high 20 (18%), indoor 19 (17%), outdoor-medium 12 (11%).*
*Keeping ONLY audited-outdoor games (n=32): +1.9231 (full +1.9494).*

### Design B Δ mean z², windy 14+ − outdoor calm (published +0.224 [−0.160, +0.585])
*an event can sit in BOTH arms (different match hours)*

full = **+0.2244**   n = 1747   events = 62   LOEO range [+0.1181, +0.2985]   biggest single event = 7% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | 2024 Lapiplasty Pickleball World Championships (2024-11-04, Brookhaven Country Club) [UNAUDITED — heuristic label] | 51 | -0.1063 | +0.1181 |
| 2 | PPA Tour: 2026 Carvana Mesa Cup (2026-02-16, Arizona Athletic Grounds) [audit: outdoor/medium] | 31 | +0.0741 | +0.2985 |
| 3 | PPA Tour: Veolia Texas Open (2026-03-09, The Courts McKinney Pickleball & Tennis Center) [audit: outdoor/high] | 34 | -0.0692 | +0.1552 |

## 1b. Design C — serve-rate swing z², windy 14+ (published 1.34, n=162 team-halves)

### Design C mean z², OUTDOOR windy 14+

full = **+1.3445**   n = 162   events = 15   LOEO range [+1.2858, +1.4089]   biggest single event = 18% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: Stratusphere Gin Virginia Beach Cup (2024-09-30, Pickleball Virginia Beach) [UNAUDITED — heuristic label] | 12 | +0.0644 | +1.4089 |
| 2 | PPA Tour: Veolia Texas Open (2026-03-09, The Courts McKinney Pickleball & Tennis Center) [audit: outdoor/high] | 29 | +0.0587 | +1.4032 |
| 3 | 2024 Lapiplasty Pickleball World Championships (2024-11-04, Brookhaven Country Club) [UNAUDITED — heuristic label] | 21 | -0.0587 | +1.2858 |

*Audit composition of the Design C windy-14+ bin: unaudited 87 (54%), outdoor-high 29 (18%), indoor 28 (17%), outdoor-medium 18 (11%).*
*Keeping ONLY audited-outdoor games (n=47): +1.1433 (full +1.3445).*

### Design C Δ mean z², windy 14+ − outdoor calm (published +0.190 [−0.067, +0.544])

full = **+0.1896**   n = 2821   events = 62   LOEO range [+0.1300, +0.2579]   biggest single event = 6% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: Stratusphere Gin Virginia Beach Cup (2024-09-30, Pickleball Virginia Beach) [UNAUDITED — heuristic label] | 39 | +0.0683 | +0.2579 |
| 2 | 2024 Lapiplasty Pickleball World Championships (2024-11-04, Brookhaven Country Club) [UNAUDITED — heuristic label] | 77 | -0.0596 | +0.1300 |
| 3 | PPA Tour: Veolia Texas Open (2026-03-09, The Courts McKinney Pickleball & Tennis Center) [audit: outdoor/high] | 53 | +0.0578 | +0.2474 |

## 1c. The continuous versions (published as near-nulls)

### Design B slope of z² on match-hour wind, outdoor (published +0.165 per +10 mph [−0.053, +0.364])

full = **+0.1652**   n = 2403   events = 63   LOEO range [+0.1197, +0.2017]   biggest single event = 7% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: CIBC Texas Open - 2025 (2025-03-12, Courts of McKinney) [UNAUDITED — heuristic label] | 55 | -0.0455 | +0.1197 |
| 2 | Edward Jones MLP Mid-Season Tournament (2025-07-09, Belknap Park) [audit: outdoor/high] | 129 | +0.0365 | +0.2017 |
| 3 | PPA Tour: Stratusphere Gin Virginia Beach Cup (2024-09-30, Pickleball Virginia Beach) [UNAUDITED — heuristic label] | 45 | +0.0331 | +0.1983 |

### Design C slope of z² on match-hour wind, outdoor (published +0.038 per +10 mph [−0.093, +0.166])

full = **+0.0382**   n = 3886   events = 63   LOEO range [+0.0115, +0.0643]   biggest single event = 6% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | 2024 Lapiplasty Pickleball World Championships (2024-11-04, Brookhaven Country Club) [UNAUDITED — heuristic label] | 104 | -0.0267 | +0.0115 |
| 2 | PPA Tour: Stratusphere Gin Virginia Beach Cup (2024-09-30, Pickleball Virginia Beach) [UNAUDITED — heuristic label] | 78 | +0.0261 | +0.0643 |
| 3 | 2025 Edward Jones MLP Cup (2025-10-31, Brookhaven Country Club) [audit: outdoor/medium] | 169 | +0.0202 | +0.0584 |

## 2. Favourite edge (obs − pred) tail bins, weather_report.py

### outdoor 20+ mph at match hour (published −0.047, n=82)

full = **-0.0466**   n = 82   events = 4   LOEO range [-0.0646, +0.0096]   biggest single event = 77% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | 2024 Lapiplasty Pickleball World Championships (2024-11-04, Brookhaven Country Club) [UNAUDITED — heuristic label] | 63 | +0.0562 | +0.0096 |
| 2 | PPA Tour: Fasenra Virginia Beach Cup presented by Joola (2025-10-06, Pickleball Virginia Beach) [UNAUDITED — heuristic label] | 9 | -0.0181 | -0.0646 |
| 3 | PPA Tour: CIBC Texas Open - 2025 (2025-03-12, Courts of McKinney) [UNAUDITED — heuristic label] | 8 | +0.0143 | -0.0323 |

*Audit composition of this bin: unaudited 80 (98%), outdoor-high 2 (2%).*
*Audited-outdoor games alone: only 2 — the bin cannot be re-estimated on verified-outdoor play.*

### outdoor 14–20 mph at match hour — the '−6.0pp' result (published −0.060, n=1151)

full = **-0.0599**   n = 1151   events = 21   LOEO range [-0.0655, -0.0564]   biggest single event = 12% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: 2026 Carvana Mesa Cup (2026-02-16, Arizona Athletic Grounds) [audit: outdoor/medium] | 110 | -0.0056 | -0.0655 |
| 2 | PPA Tour NZ: Pickleball Cup (2025-04-24, TotalCoach Tennis Centre) [audit: unknown/low] | 77 | -0.0039 | -0.0638 |
| 3 | 2026 Opelika PPA Challenger (2026-04-18, Opelika Sportsplex) [audit: outdoor/medium] | 113 | +0.0035 | -0.0564 |

*Audit composition of this bin: unaudited 433 (38%), indoor 255 (22%), outdoor-medium 248 (22%), outdoor-high 138 (12%), unknown 77 (7%).*
*Keeping ONLY audited-outdoor games (n=386): -0.0540 (full -0.0599).*

### outdoor 92+ °F at match hour (published −0.049, n=644)

full = **-0.0491**   n = 644   events = 6   LOEO range [-0.0540, -0.0470]   biggest single event = 62% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | Punta Gorda PPA Challenger (2025-05-24, PicklePlex) [UNAUDITED — heuristic label] | 22 | -0.0049 | -0.0540 |
| 2 | PPA Tour Asia, Panas Malaysia Open 2025 (2025-07-03, 9Pickle) [audit: indoor/medium] | 19 | +0.0021 | -0.0470 |
| 3 | PPA Tour: Guaranteed Rate Las Vegas Open (2024-10-09, Darling Tennis Center) [UNAUDITED — heuristic label] | 93 | +0.0020 | -0.0471 |

*Audit composition of this bin: indoor 529 (82%), unaudited 115 (18%).*
*Audited-outdoor games alone: only 0 — the bin cannot be re-estimated on verified-outdoor play.*

### outdoor 92+ °F daily max (published −0.055, n=1755)

full = **-0.0552**   n = 1755   events = 7   LOEO range [-0.0603, -0.0510]   biggest single event = 27% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | Punta Gorda PPA Challenger (2025-05-24, PicklePlex) [UNAUDITED — heuristic label] | 190 | -0.0051 | -0.0603 |
| 2 | 2026 Macon PPA Challenger (2026-07-18, Rhythm & Rally Sports & Events) [audit: indoor/high] | 230 | +0.0042 | -0.0510 |
| 3 | Boise PPA Challenger (2025-05-31, The Flying Pickle) [audit: indoor/high] | 145 | +0.0026 | -0.0526 |

*Audit composition of this bin: indoor 1089 (62%), unaudited 666 (38%).*
*Audited-outdoor games alone: only 0 — the bin cannot be re-estimated on verified-outdoor play.*

### outdoor 20+ mph daily max (published −0.059, n=313)

full = **-0.0594**   n = 313   events = 4   LOEO range [-0.0641, -0.0527]   biggest single event = 50% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | 2024 Lapiplasty Pickleball World Championships (2024-11-04, Brookhaven Country Club) [UNAUDITED — heuristic label] | 132 | +0.0067 | -0.0527 |
| 2 | PPA Tour: Fasenra Virginia Beach Cup presented by Joola (2025-10-06, Pickleball Virginia Beach) [UNAUDITED — heuristic label] | 9 | -0.0047 | -0.0641 |
| 3 | PPA Tour: CIBC Texas Open - 2025 (2025-03-12, Courts of McKinney) [UNAUDITED — heuristic label] | 155 | +0.0034 | -0.0560 |

*Audit composition of this bin: unaudited 296 (95%), outdoor-high 17 (5%).*
*Audited-outdoor games alone: only 17 — the bin cannot be re-estimated on verified-outdoor play.*

### INDOOR control 20+ mph at match hour (published −0.113, n=222)

full = **-0.1134**   n = 222   events = 2   LOEO range [-0.1159, -0.0918]   biggest single event = 90% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: Picklr Utah Open (2024-08-21, Salt Palace Convention Center) [audit: indoor/high] | 199 | +0.0216 | -0.0918 |
| 2 | PPA Tour: CIBC Atlanta Slam (2024-09-09, Life Time Peachtree Corners) [audit: mixed/high] | 23 | -0.0025 | -0.1159 |

*Audit composition of this bin: indoor 199 (90%), mixed 23 (10%).*
*Audited-outdoor games alone: only 0 — the bin cannot be re-estimated on verified-outdoor play.*

### outdoor 0–8 mph at match hour — the calm REFERENCE (published −0.040, n=17139)

full = **-0.0404**   n = 17139   events = 65   LOEO range [-0.0415, -0.0392]   biggest single event = 3% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | 2026 Macon PPA Challenger (2026-07-18, Rhythm & Rally Sports & Events) [audit: indoor/high] | 319 | +0.0012 | -0.0392 |
| 2 | PPA Tour: Rate Vegas Cup (2025-10-20, Darling Tennis Center) [UNAUDITED — heuristic label] | 436 | +0.0012 | -0.0392 |
| 3 | 2026 Wilson PPA Challenger (2026-05-23, Wilson Pickleball Facility) [UNAUDITED — heuristic label] | 411 | -0.0011 | -0.0415 |

*Audit composition of this bin: unaudited 9685 (57%), outdoor-medium 2243 (13%), indoor 2110 (12%), outdoor-high 1334 (8%), unknown 1055 (6%), mixed 712 (4%).*
*Keeping ONLY audited-outdoor games (n=3577): -0.0483 (full -0.0404).*

### 2b. The CONTRAST that the claim rests on: tail bin − calm bin

### outdoor 14–20 mph − outdoor calm <8 (match hour)

full = **-0.0195**   n = 18290   events = 66   LOEO range [-0.0245, -0.0156]   biggest single event = 3% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: 2026 Carvana Mesa Cup (2026-02-16, Arizona Athletic Grounds) [audit: outdoor/medium] | 360 | -0.0050 | -0.0245 |
| 2 | PPA Tour NZ: Pickleball Cup (2025-04-24, TotalCoach Tennis Centre) [audit: unknown/low] | 77 | -0.0039 | -0.0234 |
| 3 | 2026 Opelika PPA Challenger (2026-04-18, Opelika Sportsplex) [audit: outdoor/medium] | 365 | +0.0038 | -0.0156 |

*Audit composition of this contrast: unaudited 10118 (55%), outdoor-medium 2491 (14%), indoor 2365 (13%), outdoor-high 1472 (8%), unknown 1132 (6%), mixed 712 (4%).*
*Keeping ONLY audited-outdoor games (386 treated / 3577 reference): -0.0057 (full -0.0195).*

### outdoor 20+ mph − outdoor calm <8 (match hour)

full = **-0.0062**   n = 17221   events = 65   LOEO range [-0.0240, +0.0496]   biggest single event = 3% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | 2024 Lapiplasty Pickleball World Championships (2024-11-04, Brookhaven Country Club) [UNAUDITED — heuristic label] | 334 | +0.0558 | +0.0496 |
| 2 | PPA Tour: Fasenra Virginia Beach Cup presented by Joola (2025-10-06, Pickleball Virginia Beach) [UNAUDITED — heuristic label] | 98 | -0.0178 | -0.0240 |
| 3 | PPA Tour: CIBC Texas Open - 2025 (2025-03-12, Courts of McKinney) [UNAUDITED — heuristic label] | 57 | +0.0143 | +0.0081 |

*Audit composition of this contrast: unaudited 9765 (57%), outdoor-medium 2243 (13%), indoor 2110 (12%), outdoor-high 1336 (8%), unknown 1055 (6%), mixed 712 (4%).*
*Audited-outdoor only: 2 treated / 3577 reference games — too thin to re-estimate the contrast on verified-outdoor play.*

### outdoor 92+ °F − outdoor <70 °F (match hour)

full = **-0.0098**   n = 10267   events = 56   LOEO range [-0.0147, -0.0078]   biggest single event = 5% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | Punta Gorda PPA Challenger (2025-05-24, PicklePlex) [UNAUDITED — heuristic label] | 22 | -0.0049 | -0.0147 |
| 2 | PPA Tour Asia, Panas Malaysia Open 2025 (2025-07-03, 9Pickle) [audit: indoor/medium] | 19 | +0.0021 | -0.0078 |
| 3 | PPA Tour: Guaranteed Rate Las Vegas Open (2024-10-09, Darling Tennis Center) [UNAUDITED — heuristic label] | 93 | +0.0020 | -0.0079 |

*Audit composition of this contrast: unaudited 4854 (47%), outdoor-medium 2089 (20%), indoor 1989 (19%), mixed 659 (6%), unknown 390 (4%), outdoor-high 286 (3%).*
*Audited-outdoor only: 0 treated / 2375 reference games — too thin to re-estimate the contrast on verified-outdoor play.*

## 3. favorites_wind regression 1 — is the NULL itself fragile?

### outdoor: d (skill×wind per +10 mph), published +0.002 [−0.060, +0.064]

full = **+0.0017**   n = 24819   events = 67   LOEO range [-0.0059, +0.0167]   biggest single event = 3% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: North Carolina Cup (2024-04-01, Cary Tennis Park) [UNAUDITED — heuristic label] | 356 | +0.0151 | +0.0167 |
| 2 | PPA Tour: CIBC Texas Open - 2025 (2025-03-12, Courts of McKinney) [UNAUDITED — heuristic label] | 506 | +0.0135 | +0.0152 |
| 3 | PPA Tour NZ: Pickleball Cup (2025-04-24, TotalCoach Tennis Centre) [audit: unknown/low] | 298 | -0.0075 | -0.0059 |

### outdoor: d/b = fraction of the favourite's edge erased per +10 mph

full = **+0.0016**   n = 24819   events = 67   LOEO range [-0.0056, +0.0162]   biggest single event = 3% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: North Carolina Cup (2024-04-01, Cary Tennis Park) [UNAUDITED — heuristic label] | 356 | +0.0146 | +0.0162 |
| 2 | PPA Tour: CIBC Texas Open - 2025 (2025-03-12, Courts of McKinney) [UNAUDITED — heuristic label] | 506 | +0.0131 | +0.0147 |
| 3 | PPA Tour NZ: Pickleball Cup (2025-04-24, TotalCoach Tennis Centre) [audit: unknown/low] | 298 | -0.0072 | -0.0056 |

### indoor: d (skill×wind per +10 mph), published −0.080 [−0.150, +0.020]

full = **-0.0805**   n = 11699   events = 43   LOEO range [-0.0960, -0.0410]   biggest single event = 6% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: Picklr Utah Open (2024-08-21, Salt Palace Convention Center) [audit: indoor/high] | 496 | +0.0395 | -0.0410 |
| 2 | PPA Tour: Veolia Cape Coral Open (2025-03-05, Lake Kennedy Racquet Center) [audit: outdoor/high] | 501 | -0.0155 | -0.0960 |
| 3 | PPA Tour: Indoor National Championships (2026-01-19, Lifetime Lakeville) [audit: indoor/high] | 392 | -0.0147 | -0.0951 |

### indoor: d/b = fraction of the favourite's edge erased per +10 mph

full = **-0.0726**   n = 11699   events = 43   LOEO range [-0.0854, -0.0378]   biggest single event = 6% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: Picklr Utah Open (2024-08-21, Salt Palace Convention Center) [audit: indoor/high] | 496 | +0.0348 | -0.0378 |
| 2 | PPA Tour: Veolia Cape Coral Open (2025-03-05, Lake Kennedy Racquet Center) [audit: outdoor/high] | 501 | -0.0128 | -0.0854 |
| 3 | PPA Tour: Indoor National Championships (2026-01-19, Lifetime Lakeville) [audit: indoor/high] | 392 | -0.0126 | -0.0852 |

### 3b. Leave-one-TOUR-out and leave-one-YEAR-out, outdoor d

| dropped | n | d | d/b |
|---|---|---|---|
| — (full) | 24819 | +0.0017 | +0.0016 |
| tour=MLP | 24263 | +0.0002 | +0.0002 |
| tour=PPA | 556 | +0.1287 | +0.1231 |
| year=2024 | 17733 | +0.0280 | +0.0263 |
| year=2025 | 13907 | -0.0151 | -0.0147 |
| year=2026 | 17998 | -0.0003 | -0.0003 |

| kept ONLY | n | d | d/b |
|---|---|---|---|
| tour=MLP | 556 | +0.1287 | +0.1231 |
| tour=PPA | 24263 | +0.0002 | +0.0002 |
| year=2024 | 7086 | -0.0088 | -0.0092 |
| year=2025 | 10912 | +0.0179 | +0.0169 |
| year=2026 | 6821 | +0.0771 | +0.0725 |

## 4. H1 serve-point-rate slope vs wind, outdoor (published null)

### serve rate per +10 mph, match-hour wind (published +0.0030 [−0.0009, +0.0072])

full = **+0.0030**   n = 7905   events = 63   LOEO range [+0.0020, +0.0038]   biggest single event = 3% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: North Carolina Cup (2024-04-01, Cary Tennis Park) [UNAUDITED — heuristic label] | 127 | -0.0009 | +0.0020 |
| 2 | PPA Tour: Veolia Texas Open (2026-03-09, The Courts McKinney Pickleball & Tennis Center) [audit: outdoor/high] | 197 | +0.0009 | +0.0038 |
| 3 | PPA Tour: Masters (2025-01-06, Mission Hills Country Club) [UNAUDITED — heuristic label] | 181 | -0.0007 | +0.0022 |

### serve rate per +10 mph, daily max wind (published +0.0017 [−0.0024, +0.0061])

full = **+0.0017**   n = 7922   events = 63   LOEO range [+0.0008, +0.0029]   biggest single event = 3% of the sample

| rank | event dropped | n in stat | Δ estimate | estimate w/o it |
|---|---|---|---|---|
| 1 | PPA Tour: Veolia Texas Open (2026-03-09, The Courts McKinney Pickleball & Tennis Center) [audit: outdoor/high] | 197 | +0.0012 | +0.0029 |
| 2 | PPA Tour: North Carolina Cup (2024-04-01, Cary Tennis Park) [UNAUDITED — heuristic label] | 127 | -0.0009 | +0.0008 |
| 3 | PPA Tour: Veolia Lakeland Open (2025-11-18, Beerman Family Tennis Center) [UNAUDITED — heuristic label] | 194 | +0.0009 | +0.0026 |
