# Weather report — first pass (day-level join)

Joined rows: 12902 matches with rally logs, 36608 games with full v2 ratings.


## OUTDOOR

### A. Serve-point rate vs daily max wind (7922 matches)
| wind | matches | rallies | serve-point rate |
|---|---|---|---|
| 0–8 mph | 2503 | 204965 | 0.4476 |
| 8–14 mph | 4105 | 336012 | 0.4500 |
| 14–20 mph | 1193 | 100275 | 0.4490 |
| 20+ mph | 121 | 10847 | 0.4540 |

WLS slope: per +10 mph: +0.0017 (95% cluster-bootstrap CI [-0.0024, +0.0061])

### B. Favorites vs wind (daily max) (24909 games)
| wind | games | predicted fav % | observed fav % | edge (obs−pred) | Brier |
|---|---|---|---|---|---|
| 0–8 mph | 7550 | 0.815 | 0.775 | -0.040 [-0.051,-0.029] | 0.1553 |
| 8–14 mph | 13212 | 0.814 | 0.774 | -0.040 [-0.049,-0.031] | 0.1581 |
| 14–20 mph | 3834 | 0.815 | 0.769 | -0.046 [-0.058,-0.032] | 0.1598 |
| 20+ mph | 313 | 0.801 | 0.741 | -0.059 [-0.067,+0.008] | 0.1678 |

### C. Favorites vs heat (daily max) (24909 games)
| tmax | games | predicted fav % | observed fav % | edge (obs−pred) | Brier |
|---|---|---|---|---|---|
| 0–70 °F | 5581 | 0.816 | 0.768 | -0.048 [-0.062,-0.035] | 0.1620 |
| 70–82 °F | 10905 | 0.814 | 0.775 | -0.039 [-0.048,-0.030] | 0.1558 |
| 82–92 °F | 6668 | 0.814 | 0.779 | -0.036 [-0.047,-0.024] | 0.1538 |
| 92+ °F | 1755 | 0.811 | 0.756 | -0.055 [-0.075,-0.037] | 0.1700 |

## INDOOR  *(control arm — no direct wind exposure expected; heat/HVAC effects still possible)*

### A. Serve-point rate vs daily max wind (4980 matches)
| wind | matches | rallies | serve-point rate |
|---|---|---|---|
| 0–8 mph | 1306 | 95999 | 0.4483 |
| 8–14 mph | 3048 | 211281 | 0.4489 |
| 14–20 mph | 493 | 36811 | 0.4482 |
| 20+ mph | 133 | 12219 | 0.4459 |

WLS slope: per +10 mph: +0.0005 (95% cluster-bootstrap CI [-0.0052, +0.0059])

### B. Favorites vs wind (daily max) (11699 games)
| wind | games | predicted fav % | observed fav % | edge (obs−pred) | Brier |
|---|---|---|---|---|---|
| 0–8 mph | 2827 | 0.818 | 0.773 | -0.045 [-0.064,-0.025] | 0.1544 |
| 8–14 mph | 6921 | 0.815 | 0.773 | -0.042 [-0.052,-0.032] | 0.1579 |
| 14–20 mph | 1617 | 0.806 | 0.774 | -0.032 [-0.066,-0.002] | 0.1604 |
| 20+ mph | 334 | 0.803 | 0.716 | -0.088 [-0.147,-0.048] | 0.1830 |

### C. Favorites vs heat (daily max) (11699 games)
| tmax | games | predicted fav % | observed fav % | edge (obs−pred) | Brier |
|---|---|---|---|---|---|
| 0–70 °F | 2757 | 0.817 | 0.776 | -0.041 [-0.059,-0.019] | 0.1551 |
| 70–82 °F | 4908 | 0.814 | 0.769 | -0.046 [-0.064,-0.030] | 0.1599 |
| 82–92 °F | 2770 | 0.807 | 0.764 | -0.043 [-0.061,-0.028] | 0.1616 |
| 92+ °F | 1226 | 0.823 | 0.789 | -0.035 [-0.055,-0.015] | 0.1499 |

---
*Caveats: day-level weather (attenuates), v2 current-form values applied retroactively (fine for interactions, not levels), indoor/outdoor labels are heuristic (see scraper/weather.py). Hourly join is the designed next step.*
