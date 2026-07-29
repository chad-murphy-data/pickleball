# model/weather_review/ — second-pass audit of the weather thread

Working scripts from the neutral re-review of the 2026-07-28 weather
session (the "six hypotheses, six nulls" result). They are **additions,
not edits**: the committed `model/weather_report.py`,
`model/favorites_wind.py`, `model/end_effects.py` and
`model/wind_skill.py` are deliberately left untouched so the published
numbers stay reproducible for comparison.

Each script is standalone, stdlib/numpy, deterministic (seeded), and
reads only committed data. Naming: `a7_*` reproduction checks, `heat_*`
the continuous heat tests, `label_arms_*` re-runs under the web-verified
venue labels in `data/venue_overrides.csv`, and so on.

Findings are collected in `model/weather_review_interim.md` (phase 1) and
the final review document; this directory is the receipts.
