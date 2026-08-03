# Rally duration — the first DV outside the win/loss closure

*Session 2026-08-03. Sample: 400 matches, 21,614 clean rallies (100% log
fetch success). NOT the full archive — a first look.*

## Why this is different from everything else in the clutch thread

Every prior DV in this project is a function of **who won each rally**:
points, margins, win probability, hazards, hit rates. Skill is *defined* as
what predicts winning, so any second factor competes for residual variance in
a quantity the skill term was fitted to explain. That is why the zero-clutch
simulation (`clutch_endogeneity.py`) could reproduce every clutch index built
today — it generates the binary sequence, and the binary sequence *was* the
data.

Duration sits outside that closure. The simulation cannot reproduce it,
because it never modelled time.

## Where the data comes from

Referee logs carry `date_created` per entry, and `log_type 12` marks *rally
underway*, so the gap to the resolving entry (14 point / 16 side-out /
23 second server) is a per-rally duration. Validated against the explicit
`point_log` payload (`time_started` / `time_ended`), which matches exactly —
but only appears on point rows, so the type-12 gap is what covers all rallies.

Rallies within two log entries of a `timeout_log`, `line_review_log` or
`penalty_log` are dropped; those produced the 3-minute outliers. Kept range
2–90s. **Not in `pb_rally`** — this needs raw logs, so archive-scale work
means a backfill.

Overall: median 17.0s, mean 18.3s, sd 8.1.

## Big points are played differently

| score state | mean |
|---|---|
| early (both ≤ 4) | 18.0s |
| middle | 18.5s |
| someone ≥ 9 | 18.7s |
| **9+ and close** | **19.6s** |
| 9+ but blowout | 17.4s |

Confounded by matchup (close games select evenly-matched pairs, who rally
longer), so the test is **within match** — same four players, big-and-close
rallies against their own early rallies:

| contrast | effect | 95% CI |
|---|---|---|
| **big-and-close − early** | **+1.74s** | **[+1.24, +2.22]** |
| late-but-**decided** − early (control) | −0.27s | [−0.77, +0.26] |

72% of matches show it. The control is the load-bearing half: general
late-game slowdown or fatigue would move the decided arm too. It doesn't.

## No choke signature — and this is the interesting part

Duration separates the two ways to lose a rally: fast = an error, slow = a
battle you were outplayed in. It is the closest thing to an unforced-error
measure available without shot data.

Relative to each match's own average:

| situation | outcome | n | vs match avg |
|---|---|---|---|
| serving for the game | lost | 847 | −0.41s |
| serving for the game | won | 611 | +0.63s |
| staving off game point | lost | 879 | −0.24s |
| staving off game point | won | 466 | **+0.98s** [+0.18, +1.77] |
| ordinary early | lost | 7,518 | −0.38s |
| ordinary early | won | 5,619 | +0.04s |

**Rallies lost under pressure take the same time as ordinary lost rallies**
(−0.03s and +0.14s difference). Players do not tighten up and miss quickly.

The whole effect is in rallies **won**: +0.59s and +0.94s versus ordinary won
rallies. Pressure does not produce errors; it makes points **harder to
finish**. That also explains the +1.74s above — it is won-rally lengthening,
not a general slowdown.

## Limits

* Only *staving off / won* clears zero on its own; the other cells span it.
* **Prep vs rally is unresolved.** Median 17s is long for a pro rally
  (typically ~10s), so the type-12 marker probably fires at point start, not
  serve strike — meaning this may bundle serve preparation. "Players take
  longer *before* big points" and "big-point rallies are longer" are
  different claims and this cannot yet separate them. Resolvable with data
  that exists.
* Timestamps are referee taps, not true rally boundaries. Latency under
  pressure is a rival explanation, partly cancelled since both endpoints
  would shift.
* 400 matches, not the archive.

## Next

Separate prep from rally (compare the resolution→next-type-12 gap, which is
pure between-point time, against the type-12→resolution gap). If the effect
lives in the between-point gap it is a pacing/routine finding; if in the
rally itself it is a play-style finding. Then per-player contrasts — but
those need the within-match design, since who reaches pressure states is
skill-determined.
