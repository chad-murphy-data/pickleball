# Pin realignment verdict (2026-08-16, ultracode workflow + parity tiebreak)

**Mapping: IDENTITY. Rows 1-16 of shot_labels pin and describe rallies
1-16.** One anomaly: **row 3's pin (91.19 s) marks a full-speed
broadcast REPLAY of rally 3**, not its live serve. Rally 3 aired live
~56-79 s (scorebug 0-0, Tuionetoa serving — the match's first point,
the classic full re-air candidate); the replay carries the updated
UTAH 1 - CHICAGO 0 bug, which is exactly what the user's screenshot
showed and what triggered the scare. Rally 17 has a 2-shot label stub
and no pin; the stray rally-18 mark (91.11 s) is junk from the broken
seek and proves the old tool dumped multiple rows at the ~91 s region.

## How it was decided (two independent solvers split; evidence broke it)

- Solver A (exhaustive DP): +1 shift on rows 3-8, row 9 = replay pin.
- Solver B (signature-anchored): identity, row 3 = replay pin.
- Adversarial verifier sided with A, but its refutation of the replay
  reading conflated REAL time with BROADCAST time (a replay inserts
  footage; a condensed VOD keeps it) — discounted.
- **Tiebreak, non-circular, from the user's own typed hitters**: teams
  strictly alternate shots, so odd shots ≥3 belong to the serving team.
  Every disputed row is parity-PERFECT one way: row 6 = 5/5 Utah-served
  — impossible under A (its rally 7 is Chicago-served), exact under B.
  Rows 3-5, 7-9 parity-consistent with identity throughout.
- Corroboration: A itself called row 5 -> rally 6 "the one strained
  fit" (14 shots cannot fit rally 6's 19 s duration); B's unconstrained
  optimum was identity (cost 82.4, all residual gaps explained by
  documented replay/filler segments); gap arithmetic pin2->pin3
  (50.55 s) = rally 2 play + rally 3 LIVE + transitions + replay start.

Confidence: high. The 30-second full confirmation, if ever wanted: the
segment at ~91 s should be the SAME rally as ~56-79 s (with a replay
wipe/graphic at its start). Not required for labeling — the scorebug
protocol routes around the replay regardless.

## Consequences

- **Prefills are correctly attached** (row k describes rally k). The
  "prefill demoted" protocol stays as a safety net, but ⏎-flow is
  legitimate again wherever the screen tracks it.
- **Pins are a valid jitter reference again** for rows 1-2 and 4-16
  (15 pins); rally 3's pin is excluded (replay, not the live serve).
- The parity check doubles as a label-quality result: the typed
  sequences alternate perfectly in every disputed row, so pop-era
  coding damage is bounded to shot TYPES and occasional count slips,
  not structure.
- **New protocol rule**: at a seek, if the bug shows a LATER score than
  the header, you are on a REPLAY — the live serve is EARLIER; scrub
  backward to the serve whose bug matches the header. (Replays show the
  replayed rally's END state.)
- Rally 3 labels at its live airing: serve ≈ 56-60 s, bug 0-0, Utah
  serving. The tool now seeks there.

## Addendum 2026-09-02 — the "replay at ~91 s" call is WRONG on the video time

Frames pulled straight from the VOD (imageio-ffmpeg, no browser) while
staging the r2–r5 ball tools:

| t (s) | scorebug | what is on screen |
|---|---|---|
| 40.6 | 0-0 | Jones serving, near-right court → rally 2 serve |
| 59.5 | 0-0 | Tuionetoa tossing to serve, near-LEFT court → rally 3 LIVE serve |
| 76.0 | 0-0 | rally 3 still in play (Utah at the kitchen) |
| 91.3 | **1-0** | Tuionetoa serving, near-RIGHT court → **rally 4 live serve** |
| 103.9 | 1-0 | rally 4 in play at the kitchen (ends ~103.7, Wei into the net) |
| 118.9 | 2-0 | Tuionetoa serving, near-left → rally 5 serve |
| 132.6 | 2-0 | rally 5 dinking at the kitchen |
| 146.4 | 3-0 | serve → rally 6 (matches the manual r6 serve 146.34) |

A replay of rally 3 at 91 s would repeat the 59.5 s picture (same court
side, same server position); it does not, and the duration arithmetic
never allowed it (rally 4 = 25 s of log between 91 and the rally-5 serve
at 118.8). So the 91.19 s pin is rally 4's live serve, and the v4
window rows are one rally LATE from row 3 onward (row 3 = rally 4,
row 4 = rally 5, row 5 = rally 6, …). The prefill `t_tap_s` values for
r2–r5 sit on the right rallies (serves at 40.54 / 59.43 / 91.16 /
118.78). The row-content mapping (identity) stands; only the pin-time
reading changes. Consumers: the contact tool's per-row pin seek is
shifted for rows ≥ 3 (seek by time instead: 40.5 / 59.4 / 91.2 /
118.8); `windows_from_labels` in pose_extract is unaffected (it reads
the contact rows, which are right).
