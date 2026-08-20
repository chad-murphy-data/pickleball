# arm6m — marked 6x6, calls locked 2026-08-20

Nine windows, 36 cells each at 0.15 s, span 5.25 s. Cells numbered 1-36
row-major. Sides: N = near/black (bottom), F = far (top).

METHOD, stated because it constrains the result: ONE read of the
delivered grid per window. No crops, no upscaling. A row-crop helper was
written and deleted unviewed — crop-assisted scoring is 7 images per
window (~$85/match), which is strictly dominated by just running plain
3x3 at $44, so a crop-assisted number would measure a configuration
nobody would ever buy.

CALL RULE: a contact is called only on STRIKE EVIDENCE — arm/paddle
extended into a hitting posture, or ball visibly adjacent to a paddle.
A magenta mark near a player is NOT sufficient. Stated before scoring
because the liberal mark-driven read gives ~91 calls across the nine
windows (one per 0.52 s), which is faster than real dinking and is
exactly falsifier 3 — believing a 44%-precision tracker.

  w01  3N  6F  10F  13F  18F  21N  25N  28F  31N          (9)
  w02  22N 25N 27N  30F  32N                              (5)   c1-c20 EMPTY
  w03  9N  13N 16N  19N  21F  24N  27F  30N  33F  35N    (10)
  w04  5F  8N  11N  16F  20N  22F  24N  27F  31F  35N    (10)
  w05  3N  6N  8N   11F  13N  16N  20F  21N  24F  28F    (10)
  w06  34N                                                (1)   c1-c30 EMPTY
  w07  8N  10F 12N  14F  18N  20F  24N  27F  30N  33F    (10)
  w08  2N  6F  8F   11F  14F  17F  23N  26N  30N  33N    (10)
  w09  2F  3N  8F   12F  16N  19F  23N  25N  28N  31N    (10)

TOTAL 75 calls across 9 windows.

TWO WINDOWS ARE MOSTLY DEAD TIME and I am calling them that way. w06 is
a server standing with the ball and a line judge in frame for 30 of 36
cells; w02 is ~20 cells of players milling between points. This is the
sampler working as written, not a bug: sample_windows draws t0 uniformly
over [first_contact - PRE_PAD, last_contact] with PRE_PAD = 4.0 s fixed
at every rung. Expected dead fraction is 8/((S+4)*span) — about 11% for
a 10 s rally, 22% for a 3 s one — but the tail is what bites: t0 within
1 s of the pad's start gives a ~75%-dead window, at probability
1/(S+4) ~ 8%, so 9 windows should throw about one. Two showed up.

CONSEQUENCE: the realized contact count is below the ~55 the arm was
sized for, so this arm is under-powered relative to plan and its error
bar is wider than the +/-7pp the sizing assumed. Not a reason to redraw
- redrawing after seeing the images would be selection - but the number
should be read with that in mind. If the ladder is run again, PRE_PAD
should scale with the rung rather than sitting fixed at 4.0 s.

REGISTERED, for the record (swing_explore_notes.md, before any grid
existed): recall 70-85%; falsifier 2 = arm6m <= arm6 + 5pp; falsifier 3
= arm6m precision below arm6 precision; miss decomposition 60/40
tracker-bound over reading-bound.
