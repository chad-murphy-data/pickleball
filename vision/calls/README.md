# Locked call sheets

One row per call: `window,rank,cell` (cell 1-based row-major, rank 1 =
most confident). Scored by `vision/score_localization.py` against the
matching `ANSWER_KEY_LOC.csv`, which stays on the user's machine until
the calls here are committed.

METHOD, identical across arms and stated because it constrains the
result: ONE read of the delivered grid per window. No crops, no
upscaling, no second pass. Crop-assisted scoring is ~7 images per
window (~$85/match), which is strictly dominated by just running plain
3x3 at $44 — so a crop-assisted number would measure a configuration
nobody would ever buy.

CALL RULE: a contact is called only on STRIKE EVIDENCE — arm/paddle
driven forward into a hitting posture, or ball visibly adjacent to a
paddle. On marked arms a magenta tracker mark near a player is NOT
sufficient on its own.

| arm    | packing | windows | cells | calls | rate  | status |
|--------|---------|---------|-------|-------|-------|--------|
| arm6m  | 6x6 mkd | 9       | 324   | 75    | 23.1% | SCORED, key seen — spent |
| arm5   | 5x5     | 13      | 325   | 65    | 20.0% | locked, awaiting key |
| arm6m2 | 6x6 mkd | 9       | 324   | 45    | 13.9% | locked, awaiting key |

arm6 (6x6 plain) was DISCARDED UNOPENED: its nine windows are the same
nine windows as arm6m, whose key has been read, so scoring it would
have been scoring with the answers in hand. There is currently no
clean plain-6x6 control.

Call rate fell 23.1% -> 13.9% at the same packing between arm6m and
arm6m2. That is the new scoring rule working as intended: under a
call-count-indexed null every extra call raises the bar it is measured
against, so over-calling stopped being free.
