# Ball-click label picks (2026-09-02) — for the owner to click

Purpose: 2–4 more rallies with **ball V/S clicks** (the `ball_path_r{N}.csv`
kind, per `labeling_protocol.md`), chosen for FAST EXCHANGES, because
that is where the tracker loses the ball (owner overlay observations +
events autopsy: the misses are arrive/depart pairs around a lost gap next
to a player). Contact labels already exist for these rallies (typed
serve/return/slow/fast/lunge/whiff with times), so the click job is only
the ball, and the fast shots are already timestamped to click around.

Rules that shaped the pick (unchanged): r6/r7 are the only TRAIN rallies
today; r9/r10 are spent evaluation; **r20 is the reserved next seal**;
rallies 22–30 are temporal-gate HOLDOUT and stay untouched. Only the
`train` rows of `data/vision/label_split.csv` are eligible. Half train,
half fresh seals, so the model can learn from fast exchanges AND be
graded on unseen ones.

| pick | rally (game 1 womens) | role | why | contacts (labels) | dur |
|---|---|---|---|---|---|
| 1 | **r17** | TRAIN (joins r6/r7) | densest fast rally in the train split: 7 fast + 6 slow + 2 whiffs on 17 contacts; a proper hands battle | 17 | 23 s |
| 2 | **r21** | FRESH SEAL | short, 2 fast + 1 lunge on 6 contacts: a quick kill after the return, the exact "fast short flight" shape the hand-off pass targets | 6 | 13 s |
| 3 | r4 | TRAIN | 8 dinks then 2 speed-ups + 4 counters on 16 contacts (old-era types, times valid): slow→fast transition inside one rally | 16 | — |
| 4 | r20 | SEAL (already reserved) | 16 slow + 2 fast on 20 contacts, 44 s: the long-dink control case; grade only, never tune | 20 | 44 s |

Do 1 and 2 first (one train, one seal). 3 and 4 if the appetite holds.
r11 and r12 are unlabeled at every level (no contacts either) and would
cost a contact pass first — skip unless a fresh unseen-everything rally
is wanted. Do not click any of r22–r30.

Before clicking, stage the inputs the way r9/r10 were staged (HANDOFF.md
regeneration recipe): `r{N}_clip.mp4` cut from the same VOD at the
rally window, pose npz `r00{N}.npz`, then `c3_lab` cache, emission
p-cache, `cands_*`. The audit tool's chained seek reads the contact
times, so the fast shots are one keypress away.

What a new train rally buys: the emission scorer refits with r17 in the
fold (cross-fold caches `_x` go three-way), the hand-off grid re-tunes
on r6+r7+r17 with the selection rule as written in handoff_gate.md, and
r21 becomes a seal alongside r20. Nothing about r9/r10 changes.
