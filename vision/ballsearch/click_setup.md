# Click package (2026-09-02) — everything staged for the owner's ball clicks

Owner ask: "set up the stuff I need to do to click those rallies …
don't give me the minimum amount." This is the full package: nine
rallies in three sittings, ordered so every sitting is useful on its
own. Everything the machine can do is done and committed; what is left
is the clicking (and one contact pass for sitting 3).

Rules that shaped it (unchanged, see HANDOFF.md): TRAIN = r6 + r7
today; r9 / r10 are spent evaluation; **r20 is the reserved next seal,
r21 becomes the second seal**; rallies 22–34 are temporal-gate HOLDOUT
and stay untouched at every level. Positives come ONLY from your V
clicks; S clicks are ignore-zones; nothing is ever trained on tracker
output. No graded run on r20 / r21 happens without your explicit go.

## What is ready right now (no prerequisites)

Five rallies already have your manual contact taps, so their click
tools are built and committed — open the HTML, pick the video, click.

| sitting | rally | role after clicking | contacts (fast) | frames to click (30 fps) | why |
|---|---|---|---|---|---|
| 1 | **r17** ✅ clicked 2026-09-02 | TRAIN (joins r6 / r7) | 15 (7 fast) | 448 | the densest fast rally in the train split: the hands battle the tracker keeps losing. Delivered: 403 frames (through 440.7 s, the ball dead), 269 V / 110 S / 24 I / 0 N — streak share 27 %, the highest of any clicked rally |
| 1 | **r21** | SEAL #2 (grade only) | 6 (2 fast) | 228 | quick kill after the return — the fast-short-flight shape the hand-off pass targets |
| 1 | r18 | TRAIN | 3 (1 fast) | 186 | serve, return, one fast put-away: a free three-contact sample |
| 1 | r19 | TRAIN | 2 | 133 | serve + return only (a missed return): two flights, nearly free |
| 2 | **r20** | SEAL #1 (reserved) | 20 (2 fast) | 670 | the long dink-control case; the tracker's easy regime, graded not tuned |

Sitting 1 ≈ 1,000 frames (r9 was ~870), sitting 2 ≈ 670.

Tools (committed, self-contained; the video never leaves your Mac):

    data/vision/ball_audit_r17.html
    data/vision/ball_audit_r21.html
    data/vision/ball_audit_r18.html
    data/vision/ball_audit_r19.html
    data/vision/ball_audit_r20.html

Each tool spans first tap − 1.0 s to last tap + 2.0 s, the same span
the r6–r10 paths were clicked on. The clip / candidate / c3 window
entries for these five were cut to that span too (below), so your
clicks, the pose stream and the tracker all see the same frames.

## How to click (per rally)

1. Open the HTML in Chrome/Safari. Pick `full_match.mp4.webm` (the
   SAME file you labeled contacts on — the frame times are absolute
   VOD seconds, so a different cut of the broadcast would mis-seek).
2. Frame 0 is one second before your serve tap. Sanity-check the
   scorebug on frame 0 before clicking (expected: r17 UTAH 5 – CHICAGO
   4 Chicago serving; r18 / r19 are the next two points; r20 UTAH 6 –
   4 Utah serving; r21 UTAH 7 – 4 Utah serving). A replay carries the
   rally's END score on the bug — if the bug is already past the
   expected score, stop and say so; do not click a replay.
3. One answer per frame, four kinds, exactly as the r6–r10 paths:
   click = V (clean ball), **S** + click = streak / blur that IS the
   ball (click its centre, be consistent), **I** + click = ball hidden
   behind a body or paddle (your trajectory sense places it), **N** =
   genuinely can't place it. `T` toggles the trail when you lose it;
   `←`/`→` step, `,`/`.` jump ±10, `⌫` clears a frame.
4. Centre-of-blob is fine. Don't agonize; consistency on streaks
   matters more than precision.
5. **Export at the end of every sitting** (⬇) and save as
   `data/vision/ball_path_r{N}.csv` — partial is fine, ⬆ import
   restores it. Then share the CSV in-thread (or push it); nothing else
   from your machine is needed. Progress autosaves in the browser
   (localStorage key `ball_audit_r{N}`), but the export is the backup.
6. Skip `--score` on r20 / r21 for now; export only. Scoring is a
   grade, and grades on the seals wait for your go.

## Sitting 3 — the fast rallies that need a contact pass first

r2–r5 carry only PREFILL contact rows (pop-era types and approximate
times, `source = prefill`). The ball tool refuses to build off
prefills by design (its answer key must be your taps), so these need
your contact pass first — the same job you did for r17–r27, about one
keypress per shot with the chained seek.

| order | rally | old-era shot mix | why |
|---|---|---|---|
| 1 | **r3** | 17 shots: 10 fast-type (counters / speed-ups) | the fastest rally in game 1 by type count; note its pin marks a REPLAY (pin_realignment.md) — the live rally is at ~56–79 s, scorebug 0-0, Tuionetoa serving |
| 2 | **r4** | 16 shots: 8 dinks → 2 speed-ups + 4 counters | slow→fast transition inside one rally |
| 3 | r2 | 11 shots, 4 fast-type | short and quick |
| 4 | r5 | 14 shots | a fourth train rally if the appetite holds |

Steps:

1. Open `data/vision/contact_audit_chicago0725.html`, go to the rally,
   run the tap pass then the pace pass (F/S per contact), export
   `contact_labels_chicago0725.csv` at the end of the sitting and
   share it — it replaces the committed one (the prefill rows for those
   rallies become `manual`).
2. Then either run locally

       python3 vision/make_ball_audit.py --rally 3     # -> data/vision/ball_audit_r3.html

   and click as above, or hand me the contact CSV and I generate the
   tools, cut the clips, extract candidates and commit them (five
   minutes here) before you click.

r14 and r16 are the same shape (prefill only, 11 and 7 shots) and are
the next two after r5 if you keep going. **Do not click r22–r34.**

## The one thing the machine cannot do here: pose streams

The tracker's floor / player-box and the hand-off pass read the pose
npz (`r00NN.npz`). This container has no GPU and no torch, so the
five new rallies need ONE Colab run (gpu_runbook.md, same folder as
before — your Drive folder already holds r0002–r0010, so r2–r5 are
covered for sitting 3 already):

    !python pose_extract.py --video full_match.mp4.webm --device cuda \
        --labels contact_labels_chicago0725.csv \
        --windows rally_windows_chicago0725_v4.csv \
        --rallies 17,18,19,20,21 --out-dir pose

Its windows come from your contact taps (serve − 1.5 s to a cadence
cap), which cover the click span with margin. Drop the five npz into
the Drive pose folder (or share them) and the staging below is
complete. If Colab is a hassle, say so: the CPU RTMPose fallback runs
here at a few minutes per rally, at the production-spine quality
rather than the verdict backbone, and for the ball tracker that is
sufficient (the pose stream is a floor / occlusion prior, not the
instrument under test).

## What is staged on the machine side (done, committed)

- `data/vision/ball_candidates_r{17,18,19,20,21}.csv.gz` — classical
  candidates extracted from clips cut from the full match at the
  audit-tool span (offsets 426.8 / 455.1 / 476.7 / 483.7 / 515.4 s;
  957 / 439 / 333 / 1406 / 523 frames at 60 fps). `check_clip.py` PASS
  at d = 0 for r17 and r21 against the committed CSVs.
- `cut_clip.py --stage N full_match.mp4.webm` reproduces those clips
  from your Drive copy of the match (default mode reproduces them from
  the committed CSV once you have it). The clips themselves are not in
  git (house rule) and this container is ephemeral — re-cut with that
  one line when a fresh thread needs them, then `check_clip.py N`.
- `c3_lab.py` WINDOWS carries (serve, end, npz) for 17–21 at the
  audit-tool span; `python3 c3_lab.py N` runs the moment the npz lands.

## r17 — DONE end to end (2026-09-02, same day as the delivery)

Pose came from the CPU fallback (rtmpose-balanced, 20.7 min, 7,063
detections / 36 tracks) — Colab is optional for the rest; the npz is
gitignored, so a fresh thread re-extracts (`python3 vision/pose_extract.py
--video vision/ballsearch/full_match.webm --backend rtmpose --rtm-mode
balanced --rallies 17 --out-dir vision/ballsearch --labels
data/vision/contact_labels_chicago0725.csv --windows
data/vision/rally_windows_chicago0725_v4.csv`) or drops in your Colab
npz. Chain after that: `ball_grade.py` (train dry-run → anchors,
committed) → `c3_lab.py 17` → `emission.py cache 17` → `train_read.py 17`
(→ `train_read_r17.txt`, committed; the read itself is in HANDOFF.md).
Headline: 245 / 379 clicks at 1.00 precision, one wrong point in the
whole rally; every miss is a hole, and the holes are the SLOW contacts
next to a player (432.1 / 432.8 / 436.0 s) and the serve — the fast
exchanges are the best-covered bucket. That is the opposite of the
premise that picked r17 ("the hands battle the tracker keeps losing").

## What happens when the CSVs arrive

1. r17 (+ r18, r19, later r3/r4/r2/r5) join TRAIN: `emission.py train`
   refits the candidate scorer with the new fold (cross-fold `_x`
   caches go n-way), `geom_fix.py tune` / `handoff.py tune` re-tune on
   the enlarged train with the selection rules as written in their
   gates. Nothing about r9 / r10 changes; they stay spent evaluation.
2. r20 / r21 become the sealed pair. Their one-shot happens only on
   your explicit authorization, on a selection rule written down
   before the numbers (`gapfill_gate.md` / `handoff_gate.md` pattern).
3. ~~The first read that matters~~ DONE for r17 (above): does the hand-off pass's fast-flight
   miss rate on r17 look like r6 / r7 (arrive/depart pairs lost next to
   a player), or different? That read decides whether the streak
   detector question (a timing instrument, per the 2026-09-02 thread)
   is worth building — it needs your rule change (S clicks as positives
   for that instrument only) before any code exists.
