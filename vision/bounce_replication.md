# Where the ball bounced — the tracker vs the owner's clicks

**2026-09-04.** Answers the first of the three product questions
("the final product is going to say *the ball bounced somewhere* — how
do we get that best?").

Reproduce:

```bash
cd vision
python3 ball_replicate.py --rally 9 --npz ballsearch/r0009.npz \
        --anchors ballsearch/anchors_grade_r9.csv
```

## The two candidate instruments

| | what it looks at | median error |
|---|---|---|
| `ballsearch/bounce_proxy.py` | player feet + a fixed lead offset. **No ball.** | 4.0 ft train / **5.3 ft** eval |
| `ball_replicate.py` (this) | the tracker's own path, lifted to 3D and fit as arcs | **1.9 ft** |

The proxy was built to test whether the ball could be skipped entirely.
It can't, and it doesn't have to be. It is a floor, not an answer, and
it should stop being quoted as one.

## Result — fully automated path vs the owner's clicked path

Both sides go through the same `court3d` two-pass reconstruction; the
tracked side is decoded and segmented with no clicks anywhere. Truth is
the bounce point fit from the owner's clicks.

| rally | bounces matched | median | contact impacts (same run) | bounces tracked/human |
|---|---|---|---|---|
| r7 | 3/3 | 2.05 ft | 2.23 ft | 7 / 3 |
| r9 | 5/13 | 0.83 ft | 3.60 ft | 19 / 13 |
| r10 | 2/13 | 1.60 ft | 2.60 ft | 11 / 13 |
| r17 | 3/5 | 1.91 ft | 4.66 ft | 9 / 5 |

**Pooled n=13: median 1.91 ft, mean 1.78, p90 3.52, max 4.10.**
46% land within 1 ft, 54% within 2 ft, 77% within 3 ft.

Court scale for reading those: kitchen 7 ft deep, half-court 22 ft,
court 20 ft wide. A 1.9 ft median is well inside the kitchen — it
supports a real bounce map. A 5.3 ft median does not.

## The bottleneck is RECALL, not accuracy

Only **13 of 34** human bounces (38%) got a tracked counterpart within
0.30 s. That is the number to work on, and it is not shyness — the
tracker *over*-calls bounces on three of the four rallies (r9: 19 vs
13, r7: 7 vs 3, r17: 9 vs 5). It is emitting bounce events at the wrong
times, and only the ones where segmentation happened to agree get
matched. Segmentation, not detection.

## Why bounces beat contacts, structurally

Contact impacts in the same runs: 2.23 / 3.60 / 2.60 / 4.66 ft — worse
than bounces on every rally. That ordering is not luck:

* a bounce sits on z=0, the plane the homography solves to 0.06 ft,
  so it carries no depth degeneracy;
* it is bracketed by a fitted arc on each side, both pinned by
  mid-flight points — the tracker's strongest region (~90%);
* a contact is where the ball is occluded by the paddle and the arm,
  and it is the one place the path has a hole (~72-78%).

So "where did it bounce" is the easy question and "where did it meet
the paddle" is the hard one, and the ball-visibility number that closed
this thread (64% findable per in-play *frame*) was the wrong
denominator for either: an arc needs three well-spread points per
*flight*, not every frame.
