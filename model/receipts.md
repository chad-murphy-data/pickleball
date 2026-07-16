# Receipts ledger

Every forecast we commit *before* an event, graded *after* it — wins and losses
alike. The model being publicly wrong ~23% of the time is part of the product;
this file is the running scorecard that keeps us honest.

Method: a forecast is committed to the repo before first serve (timestamped by
commit). After the result finalizes in the API, we record the actual outcome,
mark the overall call HIT/MISS on the favorite, and note a Brier score where the
forecast gave a probability. No retro-editing the prediction — only appending the
grade.

| # | committed | event | call (pre-match) | result | overall |
|--:|:--|:--|:--|:--|:--|
| 1 | 2026-07-12 | MLP Mid-Season **Gold Final** — NJ 5s vs STL Shock | **St. Louis 60.7%** (v2) · 57.0% (v1) | STL won **3–0** | ✅ **HIT** |

### Entry 1 — MLP Mid-Season Gold Final
`model/prediction_midseason_final.md` (committed pre-first-serve 2026-07-12,
graded 2026-07-16).

- **Overall: HIT.** Favored St. Louis under both models; St. Louis won 3–0.
  Match-level Brier **v2 0.154 / v1 0.185** (both beat 0.25 coin-flip).
- **Right answer, wrong path.** The forecast's central scenario was a 46.5%
  DreamBreaker "hinge"; instead STL swept in regulation.
- **Notable miss:** the v2 **88%-NJ women's-doubles call** (Waters/Johnson) —
  STL won it 11–6. Anna Leigh Waters lost both her lines (WD 6–11, MXD1 8–11).
- **Per-line Brier** (3 played lines): v2 0.358, v1 0.269 — dominated by the WD
  miss; the less-confident v1 scored better line-by-line (n=3, read gently).

Full breakdown in the prediction file's RESULT section.

---

*Next scheduled grading:* `model/registered_predictions.md` (frozen 2026-07-12) vs
games dated after 2026-07-12 — due September 2026 per CLAUDE.md.
