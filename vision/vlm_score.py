"""Score the blind VLM hitter-call test (2026-08-19).

The 20 calls in CALLS were locked in-thread BEFORE the key existed on
this machine, and are transcribed verbatim. Key committed alongside as
data/vision/vlm_test_key_20260819.csv, so the whole test is
reproducible:

    python3 vision/vlm_score.py data/vision/vlm_test_key_20260819.csv

Scoring caveat that is the point, not a bug: the key names HITTERS, the
calls name COURT POSITIONS, and the map between them (who stood
camera-left) is not in any committed artifact. Side/team and pace score
exactly; within-team left/right needs one human glance to orient. See
swing_explore_notes.md 2026-08-19."""
import csv, sys
from collections import Counter, defaultdict
sys.path.insert(0, "vision")
from fastslow_check import classify_type

CALLS = {  # question -> (position, pace)   [camera view]
 1:("far-right","slow"),  2:("far-right","slow"),  3:("near-left","slow"),
 4:("far-right","fast"),  5:("far-left","slow"),   6:("near-left","slow"),
 7:("far-left","slow"),   8:("near-left","slow"),  9:("far-left","slow"),
10:("near-left","slow"), 11:("far-right","fast"), 12:("near-left","fast"),
13:("near-right","fast"),14:("far-left","fast"),  15:("far-right","slow"),
16:("near-left","slow"), 17:("near-left","fast"), 18:("far-right","slow"),
19:("far-right","fast"), 20:("near-left","slow")}

UTAH = {"Allyce Jones", "Etta Tuionetoa"}          # black kit
CHI  = {"Emma Nelson", "Ting Chieh Wei"}           # white kit
# MLP switches ends at 6; strips at 6-4 (q16, q19) show the swap.
SWITCHED = {16, 19}

key = {}
for r in csv.DictReader(open(sys.argv[1])):
    q = int(r["question"].split(".")[0][1:])
    key[q] = (r["hitter_name"], r["shot_type"], r["pace"],
              int(r["rally_cum"]), int(r["shot_index"]))

side_ok = pace_ok = 0
by_truth_pace = defaultdict(lambda: [0, 0])
pos_by_name = defaultdict(Counter)
rows = []
for q in sorted(key):
    pos, pace_call = CALLS[q]
    name, ty, pace_true, rally, shot = key[q]
    near_team = CHI if q in SWITCHED else UTAH
    called_near = pos.startswith("near")
    truth_near = name in near_team
    s_ok = called_near == truth_near
    p_ok = pace_call == pace_true
    side_ok += s_ok
    pace_ok += p_ok
    by_truth_pace[pace_true][0] += p_ok
    by_truth_pace[pace_true][1] += 1
    pos_by_name[name][pos] += 1
    rows.append((q, rally, shot, name, ty, pace_true, pos, pace_call,
                 s_ok, p_ok))

print("q   rally.shot  truth                       call                 side pace")
for q, ra, sh, name, ty, pt, pos, pc, s, p in rows:
    print(f"q{q:02d}  r{ra:>2}.{sh:<2}  {name:<16} {ty:<9}{pt:<5} "
          f"{pos:<11}{pc:<5}  {'ok' if s else 'MISS':<5}{'ok' if p else 'MISS'}")

n = len(rows)
print(f"\nSIDE/TEAM      {side_ok}/{n} = {side_ok/n:.0%}   (chance 50%)")
print(f"PACE           {pace_ok}/{n} = {pace_ok/n:.0%}   (chance 50%)")
for k in ("fast", "slow"):
    a, b = by_truth_pace[k]
    print(f"   truth {k:<5} {a}/{b} = {a/b:.0%}")
gap = by_truth_pace['slow'][0]/by_truth_pace['slow'][1] - \
      by_truth_pace['fast'][0]/by_truth_pace['fast'][1]
print(f"   slow-minus-fast gap: {gap:+.0%}  (registered 15-25pp worse on fast)")

print("\nposition calls by true player:")
for name, c in sorted(pos_by_name.items(), key=lambda kv: -sum(kv[1].values())):
    tot = sum(c.values())
    top, topn = c.most_common(1)[0]
    print(f"   {name:<16} n={tot:<3} {dict(c)}   -> modal {top} {topn}/{tot}")
absent = (UTAH | CHI) - set(pos_by_name)
print(f"   never drawn: {absent or 'none'}")

print("\npace errors, by truth type:")
for q, ra, sh, name, ty, pt, pos, pc, s, p in rows:
    if not p:
        print(f"   q{q:02d} r{ra}.{sh} truth {ty}({pt}) - I called {pc}")
