"""Cross-team talent overlap: how often can a stronger team's best woman
outrank a weaker team's men?

The "every man better than every woman" fact is only revealed WITHIN a
team (176/176 men-first DreamBreaker orders). It is NOT a league-wide
ordering: a superteam's woman can outrank a budget team's man. This
quantifies how often that overlap exists under a within-team-only draw
(women below their OWN team's weakest man; cross-team overlap allowed),
supporting the "spectacle is a real favored matchup, not a sacrifice, in
budget-mismatch games" point in db_crossgender_conditions.md (5c).

Pure value comparisons + rally sigmoid; no DP needed. Illustrative (the
percentages depend on the draw parameters; the direction is robust).

Run: python model/db_crossteam_overlap.py
"""
import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_scenarios import sigmoid, K_RALLY  # noqa: E402

k = K_RALLY
rng = random.Random(20260723)

def draw_team():
    L = rng.uniform(0, 2)                 # team budget/level
    men = sorted(L + rng.uniform(0, 0.8) for _ in range(2))  # ascending
    wmax = min(men) - 0.05
    women = sorted(rng.uniform(L - 1.0, wmax) for _ in range(2))
    return {"L": L, "men": men, "women": women}  # each ascending

N = 20000
overlap_weakman = 0   # T1 best woman > T2 weakest man
overlap_bestman = 0   # T1 best woman > T2 best man
fav_and_overlap = 0   # T1 clear favorite AND showcase available (competitive/fav vs a man)
fav_n = 0
comp_available = 0    # some T1 woman is competitive (rally .45-.55) vs some T2 man
for _ in range(N):
    t1, t2 = draw_team(), draw_team()
    bw1 = t1["women"][-1]                 # T1 best woman
    # team strength proxy = mean of 4 values
    s1 = mean(t1["men"] + t1["women"]); s2 = mean(t2["men"] + t2["women"])
    if bw1 > t2["men"][0]: overlap_weakman += 1
    if bw1 > t2["men"][-1]: overlap_bestman += 1
    # competitive showcase: any T1 woman vs any T2 man with rally p in [.45,.55]
    comp = any(0.45 <= sigmoid(k*(w - m)) <= 0.55
               for w in t1["women"] for m in t2["men"])
    if comp: comp_available += 1
    # heavy favorite regime
    if s1 - s2 > 0.5:                     # T1 clearly stronger team
        fav_n += 1
        # showcase = T1 best woman competitive-or-favored vs some T2 man
        showcase = any(sigmoid(k*(bw1 - m)) >= 0.45 for m in t2["men"])
        if showcase: fav_and_overlap += 1

print(f"N = {N} random asymmetric pairs (within-team men>women; cross-team overlap allowed)")
print(f"T1 best woman > T2 WEAKEST man:            {100*overlap_weakman/N:.1f}%")
print(f"T1 best woman > T2 BEST man:               {100*overlap_bestman/N:.1f}%")
print(f"some competitive W-v-M matchup exists:     {100*comp_available/N:.1f}%")
print(f"among T1 clear-favorite pairs (n={fav_n}):")
print(f"  T1 star woman competitive/favored vs a T2 man: {100*fav_and_overlap/fav_n:.1f}%")
