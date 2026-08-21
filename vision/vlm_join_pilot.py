"""The JOIN pilot: does 90% partner accuracy survive decoder-placed frames?

WHAT THIS MEASURES AND WHY (2026-08-21, user direction). The pieces on
the shelf, per STATUS.md: pose tracking (solid), shot COUNTS per rally
via the alternation decoder (161/162), side per shot (exact, free from
alternation + the referee log's serve side), and WHO-hit-it partner
recognition at 90% on Sonnet (vlm_tier_test --mode partner). The plan
for coding a whole match is to glue these: decode each rally's contact
sequence, cut a strip at each decoded time, ask the VLM which partner
hit it, sum into touch counts.

Every link in that chain is measured EXCEPT ONE: the 90% was measured
on strips cut at TRUE label times, while production strips would be cut
at DECODER times — and the decoder's placement is only ~51% precise at
+/-0.30s even though its counts are near-exact. If the VLM's partner
call is robust to that placement slop (the strip still shows the right
moment-ish, hitter mid-prep or mid-follow-through), the glue holds and
the whole-match run is licensed. If accuracy craters, placement is the
real blocker and no overnight run should be bought until it's fixed.
This pilot measures exactly that link, on rallies where the truth is
already hand-labeled — so the answer costs a few dollars, not a blind
overnight run on unlabeled footage.

EXPLORATION, not a gate (same standing as swing_explore.py): TRAIN
rallies only, leave-one-rally-out scorer fits mirroring swing_explore's
loop, holdout untouched. r9/r10 are dropped entirely (contact_gate.md:
their label spans exceed log durations ~24 s; their timing is suspect,
and timing is the very thing under test). The serve side s0 comes from
the true first contact's team — legitimate, not leakage: in deployment
that bit comes from the referee log, which is free and exact.

THE JOIN IS BY ORDER, NOT BY TIME. Production has no true times to
match against — the k-th decoded contact of team T simply IS that
team's k-th touch as far as the product is concerned. So the pilot
scores exactly that correspondence (k-th decoded event of team T vs
k-th true contact of team T), and reports time-tolerance agreement
only as a diagnostic. Decoded ghosts claim no timestamp and cut no
strip; they shift ordinals only, exactly as in decoded_events.

Two stages, cheapest first:

  --dry     no API, no video: decode + order-join only. Prints count
            deltas and the |dt| distribution of joined pairs — how
            often the decoded time is even close enough for the strip
            (t-0.1/t/t+0.1) to contain the true contact. Run this
            first; if placement is hopeless it shows here for free.

  (full)    also needs --video + ANTHROPIC_API_KEY: cuts strips at
            DECODED times with the SAME cut_strip as vlm_frame_sample
            (identical instrument to the 90% test — the only changed
            variable is the timestamp source) and asks ONE call per
            decoded event: name A / name B / '{OTHER}' / '{DEAD}'.
            The two escapes are the dry run's defect counters riding
            channels already measured to work (play/no-play 30/30,
            side 95%) — DEAD filters trailing-junk events, OTHER
            flags wrong-team offerings from ordinal slips. Reports:
            answer mix, accuracy under the order-join AND a
            time-tolerant join (separating join slips from VLM
            error), accuracy by |dt| bin, DEAD-heavy rallies to
            re-decode, and the product metric — per-player touch
            counts vs truth. Calls CSV is written next to the strips
            so nothing needs re-spending. Note the benchmark caveat:
            the 90% was forced-choice; the escapes give the model a
            hedge, so watch the answer mix for over-hedging on
            good-|dt| events.

Run on the Mac (pose_rtm/ + video live there):
    python3 vlm_join_pilot.py --dry
    python3 vlm_join_pilot.py --video full_match.mp4.webm
    python3 vlm_join_pilot.py --selftest      (anywhere, no deps)
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

LABELS = "contact_labels_chicago0725.csv"
WINDOWS_V4 = "rally_windows_chicago0725_v4.csv"
SPLIT = "label_split.csv"
POSE_DIR = "pose_rtm"
EXCLUDE = {9, 10}          # contact_gate.md span anomaly — timing suspect
STRIP_OFFS = (-0.10, 0.0, 0.10)   # vlm_frame_sample's OFFSETS, frozen
DT_BINS = ((0.0, 0.2), (0.2, 0.5), (0.5, float("inf")))
TRAIL_PAD = 1.5    # events later than last true contact + this = trailing
TIME_TOL = 0.5     # secondary time-join tolerance (diagnostic)
OTHER = "other player"      # escape: hitter is neither offered name
DEAD = "no live shot"       # escape: between points / no rally shown

# DRY-RUN FINDINGS 2026-08-21 (15 train rallies) that shaped the API
# stage — three defects, each with machinery already on the shelf:
#   A TRAILING JUNK: v4 windows carry 7-13s of post-rally dead time and
#     the decoder's span constraint marches into it (2-contact rallies
#     decode 6-8 events; over-count rallies still join at 82-100%
#     <=0.2s — real contacts found, junk appended). Counter: the DEAD
#     escape below (play/no-play measured 30/30) drops junk events in
#     the same paid call.
#   B WRONG-SEGMENT (r5/r6/r17): weak-dink rallies lose the scoring
#     contest to that same trailing movement — decode lands 8-11s away
#     wholesale. DEAD-heavy rallies are flagged for re-decode;
#     scorebug flip-sync end-clipping is the reserve fix.
#   C OFF-BY-ONE CASCADES (r1/r3/r4/r16): one mid-rally slip shifts
#     every later ordinal (median |dt| ~ one dink gap) and flips the
#     decoded parity, so the partner question offers the WRONG TEAM's
#     names. Counter: the OTHER escape (side is a 95% call for the
#     VLM) turns forced-wrong answers into detectable flags; the
#     time-join secondary scoring separates join slips from VLM error.


def train_only(split_path):
    p = Path(split_path)
    if not p.exists():
        raise SystemExit(f"{split_path} not found — the split is mandatory")
    return {int(r["rally_cum"]) for r in csv.DictReader(open(p))
            if r["split"] == "train"}


def name_map(labels_path):
    """uuid -> display name, straight from the labels file."""
    out = {}
    for r in csv.DictReader(open(labels_path)):
        out[r["hitter_uuid"].lower()] = r["hitter_name"]
    return out


def true_hitters(labels_path, train):
    """rally_cum -> [(t, hitter_name)] real contacts, time-sorted."""
    out = defaultdict(list)
    for r in csv.DictReader(open(labels_path)):
        cum = int(r["rally_cum"])
        if cum not in train or cum in EXCLUDE:
            continue
        if r.get("contact", "1") == "0":
            continue
        t = float(r["t_refined_s"] or r["t_tap_s"])
        out[cum].append((t, r["hitter_name"]))
    for v in out.values():
        v.sort()
    return dict(out)


def decode_train_rallies(labels_path, windows_path, pose_dir, train):
    """Leave-one-rally-out decode of every eligible train rally,
    mirroring swing_explore.run()'s loop exactly (same loaders, same
    m-mapping, same s0). Returns
    {cum: {"events": [(t, team)], "ghosts": n, "teams": (setA, setB)}}
    with team 0/1 in label coordinates (0 = teamA of the windows file).
    Heavy imports live here so --selftest/--dry-help run anywhere."""
    import numpy as np  # noqa: F401  (swing_explore needs it)
    import swing_explore as SE
    from contact_ceiling import (load_rosters, load_labels,
                                 rally_candidates, rally_coverage)

    rosters = load_rosters(Path(windows_path))
    labels = load_labels(Path(labels_path), rosters)
    rallies = {}
    for cum, d in labels.items():
        if cum not in train or cum in EXCLUDE or not d["contacts"]:
            continue
        rd = SE.load_rally(pose_dir, cum)
        if rd is None:
            continue
        cands, _b = rally_candidates(rd["z"])
        _fl, m_raw = rally_coverage(d["contacts"], cands, 2, SE.TOL_S)
        m_srv, margin = SE.serve_mapping(rd, d["contacts"])
        m = m_srv if margin >= 1.25 else m_raw
        rallies[cum] = {"rd": rd, "contacts": d["contacts"],
                        "whiffs": d["whiffs"], "m": m}
    if len(rallies) < 3:
        raise SystemExit(f"need >=3 train rallies with labels + pose, "
                         f"found {len(rallies)} — check --pose-dir")

    out = {}
    for held in sorted(rallies):
        Xtr, ytr = [], []
        for cum, r in rallies.items():
            if cum == held:
                continue
            X, y = SE.rally_instances(r["rd"], r["contacts"], r["whiffs"],
                                      r["m"])
            Xtr += X
            ytr += y
        model = SE.fit_logreg(np.stack(Xtr), np.array(ytr, float))
        r = rallies[held]
        dets = SE.score_rally(model, r["rd"])
        s0 = r["contacts"][0][1] ^ r["m"]
        path = SE.decode_rally(dets, s0)
        events = [(t, side ^ r["m"]) for t, side, _sc, _g in path]
        out[held] = {"events": events,
                     "ghosts": sum(g for *_x, g in path),
                     "teams": rosters[held]}
    return out


def order_join(events, truths_by_team):
    """k-th decoded event of team T <-> k-th true contact of team T.
    events: [(t, team)]; truths_by_team: {team: [(t, name)]}.
    Returns (pairs, extra, missing): pairs =
    [(t_dec, team, k, t_true, name_true, dt)]; extra = decoded events
    past their team's true count; missing = true contacts never
    reached by a decoded event."""
    pairs, extra = [], []
    seen = Counter()
    for t, team in events:
        k = seen[team]
        seen[team] += 1
        truths = truths_by_team.get(team, [])
        if k < len(truths):
            tt, name = truths[k]
            pairs.append((t, team, k, tt, name, abs(t - tt)))
        else:
            extra.append((t, team, k))
    missing = [(team, k, tt, name)
               for team, truths in truths_by_team.items()
               for k, (tt, name) in enumerate(truths) if k >= seen[team]]
    return pairs, extra, missing


def split_truth_by_team(hitters, teams, names):
    """[(t, name)] + (setA_uuids, setB_uuids) -> {0: [...], 1: [...]}
    preserving time order within team."""
    name_to_team = {}
    for team, uuids in ((0, teams[0]), (1, teams[1])):
        for u in uuids:
            if u in names:
                name_to_team[names[u]] = team
    out = {0: [], 1: []}
    for t, name in hitters:
        if name not in name_to_team:
            raise SystemExit(f"hitter {name!r} not in either roster")
        out[name_to_team[name]].append((t, name))
    return out


def team_names(teams, names, team):
    got = sorted(names[u] for u in teams[team] if u in names)
    if len(got) != 2:
        raise SystemExit(f"team {team} resolves to {got} — need exactly "
                         f"2 named players (labels never show one of them?)")
    return got


def dt_bin(dt):
    for lo, hi in DT_BINS:
        if lo <= dt < hi:
            return f"{lo:.1f}-{hi:.1f}s" if hi < 9 else f">{lo:.1f}s"
    return "?"


def trim_trailing(events, last_true_t, pad=TRAIL_PAD):
    """(kept, dropped): events later than last_true + pad are trailing.
    Diagnostic only — it peeks at truth. Its production analog is the
    DEAD escape (play/no-play, 30/30) or a scorebug end-clip."""
    kept = [(t, team) for t, team in events if t <= last_true_t + pad]
    return kept, len(events) - len(kept)


def print_dry(decoded, truths, names):
    all_dt, trim_dt = [], []
    tot_true = tot_dec = tot_ghost = tot_trail = 0
    print(f"{'rally':>5} {'true':>5} {'dec':>4} {'ghost':>5} "
          f"{'extra':>5} {'miss':>4} {'trail':>5}  "
          f"joined |dt| med/frac<=0.2  ->  TRIMMED med/frac<=0.2")
    for cum in sorted(decoded):
        d = decoded[cum]
        tbt = split_truth_by_team(truths[cum], d["teams"], names)
        pairs, extra, missing = order_join(d["events"], tbt)
        dts = sorted(p[5] for p in pairs)
        all_dt += dts
        n_true = sum(len(v) for v in tbt.values())
        last_true = max(t for t, _n in truths[cum])
        kept, n_trail = trim_trailing(d["events"], last_true)
        tpairs, _te, _tm = order_join(kept, tbt)
        tdts = sorted(p[5] for p in tpairs)
        trim_dt += tdts
        tot_true += n_true
        tot_dec += len(d["events"])
        tot_ghost += d["ghosts"]
        tot_trail += n_trail

        def stat(xs):
            if not xs:
                return "   -  /  - "
            fr = sum(1 for x in xs if x <= 0.2) / len(xs)
            return f"{xs[len(xs) // 2]:5.2f} / {fr:3.0%}"

        print(f"{cum:>5} {n_true:>5} {len(d['events']):>4} "
              f"{d['ghosts']:>5} {len(extra):>5} {len(missing):>4} "
              f"{n_trail:>5}  {stat(dts)}       ->  {stat(tdts)}")
    all_dt.sort()
    trim_dt.sort()
    n = len(all_dt)
    print(f"\nTOTAL {tot_true} true / {tot_dec} decoded "
          f"(+{tot_ghost} ghosts, {tot_trail} trailing) — "
          f"{n} joined pairs")
    if n:
        print(f"|dt|  median {all_dt[n // 2]:.2f}s   "
              f"p90 {all_dt[int(0.9 * n)]:.2f}s")
        for lo, hi in DT_BINS:
            c = sum(1 for x in all_dt if lo <= x < hi)
            print(f"      {dt_bin(lo):>9}: {c:>4}  ({c / n:.0%})")
    if trim_dt:
        m = len(trim_dt)
        near = sum(1 for x in trim_dt if x <= 0.2) / m
        print(f"TRIMMED (drop events > last true contact + {TRAIL_PAD}s "
              f"— truth-informed upper bound on the DEAD-escape filter): "
              f"{m} pairs, median {trim_dt[m // 2]:.2f}s, "
              f"{near:.0%} <=0.2s")
    print(f"\nA strip spans t+/-0.1s; |dt|<=0.2s means the true "
          f"contact is inside or adjacent to it. The API stage "
          f"measures whether the VLM's call survives the rest — "
          f"this table only predicts it.")


def escape_tool(offered):
    """Hitter call over the OFFERED names plus the two escapes. DEAD
    rides the measured 30/30 play/no-play channel and filters defect
    A's trailing junk; OTHER catches a hitter outside the offered set.

    `offered` is 2 names in --names team mode, 4 in all4 mode. The
    2-name form makes OTHER load-bearing for defect C (wrong-team
    offering); the 4-name form removes that failure entirely and turns
    parity into something we CHECK rather than something we assume —
    see the 2026-08-21 API findings below."""
    return {
        "name": "call_shot",
        "description": ("Report who hit the shot shown, or that no live "
                        "shot is shown."),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": list(offered) + [OTHER, DEAD],
                    "description": (
                        "The named player hitting the ball in these "
                        f"frames; '{OTHER}' if someone IS hitting but "
                        f"it is none of the named players; '{DEAD}' if "
                        "no one is hitting — players walking, "
                        "resetting, celebrating, between points."),
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    }


def escape_prompt(offered):
    who = " or ".join([", ".join(offered[:-1]), offered[-1]]) \
        if len(offered) > 2 else " or ".join(offered)
    return (
        "This is a strip of 3 frames (0.1 s apart) from a pro "
        "pickleball broadcast. If a shot is being hit in these frames "
        f"by {who}, name the hitter. If someone else is hitting, "
        f"answer '{OTHER}'. If no shot is happening — players between "
        f"points, resetting, celebrating — answer '{DEAD}'. Use the "
        "call_shot tool."
    )


def time_join(events, truths_by_team, tol=TIME_TOL):
    """Secondary diagnostic join: nearest same-team true contact within
    tol, greedy one-to-one by |dt|. Separates 'the VLM read the strip
    wrong' from 'the order-join paired it against the wrong truth row'
    when defect C shifts ordinals. Returns {event_index: (t, name)}."""
    cands = []
    for i, (t, team) in enumerate(events):
        for j, (tt, name) in enumerate(truths_by_team.get(team, [])):
            if abs(t - tt) <= tol:
                cands.append((abs(t - tt), i, j, team, tt, name))
    cands.sort()
    used_i, used_j, out = set(), set(), {}
    for _d, i, j, team, tt, name in cands:
        if i in used_i or (team, j) in used_j:
            continue
        used_i.add(i)
        used_j.add((team, j))
        out[i] = (tt, name)
    return out


def rescore(calls_path, labels_path, windows_path, split_path):
    """Re-read an existing join_pilot_calls.csv and test the PARITY
    HYPOTHESIS for free — no API, no video. Costs nothing, so it runs
    before any re-spend.

    The 2026-08-21 run answered OTHER 73 times out of 186. Two readings:
    (a) the model hedges when unsure, or (b) we OFFERED THE WRONG TEAM
    and it correctly refused both names.

    VERSION 1 OF THIS CHECK WAS VACUOUS and its 0-of-65 result must not
    be quoted. It read the truth out of the CSV's order_truth/time_truth
    columns — but BOTH joins are same-team-constrained by construction
    (each looks up truths_by_team[event's own team]), so the attached
    truth is ALWAYS from the offered team and 'OFF' was unreachable.
    A test that cannot fail measures nothing.

    The fix: join each decoded event to the nearest true contact in the
    WHOLE rally, ignoring team, then ask whose it was. That can land on
    either team, so (b) is now falsifiable. Nearest-|dt| is reported
    alongside, because the third reading the vacuous version could not
    see is that OTHER answers sit on frames with no contact nearby at
    all — junk the DEAD escape should have caught."""
    import csv as _csv
    rows = list(_csv.DictReader(open(calls_path)))
    train = train_only(split_path)
    names = name_map(labels_path)
    truths = true_hitters(labels_path, train)
    from contact_ceiling import load_rosters
    rosters = load_rosters(Path(windows_path))

    tally = Counter()
    dts = defaultdict(list)
    per_rally = defaultdict(Counter)
    for r in rows:
        cum = int(r["rally_cum"])
        if cum not in train or cum not in truths:
            tally["rally not in train/labels"] += 1
            continue
        t_dec = float(r["t_decoded"])
        tt, tname = min(truths[cum], key=lambda x: abs(x[0] - t_dec))
        dt = abs(tt - t_dec)
        team = int(r["team"])
        offered = set(team_names(rosters[cum], names, team))
        on_offered = tname in offered
        ans = r["answer"]
        kind = ("named" if ans not in (OTHER, DEAD) else
                "OTHER" if ans == OTHER else "DEAD")
        tally[f"{kind}: nearest truth {'ON' if on_offered else 'OFF'} "
              f"offered team"] += 1
        dts[kind].append(dt)
        if kind == "OTHER":
            per_rally[cum]["off" if not on_offered else "on"] += 1
        if kind == "named":
            tally["named: model agreed with nearest truth"] += (ans == tname)

    print(f"PARITY DIAGNOSTIC v2 — {len(rows)} calls "
          f"(nearest contact in the rally, TEAM-BLIND)\n")
    for k in sorted(tally):
        print(f"  {k:<50} {tally[k]:>4}")
    print("\nnearest-truth |dt| by answer kind (how close a real "
          "contact was):")
    for kind in ("named", "OTHER", "DEAD"):
        v = sorted(dts[kind])
        if not v:
            continue
        n = len(v)
        near = sum(1 for x in v if x <= 0.2) / n
        print(f"  {kind:<6} n={n:<4} median {v[n // 2]:6.2f}s   "
              f"{near:.0%} within 0.2s")
    off = tally["OTHER: nearest truth OFF offered team"]
    on = tally["OTHER: nearest truth ON offered team"]
    if off + on:
        print(f"\nOf {off + on} OTHER answers, {off} "
              f"({off / (off + on):.0%}) had the nearest true contact "
              f"on the team we did NOT offer.")
        if off > on:
            print("  -> PARITY IS A REAL CULPRIT: the model was refusing "
                  "wrong-team offerings. --names all4 removes it.")
        else:
            print("  -> parity is NOT the main story. Check the |dt| "
                  "table above: if OTHER sits far from any contact, "
                  "those are junk frames (a DEAD-escape miss, not a "
                  "recognition failure).")
    if per_rally:
        print("\nper-rally OTHER split (off/on offered team):")
        for cum in sorted(per_rally):
            c = per_rally[cum]
            print(f"  r{cum:<4} off={c['off']:<3} on={c['on']}")


def run_api(decoded, truths, names, video, model, out_dir, limit, width,
            mode="team"):
    from vlm_frame_sample import cut_strip
    from vlm_tier_test import PRICE, image_media_type
    import base64
    import anthropic
    client = anthropic.Anthropic()

    def ask(img, offered):
        b64 = base64.standard_b64encode(Path(img).read_bytes()).decode()
        resp = client.messages.create(
            model=model, max_tokens=256,
            tools=[escape_tool(offered)],
            tool_choice={"type": "tool", "name": "call_shot"},
            messages=[{"role": "user", "content": [
                {"type": "image",
                 "source": {"type": "base64",
                            "media_type": image_media_type(img),
                            "data": b64}},
                {"type": "text", "text": escape_prompt(offered)},
            ]}],
        )
        call = next(b for b in resp.content if b.type == "tool_use")
        return call.input["answer"], resp.usage

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    rows, done = [], 0
    answers, parity = Counter(), Counter()
    o_ok = o_n = t_ok = t_n = 0
    ok_by_bin, n_by_bin = Counter(), Counter()
    touch_pipe, touch_true = Counter(), Counter()
    dead_by_rally = Counter()
    in_tok = out_tok = 0
    for cum in sorted(decoded):
        d = decoded[cum]
        tbt = split_truth_by_team(truths[cum], d["teams"], names)
        events = d["events"]
        opairs, _extra, _missing = order_join(events, tbt)
        omap = {}
        for t_dec, team, k, t_true, name_true, dt in opairs:
            omap.setdefault((t_dec, team), (t_true, name_true, dt))
        tmap = time_join(events, tbt)
        for t, name in truths[cum]:
            touch_true[name] += 1
        for i, (t_dec, team) in enumerate(events):
            if limit and done >= limit:
                break
            done += 1
            own = team_names(d["teams"], names, team)
            offered = (own if mode == "team" else
                       sorted(own + team_names(d["teams"], names, team ^ 1)))
            img = out / f"r{cum:03d}_e{i:02d}_t{t_dec:07.2f}.png"
            if not img.exists():
                cut_strip(video, t_dec, img, width)
            ans, usage = ask(img, offered)
            in_tok += usage.input_tokens
            out_tok += usage.output_tokens
            answers[ans] += 1
            named = ans not in (OTHER, DEAD)
            if named:
                touch_pipe[ans] += 1
            if ans == DEAD:
                dead_by_rally[cum] += 1
            o_truth = omap.get((t_dec, team))
            t_truth = tmap.get(i)
            if o_truth and named:
                _tt, name_true, dt = o_truth
                hit = ans == name_true
                o_ok += hit
                o_n += 1
                b = dt_bin(dt)
                ok_by_bin[b] += hit
                n_by_bin[b] += 1
            if t_truth and named:
                t_ok += ans == t_truth[1]
                t_n += 1
            # PARITY CHECK (all4 only): the decoder said this event
            # belonged to `team`; the model named someone. Agreement is
            # now measurable per event instead of assumed — and a
            # disagreement rate near 50% would mean the decoder's team
            # assignment is noise, which the 2-name mode could only see
            # as an unexplained pile of OTHER answers.
            if mode == "all4" and named:
                parity[("agree" if ans in own else "flip")] += 1
            rows.append([
                cum, i, team, f"{t_dec:.3f}", ans,
                "|".join(offered),
                o_truth[1] if o_truth else "",
                f"{o_truth[2]:.3f}" if o_truth else "",
                t_truth[1] if t_truth else "",
            ])

    calls_csv = out / "join_pilot_calls.csv"
    with open(calls_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rally_cum", "event", "team", "t_decoded", "answer",
                    "offered", "order_truth", "order_dt", "time_truth"])
        w.writerows(rows)

    i_, o_ = PRICE[model]
    print(f"\nPARTNER ON DECODED FRAMES ({model}, {done} events)")
    print(f"  answers: " + ", ".join(f"{k} x{v}"
                                     for k, v in answers.most_common()))
    if o_n:
        print(f"  order-join accuracy {o_ok}/{o_n} = {o_ok / o_n:.0%}  "
              f"(named answers only; true-frame benchmark 90%, chance 50%)")
        for lo, hi in DT_BINS:
            b = dt_bin(lo)
            if n_by_bin[b]:
                print(f"    |dt| {b:>9}: {ok_by_bin[b]}/{n_by_bin[b]} = "
                      f"{ok_by_bin[b] / n_by_bin[b]:.0%}")
    if t_n:
        print(f"  time-join accuracy  {t_ok}/{t_n} = {t_ok / t_n:.0%}  "
              f"(nearest same-team truth within {TIME_TOL}s — reads "
              f"through defect C's ordinal slips)")
    if parity:
        tot = sum(parity.values())
        print(f"  DECODER PARITY: {parity['agree']}/{tot} = "
              f"{parity['agree'] / tot:.0%} of named calls landed on the "
              f"team the decoder assigned (50% would mean the team "
              f"assignment carries no information)")
    heavy = [c for c, k in dead_by_rally.items()
             if k >= max(2, len(decoded[c]['events']) // 2)]
    if heavy:
        print(f"  DEAD-heavy rallies (re-decode candidates): "
              f"{sorted(heavy)}")
    print(f"  spend ${in_tok * i_ / 1e6 + out_tok * o_ / 1e6:.3f} "
          f"({in_tok} in / {out_tok} out tok)")
    print(f"\nTOUCH COUNTS (product metric; named answers only)")
    for name in sorted(set(touch_true) | set(touch_pipe)):
        tp, tt = touch_pipe[name], touch_true[name]
        print(f"  {name:<22} pipeline {tp:>3}  true {tt:>3}  "
              f"delta {tp - tt:+d}")
    print(f"\ncalls -> {calls_csv} (re-scorable without re-spending)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--windows", default=WINDOWS_V4)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--pose-dir", default=POSE_DIR)
    ap.add_argument("--video", help="needed for the API stage")
    ap.add_argument("--model", default="claude-sonnet-5",
                    help="the tier the partner test picked")
    ap.add_argument("--out-dir", default="vlm_join")
    ap.add_argument("--width", type=int, default=1280,
                    help="per-frame strip width, same as the 90%% test")
    ap.add_argument("--limit", type=int,
                    help="cap API calls for a smoke run")
    ap.add_argument("--names", choices=["team", "all4"], default="all4",
                    help="team = offer the decoded team's 2 players "
                         "(2026-08-21 run: 73/186 answered OTHER); "
                         "all4 = offer every player and CHECK the "
                         "decoder's parity instead of assuming it")
    ap.add_argument("--dry", action="store_true",
                    help="decode + join only; no video, no API, free")
    ap.add_argument("--rescore",
                    help="path to an existing join_pilot_calls.csv: "
                         "runs the parity diagnostic on calls already "
                         "paid for; no API, no video, free")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.rescore:
        return rescore(a.rescore, a.labels, a.windows, a.split)

    train = train_only(a.split)
    names = name_map(a.labels)
    truths = true_hitters(a.labels, train)
    decoded = decode_train_rallies(a.labels, a.windows, a.pose_dir, train)
    truths = {c: truths[c] for c in decoded}
    print(f"{len(decoded)} train rallies decoded "
          f"(r9/r10 excluded per contact_gate.md)\n")
    print_dry(decoded, truths, names)
    if a.dry:
        return
    if not a.video:
        raise SystemExit("\n--video required for the API stage "
                         "(or pass --dry)")
    run_api(decoded, truths, names, a.video, a.model, a.out_dir,
            a.limit, a.width, a.names)


def selftest():
    """Join + touch math only — no pose, no numpy, no API."""
    names = {"ua": "Ann", "ub": "Bea", "uc": "Cal", "ud": "Dee"}
    teams = (frozenset({"ua", "ub"}), frozenset({"uc", "ud"}))
    # truth: A-team serve, strict alternation, 5 contacts
    hitters = [(10.0, "Ann"), (11.0, "Cal"), (11.8, "Bea"),
               (12.5, "Dee"), (13.2, "Ann")]
    tbt = split_truth_by_team(hitters, teams, names)
    assert [n for _t, n in tbt[0]] == ["Ann", "Bea", "Ann"]
    assert [n for _t, n in tbt[1]] == ["Cal", "Dee"]
    assert team_names(teams, names, 0) == ["Ann", "Bea"]

    # decoder: right counts, sloppy times -> all joined, dt as expected
    events = [(10.1, 0), (11.4, 1), (11.75, 0), (12.9, 1), (13.2, 0)]
    pairs, extra, missing = order_join(events, tbt)
    assert len(pairs) == 5 and not extra and not missing
    assert abs(pairs[1][5] - 0.4) < 1e-9        # 11.4 vs 11.0
    assert pairs[2][4] == "Bea"                 # k=1 of team 0

    # decoder over-counts team 1 and misses team 0's last touch
    events = [(10.1, 0), (11.4, 1), (12.9, 1), (13.5, 1)]
    pairs, extra, missing = order_join(events, tbt)
    assert len(pairs) == 3 and len(extra) == 1 and len(missing) == 2
    assert extra[0][1] == 1 and extra[0][2] == 2
    assert {(m[0], m[1]) for m in missing} == {(0, 1), (0, 2)}

    # dt binning covers the line
    assert dt_bin(0.0) == "0.0-0.2s" and dt_bin(0.35) == "0.2-0.5s"
    assert dt_bin(0.5) == ">0.5s" and dt_bin(4.0) == ">0.5s"

    # trailing trim: junk after last true contact + pad drops; play stays
    ev = [(10.1, 0), (11.4, 1), (13.3, 0), (15.5, 1), (16.8, 0)]
    kept, n_trail = trim_trailing(ev, last_true_t=13.2)
    assert n_trail == 2 and [t for t, _s in kept] == [10.1, 11.4, 13.3]

    # time-join: greedy nearest within tol, one-to-one, same team only
    tj = time_join([(10.15, 0), (10.3, 0), (11.35, 1), (99.0, 1)], tbt)
    assert tj[0] == (10.0, "Ann")        # nearest claims it
    assert 1 not in tj                    # one-to-one: 10.3 loses to 10.15
    assert tj[2] == (11.0, "Cal")
    assert 3 not in tj                    # out of tolerance

    # escape tool: offered names + both escapes, nothing else
    et = escape_tool(["Ann", "Bea"])
    assert et["input_schema"]["properties"]["answer"]["enum"] == \
        ["Ann", "Bea", OTHER, DEAD]
    assert OTHER in escape_prompt(["Ann", "Bea"])
    # all4 form: every player offered, escapes still present
    e4 = escape_tool(["Ann", "Bea", "Cal", "Dee"])
    assert e4["input_schema"]["properties"]["answer"]["enum"] == \
        ["Ann", "Bea", "Cal", "Dee", OTHER, DEAD]
    p4 = escape_prompt(["Ann", "Bea", "Cal", "Dee"])
    assert all(n in p4 for n in ("Ann", "Bea", "Cal", "Dee")), p4
    # THE TEST THAT WOULD HAVE CAUGHT THE VACUOUS v1 DIAGNOSTIC: the
    # parity check must be able to return OFF. v1 read truth from the
    # same-team-constrained join columns, so OFF was unreachable and it
    # reported a guaranteed 0. Any future rewrite has to keep this
    # property: a decoded event whose nearest real contact belongs to
    # the OTHER team must be classifiable as such.
    rally_truth = [(10.0, "Ann"), (11.0, "Cal"), (11.8, "Bea")]
    teams_by_name = {"Ann": 0, "Bea": 0, "Cal": 1, "Dee": 1}
    for t_dec, team, want_on in ((10.05, 0, True),    # Ann, offered 0
                                 (11.05, 0, False),   # Cal, offered 0
                                 (11.05, 1, True)):   # Cal, offered 1
        _tt, tname = min(rally_truth, key=lambda x: abs(x[0] - t_dec))
        assert (teams_by_name[tname] == team) is want_on, (t_dec, tname)
    print("selftest OK: team split, order join, extras/missing, dt bins, "
          "trailing trim, time join, escape tool, team-blind parity "
          "check can return OFF")


if __name__ == "__main__":
    main()
