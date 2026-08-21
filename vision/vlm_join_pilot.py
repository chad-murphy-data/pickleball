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
            variable is the timestamp source), calls the partner tool,
            and reports: partner accuracy on joined pairs (compare to
            90%), accuracy binned by |dt|, and the product metric —
            per-player touch counts, pipeline vs truth. Calls CSV is
            written next to the strips so nothing needs re-spending.

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


def print_dry(decoded, truths, names):
    all_dt, tot_true, tot_dec, tot_ghost = [], 0, 0, 0
    print(f"{'rally':>5} {'true':>5} {'dec':>4} {'ghost':>5} "
          f"{'extra':>5} {'miss':>4}  joined |dt|: med / p90 / frac<=0.2s")
    for cum in sorted(decoded):
        d = decoded[cum]
        tbt = split_truth_by_team(truths[cum], d["teams"], names)
        pairs, extra, missing = order_join(d["events"], tbt)
        dts = sorted(p[5] for p in pairs)
        all_dt += dts
        n_true = sum(len(v) for v in tbt.values())
        tot_true += n_true
        tot_dec += len(d["events"])
        tot_ghost += d["ghosts"]
        med = dts[len(dts) // 2] if dts else float("nan")
        p90 = dts[int(0.9 * len(dts))] if dts else float("nan")
        fr = (sum(1 for x in dts if x <= 0.2) / len(dts)) if dts else 0.0
        print(f"{cum:>5} {n_true:>5} {len(d['events']):>4} "
              f"{d['ghosts']:>5} {len(extra):>5} {len(missing):>4}  "
              f"{med:>7.2f} / {p90:.2f} / {fr:.0%}")
    all_dt.sort()
    n = len(all_dt)
    print(f"\nTOTAL {tot_true} true / {tot_dec} decoded "
          f"(+{tot_ghost} ghosts) — {n} joined pairs")
    if n:
        print(f"|dt|  median {all_dt[n // 2]:.2f}s   "
              f"p90 {all_dt[int(0.9 * n)]:.2f}s")
        for lo, hi in DT_BINS:
            c = sum(1 for x in all_dt if lo <= x < hi)
            print(f"      {dt_bin(lo):>9}: {c:>4}  ({c / n:.0%})")
        near = sum(1 for x in all_dt if x <= 0.2) / n
        print(f"\nA strip spans t+/-0.1s; |dt|<=0.2s means the true "
              f"contact is inside or adjacent to it ({near:.0%} of "
              f"pairs). The API stage measures whether the VLM's call "
              f"survives the rest — this table only predicts it.")


def run_api(decoded, truths, names, video, model, out_dir, limit, width):
    from vlm_frame_sample import cut_strip
    from vlm_tier_test import PRICE, call_partner
    import anthropic
    client = anthropic.Anthropic()

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    rows, done = [], 0
    ok_by_bin, n_by_bin = Counter(), Counter()
    touch_pipe, touch_true = Counter(), Counter()
    in_tok = out_tok = ok = n = 0
    for cum in sorted(decoded):
        d = decoded[cum]
        tbt = split_truth_by_team(truths[cum], d["teams"], names)
        pairs, extra, _missing = order_join(d["events"], tbt)
        for t, name in truths[cum]:
            touch_true[name] += 1
        for t_dec, team, k, t_true, name_true, dt in pairs:
            if limit and done >= limit:
                break
            done += 1
            na, nb = team_names(d["teams"], names, team)
            img = out / f"r{cum:03d}_k{k:02d}_t{t_dec:07.2f}.png"
            if not img.exists():
                cut_strip(video, t_dec, img, width)
            called, usage = call_partner(client, model, img, na, nb)
            in_tok += usage.input_tokens
            out_tok += usage.output_tokens
            hit = called == name_true
            ok += hit
            n += 1
            b = dt_bin(dt)
            ok_by_bin[b] += hit
            n_by_bin[b] += 1
            touch_pipe[called] += 1
            rows.append([cum, k, team, f"{t_dec:.3f}", f"{t_true:.3f}",
                         f"{dt:.3f}", called, name_true, int(hit)])
        # extras still produce touches in production — attribute them too
        for t_dec, team, k in extra:
            if limit and done >= limit:
                break
            done += 1
            na, nb = team_names(d["teams"], names, team)
            img = out / f"r{cum:03d}_x{k:02d}_t{t_dec:07.2f}.png"
            if not img.exists():
                cut_strip(video, t_dec, img, width)
            called, usage = call_partner(client, model, img, na, nb)
            in_tok += usage.input_tokens
            out_tok += usage.output_tokens
            touch_pipe[called] += 1
            rows.append([cum, k, team, f"{t_dec:.3f}", "", "", called,
                         "", ""])

    calls_csv = out / "join_pilot_calls.csv"
    with open(calls_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rally_cum", "k", "team", "t_decoded", "t_true",
                    "dt", "called", "truth", "ok"])
        w.writerows(rows)

    i, o = PRICE[model]
    print(f"\nPARTNER ON DECODED FRAMES ({model}, {n} joined pairs)")
    print(f"  accuracy {ok}/{n} = {ok / n:.0%}    "
          f"(true-frame benchmark: 90%; chance: 50%)")
    for lo, hi in DT_BINS:
        b = dt_bin(lo)
        if n_by_bin[b]:
            print(f"  |dt| {b:>9}: {ok_by_bin[b]}/{n_by_bin[b]} = "
                  f"{ok_by_bin[b] / n_by_bin[b]:.0%}")
    print(f"  spend ${in_tok * i / 1e6 + out_tok * o / 1e6:.3f} "
          f"({in_tok} in / {out_tok} out tok)")
    print(f"\nTOUCH COUNTS (product metric, pipeline vs truth)")
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
    ap.add_argument("--dry", action="store_true",
                    help="decode + join only; no video, no API, free")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

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
            a.limit, a.width)


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
    print("selftest OK: team split, order join, extras/missing, dt bins")


if __name__ == "__main__":
    main()
