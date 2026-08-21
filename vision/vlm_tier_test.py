"""VLM tier/cost test — runs the hitter-call test through the REAL API
across model price tiers and scores each against the same answer key.

Fills the gap between two existing scripts: vlm_pack.py prices a grid
scan (tokens/cost only, no accuracy) and vlm_score.py scored one
MANUAL in-thread run (accuracy only, whatever model happened to be
serving that session, never recorded). Neither answers the actual
question (2026-08-21, user): do we need the top tier, or does a
cheaper one hold the same accuracy on the channel that already works
(95% side / 85% four-way / 75% pace, vlm_score.py 2026-08-19)?

This scores SIDE and PACE only (not full 4-way hitter identity) —
that needs a --near-team/--far-team split identical to vlm_score.py's
hardcoded UTAH/CHI, parameterized here so the script isn't tied to one
match. Pass whichever match's rosters you're testing against.

Needs the q*.png strips + ANSWER_KEY.csv that vlm_frame_sample.py
writes (gitignored, video-derived) — run wherever that video lives,
not in a footage-less session.

Usage:
    pip install anthropic
    python3 vlm_tier_test.py --dir vlm_test --key vlm_test/ANSWER_KEY.csv \\
        --near-team "Allyce Jones,Etta Tuionetoa" \\
        --far-team "Emma Nelson,Ting Chieh Wei" --switched 16,19 \\
        --models claude-haiku-4-5,claude-sonnet-5,claude-opus-5

    python3 vlm_tier_test.py --selftest      (no API key, no images needed)

Reports, per model: side/pace accuracy vs the key, ACTUAL dollars spent
(from response.usage, not an estimate), and the $/match this would
extrapolate to at vlm_pack.py's 3x3-grid, 47-min-of-rally-time rate —
the same basis as the $44/$8/$3 figures in STATUS.md, so the two are
directly comparable.
"""
import argparse
import base64
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vlm_pack import cost_table  # noqa: E402

TOOL = {
    "name": "call_shot",
    "description": "Report the court position and pace of the hitter shown.",
    "input_schema": {
        "type": "object",
        "properties": {
            "position": {"type": "string",
                         "enum": ["near-left", "near-right",
                                  "far-left", "far-right"]},
            "pace": {"type": "string", "enum": ["fast", "slow"]},
        },
        "required": ["position", "pace"],
        "additionalProperties": False,
    },
}

PROMPT = (
    "This is a strip of frames from a pro pickleball broadcast centered "
    "on one shot. Camera view: the NEAR court is the bottom/foreground, "
    "the FAR court is the top/background. Call the HITTER's court "
    "position (near-left / near-right / far-left / far-right) and "
    "whether the shot's pace was fast or slow. Use the call_shot tool."
)

# $/M tokens, list rate (input, output) — 2026-06-24 cached pricing.
# Batch halves both; this script uses the synchronous Messages API, so
# report list-rate actual spend and note the batch-equivalent alongside.
PRICE = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
}


def load_key(path):
    return list(csv.DictReader(open(path)))


def image_media_type(path):
    ext = Path(path).suffix.lower()
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp"}.get(ext.lstrip("."), "image/png")


def call_model(client, model, img_path):
    b64 = base64.standard_b64encode(Path(img_path).read_bytes()).decode()
    resp = client.messages.create(
        model=model, max_tokens=256,
        tools=[TOOL], tool_choice={"type": "tool", "name": "call_shot"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                          "media_type": image_media_type(img_path),
                                          "data": b64}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    call = next(b for b in resp.content if b.type == "tool_use")
    return call.input["position"], call.input["pace"], resp.usage


def score_model(client, model, rows, img_dir, near_team, far_team, switched):
    """Mirrors vlm_score.py's side/pace matching, generalized to any
    two-team roster and any switched-rally set."""
    side_ok = pace_ok = n = 0
    in_tok = out_tok = 0
    errors = []
    for r in rows:
        qfile = r.get("question") or r.get("window")
        img_path = Path(img_dir) / qfile
        if not img_path.exists():
            errors.append(f"missing {img_path}")
            continue
        rally = int(r.get("rally_cum") or r.get("rally"))
        truth_name = r["hitter_name"]
        truth_pace = r["pace"]
        near_set = far_team if rally in switched else near_team
        truth_near = truth_name in near_set

        pos, pace_call, usage = call_model(client, model, img_path)
        in_tok += usage.input_tokens
        out_tok += usage.output_tokens
        called_near = pos.startswith("near")
        side_ok += called_near == truth_near
        pace_ok += pace_call == truth_pace
        n += 1
    i, o = PRICE[model]
    actual_cost = in_tok * i / 1e6 + out_tok * o / 1e6
    return {
        "n": n, "side": side_ok / n if n else 0.0,
        "pace": pace_ok / n if n else 0.0,
        "in_tok": in_tok, "out_tok": out_tok,
        "cost": actual_cost, "errors": errors,
    }


def per_match_extrapolation(model, rally_minutes=47):
    """$/match at the vlm_pack.py 3x3-grid token rate for THIS model's
    price, so it lines up with the $44/$8/$3 figures already on record."""
    i, o = PRICE[model]
    rows = cost_table(rally_minutes * 60, tiers={"m": (i, o)}, batch=True)
    n3x3 = rows[0]  # n=3 is first row
    return n3x3[-1]["m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="folder holding the question images")
    ap.add_argument("--key", help="ANSWER_KEY.csv from vlm_frame_sample.py")
    ap.add_argument("--near-team", default="", help="comma-separated names, "
                    "the NEAR-camera team in un-switched rallies")
    ap.add_argument("--far-team", default="")
    ap.add_argument("--switched", default="", help="comma-separated rally_cum "
                     "values where ends swapped (MLP switches at 6)")
    ap.add_argument("--models", default="claude-haiku-4-5,claude-sonnet-5,"
                     "claude-opus-5",
                     help="comma-separated model IDs to compare")
    ap.add_argument("--limit", type=int, help="only score the first N rows "
                     "(cheap smoke test before spending on the full set)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if not (a.dir and a.key):
        raise SystemExit("--dir and --key are required (or --selftest)")

    import anthropic
    client = anthropic.Anthropic()

    near_team = set(x.strip() for x in a.near_team.split(",") if x.strip())
    far_team = set(x.strip() for x in a.far_team.split(",") if x.strip())
    switched = set(int(x) for x in a.switched.split(",") if x.strip())
    rows = load_key(a.key)
    if a.limit:
        rows = rows[:a.limit]
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    print(f"scoring {len(rows)} questions x {len(models)} models "
          f"({len(rows) * len(models)} API calls)\n")

    results = {}
    for model in models:
        t0 = time.time()
        res = score_model(client, model, rows, a.dir, near_team, far_team,
                           switched)
        res["wall_s"] = time.time() - t0
        results[model] = res
        extrap = per_match_extrapolation(model)
        print(f"{model:<22} n={res['n']:<4} "
              f"side={res['side']:.0%}  pace={res['pace']:.0%}   "
              f"actual ${res['cost']:.4f} on this test "
              f"({res['in_tok']} in / {res['out_tok']} out tok)   "
              f"~${extrap:.2f}/match at 3x3-grid scale (batch)")
        if res["errors"]:
            print(f"  {len(res['errors'])} missing images, e.g. "
                  f"{res['errors'][0]}")

    if len(results) > 1:
        best = max(results, key=lambda m: results[m]["side"] + results[m]["pace"])
        cheapest = min(results, key=lambda m: PRICE[m][0])
        if best != cheapest:
            drop_side = results[best]["side"] - results[cheapest]["side"]
            drop_pace = results[best]["pace"] - results[cheapest]["pace"]
            print(f"\n{cheapest} vs {best}: side {drop_side:+.0%}, "
                  f"pace {drop_pace:+.0%} — weigh that drop against the "
                  f"${per_match_extrapolation(best) - per_match_extrapolation(cheapest):.2f}/match "
                  f"you'd save.")


def selftest():
    """No API key needed: checks the accounting math only."""
    assert PRICE["claude-sonnet-5"] == (3.0, 15.0)
    extrap = per_match_extrapolation("claude-sonnet-5")
    assert 8.5 < extrap < 9.1, extrap   # matches STATUS.md's ~$8.87 "mid tier"
    extrap_h = per_match_extrapolation("claude-haiku-4-5")
    assert 2.9 < extrap_h < 3.0, extrap_h  # matches ~$2.96 "small tier"
    print(f"selftest OK: sonnet ${extrap:.2f}/match, haiku ${extrap_h:.2f}/match "
          f"(3x3 grid, 47 min rally time, batch) — matches STATUS.md's "
          f"mid/small tiers")


if __name__ == "__main__":
    main()
