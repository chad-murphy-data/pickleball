"""VLM tier/cost test — runs the hitter-call test through the REAL API
across model price tiers and scores each against the same answer key.

Fills the gap between two existing scripts: vlm_pack.py prices a grid
scan (tokens/cost only, no accuracy) and vlm_score.py scored one
MANUAL in-thread run (accuracy only, whatever model happened to be
serving that session, never recorded). Neither answers the actual
question (2026-08-21, user): do we need the top tier, or does a
cheaper one hold the same accuracy on the channel that already works
(95% side / 85% four-way / 75% pace, vlm_score.py 2026-08-19)?

Two modes, --mode side-pace (default) or --mode partner:

  side-pace: position (4-way) + pace, scored down to SIDE (near/far)
  against a --near-team/--far-team split. This is NOT the touch-share
  question — side is already known for free from exact alternation
  (0 violations / 229 contacts, no VLM needed), so this mode only
  re-derives something the project already has.

  partner: the actual touch-share call (2026-08-21, user). Side is
  taken as GIVEN (same as production would have it from alternation)
  and the model only picks WHICH of the two named players on that
  known side hit the shot — the ~89%-in-one-manual-test question from
  STATUS.md's "Buildable now: touch share" section. Requires each
  team to have exactly two members.

Both modes take a --near-team/--far-team split identical to
vlm_score.py's hardcoded UTAH/CHI, parameterized here so the script
isn't tied to one match. Pass whichever match's rosters you're
testing against.

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


def partner_tool(name_a, name_b):
    """Dynamic tool: the enum is the two named players on the ALREADY-
    KNOWN side, not a fixed 4-way position. This is the touch-share
    call — side is given for free from alternation, so the only open
    question per shot is which partner hit it."""
    return {
        "name": "call_partner",
        "description": "Report which of the two named players hit the shot shown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hitter": {"type": "string", "enum": [name_a, name_b]},
            },
            "required": ["hitter"],
            "additionalProperties": False,
        },
    }


def partner_prompt(name_a, name_b):
    return (
        "This is a strip of frames from a pro pickleball broadcast "
        "centered on one shot. The shot was hit by one of two "
        f"partners on the same side of the court: {name_a} or "
        f"{name_b}. Call which one hit it, using the call_partner tool."
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


def call_partner(client, model, img_path, name_a, name_b):
    b64 = base64.standard_b64encode(Path(img_path).read_bytes()).decode()
    tool = partner_tool(name_a, name_b)
    resp = client.messages.create(
        model=model, max_tokens=256,
        tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                          "media_type": image_media_type(img_path),
                                          "data": b64}},
            {"type": "text", "text": partner_prompt(name_a, name_b)},
        ]}],
    )
    call = next(b for b in resp.content if b.type == "tool_use")
    return call.input["hitter"], resp.usage


def side_team(truth_name, near_team, far_team):
    """The roster (near_team or far_team) truth_name actually belongs
    to. Rosters don't swap mid-match — only which screen side (near/
    far) a roster appears on does, and partner mode never asks about
    screen side, so no rally/switched argument is needed here (that
    swap only matters to side-pace mode's near/far scoring)."""
    if truth_name in near_team:
        return near_team
    if truth_name in far_team:
        return far_team
    return None


def score_partner_model(client, model, rows, img_dir, near_team, far_team):
    """The real touch-share test: side given, model picks between the
    two named players on it. Requires exactly 2 players/side. No
    --switched needed — see side_team's docstring."""
    ok = n = 0
    in_tok = out_tok = 0
    errors = []
    for r in rows:
        qfile = r.get("question") or r.get("window")
        img_path = Path(img_dir) / qfile
        if not img_path.exists():
            errors.append(f"missing {img_path}")
            continue
        truth_name = r["hitter_name"]
        team = side_team(truth_name, near_team, far_team)
        if not team or len(team) != 2:
            errors.append(f"{qfile}: hitter {truth_name!r} not in a "
                          f"2-player team (got {team})")
            continue
        name_a, name_b = sorted(team)

        called, usage = call_partner(client, model, img_path, name_a, name_b)
        in_tok += usage.input_tokens
        out_tok += usage.output_tokens
        ok += called == truth_name
        n += 1
    i, o = PRICE[model]
    actual_cost = in_tok * i / 1e6 + out_tok * o / 1e6
    return {
        "n": n, "partner": ok / n if n else 0.0,
        "in_tok": in_tok, "out_tok": out_tok,
        "cost": actual_cost, "errors": errors,
    }


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
    ap.add_argument("--mode", choices=["side-pace", "partner"],
                     default="side-pace",
                     help="side-pace = 4-way position + pace, scored to "
                     "side (re-derives what alternation already gives "
                     "for free); partner = the real touch-share call, "
                     "side given, which of 2 named players hit it")
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

    print(f"mode={a.mode}  scoring {len(rows)} questions x {len(models)} "
          f"models ({len(rows) * len(models)} API calls)\n")

    results = {}
    for model in models:
        t0 = time.time()
        if a.mode == "partner":
            res = score_partner_model(client, model, rows, a.dir, near_team,
                                       far_team)
        else:
            res = score_model(client, model, rows, a.dir, near_team,
                               far_team, switched)
        res["wall_s"] = time.time() - t0
        results[model] = res
        extrap = per_match_extrapolation(model)
        if a.mode == "partner":
            print(f"{model:<22} n={res['n']:<4} "
                  f"partner={res['partner']:.0%}   "
                  f"actual ${res['cost']:.4f} on this test "
                  f"({res['in_tok']} in / {res['out_tok']} out tok)   "
                  f"~${extrap:.2f}/match at 3x3-grid scale (batch)")
        else:
            print(f"{model:<22} n={res['n']:<4} "
                  f"side={res['side']:.0%}  pace={res['pace']:.0%}   "
                  f"actual ${res['cost']:.4f} on this test "
                  f"({res['in_tok']} in / {res['out_tok']} out tok)   "
                  f"~${extrap:.2f}/match at 3x3-grid scale (batch)")
        if res["errors"]:
            print(f"  {len(res['errors'])} skipped, e.g. {res['errors'][0]}")

    if len(results) > 1:
        key_fn = ((lambda m: results[m]["partner"]) if a.mode == "partner"
                  else (lambda m: results[m]["side"] + results[m]["pace"]))
        best = max(results, key=key_fn)
        cheapest = min(results, key=lambda m: PRICE[m][0])
        if best != cheapest:
            drop = key_fn(best) - key_fn(cheapest)
            unit = "partner" if a.mode == "partner" else "side+pace"
            print(f"\n{cheapest} vs {best}: {unit} {drop:+.0%} — weigh "
                  f"that drop against the "
                  f"${per_match_extrapolation(best) - per_match_extrapolation(cheapest):.2f}/match "
                  f"you'd save.")


def selftest():
    """No API key needed: checks the accounting math and the partner-
    mode side lookup/tool construction — the parts that don't need a
    live call."""
    assert PRICE["claude-sonnet-5"] == (3.0, 15.0)
    extrap = per_match_extrapolation("claude-sonnet-5")
    assert 8.5 < extrap < 9.1, extrap   # matches STATUS.md's ~$8.87 "mid tier"
    extrap_h = per_match_extrapolation("claude-haiku-4-5")
    assert 2.9 < extrap_h < 3.0, extrap_h  # matches ~$2.96 "small tier"
    print(f"selftest OK: sonnet ${extrap:.2f}/match, haiku ${extrap_h:.2f}/match "
          f"(3x3 grid, 47 min rally time, batch) — matches STATUS.md's "
          f"mid/small tiers")

    near_team = {"Alshon", "Black"}
    far_team = {"Patriquin", "Bright"}
    # roster lookup — a player's team doesn't change when ends swap,
    # so this needs no rally/switched argument at all
    assert side_team("Alshon", near_team, far_team) == near_team
    assert side_team("Black", near_team, far_team) == near_team
    assert side_team("Bright", near_team, far_team) == far_team
    # a name in neither roster is caught, not silently mis-teamed
    assert side_team("Nobody", near_team, far_team) is None

    tool = partner_tool("Alshon", "Black")
    assert tool["input_schema"]["properties"]["hitter"]["enum"] == \
        ["Alshon", "Black"]
    assert "Alshon" in partner_prompt("Alshon", "Black")
    print("selftest OK: side_team roster lookup, partner tool/prompt "
          "construction")


if __name__ == "__main__":
    main()
