"""value_cap/price_list.py -- the shipped price list, one table, with the
doubles-only value next to the price so the DreamBreaker (singles) lift
is visible.

    python value_cap/price_list.py            # -> price_list.csv + price_list.md

Shipped rule (2026-09-04): alpha = 1, one joint $20M pool, $30k floor,
Anna Leigh Waters franchise-tagged (`phase2_pricing.prices_tagged`).
Columns: price (the tag list), curve price (untagged, for comparison),
phi (the value the price is proportional to), doubles value + rank within
gender (v2 `value_now_mean`, the per-point logit), singles value + games
(singles suite; 0 games = imputed from doubles, wide error bar).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.argv = [sys.argv[0]]                       # phase2_pricing parses argv on import
from phase2_pricing import DOUBLES, NAME, pid_named, prices, prices_tagged  # noqa: E402
from pool import load_pool  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TAG = "Anna Leigh Waters"


def singles_table():
    out = {}
    for r in csv.DictReader((DATA / "singles_players.csv").open()):
        out[r["player_id"]] = (float(r["singles_value"]), int(r["singles_games"]))
    return out


def build(alpha=1.0, floor=30_000):
    pool = load_pool("phi")
    tag = pid_named(TAG)
    curve = prices(pool, alpha, "joint", floor)
    price = prices_tagged(pool, alpha, tag, "joint", floor)
    phi = {pid: v for g in pool for pid, _, v in pool[g]}
    sing = singles_table()
    # doubles rank within gender over EVERY tracked player, not just the pool
    drank = {}
    for g in ("M", "F"):
        order = sorted((u for u in DOUBLES if DOUBLES[u]["gender"] == g), key=lambda u: -DOUBLES[u]["v"])
        drank.update({u: i + 1 for i, u in enumerate(order)})
    rows = []
    for u in sorted(price, key=lambda u: -price[u]):
        sv, sg = sing.get(u, (None, 0))
        rows.append({
            "player_id": u, "full_name": NAME[u], "gender": DOUBLES[u]["gender"],
            "price": round(price[u]), "curve_price": round(curve[u]), "phi": round(phi[u], 4),
            "doubles_value": DOUBLES[u]["v"], "doubles_rank": drank[u],
            "singles_value": None if sv is None else round(sv, 3), "singles_games": sg,
            "tagged": int(u == tag),
        })
    return rows, alpha, floor, price[tag], curve[tag]


def render(rows, alpha, floor, tag_price, tag_curve):
    n_f = sum(r["gender"] == "F" for r in rows)
    women_share = sum(r["price"] for r in rows if r["gender"] == "F") / sum(r["price"] for r in rows)
    lines = ["# The price list (shipped rule)", "",
             f"alpha = {alpha:g}, one joint $20M pool over the self-consistent top-{n_f}-per-gender phi pool, "
             f"${floor/1e3:,.0f}k floor, **{TAG} franchise-tagged at ${tag_price/1e3:,.0f}k** (her curve price is "
             f"${tag_curve/1e3:,.0f}k, which no legal roster can carry; the tag is cap minus the cheapest legal "
             f"completion -- the two cheapest priced women plus the three cheapest priced men -- and the "
             f"${(tag_curve-tag_price)/1e3:,.0f}k gap is spread over the other {len(rows)-1} players in proportion "
             f"to their value, so the pool still sums to 20 caps). Women's share of the pool {100*women_share:.1f}%. "
             f"Built by `price_list.py`; full table in `price_list.csv`.", "",
             "**Read the doubles column next to the price.** Price is proportional to phi, a player's average "
             "contribution to winning an MLP tie (four doubles games plus a DreamBreaker at 2-2, which is "
             "singles). Where a player's price rank sits above their doubles rank, the DreamBreaker channel "
             "is doing it: Parris Todd and Kate Fahey price above women who are a hair better in doubles "
             "because they are the two best singles players in the women's field after Waters; Christopher "
             "Haworth prices into the pool on singles alone. Singles with 0 games = imputed from doubles, wide "
             "error bar (that is why Jade Kawamoto and Tina Pisnik sit below Todd and Fahey here).", ""]

    def table(sub, title):
        out = [f"## {title}", "", "| # | player | price | curve price | phi | doubles value | doubles rank | singles value | singles games |",
               "|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(sub, 1):
            sv = "--" if r["singles_value"] is None else f"{r['singles_value']:+.2f}"
            out.append(f"| {i} | {r['full_name']}{' (tag)' if r['tagged'] else ''} | ${r['price']/1e3:,.0f}k | "
                       f"${r['curve_price']/1e3:,.0f}k | {r['phi']:.3f} | {r['doubles_value']:+.2f} | "
                       f"#{r['doubles_rank']}{r['gender']} | {sv} | {r['singles_games']} |")
        return out + [""]
    lines += table(rows[:10], "Top 10 overall")
    lines += table([r for r in rows if r["gender"] == "F"], "Women (all priced)")
    lines += table([r for r in rows if r["gender"] == "M"], "Men (all priced)")
    return "\n".join(lines)


def main():
    rows, alpha, floor, tp, tc = build()
    with open(HERE / "price_list.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = render(rows, alpha, floor, tp, tc)
    (HERE / "price_list.md").write_text(md)
    print("\n".join(md.splitlines()[:22]))
    print(f"... wrote price_list.csv ({len(rows)} rows) and price_list.md")


if __name__ == "__main__":
    main()
