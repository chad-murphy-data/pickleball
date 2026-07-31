"""C1d — how much attenuation does the wind PROXY impose?

    python model/weather_review/c1_attenuation.py

Two measurable components of measurement error in the wind regressor:

  (a) TIMING. For matches that carry both a true-UTC game-end stamp and a
      planned start time, the ERA5 hourly wind can be read at either
      hour. Treat them as two noisy readings of "the wind while this match
      was on". Their correlation r bounds the reliability of either one;
      a classical-errors slope is attenuated by the reliability ratio, so
      the true slope is roughly beta_observed / r.

  (b) GRID vs COURT. ERA5 10 m wind on a ~9 km grid cell versus what the
      ball feels 1 m over a court, behind windscreens and stands. No data
      here can measure this; the published literature puts hourly
      reanalysis-vs-station wind correlations around 0.7-0.9, which would
      add another 1/0.7-1/0.9 inflation. Reported as a stated assumption,
      not a measurement.

Both push the same way: every slope in this review is a LOWER bound on
the magnitude of the true effect, and every MDE is an UPPER bound on how
small an effect the data could have seen.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_lib import (ROOT, event_tz, get_tz, label_arms, load_hourly,  # noqa
                    local_hour_key, parse_utc, read_csv)
from c1_build_pace import END_COLS, calibrate_offsets, naive_local  # noqa: E402


def corr(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** .5
    db = sum((y - mb) ** 2 for y in b) ** .5
    return num / (da * db)


def main():
    mt_rows = read_csv(ROOT / "data/match_times.csv")
    tzs = event_tz()
    offsets = calibrate_offsets(mt_rows, tzs)
    hourly = load_hourly()
    arms = label_arms()
    pairs, pairs_g = [], []
    same_hour = 0
    for r in mt_rows:
        ev = r["event_id"]
        if arms["corrected_all"].get(ev) != "outdoor":
            continue
        tzname = tzs.get(ev, "")
        if get_tz(tzname) is None or ev not in offsets:
            continue
        ends = [parse_utc(r[c]) for c in END_COLS]
        ends = [e for e in ends if e]
        pl = naive_local(r["planned_start_local"])
        if not ends or pl is None:
            continue
        h_true = local_hour_key(ends[0], tzname)
        h_plan = (pl - offsets[ev]).strftime("%Y-%m-%dT%H")
        a, b = hourly.get((ev, h_true)), hourly.get((ev, h_plan))
        if not a or not b or a["wind"] is None or b["wind"] is None:
            continue
        if h_true == h_plan:
            same_hour += 1
        pairs.append((a["wind"], b["wind"]))
        if a["gust"] is not None and b["gust"] is not None:
            pairs_g.append((a["gust"], b["gust"]))

    n = len(pairs)
    r_w = corr([p[0] for p in pairs], [p[1] for p in pairs])
    r_g = corr([p[0] for p in pairs_g], [p[1] for p in pairs_g])
    print("outdoor matches with both a true game hour and a planned hour: %d" % n)
    print("  same hour after rounding: %d (%.0f%%)" % (same_hour, 100 * same_hour / n))
    print("  corr(wind @ true hour, wind @ planned hour)  = %.3f" % r_w)
    print("  corr(gust @ true hour, gust @ planned hour)  = %.3f" % r_g)
    print("  => timing-only attenuation factor ~ %.3f; a published slope of"
          " b is really ~ %.2f x b" % (r_w, 1 / r_w))
    diffs = [abs(a - b) for a, b in pairs]
    diffs.sort()
    print("  |wind(true) - wind(planned)|: median %.1f mph, p90 %.1f mph"
          % (diffs[len(diffs) // 2], diffs[int(.9 * len(diffs))]))
    print("\n  For reference, the committed tests (weather_report.py,"
          " favorites_wind.py,\n  end_effects.py, wind_skill.py) join on"
          " start_local OR planned_start_local,\n  i.e. the noisier of these"
          " two readings for the ~%.0f%% of matches where the\n  two hours"
          " disagree." % (100 * (1 - same_hour / n)))


if __name__ == "__main__":
    main()
