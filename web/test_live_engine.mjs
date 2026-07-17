// Cross-check web/sitelib/live_engine.js against the Python reference
// (race.py / winprob.py / replay_winprob.py). Run:  node web/test_live_engine.mjs
// It shells out to python3 for the reference values, so it needs the repo root.
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const PKL = require(path.join(HERE, "sitelib", "live_engine.js"));

const py = `
import json, sys
sys.path.insert(0, "web")
from sitelib.race import race_dist, sigmoid, team_eta, set_calibration, calibrate
from sitelib.winprob import (ServeDP, serve_probs, eta_anchor, rally_race_p,
                             display_floor, A1, A2, B1, B2)
set_calibration(0.06, 0.9, 0.02118)
out = {}
out["race"] = [[p, T, race_dist(p, T)["p_win"]] for p in (0.35, 0.5, 0.62) for T in (11, 15, 21)]
out["serve_probs"] = [serve_probs(e, 0.43) for e in (-0.8, 0.0, 0.55)]
cases = []
for eta in (-0.6, 0.0, 0.45, 1.2):
    dp = ServeDP(eta, 0.43, 11)
    for (a, b, s) in ((0,0,A2),(0,0,B2),(4,7,A1),(7,4,B1),(10,10,A2),(10,10,B1),(9,10,B2),(13,12,A1)):
        cases.append([eta, a, b, s, dp.p(a, b, s)])
dp15 = ServeDP(0.3, 0.43, 15)
cases.append([0.3, 12, 9, B2, dp15.p(12, 9, B2)])
out["dp"] = cases
out["anchor"] = [eta_anchor(p, 0.43, 11) for p in (0.5, 0.65, 0.88)]
out["rally"] = [[a, b, p, rally_race_p(a, b, p, 21)] for (a, b, p) in ((0,0,0.5),(0,0,0.55),(14,18,0.52),(20,20,0.5))]
out["cal"] = [calibrate(p) for p in (0.5, 0.75, 0.999)]
out["floor"] = display_floor(0.97)
out["team_eta"] = team_eta(0.7, 0.2, 0.4, 0.35)
print(json.dumps(out))
`;
const ref = JSON.parse(execFileSync("python3", ["-c", py], { cwd: path.join(HERE, "..") }).toString());

PKL.configure({ cal: { a: 0.06, b: 0.9, eps: 0.02118 } });

let fails = 0;
const check = (label, got, want, tol = 1e-9) => {
  if (Math.abs(got - want) > tol) { console.error(`FAIL ${label}: js=${got} py=${want}`); fails++; }
};

for (const [p, T, want] of ref.race) check(`race(${p},${T})`, PKL.raceDist(p, T).pw, want);
ref.serve_probs.forEach(([ka, kb], i) => {
  const etas = [-0.8, 0.0, 0.55];
  const [ja, jb] = PKL.serveProbs(etas[i], 0.43);
  check(`serveProbs(${etas[i]}).A`, ja, ka); check(`serveProbs(${etas[i]}).B`, jb, kb);
});
const dps = {};
for (const [eta, a, b, s, want] of ref.dp) {
  const T = eta === 0.3 ? 15 : 11;
  const key = `${eta}|${T}`;
  dps[key] = dps[key] || PKL.ServeDP(eta, 0.43, T);
  check(`dp(${eta})[${a},${b},${s}]`, dps[key].p(a, b, s), want);
}
[0.5, 0.65, 0.88].forEach((p, i) => check(`anchor(${p})`, PKL.etaAnchor(p, 0.43, 11), ref.anchor[i], 1e-6));
for (const [a, b, p, want] of ref.rally) check(`rally(${a},${b},${p})`, PKL.rallyRaceTable(p, 21).p(a, b), want);
[0.5, 0.75, 0.999].forEach((p, i) => check(`cal(${p})`, PKL.calibrate(p), ref.cal[i]));
check("floor(0.97)", PKL.displayFloor(0.97), ref.floor);
check("teamEta", PKL.teamEta(0.7, 0.2, 0.4, 0.35), ref.team_eta);

// matchup composition sanity (JS-only invariants)
check("matchup all-even", PKL.matchupProb(0, 0, [0.5, 0.5, 0.5, 0.5], 0.5), 0.5, 1e-12);
check("matchup 2-2", PKL.matchupProb(2, 2, [], 0.37), 0.37, 1e-12);
check("bestOf3 even", PKL.bestOfProb(0, 0, 2, 0.5), 0.5, 1e-12);
const g = 0.62;
check("bestOf3 closed form", PKL.bestOfProb(0, 0, 2, g), g * g * (3 - 2 * g), 1e-12);

// serve-DP internal consistency: point share pins sigmoid(eta)
const [kA, kB] = PKL.serveProbs(0.7, 0.43);
const share = (kA / (1 - kA)) / (kA / (1 - kA) + kB / (1 - kB));
check("odds-split share", share, PKL.sig(0.7), 1e-12);

if (fails) { console.error(`${fails} FAILURES`); process.exit(1); }
console.log(`all checks passed (${ref.race.length + ref.dp.length + ref.rally.length + 14} comparisons)`);
