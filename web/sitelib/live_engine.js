/* PICKLES live win-probability engine — JS twin of web/sitelib/race.py +
 * web/sitelib/winprob.py (serve-aware side-out DP) plus the matchup
 * composition used by web/replay_winprob.py and the DreamBreaker model of
 * web/make_forecast.py. Keep the four in sync — the Python versions are
 * the reference implementations, validated against the graded receipts.
 *
 * Everything here is pure math on plain numbers; no DOM, no fetch. Loaded
 * by live.html in the browser and by web/test_live_engine.mjs under Node.
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory();
  else root.PKL = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Constants mirrored from the Python modules (single source: v2 posterior
  // + web/calibration.json values are INJECTED by the page at runtime via
  // configure(); these are fallbacks that match the committed values).
  const DEFAULTS = {
    gamma: -0.1829,          // weakest-link (race.py GAMMA)
    kDoubles: 0.43,          // measured serve-rally win rate (winprob.py)
    epsFloor: 0.021,         // display floor (winprob.py EPS_FLOOR)
    cal: { a: 0.0, b: 1.0, eps: 0.0 },   // race.py set_calibration
    kDbSingles: 0.42,        // make_forecast.py K_DB_SINGLES
    singlesImpute: [0.28, 1.14],         // make_forecast.py SINGLES_IMPUTE
  };
  const C = Object.assign({}, DEFAULTS);

  function configure(opts) { Object.assign(C, opts || {}); }

  const sig = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
  const clamp = (p, lo, hi) => Math.min(Math.max(p, lo), hi);

  function comb(n, k) {
    let r = 1;
    for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
    return r;
  }

  // ---- race.py: serve-blind race to T, win by 2 ------------------------
  function raceDist(p, T) {
    p = clamp(p, 1e-9, 1 - 1e-9);
    const q = 1 - p, win = [], lose = [];
    for (let b = 0; b <= T - 2; b++) win.push([T, b, comb(T - 1 + b, b) * p ** T * q ** b]);
    for (let a = 0; a <= T - 2; a++) lose.push([a, T, comb(T - 1 + a, a) * q ** T * p ** a]);
    const deuce = comb(2 * T - 2, T - 1) * (p * q) ** (T - 1);
    const dwin = deuce * ((p * p) / (p * p + q * q));
    const pw = win.reduce((s, x) => s + x[2], 0) + dwin;
    return { pw, win, lose, deuce, dwin };
  }

  function teamEta(v1, v2, v3, v4, gamma) {
    const g = gamma === undefined ? C.gamma : gamma;
    return (v1 + v2 + g * Math.abs(v1 - v2)) - (v3 + v4 + g * Math.abs(v3 - v4));
  }

  function calibrate(p) {
    p = clamp(p, 1e-12, 1 - 1e-12);
    const l = Math.log(p / (1 - p));
    return (1 - C.cal.eps) * sig(C.cal.a + C.cal.b * l) + C.cal.eps / 2;
  }

  const displayFloor = (p) => (1 - C.epsFloor) * p + C.epsFloor / 2;

  // ---- winprob.py: serve-aware side-out DP -----------------------------
  // Serve states: 0 = A serves #1, 1 = A serves #2, 2 = B #1, 3 = B #2.
  const A1 = 0, A2 = 1, B1 = 2, B2 = 3;

  const odds = (p) => { p = clamp(p, 1e-9, 1 - 1e-9); return p / (1 - p); };

  function serveProbs(eta, k) {
    const r = Math.sqrt(odds(sig(eta)));
    const oa = odds(k) * r, ob = odds(k) / r;
    return [oa / (1 + oa), ob / (1 + ob)];
  }

  // V is a Float64Array over (a, b, state), a,b in [0, cap]; terminal cells
  // resolved inline. Same backward induction + cell algebra as _table().
  function ServeDP(eta, k, T) {
    k = k === undefined ? C.kDoubles : k;
    T = T === undefined ? 11 : T;
    const cap = T + 40;
    let [kA, kB] = serveProbs(eta, k);
    // Python builds its table from round(kA, 6) (lru_cache key) — mirror it
    // so both implementations return identical numbers.
    kA = Math.round(kA * 1e6) / 1e6;
    kB = Math.round(kB * 1e6) / 1e6;
    const qA = 1 - kA, qB = 1 - kB;
    const denom = 1 - qA * qA * qB * qB;
    const N = cap + 1;
    const V = new Float64Array(N * N * 4);
    const done = (a, b) => (a >= T && a - b >= 2 ? 1 : b >= T && b - a >= 2 ? 0 : null);
    const get = (a, b, s) => {
      const d = done(a, b);
      if (d !== null) return d;
      if (a >= cap || b >= cap) return 0.5;   // unreachable-in-practice deuce tail
      return V[(a * N + b) * 4 + s];
    };
    for (let a = cap - 1; a >= 0; a--) {
      for (let b = cap - 1; b >= 0; b--) {
        if (done(a, b) !== null) continue;
        const wa1 = get(a + 1, b, A1), wa2 = get(a + 1, b, A2);
        const lb1 = get(a, b + 1, B1), lb2 = get(a, b + 1, B2);
        const va1 = (kA * wa1 + qA * kA * wa2 + qA * qA * kB * lb1 + qA * qA * qB * kB * lb2) / denom;
        const va2 = kA * wa2 + qA * (kB * lb1 + qB * kB * lb2 + qB * qB * va1);
        const vb2 = kB * lb2 + qB * va1;
        const vb1 = kB * lb1 + qB * vb2;
        const base = (a * N + b) * 4;
        V[base + A1] = va1; V[base + A2] = va2; V[base + B1] = vb1; V[base + B2] = vb2;
      }
    }
    return {
      eta, k, T,
      p(a, b, s) {
        const d = done(a, b);
        if (d !== null) return d;
        if (a >= cap || b >= cap) return 0.5;
        return V[(a * N + b) * 4 + s];
      },
    };
  }

  // Anchor: find eta' whose serve-DP start-of-game prob (mean of the two
  // opening #2-server states) equals the CALIBRATED pre-match probability.
  // Mirrors winprob.eta_anchor — keeps live curves consistent with the
  // graded receipts at rally zero.
  function etaAnchor(targetP, k, T) {
    targetP = clamp(targetP, 1e-6, 1 - 1e-6);
    let lo = -8, hi = 8;
    for (let i = 0; i < 40; i++) {
      const mid = 0.5 * (lo + hi);
      const dp = ServeDP(mid, k, T);
      if (0.5 * (dp.p(0, 0, A2) + dp.p(0, 0, B2)) < targetP) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  // ---- DreamBreaker: rally-scored race (winprob.rally_race_p) ----------
  // Iterative DP table (the Python version memoizes a recursion).
  function rallyRaceTable(p, T) {
    T = T === undefined ? 21 : T;
    const cap = T + 40, N = cap + 1;
    const V = new Float64Array(N * N);
    const done = (x, y) => (x >= T && x - y >= 2 ? 1 : y >= T && y - x >= 2 ? 0 : null);
    for (let x = cap - 1; x >= 0; x--) {
      for (let y = cap - 1; y >= 0; y--) {
        if (done(x, y) !== null) continue;
        const up = x + 1 >= cap ? (done(x + 1, y) ?? 0.5) : (done(x + 1, y) ?? V[(x + 1) * N + y]);
        const dn = y + 1 >= cap ? (done(x, y + 1) ?? 0.5) : (done(x, y + 1) ?? V[x * N + y + 1]);
        V[x * N + y] = p * up + (1 - p) * dn;
      }
    }
    return {
      p(x, y) {
        const d = done(x, y);
        if (d !== null) return d;
        if (x >= cap || y >= cap) return 0.5;
        return V[x * N + y];
      },
    };
  }

  // Solve the per-rally prob whose fresh-start rally race equals pDb
  // (replay_winprob.py does the same bisection).
  function rallyPForTarget(pDb, T) {
    let lo = 1e-4, hi = 1 - 1e-4;
    for (let i = 0; i < 40; i++) {
      const mid = 0.5 * (lo + hi);
      if (rallyRaceTable(mid, T).p(0, 0) < pDb) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  // ---- make_forecast.py: DreamBreaker pre-match prob from singles ------
  // rosters: arrays of {sv: singles value or null, v: doubles value or null}
  function dbWinProb(roster1, roster2) {
    const [ia, ib] = C.singlesImpute;
    const sOf = (p) => (p.sv !== null && p.sv !== undefined ? p.sv
      : p.v !== null && p.v !== undefined ? ia + ib * p.v : null);
    const s1 = roster1.map(sOf), s2 = roster2.map(sOf);
    if (!s1.length || !s2.length || s1.some((v) => v === null) || s2.some((v) => v === null))
      return 0.5;
    const gap = s1.reduce((a, b) => a + b, 0) / s1.length - s2.reduce((a, b) => a + b, 0) / s2.length;
    const p = raceDist(sig(C.kDbSingles * gap), 21).pw;
    const eps = C.cal.eps || C.epsFloor;
    return clamp(p, eps / 2, 1 - eps / 2);
  }

  // ---- matchup composition (replay_winprob.matchup_prob) ---------------
  // First to 3 of the 4 MLP games; 2-2 goes to the DreamBreaker at pDb.
  function matchupProb(w1, w2, futurePs, pDb) {
    if (w1 >= 3) return 1;
    if (w2 >= 3) return 0;
    if (!futurePs.length) return pDb;
    const p = futurePs[0], rest = futurePs.slice(1);
    return p * matchupProb(w1 + 1, w2, rest, pDb) + (1 - p) * matchupProb(w1, w2 + 1, rest, pDb);
  }

  // Best-of-N with a single per-game prob (PPA); returns match win prob.
  function bestOfProb(wonA, wonB, need, pGame) {
    if (wonA >= need) return 1;
    if (wonB >= need) return 0;
    return pGame * bestOfProb(wonA + 1, wonB, need, pGame)
      + (1 - pGame) * bestOfProb(wonA, wonB + 1, need, pGame);
  }

  // House formatting: no probability ever displays as 0% or 100%.
  const fp = (p) => (p < 0.005 ? "<1" : p > 0.995 ? ">99" : (100 * p).toFixed(0));

  return {
    configure, C, sig, comb, raceDist, teamEta, calibrate, displayFloor,
    serveProbs, ServeDP, etaAnchor, rallyRaceTable, rallyPForTarget,
    dbWinProb, matchupProb, bestOfProb, fp,
    A1, A2, B1, B2,
  };
});
