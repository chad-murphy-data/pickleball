"""Better per-candidate learner (learner_gate.md): gradient-boosted trees
on emission.py's 14 features, same labels, same discipline.

    python3 learner.py train      # r6/r7 cross-fold, gate 1, pooled model
    python3 learner.py cache      # _gbt p-caches (r9/r10 pooled, r6/r7 cross)
    python3 learner.py tune       # PF_PXS=_gbt path-first grid on r6/r7, gate 2
    python3 learner.py grade <r>  # the one shot (r9/r10), knobs frozen

Positives come ONLY from the owner's V clicks (emission.harvest_train);
r9/r10 clicks are grading-only. pathfirst.py is untouched apart from the
PF_PXS cache-suffix hook; the frozen pathfirst_tune.json is never
written.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import emission as em                                       # noqa: E402

SP = Path(__file__).parent
SFX = "_gbt"
MODEL_PKL = SP / "emission_gbt.pkl"
MODEL_JSON = SP / "emission_gbt.json"
TUNE_JSON = SP / "pathfirst_tune_gbt.json"
CONFIGS = {
    "A": dict(max_leaf_nodes=15, max_iter=300, learning_rate=0.05),
    "B": dict(max_leaf_nodes=31, max_iter=300, learning_rate=0.05),
    "C": dict(max_leaf_nodes=15, max_iter=600, learning_rate=0.03),
}
FIXED = dict(l2_regularization=1.0, min_samples_leaf=20, early_stopping=False,
             random_state=7)
INC_LOGIT = {"6->7": dict(auc=0.9042, neg_kept=0.7229),
             "7->6": dict(auc=0.9394, neg_kept=0.3192)}
INC_TRAIN = dict(h12=263, prec=0.807)
INC_EVAL = {9: dict(h12=537, prec=0.87, f1=0.731), 10: dict(h12=422, prec=0.88, f1=0.675)}


def fit(X, y, cfg):
    w = np.where(y == 1, (y == 0).sum() / max(y.sum(), 1), 1.0)
    m = HistGradientBoostingClassifier(**cfg, **FIXED)
    m.fit(X, y, sample_weight=w)
    return m


def xval_stats(sc, yte):
    a = em.auc(sc, yte)
    pos_sc = np.sort(sc[yte == 1])
    thr97 = pos_sc[max(0, int(0.03 * len(pos_sc)) - 1)]
    keep_neg = float((sc[yte == 0] >= thr97).mean())
    return dict(auc=round(float(a), 4), neg_kept=round(keep_neg, 4))


def train():
    packs = {r: em.harvest_train(r) for r in (6, 7)}
    for r, (F, y) in packs.items():
        print(f"r{r}: {len(y)} labeled cands, {int(y.sum())} pos")
    results = {}
    for name, cfg in CONFIGS.items():
        st = {}
        for tr_r, te_r in ((6, 7), (7, 6)):
            m = fit(packs[tr_r][0], packs[tr_r][1], cfg)
            sc = m.predict_proba(packs[te_r][0])[:, 1]
            st[f"{tr_r}->{te_r}"] = xval_stats(sc, packs[te_r][1])
        mean_auc = (st["6->7"]["auc"] + st["7->6"]["auc"]) / 2
        results[name] = dict(cfg=cfg, xval=st, mean_auc=round(mean_auc, 4))
        print(f"  cfg {name} {cfg}: 6->7 {st['6->7']}  7->6 {st['7->6']}  mean AUC {mean_auc:.4f}")
    print(f"  incumbent logistic: {INC_LOGIT}")
    order = sorted(results, key=lambda k: (-results[k]["mean_auc"], k))   # ties -> A (smaller)
    best = order[0]
    st = results[best]["xval"]
    ok = all(st[k]["auc"] >= INC_LOGIT[k]["auc"] and st[k]["neg_kept"] <= INC_LOGIT[k]["neg_kept"]
             for k in INC_LOGIT)
    out = dict(selected=best, results=results, gate1="PASS" if ok else "DEAD",
               feats=em.FEATS, train_rallies=[6, 7])
    if ok:
        X = np.vstack([packs[6][0], packs[7][0]])
        y = np.concatenate([packs[6][1], packs[7][1]])
        m = fit(X, y, CONFIGS[best])
        p = m.predict_proba(X)[:, 1]
        print(f"pooled: in-sample AUC {em.auc(p, y):.4f}")
        folds = {r: fit(packs[r][0], packs[r][1], CONFIGS[best]) for r in (6, 7)}
        pickle.dump(dict(pooled=m, folds=folds), open(MODEL_PKL, "wb"))
        out.update(n=int(len(y)), n_pos=int(y.sum()))
    MODEL_JSON.write_text(json.dumps(out, indent=1))
    print(f"GATE 1: {out['gate1']} (selected {best}); wrote {MODEL_JSON.name}"
          f"{' + ' + MODEL_PKL.name if ok else ''}")


def _score_rows(rally, model, sfx):
    rows_by = {}
    for mode in ("cc", "peak"):
        rows = em.cands_rows(rally, mode)
        rows_by[mode] = (rows, np.zeros((len(rows), len(em.FEATS)), np.float32))
    em.featurize(rally, rows_by)
    for mode, (rows, F) in rows_by.items():
        p = model.predict_proba(F)[:, 1].astype(np.float32)
        out = SP / f"p_r{rally}_{mode}_{em.THR}{sfx}.npz"
        np.savez_compressed(out, p=p, fxy=rows[:, :3].astype(np.float32))
        print(f"{out.name}: {len(p)} rows, p median {np.median(p):.3f}, p>=0.4 {float((p >= 0.4).mean()):.1%}")


def cache():
    mj = json.loads(MODEL_JSON.read_text())
    assert mj["gate1"] == "PASS", "gate 1 not passed; no caches"
    mdl = pickle.load(open(MODEL_PKL, "rb"))
    for target, src in ((6, 7), (7, 6)):
        _score_rows(target, mdl["folds"][src], "_x" + SFX)
    for r in (9, 10):
        _score_rows(r, mdl["pooled"], SFX)


def _pf():
    assert os.environ.get("PF_PXS") == SFX, f"run with PF_PXS={SFX}"
    import pathfirst as pf
    import geom_fix
    return pf, geom_fix


def tune():
    pf, geom_fix = _pf()
    ctxs = [pf.context(r) for r in (6, 7)]
    for ctx in ctxs:
        print(f"rally {ctx['rally']}: p-cache '{ctx['pxs']}', decode@12 {sum(ctx['dec'])}/{len(ctx['dec'])}")
    rows = []
    for p_seed in pf.GRID_PSEED:
        for s_min in pf.GRID_SMIN:
            for gap in pf.GRID_GAP:
                h12 = have = 0
                nd = nt = 0
                per = []
                for ctx in ctxs:
                    res = pf.run(ctx, p_seed, s_min, gap)
                    h, hv, _ = geom_fix.grade(res["track"], ctx["truth"], ctx["t0"], ctx["dec"])
                    rng = np.random.default_rng(pf.NULL_SEED + ctx["rally"])
                    d_ = geom_fix.grade(pf.displaced(res["track"], rng), ctx["truth"], ctx["t0"], ctx["dec"])[0]
                    t_ = geom_fix.grade(pf.timeshift(res["track"], ctx, rng), ctx["truth"], ctx["t0"], ctx["dec"])[0]
                    nd, nt = max(nd, d_), max(nt, t_)
                    h12 += h
                    have += hv
                    per.append(f"r{ctx['rally']} {h}/{hv} {len(res['chosen'])}fl")
                prec = h12 / max(1, have)
                cell = dict(p_seed=p_seed, s_min=s_min, gap=gap)
                rows.append((cell, h12, prec, nd, nt))
                print(f"  {cell}  r@12 {h12} prec {prec:.3f}  nulls {nd}/{nt}  {' '.join(per)}")
    ok = [x for x in rows if x[2] >= INC_TRAIN["prec"] - 0.03 and x[3] <= 3 and x[4] <= 3]
    ok.sort(key=lambda x: (-x[1], -x[0]["s_min"], x[0]["gap"], -x[0]["p_seed"]))
    if not ok or ok[0][1] <= INC_TRAIN["h12"]:
        print(f"GATE 2 DEAD: no cell beats the incumbent {INC_TRAIN} under the rule")
        TUNE_JSON.write_text(json.dumps(dict(dead=True, grid=[dict(c, h12=h, prec=round(p, 3)) for c, h, p, _, _ in rows])))
        return
    best = ok[0]
    print(f"GATE 2 SELECTED {best[0]}  r@12 {best[1]} prec {best[2]:.3f} (incumbent {INC_TRAIN})")
    TUNE_JSON.write_text(json.dumps(dict(best[0], dead=False, train_h12=best[1], train_prec=round(best[2], 3),
                                         grid=[dict(c, h12=h, prec=round(p, 3)) for c, h, p, _, _ in rows])))


def grade(rally):
    pf, geom_fix = _pf()
    import corridor_dp as cdp
    import events as evm
    cell = json.loads(TUNE_JSON.read_text())
    assert not cell.get("dead"), "gate 2 dead; no shot"
    cell = {k: cell[k] for k in ("p_seed", "s_min", "gap")}
    ctx = pf.context(rally)
    res = pf.run(ctx, cell["p_seed"], cell["s_min"], cell["gap"])
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    inc = INC_EVAL[rally]
    print(f"rally {rally}: p-cache '{ctx['pxs']}', cell {cell}; incumbent path-first {inc}")
    cdp.score(res["track"], truth, t0, dec, "gbt-pathfirst")
    for vis in "VS":
        tt = [x for x in truth if x[3] == vis]
        dd = [d for x, d in zip(truth, dec) if x[3] == vis]
        cdp.score(res["track"], tt, t0, dd, f"  [{vis}]")
    rng = np.random.default_rng(pf.NULL_SEED + rally)
    cdp.score(pf.displaced(res["track"], rng), truth, t0, dec, "null-disp")
    cdp.score(pf.timeshift(res["track"], ctx, rng), truth, t0, dec, "null-tshift")
    h, hv, _ = geom_fix.grade(res["track"], truth, t0, dec)
    prec = h / max(1, hv)
    rng = np.random.default_rng(pf.NULL_SEED + rally)
    nd = geom_fix.grade(pf.displaced(res["track"], rng), truth, t0, dec)[0]
    nt = geom_fix.grade(pf.timeshift(res["track"], ctx, rng), truth, t0, dec)[0]
    ec = json.loads((SP / "events_tune_v3.json").read_text())
    evs = evm.events(ctx, res["chosen"], ec["r_seam"], ec["a_seam"], ec["dt_pair"],
                     ec["off"], d_pair=ec["d_pair"])
    cont, bnc = evm.truth_events(ctx["c"])
    pr_ = evm.prf([e["t"] for e in evs], sorted(cont + bnc))
    rec, pr, f1 = pr_["recall"], pr_["precision"], pr_["f1"]
    print(f"  flights {len(res['chosen'])} (hyp {res['n_hyp']}, kept {res['n_kept']}); "
          f"events on new track: n={len(evs)} recall {rec:.3f} prec {pr:.3f} F1 {f1:.3f} (adopted {inc['f1']})")
    bars = [h > inc["h12"], prec >= inc["prec"] - 0.02, nd <= 3, nt <= 3, f1 >= inc["f1"] - 0.03]
    print(f"  BARS: r@12 {h} > {inc['h12']}: {bars[0]}; prec {prec:.3f} >= {inc['prec'] - 0.02:.2f}: {bars[1]}; "
          f"nulls {nd}/{nt} <= 3: {bars[2] and bars[3]}; events F1 {f1:.3f} >= {inc['f1'] - 0.03:.3f}: {bars[4]}"
          f"  =>  {'PASS' if all(bars) else 'FAIL'}")
    return all(bars)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "cache":
        cache()
    elif cmd == "tune":
        tune()
    elif cmd == "grade":
        assert len(sys.argv) == 3, "grade takes the rally only; knobs are frozen"
        grade(int(sys.argv[2]))
