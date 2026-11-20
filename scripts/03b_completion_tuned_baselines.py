"""Tuned imputation baselines for the pattern-completion comparison (same folds and panels as 03_completion.py).

Within every outer training set, each baseline's configuration is selected on study-grouped inner folds by the
RMSE of masked reconstruction (half sparse panels with 1-3 anchors, half leave-one-element-out masks).
LC-HG and the multivariate-normal EM have no tuned hyper-parameters and are not refitted here.
Usage: python 03b_completion_tuned_baselines.py 0,1,2 [n_workers]
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import sys, time, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from ree_common import *
from lchg import grouped_folds

SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0]
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 8
NFOLD = 10
df = pd.read_csv(PROC / "biota_model.csv")
N = [f"n_{e}" for e in REE]
Y = df[N].to_numpy(float)
META = ["Sample_group", "Phylum", "Class", "Tissue_class", "Tissues", "Trophic_position", "Habitat", "Ecosystem", "Weight"]
X = df[META].copy()
for c in META:
    X[c] = X[c].astype("category")
groups = df.study.to_numpy()
idx = {e: j for j, e in enumerate(REE)}
PANELS = {"meta-only": [], "La": ["La"], "Tb": ["Tb"], "La+Ce": ["La", "Ce"], "Pr+Yb": ["Pr", "Yb"],
          "La+Ce+Nd": ["La", "Ce", "Nd"], "Ce+Nd+Yb": ["Ce", "Nd", "Yb"], "LOEO": None}

LGB_BASE = dict(subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=1, random_state=0)
CAND = {
    "LightGBM-mask": [
        dict(tag="default", meta=True, reps=6, kw=dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20, colsample_bytree=0.8)),
        dict(tag="shallow-regularised", meta=True, reps=6, kw=dict(n_estimators=300, learning_rate=0.05, num_leaves=4, min_child_samples=40, reg_lambda=10.0, colsample_bytree=0.8)),
        dict(tag="linear-tree", meta=True, reps=6, kw=dict(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=40, linear_tree=True, linear_lambda=1.0)),
        dict(tag="linear-tree-shallow-no-meta", meta=False, reps=6, kw=dict(n_estimators=200, learning_rate=0.05, num_leaves=4, min_child_samples=60, linear_tree=True, linear_lambda=10.0)),
        dict(tag="no-meta", meta=False, reps=6, kw=dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20, colsample_bytree=0.8)),
        dict(tag="more-augmentation", meta=True, reps=12, kw=dict(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=30, reg_lambda=2.0, colsample_bytree=0.5)),
        dict(tag="no-meta-small", meta=False, reps=6, kw=dict(n_estimators=400, learning_rate=0.05, num_leaves=8, min_child_samples=30, reg_lambda=2.0, colsample_bytree=0.8)),
        dict(tag="no-meta-more-augmentation", meta=False, reps=12, kw=dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20, colsample_bytree=0.8)),
        dict(tag="no-meta-deep", meta=False, reps=6, kw=dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10, colsample_bytree=0.8)),
    ],
    "KNN": [dict(tag=f"k{k}-{w}", k=k, w=w) for k in (3, 5, 10, 20, 40) for w in ("uniform", "distance")],
    "MICE": [dict(tag="iter15", max_iter=15, nn=None), dict(tag="iter40", max_iter=40, nn=None),
             dict(tag="iter15-nearest5", max_iter=15, nn=5), dict(tag="iter40-nearest8", max_iter=40, nn=8)],
    "missForest": [dict(tag="leaf3", leaf=3, mf=1.0, it=5), dict(tag="leaf1", leaf=1, mf=1.0, it=5),
                   dict(tag="leaf10-mf0.5", leaf=10, mf=0.5, it=5), dict(tag="leaf3-mf0.5-it8", leaf=3, mf=0.5, it=8)],
}


# Round 2: the first search selected boundary configurations for three baselines, so the space is extended outwards
# (incumbent kept as a candidate). Usage: python 03b_completion_tuned_baselines.py 0 8 r2
ROUND2 = len(sys.argv) > 3 and sys.argv[3] == "r2"
if ROUND2:
    CAND = {
        "LightGBM-mask": [
            dict(tag="no-meta-deep", meta=False, reps=6, kw=dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10, colsample_bytree=0.8)),
            dict(tag="no-meta-deeper", meta=False, reps=6, kw=dict(n_estimators=1000, learning_rate=0.03, num_leaves=63, min_child_samples=5, colsample_bytree=0.8)),
            dict(tag="no-meta-deep-aug12", meta=False, reps=12, kw=dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10, colsample_bytree=0.8)),
            dict(tag="no-meta-deep-aug20-allcols", meta=False, reps=20, kw=dict(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=10, colsample_bytree=1.0)),
        ],
        "MICE": [dict(tag="iter40-nearest8", max_iter=40, nn=8), dict(tag="iter100", max_iter=100, nn=None, tol=1e-4),
                 dict(tag="iter300", max_iter=300, nn=None, tol=1e-5), dict(tag="iter100-nearest8", max_iter=100, nn=8, tol=1e-4)],
        "missForest": [dict(tag="leaf3-mf0.5-it8", leaf=3, mf=0.5, it=8), dict(tag="leaf3-mf0.3-it8", leaf=3, mf=0.3, it=8),
                       dict(tag="leaf3-mf0.5-it12", leaf=3, mf=0.5, it=12), dict(tag="leaf5-mf0.33-it12", leaf=5, mf=0.33, it=12)],
    }
SUFFIX = " (tuned r2)" if ROUND2 else " (tuned)"
FILETAG = "_r2" if ROUND2 else ""


# ---------------------------------------------------------------- model wrappers: fit(Ytr, Xtr, rng, cfg) -> predict(Yo, Xo)
def fit_lgbm(Ytr, Xtr, rng, cfg):
    F, rows = [], []
    for rep in range(cfg["reps"]):
        M = Ytr.copy()
        for i in range(len(M)):
            o = np.where(~np.isnan(M[i]))[0]
            k = rng.integers(0, len(o) + 1)
            M[i, rng.choice(o, size=len(o) - k, replace=False)] = np.nan
        F.append(M); rows.append(np.arange(len(M)))
    F, rows = np.vstack(F), np.concatenate(rows)
    Xa = Xtr.iloc[rows].reset_index(drop=True)
    models = []
    for e in range(14):
        t = Ytr[rows, e]; keep = ~np.isnan(t)
        Fe = F.copy(); Fe[:, e] = np.nan
        D = pd.DataFrame(Fe, columns=N)
        if cfg["meta"]:
            D = pd.concat([D, Xa], axis=1)
        models.append(lgb.LGBMRegressor(**LGB_BASE, **cfg["kw"]).fit(D[keep], t[keep]))

    def predict(Yo, Xo):
        out = np.zeros_like(Yo)
        for e in range(14):
            Fe = Yo.copy(); Fe[:, e] = np.nan
            D = pd.DataFrame(Fe, columns=N)
            if cfg["meta"]:
                D = pd.concat([D, Xo.reset_index(drop=True)], axis=1)
            out[:, e] = models[e].predict(D)
        return out
    return predict


def fit_knn(Ytr, Xtr, rng, cfg):
    m = KNNImputer(n_neighbors=cfg["k"], weights=cfg["w"]).fit(Ytr)
    return lambda Yo, Xo: m.transform(Yo.copy())


def fit_mice(Ytr, Xtr, rng, cfg):
    m = IterativeImputer(BayesianRidge(), max_iter=cfg["max_iter"], n_nearest_features=cfg["nn"], tol=cfg.get("tol", 1e-3), random_state=0).fit(Ytr)
    return lambda Yo, Xo: m.transform(Yo.copy())


def fit_mforest(Ytr, Xtr, rng, cfg):
    m = IterativeImputer(ExtraTreesRegressor(n_estimators=50, min_samples_leaf=cfg["leaf"], max_features=cfg["mf"], n_jobs=1, random_state=0),
                         max_iter=cfg["it"], random_state=0).fit(Ytr)
    return lambda Yo, Xo: m.transform(Yo.copy())


FIT = {"LightGBM-mask": fit_lgbm, "KNN": fit_knn, "MICE": fit_mice, "missForest": fit_mforest}


def val_masks(Yb, rng):
    """Two masked copies of the validation rows: sparse anchors (1-3 kept) and leave-one-element-out."""
    o = ~np.isnan(Yb)
    A = np.full_like(Yb, np.nan); B = Yb.copy()
    for i in range(len(Yb)):
        ob = np.where(o[i])[0]
        if len(ob) < 2:
            B[i] = np.nan; continue
        keep = rng.choice(ob, size=min(int(rng.integers(1, 4)), len(ob) - 1), replace=False)
        A[i, keep] = Yb[i, keep]
        B[i, rng.choice(ob)] = np.nan
    return [(A, o & np.isnan(A) & ~np.isnan(A).all(1)[:, None]), (B, o & np.isnan(B) & ~np.isnan(B).all(1)[:, None])]


def panel_inputs(Yte, panel):
    o = ~np.isnan(Yte)
    if panel is None:
        res = []
        for e in range(14):
            Yo = Yte.copy(); Yo[:, e] = np.nan
            sm = np.zeros_like(o); sm[:, e] = o[:, e] & (o.sum(1) >= 2)
            res.append((Yo, sm))
        return res
    pj = [idx[e] for e in panel]
    ok = o[:, pj].all(1) if pj else np.ones(len(Yte), bool)
    Yo = np.full_like(Yte, np.nan); Yo[:, pj] = Yte[:, pj]
    sm = o.copy(); sm[:, pj] = False; sm[~ok] = False
    return [(Yo, sm)]


def run_fold(seed, f, tr, te):
    rng = np.random.default_rng(5000 + 1000 * seed + f)
    Ytr, Yte, gtr = Y[tr], Y[te], groups[tr]
    Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    inner = grouped_folds(gtr, 3, 300 + 10 * seed + f)
    masks = [val_masks(Ytr[b], np.random.default_rng(seed * 100 + f * 10 + j)) for j, (a, b) in enumerate(inner)]
    chosen, final = {}, {}
    for name, cands in CAND.items():
        best, best_cfg, scores = np.inf, cands[0], {}
        for cfg in cands:
            sse, cnt = 0.0, 0
            try:
                for (a, b), mk in zip(inner, masks):
                    pred = FIT[name](Ytr[a], Xtr.iloc[a].reset_index(drop=True), np.random.default_rng(1), cfg)
                    for Yo, sm in mk:
                        rows = sm.any(1)
                        if rows.any():
                            P = pred(Yo[rows], Xtr.iloc[b].reset_index(drop=True)[rows].reset_index(drop=True))
                            sse += float((((Ytr[b][rows] - P) ** 2)[sm[rows]]).sum()); cnt += int(sm[rows].sum())
                r = np.sqrt(sse / max(cnt, 1))
            except Exception as ex:
                r = np.inf
            scores[cfg["tag"]] = r
            if r < best:
                best, best_cfg = r, cfg
        chosen[name] = {"seed": seed, "fold": f, "model": name, "chosen": best_cfg["tag"], "inner_rmse": best,
                        "all_scores": json.dumps({k: (None if not np.isfinite(v) else round(v, 4)) for k, v in scores.items()})}
        final[name] = FIT[name](Ytr, Xtr, rng, best_cfg)
    records = []
    for pname, panel in PANELS.items():
        for Yo, sm in panel_inputs(Yte, panel):
            if not sm.any():
                continue
            rows = sm.any(1)
            Yo_, Xo_ = Yo[rows], Xte[rows].reset_index(drop=True)
            ii, jj = np.where(sm[rows]); gi = te[np.where(rows)[0][ii]]
            base = pd.DataFrame({"seed": seed, "fold": f, "panel": pname, "row": gi, "el": np.array(REE)[jj], "y": Yte[rows][ii, jj]})
            for name in CAND:
                if name != "LightGBM-mask" and not (~np.isnan(Yo_)).any():
                    continue
                b = base.copy(); b["model"] = name + SUFFIX; b["pred"] = final[name](Yo_, Xo_)[ii, jj]; b["sd"] = np.nan
                records.append(b)
    return pd.concat(records, ignore_index=True), list(chosen.values())


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(seed, f, tr, te) for seed in SEEDS for f, (tr, te) in enumerate(grouped_folds(groups, NFOLD, seed))]
    out = Parallel(n_jobs=NW, verbose=5)(delayed(run_fold)(*j) for j in jobs)
    print(f"all folds done {time.time() - t0:.0f}s", flush=True)
    R = pd.concat([o[0] for o in out], ignore_index=True)
    tag = "-".join(map(str, SEEDS))
    R.to_parquet(RES / f"completion_tuned{FILETAG}_predictions_seeds{tag}.parquet", index=False)
    Cn = pd.DataFrame([c for o in out for c in o[1]])
    Cn.to_csv(RES / f"completion_tuned{FILETAG}_chosen_seeds{tag}.csv", index=False)
    S = R.assign(e2=(R.y - R.pred) ** 2).groupby(["panel", "model"]).e2.mean().pow(0.5).unstack("panel")[list(PANELS)]
    pd.set_option("display.width", 200)
    print(S.round(3).to_string())
    print(Cn.groupby(["model", "chosen"]).size().to_string())
