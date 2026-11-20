"""Leave-study-out comparison of REE pattern-completion methods under controlled anchor panels."""
import numpy as np, pandas as pd, warnings, time, sys
warnings.filterwarnings("ignore")
from ree_common import *
from lchg import LCHG, grouped_folds, lambda_ols
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor

SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0]
NFOLD = 10
df = pd.read_csv(PROC / "biota_model.csv")
N = [f"n_{e}" for e in REE]
Y = df[N].to_numpy(float); n = len(Y); obs = ~np.isnan(Y)
META = ["Sample_group", "Phylum", "Class", "Tissue_class", "Tissues", "Trophic_position", "Habitat", "Ecosystem", "Weight"]
X = df[META].copy()
for c in META:
    X[c] = X[c].astype("category")
groups = df.study.to_numpy()
idx = {e: j for j, e in enumerate(REE)}
PANELS = {"meta-only": [], "La": ["La"], "Tb": ["Tb"], "La+Ce": ["La", "Ce"], "Pr+Yb": ["Pr", "Yb"],
          "La+Ce+Nd": ["La", "Ce", "Nd"], "Ce+Nd+Yb": ["Ce", "Nd", "Yb"], "LOEO": None}
POW = (1 << np.arange(14))


def mvn_em(Ytr, n_iter=60):
    mu = np.nanmean(Ytr, 0); S = np.diag(np.nanvar(Ytr, 0)); o = ~np.isnan(Ytr)
    keys = o @ POW
    for _ in range(n_iter):
        Z = Ytr.copy(); C = np.zeros((14, 14))
        for key in np.unique(keys):
            r = np.where(keys == key)[0]; oo = o[r[0]]; mm = ~oo
            if mm.any():
                A = S[np.ix_(mm, oo)] @ np.linalg.inv(S[np.ix_(oo, oo)])
                Z[np.ix_(r, mm)] = mu[mm] + (Ytr[np.ix_(r, oo)] - mu[oo]) @ A.T
                C[np.ix_(mm, mm)] += len(r) * (S[np.ix_(mm, mm)] - A @ S[np.ix_(oo, mm)])
        mu = Z.mean(0); Zc = Z - mu; S = (Zc.T @ Zc + C) / len(Z) + 1e-8 * np.eye(14)
    return mu, S


def mvn_predict(mu, S, Yo):
    out = np.tile(mu, (len(Yo), 1)); o = ~np.isnan(Yo); keys = o @ POW
    for key in np.unique(keys):
        r = np.where(keys == key)[0]; oo = o[r[0]]
        if oo.any():
            A = S[:, oo] @ np.linalg.inv(S[np.ix_(oo, oo)])
            out[r] = mu + (Yo[np.ix_(r, oo)] - mu[oo]) @ A.T
    return out


def augment(Ytr, rng, reps=6):
    """Random-panel masking augmentation for the black-box booster."""
    F, rows = [], []
    for rep in range(reps):
        M = Ytr.copy()
        for i in range(len(M)):
            o = np.where(~np.isnan(M[i]))[0]
            k = rng.integers(0, len(o) + 1)
            M[i, rng.choice(o, size=len(o) - k, replace=False)] = np.nan
        F.append(M); rows.append(np.arange(len(M)))
    return np.vstack(F), np.concatenate(rows)


def lgbm_mask_fit(Ytr, Xtr, rng):
    F, rows = augment(Ytr, rng)
    Xa = Xtr.iloc[rows].reset_index(drop=True)
    models = []
    for e in range(14):
        t = Ytr[rows, e]; keep = ~np.isnan(t)
        Fe = F.copy(); Fe[:, e] = np.nan
        D = pd.concat([pd.DataFrame(Fe, columns=N), Xa], axis=1)
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2, random_state=0)
        models.append(m.fit(D[keep], t[keep]))
    return models


def lgbm_mask_predict(models, Yo, Xte):
    out = np.zeros_like(Yo)
    for e in range(14):
        Fe = Yo.copy(); Fe[:, e] = np.nan
        D = pd.concat([pd.DataFrame(Fe, columns=N), Xte.reset_index(drop=True)], axis=1)
        out[:, e] = models[e].predict(D)
    return out


def panel_inputs(Yte, panel):
    """List of (masked observation matrix, score mask) for a panel on test rows."""
    o = ~np.isnan(Yte)
    if panel is None:   # leave-one-element-out
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


LC = ["LC-HG (proposed)", "LC-HG no-meta", "LC-HG no-anomaly", "LC-HG deg2", "HG learned-basis"]


def run_fold(seed, f, tr, te):
    records = []
    rng = np.random.default_rng(1000 * seed + f)
    Ytr, Yte, gtr = Y[tr], Y[te], groups[tr]
    Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    M = {"LC-HG (proposed)": LCHG(seed=seed).fit(Ytr, Xtr, gtr),
         "LC-HG no-meta": LCHG(use_meta=False, seed=seed).fit(Ytr),
         "LC-HG no-anomaly": LCHG(anomalies=(), seed=seed).fit(Ytr, Xtr, gtr),
         "LC-HG deg2": LCHG(degree=2, seed=seed).fit(Ytr, Xtr, gtr),
         "HG learned-basis": LCHG(basis="learned", seed=seed).fit(Ytr, Xtr, gtr)}
    for k in LC:
        M[k].nu = 4.0            # Gaussian EM fit, coherence-adaptive predictive variance
    mu, S = mvn_em(Ytr)
    mice = IterativeImputer(BayesianRidge(), max_iter=15, random_state=seed).fit(Ytr)
    mforest = IterativeImputer(ExtraTreesRegressor(n_estimators=50, min_samples_leaf=3, n_jobs=2, random_state=seed),
                               max_iter=5, random_state=seed).fit(Ytr)
    knn = KNNImputer(n_neighbors=10, weights="distance").fit(Ytr)
    lm = lgbm_mask_fit(Ytr, Xtr, rng)
    gmean = np.nanmean(Ytr, 0)
    for pname, panel in PANELS.items():
        for Yo, sm in panel_inputs(Yte, panel):
            if not sm.any():
                continue
            rows = sm.any(1)
            Yo_, Xo_ = Yo[rows], Xte[rows].reset_index(drop=True)
            P = {}
            for k in LC:
                P[k] = M[k].predict(Yo_, Xo_ if M[k].use_meta else None)
            P["MVN-EM"] = (mvn_predict(mu, S, Yo_), None)
            P["LightGBM-mask"] = (lgbm_mask_predict(lm, Yo_, Xo_), None)
            P["Global mean"] = (np.tile(gmean, (len(Yo_), 1)), None)
            if (~np.isnan(Yo_)).any():
                P["MICE"] = (mice.transform(Yo_.copy()), None)
                P["missForest"] = (mforest.transform(Yo_.copy()), None)
                P["KNN"] = (knn.transform(Yo_.copy()), None)
                P["Lambda-OLS (per sample)"] = (lambda_ols(Yo_), None)
            ii, jj = np.where(sm[rows])
            gi = te[np.where(rows)[0][ii]]
            base = pd.DataFrame({"seed": seed, "fold": f, "panel": pname, "row": gi, "el": np.array(REE)[jj],
                                 "y": Yte[rows][ii, jj]})
            for k, (mean, sd) in P.items():
                b = base.copy(); b["model"] = k; b["pred"] = mean[ii, jj]
                b["sd"] = sd[ii, jj] if sd is not None else np.nan
                records.append(b)
    return pd.concat(records, ignore_index=True)


from joblib import Parallel, delayed
t0 = time.time()
jobs = [(seed, f, tr, te) for seed in SEEDS for f, (tr, te) in enumerate(grouped_folds(groups, NFOLD, seed))]
records = Parallel(n_jobs=6)(delayed(run_fold)(*j) for j in jobs)
print(f"all folds done {time.time() - t0:.0f}s", flush=True)

R = pd.concat(records, ignore_index=True)
tag = "-".join(map(str, SEEDS))
R.to_parquet(RES / f"completion_predictions_seeds{tag}.parquet", index=False)


def summ(g):
    e = g.y - g.pred
    return pd.Series({"n": len(g), "R2": 1 - (e ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum(),
                      "RMSE": np.sqrt((e ** 2).mean()), "MAE": e.abs().mean()})


S1 = R.groupby(["panel", "model"]).apply(summ).reset_index()
S1.to_csv(RES / f"completion_summary_by_panel_seeds{tag}.csv", index=False)
print("RMSE (log10 units)")
print(S1.pivot(index="model", columns="panel", values="RMSE")[list(PANELS)].round(3).to_string())
print("R2")
print(S1.pivot(index="model", columns="panel", values="R2")[list(PANELS)].round(3).to_string())
print("n scored", S1.pivot(index="model", columns="panel", values="n")[list(PANELS)].iloc[0].to_dict())
