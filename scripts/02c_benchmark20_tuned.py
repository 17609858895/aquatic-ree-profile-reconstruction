"""Nested hyper-parameter tuning of the 20-learner benchmark (tasks L0 and L1; study-grouped vs random splits).

For every outer fold, each learner's configuration is selected on inner folds built from the outer-training
data only (study-grouped inner folds for the leave-study-out scheme, random inner folds for the random scheme).
The inner out-of-fold predictions of the selected configuration feed the stacking meta-learner (NNLS).
Usage: python 02c_benchmark20_tuned.py [n_workers]
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import sys, time, json, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from ree_common import *
from lchg import grouped_folds

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CAT = ["Sample_group", "Phylum", "Class", "Tissue_class", "Tissues", "Trophic_position", "Habitat", "Ecosystem", "Weight"]
df = pd.read_csv(PROC / "biota_model.csv")
N = [f"n_{e}" for e in REE]
df["level"] = df[N].mean(1)
TASKS = {"L0": ("level", []), "L1": ("n_Yb", ["n_La"])}


def grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, v)) for v in itertools.product(*kw.values())]


def sample(cands, n, seed=0):
    if len(cands) <= n:
        return cands
    idx = np.random.default_rng(seed).choice(len(cands), n, replace=False)
    return [cands[i] for i in sorted(idx)]


GPK = {"rbf": lambda: ConstantKernel(1.0) * RBF(3.0) + WhiteKernel(0.5),
       "matern": lambda: ConstantKernel(1.0) * Matern(3.0, nu=1.5) + WhiteKernel(0.5),
       "rq": lambda: ConstantKernel(1.0) * RationalQuadratic(3.0, 1.0) + WhiteKernel(0.5)}
# name -> (constructor(params), candidate list)
SPACE = {
    "Ridge": (lambda p: Ridge(**p), grid(alpha=list(np.logspace(-2, 3, 8)))),
    "Lasso": (lambda p: Lasso(max_iter=5000, **p), grid(alpha=list(np.logspace(-4, 0, 8)))),
    "ElasticNet": (lambda p: ElasticNet(max_iter=5000, **p), sample(grid(alpha=list(np.logspace(-4, 0, 5)), l1_ratio=[0.2, 0.5, 0.8]), 10)),
    "BayesianRidge": (lambda p: BayesianRidge(**p), [dict(alpha_1=a, lambda_1=b) for a, b in [(1e-6, 1e-6), (1e-4, 1e-6), (1e-6, 1e-4), (1e-2, 1e-2)]]),
    "Huber": (lambda p: HuberRegressor(max_iter=500, **p), sample(grid(epsilon=[1.2, 1.35, 1.6, 2.0], alpha=[1e-3, 1e-1, 1.0, 10.0]), 8)),
    "PLS": (lambda p: PLSRegression(**p), grid(n_components=[1, 2, 3, 4, 6, 8, 10, 14])),
    "KNN": (lambda p: KNeighborsRegressor(**p), grid(n_neighbors=[5, 10, 20, 40, 80], weights=["uniform", "distance"])),
    "SVR": (lambda p: SVR(**p), sample(grid(C=[0.3, 1.0, 3.0, 10.0], gamma=["scale", 0.01, 0.03], epsilon=[0.05, 0.2]), 10)),
    "KernelRidge": (lambda p: KernelRidge(kernel="rbf", **p), grid(alpha=[0.1, 1.0, 10.0], gamma=[0.005, 0.02, 0.08])),
    "GPR": (lambda p: GaussianProcessRegressor(GPK[p["kernel"]](), normalize_y=True, alpha=1e-6), grid(kernel=["rbf", "matern", "rq"])),
    "DecisionTree": (lambda p: DecisionTreeRegressor(random_state=0, **p), grid(max_depth=[3, 5, 8], min_samples_leaf=[5, 15, 40])),
    "RandomForest": (lambda p: RandomForestRegressor(200, n_jobs=1, random_state=0, **p),
                     sample(grid(min_samples_leaf=[1, 3, 5, 10, 20], max_features=[0.2, 0.4, 0.7]), 8)),
    "ExtraTrees": (lambda p: ExtraTreesRegressor(200, n_jobs=1, random_state=0, **p),
                   sample(grid(min_samples_leaf=[1, 3, 5, 10, 20], max_features=[0.2, 0.4, 0.7]), 8)),
    "AdaBoost": (lambda p: AdaBoostRegressor(DecisionTreeRegressor(max_depth=p["depth"]), n_estimators=100, learning_rate=p["lr"], random_state=0),
                 sample(grid(depth=[2, 3, 4], lr=[0.05, 0.3, 1.0]), 6)),
    "GBDT": (lambda p: GradientBoostingRegressor(n_estimators=p["n"], learning_rate=p["lr"], max_depth=p["depth"], subsample=0.8, random_state=0),
             sample([dict(n=n, lr=lr, depth=d) for (n, lr) in [(150, 0.05), (300, 0.03), (500, 0.02)] for d in [2, 3, 4]], 8)),
    "HistGB": (lambda p: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0, **p),
               sample(grid(max_leaf_nodes=[4, 8, 16], min_samples_leaf=[10, 20, 40], l2_regularization=[0.0, 1.0, 10.0]), 8)),
    "XGBoost": (lambda p: XGBRegressor(n_estimators=300, learning_rate=0.03, subsample=0.8, n_jobs=1, random_state=0, **p),
                sample(grid(max_depth=[2, 3, 4, 6], min_child_weight=[1, 5, 15], reg_lambda=[1.0, 5.0, 20.0], colsample_bytree=[0.5, 0.8]), 10)),
    "LightGBM": (lambda p: LGBMRegressor(n_estimators=300, learning_rate=0.03, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=1, random_state=0, **p),
                 sample(grid(num_leaves=[4, 8, 16], min_child_samples=[10, 20, 40], reg_lambda=[0.0, 2.0, 10.0], colsample_bytree=[0.5, 0.8]), 10)),
    "CatBoost": (lambda p: CatBoostRegressor(iterations=300, verbose=0, thread_count=1, random_seed=0, allow_writing_files=False, **p),
                 sample(grid(depth=[3, 4, 6], l2_leaf_reg=[1.0, 5.0, 20.0], learning_rate=[0.03, 0.08]), 6)),
    "MLP": (lambda p: MLPRegressor(max_iter=600, early_stopping=True, random_state=0, **p),
            sample(grid(hidden_layer_sizes=[(32,), (64, 32), (128, 64)], alpha=[1e-4, 1e-2, 1.0], learning_rate_init=[1e-3, 3e-3]), 8)),
}
NAMES = list(SPACE)
assert len(NAMES) == 20


def encode(Xa, Xb, num):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False), CAT),
                             ("num", StandardScaler(), num)])
    return pre.fit_transform(Xa), pre.transform(Xb)


def fitpred(name, params, A, ya, Bm):
    m = SPACE[name][0](params)
    if name == "PLS":
        m.set_params(n_components=min(params["n_components"], A.shape[1], A.shape[0] - 1))
    m.fit(A, ya)
    return np.asarray(m.predict(Bm)).ravel()


def run_fold(task, scheme, f, tr, te):
    target, num = TASKS[task]
    D = df[df[target].notna() & df[num].notna().all(1)].reset_index(drop=True)
    X, y, g = D[CAT + num], D[target].to_numpy(), D.study.to_numpy()
    Xtr, ytr = X.iloc[tr], y[tr]
    inner = grouped_folds(g[tr], 4, 100 + f) if scheme == "study" else list(KFold(4, shuffle=True, random_state=f).split(Xtr))
    enc_inner = [encode(Xtr.iloc[a], Xtr.iloc[b], num) for a, b in inner]
    Atr, Ate = encode(Xtr, X.iloc[te], num)
    O = np.zeros((len(tr), 20)); T = np.zeros((len(te), 20)); chosen = {}; inner_rmse = {}
    for j, name in enumerate(NAMES):
        best, best_oof, best_p = np.inf, None, None
        for p in SPACE[name][1]:
            oof = np.zeros(len(tr))
            try:
                for (a, b), (Aa, Ab) in zip(inner, enc_inner):
                    oof[b] = fitpred(name, p, Aa, ytr[a], Ab)
                r = float(np.sqrt(np.mean((oof - ytr) ** 2)))
            except Exception:
                r = np.inf
            if r < best:
                best, best_oof, best_p = r, oof, p
        O[:, j] = best_oof; chosen[name] = best_p; inner_rmse[name] = best
        T[:, j] = fitpred(name, best_p, Atr, ytr, Ate)
    w, _ = nnls(O, ytr); w = w / w.sum()
    top = np.argsort([inner_rmse[n] for n in NAMES])[:5]
    w5, _ = nnls(O[:, top], ytr); w5 = w5 / w5.sum()
    P = {n: T[:, j] for j, n in enumerate(NAMES)}
    P["Stack-NNLS"] = T @ w; P["Stack-top5"] = T[:, top] @ w5; P["Average"] = T.mean(1)
    return {"task": task, "scheme": scheme, "fold": f, "te": te, "pred": P, "y": y[te], "study": g[te],
            "chosen": {k: json.dumps(v, default=str) for k, v in chosen.items()}, "weights": dict(zip(NAMES, w)), "inner_rmse": inner_rmse}


if __name__ == "__main__":
    t0 = time.time()
    jobs = []
    for task, (target, num) in TASKS.items():
        D = df[df[target].notna() & df[num].notna().all(1)].reset_index(drop=True)
        g = D.study.to_numpy()
        for scheme in ["study", "random"]:
            folds = grouped_folds(g, 10, 0) if scheme == "study" else list(KFold(10, shuffle=True, random_state=0).split(D))
            jobs += [(task, scheme, f, tr, te) for f, (tr, te) in enumerate(folds)]
    out = Parallel(n_jobs=NW, verbose=5)(delayed(run_fold)(*j) for j in jobs)
    print(f"all jobs done {time.time() - t0:.0f}s", flush=True)
    rows, chosen, weights = [], [], []
    for task in TASKS:
        for scheme in ["study", "random"]:
            sel = [o for o in out if o["task"] == task and o["scheme"] == scheme]
            y = np.concatenate([o["y"] for o in sel]); st = np.concatenate([o["study"] for o in sel])
            P = {k: np.concatenate([o["pred"][k] for o in sel]) for k in sel[0]["pred"]}
            pd.DataFrame(P).assign(y=y, study=st).to_csv(RES / f"benchmark20_tuned_{task}_{scheme}_predictions.csv", index=False)
            for k, p in P.items():
                e = y - p
                rows.append({"task": task, "scheme": scheme, "model": k, "n": len(y), "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                             "RMSE": np.sqrt((e ** 2).mean()), "MAE": np.abs(e).mean(),
                             "median_study_RMSE": pd.Series(e ** 2).groupby(st).mean().pow(0.5).median()})
            for o in sel:
                chosen.append({"task": task, "scheme": scheme, "fold": o["fold"], **o["chosen"]})
                weights.append({"task": task, "scheme": scheme, "fold": o["fold"], **o["weights"]})
    S = pd.DataFrame(rows)
    S.to_csv(RES / "benchmark20_tuned_summary.csv", index=False)
    pd.DataFrame(chosen).to_csv(RES / "benchmark20_tuned_chosen_params.csv", index=False)
    pd.DataFrame(weights).to_csv(RES / "benchmark20_tuned_stack_weights.csv", index=False)
    pd.set_option("display.width", 200)
    for task in TASKS:
        pv = S[S.task == task].pivot(index="model", columns="scheme", values="R2")
        pv["gap"] = pv["random"] - pv["study"]
        print(task); print(pv.sort_values("study", ascending=False).round(3).to_string())
