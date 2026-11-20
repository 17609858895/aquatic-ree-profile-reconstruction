"""(1) Closed-form optimal anchor-panel design from LC-HG and its leave-study-out validation.
(2) Study-grouped (Mondrian by panel size) conformal calibration of LC-HG predictive SDs,
    Gaussian vs robust multivariate-t (coherence-adaptive) variants on identical masks."""
import numpy as np, pandas as pd, warnings, time, itertools
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from scipy.stats import norm
from ree_common import *
from lchg import LCHG, grouped_folds

df = pd.read_csv(PROC / "biota_model.csv")
N = [f"n_{e}" for e in REE]
Y = df[N].to_numpy(float); obs = ~np.isnan(Y); n = len(Y)
META = ["Sample_group", "Phylum", "Class", "Tissue_class", "Tissues", "Trophic_position", "Habitat", "Ecosystem", "Weight"]
X = df[META].copy()
for c in META:
    X[c] = X[c].astype("category")
g = df.study.to_numpy()
VARIANTS = {"gauss": (None, None), "t4": (4.0, 4.0), "adaptive": (None, 4.0)}   # (fit nu, predictive nu)
ALPHAS = (0.8, 0.9, 0.95)


def analytic_rmse(model, panel):
    Si = np.linalg.inv(model.Sigma)
    if len(panel):
        Bo = model.B[list(panel)]
        V = np.linalg.inv(Si + (Bo.T / model.s2[list(panel)]) @ Bo)
    else:
        V = model.Sigma
    var = np.einsum("ek,kl,el->e", model.B, V, model.B) + model.s2
    rest = [e for e in range(14) if e not in panel]
    return np.sqrt(var[rest].mean())


def random_panel_masks(Yb, rng, kfix=None):
    """Keep k random observed elements per row (k random in 0..5 unless fixed)."""
    Yo = np.full_like(Yb, np.nan); ks = np.zeros(len(Yb), int)
    for i in range(len(Yb)):
        o = np.where(~np.isnan(Yb[i]))[0]
        if len(o) < 2:
            ks[i] = -1; continue
        k = min(int(rng.integers(0, 6)) if kfix is None else kfix, len(o) - 1)
        if kfix is not None and k < kfix:
            ks[i] = -1; continue
        keep = rng.choice(o, size=k, replace=False) if k > 0 else []
        Yo[i, keep] = Yb[i, keep]; ks[i] = k
    return Yo, ks


def run_fold(f, tr, te, val_panels):
    rng = np.random.default_rng(100 + f)
    Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    mods = {}
    for v, (nf, npred) in VARIANTS.items():
        mods[v] = LCHG(nu=nf).fit(Y[tr], Xtr, g[tr]); mods[v].nu = npred
    emp = {}
    for name, pj in val_panels:
        pj = list(pj)
        ok = obs[te][:, pj].all(1) if pj else np.ones(len(te), bool)
        if ok.sum() == 0:
            continue
        Yo = np.full((len(te), 14), np.nan); Yo[:, pj] = Y[te][:, pj]
        mean, sd = mods["gauss"].predict(Yo[ok], Xte[ok].reset_index(drop=True))
        sm = obs[te][ok].copy(); sm[:, pj] = False
        emp[name] = (Y[te][ok][sm] - mean[sm]) ** 2
    # conformal scores from inner study-grouped folds (identical masks for both variants)
    scores = {v: {k: [] for k in range(6)} for v in VARIANTS}
    for a, b in grouped_folds(g[tr], 4, 50 + f):
        Yb, Xb = Y[tr][b], Xtr.iloc[b].reset_index(drop=True)
        masks = [random_panel_masks(Yb, rng) for _ in range(3)]
        for v, (nf, npred) in VARIANTS.items():
            mi = LCHG(nu=nf, n_em=3).fit(Y[tr][a], Xtr.iloc[a].reset_index(drop=True), g[tr][a]); mi.nu = npred
            for Yo, ks in masks:
                mean, sd = mi.predict(Yo, Xb)
                sm = ~np.isnan(Yb) & np.isnan(Yo)
                z = np.abs(Yb - mean) / sd
                for k in range(6):
                    scores[v][k].append(z[(ks == k)[:, None] & sm])
    conf = []
    for k in range(6):
        Yt = Y[te]
        Yo, ks = random_panel_masks(Yt, rng, kfix=k)
        elig = ks == k
        if elig.sum() == 0:
            continue
        sm = ~np.isnan(Yt[elig]) & np.isnan(Yo[elig])
        ii, jj = np.where(sm)
        for v in VARIANTS:
            mean, sd = mods[v].predict(Yo[elig], Xte[elig].reset_index(drop=True))
            d = pd.DataFrame({"fold": f, "variant": v, "k": k, "row": te[np.where(elig)[0][ii]], "el": np.array(REE)[jj],
                              "y": Yt[elig][ii, jj], "pred": mean[ii, jj], "sd": sd[ii, jj]})
            for a in ALPHAS:
                d[f"q{int(a * 100)}"] = np.quantile(np.concatenate(scores[v][k]), a)
            conf.append(d)
    return emp, pd.concat(conf, ignore_index=True)


if __name__ == "__main__":
    t0 = time.time()
    full = LCHG().fit(Y, X, g)
    rank = []
    for k in range(0, 5):
        for p in itertools.combinations(range(14), k):
            rank.append({"k": k, "panel": "+".join(REE[e] for e in p) if p else "none", "idx": p, "analytic_RMSE": analytic_rmse(full, p)})
    rank = pd.DataFrame(rank)
    best = rank.sort_values("analytic_RMSE").groupby("k").head(3)
    worst = rank.sort_values("analytic_RMSE").groupby("k").tail(1)
    common = rank[rank.panel.isin(["La", "Ce", "La+Ce", "La+Ce+Nd", "La+Ce+Nd+Sm"])]
    val = pd.concat([best, worst, common, rank[rank.k.between(1, 4)].groupby("k").sample(12, random_state=0)]).drop_duplicates("panel")
    rank.drop(columns="idx").to_csv(RES / "panel_design_all_analytic.csv", index=False)
    print(best[["k", "panel", "analytic_RMSE"]].round(3).to_string()); print(worst[["k", "panel", "analytic_RMSE"]].round(3).to_string(), flush=True)

    folds = grouped_folds(g, 10, 0)
    vp = list(zip(val.panel, val.idx))
    out = Parallel(n_jobs=5)(delayed(run_fold)(f, tr, te, vp) for f, (tr, te) in enumerate(folds))
    print(f"folds done {time.time() - t0:.0f}s", flush=True)
    emp = {p: np.concatenate([o[0][p] for o in out if p in o[0]]) for p in val.panel}
    val = val.assign(empirical_RMSE=[np.sqrt(emp[p].mean()) for p in val.panel], n_scored=[len(emp[p]) for p in val.panel]).drop(columns="idx")
    val.to_csv(RES / "panel_design_validation.csv", index=False)
    print(val.sort_values(["k", "analytic_RMSE"]).round(3).to_string())
    print("analytic vs empirical RMSE: Spearman %.3f | Pearson %.3f" % (
        val[["analytic_RMSE", "empirical_RMSE"]].corr(method="spearman").iloc[0, 1], val[["analytic_RMSE", "empirical_RMSE"]].corr().iloc[0, 1]))

    Cf = pd.concat([o[1] for o in out], ignore_index=True); Cf["study"] = g[Cf.row.to_numpy()]
    Cf.to_parquet(RES / "conformal_predictions.parquet", index=False)
    Cf["z"] = (Cf.y - Cf.pred).abs() / Cf.sd
    rows = []
    for (v, k), d in Cf.groupby(["variant", "k"]):
        r = {"variant": v, "k": k, "n": len(d), "RMSE": np.sqrt(((d.y - d.pred) ** 2).mean())}
        for a in ALPHAS:
            r[f"model_cov{int(a * 100)}"] = (d.z <= norm.ppf(0.5 + a / 2)).mean()
            r[f"conf_cov{int(a * 100)}"] = (d.z <= d[f"q{int(a * 100)}"]).mean()
        r["conf_halfwidth90_median"] = (d.q90 * d.sd).median(); r["conf_halfwidth90_mean"] = (d.q90 * d.sd).mean()
        per = (d.z <= d.q90).groupby(d.study).mean()
        r["study_cov90_median"] = per.median(); r["study_cov90_q10"] = per.quantile(0.1); r["studies_below_80"] = (per < 0.8).mean()
        rows.append(r)
    U = pd.DataFrame(rows); U.to_csv(RES / "conformal_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(U.round(3).to_string())
