"""Sensitivity analyses of the pattern-completion and transfer-gap results.

Scenario groups (each evaluated with 10 held-out folds):
  exclusion      drop wet-weight / modelled / laboratory records, legacy techniques, the largest study, incomplete patterns
  normalisation  PAAS (baseline) vs CI chondrite vs NASC vs none
  structure      tetrad-effect terms, polynomial degree, anomaly set
  prior          metadata feature set of the prior
  validation     random rows vs leave-event-out vs leave-study-out (baseline)
Extras from the baseline run: common-record comparison of anchor panels; nu of the coherence-adaptive variance.
Usage: python 11_sensitivity.py [n_workers]
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import sys, time, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from ree_common import *
from lchg import LCHG, grouped_folds, lambda_ols

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 10
df = pd.read_csv(PROC / "biota_model.csv")
N = [f"n_{e}" for e in REE]
YP = df[N].to_numpy(float)                       # log10 PAAS-normalised
study = df.study.to_numpy(); event = df.Event.to_numpy()
ix = {e: j for j, e in enumerate(REE)}
META = ["Sample_group", "Phylum", "Class", "Tissue_class", "Tissues", "Trophic_position", "Habitat", "Ecosystem", "Weight"]
META_SETS = {"default": META, "biology only": ["Sample_group", "Tissue_class", "Trophic_position", "Habitat", "Ecosystem", "Weight"],
             "all metadata": META + ["REE_contamination", "Area", "Latitude", "Longitude", "abs_lat", "Species", "Device", "Reference_year", "is_LAB"]}
PANELS = {"meta-only": [], "La": ["La"], "Pr+Yb": ["Pr", "Yb"], "La+Ce+Nd": ["La", "Ce", "Nd"], "Ce+Nd+Yb": ["Ce", "Nd", "Yb"], "LOEO": None}
PAIRS = [("La", "Tb"), ("La+Ce", "Pr+Yb"), ("La+Ce+Nd", "Ce+Nd+Yb"), ("La+Ce+Nd+Sm", "Ce+Nd+Ho+Lu")]
NUS = [None, 2.0, 4.0, 8.0, 16.0]

ROWS = {
    "all records": np.ones(len(df), bool),
    "no wet-weight": (df.Weight != "WW").to_numpy(),
    "no modelled data": (df.Device != "Mod.").to_numpy(),
    "no laboratory": (df.is_LAB == 0).to_numpy(),
    "no WW/modelled/lab": ((df.Weight != "WW") & (df.Device != "Mod.") & (df.is_LAB == 0)).to_numpy(),
    "no legacy techniques": (~df.Device.isin(["X-ray emission", "INAA"])).to_numpy(),
    "no largest study": (df.study != df.study.value_counts().index[0]).to_numpy(),
    "complete patterns only": (df.n_el == 14).to_numpy(),
}
SCN = [dict(name=k, group="exclusion", rows=k) for k in ROWS]
SCN += [dict(name=k, group="normalisation", norm=k) for k in ("CI chondrite", "NASC", "none")]
SCN += [dict(name="+ tetrad terms", group="structure", kw=dict(tetrad=True)), dict(name="linear shape", group="structure", kw=dict(degree=1)),
        dict(name="quartic shape", group="structure", kw=dict(degree=4)), dict(name="Ce+Eu anomalies only", group="structure", kw=dict(anomalies=("Ce", "Eu")))]
SCN += [dict(name="prior: biology only", group="prior", meta="biology only"), dict(name="prior: all metadata", group="prior", meta="all metadata")]
SCN += [dict(name="random rows", group="validation", split="random"), dict(name="leave-event-out", group="validation", split="event")]


def design(cols, rows):
    X = df.loc[rows, cols].copy().reset_index(drop=True)
    for c in cols:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    return X


def cases(Yte, panel):
    o = ~np.isnan(Yte)
    if panel is None:
        for e in range(14):
            Yo = Yte.copy(); Yo[:, e] = np.nan
            sm = np.zeros_like(o); sm[:, e] = o[:, e] & (o.sum(1) >= 2)
            yield Yo, sm
    else:
        pj = [ix[e] for e in panel]
        ok = o[:, pj].all(1) if pj else np.ones(len(Yte), bool)
        Yo = np.full_like(Yte, np.nan); Yo[:, pj] = Yte[:, pj]
        sm = o.copy(); sm[:, pj] = False; sm[~ok] = False
        yield Yo, sm


def get_folds(scn, rows):
    split = scn.get("split", "study")
    if split == "random":
        return list(KFold(10, shuffle=True, random_state=0).split(np.arange(rows.sum())))
    return grouped_folds((event if split == "event" else study)[rows], 10, 0)


def run(si, f):
    scn = SCN[si]
    rows = ROWS[scn.get("rows", "all records")]
    Y = YP[rows] + LOG_PAAS - LOG_REF[scn.get("norm", "PAAS")]
    X = design(META_SETS[scn.get("meta", "default")], rows); g = study[rows]
    tr, te = get_folds(scn, rows)[f]
    mod = LCHG(**scn.get("kw", {})).fit(Y[tr], X.iloc[tr].reset_index(drop=True), g[tr])
    mod.nu = 4.0
    Xte = X.iloc[te].reset_index(drop=True)
    out, extra = [], {}
    base = scn["name"] == "all records"
    for pn, panel in PANELS.items():
        acc = {"LC-HG": [0.0, 0, 0, 0.0, 0.0], "lambda-OLS": [0.0, 0, 0, 0.0, 0.0]}
        for Yo, sm in cases(Y[te], panel):
            r = sm.any(1)
            if not r.any():
                continue
            mean, sd = mod.predict(Yo[r], Xte[r].reset_index(drop=True))
            yt = Y[te][r][sm[r]]; e = yt - mean[sm[r]]
            a = acc["LC-HG"]; a[0] += float((e ** 2).sum()); a[1] += len(e); a[2] += int((np.abs(e) / sd[sm[r]] < 1.645).sum())
            a[3] += float(yt.sum()); a[4] += float((yt ** 2).sum())
            if scn["group"] in ("normalisation",) or base:
                if panel:   # lambda-OLS needs at least one anchor
                    e2 = yt - lambda_ols(Yo[r])[sm[r]]
                    b = acc["lambda-OLS"]; b[0] += float((e2 ** 2).sum()); b[1] += len(e2); b[3] += float(yt.sum()); b[4] += float((yt ** 2).sum())
                elif panel is None:
                    e2 = yt - lambda_ols(Yo[r])[sm[r]]
                    b = acc["lambda-OLS"]; b[0] += float((e2 ** 2).sum()); b[1] += len(e2); b[3] += float(yt.sum()); b[4] += float((yt ** 2).sum())
        for m, a in acc.items():
            if a[1]:
                out.append(dict(scenario=scn["name"], group=scn["group"], fold=f, panel=pn, model=m, sse=a[0], n=a[1], hit90=a[2], sy=a[3], syy=a[4]))
    if base:
        # (i) anchor panels compared on identical records and identical target elements
        comp = []
        for pa, pb in PAIRS:
            ea, eb = [ix[e] for e in pa.split("+")], [ix[e] for e in pb.split("+")]
            un = sorted(set(ea) | set(eb)); o = ~np.isnan(Y[te]); ok = o[:, un].all(1)
            if ok.sum() == 0:
                continue
            sm = o[ok].copy(); sm[:, un] = False
            for lab, pj in ((pa, ea), (pb, eb)):
                Yo = np.full((ok.sum(), 14), np.nan); Yo[:, pj] = Y[te][ok][:, pj]
                mean, _ = mod.predict(Yo, Xte[ok].reset_index(drop=True))
                comp.append(dict(pair=f"{pa} vs {pb}", panel=lab, fold=f, sse=float((((Y[te][ok] - mean) ** 2)[sm]).sum()), n=int(sm.sum()),
                                 n_records=int(ok.sum())))
        extra["common"] = comp
        # (ii) nu of the coherence-adaptive variance: normalised scores for cross-fold conformal calibration
        zs = []
        for pn in ("Pr+Yb", "LOEO"):
            for nu in NUS:
                mod.nu = nu
                for Yo, sm in cases(Y[te], PANELS[pn]):
                    r = sm.any(1)
                    if not r.any():
                        continue
                    mean, sd = mod.predict(Yo[r], Xte[r].reset_index(drop=True))
                    zs.append(pd.DataFrame({"panel": pn, "nu": "Gaussian" if nu is None else str(int(nu)), "fold": f,
                                            "z": (np.abs(Y[te][r] - mean) / sd)[sm[r]], "sd": sd[sm[r]]}))
        mod.nu = 4.0
        extra["nu"] = pd.concat(zs, ignore_index=True)
    return out, extra


def analytic_table(name):
    rows = ROWS[name]
    m = LCHG().fit(YP[rows], design(META, rows), study[rows])
    Si = np.linalg.inv(m.Sigma); res = {}
    for k in (1, 2, 3):
        for p in itertools.combinations(range(14), k):
            Bo = m.B[list(p)]
            V = np.linalg.inv(Si + (Bo.T / m.s2[list(p)]) @ Bo)
            var = np.einsum("ek,kl,el->e", m.B, V, m.B) + m.s2
            res["+".join(REE[e] for e in p)] = float(np.sqrt(var[[e for e in range(14) if e not in p]].mean()))
    return name, pd.Series(res)


def anomaly_table(norm):
    Y = YP + LOG_PAAS - LOG_REF[norm]
    m = LCHG().fit(Y, design(META, np.ones(len(df), bool)), study)
    _, _, C, _ = m.predict(Y, design(META, np.ones(len(df), bool)), return_coef=True)
    return norm, pd.DataFrame({c: C[:, m.cols.index(c)] for c in m.cols})


def l0_gap(name):
    """Metadata-only REE level: random vs leave-event-out vs leave-study-out, fixed regularised learners."""
    rows = ROWS[name]
    y = np.nanmean(YP[rows], 1); X = design(META, rows); g, ev = study[rows], event[rows]
    out = {"scenario": name, "n": int(rows.sum()), "n_studies": len(np.unique(g))}
    splits = {"random": list(KFold(10, shuffle=True, random_state=0).split(X)), "event": grouped_folds(ev, 10, 0), "study": grouped_folds(g, 10, 0)}
    for sn, folds in splits.items():
        p1 = np.zeros(len(y)); p2 = np.zeros(len(y))
        for tr, te in folds:
            p1[te] = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, reg_lambda=2.0, cat_smooth=10, verbose=-1, n_jobs=1, random_state=0).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            oh = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False).fit(X.iloc[tr].astype(str))
            p2[te] = Ridge(alpha=3.0).fit(oh.transform(X.iloc[tr].astype(str)), y[tr]).predict(oh.transform(X.iloc[te].astype(str)))
        sst = ((y - y.mean()) ** 2).sum()
        out[f"LightGBM_{sn}"] = 1 - ((y - p1) ** 2).sum() / sst; out[f"Ridge_{sn}"] = 1 - ((y - p2) ** 2).sum() / sst
    return out


if __name__ == "__main__":
    t0 = time.time()
    jobs = [delayed(run)(si, f) for si in range(len(SCN)) for f in range(10)]
    res = Parallel(n_jobs=NW, verbose=5)(jobs)
    print(f"scenario folds done {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame([r for o, _ in res for r in o])
    A = R.groupby(["group", "scenario", "panel", "model"], sort=False).agg(sse=("sse", "sum"), n=("n", "sum"), hit=("hit90", "sum"), sy=("sy", "sum"), syy=("syy", "sum")).reset_index()
    A["RMSE"] = np.sqrt(A.sse / A.n); A["R2"] = 1 - A.sse / (A.syy - A.sy ** 2 / A.n); A["raw_cov90"] = np.where(A.model == "LC-HG", A.hit / A.n, np.nan)
    nrec = {s["name"]: int(ROWS[s.get("rows", "all records")].sum()) for s in SCN}
    nst = {s["name"]: len(np.unique(study[ROWS[s.get("rows", "all records")]])) for s in SCN}
    A["n_records"] = A.scenario.map(nrec); A["n_studies"] = A.scenario.map(nst)
    A.drop(columns=["sse", "hit", "sy", "syy"]).to_csv(RES / "sensitivity_completion.csv", index=False)
    pd.set_option("display.width", 250)
    L = A[A.model == "LC-HG"]
    print(L.pivot(index="scenario", columns="panel", values="RMSE").loc[[s["name"] for s in SCN], list(PANELS)].round(3).to_string())
    print("lambda-OLS RMSE by normalisation")
    print(A[A.model == "lambda-OLS"].pivot(index="scenario", columns="panel", values="RMSE").round(3).to_string())
    print("raw 90% coverage (LC-HG)"); print(L.pivot(index="scenario", columns="panel", values="raw_cov90").loc[[s["name"] for s in SCN], list(PANELS)].round(3).to_string())

    Cm = pd.DataFrame([r for _, ex in res for r in ex.get("common", [])])
    Cm = Cm.groupby(["pair", "panel"], sort=False).agg(sse=("sse", "sum"), n=("n", "sum"), n_records=("n_records", "sum")).reset_index()
    Cm["RMSE"] = np.sqrt(Cm.sse / Cm.n); Cm.drop(columns="sse").to_csv(RES / "sensitivity_common_record_panels.csv", index=False)
    print(Cm.round(3).to_string())

    Z = pd.concat([ex["nu"] for _, ex in res if "nu" in ex], ignore_index=True)
    rows = []
    for (pn, nu), d in Z.groupby(["panel", "nu"], sort=False):
        hw, hit = [], 0
        for f in range(10):
            q = np.quantile(d.z[d.fold != f], 0.9); te = d[d.fold == f]
            hw.append(q * te.sd.to_numpy()); hit += int((te.z <= q).sum())
        hw = np.concatenate(hw)
        rows.append(dict(panel=pn, nu=nu, raw_cov90=float((d.z < 1.645).mean()), conformal_cov90=hit / len(d), median_halfwidth90=float(np.median(hw)),
                         mean_halfwidth90=float(hw.mean())))
    NU = pd.DataFrame(rows); NU.to_csv(RES / "sensitivity_nu.csv", index=False); print(NU.round(3).to_string())

    T = dict(Parallel(n_jobs=min(NW, 8))(delayed(analytic_table)(k) for k in ROWS))
    T = pd.DataFrame(T); T.to_csv(RES / "sensitivity_panel_ranking_analytic.csv")
    k = T.index.str.count(r"\+") + 1
    rk = []
    for c in T.columns:
        r = {"scenario": c, "spearman_vs_all": spearmanr(T["all records"], T[c])[0]}
        for kk in (1, 2, 3):
            r[f"best_k{kk}"] = T.loc[k == kk, c].idxmin(); r[f"best_k{kk}_RMSE"] = T.loc[k == kk, c].min()
            b0 = T.loc[k == kk, "all records"].idxmin()
            r[f"rank_of_baseline_best_k{kk}"] = int(T.loc[k == kk, c].rank()[b0])
        rk.append(r)
    RK = pd.DataFrame(rk); RK.to_csv(RES / "sensitivity_panel_ranking_summary.csv", index=False); print(RK.round(3).to_string())

    An = dict(Parallel(n_jobs=4)(delayed(anomaly_table)(k) for k in LOG_REF))
    rows = []
    for nm, d in An.items():
        for c in ["lambda0", "lambda1", "lambda2", "lambda3", "anom_Ce", "anom_Eu", "anom_Gd"]:
            rows.append(dict(normalisation=nm, coefficient=c, r_vs_PAAS=float(np.corrcoef(An["PAAS"][c], d[c])[0, 1]),
                             median_offset=float(np.median(d[c] - An["PAAS"][c])), sd_of_difference=float(np.std(d[c] - An["PAAS"][c]))))
    AN = pd.DataFrame(rows); AN.to_csv(RES / "sensitivity_normalisation_coefficients.csv", index=False)
    print(AN.pivot(index="coefficient", columns="normalisation", values="r_vs_PAAS").round(4).to_string())
    print(AN.pivot(index="coefficient", columns="normalisation", values="sd_of_difference").round(4).to_string())

    G = pd.DataFrame(Parallel(n_jobs=min(NW, 8))(delayed(l0_gap)(k) for k in ROWS))
    G.to_csv(RES / "sensitivity_transfer_gap.csv", index=False); print(G.round(3).to_string())
    print(f"total {time.time() - t0:.0f}s")
