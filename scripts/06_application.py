"""Applications of the fitted LC-HG model on the full compilation.

(1) Harmonised 14-element patterns and SumREE with posterior uncertainty for every biota record.
(2) Anomaly atlas: Ce, Eu, Gd anomalies with posterior SD, compared with neighbour-interpolation formulas.
(3) Missingness mechanism check (are unreported elements those expected to be low?).
(4) Dry-weight organism-group comparisons in records reported as uncontaminated.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from ree_common import *
from lchg import LCHG

df = pd.read_csv(PROC / "biota_model.csv")
N = [f"n_{e}" for e in REE]
Y = df[N].to_numpy(float); obs = ~np.isnan(Y); n = len(Y)
META = ["Sample_group", "Phylum", "Class", "Tissue_class", "Tissues", "Trophic_position", "Habitat", "Ecosystem", "Weight"]
X = df[META].copy()
for c in META:
    X[c] = X[c].astype("category")
g = df.study.to_numpy()
m = LCHG().fit(Y, X, g)
mean, sd, C, V = m.predict(Y, X, return_coef=True)
cols = m.cols

# ---- (1) harmonised patterns and SumREE (Monte Carlo over posterior)
rng = np.random.default_rng(0)
S = 400
L = np.linalg.cholesky(V + 1e-10 * np.eye(V.shape[1]))
draw = C[None] + np.einsum("nkl,snl->snk", L, rng.normal(size=(S, n, V.shape[1])))
Ys = draw @ m.B.T + m.r0 + rng.normal(size=(S, n, 14)) * np.sqrt(m.s2)
Ys = np.where(obs[None], Y[None], Ys)                       # measured values are kept as measured
conc = 10 ** (Ys + LOG_PAAS)                                # ug/kg
sumree = conc.sum(2)
H = df[["Data_code", "study", "Event", "Sample_group", "Phylum", "Class", "Species", "Tissues", "Tissue_class", "Trophic_position",
        "Habitat", "Ecosystem", "Area", "Weight", "REE_contamination", "n_el", "SumREE"]].copy()
comp = np.where(obs, Y, mean)
for j, e in enumerate(REE):
    H[f"{e}_harm_ugkg"] = 10 ** (comp[:, j] + LOG_PAAS[j])
    H[f"{e}_measured"] = obs[:, j]
    H[f"{e}_sd_log10"] = np.where(obs[:, j], 0.0, sd[:, j])
H["SumREE14_median"] = np.median(sumree, 0)
H["SumREE14_lo95"], H["SumREE14_hi95"] = np.percentile(sumree, [2.5, 97.5], 0)
H["SumREE14_fold_unc"] = H.SumREE14_hi95 / H.SumREE14_lo95
for k, c in enumerate(cols):
    H[c] = C[:, k]; H[c + "_sd"] = np.sqrt(V[:, k, k])
H.to_csv(RES / "harmonised_biota_REE.csv", index=False, encoding="utf-8-sig")
rep = H[H.SumREE.notna() & (H.n_el >= 1)]
print("records:", n, "| complete as measured:", int((H.n_el == 14).sum()), "| completed by model:", int((H.n_el < 14).sum()))
print("median 95% fold-uncertainty of SumREE14 by n measured:")
print(H.groupby(pd.cut(H.n_el, [0, 1, 3, 6, 10, 13, 14])).SumREE14_fold_unc.median().round(2).to_string())
print("author SumREE vs harmonised (n=%d): r(log)=%.3f, median ratio harm/author=%.2f" % (
    len(rep), np.corrcoef(np.log10(rep.SumREE), np.log10(rep.SumREE14_median))[0, 1], (rep.SumREE14_median / rep.SumREE).median()))

# ---- (2) anomaly atlas
An = pd.DataFrame({"Data_code": df.Data_code})
sm = np.array([e not in ("Ce", "Eu", "Gd") for e in REE])
neigh = {"Ce": ("La", "Pr", 0.5, 0.5), "Eu": ("Sm", "Gd", 0.5, 0.5), "Gd": ("Sm", "Tb", 1 / 3, 2 / 3)}
ix = {e: j for j, e in enumerate(REE)}
for a in ("Ce", "Eu", "Gd"):
    k = cols.index(f"anom_{a}")
    An[f"{a}_anom_log10"] = C[:, k]; An[f"{a}_anom_sd"] = np.sqrt(V[:, k, k])
    An[f"{a}_measured"] = obs[:, ix[a]]
    l, r, wl, wr = neigh[a]
    An[f"{a}_classic_log10"] = Y[:, ix[a]] - (wl * Y[:, ix[l]] + wr * Y[:, ix[r]])   # geometric interpolation
    okc = An[f"{a}_classic_log10"].notna()
    okm = An[f"{a}_measured"] & (An[f"{a}_anom_sd"] < 0.15)
    both = okc & okm
    print(f"{a}: classical computable n={okc.sum()} | LC-HG resolvable (measured & sd<0.15) n={okm.sum()} "
          f"| agreement r={np.corrcoef(An.loc[both, f'{a}_classic_log10'], An.loc[both, f'{a}_anom_log10'])[0, 1]:.3f} "
          f"| significant (|z|>2): {int(((An[f'{a}_anom_log10'].abs() / An[f'{a}_anom_sd']) > 2)[okm].sum())}")
An = pd.concat([An, df[["study", "Sample_group", "Tissue_class", "Ecosystem", "Habitat", "REE_contamination", "Area"]]], axis=1)
An.to_csv(RES / "anomaly_atlas.csv", index=False, encoding="utf-8-sig")
q = An[An.Gd_measured & (An.Gd_anom_sd < 0.15)]
print("Gd anomaly (log10) medians by ecosystem x contamination:")
print(q.groupby(["Ecosystem", "REE_contamination"]).Gd_anom_log10.agg(["median", "count"]).round(3).to_string())
q = An[An.Ce_measured & (An.Ce_anom_sd < 0.15)]
print("Ce anomaly by ecosystem:", q.groupby("Ecosystem").Ce_anom_log10.median().round(3).to_dict())
q = An[An.Eu_measured & (An.Eu_anom_sd < 0.15)]
print("Eu anomaly by tissue class:", q.groupby("Tissue_class").Eu_anom_log10.median().round(3).to_dict())

# ---- (3) missingness mechanism: predicted level of unreported elements vs reported, within records with >=6 elements
rows = []
rich = obs.sum(1) >= 6
for j, e in enumerate(REE):
    miss = rich & ~obs[:, j]; have = rich & obs[:, j]
    if miss.sum() >= 15:
        # raw (un-normalised) log10 ug/kg predicted for missing vs measured for reported
        pm = mean[miss, j] + LOG_PAAS[j]; ph = Y[have, j] + LOG_PAAS[j]
        rows.append({"element": e, "n_missing": int(miss.sum()), "n_reported": int(have.sum()),
                     "median_pred_missing_log10ugkg": np.median(pm), "median_reported_log10ugkg": np.median(ph),
                     "share_missing_below_q10_reported": np.mean(pm < np.quantile(ph, 0.10))})
MN = pd.DataFrame(rows); MN.to_csv(RES / "missingness_mechanism.csv", index=False)
print(MN.round(3).to_string())

# ---- (4) Organism-group summaries
Hn = H[(H.Weight == "DW") & (H.REE_contamination == "No")]
rows = []
for grp, d in Hn.groupby("Sample_group"):
    c = d[d.n_el == 14]
    rows.append(dict(group=grp, n_complete=len(c), n_harmonised=len(d), studies_complete=c.study.nunique(), studies_harmonised=d.study.nunique(),
                     median_complete=c.SumREE14_median.median() if len(c) else np.nan, median_harmonised=d.SumREE14_median.median(),
                     q25=d.SumREE14_median.quantile(0.25), q75=d.SumREE14_median.quantile(0.75)))
Cb = pd.DataFrame(rows); Cb["fold_change"] = Cb.median_harmonised / Cb.median_complete
Cb.to_csv(RES / "ext_baseline_shift.csv", index=False)
print(Cb.round(2).to_string(), flush=True)

