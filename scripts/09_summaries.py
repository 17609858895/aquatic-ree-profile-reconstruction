"""Summary tables from the completion comparison: pooled metrics, per-study paired tests, per-element metrics."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from ree_common import *

R = pd.concat([pd.read_parquet(p) for p in sorted(RES.glob("completion_predictions_seeds*.parquet"))
               + sorted(RES.glob("completion_tuned*_predictions_seeds*.parquet"))], ignore_index=True)
df = pd.read_csv(PROC / "biota_model.csv")
R["study"] = df.study.to_numpy()[R.row.to_numpy()]
R["e2"] = (R.y - R.pred) ** 2
PAN = ["meta-only", "La", "Tb", "La+Ce", "Pr+Yb", "La+Ce+Nd", "Ce+Nd+Yb", "LOEO"]
PROP = "LC-HG (proposed)"


def pooled(g):
    return pd.Series({"n": len(g), "R2": 1 - g.e2.sum() / ((g.y - g.y.mean()) ** 2).sum(), "RMSE": np.sqrt(g.e2.mean()),
                      "MAE": np.sqrt(g.e2).mean(), "medAE": np.sqrt(g.e2).median()})


S = R.groupby(["panel", "model"]).apply(pooled).reset_index()
# one value per study (errors pooled over repeated splits) so that repeats are not counted as independent units
ps = R.groupby(["panel", "model", "study"]).e2.mean().pow(0.5).reset_index(name="rmse")
med = ps.groupby(["panel", "model"]).rmse.median().rename("median_study_RMSE").reset_index()
S = S.merge(med)
rows = []
for p in PAN:
    a = ps[(ps.panel == p) & (ps.model == PROP)].set_index("study").rmse
    for m in ps[ps.panel == p].model.unique():
        if m == PROP:
            continue
        b = ps[(ps.panel == p) & (ps.model == m)].set_index("study").rmse
        j = a.index.intersection(b.index)
        d = b[j] - a[j]
        rows.append({"panel": p, "model": m, "n_studies": len(j), "share_studies_LCHG_better": float((d > 0).mean()),
                     "median_diff": float(d.median()), "wilcoxon_p": wilcoxon(b[j], a[j]).pvalue if (d != 0).any() else np.nan})
W = pd.DataFrame(rows)
W["p_BH"] = multipletests(W.wilcoxon_p.fillna(1), method="fdr_bh")[1]
S = S.merge(W, how="left", on=["panel", "model"])
S.to_csv(RES / "completion_summary_full.csv", index=False)
pd.set_option("display.width", 260)
for metric in ["RMSE", "R2", "median_study_RMSE"]:
    print(metric); print(S.pivot(index="model", columns="panel", values=metric)[PAN].round(3).to_string())
print("share of studies where LC-HG is better / BH-adjusted p")
print(W.pivot(index="model", columns="panel", values="share_studies_LCHG_better")[PAN].round(2).to_string())
print(W.pivot(index="model", columns="panel", values="p_BH")[PAN].map(lambda v: f"{v:.1e}").to_string())
E = R[R.model == PROP].groupby(["panel", "el"]).apply(pooled).reset_index()
E.to_csv(RES / "completion_per_element_LCHG.csv", index=False)
print("LC-HG per-element R2"); print(E.pivot(index="el", columns="panel", values="R2").reindex(REE)[PAN].round(3).to_string())
L = R[(R.model == PROP)]
L = L.assign(z=np.sqrt(L.e2) / L.sd)
print("LC-HG raw 90% coverage by panel:", L.groupby("panel").z.apply(lambda z: float((z < 1.645).mean())).reindex(PAN).round(3).to_dict())
