"""Build the analysis table: biota rows, log10 shale(PAAS)-normalised REE, study groups."""
import numpy as np, pandas as pd
from ree_common import *

d = load_raw()
d["medium"] = np.where(d.Sample_group == "Water", "water", np.where(d.Sample_group == "Sediment", "sediment", "biota"))
d["n_el"] = d[REE].notna().sum(1)
bio = d[(d.medium == "biota")].copy()
for j, e in enumerate(REE):
    bio[f"n_{e}"] = np.log10(bio[e]) - LOG_PAAS[j]          # log10 PAAS-normalised
bio["study"] = bio["Reference"]
bio.to_csv(PROC / "biota_all.csv", index=False, encoding="utf-8-sig")
use = bio[bio.n_el >= 1].reset_index(drop=True)
use.to_csv(PROC / "biota_model.csv", index=False, encoding="utf-8-sig")
d.to_csv(PROC / "all_media.csv", index=False, encoding="utf-8-sig")
print("biota rows", len(bio), "| with >=1 element", len(use), "| complete 14", int((use.n_el == 14).sum()),
      "| studies", use.study.nunique(), "| events", use.Event.nunique())
print("WW rows:", int((use.Weight == "WW").sum()), "| LAB rows:", int(use.is_LAB.sum()), "| Mod. device:", int((use.Device == "Mod.").sum()))
print(use.groupby("study").size().describe().round(1).to_dict())
