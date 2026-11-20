"""Shared constants and helpers for the global aquatic REE ML project.

Source data: Pereto, Baudrimont & Coynel (2024) PANGAEA 10.1594/PANGAEA.963631 (CC BY 4.0).
Concentrations: ug/kg for biota and sediment, ug/L for water. The raw workbook is never modified.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data_extracted" / "Global_compilation_of_Rare_Earth_Element_PERETO-etal_2023.xlsx"
PROC = ROOT / "data_processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
for p in (PROC, RES, FIG):
    p.mkdir(exist_ok=True)

REE = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Z = np.array([57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71])
# Shannon (1976) effective ionic radii of REE3+ in 8-fold coordination (Angstrom)
R8 = np.array([1.160, 1.143, 1.126, 1.109, 1.079, 1.066, 1.053, 1.040, 1.027, 1.015, 1.004, 0.994, 0.985, 0.977])
# Revised PAAS: McLennan (1989), Table 2, p. 172; ug/g -> ug/kg below.
PAAS_PPM = np.array([38.2, 79.6, 8.83, 33.9, 5.55, 1.08, 4.66, 0.774, 4.68, 0.991, 2.85, 0.405, 2.82, 0.433])
LOG_PAAS = np.log10(PAAS_PPM * 1000.0)  # log10 ug/kg

TISSUE_CLASS = {
    "whole organism": "whole", "carcass": "whole", "skinless body": "whole",
    "shell": "hard", "carapace": "hard", "tube": "hard", "scute": "hard", "skeleton": "hard",
    "beak": "hard", "gladius": "hard", "feather": "hard", "fur": "hard", "cartilage": "hard",
    "muscle": "muscle", "foot": "muscle", "mantle": "muscle", "arm": "muscle", "fin": "muscle",
    "funnel": "muscle", "mouth": "muscle", "head": "muscle",
    "gill": "organ", "liver": "organ", "digestive gland": "organ", "kidney": "organ",
    "hepatopancreas": "organ", "spleen": "organ", "pancreas": "organ", "internal organs": "organ",
    "stomach": "organ", "caecum": "organ", "bile": "organ", "urinary system": "organ",
    "heart": "organ", "blood": "organ", "ganglion": "organ", "eye": "organ", "faeces": "organ",
    "gonad": "reproductive", "oocytes": "reproductive", "mature ova": "reproductive",
    "ovarian sac": "reproductive", "seamen fluid": "reproductive", "nidamental gland": "reproductive",
    "ink sac": "reproductive",
    "skin": "integument", "blubber tissue": "integument",
    "leaves": "plant_aerial", "stem": "plant_aerial", "flower": "plant_aerial",
    "flower stalks": "plant_aerial", "shoot": "plant_aerial", "bud": "plant_aerial",
    "root": "plant_below", "rhizome": "plant_below",
}

CAT_COLS = ["Sample_group", "Phylum", "Class", "Species", "Tissues", "Tissue_class", "Trophic_position",
            "Habitat", "Ecosystem", "Area", "Country", "Weight", "Device", "REE_contamination"]
NUM_COLS = ["Latitude", "Longitude", "abs_lat", "Reference_year", "is_LAB"]
# Predictors used for metadata-only prediction. Country/Species are high-cardinality.
FEATS_CAT = ["Sample_group", "Phylum", "Class", "Species", "Tissue_class", "Tissues", "Trophic_position",
             "Habitat", "Ecosystem", "Area", "Weight", "Device", "REE_contamination"]
FEATS_NUM = ["Latitude", "Longitude", "abs_lat", "Reference_year", "is_LAB"]


def load_raw():
    d = pd.read_excel(RAW_XLSX, sheet_name="Data table")
    sum_col = [c for c in d.columns if "REE" in c and c not in REE and c != "REE_contamination"][0]
    d = d.rename(columns={sum_col: "SumREE"})
    for c in REE + ["SumREE"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["Sample_group", "Phylum", "Class", "Species", "Tissues", "Trophic_position", "Habitat",
              "Ecosystem", "Area", "Country", "Weight", "Device", "REE_contamination"]:
        d[c] = d[c].astype(str).str.strip()
    d["Tissues"] = d["Tissues"].str.lower()
    d["Tissue_class"] = d["Tissues"].map(TISSUE_CLASS)
    d.loc[d.Sample_group == "Water", "Tissue_class"] = "water"
    d.loc[d.Sample_group == "Sediment", "Tissue_class"] = "sediment"
    d["Tissue_class"] = d["Tissue_class"].fillna("other")
    d["is_LAB"] = (d["Latitude"].astype(str).str.upper() == "LAB").astype(int)
    d["Latitude"] = pd.to_numeric(d["Latitude"], errors="coerce")
    d["Longitude"] = pd.to_numeric(d["Longitude"], errors="coerce")
    d["abs_lat"] = d["Latitude"].abs()
    return d


# Other normalising references (ug/g). Constant per-element shifts in log space; values to be re-checked against the
# primary sources before publication (CI chondrite: McDonough & Sun 1995; NASC: Gromet et al. 1984 / Haskin et al. 1968).
CI_PPM = np.array([0.237, 0.613, 0.0928, 0.457, 0.148, 0.0563, 0.199, 0.0361, 0.246, 0.0546, 0.160, 0.0247, 0.161, 0.0246])
NASC_PPM = np.array([32.0, 73.0, 7.9, 33.0, 5.7, 1.24, 5.2, 0.85, 5.8, 1.04, 3.4, 0.50, 3.1, 0.48])
LOG_REF = {"PAAS": LOG_PAAS, "CI chondrite": np.log10(CI_PPM * 1000.0), "NASC": np.log10(NASC_PPM * 1000.0), "none": np.zeros(14)}
# number of 4f electrons of REE3+ (Pm, q = 4, is absent) and the four tetrads used for the optional tetrad-effect terms
Q4F = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], float)
TETRADS = [(0.0, 3.5), (3.5, 7.0), (7.0, 10.5), (10.5, 14.0)]


def basis_matrix(anomalies=("Ce", "Eu", "Gd"), degree=3, tetrad=False):
    """Orthonormal polynomial basis in ionic radius (lambda-shape) + unit anomaly vectors.

    The smooth basis is orthogonalised over the elements that are not anomaly-prone,
    following the lambda-coefficient idea of O'Neill (2016) for lanthanide-contraction
    controlled patterns. Returns B (14 x K) and the column names.
    """
    mask = np.array([e not in anomalies for e in REE])
    r = R8 - R8[mask].mean()
    V = np.vstack([r ** k for k in range(degree + 1)]).T  # 14 x (d+1)
    # Gram-Schmidt on the non-anomalous rows only
    Q = np.zeros_like(V)
    for k in range(V.shape[1]):
        v = V[:, k].copy()
        for j in range(k):
            v -= (v[mask] @ Q[mask, j]) / (Q[mask, j] @ Q[mask, j]) * Q[:, j]
        Q[:, k] = v / np.sqrt(v[mask] @ v[mask])
    cols = [f"lambda{k}" for k in range(degree + 1)]
    if tetrad:   # one downward-opening parabola per tetrad, zero at the tetrad boundaries (Anenburg & Williams 2022 style)
        T = [np.where((Q4F >= a) & (Q4F <= b), (Q4F - a) * (b - Q4F) / ((b - a) / 2) ** 2, 0.0) for a, b in TETRADS]
        Q = np.hstack([Q, np.array(T).T]); cols += [f"tau{i + 1}" for i in range(4)]
    A = []
    for a in anomalies:
        e = np.zeros(14); e[REE.index(a)] = 1.0
        A.append(e); cols.append(f"anom_{a}")
    B = np.hstack([Q, np.array(A).T]) if A else Q
    return B, cols
