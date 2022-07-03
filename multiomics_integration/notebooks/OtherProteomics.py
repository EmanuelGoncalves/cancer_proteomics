# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %autosave 0
# %load_ext autoreload
# %autoreload 2

import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.Utils import Utils
from crispy.GIPlot import GIPlot
from scipy.stats import skew, zscore
from crispy.DataImporter import PPI
from sklearn.decomposition import PCA
from crispy.LMModels import LMModels, LModel
from multiomics_integration.notebooks import DataImport, two_vars_correlation


LOG = logging.getLogger("multiomics_integration")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("multiomics_integration", "plots/DIANN/")


# ### Imports

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
protein = pd.read_csv(
    f"{DPATH}/e0022_diann_051021_working_matrix_averaged.txt",
    sep="\t",
    index_col=0,
).T

protein.columns = [c.split(";")[1] for c in protein]
protein.index = [c.split(";")[1] for c in protein.index]

# protein = zscore(protein, nan_policy="omit")
protein = zscore(protein.T, nan_policy="omit").T

# Other proteomics
prot_ccle = pd.read_csv(f"{DPATH}/other_proteomics/ccle_proteins_sanger_name.csv")
prot_ccle = prot_ccle[~prot_ccle["Cell_line"].isna()].set_index("Cell_line").T
prot_ccle = prot_ccle.reindex(
    columns=set.intersection(set(protein), set(prot_ccle)),
    index=set.intersection(set(protein.index), set(prot_ccle.index)),
)
# prot_ccle = zscore(prot_ccle, nan_policy="omit")
prot_ccle = zscore(prot_ccle.T, nan_policy="omit").T
print(f"Broad CCLE: {prot_ccle.shape}")

prot_coread = pd.read_csv(
    f"{DPATH}/other_proteomics/coread_processed.csv", index_col=0
).T
prot_coread = prot_coread.reindex(
    columns=set.intersection(set(protein), set(prot_coread)),
    index=set.intersection(set(protein.index), set(prot_coread.index)),
)
# prot_coread = zscore(prot_coread, nan_policy="omit")
prot_coread = zscore(prot_coread.T, nan_policy="omit").T
print(f"Roumelliotis et al. 2017: {prot_coread.shape}")

prot_nci60 = pd.read_csv(
    f"{DPATH}/other_proteomics/tianan_protein_processed_common.csv", index_col=0
).T
prot_nci60 = prot_nci60.reindex(
    columns=set.intersection(set(protein), set(prot_nci60)),
    index=set.intersection(set(protein.index), set(prot_nci60.index)),
)
# prot_nci60 = zscore(prot_nci60, nan_policy="omit")
prot_nci60 = zscore(prot_nci60.T, nan_policy="omit").T
print(f"Guo et al. 2019: {prot_nci60.shape}")

prot_tnbc = pd.read_csv(f"{DPATH}/other_proteomics/TNBC_processed.csv", index_col=0).T
prot_tnbc = prot_tnbc.reindex(
    columns=set.intersection(set(protein), set(prot_tnbc)),
    index=set.intersection(set(protein.index), set(prot_tnbc.index)),
)
# prot_tnbc = zscore(prot_tnbc, nan_policy="omit")
prot_tnbc = zscore(prot_tnbc.T, nan_policy="omit").T
print(f"Lawrence et al. 2015: {prot_tnbc.shape}")

prot_frejno = pd.read_csv(
    f"{DPATH}/other_proteomics/protein_combined_common_processed.csv", index_col=0
).T
prot_frejno = prot_frejno.T.groupby(prot_frejno.columns).mean().T
prot_frejno = prot_frejno.reindex(
    columns=set.intersection(set(protein), set(prot_frejno)),
    index=set.intersection(set(protein.index), set(prot_frejno.index)),
)
# prot_frejno = zscore(prot_frejno, nan_policy="omit")
prot_frejno = zscore(prot_frejno.T, nan_policy="omit").T
print(f"Frenjo et al. 2020: {prot_frejno.shape}")

## Correlate samples
dfs = dict(
    ccle={"df": prot_ccle, "name": "Nusinow et al. 2020"},
    coread={"df": prot_coread, "name": "Roumelliotis et al. 2017"},
    nci60={"df": prot_nci60, "name": "Guo et al. 2019"},
    tnbc={"df": prot_tnbc, "name": "Lawrence et al. 2015"},
    frejno={"df": prot_frejno, "name": "Frenjo et al. 2020"},
)

# Samples correlation
reps_corr = pd.DataFrame(
    [
        {
            **two_vars_correlation(protein[c], dfs[n]["df"][c]),
            **dict(
                dataset=n,
                dataset_name=dfs[n]["name"],
                cellline=c,
            ),
        }
        for n in dfs
        for c in dfs[n]["df"]
    ]
)

fig, ax = plt.subplots(1, 1, figsize=(2, 1.25), dpi=600)

sns.boxplot(
    "corr",
    "dataset",
    data=reps_corr,
    orient="h",
    color=GIPlot.PAL_DTRACE[0],
    saturation=1,
    showcaps=False,
    sym="",
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    flierprops=dict(
        marker="o",
        markerfacecolor="black",
        markersize=1.0,
        linestyle="none",
        markeredgecolor="none",
        alpha=0.6,
    ),
    ax=ax,
)

sns.stripplot(
    "corr",
    "dataset",
    data=reps_corr,
    color=GIPlot.PAL_DTRACE[2],
    orient="h",
    edgecolor="white",
    linewidth=0.1,
    s=2,
    ax=ax,
)

ax.set_yticklabels([f"{dfs[d.get_text()]['name']} (N={dfs[d.get_text()]['df'].shape[1]}, median r={reps_corr[reps_corr['dataset'] == d.get_text()]['corr'].median():.2f})" for d in ax.get_yticklabels()])

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel("Pearson's r\nsame cell line")
ax.set_ylabel("")

plt.savefig(f"{RPATH}/OtherProteomics_corr.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/OtherProteomics_corr.png", bbox_inches="tight", dpi=600)
plt.close("all")

# All versus all
dfs = dict(
    ccle={"df": prot_ccle, "name": "Nusinow et al. 2020"},
    coread={"df": prot_coread, "name": "Roumelliotis et al. 2017"},
    nci60={"df": prot_nci60, "name": "Guo et al. 2019"},
    tnbc={"df": prot_tnbc, "name": "Lawrence et al. 2015"},
    frejno={"df": prot_frejno, "name": "Frenjo et al. 2020"},
    cmrisanger={"df": protein, "name": "ProCan-DepMapSanger"},
)

reps_corr = pd.DataFrame(
    [
        {
            **two_vars_correlation(dfs[n1]["df"][c], dfs[n2]["df"][c]),
            **dict(
                dataset1=n1,
                dataset1_name=dfs[n1]["name"],
                dataset2=n2,
                dataset2_name=dfs[n2]["name"],
                cellline=c,
            ),
        }
        for n1 in dfs
        for n2 in dfs
        for c in set(dfs[n1]["df"]).intersection(dfs[n2]["df"])
    ]
)

reps_corr_m = pd.pivot_table(reps_corr, index="dataset1_name", columns="dataset2_name", values="corr", aggfunc=np.mean)

fig = sns.clustermap(
    reps_corr_m.replace(np.nan, 0),
    cmap="RdBu",
    center=0,
    annot=True,
    fmt=".2f",
    mask=reps_corr_m.isna(),
    annot_kws={"size": 7},
    figsize=(4, 4),
)
fig.ax_heatmap.set_xlabel("Data-sets")
fig.ax_heatmap.set_ylabel("Data-sets")

fig.ax_heatmap.set_yticklabels(fig.ax_heatmap.get_yticklabels(), rotation=0)

fig.ax_col_dendrogram.set_title("Mean cell lines pearson's r")

plt.savefig(f"{RPATH}/OtherProteomics_all_clustermap.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/OtherProteomics_all_clustermap.png", bbox_inches="tight", dpi=600)
plt.close("all")


#
