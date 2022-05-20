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

import gseapy
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import numpy.ma as ma
import itertools as it
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.GIPlot import GIPlot
from adjustText import adjust_text
from crispy.MOFA import MOFA, MOFAPlot
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from crispy.DataImporter import CORUM, BioGRID, PPI, HuRI
from multiomics_integration.notebooks import (
    DataImport,
    two_vars_correlation,
    PALETTE_TTYPE,
    PALETTE_PERTURB,
)


LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/")


# ### Import MOFA analysis

factors, weights, rsquare = MOFA.read_mofa_hdf5(f"{TPATH}/MultiOmics_broad.hdf5")

# ### Import manifest

manifest = DataImport.read_protein_perturbation_manifest()

# Remove MDA-MB-468
manifest = manifest[~manifest["External Patient ID"].isin(["MDA-MB-468"])]

# Remove low quality samples
manifest = manifest[
    ~((manifest["Cell Line"] == "BT-549 1% FBS") & (manifest["Date on sample"] == "4/7/19 "))
]

manifest = manifest.drop(
    ["200627_b2-1-t5-1_00wuz_00yid_m03_s_1", "200623_b2-1-t4-1_00wuz_00yh3_m01_s_1"]
)

# Remove low FBS levels
manifest = manifest[["0.5%FBS" not in v for v in manifest["Cell Line"]]]


# ### Import proteomics

prot = DataImport.read_protein_perturbation(map_protein=True)[manifest.index]


# ### Replicates correlation

reps_corr = pd.DataFrame(
    [
        {
            **two_vars_correlation(prot[c1], prot[c2]),
            **dict(
                sample1=c1,
                sample2=c2,
                cellline1=manifest.loc[c1, "Cell Line"],
                cellline2=manifest.loc[c2, "Cell Line"],
            ),
        }
        for c1, c2 in it.combinations(list(prot), 2)
    ]
)

# Plot
plot_df = reps_corr.query("cellline1 == cellline2")

fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

sns.boxplot(
    "corr",
    "cellline1",
    data=plot_df,
    order=list(PALETTE_PERTURB.keys()),
    orient="h",
    saturation=1,
    palette=PALETTE_PERTURB,
    showcaps=False,
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
    "cellline1",
    data=plot_df,
    order=list(PALETTE_PERTURB.keys()),
    orient="h",
    edgecolor="white",
    palette=PALETTE_PERTURB,
    linewidth=0.1,
    s=3,
    ax=ax,
)

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel("Replicates correlation\n(Pearson's R)")
ax.set_ylabel("Condition")

plt.savefig(f"{RPATH}/Perturb_rep_corr.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/Perturb_rep_corr.png", bbox_inches="tight", dpi=600)
plt.close("all")


# ### Calculate comparisons

# Define comparisons
comparisons = dict(
    Arrested_5days=dict(
        control=list(manifest[["EXPO" in v for v in manifest["Cell Line"]]].index),
        condition=list(manifest[["Arrested (5 days" in v for v in manifest["Cell Line"]]].index),
    ),
    Arrested_8days=dict(
        control=list(manifest[["EXPO" in v for v in manifest["Cell Line"]]].index),
        condition=list(manifest[["Arrested (8 days" in v for v in manifest["Cell Line"]]].index),
    ),
    FBS=dict(
        control=list(manifest[["10% FBS" in v for v in manifest["Cell Line"]]].index),
        condition=list(manifest[["1% FBS" in v for v in manifest["Cell Line"]]].index),
    ),
)

# Differential protein abundance
comparisons_fc = []
for k, v in comparisons.items():
    df = pd.DataFrame(
        ttest_ind(
            prot[v["control"]].T,
            prot[v["condition"]].T,
            equal_var=False,
            nan_policy="omit",
        ),
        index=["tstat", "pvalue"],
        columns=prot.index,
    ).T.astype(float).sort_values("pvalue").dropna()

    df["comparison"] = k
    df["fdr"] = multipletests(df["pvalue"], method="fdr_bh")[1]
    df["diff"] = prot.loc[df.index, v["control"]].median(1) - prot.loc[df.index, v["condition"]].mean(1)

    comparisons_fc.append(df.reset_index())
comparisons_fc = pd.concat(comparisons_fc).sort_values("fdr")
comparisons_diff = pd.pivot_table(comparisons_fc, index="GeneSymbol", columns="comparison", values="diff")
comparisons_fc.to_csv(f"{DPATH}/perturbation_proteomics_diff_analysis.csv", index=False)

# Plot distribtuions
fig, ax = plt.subplots(1, 1, figsize=(2, 1), dpi=600)

sns.boxplot(
    "diff",
    "comparison",
    data=comparisons_fc,
    orient="h",
    saturation=1,
    showcaps=False,
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

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel("Differential protein abundance")
ax.set_ylabel("")

plt.savefig(f"{RPATH}/Perturb_differential_abundance_boxplot.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/Perturb_differential_abundance_boxplot.png", bbox_inches="tight", dpi=600)
plt.close("all")

# ###

comparisons_corr = (
    pd.DataFrame(
        [
            {
                **two_vars_correlation(weights["proteomics"][f], comparisons_diff[c]),
                **dict(factor=f, comparison=c),
            }
            for f in weights["proteomics"]
            for c in comparisons_diff
        ]
    )
    .sort_values("pval")
    .dropna()
)

# Plot
plot_df = pd.pivot_table(comparisons_corr, index="comparison", columns="factor", values="corr")

g = sns.clustermap(
    plot_df,
    cmap="RdYlGn",
    annot=True,
    center=0,
    fmt=".2f",
    annot_kws=dict(size=4),
    lw=0.05,
    figsize=(5, 1.5),
)
plt.savefig(f"{RPATH}/Perturb_corr_clustermap.pdf", bbox_inches="tight")
plt.savefig(
    f"{RPATH}/Perturb_corr_clustermap.png", bbox_inches="tight", dpi=600
)
plt.close("all")
