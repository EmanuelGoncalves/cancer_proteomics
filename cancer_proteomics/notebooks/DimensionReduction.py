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
from crispy.CrispyPlot import CrispyPlot
from cancer_proteomics.notebooks import DataImport, DimReduction, two_vars_correlation


LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/")


# ### Imports

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
prot = DataImport.read_protein_matrix()

# Read Transcriptomics
gexp = DataImport.read_gene_matrix()


# ### Dimension reduction

# Run PCA and tSNE
prot_dimred = DimReduction.dim_reduction(prot)

# Plot cell lines in 2D coloured by tissue type
fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=600)

DimReduction.plot_dim_reduction(
    prot_dimred, ctype="tsne", hue_by=ss["tissue"], palette=CrispyPlot.PAL_TISSUE, ax=ax
)

plt.savefig(f"{RPATH}/DimensionReduction_Proteomics_tSNE.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/DimensionReduction_Proteomics_tSNE.png", bbox_inches="tight")
plt.close("all")


# ### Covariates

covariates = pd.concat(
    [
        ss["CopyNumberAttenuation"],
        ss["GeneExpressionAttenuation"],
        ss["EMT"],
        ss["Proteasome"],
        ss["TranslationInitiation"],
        ss["CopyNumberInstability"],
        prot.loc[["CADH1_HUMAN", "VIME_HUMAN"]].T.add_suffix("_prot"),
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_gexp"),
        pd.get_dummies(ss["media"]),
        pd.get_dummies(ss["growth_properties"]),
        pd.get_dummies(ss["tissue"])[["Haematopoietic and Lymphoid", "Lung"]],
        ss[["ploidy", "mutational_burden", "growth", "size"]],
        ss["replicates_correlation"].rename("RepsCorrelation"),
    ],
    axis=1,
)

# Plot
n_pcs = 30
pcs_order = DimReduction.pc_labels(n_pcs)

# Covariates correlation
covs_corr = (
    pd.DataFrame(
        [
            {
                **two_vars_correlation(prot_dimred["pcs"][pc], covariates[c]),
                **dict(pc=pc, covariate=c),
            }
            for pc in pcs_order
            for c in covariates
        ]
    )
    .sort_values("pval")
    .dropna()
)

# Plot
df_vexp = prot_dimred["vexp"][pcs_order]
df_corr = pd.pivot_table(covs_corr, index="covariate", columns="pc", values="corr").loc[
    covariates.columns, pcs_order
]

f, (axb, axh) = plt.subplots(
    2,
    1,
    sharex="col",
    sharey="row",
    figsize=(n_pcs * 0.225, df_corr.shape[0] * 0.225 + 0.5),
    gridspec_kw=dict(height_ratios=[1, 4]),
    dpi=600,
)

axb.bar(np.arange(n_pcs) + 0.5, df_vexp, color=CrispyPlot.PAL_DTRACE[2], linewidth=0)
axb.set_yticks(np.arange(0, df_vexp.max() + 0.05, 0.05))
axb.set_title(f"Principal component analysis")
axb.set_ylabel("Total variance")

axb_twin = axb.twinx()
axb_twin.scatter(
    np.arange(n_pcs) + 0.5, df_vexp.cumsum(), c=CrispyPlot.PAL_DTRACE[1], s=6
)
axb_twin.plot(
    np.arange(n_pcs) + 0.5,
    df_vexp.cumsum(),
    lw=0.5,
    ls="--",
    c=CrispyPlot.PAL_DTRACE[1],
)
axb_twin.set_yticks(np.arange(0, df_vexp.cumsum().max() + 0.1, 0.1))
axb_twin.set_ylabel("Cumulative variance")

g = sns.heatmap(
    df_corr,
    cmap="Spectral",
    annot=True,
    cbar=False,
    fmt=".2f",
    linewidths=0.3,
    ax=axh,
    center=0,
    annot_kws={"fontsize": 5},
)
axh.set_xlabel("Principal components")
axh.set_ylabel("")

plt.subplots_adjust(hspace=0.01)
plt.savefig(
    f"{RPATH}/DimensionReduction_Proteomics_PCA_heatmap.pdf", bbox_inches="tight"
)
plt.savefig(
    f"{RPATH}/DimensionReduction_Proteomics_PCA_heatmap.png", bbox_inches="tight"
)
plt.close("all")
