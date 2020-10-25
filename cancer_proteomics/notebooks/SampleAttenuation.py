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

import igraph
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
from crispy.MOFA import MOFA
import matplotlib.pyplot as plt
from scipy.stats import skew
from crispy.GIPlot import GIPlot
from crispy.DataImporter import PPI
from sklearn.decomposition import PCA
from crispy.Enrichment import Enrichment
from crispy.LMModels import LMModels, LModel
from cancer_proteomics.notebooks import DataImport
from scripts.CProtUtils import two_vars_correlation

LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/")


# ### Imports

# Pathways
emt_sig = Enrichment.read_gmt(f"{DPATH}/pathways/emt.symbols.gmt")
emt_sig = emt_sig["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"]

proteasome_sig = Enrichment.read_gmt(f"{DPATH}/pathways/proteasome.symbols.gmt")
proteasome_sig = proteasome_sig["BIOCARTA_PROTEASOME_PATHWAY"]

translation_sig = Enrichment.read_gmt(f"{DPATH}/pathways/translation_initiation.symbols.gmt")
translation_sig = translation_sig["GO_TRANSLATIONAL_INITIATION"]

# MOFA analysis

factors, weights, rsquare = MOFA.read_mofa_hdf5(f"{TPATH}/MultiOmics_broad.hdf5")

# Perturbation proteomics differential analysis
perturb = pd.read_csv(f"{DPATH}/perturbation_proteomics_diff_analysis.csv")
perturb_diff = pd.pivot_table(perturb, index="GeneSymbol", columns="comparison", values="diff")
perturb_corr = pd.DataFrame(
    {
        f: {
            c: two_vars_correlation(weights["proteomics"][f], perturb_diff[c])["corr"]
            for c in perturb_diff
        }
        for f in weights["proteomics"]
    }
)

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
prot = DataImport.read_protein_matrix(map_protein=True)
prot_reps = pd.read_csv(f"{DPATH}/E0022_P06_Protein_Matrix_Raw_Mean_Intensities.tsv.gz", sep="\t", index_col=0)

# Read Transcriptomics
gexp = DataImport.read_gene_matrix()

# Read CRISPR
crispr = DataImport.read_crispr_matrix()

# Read Methylation
methy = DataImport.read_methylation_matrix()

# Read Drug-response
drespo = DataImport.read_drug_response()

# Covariates
covariates = pd.concat(
    [
        ss["CopyNumberAttenuation"],
        ss["GeneExpressionAttenuation"],
        ss["EMT"],
        ss["Proteasome"],
        ss["TranslationInitiation"],
        ss["CopyNumberInstability"],
        prot.loc[["CDH1", "VIM"]].T.add_suffix("_prot"),
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_gexp"),
        pd.get_dummies(ss["media"]),
        pd.get_dummies(ss["growth_properties"]),
        pd.get_dummies(ss["tissue"])[["Haematopoietic and Lymphoid", "Lung"]],
        ss[["ploidy", "mutational_burden", "growth", "size"]],
        ss["replicates_correlation"].rename("RepsCorrelation"),
        prot.mean().rename("MeanProteomics"),
        methy.mean().rename("MeanMethylation"),
        drespo.mean().rename("MeanDrugResponse"),
    ],
    axis=1,
)

# Merged samplesheet
ss_merged = pd.concat(
    [
        ss,
        prot.reindex(proteasome_sig).mean().rename("ProteasomeMean"),
        prot.reindex(translation_sig).mean().rename("TranslationInitiationMean"),
        factors,
        prot.mean().rename("MeanProtein"),
        prot.std().rename("StdProtein"),
    ],
    axis=1,
)

# Covariates and factors correlation
n_factors_corr = {}
for f in factors:
    n_factors_corr[f] = {}

    for c in covariates:
        fc_samples = list(covariates.reindex(factors[f].index)[c].dropna().index)
        n_factors_corr[f][c] = two_vars_correlation(
            factors[f][fc_samples], covariates[c][fc_samples]
        )["corr"]
n_factors_corr = pd.DataFrame(n_factors_corr)


# ### Clustermap with weights correlation
n_heatmaps = len(rsquare)
nrows, ncols = list(rsquare.values())[0].shape

row_order = list(list(rsquare.values())[0].index)
col_order = list(list(rsquare.values())[0].columns)

f, axs = plt.subplots(
    n_heatmaps + 2,
    1,
    sharex="none",
    sharey="none",
    gridspec_kw={"height_ratios": [1.5] * n_heatmaps + [9] + [1.5]},
    figsize=(0.25 * ncols, 0.3 * n_heatmaps + 0.3 * n_factors_corr.shape[0]),
)

vmax = np.max([rsquare[k].max().max() for k in rsquare])

# Factors
for i, k in enumerate(rsquare):
    axh = axs[i]

    df = rsquare[k]

    # Heatmap
    g = sns.heatmap(
        df.loc[row_order, col_order],
        cmap="Blues",
        annot=True,
        cbar=False,
        fmt=".1f",
        linewidths=0.5,
        ax=axh,
        vmin=0,
        vmax=vmax,
        annot_kws={"fontsize": 5},
    )
    axh.set_ylabel(f"{k} cell lines")
    g.set_xticklabels([])
    g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment="right")

# Covariates
ax = axs[n_heatmaps]

g = sns.heatmap(
    n_factors_corr.loc[:, col_order],
    cmap="RdYlGn",
    center=0,
    annot=True,
    cbar=False,
    fmt=".2f",
    linewidths=0.5,
    ax=ax,
    annot_kws={"fontsize": 5},
)
ax.set_xlabel("")
ax.set_ylabel(f"Potential related factors")

# Perturbation
ax = axs[n_heatmaps + 1]

g = sns.heatmap(
    perturb_corr.loc[:, col_order],
    cmap="RdYlGn",
    center=0,
    annot=True,
    cbar=False,
    fmt=".2f",
    linewidths=0.5,
    ax=ax,
    annot_kws={"fontsize": 5},
)
ax.set_xlabel("")
ax.set_ylabel(f"")

g.set_xticklabels(g.get_xticklabels(), rotation=0, va="center")

plt.subplots_adjust(hspace=0.025)

plt.savefig(f"{RPATH}/SampleAttenuation_clustermap.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/SampleAttenuation_clustermap.png", bbox_inches="tight", dpi=600)
plt.close("all")


###
plot_df = pd.concat([
    rsquare["Haem"].T.add_prefix("Haem_"),
    rsquare["Other"].T.add_prefix("Other_"),
    n_factors_corr.T.add_prefix("Corr_"),
    perturb_corr.T.add_prefix("Pert_"),
], axis=1).T


f, axs = plt.subplots(
    3,
    1,
    sharex="col",
    sharey="none",
    gridspec_kw={"height_ratios": [1.5] * 2 + [9]},
    figsize=(plot_df.shape[1] * 0.25, plot_df.shape[0] * 0.25),
)

for i, n in enumerate(["Haem", "Other"]):
    df = plot_df[[i.startswith(n) for i in plot_df.index]]
    df.index = [i.split("_")[1] for i in df.index]
    g = sns.heatmap(
        df,
        cmap="Blues",
        annot=True,
        cbar=False,
        fmt=".1f",
        linewidths=0.5,
        ax=axs[i],
        vmin=0,
        annot_kws={"fontsize": 5},
    )
    axs[i].set_ylabel(f"{n} cell lines")
#
df = plot_df[[i.split("_")[0] not in ["Haem", "Other"] for i in plot_df.index]]
sns.heatmap(
    df,
    cmap="RdYlGn",
    center=0,
    annot=True,
    cbar=False,
    fmt=".2f",
    linewidths=0.5,
    annot_kws={"fontsize": 5},
    ax=axs[2],
)

plt.subplots_adjust(hspace=0.025)

plt.savefig(f"{RPATH}/SampleAttenuation_clustermap_merged.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/SampleAttenuation_clustermap_merged.png", bbox_inches="tight", dpi=600)
plt.close("all")


# ####
for x_var, y_var in [("MeanProtein", "replicates_correlation"), ("StdProtein", "F2"), ("MeanProtein", "F2")]:
    GIPlot.gi_regression(ss_merged[x_var], ss_merged[y_var])

    plt.savefig(f"{RPATH}/SampleAttenuation_regression_{x_var}_{y_var}.pdf", bbox_inches="tight")
    plt.savefig(f"{RPATH}/SampleAttenuation_regression_{x_var}_{y_var}.png", bbox_inches="tight", dpi=600)
    plt.close("all")
