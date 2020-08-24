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
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import spearmanr

from cancer_proteomics.notebooks import DataImport, DimReduction, two_vars_correlation


LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/")


# ### Imports

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
prot = DataImport.read_protein_matrix(map_protein=True)

# Read proteomics BROAD (Proteins x Cell lines)
prot_broad = DataImport.read_protein_matrix_broad()

# Read Transcriptomics
gexp = DataImport.read_gene_matrix()


# ### Dimension reduction

# Run PCA and tSNE
prot_dimred = DimReduction.dim_reduction(prot)
prot_broad_dimred = DimReduction.dim_reduction(prot_broad)

# Plot cell lines in 2D coloured by tissue type
fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=600)

DimReduction.plot_dim_reduction(
    prot_dimred, ctype="tsne", hue_by=ss["tissue"], palette=CrispyPlot.PAL_TISSUE, ax=ax
)

plt.savefig(f"{RPATH}/DimensionReduction_Proteomics_tSNE.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/DimensionReduction_Proteomics_tSNE.png", bbox_inches="tight")
plt.close("all")


# ### PCs enrichment
#

genesets = ["c5.all.v7.1.symbols.gmt", "c2.all.v7.1.symbols.gmt"]

enr_pcs = [("prot", "PC1"), ("prot_broad", "PC1")]

dsets_dred = dict(prot=prot_dimred, prot_broad=prot_broad_dimred)

enr_pcs = pd.concat(
    [
        gseapy.ssgsea(
            dsets_dred[dtype]["loadings"].loc[dtype_pc],
            processes=4,
            gene_sets=Enrichment.read_gmt(f"{DPATH}/pathways/{g}"),
            no_plot=True,
        )
        .res2d.assign(geneset=g)
        .assign(dtype=dtype)
        .assign(dtype_pc=dtype_pc)
        .reset_index()
        for dtype, dtype_pc in enr_pcs
        for g in genesets
    ],
    ignore_index=True,
)
enr_pcs = enr_pcs.rename(columns={"sample1": "nes"}).sort_values("nes")
enr_pcs.to_csv(
    f"{TPATH}/DimensionReduction_pcs_enr.csv.gz", compression="gzip", index=False
)

# Plot
enr_pcs_plt = [("prot", "PC1", "prot_broad", "PC1", 0.5)]

for x_dtype, x_pc, y_dtype, y_pc, thres_abs in enr_pcs_plt:
    x_label = f"{x_dtype}_{x_pc}"
    y_label = f"{y_dtype}_{y_pc}"

    plot_df = pd.concat(
        [
            enr_pcs.query(f"(dtype == '{x_dtype}') & (dtype_pc == '{x_pc}')")
            .set_index("Term|NES")["nes"]
            .rename(x_label),
            enr_pcs.query(f"(dtype == '{y_dtype}') & (dtype_pc == '{y_pc}')")
            .set_index("Term|NES")["nes"]
            .rename(y_label),
        ],
        axis=1,
    ).dropna()
    plot_df.index = [i.replace("_", " ") for i in plot_df.index]

    f, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

    ax.scatter(
        plot_df[f"{x_dtype}_{x_pc}"],
        plot_df[f"{y_dtype}_{y_pc}"],
        c=GIPlot.PAL_DBGD[2],
        s=5,
        linewidths=0,
    )

    gs_highlight = plot_df[
        (plot_df[[x_label, y_label]].abs() > thres_abs).any(1)
    ].sort_values(x_label)
    gs_highlight_dw = gs_highlight.query(f"{x_label} < 0").sort_values(
        x_label, ascending=False
    )
    gs_highlight_up = gs_highlight.query(f"{x_label} > 0").sort_values(
        x_label, ascending=True
    )
    gs_highlight_pal = pd.Series(
        sns.light_palette(
            "#3182bd", n_colors=len(gs_highlight_dw) + 1, reverse=True
        ).as_hex()[:-1]
        + sns.light_palette(
            "#e6550d", n_colors=len(gs_highlight_up) + 1, reverse=False
        ).as_hex()[1:],
        index=gs_highlight.index,
    )

    for g in gs_highlight.index:
        ax.scatter(
            plot_df.loc[g, x_label],
            plot_df.loc[g, y_label],
            c=gs_highlight_pal[g],
            s=10,
            linewidths=0,
            label=g,
        )

    cor, pval = spearmanr(plot_df[x_label], plot_df[y_label])
    annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
    ax.text(0.98, 0.02, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

    ax.legend(
        frameon=False, prop={"size": 4}, loc="center left", bbox_to_anchor=(1, 0.5)
    )

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    ax.set_xlabel(f"{x_dtype} {x_pc} ({dsets_dred[x_dtype]['vexp'][x_pc] * 100:.1f}%)")
    ax.set_ylabel(f"{y_dtype} {y_pc} ({dsets_dred[y_dtype]['vexp'][y_pc] * 100:.1f}%)")
    ax.set_title(f"PCs enrichment scores (NES)")

    plt.savefig(
        f"{RPATH}/DimensionReduction_pcs_enr_{x_dtype}_{x_pc}_{y_dtype}_{y_pc}.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{RPATH}/DimensionReduction_pcs_enr_{x_dtype}_{x_pc}_{y_dtype}_{y_pc}.png",
        bbox_inches="tight",
    )
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
        prot.loc[["CDH1", "VIM"]].T.add_suffix("_prot"),
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
