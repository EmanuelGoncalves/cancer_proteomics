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
from sklearn.metrics.ranking import auc
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from crispy.DataImporter import CORUM, BioGRID, PPI, HuRI
from cancer_proteomics.notebooks import (
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


# ### Imports

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
prot = DataImport.read_protein_matrix(map_protein=True)

# Read proteomics BROAD (Proteins x Cell lines)
prot_broad = DataImport.read_protein_matrix_broad()

# Read Transcriptomics
gexp = DataImport.read_gene_matrix()

# Read CRISPR
crispr = DataImport.read_crispr_matrix()

# Read Methylation
methy = DataImport.read_methylation_matrix()

# Read Drug-response
drespo = DataImport.read_drug_response()
drespo = drespo.set_index(pd.Series([";".join(map(str, i)) for i in drespo.index]))

dmaxc = DataImport.read_drug_max_concentration()
dmaxc.index = [";".join(map(str, i)) for i in dmaxc.index]
dmaxc = dmaxc.reindex(drespo.index)


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

# ### MOFA
#
groupby = ss.loc[prot.columns, "tissue"].apply(
    lambda v: "Haem" if v == "Haematopoietic and Lymphoid" else "Other"
)

mofa = MOFA(
    views=dict(proteomics=prot, transcriptomics=gexp, methylation=methy, drespo=drespo),
    groupby=groupby,
    iterations=2000,
    use_overlap=False,
    convergence_mode="slow",
    factors_n=15,
    from_file=f"{TPATH}/MultiOmics.hdf5",
    verbose=2,
)


# ### Factors integrated with other measurements
#
n_factors_corr = {}
for f in mofa.factors:
    n_factors_corr[f] = {}

    for c in covariates:
        fc_samples = list(covariates.reindex(mofa.factors[f].index)[c].dropna().index)
        n_factors_corr[f][c] = pearsonr(
            mofa.factors[f][fc_samples], covariates[c][fc_samples]
        )[0]
n_factors_corr = pd.DataFrame(n_factors_corr)

# Factor clustermap
MOFAPlot.factors_corr_clustermap(mofa)
plt.savefig(f"{RPATH}/MultiOmics_factors_corr_clustermap.pdf", bbox_inches="tight")
plt.savefig(
    f"{RPATH}/MultiOmics_factors_corr_clustermap.png", bbox_inches="tight", dpi=600
)
plt.close("all")

# Variance explained across data-sets
MOFAPlot.variance_explained_heatmap(mofa)
plt.savefig(f"{RPATH}/MultiOmics_factors_rsquared_heatmap.pdf", bbox_inches="tight")
plt.savefig(
    f"{RPATH}/MultiOmics_factors_rsquared_heatmap.png", bbox_inches="tight", dpi=600
)
plt.close("all")

# Covairates correlation heatmap
MOFAPlot.covariates_heatmap(n_factors_corr, mofa, ss["tissue"])
plt.savefig(
    f"{RPATH}/MultiOmics_factors_covariates_clustermap.pdf", bbox_inches="tight"
)
plt.savefig(
    f"{RPATH}/MultiOmics_factors_covariates_clustermap.png",
    bbox_inches="tight",
    dpi=600,
)
plt.close("all")


# ### MOFA Factor 1 and 2
f_x, f_y = "F1", "F3"

plot_df = pd.concat(
    [
        mofa.factors[[f_x, f_y]],
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_transcriptomics"),
        prot.loc[["CDH1", "VIM"]].T.add_suffix("_proteomics"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
)

# Tissue plot
ax = GIPlot.gi_tissue_plot(f_x, f_y, plot_df, plot_reg=False, pal=PALETTE_TTYPE)
ax.set_xlabel(f"Factor {f_x[1:]}")
ax.set_ylabel(f"Factor {f_y[1:]}")
plt.savefig(f"{RPATH}/MultiOmics_{f_x}_{f_y}_tissue_plot.pdf", bbox_inches="tight")
plt.savefig(
    f"{RPATH}/MultiOmics_{f_x}_{f_y}_tissue_plot.png", bbox_inches="tight", dpi=600
)
plt.close("all")

# Continous annotation
for z in ["VIM_proteomics", "CDH1_proteomics"]:
    ax = GIPlot.gi_continuous_plot(f_x, f_y, z, plot_df, cbar_label=z.replace("_", " "))
    ax.set_xlabel(f"Factor {f_x[1:]}")
    ax.set_ylabel(f"Factor {f_y[1:]}")
    plt.savefig(
        f"{RPATH}/MultiOmics_{f_x}_{f_y}_continous_{z}.pdf", bbox_inches="tight"
    )
    plt.savefig(
        f"{RPATH}/MultiOmics_{f_x}_{f_y}_continous_{z}.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close("all")


# ### Perturbation proteomics

manifest = DataImport.read_protein_perturbation_manifest()
manifest = manifest[~manifest["External Patient ID"].isin(["MDA-MB-468", "MRC-5"])]
manifest = manifest[
    ~((manifest["Cell Line"] == "BT-549 1% FBS") & (manifest["Date on sample"] == "4/7/19 "))
]
manifest = manifest.drop(
    ["200627_b2-1-t5-1_00wuz_00yid_m03_s_1", "200623_b2-1-t4-1_00wuz_00yh3_m01_s_1"]
)
manifest = manifest[["0.5%FBS" not in v for v in manifest["Cell Line"]]]

prot_perturb = DataImport.read_protein_perturbation(map_protein=True)
prot_perturb = prot_perturb[manifest.index]

prot_perturb_reps_corr = pd.DataFrame(
    [
        {
            **two_vars_correlation(prot_perturb[c1], prot_perturb[c2]),
            **dict(
                sample1=c1,
                sample2=c2,
                cellline1=manifest.loc[c1, "Cell Line"],
                cellline2=manifest.loc[c2, "Cell Line"],
            ),
        }
        for c1, c2 in it.combinations(list(prot_perturb), 2)
    ]
)

#
plot_df = prot_perturb_reps_corr.query("cellline1 == cellline2")
plot_df.cellline1 = plot_df.cellline1.astype("category")
plot_df.cellline1.cat.set_categories(PALETTE_PERTURB.keys(), inplace=True)
plot_df = plot_df.sort_values("cellline1")

fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

sns.boxplot(
    "corr",
    "cellline1",
    data=plot_df,
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
    orient="h",
    edgecolor="white",
    palette=PALETTE_PERTURB,
    linewidth=0.1,
    s=3,
    ax=ax,
)

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel("Replicated correlation\n(Pearson's R)")
ax.set_ylabel("Condition")

plt.savefig(f"{RPATH}/MultiOmics_perturb_repc_corr.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/MultiOmics_perturb_repc_corr.png", bbox_inches="tight", dpi=600)
plt.close("all")

#
plot_df = prot_perturb.rename(columns=manifest["Cell Line"]).corr()
col_colors = [PALETTE_PERTURB[c] for c in plot_df]
row_colors = [PALETTE_PERTURB[c] for c in plot_df.index]

g = sns.clustermap(
    plot_df,
    cmap="RdYlGn",
    row_colors=row_colors,
    col_colors=col_colors,
    annot=True,
    center=0,
    fmt=".1f",
    annot_kws=dict(size=4),
    lw=0.05,
    figsize=(8.5, 8.5),
)
plt.savefig(f"{RPATH}/MultiOmics_perturb_repc_corr_clustermap.pdf", bbox_inches="tight")
plt.savefig(
    f"{RPATH}/MultiOmics_perturb_repc_corr_clustermap.png", bbox_inches="tight", dpi=600
)
plt.close("all")

#
comparison = dict(
    control=list(manifest[["10% FBS" in v for v in manifest["Cell Line"]]].index),
    condition=list(manifest[["1% FBS" in v for v in manifest["Cell Line"]]].index),
)

diff_prot = pd.DataFrame(
    ttest_ind(
        prot_perturb[comparison["control"]].T,
        prot_perturb[comparison["condition"]].T,
        equal_var=False,
        nan_policy="omit",
    ),
    index=["tstat", "pvalue"],
    columns=prot_perturb.index,
).T.astype(float).sort_values("pvalue").dropna()
diff_prot["fdr"] = multipletests(diff_prot["pvalue"], method="fdr_bh")[1]
diff_prot["diff"] = prot_perturb.loc[diff_prot.index, comparison["control"]].mean(1) - prot_perturb.loc[diff_prot.index, comparison["condition"]].mean(1)
diff_prot["control_n"] = prot_perturb.loc[diff_prot.index, comparison["control"]].count(1)
diff_prot["condition_n"] = prot_perturb.loc[diff_prot.index, comparison["condition"]].count(1)


# ###
genesets = ["c5.all.v7.1.symbols.gmt", "c2.all.v7.1.symbols.gmt"]

dsets_dred = dict(diff_prot=diff_prot["diff"], mofa_factor=mofa.weights["proteomics"]["F2"])

enr_pcs = pd.concat(
    [
        gseapy.ssgsea(
            dsets_dred[ds],
            processes=4,
            gene_sets=Enrichment.read_gmt(f"{DPATH}/pathways/{g}"),
            no_plot=True,
        )
        .res2d.assign(geneset=g)
        .assign(dtype=ds)
        .reset_index()
        for ds in dsets_dred
        for g in genesets
    ],
    ignore_index=True,
)
enr_pcs = enr_pcs.rename(columns={"sample1": "nes"}).sort_values("nes")

# ####

plot_df = pd.pivot_table(enr_pcs, index="Term|NES", columns="dtype", values="nes")

f, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

ax.scatter(
    plot_df["diff_prot"], plot_df["mofa_factor"],
    c=GIPlot.PAL_DBGD[2],
    s=5,
    linewidths=0,
)

gs_highlight = plot_df[
    (plot_df[["diff_prot", "mofa_factor"]].abs() > .45).any(1)
].sort_values("diff_prot").head(30)
gs_highlight_dw = gs_highlight.query(f"diff_prot < 0").sort_values(
    "diff_prot", ascending=False
)
gs_highlight_up = gs_highlight.query(f"diff_prot > 0").sort_values(
    "diff_prot", ascending=True
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
        plot_df.loc[g, "diff_prot"],
        plot_df.loc[g, "mofa_factor"],
        c=gs_highlight_pal[g],
        s=10,
        linewidths=0,
        label=g,
    )

cor, pval = spearmanr(plot_df["diff_prot"], plot_df["mofa_factor"], nan_policy="omit")
annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
ax.text(0.98, 0.02, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

ax.legend(
    frameon=False, prop={"size": 4}, loc="center left", bbox_to_anchor=(1, 0.5)
)

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel("FBS differential protein abundance\n(10% - 1%)")
ax.set_ylabel("Factor 2")
ax.set_title(f"PCs enrichment scores (NES)")

plt.savefig(
    f"{RPATH}/MultiOmics_perturb_gsea_corr.pdf",
    bbox_inches="tight",
)
plt.savefig(
    f"{RPATH}/MultiOmics_perturb_gsea_corr.png",
    bbox_inches="tight",
)
plt.close("all")
