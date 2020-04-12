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

import os
import sys
import logging
import argparse
import numpy as np
import numpy.ma as ma
import pandas as pd
import pkg_resources
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
from natsort import natsorted
from statsmodels.stats.weightstats import ztest
from sklearn.metrics import jaccard_score
from crispy.GIPlot import GIPlot
from itertools import zip_longest
from crispy.QCPlot import QCplot
from crispy.Utils import Utils
from sklearn.metrics.ranking import auc
from cancer_proteomics.eg.LMModels import LMModels
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import quantile_transform
from crispy.DimensionReduction import dim_reduction, plot_dim_reduction
from scipy.stats import spearmanr, pearsonr, rankdata, ttest_ind, zscore
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    CRISPR,
    CORUM,
    Sample,
    BioGRID,
    PPI,
    HuRI,
    DPATH,
    WES,
)


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


# Data-sets
#
prot_obj = Proteomics()
gexp_obj = GeneExpression()
crispr_obj = CRISPR()
wes_obj = WES()


# Samples
#

ss = prot_obj.ss.copy()


# Filter data-sets
#

prot = prot_obj.filter()
prot = prot.loc[prot.count(1) > 10]
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp_obj.filter()
LOG.info(f"Transcriptomics: {gexp.shape}")

crispr = crispr_obj.filter(dtype="merged")
LOG.info(f"CRISPR: {crispr.shape}")

wes = wes_obj.filter(as_matrix=False)


# CORUM + BioGRID
#

corum_db = CORUM()
biogrid_db = BioGRID()
huri_db = HuRI()


# Genes
#
cgenes = pd.read_csv(f"{DPATH}/cancer_genes_latest.csv.gz", index_col=0).iloc[:, 0]

genes = natsorted(
    list(set.intersection(set(prot.index), set(gexp.index), set(crispr.index)))
)
LOG.info(f"Genes: {len(genes)}")


# Paralog information
#

pinfo = pd.read_excel(
    f"{DPATH}/msb198871-sup-0004-datasetev1.xlsx", sheet_name="gene annotations"
)
pinfo = pinfo.groupby("gene name")[
    ["paralog or singleton", "homomer or not (all PPI)"]
].first()

paralog_ds = pd.read_excel(
    f"{DPATH}/msb198871-sup-0004-datasetev1.xlsx", sheet_name="paralog annotations"
)


# Correlation matrices
#


def df_correlation(df_matrix, min_obs=10):
    LOG.info(df_matrix.shape)

    x = ma.masked_invalid(df_matrix.values)
    n_vars = x.shape[1]

    rs = np.full((n_vars, n_vars), np.nan, dtype=float)
    prob = np.full((n_vars, n_vars), np.nan, dtype=float)

    for var1 in range(n_vars - 1):
        for var2 in range(var1 + 1, n_vars):
            xy = ma.mask_rowcols(x[:, [var1, var2]], axis=0)
            xy = xy[~xy.mask.any(axis=1), :]

            if xy.shape[0] > min_obs:
                rs[var1, var2], prob[var1, var2] = pearsonr(xy[:, 0], xy[:, 1])

    df = pd.DataFrame(dict(corr=rs.ravel(), pvalue=prob.ravel()))
    df["protein1"], df["protein2"] = zip(
        *(it.product(df_matrix.columns, df_matrix.columns))
    )
    df = df.dropna().set_index(["protein1", "protein2"])
    df["fdr"] = multipletests(df["pvalue"], method="fdr_bh")[1]

    return df


df_corr = (
    pd.concat(
        [
            df_correlation(prot.reindex(genes).T).add_prefix("prot_"),
            df_correlation(gexp.reindex(genes).T).add_prefix("gexp_"),
            df_correlation(crispr.reindex(genes).T).add_prefix("crispr_"),
        ],
        axis=1,
    )
    .sort_values("prot_pvalue")
    .reset_index()
).dropna()

# CORUM
df_corr["corum"] = [
    int((p1, p2) in corum_db.db_melt_symbol)
    for p1, p2 in df_corr[["protein1", "protein2"]].values
]

# BioGRID
df_corr["biogrid"] = [
    int((p1, p2) in biogrid_db.biogrid)
    for p1, p2 in df_corr[["protein1", "protein2"]].values
]

# HuRI
df_corr["huri"] = [
    int((p1, p2) in huri_db.huri) for p1, p2 in df_corr[["protein1", "protein2"]].values
]

# String distance
ppi = PPI().build_string_ppi(score_thres=900)
df_corr = PPI.ppi_annotation(
    df_corr, ppi, x_var="protein1", y_var="protein2", ppi_var="string_dist"
)
df_corr = df_corr.assign(string=(df_corr["string_dist"] == "1").astype(int))

# Number of observations
p_obs = {p: set(v.dropna().index) for p, v in prot.reindex(genes).iterrows()}
df_corr["nobs"] = [
    len(set.intersection(p_obs[p1], p_obs[p2]))
    for p1, p2 in df_corr[["protein1", "protein2"]].values
]

# Cancer genes
df_corr["protein1_cgene"] = df_corr["protein1"].isin(set(cgenes.index)).astype(int)
df_corr["protein1_cgene_type"] = cgenes.reindex(df_corr["protein1"]).values

df_corr["protein2_cgene"] = df_corr["protein2"].isin(set(cgenes.index)).astype(int)
df_corr["protein2_cgene_type"] = cgenes.reindex(df_corr["protein2"]).values

# Export
df_corr.to_csv(
    f"{RPATH}/2.SLProteinInteractions.csv.gz", compression="gzip", index=False
)
# df_corr = pd.read_csv(f"{RPATH}/2.SLProteinInteractions.csv.gz")


rc_dict = dict()
for y in ["corum", "biogrid", "string", "huri"]:
    rc_dict[y] = dict()
    for x in ["prot_corr", "gexp_corr", "crispr_corr"]:
        rc_df = (
            df_corr.sort_values(f"{x.split('_')[0]}_pvalue")[[x, y]]
            .reset_index(drop=True)
            .copy()
        )

        rc_df_y = np.cumsum(rc_df[y]) / sum(rc_df[y])
        rc_df_x = np.array(rc_df.index) / rc_df.shape[0]
        rc_df_auc = auc(rc_df_x, rc_df_y)

        rc_dict[y][x] = dict(x=list(rc_df_x), y=list(rc_df_y), auc=rc_df_auc)

rc_pal = dict(
    biogrid=sns.color_palette("tab20c").as_hex()[0:3],
    corum=sns.color_palette("tab20c").as_hex()[4:7],
    string=sns.color_palette("tab20c").as_hex()[8:11],
    huri=sns.color_palette("tab20c").as_hex()[12:15],
)


# Recall curves
#

_, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

for db in rc_dict:
    for i, ds in enumerate(rc_dict[db]):
        ax.plot(
            rc_dict[db][ds]["x"],
            rc_dict[db][ds]["y"],
            label=f"{db} {ds.split('_')[0]} (AUC {rc_dict[db][ds]['auc']:.2f})",
            c=rc_pal[db][i],
        )

ax.plot([0, 1], [0, 1], "k--", lw=0.3)
ax.legend(loc="lower right", frameon=False)

ax.set_ylabel("Cumulative sum")
ax.set_xlabel("Ranked correlation")
ax.grid(True, axis="both", ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.savefig(
    f"{RPATH}/2.SLProtein_roc_curves.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


# Recall barplot
#

plot_df = pd.DataFrame(
    [
        dict(ppi=db, dtype=ds, auc=rc_dict[db][ds]["auc"])
        for db in rc_dict
        for ds in rc_dict[db]
    ]
)
plot_df["dtype"] = plot_df["dtype"].apply(lambda v: v.split("_")[0])

hue_order = ["corum", "biogrid", "string", "huri"]

_, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

sns.barplot(
    "auc",
    "ppi",
    "dtype",
    data=plot_df,
    orient="h",
    linewidth=0.0,
    saturation=1.0,
    palette="Blues_d",
    ax=ax,
)

ax.set_ylabel("")
ax.set_xlabel("Recall curve AUC")
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

ax.legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    prop={"size": 4},
    frameon=False,
    title="Data-set",
).get_title().set_fontsize("5")

plt.savefig(
    f"{RPATH}/2.SLProtein_roc_barplot.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


#
#

for dt in ["prot", "gexp", "crispr"]:
    _, ax = plt.subplots(1, 1, figsize=(3, 5), dpi=600)

    ax.hexbin(
        df_corr[f"{dt}_corr"],
        -np.log(df_corr[f"{dt}_pvalue"]),
        cmap="Spectral_r",
        gridsize=100,
        mincnt=1,
        bins="log",
        lw=0,
    )

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
    ax.set_ylabel("Significance (-log p-value)")
    ax.set_xlabel("Effect size (Pearson's R)")
    ax.set_title(dt)

    plt.savefig(
        f"{RPATH}/2.SLProtein_volcano_{dt}.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")


#
#

_, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

ax.hexbin(
    df_corr.eval("prot_corr - crispr_corr"),
    df_corr.eval("gexp_corr - crispr_corr"),
    cmap="Spectral_r",
    gridsize=100,
    mincnt=1,
    bins="log",
    lw=0,
)

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
ax.set_xlabel("PPI Protein - CRISPR")
ax.set_ylabel("PPI GExp - CRISPR")

plt.savefig(
    f"{RPATH}/2.SLProtein_ppi_attenuation.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


#
#

row_order = ["prot", "gexp", "crispr"]

plot_df = pd.concat(
    [
        df_corr.query(f"({d}_fdr < .05) & ({d}_corr > 0.5)")["string_dist"]
        .value_counts()
        .reset_index()
        .assign(dtype=d)
        for d in row_order
    ]
)
_, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

sns.barplot(
    "dtype",
    "string_dist",
    "index",
    data=plot_df,
    linewidth=0.0,
    saturation=1.0,
    palette=GIPlot.PPI_PAL,
    hue_order=GIPlot.PPI_ORDER[1:],
    ax=ax,
)

ax.set_ylabel("")
ax.set_xlabel("String PPI distance")
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

ax.legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    prop={"size": 4},
    frameon=False,
    title="Distance",
).get_title().set_fontsize("5")

plt.savefig(
    f"{RPATH}/2.SLProtein_string_dist_barplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


#
#

for x_var, y_var in [
    ("prot_corr", "crispr_corr"),
    ("prot_corr", "gexp_corr"),
    ("crispr_corr", "gexp_corr"),
]:
    x, y = df_corr[x_var], df_corr[y_var]

    _, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

    ax.hexbin(x, y, cmap="Spectral_r", gridsize=100, mincnt=1, bins="log", lw=0)

    ax.set_xlabel(f"PPI {x_var.split('_')[0]} Pearsons'R")
    ax.set_ylabel(f"PPI {y_var.split('_')[0]} Pearsons'R")

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    cor, _ = spearmanr(x, y)
    annot_text = f"Spearman's R={cor:.2g}"
    ax.text(0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

    plt.savefig(
        f"{RPATH}/2.SLProtein_hexbin_{x_var.split('_')[0]}_{y_var.split('_')[0]}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


#
#

for dt in ["prot", "gexp", "crispr"]:
    _, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

    sns.distplot(
        df_corr[f"{dt}_corr"],
        kde=False,
        bins=60,
        hist_kws=dict(lw=0, alpha=1),
        color=GIPlot.PAL_DTRACE[2],
        ax=ax,
    )

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

    ax.set_title(f"Mean correlation = {df_corr[f'{dt}_corr'].mean():.2f}")

    plt.savefig(
        f"{RPATH}/2.SLProtein_histogram_{dt}.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")


#
#

df_corr.query(f"(prot_fdr < .05) & (prot_corr > 0.5)").sort_values(
    "crispr_pvalue"
).dropna().head(60)

# x_var, y_var = "ARID1A", "SMARCB1"
for x_var, y_var in [("CCT4", "XPO1"), ("ADSL", "ATIC"), ("CAD", "UMPS")]:
    plot_df = pd.concat(
        [
            prot.loc[[x_var, y_var]].T.add_prefix("prot_"),
            crispr_obj.merged.loc[[x_var, y_var]].T.add_prefix("crispr_"),
            ss["tissue"],
            crispr_obj.merged_institute.rename("institute"),
        ],
        axis=1,
    )

    _, axs = plt.subplots(1, 2, figsize=(4.3, 2), dpi=600)

    for i, dt in enumerate(["prot", "crispr"]):
        ax = axs[i]

        GIPlot.gi_regression_no_marginals(
            f"{dt}_{x_var}",
            f"{dt}_{y_var}",
            plot_df,
            "tissue",
            palette=GIPlot.PAL_TISSUE,
            style="institute" if dt == "crispr" else None,
            plot_hue_legend=(dt == "crispr"),
            plot_style_legend=(dt == "crispr"),
            ax=ax,
        )

        ax.set_xlabel(x_var)
        ax.set_ylabel(None if i != 0 else y_var)
        ax.set_title(dt)

    plt.subplots_adjust(hspace=0, wspace=0.3)
    plt.savefig(
        f"{RPATH}/2.SLProtein_giplot_{x_var}_{y_var}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


#
#

df_corr_ppi = df_corr.query(f"(prot_fdr < .05) & (prot_corr > 0.5)")
df_corr_ppi = pd.concat(
    [df_corr_ppi["protein1"], df_corr_ppi["protein2"]], ignore_index=True
).value_counts()
df_corr_ppi = pd.concat(
    [
        pd.qcut(df_corr_ppi, np.arange(0, 1.1, 0.1))
        .apply(lambda v: f"{round(v.left):.0f}-{round(v.right):.0f}")
        .rename("n_ppi"),
        df_corr_ppi.pipe(np.log2).rename("ppi"),
        gexp.loc[df_corr_ppi.index].mean(1).rename("gexp"),
        crispr.loc[df_corr_ppi.index].mean(1).rename("crispr"),
        prot_obj.peptide_raw_mean.reindex(df_corr_ppi.index).rename("prot"),
        pinfo.reindex(df_corr_ppi.index),
    ],
    axis=1,
).sort_values("crispr", ascending=False)
df_corr_ppi.to_csv(f"{RPATH}/2.SLProteinInteractions_ppi.csv.gz", compression="gzip")

for dt in ["gexp", "prot"]:
    fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

    hb = ax.hexbin(
        x=df_corr_ppi[dt],
        y=df_corr_ppi["ppi"],
        C=df_corr_ppi["crispr"],
        reduce_C_function=np.mean,
        cmap="Spectral",
        gridsize=50,
        lw=0,
    )

    cor, pval = spearmanr(df_corr_ppi[dt], df_corr_ppi["ppi"])
    annot_text = f"R={cor:.2g}, p={pval:.1e}"
    ax.text(0.95, 0.05, annot_text, fontsize=6, transform=ax.transAxes, ha="right")

    axins1 = inset_axes(ax, width="30%", height="5%", loc="upper left")
    cb = fig.colorbar(hb, cax=axins1, orientation="horizontal")
    cb.ax.tick_params(labelsize=4)
    cb.ax.set_title("CRISPR\n(mean scaled FC)", fontsize=4)

    ax.set_xlabel(
        "Gene expression (mean voom)"
        if dt == "gexp"
        else "Protein abundance (mean intensities)"
    )
    ax.set_ylabel("PPI interactions (log2)")

    plt.savefig(
        f"{RPATH}/2.SLProtein_robustness_landscape_{dt}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


#
#

dsets = ["crispr", "gexp", "prot"]

order = natsorted(set(df_corr_ppi["n_ppi"]))
pal = pd.Series(sns.color_palette("Blues_d", n_colors=len(order)).as_hex(), index=order)

_, axs = plt.subplots(1, len(dsets), figsize=(2 * len(dsets), 2), dpi=600)

for i, dt in enumerate(dsets):
    ax = axs[i]

    ax = GIPlot.gi_classification(
        dt, "n_ppi", df_corr_ppi, orient="h", palette=pal.to_dict(), order=order, ax=ax
    )

    if dt == "crispr":
        xlabel, title = "Gene essentiality\n(mean scaled FC)", "CRISPR-Cas9"
    elif dt == "gexp":
        xlabel, title = "Gene expression\n(mean voom)", "RNA-Seq"
    else:
        xlabel, title = "Protein abundance\n(mean intensities)", "SWATH-MS"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of protein interactions" if i == 0 else None)
    ax.set_title(title)

    if i != 0:
        ax.axes.yaxis.set_ticklabels([])

plt.subplots_adjust(hspace=0, wspace=0.05)
plt.savefig(
    f"{RPATH}/2.SLProtein_ppi_regression.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


#
#

order = natsorted(set(df_corr_ppi["n_ppi"]))

pals = {
    "paralog or singleton": {
        "unclassified": GIPlot.PAL_DTRACE[0],
        "paralog": GIPlot.PAL_DTRACE[1],
        "singleton": GIPlot.PAL_DTRACE[2],
    },
    "homomer or not (all PPI)": {
        "not-homomer": GIPlot.PAL_DTRACE[2],
        "homomer": GIPlot.PAL_DTRACE[1],
    },
}

hue_orders = {
    "paralog or singleton": ["paralog", "singleton", "unclassified"],
    "homomer or not (all PPI)": ["homomer", "not-homomer"],
}

for ft in ["paralog or singleton", "homomer or not (all PPI)"]:
    _, axs = plt.subplots(1, len(dsets), figsize=(2 * len(dsets), 2), dpi=600)

    for i, dt in enumerate(dsets):
        ax = axs[i]

        ax = GIPlot.gi_classification(
            dt,
            "n_ppi",
            df_corr_ppi,
            hue=ft,
            hue_order=hue_orders[ft],
            orient="h",
            palette=pals[ft],
            order=order,
            plot_legend=(i == (len(dsets) - 1)),
            legend_kws=dict(
                loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 6}
            ),
            ax=ax,
        )

        if dt == "crispr":
            xlabel, title = "Gene essentiality\n(mean scaled FC)", "CRISPR-Cas9"
        elif dt == "gexp":
            xlabel, title = "Gene expression\n(mean voom)", "RNA-Seq"
        else:
            xlabel, title = "Protein abundance\n(mean log2 intensities)", "SWATH-MS"

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of protein interactions" if i == 0 else None)
        ax.set_title(title)

        if i != 0:
            ax.axes.yaxis.set_ticklabels([])

    plt.subplots_adjust(hspace=0, wspace=0.05)
    plt.savefig(
        f"{RPATH}/2.SLProtein_ppi_regression_{ft}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


#
#

novel_ppis = df_corr.query(f"(prot_fdr < .05) & (prot_corr > 0.5)")
plot_df = novel_ppis["string_dist"].value_counts().reset_index()

_, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=600)

sns.barplot(
    "index",
    "string_dist",
    data=plot_df,
    linewidth=0.0,
    saturation=1.0,
    palette=GIPlot.PPI_PAL,
    order=GIPlot.PPI_ORDER[1:],
    ax=ax,
)

ax.set_ylabel("")
ax.set_xlabel("String PPI distance")
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

plt.savefig(
    f"{RPATH}/2.SLProtein_novel_ppi_string_dist_barplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


#
#

ppi_matrix = pd.DataFrame(
    [
        dict(protein1=p1, protein2=p2, value=c)
        for px, py, c in df_corr[["protein1", "protein2", "prot_corr"]].values
        for p1, p2 in [(px, py), (py, px)]
    ]
)
ppi_matrix = pd.pivot_table(
    ppi_matrix, index="protein1", columns="protein2", values="value", fill_value=np.nan
)

paralog_df = paralog_ds[paralog_ds["gene1 name"].isin(ppi_matrix.index)]
paralog_df = paralog_df[paralog_ds["gene2 name"].isin(ppi_matrix.index)]
paralog_df = paralog_df.groupby(["gene1 name", "gene2 name"])["dS"].first()

plot_df = pd.DataFrame(
    [
        dict(
            p1=p1,
            p2=p2,
            ds=paralog_df[(p1, p2)],
            corr=spearmanr(ppi_matrix.loc[p1], ppi_matrix.loc[p2], nan_policy="omit")[0],
        )
        for p1, p2 in paralog_df.index
    ]
)

ax = GIPlot.gi_regression("ds", "corr", plot_df)

plt.savefig(
    f"{RPATH}/2.SLProtein_paralogs_ds_regression.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


#
#

novel_ppi_matrix = pd.DataFrame(
    [
        dict(protein1=p1, protein2=p2, pval=pval)
        for px, py, pval in novel_ppis[["protein1", "protein2", "prot_pvalue"]].values
        for p1, p2 in [(px, py), (py, px)]
    ]
)
novel_ppi_matrix = pd.pivot_table(
    novel_ppi_matrix,
    index="protein1",
    columns="protein2",
    values="pval",
    fill_value=np.nan,
)

fs_mut = wes_obj.filter(mutation_class={"frameshift", "nonsense", "ess_splice"})
fs_mut = fs_mut.reindex(
    index=novel_ppi_matrix.index, columns=set(fs_mut).intersection(prot.columns)
).dropna()


fs_associations = []
for p in fs_mut.index:
    pis = novel_ppi_matrix.loc[p].dropna()

    for pi in pis.index:
        p_df = pd.concat([fs_mut.loc[p], crispr.loc[pi]], axis=1).dropna()

        n_muts = p_df[p].sum()

        if n_muts <= 10:
            continue

        ttest_stat, ttest_pval = ttest_ind(
            p_df[p_df[p] == 0][pi], p_df[p_df[p] == 1][pi], equal_var=False
        )

        fs_associations.append(
            dict(
                px=p,
                p_ppi=pi,
                ttest_stat=ttest_stat,
                ttest_pval=ttest_pval,
                n_muts=n_muts,
                n_obs=p_df.shape[0],
            )
        )
fs_associations = pd.DataFrame(fs_associations)
fs_associations["ttest_fdr"] = multipletests(fs_associations["ttest_pval"], method="fdr_bh")[1]
print(fs_associations.sort_values("ttest_fdr"))

g1, g2 = "STAG2", "SMC3"

plot_df = pd.concat(
    [prot.loc[g1], prot.loc[g2], fs_mut.loc[g1].rename("mutation")], axis=1
).dropna()

GIPlot.gi_regression_marginal(g1, g2, "mutation", plot_df, marginal_notch=True)

plt.savefig(
    f"{RPATH}/2.SLProtein_mutation_regression_{g1}_{g2}.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


#
#

ppi_matrix = pd.DataFrame(
    [
        dict(protein1=p1, protein2=p2, value=c)
        for px, py, c in df_corr[["protein1", "protein2", "prot_corr"]].values
        for p1, p2 in [(px, py), (py, px)]
    ]
)

ppi_matrix = pd.pivot_table(
    ppi_matrix, index="protein1", columns="protein2", values="value", fill_value=np.nan
)


sns.clustermap(ppi_matrix.fillna(0), mask=ppi_matrix.isnull(), cmap="Spectral_r")
plt.savefig(
    f"{RPATH}/2.SLProtein_ppi_matrix_clustermap.png",
    bbox_inches="tight",
    transparent=True,
    dpi=600,
)
plt.close("all")


# ppi_matrix = ppi_matrix.groupby("protein1")["protein2"].agg(set).to_dict()


def protein_regulation(sample, min_events=3):
    LOG.info(f"PPI: {sample.name}")
    sample_zscore = pd.Series(zscore(sample, nan_policy="omit"), index=sample.index)

    sample_zscores = {}

    for p in ppi_matrix:
        sample_ppi = sample_zscore.loc[ppi_matrix[p]].dropna()

        if len(sample_ppi) < min_events:
            continue

        sample_not_ppi = sample_zscore.drop(ppi_matrix[p]).dropna()

        tstat, tpvalue = ztest(sample_ppi, sample_not_ppi)
        ztest_score = np.log10(tpvalue) * (1 if tstat < 0 else -1)

        sample_zscores[p] = ztest_score

    sample_zscores = pd.Series(sample_zscores)

    return sample_zscores


ppi_dot = pd.DataFrame({s: protein_regulation(s_df) for s, s_df in prot_obj.protein_raw.T.iterrows()})

ppi_dot_norm = (ppi_dot - ppi_dot.mean()).divide(ppi_dot.std()).fillna(0)

ppi_dot_tsne, ppi_dot_pca = dim_reduction(ppi_dot_norm)

dimred = dict(
    tSNE=dict(proteomics=ppi_dot_tsne),
    pca=dict(proteomics=ppi_dot_pca),
)

for ctype in dimred:
    for dtype in dimred[ctype]:
        plot_df = pd.concat(
            [dimred[ctype][dtype], ss["tissue"]], axis=1, sort=False
        ).dropna()

        ax = plot_dim_reduction(plot_df, ctype=ctype, palette=GIPlot.PAL_TISSUE_2)
        ax.set_title(f"{ctype} - {dtype}")
        plt.savefig(
            f"{RPATH}/2.SLProtein_dimension_reduction_{dtype}_{ctype}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")
