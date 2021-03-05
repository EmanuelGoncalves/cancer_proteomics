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
import numpy.ma as ma
import itertools as it
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.GIPlot import GIPlot
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cancer_proteomics.notebooks import DataImport
from crispy.DataImporter import (
    CORUM,
    BioGRID,
    PPI,
    HuRI,
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
peptide_raw_mean = DataImport.read_peptide_raw_mean()

# Read Transcriptomics
gexp = DataImport.read_gene_matrix()

# Read CRISPR
crispr = DataImport.read_crispr_matrix()


# ### Protein-Protein interactions
#

corum_db = CORUM(ddir=PPIPATH)
biogrid_db = BioGRID(ddir=PPIPATH)
huri_db = HuRI(ddir=PPIPATH)


# ###
# Gene lists
cgenes = set(pd.read_csv(f"{DPATH}/cancer_genes_latest.csv.gz")["gene_symbol"])

genes = natsorted(
    list(set.intersection(set(prot.index), set(gexp.index), set(crispr.index)))
)
LOG.info(f"Genes: {len(genes)}")

# Paralog information
pinfo = pd.read_excel(
    f"{DPATH}/msb198871-sup-0004-datasetev1.xlsx", sheet_name="gene annotations"
)
pinfo = pinfo.groupby("gene name")[
    ["paralog or singleton", "homomer or not (all PPI)"]
].first()

paralog_ds = pd.read_excel(
    f"{DPATH}/msb198871-sup-0004-datasetev1.xlsx", sheet_name="paralog annotations"
)


# ### Protien-Protein correlations

def df_correlation(df_matrix, min_obs=15):
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


samples_overlap = list(set.intersection(set(prot), set(gexp), set(crispr)))
LOG.info(f"Overlapping Samples = {len(samples_overlap)}")

df_corr = (
    pd.concat(
        [
            df_correlation(prot.reindex(index=genes).T).add_prefix("prot_"),
            df_correlation(gexp.reindex(index=genes).T).add_prefix("gexp_"),
            df_correlation(crispr.reindex(index=genes).T).add_prefix("crispr_"),
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
    int((p1, p2) in huri_db.huri)
    for p1, p2 in df_corr[["protein1", "protein2"]].values
]

# String distance
ppi = PPI(ddir=PPIPATH).build_string_ppi(score_thres=900)
df_corr = PPI.ppi_annotation(
    df_corr, ppi, x_var="protein1", y_var="protein2", ppi_var="string_dist"
)
df_corr = df_corr.assign(string=(df_corr["string_dist"] == "1").astype(int))

# Number of observations
p_obs = {p: set(v.dropna().index) for p, v in prot.reindex(genes, columns=samples_overlap).iterrows()}
df_corr["nobs"] = [
    len(set.intersection(p_obs[p1], p_obs[p2]))
    for p1, p2 in df_corr[["protein1", "protein2"]].values
]

# Cancer genes
df_corr["protein1_cgene"] = df_corr["protein1"].isin(cgenes).astype(int)
df_corr["protein2_cgene"] = df_corr["protein2"].isin(cgenes).astype(int)

# Merged score
df_corr["merged_pvalue"] = df_corr[
    ["prot_pvalue", "gexp_pvalue", "crispr_pvalue"]
].prod(1)

# Export
df_corr.to_csv(f"{TPATH}/PPInteractions.csv.gz", compression="gzip", index=False)
# df_corr = pd.read_csv(f"{TPATH}/PPInteractions.csv.gz")

#
#
pal = sns.color_palette("tab10").as_hex()

_, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=600)

for i, n in enumerate(["prot", "gexp", "crispr"]):
    sns.ecdfplot(df_corr.query("nobs > 300")[f"{n}_corr"], color=pal[i], label=f"{n}", ax=ax)
    f"{n}={sum(df_corr[f'{n}_corr'].abs() > .5)}"

ax.set_xlabel("Protein-protein correlations\n(Pearson's r)")
ax.set_ylabel("Proportion")
ax.grid(True, axis="both", ls="-", lw=0.1, alpha=1.0, zorder=0)

ax.legend(frameon=False, prop={"size": 6})

plt.savefig(
    f"{RPATH}/PPInteractions_ecdf_correlations.pdf", bbox_inches="tight"
)
plt.savefig(
    f"{RPATH}/PPInteractions_ecdf_correlations.png", bbox_inches="tight"
)
plt.close("all")

##
igraph_nets = {}
for n in ["prot", "gexp", "crispr"]:
    # Network
    net = df_corr.query(f"{n}_corr > .5")
    net_i = igraph.Graph(directed=False)

    # Initialise network lists
    edges = [(px, py) for px, py in net[["protein1", "protein2"]].values]
    vertices = list(set(net["protein1"]).union(net["protein2"]))

    # Add nodes
    net_i.add_vertices(vertices)

    # Add edges
    net_i.add_edges(edges)

    # Simplify
    net_i = net_i.simplify(combine_edges="max")
    LOG.info(net_i.summary())
    LOG.info(net_i.transitivity_undirected())

    igraph_nets[n] = net_i

# ### Define novel PPIs
novel_ppis = df_corr.query(f"(prot_fdr < .05) & (prot_corr > 0.5)")

# Export
ppis = df_corr[
    (df_corr["prot_corr"].abs() > .5) | (df_corr["gexp_corr"].abs() > .5) | (df_corr["crispr_corr"].abs() > .5)
    ]
ppis.round(4).to_csv(f"{TPATH}/PPInteractions_filtered.csv", index=False)


#
rc_dict = dict()
for y in ["corum", "biogrid", "string", "huri"]:
    rc_dict[y] = dict()
    for x in ["prot", "gexp", "crispr", "merged"]:
        rc_df = df_corr.sort_values(f"{x}_pvalue")[y].reset_index(drop=True).copy()

        rc_df_y = np.cumsum(rc_df) / np.sum(rc_df)
        rc_df_x = np.array(rc_df.index) / rc_df.shape[0]
        rc_df_auc = auc(rc_df_x, rc_df_y)

        rc_dict[y][x] = dict(x=list(rc_df_x), y=list(rc_df_y), auc=rc_df_auc)

rc_pal = dict(
    biogrid=sns.color_palette("tab20c").as_hex()[0:4],
    corum=sns.color_palette("tab20c").as_hex()[4:8],
    string=sns.color_palette("tab20c").as_hex()[8:12],
    huri=sns.color_palette("tab20c").as_hex()[12:16],
)

# Recall curves
_, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

for db in rc_dict:
    for i, ds in enumerate(rc_dict[db]):
        ax.plot(
            rc_dict[db][ds]["x"],
            rc_dict[db][ds]["y"],
            label=f"{db} {ds} (AUC {rc_dict[db][ds]['auc']:.2f})",
            c=rc_pal[db][i],
        )

ax.plot([0, 1], [0, 1], "k--", lw=0.3)
ax.legend(loc="lower right", frameon=False)

ax.set_ylabel("Cumulative sum")
ax.set_xlabel("Ranked correlation")
ax.grid(True, axis="both", ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.savefig(
    f"{RPATH}/PPInteractions_roc_curves_overlap.pdf", bbox_inches="tight"
)
plt.savefig(
    f"{RPATH}/PPInteractions_roc_curves_overlap.png", bbox_inches="tight"
)
plt.close("all")

# Recall barplot
plot_df = pd.DataFrame(
    [
        dict(ppi=db, dtype=ds, auc=rc_dict[db][ds]["auc"])
        for db in rc_dict
        for ds in rc_dict[db]
    ]
)

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
    f"{RPATH}/PPInteractions_roc_barplot_overlap.pdf", bbox_inches="tight"
)
plt.savefig(
    f"{RPATH}/PPInteractions_roc_barplot_overlap.png", bbox_inches="tight"
)
plt.close("all")

# PPI types
#
df_corr_ppi = df_corr.query(f"(prot_fdr < .05) & (prot_corr > 0.5)")
df_corr_ppi = pd.concat(
    [df_corr_ppi["protein1"], df_corr_ppi["protein2"]], ignore_index=True
).value_counts()
df_corr_ppi = pd.concat(
    [
        pd.qcut(df_corr_ppi, np.arange(0, 1.1, 0.1), duplicates="drop")
            .apply(lambda v: f"{round(v.left):.0f}-{round(v.right):.0f}")
            .rename("n_ppi"),
        df_corr_ppi.pipe(np.log2).rename("ppi"),
        gexp.loc[df_corr_ppi.index].mean(1).rename("gexp"),
        crispr.loc[df_corr_ppi.index].mean(1).rename("crispr"),
        peptide_raw_mean.reindex(df_corr_ppi.index).rename("prot"),
        pinfo.reindex(df_corr_ppi.index),
    ],
    axis=1,
).sort_values("crispr", ascending=False)

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
        f"{RPATH}/PPInteractions_robustness_landscape_{dt}.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{RPATH}/PPInteractions_robustness_landscape_{dt}.png",
        bbox_inches="tight",
    )
    plt.close("all")

#
dsets = ["crispr", "gexp", "prot"]

order = natsorted(set(df_corr_ppi["n_ppi"]))
pal = pd.Series(
    sns.color_palette("Blues_d", n_colors=len(order)).as_hex(), index=order
)

_, axs = plt.subplots(1, len(dsets), figsize=(3 * len(dsets), 2), dpi=600)

for i, dt in enumerate(dsets):
    ax = axs[i]

    ax = GIPlot.gi_classification(
        dt,
        "n_ppi",
        df_corr_ppi,
        orient="h",
        palette=pal.to_dict(),
        order=order,
        ax=ax,
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
    f"{RPATH}/PPInteractions_ppi_regression.pdf",
    bbox_inches="tight",
)
plt.savefig(
    f"{RPATH}/PPInteractions_ppi_regression.png",
    bbox_inches="tight",
)
plt.close("all")

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
        f"{RPATH}/PPInteractions_ppi_regression_{ft}.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{RPATH}/PPInteractions_ppi_regression_{ft}.png",
        bbox_inches="tight",
    )
    plt.close("all")
