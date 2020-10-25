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
import numpy.ma as ma
import itertools as it
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.GIPlot import GIPlot
from scipy.stats import pearsonr, spearmanr
from adjustText import adjust_text
from matplotlib_venn import venn2, venn2_circles
from sklearn.metrics.ranking import auc
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cancer_proteomics.notebooks import DataImport, two_vars_correlation
from crispy.DataImporter import CORUM, BioGRID, PPI, HuRI


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

# Read Drug-response
drespo = DataImport.read_drug_response()

dmaxc = DataImport.read_drug_max_concentration()
dmaxc = dmaxc.reindex(drespo.index)

# LM associations
#
lm_drug = pd.read_csv(f"{TPATH}/lm_sklearn_degr_drug_annotated.csv.gz")
lm_crispr = pd.read_csv(f"{TPATH}/lm_sklearn_degr_crispr_annotated.csv.gz")


# Selective and predictive dependencies
#
R2_THRES = 0.4
SKEW_THRES = -1
FDR_THRES = 0.1

dep_df = pd.concat(
    [
        lm_drug.groupby("y_id")[["r2", "skew"]]
        .first()
        .reset_index()
        .assign(dtype="drug"),
        lm_crispr.groupby("y_id")[["r2", "skew"]]
        .first()
        .reset_index()
        .assign(dtype="crispr"),
    ]
).dropna()

x = dep_df.query("dtype == 'drug'")[["r2"]]

pd.Series(StandardScaler().fit_transform(x)[:, 0]).hist(); plt.show()

# Selectivity plot
#
_, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

plot_info = [("crispr", "o", "#009EAC"), ("drug", "X", "#FEC041")]
for i, (n, m, c) in enumerate(plot_info):
    n_df = dep_df.query(f"dtype == '{n}'")

    n_ax = ax if n == "crispr" else ax.twiny()

    n_ax.scatter(
        n_df["skew"], n_df["r2"], marker=m, s=3, c=c, zorder=(i + 1), alpha=0.8, lw=0
    )

    n_ax.set_xlabel(f"{n} skewness", color=c)
    n_ax.set_ylabel("R2")

    labels = n_df.query(f"r2 > {R2_THRES}").sort_values("skew").head(20)
    labels = [
        n_ax.text(
            row["skew"],
            row["r2"],
            row["y_id"] if n == "crispr" else row["y_id"].split(";")[1],
            color="k",
            fontsize=4,
            zorder=3,
        )
        for _, row in labels.iterrows()
    ]
    adjust_text(
        labels, arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3), ax=n_ax
    )

ax.grid(axis="y", lw=0.1, color="#e1e1e1", zorder=0)
ax.axhline(R2_THRES, c="#E3213D", lw=0.3, ls="--")

plt.savefig(f"{RPATH}/TopHits_selectivity_predictive_scatter.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/TopHits_selectivity_predictive_scatter.png", bbox_inches="tight")
plt.close("all")


# Predictive features of selective and predictive dependencies
#
tophits = dep_df.query(f"(r2 > {R2_THRES}) & (skew < {SKEW_THRES})")

tophits_feat_drug = set(lm_drug.query(f"fdr < {FDR_THRES}")["x_id"])
tophits_feat_crispr = set(lm_crispr.query(f"fdr < {FDR_THRES}")["x_id"])
tophits_feat_union = set.union(tophits_feat_drug, tophits_feat_crispr)

_, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=600)
venn_groups = [tophits_feat_drug, tophits_feat_crispr]
venn2(
    venn_groups, set_labels=["Drug", "CRISPR"], set_colors=["#FEC041", "#009EAC"], ax=ax
)
venn2_circles(venn_groups, linewidth=0.5, ax=ax)
plt.title(f"Top protein features (FDR < {FDR_THRES * 100:.0f}%)")
plt.savefig(f"{RPATH}/TopHits_features_venn.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/TopHits_features_venn.png", bbox_inches="tight")
plt.close("all")

tophits_feat = pd.concat(
    [
        lm_drug[lm_drug["x_id"].isin(tophits_feat_union)].assign(dtype="drug"),
        lm_crispr[lm_crispr["x_id"].isin(tophits_feat_union)].assign(dtype="crispr"),
    ]
)
tophits_feat = tophits_feat[tophits_feat["y_id"].isin(tophits["y_id"])]
tophits_feat = tophits_feat.query(f"fdr < {FDR_THRES}")


def calculate_score(pval, beta):
    s = np.log10(pval)
    if beta > 0:
        s *= -1
    return s


tophits_feat["score"] = [
    calculate_score(p, b) for p, b in tophits_feat[["pval", "beta"]].values
]

# Top dependencies
#

topdep = ["FOXA1"]

# Top associations
for y_id in topdep:
    # y_id = "TP63"
    plot_df = (
        tophits_feat.query(f"y_id == '{y_id}'")
        .head(10)
        .reset_index(drop=True)
        .reset_index()
    )
    plot_df = plot_df.assign(logpval=-np.log10(plot_df["pval"]).values)
    plot_df = plot_df.fillna("X")

    fig, ax = plt.subplots(1, 1, figsize=(plot_df.shape[0] * 0.2, 1.5))

    for t, df in plot_df.groupby("ppi"):
        ax.bar(
            df["index"].values,
            df["logpval"].values,
            color=GIPlot.PPI_PAL[t],
            align="center",
            zorder=5,
            linewidth=0,
        )

    for g, p in plot_df[["x_id", "index"]].values:
        ax.text(
            p,
            0.1,
            g,
            ha="center",
            va="bottom",
            fontsize=6,
            zorder=10,
            rotation="vertical",
            color="white",
        )

    for x, y, t, b in plot_df[["index", "logpval", "ppi", "beta"]].values:
        if str(t) != "nan":
            c = GIPlot.PAL_DTRACE[1] if t == "T" else GIPlot.PAL_DTRACE[2]

            ax.text(x, y + 0.25, t, color=c, ha="center", fontsize=6, zorder=10)
            ax.text(
                x,
                -0.5,
                f"{b:.1f}",
                color=c,
                ha="center",
                va="top",
                fontsize=6,
                rotation="vertical",
                zorder=10,
            )

    ax.set_title(f"{y_id} (R-squared={plot_df['r2'].max():.2f})")
    plt.ylabel(f"Linear regressions\n(p-value log10)")

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
    ax.axes.get_xaxis().set_ticks([])

    plt.savefig(f"{RPATH}/TopHits_top_associations_{y_id}.pdf", bbox_inches="tight")
    plt.savefig(f"{RPATH}/TopHits_top_associations_{y_id}.png", bbox_inches="tight", dpi=600)
    plt.close("all")

#
#
gi_pairs = [
    # (
    #     "RPL22",
    #     "WRN",
    #     "crispr",
    #     ["Large Intestine", "Endometrium", "Stomach", "Ovary"],
    # ),
    # ("RAD21", "STAG1", "crispr", ["Bone", "Central Nervous System", "Breast"]),
    # ("MET", "1403;AZD6094;GDSC1", "drug", ["Stomach", "Esophagus", "Lung"]),
    # ("ACIN1", "BRAF", "crispr", ["Skin", "Breast", "Large Intestine", "Ovary"]),
    ("BSG", "FOXA1", "crispr", ["Breast"]),
    ("PRKAR1A", "PRKAR1A", "crispr", None),
    ("HNRNPH1", "HNRNPH1", "crispr", None),
    ("TP53", "MDM2", "crispr", None),
    ("REXO4", "TP53", "crispr", None),
]

for p, c, dtype, ctissues in gi_pairs:
    # p, c, dtype, ctissues = ("BSG", "FOXA1", "crispr", ["Breast"])

    plot_df = pd.concat(
        [
            drespo.loc[[c]].T.add_suffix("_y")
            if dtype == "drug"
            else crispr.loc[[c]].T.add_suffix("_y"),
            prot.loc[[p]].T.add_suffix("_prot"),
            gexp.loc[[p]].T.add_suffix("_gexp"),
            ss["tissue"],
        ],
        axis=1,
        sort=False,
    ).dropna(subset=[f"{c}_y", f"{p}_prot"])

    # Protein
    ax = GIPlot.gi_tissue_plot(f"{p}_prot", f"{c}_y", plot_df)

    if dtype == "drug":
        ax.axhline(np.log(dmaxc[c]), ls="--", lw=0.3, color=CrispyPlot.PAL_DTRACE[1])

    ax.set_xlabel(f"{p}\nProtein intensities")
    ax.set_ylabel(
        f"{c}\n{'Drug response IC50' if dtype == 'drug' else 'CRISPR-Cas9 (log2 FC)'}"
    )
    plt.savefig(
        f"{RPATH}/TopHits_{p}_{c}_{dtype}_regression_tissue_plot.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{RPATH}/TopHits_{p}_{c}_{dtype}_regression_tissue_plot.png",
        bbox_inches="tight",
    )
    plt.close("all")

    # Protein
    if ctissues is not None:
        ax = GIPlot.gi_tissue_plot(
            f"{p}_prot", f"{c}_y", plot_df[plot_df["tissue"].isin(ctissues)]
        )

        if dtype == "drug":
            ax.axhline(
                np.log(dmaxc[c]), ls="--", lw=0.3, color=CrispyPlot.PAL_DTRACE[1]
            )

        ax.set_xlabel(f"{p}\nProtein intensities")
        ax.set_ylabel(
            f"{c}\n{'Drug response IC50' if dtype == 'drug' else 'CRISPR-Cas9 (log2 FC)'}"
        )
        plt.savefig(
            f"{RPATH}/TopHits_{p}_{c}_{dtype}_regression_tissue_plot_selected.pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            f"{RPATH}/TopHits_{p}_{c}_{dtype}_regression_tissue_plot_selected.png",
            bbox_inches="tight",
        )
        plt.close("all")

    # Gene expression
    ax = GIPlot.gi_tissue_plot(f"{p}_gexp", f"{c}_y", plot_df)

    if dtype == "drug":
        ax.axhline(np.log(dmaxc[c]), ls="--", lw=0.3, color=CrispyPlot.PAL_DTRACE[1])

    ax.set_xlabel(f"{p}\nGene expression (RNA-Seq voom)")
    ax.set_ylabel(
        f"{c}\n{'Drug response IC50' if dtype == 'drug' else 'CRISPR-Cas9 (log2 FC)'}"
    )
    plt.savefig(
        f"{RPATH}/TopHits_{p}_{c}_{dtype}_regression_tissue_plot_gexp.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{RPATH}/TopHits_{p}_{c}_{dtype}_regression_tissue_plot_gexp.png",
        bbox_inches="tight",
    )
    plt.close("all")

#
breast_subtypes = pd.read_csv(f"{DPATH}/breast_subtypes.txt", sep="\t", index_col=0)

corder = ["FOXA1", "BSG"]

plot_df = pd.concat(
    [crispr.loc["FOXA1"], prot.loc["BSG"], breast_subtypes["pam50"]], axis=1
).dropna(subset=corder)

plot_df = pd.melt(plot_df, value_vars=["FOXA1", "BSG"], id_vars=["pam50"]).dropna()

g = sns.catplot(
    "pam50",
    "value",
    data=plot_df,
    col="variable",
    facet_kws=dict(despine=False),
    sharex="row",
    sharey="col",
    height=2.5,
    kind="swarm",
    col_order=corder,
)

g.set_axis_labels("Breast cancer PAM50 subtypes", "")

titles = ["FOXA1\nCRISPR-Cas9 (log2 FC)", "BSG\nProtein intensities"]
for i, ax in enumerate(g.axes[0]):
    ax.set_title(titles[i])
    sns.boxplot(
        "pam50",
        "value",
        data=plot_df.query(f"variable == '{corder[i]}'"),
        sym="",
        boxprops=dict(facecolor=(0, 0, 0, 0)),
        zorder=2,
        ax=ax,
    )
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")
    ax.set_ylabel("")

plt.savefig(f"{RPATH}/TopHits_FOXA1_BSG_Breast_Subtypes.pdf", bbox_inches="tight")
plt.savefig(
    f"{RPATH}/TopHits_FOXA1_BSG_Breast_Subtypes.png", bbox_inches="tight", dpi=600
)
plt.close("all")

#

x = pd.concat([
    drespo.loc[["1909;Venetoclax;GDSC2", "1373;Dabrafenib;GDSC1", "1427;AZD5582;GDSC1"]].T,
    prot.loc[["TSNAX", "DBNL", "NOC2L"]].T,
    ss["tissue"]
], axis=1)
