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
from crispy.GIPlot import GIPlot
from adjustText import adjust_text
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import pearsonr
from multiomics_integration.notebooks import DataImport, PALETTE_TTYPE


LOG = logging.getLogger("multiomics_integration")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("multiomics_integration", "plots/DIANN/")


# ### Imports

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
prot = DataImport.read_protein_matrix(map_protein=True)

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
lm_drug = pd.read_csv(f"{TPATH}/lm_sklearn_degr_drug_annotated_diann_051021.csv.gz")
lm_crispr = pd.read_csv(f"{TPATH}/lm_sklearn_degr_crispr_annotated_diann_051021.csv.gz")

lm_drug_matrix = pd.pivot_table(lm_drug, index="y_id", columns="x_id", values="nc_pval", fill_value=np.nan)
lm_drug_matrix.to_csv(f"{TPATH}/drug_protein_associations_nc_pval_matrix.csv.gz", compression="gzip")

# Selective and predictive dependencies
#
R2_THRES = 0.3
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

# Selectivity plot
#
plot_info = [
    ("CRISPR-Cas9", "o", CrispyPlot.PAL_DTRACE[3]),
    ("Drug Response", "o", CrispyPlot.PAL_DTRACE[1]),
]

for i, (n, m, c) in enumerate(plot_info):
    _, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

    labels = []

    n_df = dep_df.replace({
        "dtype": {"drug": "Drug Response", "crispr": "CRISPR-Cas9"},
    }).query(f"dtype == '{n}'").copy()

    ax.scatter(
        n_df["skew"],
        n_df["r2"],
        marker=m,
        s=3,
        c=c,
        zorder=0,
        alpha=0.8,
        lw=0,
        label=n,
    )

    ax.set_xlabel("Standardised skewness")
    ax.set_ylabel("Pearson's r")
    ax.set_title(n)

    for _, row in (
        n_df.query(f"r2 > {R2_THRES}").sort_values("skew").head(15).iterrows()
    ):
        labels.append(row)

    labels = [
        ax.text(
            row["skew"],
            row["r2"],
            row["y_id"] if row["dtype"] == "CRISPR-Cas9" else row["y_id"].split(";")[1],
            color="k",
            fontsize=5,
            zorder=3,
        )
        for row in labels
    ]
    adjust_text(
        labels, arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3), ax=ax
    )

    ax.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)
    ax.axhline(R2_THRES, c="black", lw=0.3, ls="--", zorder=2)
    ax.axvline(-2 if n == "CRISPR-Cas9" else -1, c="black", lw=0.3, ls="--", zorder=2)

    plt.savefig(
        f"{RPATH}/TopHits_selectivity_predictive_scatter_{n}.pdf", bbox_inches="tight"
    )
    plt.savefig(
        f"{RPATH}/TopHits_selectivity_predictive_scatter_{n}.png", bbox_inches="tight"
    )


# Predictive features of selective and predictive dependencies
#
tophits = dep_df.query(f"(r2 > {R2_THRES}) & (skew < {SKEW_THRES})")

tophits_feat_drug = set(lm_drug.query(f"fdr < {FDR_THRES}")["x_id"])
tophits_feat_crispr = set(lm_crispr.query(f"fdr < {FDR_THRES}")["x_id"])
tophits_feat_union = set.union(tophits_feat_drug, tophits_feat_crispr)

tophits_feat = pd.concat(
    [
        lm_drug[lm_drug["x_id"].isin(tophits_feat_union)].assign(dtype="drug"),
        lm_crispr[lm_crispr["x_id"].isin(tophits_feat_union)].assign(dtype="crispr"),
    ]
)
tophits_feat = tophits_feat[tophits_feat["y_id"].isin(tophits["y_id"])]
tophits_feat = tophits_feat.query(f"fdr < {FDR_THRES}")


# Top dependencies
#

topdep = ["FOXA1"]

for y_id in topdep:
    # y_id = "TP63"
    plot_df = (
        tophits_feat.query(f"y_id == '{y_id}'")
        .query("n > 60")
        .head(10)
        .sort_values("pval")
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
    plt.savefig(
        f"{RPATH}/TopHits_top_associations_{y_id}.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

# Association plots
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
    # ("REXO4", "TP53", "crispr", None),
    ("BSG", "FOXA1", "crispr", ["Breast"]),
    ("PRKAR1A", "PRKAR1A", "crispr", None),
    ("HNRNPH1", "HNRNPH1", "crispr", None),
    ("TP53", "MDM2", "crispr", None),
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
            ss["Tissue_type"],
        ],
        axis=1,
        sort=False,
    ).dropna(subset=[f"{c}_y", f"{p}_prot"])

    # Protein
    ax = GIPlot.gi_tissue_plot(
        f"{p}_prot", f"{c}_y", plot_df, pal=PALETTE_TTYPE, hue="Tissue_type"
    )

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
            f"{p}_prot",
            f"{c}_y",
            plot_df[plot_df["Tissue_type"].isin(ctissues)],
            pal=PALETTE_TTYPE,
            hue="Tissue_type",
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
    ax = GIPlot.gi_tissue_plot(
        f"{p}_gexp", f"{c}_y", plot_df, pal=PALETTE_TTYPE, hue="Tissue_type"
    )

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

##
#

pam50 = pd.read_csv(f"{DPATH}/breast_subtypes.txt", sep="\t", index_col=0)

p, c, dtype, ctissues = ("BSG", "FOXA1", "crispr", ["Breast"])

plot_df = pd.concat(
    [
        crispr.loc[["FOXA1"]].T,
        prot.loc[["BSG", "VIM"]].T,
        ss["Tissue_type"],
        pam50[["pam50", "Phenotype"]],
        ss[["F2", "Cell_line"]],
    ],
    axis=1,
    sort=False,
).dropna(subset=[p, c])

plot_df = plot_df.replace({"pam50": {np.nan: "NA"}})

pal = {"NA": "#e1e1e1", 0: CrispyPlot.PAL_DTRACE[1], "LumB": '#1f77b4', "Her2": '#ff7f0e', "Basal": '#2ca02c', "LumA": '#d62728', "Normal": '#9467bd'}

# BSG ~ FOXA1 scatter
g = GIPlot.gi_regression_marginal(
    "BSG", "FOXA1", "pam50",
    plot_df,
    discrete_pal=pal,
    plot_reg=False,
    plot_annot=False,
    legend_title="Breast - PAM50",
    hue_order=pal.keys(),
    scatter_kws=dict(edgecolor="w", lw=0.1, s=16),
)

for c in list(pal.keys())[2:-1]:
    df = plot_df.query(f"pam50 == '{c}'").dropna(subset=["BSG", "FOXA1"])

    r, p = pearsonr(df["BSG"], df["FOXA1"])
    print(f"{c}: Pearson's r = {r:.2f}, p-value = {p:.2g}")

    sns.regplot(
        x=df["BSG"],
        y=df["FOXA1"],
        data=plot_df,
        color=pal[c],
        truncate=True,
        fit_reg=True,
        scatter=False,
        ci=None,
        line_kws=dict(lw=1.0, color=pal[c]),
        ax=g.ax_joint,
    )

g.ax_joint.set_xlabel(f"BSG\nProtein intensities")
g.ax_joint.set_ylabel(f"FOXA1\nCRISPR-Cas9 (log2 FC)")

plt.gcf().set_size_inches(2, 2)

plt.savefig(f"{RPATH}/TopHits_FOXA1_BSG_scatter.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/TopHits_FOXA1_BSG_scatter.png", bbox_inches="tight", dpi=600)

#

for n, df in plot_df.query("pam50 != 'NA' & pam50 != 'Normal'").groupby("pam50"):
    r, p = pearsonr(df["BSG"], df["FOXA1"])
    print(f"{n}: Pearson's r = {r:.2f}, p-value = {p:.2g}")

g = sns.lmplot(
    data=plot_df.query("pam50 != 'NA'"),
    x="BSG",
    y="FOXA1",
    hue="pam50",
    ci=None,
    palette=pal,
    legend_out=False
)

sns.despine(top=False, right=False)

g.ax.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)
g.ax.set_xlabel(f"BSG\nProtein intensities")
g.ax.set_ylabel(f"FOXA1\nCRISPR-Cas9 (log2 FC)")

plt.gcf().set_size_inches(2, 2)

plt.savefig(f"{RPATH}/TopHits_FOXA1_BSG_lm_plot.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/TopHits_FOXA1_BSG_lm_plot.png", bbox_inches="tight", dpi=600)

plt.close("all")

#

for n, df in plot_df.dropna(subset=["Phenotype"]).groupby("Phenotype"):
    if df.shape[0] > 2:
        r, p = pearsonr(df["BSG"], df["FOXA1"])
        print(f"{n}: Pearson's r = {r:.2f}, p-value = {p:.2g}")


g = sns.lmplot(
    data=plot_df.dropna(subset=["Phenotype"]),
    x="BSG",
    y="FOXA1",
    hue="Phenotype",
    legend_out=False
)

sns.despine(top=False, right=False)

g.ax.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)
g.ax.set_xlabel(f"BSG\nProtein intensities")
g.ax.set_ylabel(f"FOXA1\nCRISPR-Cas9 (log2 FC)")

plt.gcf().set_size_inches(2.5, 2.5)

plt.savefig(f"{RPATH}/TopHits_FOXA1_BSG_lm_plot_TNBC.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/TopHits_FOXA1_BSG_lm_plot_TNBC.png", bbox_inches="tight", dpi=600)

plt.close("all")


# BSG ~ F2 scatter
g = GIPlot.gi_regression_marginal(
    "F2", "BSG", "pam50",
    plot_df,
    discrete_pal=pal,
    legend_title="Breast - PAM50",
    hue_order=pal.keys(),
    scatter_kws=dict(edgecolor="w", lw=0.1, s=16),
)

g.ax_joint.set_xlabel(f"BSG\nProtein intensities")
g.ax_joint.set_ylabel(f"MOFA Factor 2")

plt.gcf().set_size_inches(2, 2)

plt.savefig(f"{RPATH}/TopHits_F2_BSG_scatter.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/TopHits_F2_BSG_scatter.png", bbox_inches="tight", dpi=600)
