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
from cancer_proteomics.notebooks import DataImport, PALETTE_TTYPE


LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/DIANN/")


# ### Imports

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
prot = DataImport.read_protein_matrix(map_protein=True)
# Read CRISPR
crispr = DataImport.read_crispr_matrix()

# Read Drug-response
drespo = DataImport.read_drug_response()

dmaxc = DataImport.read_drug_max_concentration()
dmaxc = dmaxc.reindex(drespo.index)

# MOBEM
mobem = DataImport.read_mobem()

# LM associations
#
lm_drug = pd.read_csv(f"{TPATH}/lm_sklearn_degr_drug_annotated_diann_051021.csv.gz")
lm_crispr = pd.read_csv(f"{TPATH}/lm_sklearn_degr_crispr_annotated_diann_051021.csv.gz")


##
plot_df = lm_drug.query("nc_fdr < 0.05")

f, ax = plt.subplots(1, 1, figsize=(2, 3.5))

for l, df in plot_df.groupby("ppi"):
    ax.scatter(
        df["nc_beta"],
        -np.log10(df["nc_pval"]),
        color=CrispyPlot.PPI_PAL[l],
        marker="o",
        label=l,
        edgecolor="white",
        lw=0.1,
        alpha=0.5,
    )

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
ax.set_xlabel(f"Beta")
ax.set_ylabel(f"P-value (log10)")
ax.set_title("Drug - Protein")

texts_index = [152780, 3734420, 437633, 193807, 15827, 2590752, 969, 156096, 1465877, 12914, 75507, 1420284, 435829, 25166, 19544]
texts = [
    ax.text(x, -np.log10(y), f"{g.split(';')[1]} : {d}", color="k", fontsize=5)
    for i, (x, y, g, d, t) in plot_df.loc[texts_index][
        ["nc_beta", "nc_pval", "y_id", "x_id", "target"]
    ].iterrows()
]
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
    ax=ax,
)

plt.legend(
    frameon=False,
    prop={"size": 6},
    title="PPI distance",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
).get_title().set_fontsize("6")

plt.savefig(
    f"{RPATH}/AssociationsOverview_volcano_drug.png", bbox_inches="tight", dpi=600
)
plt.close("all")


###
plot_df = lm_crispr.query("nc_fdr < 0.05")

f, ax = plt.subplots(1, 1, figsize=(2, 3.5))

for l, df in plot_df.groupby("ppi"):
    ax.scatter(
        df["nc_beta"],
        -np.log10(df["nc_pval"]),
        color=CrispyPlot.PPI_PAL[l],
        marker="o",
        label=l,
        edgecolor="white",
        lw=0.1,
        alpha=0.5,
    )

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
ax.set_xlabel(f"Beta")
ax.set_ylabel(f"P-value (log10)")
ax.set_title("CRISPR - Protein")

texts_index = [
    0, 239, 2736, 5527191, 42171, 791944, 383621, 9, 292692, 28850344, 102680, 492, 1621, 71166160, 953640, 2546648
]
texts = [
    ax.text(x, -np.log10(y), f"{g} : {d}", color="k", fontsize=5)
    for i, (x, y, g, d) in plot_df.loc[texts_index][
        ["nc_beta", "nc_pval", "y_id", "x_id"]
    ].iterrows()
]
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
    ax=ax,
)

plt.legend(
    frameon=False,
    prop={"size": 6},
    title="PPI distance",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
).get_title().set_fontsize("6")

plt.savefig(
    f"{RPATH}/AssociationsOverview_volcano_crispr.png", bbox_inches="tight", dpi=600
)
plt.close("all")


## Selected pairs

gi_pairs = [
    ("HNRNPH1", "HNRNPH1", "crispr"),
    ("ERBB2", "ERBB2", "crispr"),
    ("EGFR", "EGFR", "crispr"),
    ("BAX", "1047;Nutlin-3a (-);GDSC2", "drug"),
    ("EGFR", "1010;Gefitinib;GDSC1", "drug"),
    ("PPA1", "PPA2", "crispr"),
    ("PPA2", "PPA1", "crispr"),
]

for p, c, dtype in gi_pairs:
    # p, c, dtype = ("PPA2", "PPA1", "crispr")

    plot_df = pd.concat(
        [
            drespo.loc[[c]].T.add_suffix("_y")
            if dtype == "drug"
            else crispr.loc[[c]].T.add_suffix("_y"),
            prot.loc[[p]].T.add_suffix("_prot"),
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
        f"{RPATH}/AssociationsOverview_scatter_{p}_{c}.png",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{RPATH}/AssociationsOverview_scatter_{p}_{c}.pdf",
        bbox_inches="tight",
    )
    plt.close("all")

# ERBB2 amplifications
plot_df = pd.concat(
    [
        crispr.loc[["ERBB2"]].T.add_prefix("CRISPR_"),
        prot.loc[["ERBB2"]].T.add_prefix("PROT_"),
        ss["Tissue_type"],
        mobem.loc[["gain.cnaPANCAN301..CDK12.ERBB2.MED24."]].T,
    ],
    axis=1,
    sort=False,
).dropna(subset=["CRISPR_ERBB2", "PROT_ERBB2"])

g = GIPlot.gi_regression_marginal(
    "PROT_ERBB2", "CRISPR_ERBB2", "gain.cnaPANCAN301..CDK12.ERBB2.MED24.",
    plot_df,
    legend_title="ERBB2 amplification",
    scatter_kws=dict(edgecolor="w", lw=0.1, s=16),
)

g.ax_joint.set_xlabel(f"ERBB2\nProtein intensities")
g.ax_joint.set_ylabel(f"ERBB2\nCRISPR-Cas9 (log2 FC)")

plt.gcf().set_size_inches(2, 2)

plt.savefig(f"{RPATH}/TopHits_ERBB2_ERBB2_scatter.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/TopHits_ERBB2_ERBB2_scatter.png", bbox_inches="tight", dpi=600)

# MET
plot_df = pd.concat(
    [
        prot.loc[["MET"]].T,
        drespo.loc[["1403;AZD6094;GDSC1"]].T,
        ss["Tissue_type"],
        mobem.loc[["gain.cnaPANCAN129..MET."]].T,
    ],
    axis=1,
    sort=False,
).dropna(subset=["MET", "1403;AZD6094;GDSC1", "gain.cnaPANCAN129..MET."])

g = GIPlot.gi_regression_marginal(
    "MET", "1403;AZD6094;GDSC1", "gain.cnaPANCAN129..MET.",
    plot_df,
    legend_title="MET amplification",
    scatter_kws=dict(edgecolor="w", lw=0.1, s=16),
)

g.ax_joint.set_xlabel(f"MET\nProtein intensities")
g.ax_joint.set_ylabel(f"AZD6094\nDrug response IC50")

plt.gcf().set_size_inches(2, 2)

plt.savefig(f"{RPATH}/TopHits_MET_1403;AZD6094;GDSC1_scatter.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/TopHits_MET_1403;AZD6094;GDSC1_scatter.png", bbox_inches="tight", dpi=600)
