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
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.GIPlot import GIPlot
from itertools import zip_longest
from sklearn.metrics.ranking import auc
from scipy.stats import pearsonr, spearmanr
from cancer_proteomics.eg.LMModels import LMModels
from sklearn.preprocessing import MinMaxScaler
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    CRISPR,
    CORUM,
    Sample,
    DPATH,
    BioGRID,
    HuRI,
    PPI,
)


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


# Data-sets
#

prot_obj = Proteomics()
gexp_obj = GeneExpression()
crispr_obj = CRISPR()


# Filter data-sets
#

ss = Sample().samplesheet

prot = prot_obj.filter()
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp_obj.filter()
LOG.info(f"Transcriptomics: {gexp.shape}")

crispr = crispr_obj.filter(dtype="merged")
LOG.info(f"CRISPR: {crispr.shape}")


# CORUM + BioGRID
#

corum_db = CORUM()
biogrid_db = BioGRID()
huri_db = HuRI()


# Gene sets
#

cgenes = pd.read_csv(f"{DPATH}/cancer_genes_latest.csv.gz")
cgenes = list(set(cgenes["gene_symbol"]))

patt = pd.read_csv(f"{RPATH}/1.ProteinAttenuation.csv.gz")
patt_low = list(patt.query("cluster == 'Low'")["gene"])
patt_high = list(patt.query("cluster == 'High'")["gene"])


# GI list
#

sl_lm = pd.read_csv(f"{RPATH}/lm_sklearn_protein_crispr.csv.gz").dropna()
gi_list = sl_lm.query("fdr < .1").query("n > 50")

# CORUM
gi_list["corum"] = [
    int((p1, p2) in corum_db.db_melt_symbol) for p1, p2 in gi_list[["y", "x"]].values
]

# BioGRID
gi_list["biogrid"] = [
    int((p1, p2) in biogrid_db.biogrid) for p1, p2 in gi_list[["y", "x"]].values
]

# HuRI
gi_list["huri"] = [
    int((p1, p2) in huri_db.huri) for p1, p2 in gi_list[["y", "x"]].values
]

# String distance
ppi = PPI().build_string_ppi(score_thres=900)
gi_list = PPI.ppi_annotation(gi_list, ppi, x_var="x", y_var="y", ppi_var="string_dist")
gi_list = gi_list.assign(string=(gi_list["string_dist"] == "1").astype(int))

# Previously reported SL
known_sl = pd.read_excel(f"{DPATH}/sang_lee_nature_comms_sl_network.xlsx")
known_sl = known_sl.query("SL == 1")
known_sl = {
    (p1, p2)
    for p in known_sl[["gene1", "gene2"]].values
    for p1, p2 in [(p[0], p[1]), (p[1], p[0])]
    if p1 != p2
}
gi_list["sl"] = [
    int((p1, p2) in known_sl) for p1, p2 in gi_list[["y", "x"]].values
]

# Attenuated protein
gi_list["attenuated"] = gi_list["y"].isin(patt_high).astype(int)

# Sort by LR p-value
gi_list = gi_list.sort_values("pval")
print(gi_list.head(60))


# Volcano
#

s_transform = MinMaxScaler(feature_range=[1, 10])

_, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

for t, df in gi_list.groupby("string_dist"):
    sc = ax.scatter(
        -np.log10(df["pval"]),
        df["b"],
        s=s_transform.fit_transform(df[["b"]].abs()),
        color=GIPlot.PPI_PAL[t],
        label=t,
        edgecolor="white",
        lw=0.1,
        alpha=0.5,
    )

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

legend1 = ax.legend(
    frameon=False,
    prop={"size": 4},
    title="PPI distance",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
legend1.get_title().set_fontsize("2")
ax.add_artist(legend1)

handles, labels = sc.legend_elements(
    prop="sizes",
    num=4,
    func=lambda x: s_transform.inverse_transform(np.array(x).reshape(-1, 1))[:, 0],
)
legend2 = (
    ax.legend(
        handles,
        labels,
        loc="upper right",
        title="Effect size (abs)",
        frameon=False,
        prop={"size": 2},
    )
        .get_title()
        .set_fontsize("2")
)

plt.ylabel("Effect size (beta)")
plt.xlabel("Association p-value (-log10)")

plt.savefig(
    f"{RPATH}/2.SL_volcano.png",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")


#
#

df_corr = pd.read_csv(f"{RPATH}/2.SLProteinInteractions.csv.gz")
novel_ppis = df_corr.query(f"(prot_fdr < .05) & (prot_corr > 0.5)")

plot_df = pd.concat([
    pd.Series(list(novel_ppis["protein1"]) + list(novel_ppis["protein2"])).value_counts().rename("ppis"),
    gi_list[gi_list["b"].abs() > .5]["y"].value_counts().rename("gis"),
], axis=1).dropna()

_, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

ax.scatter(plot_df["ppis"], plot_df["gis"], c=GIPlot.PAL_DBGD[2], s=5, linewidths=0)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel("Number of PPIs")
ax.set_ylabel(f"Number of GIs")

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")

cor, pval = spearmanr(plot_df["ppis"], plot_df["gis"])
annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
ax.text(
    0.95,
    0.05,
    annot_text,
    fontsize=4,
    transform=ax.transAxes,
    ha="right",
)

plt.savefig(
    f"{RPATH}/2.SL_ppis_gis_correlation.pdf",
    bbox_inches="tight",
)
plt.close("all")


#
#

gi_pairs = [
    ("ERBB2", "ERBB2"),
    ("SMARCA4", "SMARCA2"),
    ("RPL22L1", "WRN"),
    ("VPS4A", "VPS4B"),
    ("EMD", "LEMD2"),
    ("PRKAR1A", "PRKAR1A"),
]

# p, c = ("PCYT1A", "GUK1")
for p, c in gi_pairs:
    plot_df = pd.concat(
        [
            crispr.loc[[c]].T.add_suffix("_crispr"),
            prot.loc[[p]].T.add_suffix("_prot"),
            gexp.loc[[p]].T.add_suffix("_gexp"),
            ss["tissue"],
        ],
        axis=1,
        sort=False,
    ).dropna(subset=[f"{c}_crispr", f"{p}_prot"])

    ax = GIPlot.gi_tissue_plot(f"{p}_prot", f"{c}_crispr", plot_df)
    ax.set_xlabel(f"{p}\nProtein intensities")
    ax.set_ylabel(f"{c}\nCRISPR log FC")
    plt.savefig(
        f"{RPATH}/2.SL_{p}_{c}_regression_tissue_plot.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    ax = GIPlot.gi_tissue_plot(f"{p}_gexp", f"{c}_crispr", plot_df.dropna(subset=[f"{c}_crispr", f"{p}_gexp"]))
    ax.set_xlabel(f"{p}\nGene expression (RNA-Seq voom)")
    ax.set_ylabel(f"{c}\nCRISPR log FC")
    plt.savefig(
        f"{RPATH}/2.SL_{p}_{c}_regression_tissue_plot_gexp.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

