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
import gseapy
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
from crispy.MOFA import MOFA, MOFAPlot
from sklearn.metrics.ranking import auc
from crispy.Enrichment import Enrichment
from scipy.stats import pearsonr, spearmanr
from cancer_proteomics.eg.LMModels import LMModels
from cancer_proteomics.eg.SLinteractionsSklearn import LModel
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
    Mobem,
)


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


# Data-sets
#

prot_obj = Proteomics()
gexp_obj = GeneExpression()
crispr_obj = CRISPR()
mobem_obj = Mobem()


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


# PPIs found
#

ppis = pd.read_csv(f"{RPATH}/2.SLProteinInteractions.csv.gz")
novel_ppis = ppis.query(f"(prot_fdr < .05) & (prot_corr > 0.5)")


# Significant drug ~ crispr
#

dcrispr = pd.read_excel(f"{DPATH}/Significant Drug CRISPR Dataset EV5.xlsx", sheet_name="Dataset EV5")


# MOFA factors
#

mofa_factors, mofa_weights, mofa_rsquare = MOFA.read_mofa_hdf5(f"{RPATH}/1.MultiOmics.hdf5")


# Subtypes
#

pam50 = pd.read_csv(f"{DPATH}/breast_pam50.csv").set_index("model_id")


# GI list
#

sl_lm = pd.read_csv(f"{RPATH}/lm_sklearn_protein_crispr.csv.gz").dropna()

# CORUM
sl_lm["corum"] = [
    int((p1, p2) in corum_db.db_melt_symbol) for p1, p2 in sl_lm[["y", "x"]].values
]

# BioGRID
sl_lm["biogrid"] = [
    int((p1, p2) in biogrid_db.biogrid) for p1, p2 in sl_lm[["y", "x"]].values
]

# HuRI
sl_lm["huri"] = [
    int((p1, p2) in huri_db.huri) for p1, p2 in sl_lm[["y", "x"]].values
]

# String distance
ppi = PPI().build_string_ppi(score_thres=900)
sl_lm = PPI.ppi_annotation(sl_lm, ppi, x_var="x", y_var="y", ppi_var="string_dist")
sl_lm = sl_lm.assign(string=(sl_lm["string_dist"] == "1").astype(int))

# Previously reported SL
known_sl = pd.read_excel(f"{DPATH}/sang_lee_nature_comms_sl_network.xlsx")
known_sl = known_sl.query("SL == 1")
known_sl = {
    (p1, p2)
    for p in known_sl[["gene1", "gene2"]].values
    for p1, p2 in [(p[0], p[1]), (p[1], p[0])]
    if p1 != p2
}
sl_lm["sl"] = [int((p1, p2) in known_sl) for p1, p2 in sl_lm[["y", "x"]].values]

# Manually curated
curated_sl = pd.read_csv(f"{DPATH}/sl_list_curated.tab", sep="\t")
curated_sl = {
    (p1, p2)
    for p in curated_sl[["gene1", "gene2"]].values
    for p1, p2 in [(p[0], p[1]), (p[1], p[0])]
    if p1 != p2
}
sl_lm["sl2"] = [int((p1, p2) in curated_sl) for p1, p2 in sl_lm[["y", "x"]].values]

# Paralogs
paralog_ds = pd.read_excel(
    f"{DPATH}/msb198871-sup-0004-datasetev1.xlsx", sheet_name="paralog annotations"
)
paralog_ds = {
    (p1, p2)
    for p in paralog_ds[["gene1 name", "gene2 name"]].values
    for p1, p2 in [(p[0], p[1]), (p[1], p[0])]
    if p1 != p2
}
sl_lm["paralogs"] = [int((p1, p2) in paralog_ds) for p1, p2 in sl_lm[["y", "x"]].values]

# Attenuated protein
sl_lm["attenuated"] = sl_lm["x"].isin(patt_high).astype(int)

# Sort by LR p-value
sl_lm = sl_lm.sort_values("pval")

# Top significant associations
gi_list = sl_lm.query("fdr < .1")
print(gi_list.head(60))


# Enrichment recall curves
#

dbs = ["corum", "biogrid", "string", "huri", "attenuated", "paralogs"]
dbs_pal = dict(
    biogrid=sns.color_palette("tab20c").as_hex()[0],
    corum=sns.color_palette("tab20c").as_hex()[4],
    string=sns.color_palette("tab20c").as_hex()[8],
    huri=sns.color_palette("tab20c").as_hex()[12],
    attenuated=sns.color_palette("tab20b").as_hex()[8],
    sl=sns.color_palette("tab20b").as_hex()[4],
    sl2=sns.color_palette("tab20b").as_hex()[12],
    paralogs=sns.color_palette("tab20b").as_hex()[0],
)

dbs_rc = dict()
for db in dbs:
    db_df = gi_list.reset_index(drop=True)[db]

    rc_df_y = np.cumsum(db_df) / np.sum(db_df)
    rc_df_x = np.array(db_df.index) / db_df.shape[0]
    rc_df_auc = auc(rc_df_x, rc_df_y)

    dbs_rc[db] = dict(x=list(rc_df_x), y=list(rc_df_y), auc=rc_df_auc)

# Recall curves
_, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

for db in dbs_rc:
    ax.plot(
        dbs_rc[db]["x"],
        dbs_rc[db]["y"],
        label=f"{db} (AUC={dbs_rc[db]['auc']:.2f})",
        c=dbs_pal[db],
    )

ax.plot([0, 1], [0, 1], "k--", lw=0.3)
ax.legend(loc="lower right", frameon=False)

ax.set_ylabel("Cumulative sum")
ax.set_xlabel("Ranked correlation")
ax.grid(True, axis="both", ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.savefig(f"{RPATH}/2.SL_roc_curves.pdf", bbox_inches="tight", transparent=True)
plt.close("all")

# Barplot
plot_df = pd.DataFrame([dict(ppi=db, auc=dbs_rc[db]["auc"]) for db in dbs_rc])

_, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

sns.barplot(
    "auc",
    "ppi",
    data=plot_df,
    orient="h",
    linewidth=0.0,
    saturation=1.0,
    palette=dbs_pal,
    ax=ax,
)

ax.set_ylabel("")
ax.set_xlabel("Recall curve AUC")
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

plt.savefig(f"{RPATH}/2.SL_roc_barplot.pdf", bbox_inches="tight", transparent=True)
plt.close("all")


# Volcano
#

s_transform = MinMaxScaler(feature_range=[1, 10])

_, ax = plt.subplots(1, 1, figsize=(4.5, 2.5), dpi=600)

for t, df in gi_list.groupby("string_dist"):
    sc = ax.scatter(
        -np.log10(df["pval"]),
        df["b"],
        s=s_transform.fit_transform(df[["n"]]),
        color=GIPlot.PPI_PAL[t],
        label=t,
        lw=0.0,
        alpha=0.5,
    )

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

legend1 = ax.legend(
    frameon=False,
    prop={"size": 4},
    title="PPI distance",
    loc="lower right",
)
legend1.get_title().set_fontsize("4")
ax.add_artist(legend1)

handles, labels = sc.legend_elements(
    prop="sizes",
    num=8,
    func=lambda x: s_transform.inverse_transform(np.array(x).reshape(-1, 1))[:, 0],
)
legend2 = (
    ax.legend(
        handles,
        labels,
        loc="upper right",
        title="# samples",
        frameon=False,
        prop={"size": 4},
    )
    .get_title()
    .set_fontsize("4")
)

ax.set_ylabel("Effect size (beta)")
ax.set_xlabel("Association p-value (-log10)")
ax.set_title("CRISPR ~ Protein associations")

plt.savefig(f"{RPATH}/2.SL_volcano.png", transparent=True, bbox_inches="tight")
plt.close("all")


#
#

df_corr = pd.read_csv(f"{RPATH}/2.SLProteinInteractions.csv.gz")
novel_ppis = df_corr.query(f"(prot_fdr < .05) & (prot_corr > 0.5)")

plot_df = pd.Series(
    list(novel_ppis["protein1"]) + list(novel_ppis["protein2"])
).value_counts()
plot_df = pd.concat(
    [
        plot_df.rename("ppis"),
        gi_list["x"].value_counts().rename("gis"),
    ],
    axis=1,
).dropna()

_, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

ax.scatter(plot_df["ppis"], plot_df["gis"], c=GIPlot.PAL_DBGD[2], s=5, linewidths=0)

ax.set_yscale("log")
ax.set_xscale("log")

ax.set_xlabel("Number of PPIs")
ax.set_ylabel(f"Number of GIs")

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")

cor, pval = spearmanr(plot_df["ppis"], plot_df["gis"])
annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
ax.text(0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

plt.savefig(f"{RPATH}/2.SL_ppis_gis_correlation.pdf", bbox_inches="tight")
plt.close("all")


#
#

gi_pairs = [
    ("ERBB2", "ERBB2"),
    ("SMARCA4", "SMARCA2"),
    ("RPL22L1", "WRN"),
    ("VPS4A", "VPS4B"),
    ("EMD", "LEMD2"),
    ("HNRNPH1", "HNRNPH1"),
    ("PRKAR1A", "PRKAR1A"),
    ("BSG", "FOXA1"),
]

# p, c = "CRTAP", "FOXA1"
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

    ax = GIPlot.gi_tissue_plot(
        f"{p}_gexp", f"{c}_crispr", plot_df.dropna(subset=[f"{c}_crispr", f"{p}_gexp"])
    )
    ax.set_xlabel(f"{p}\nGene expression (RNA-Seq voom)")
    ax.set_ylabel(f"{c}\nCRISPR log FC")
    plt.savefig(
        f"{RPATH}/2.SL_{p}_{c}_regression_tissue_plot_gexp.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


#
#

p, c = ("BSG", "FOXA1")

plot_df = pd.concat(
    [
        crispr.loc[[c]].T.add_suffix("_crispr"),
        prot.loc[[p]].T.add_suffix("_prot"),
        gexp.loc[[p]].T.add_suffix("_gexp"),
        prot_obj.broad.loc[[p, "SLC16A1"]].T.add_suffix("_broad"),
        ss["tissue"],
        pam50[["PAM50", "Jiang_et_al"]],
    ],
    axis=1,
    sort=False,
).dropna(subset=[f"{c}_crispr", f"{p}_prot"])

# Association with BROAD
for p_idx in [p, "SLC16A1"]:
    ax = GIPlot.gi_tissue_plot(f"{p_idx}_broad", f"{c}_crispr", plot_df.dropna(subset=[f"{p_idx}_broad"]))
    ax.set_xlabel(f"{p_idx}\nProtein intensities (CCLE)")
    ax.set_ylabel(f"{c}\nCRISPR log FC")
    plt.savefig(
        f"{RPATH}/2.SL_{p_idx}_CCLE_{c}_regression_tissue_plot.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

#
ax = GIPlot.gi_tissue_plot(f"{p}_prot", f"{c}_crispr", plot_df[plot_df["tissue"].isin(["Breast", "Prostate"])])
ax.set_xlabel(f"{p}\nProtein intensities")
ax.set_ylabel(f"{c}\nCRISPR log FC")
plt.savefig(
    f"{RPATH}/2.SL_{p}_{c}_regression_tissue_plot_selected.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

#
pam50s = ["PAM50", "Jiang_et_al"]
huetypes = set(plot_df[pam50s[0]].dropna())
palette = pd.Series(sns.color_palette("Set1", n_colors=len(huetypes)).as_hex(), index=huetypes)
df = plot_df.dropna(subset=pam50s)
for ptype in pam50s:
    GIPlot.gi_classification(f"{c}_crispr", ptype, df, palette=palette.to_dict(), orient="h", notch=False, order=huetypes)
    plt.savefig(
        f"{RPATH}/2.SL_PAM50_{ptype}_{c}_boxplot.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
