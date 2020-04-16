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
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.GIPlot import GIPlot
from itertools import zip_longest
from cancer_proteomics.eg.LMModels import LMModels
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR, CORUM, Sample, DPATH


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


# Data-sets
#
prot, gexp, crispr = Proteomics(), GeneExpression(), CRISPR()


# Samples
#
ss = Sample().samplesheet

samples = set.intersection(
    set(prot.get_data()), set(gexp.get_data()), set(crispr.get_data())
)
LOG.info(f"Samples: {len(samples)}")

# Filter data-sets
#
prot = prot.filter(subset=samples)
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp.filter(subset=samples)
LOG.info(f"Transcriptomics: {gexp.shape}")

crispr = crispr.filter(subset=samples, dtype="merged")
LOG.info(f"CRISPR: {crispr.shape}")

cgenes = pd.read_csv(f"{DPATH}/cancer_genes_latest.csv.gz")


# Associations
#

lmm_prot = pd.read_csv(f"{RPATH}/lmm_protein_crispr.csv.gz")
lmm_gexp = pd.read_csv(f"{RPATH}/lmm_gexp_crispr.csv.gz")


# Annotate GI list
#

# Significant associations
gi_list = lmm_prot.query("fdr < .1")
gi_list = gi_list.query("nsamples > 50").sort_values("pval")

# Interactions in protein complexes
corum_db = CORUM()
gi_list["corum"] = [
    int((p1, p2) in corum_db.db_melt_symbol)
    for p1, p2 in gi_list[["y_id", "x_id"]].values
]

# Significant association transcript
gexp_signif = {(y, x) for y, x in lmm_gexp.query("fdr < .1")[["y_id", "x_id"]].values}
gi_list["gexp_signif"] = [
    int((p1, p2) in gexp_signif) for p1, p2 in gi_list[["y_id", "x_id"]].values
]

# Number of measurements per protein
gi_list["nsamples"] = prot.reindex(gi_list["x_id"]).count(1).values
print(gi_list.head(60))


#
#

gi_pairs = [("ERBB2", "ERBB2"), ("SMARCA4", "SMARCA2"), ("RPL22L1", "WRN"), ("VPS4A", "VPS4B")]

# p, c = ("VPS4A", "VPS4B")
for p, c in gi_pairs:
    plot_df = pd.concat(
        [
            crispr.loc[[c]].T.add_suffix("_crispr"),
            prot.loc[[p]].T.add_suffix("_prot"),
            ss["tissue"],
        ],
        axis=1,
        sort=False,
    ).dropna()

    grid = GIPlot.gi_regression(f"{p}_prot", f"{c}_crispr", plot_df)
    grid.set_axis_labels(f"{p}\nProtein intensities", f"{c}\nCRISPR log FC")
    plt.savefig(
        f"{RPATH}/2.SL_{p}_{c}_regression.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")

    ax = GIPlot.gi_tissue_plot(f"{p}_prot", f"{c}_crispr", plot_df)
    ax.set_xlabel(f"{p}\nProtein intensities")
    ax.set_ylabel(f"{c}\nCRISPR log FC")
    plt.savefig(
        f"{RPATH}/2.SL_{p}_{c}_regression_tissue_plot.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
