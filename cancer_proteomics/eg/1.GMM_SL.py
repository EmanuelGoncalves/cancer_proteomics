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

import sys
import os
import logging
import argparse
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.GIPlot import GIPlot
from natsort import natsorted
from itertools import zip_longest
from sklearn.mixture import GaussianMixture
from cancer_proteomics.eg.LMModels import LMModels
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR, Sample, Mobem


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

crispr = crispr.filter(subset=samples, abs_thres=0.5, min_events=5)
LOG.info(f"CRISPR: {crispr.shape}")

# Genes
#
genes = natsorted(list(set.intersection(set(prot.index), set(gexp.index))))
LOG.info(f"Genes: {len(genes)}")


#
#

lmm_prot = pd.read_csv(f"{RPATH}/lmm_protein_crispr_fillna.csv.gz")


#
#
gmm_gi_pairs = []
# y_id, x_id = "WRN", "RPL22L1"
for y_id in ["HOXD13", "KRAS", "WRN", "ERBB2", "SMARCA2", "TP53", "MCL1"]:
    LOG.info(f"y={y_id}")

    for x_id in genes:
        df = pd.concat(
            [
                crispr.loc[[y_id]].T.add_suffix("_crispr"),
                prot.loc[[x_id]].T.add_suffix("_prot"),
                gexp.loc[[x_id]].T.add_suffix("_gexp"),
            ],
            axis=1,
            sort=False,
        ).dropna()

        if df.shape[0] < 10:
            continue

        gmm_pair = dict(y_id=y_id, x_id=x_id, nsamples=df.shape[0])
        for dtype in ["prot", "gexp"]:
            df_dtype = df[[f"{y_id}_crispr", f"{x_id}_{dtype}"]]

            gi_gmm = GaussianMixture(n_components=2).fit(df_dtype)

            gmm_pair[f"{dtype}_ydelta"] = np.subtract(*np.sort(gi_gmm.means_[:, 0]))
            gmm_pair[f"{dtype}_xdelta"] = np.subtract(*np.sort(gi_gmm.means_[:, 1]))
            print(gmm_pair[f"{dtype}_ydelta"], gmm_pair[f"{dtype}_xdelta"])

            gi_gmm_pred = gi_gmm.predict(df_dtype)
            gmm_pair[f"{dtype}_ncluster"] = np.min([np.sum(1 - gi_gmm_pred), np.sum(gi_gmm_pred)])

            gmm_pair[f"{dtype}_converged"] = gi_gmm.converged_

        gmm_gi_pairs.append(gmm_pair)

gmm_gi_pairs = pd.DataFrame(gmm_gi_pairs)


#
#

self = LMModels(y=prot.loc[["RPL22L1"], :].T, x=crispr.T)


#
#

mobem = Mobem().filter()

y_id, x_id = "RPL22L1", "RPL36"

plot_df = pd.concat(
    [
        prot.loc[[y_id]].T.add_suffix("_prot"),
        crispr.loc[[x_id]].T.add_suffix("_crispr"),
        mobem.loc["TP53_mut"],
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

grid = GIPlot.gi_regression(f"{x_id}_crispr", f"{y_id}_prot", plot_df)
plt.savefig(
    f"{RPATH}/1.GMM_SL_{y_id}_{x_id}_regression.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")

ax = GIPlot.gi_tissue_plot(f"{x_id}_crispr", f"{y_id}_prot", plot_df)
plt.savefig(
    f"{RPATH}/1.GMM_SL_{y_id}_{x_id}_regression_tissue_plot.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

grid = GIPlot.gi_regression_marginal(f"{x_id}_crispr", f"{y_id}_prot", "TP53_mut", plot_df)
plt.savefig(
    f"{RPATH}/1.GMM_SL_{y_id}_{x_id}_TP53_mut_regression.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")
