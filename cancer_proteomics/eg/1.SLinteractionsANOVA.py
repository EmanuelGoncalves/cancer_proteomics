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
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.GIPlot import GIPlot
from itertools import zip_longest
from cancer_proteomics.eg.LMModels import LMModels
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR


sys.path.extend(
    [
        "/Users/eg14/Projects/crispy",
        "/Users/eg14/Projects/crispy/crispy",
        "/Users/eg14/Projects/crispy/notebooks",
    ]
)

LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# Data-sets
#
prot, gexp, crispr = Proteomics(), GeneExpression(), CRISPR()


# Samples
#
samples = set.intersection(
    set(prot.get_data()), set(gexp.get_data()), set(crispr.get_data())
)
LOG.info(f"Samples: {len(samples)}")


# Filter data-sets
#
prot = prot.filter(subset=samples, perc_measures=0.2)
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp.filter(subset=samples)
LOG.info(f"Transcriptomics: {gexp.shape}")

crispr = crispr.filter(
    subset=samples,
    binarise_thres=-0.5,
    drop_core_essential=True,
    drop_core_essential_broad=True,
)
crispr = crispr[(crispr.sum(1) > 5)]
LOG.info(f"CRISPR: {crispr.shape}")


# Genes
#

genes = natsorted(list(set.intersection(set(prot.index), set(gexp.index))))
LOG.info(f"Genes: {len(genes)}")


# Covariates
#

m = LMModels.define_covariates(cancertype=False)


# Protein ~ CRISPR LMMs
#

lmm = LMModels(
    y=prot.loc[genes].T,
    x=crispr.T,
    m=m,
    transform_x="none",
)

lmm_assoc = lmm.matrix_lmm()

lmm_assoc.to_csv(
    f"{RPATH}/lmm_protein_crispr_anova.csv.gz",
    index=False,
    compression="gzip",
)


#
#

p, c = "APPL1", "CAPRIN1"

plot_df = pd.concat([
    prot.loc[p],
    crispr.loc[c],
], axis=1, sort=False).dropna()

ax = GIPlot.gi_classification(c, p, plot_df, palette=GIPlot.PAL_DTRACE)
plt.gcf().set_size_inches(0.7, 1.5)
plt.savefig(
    f"{RPATH}/1.ANOVAS_{p}_{c}.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")
