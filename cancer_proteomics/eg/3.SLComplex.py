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
import limix
import logging
import argparse
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.GIPlot import GIPlot
from itertools import zip_longest
from crispy.MOFA import MOFA, MOFAPlot
from cancer_proteomics.eg.LMModels import LMModels
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    Methylation,
    CRISPR,
    DrugResponse,
    CORUM,
    Sample,
    DPATH,
)


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


# Data-sets
#

prot_obj = Proteomics()
gexp_obj = GeneExpression()
crispr_obj = CRISPR()
methy_obj = Methylation()
drespo_obj = DrugResponse()


# Filter data-sets
#

ss = prot_obj.ss

methy = methy_obj.filter()
LOG.info(f"Methylation: {methy.shape}")

gexp = gexp_obj.filter()
LOG.info(f"Transcriptomics: {gexp.shape}")

prot = prot_obj.filter()
LOG.info(f"Proteomics: {prot.shape}")

crispr = crispr_obj.filter(dtype="merged")
LOG.info(f"CRISPR: {crispr.shape}")

drespo = drespo_obj.filter()
LOG.info(f"Drug response: {drespo.shape}")

samples = set.intersection(set(prot), set(gexp), set(methy))
LOG.info(f"Samples: {len(samples)}")


# MOFA factors
#

factors, weights, rquared = MOFA.read_mofa_hdf5(f"{RPATH}/1.MultiOmics.hdf5")


# Gene sets
#

genes = list(set.intersection(set(prot.index), set(gexp.index)))

ess_genes = set(crispr_obj.filter(dtype="merged", subset=samples, abs_thres=0.5, min_events=5).index)

cgenes = pd.read_csv(f"{DPATH}/cancer_genes_latest.csv.gz")
cgenes = list(set(cgenes["gene_symbol"]).intersection(ess_genes))

patt = pd.read_csv(f"{RPATH}/1.ProteinAttenuation.csv.gz")
patt_low = list(set(patt.query("cluster == 'Low'")["gene"]).intersection(genes))
patt_high = list(set(patt.query("cluster == 'High'")["gene"]).intersection(genes))


#
#

m = LMModels.define_covariates(
    institute=crispr_obj.merged_institute, cancertype=False, mburden=False, ploidy=False
).assign(intercept=1)

lmms = []
# gene_y, gene_x = "ERBB2", "ERBB2"
for gene_y in ["SMARCA2"]:
    LOG.info(f"GeneY={gene_y}")

    gene_lmm = []
    for gene_x in ["SMARCA4"]:
        gene_df = pd.concat(
            [
                crispr.loc[[gene_y]].T.add_prefix("crispr_"),
                prot.loc[[gene_x]].T.add_prefix("prot_"),
                gexp.loc[[gene_x]].T.add_prefix("gexp_"),
                m,
                factors,
            ],
            axis=1,
        ).dropna()

        for f in factors:
            gene_f_lmm = []
            for dtype in ["prot", "gexp"]:
                f_lmm = limix.qtl.scan(
                    G=gene_df[f]
                    .multiply(gene_df[f"{dtype}_{gene_x}"], axis=0)
                    .rename(f"{f}_{gene_x}"),
                    Y=gene_df[f"crispr_{gene_y}"],
                    M=gene_df.reindex(columns=[f"{dtype}_{gene_x}", f] + list(m)),
                    verbose=False,
                )

                beta, beta_se = (
                    f_lmm.effsizes["h2"]
                    .iloc[-1, :][["effsize", "effsize_se"]]
                    .values
                )

                gene_f_lmm.append(
                    f_lmm.stats.drop(columns=["dof20", "scale2"]).assign(beta=beta).add_prefix(f"{dtype}_")
                )

            gene_lmm.append(pd.concat(gene_f_lmm, axis=1).assign(feature=f, gene_y=gene_y, gene_x=gene_x))

    gene_lmm = pd.concat(gene_lmm, ignore_index=True)

    gene_lmm["prot_fdr"] = multipletests(gene_lmm["prot_pv20"], method="fdr_bh")[1]
    gene_lmm["gexp_fdr"] = multipletests(gene_lmm["gexp_pv20"], method="fdr_bh")[1]

    lmms.append(gene_lmm)

lmms = pd.concat(lmms, ignore_index=True).sort_values("prot_fdr")
