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

import pylab
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import numpy.ma as ma
import itertools as it
import scipy.stats as stats
from scipy.stats import skew
import matplotlib.pyplot as plt
from natsort import natsorted
from limix.plot import qqplot
from crispy.GIPlot import GIPlot
from scipy.stats import pearsonr, spearmanr
from adjustText import adjust_text
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from multiomics_integration.notebooks import DataImport, PPI_PAL, PPI_ORDER


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

# PPIs
ppis = pd.read_csv(f"{TPATH}/PPInteractions.csv.gz")

# Hits
lm_drug = pd.read_csv(f"{TPATH}/lm_sklearn_degr_drug_annotated_diann_051021.csv.gz")
lm_crispr = pd.read_csv(f"{TPATH}/lm_sklearn_degr_crispr_annotated_diann_051021.csv.gz")

#
#
for dtype, lm_df in [
    ("Drug-Protein", lm_drug.query("n > 60")),
    # ("CRISPR-Protein", lm_crispr.query("n > 60")),
]:
    for pvar in ["nc_pval", "pval"]:
        # dtype, lm_df, pvar = "Drug-Protein", lm_drug, "nc_pval"
        # _, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

        plot_df = []
        for n in PPI_ORDER:
            df = lm_df.query(f"ppi == '{n}'")
            x_var, y_var, l_var = qqplot(
                df[pvar],
                label=n,
                # ax=ax,
                show_lambda=True,
                band_kws=dict(color=PPI_PAL[n], alpha=0.2),
                pts_kws=dict(color=PPI_PAL[n]),
            )

            plot_df.append(
                pd.DataFrame(
                    {"qnull": x_var, "qemp": y_var, "lambda": l_var, "target": n}
                )
            )

        plt.close("all")

        plot_df = pd.concat(plot_df)
        v_lambda = plot_df.groupby("target")["lambda"].first()

        _, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

        for t in CrispyPlot.PPI_ORDER:
            df = plot_df.query(f"target == '{t}'")
            ax.scatter(
                df["qnull"],
                df["qemp"],
                c=CrispyPlot.PPI_PAL[t],
                label=f"{t} ($\lambda$={v_lambda[t]:.2f})",
                s=4,
            )

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

        lims = [0, 8]
        ax.plot(lims, lims, "k-", lw=0.3, zorder=0)

        ax.legend(prop={"size": 5}, frameon=False, title="", loc=2)

        ax.set_xlabel("P-value expected (-log10)")
        ax.set_ylabel("P-value observed (-log10)")
        ax.set_title(f"{dtype} associations")

        plt.savefig(f"{RPATH}/LM_qqplot_{dtype}_{pvar}.pdf", bbox_inches="tight")
        plt.savefig(
            f"{RPATH}/LM_qqplot_{dtype}_{pvar}.png", bbox_inches="tight", dpi=600
        )
        plt.close("all")
