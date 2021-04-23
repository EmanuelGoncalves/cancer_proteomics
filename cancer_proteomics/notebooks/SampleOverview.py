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
from adjustText import adjust_text
from crispy.MOFA import MOFA, MOFAPlot
from sklearn.metrics.ranking import auc
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import pearsonr, spearmanr
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cancer_proteomics.notebooks import DataImport, two_vars_correlation, PALETTE_TTYPE
from crispy.DataImporter import CORUM, BioGRID, PPI, HuRI


LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/")


# ### Imports

# Read samplesheet
ss = DataImport.read_samplesheet()

plot_df = ss.sort_values(["tissue", "replicates_correlation"])
plot_df.tissue = plot_df.tissue.astype("category")
plot_df.tissue.cat.set_categories(ss["tissue"].value_counts().index, inplace=True)
plot_df = plot_df.sort_values(
    ["tissue", "replicates_correlation"], ascending=[True, False]
).reset_index()

theta = np.linspace(0.0, 2 * np.pi, plot_df.shape[0], endpoint=False)
width = (2 * np.pi) / plot_df.shape[0]

fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600, subplot_kw=dict(polar=True))

for t, df_t in plot_df.groupby("tissue"):
    ax.bar(
        theta[df_t.index],
        df_t["replicates_correlation"],
        width=width,
        fc=PALETTE_TTYPE[t],
        linewidth=.0005,
        label=f"{t} (N={df_t.shape[0]})"
    )

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticklabels([])
ax.set_rlabel_position(270)
ax.spines["polar"].set_visible(False)

ax.grid(True, ls="--", lw=0.1, alpha=1.0, zorder=3, c="black", axis="y")

ax.grid(True, ls="-", lw=0.0, alpha=1.0, zorder=3, c="black", axis="x")

ax.set_ylabel("Mean replicate correlation\n(Pearson's R)", fontsize=6)
ax.yaxis.set_zorder(10)

ax.set_title(f"Cancer cell lines proteomics (N={plot_df.shape[0]})")

plt.legend(frameon=False, prop={"size": 4}, loc="center left", bbox_to_anchor=(1, 0.5))

plt.savefig(f"{RPATH}/SampleOverview_replicates_radialplot.pdf", bbox_inches="tight")
plt.savefig(
    f"{RPATH}/SampleOverview_replicates_radialplot.png", bbox_inches="tight", dpi=600
)
plt.close("all")