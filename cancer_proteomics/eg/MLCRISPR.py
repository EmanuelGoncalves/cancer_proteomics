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

import gseapy
import logging
import matplotlib
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
from crispy.Utils import Utils
from crispy.GIPlot import GIPlot
from Enrichment import Enrichment
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, pearsonr
from crispy.MOFA import MOFA, MOFAPlot
from crispy.CrispyPlot import CrispyPlot
from sklearn.mixture import GaussianMixture
from cancer_proteomics.eg.LMModels import LMModels
from cancer_proteomics.eg.SLinteractionsSklearn import LModel
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    Methylation,
    CopyNumber,
    CRISPR,
    DrugResponse,
    WES,
    Mobem,
    Sample,
    Metabolomics,
)


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")
MPATH = pkg_resources.resource_filename("tables", "/")


def sample_corr(var1, var2, idx_set=None, method="pearson"):
    if idx_set is None:
        idx_set = set(var1.dropna().index).intersection(var2.dropna().index)

    else:
        idx_set = set(var1.reindex(idx_set).dropna().index).intersection(
            var2.reindex(idx_set).dropna().index
        )

    if method == "spearman":
        r, p = spearmanr(
            var1.reindex(index=idx_set), var2.reindex(index=idx_set), nan_policy="omit"
        )
    else:
        r, p = pearsonr(var1.reindex(index=idx_set), var2.reindex(index=idx_set))

    return r, p, len(idx_set)


# Data-sets
#

prot_obj = Proteomics()


# ML scores
#

ml_files = dict(
    prot="scores_202005271320_ruv_min.csv",
    mofa="scores_202005291155_mofa.csv",
    rna="scores_202005302251_rna.csv",
)
ml_dfs = {f: pd.read_csv(f"{MPATH}/{ml_files[f]}", index_col=0) for f in ml_files}
ml_scores = pd.DataFrame({f: ml_dfs[f]["val_score"] for f in ml_dfs})

ml_fimpor = dict(
    prot=pd.read_csv(
        f"{MPATH}/feature_importance_202005271320_ruv_min.csv.gz", index_col=0
    )
)


# Pairgrid

grid = sns.PairGrid(ml_scores, height=1.1, despine=False)


def triu_plot_hex(x, y, color, label, **kwargs):
    plt.hexbin(
        x, y, cmap="Spectral_r", gridsize=100, mincnt=1, bins="log", lw=0, alpha=1
    )

    lims = [ml_scores.min().min(), ml_scores.max().max()]
    plt.plot(lims, lims, ls=":", lw=0.1, c="#484848", zorder=0)
    plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")
    plt.xlim(lims)
    plt.ylim(lims)


grid.map_upper(triu_plot_hex)


def diag_plot(x, color, label, **kwargs):
    sns.distplot(x, label=label, color=GIPlot.PAL_DBGD[0])


grid.map_diag(diag_plot, kde=True, hist_kws=dict(linewidth=0), bins=30)

grid.fig.subplots_adjust(wspace=0.05, hspace=0.05)
plt.gcf().set_size_inches(3, 3)
plt.savefig(f"{RPATH}/ml_scores_pairgrid.pdf", bbox_inches="tight")
plt.close("all")


# Linear regressions
#

sl_lm = pd.read_csv(f"{RPATH}/lm_sklearn_degr_crispr.csv.gz")

lm_fimpor = dict(
    prot=pd.pivot_table(sl_lm, index="x", columns="y", values="b", fill_value=np.nan)
)


#
#

nassoc_df = pd.concat(
    [ml_scores, sl_lm.query("fdr < .1")["y"].value_counts().rename("nassoc")], axis=1
).fillna(0)

for y_var in ml_scores:
    g = GIPlot.gi_regression("nassoc", y_var, nassoc_df, lowess=True)
    g.set_axis_labels("Number of associations (LM)", f"Score (ML {y_var})")

    plt.savefig(f"{RPATH}/ml_nassoc_{y_var}.pdf", bbox_inches="tight")
    plt.close("all")


#
#

for dtype in ml_fimpor:
    # Get ML features weights
    dtype_df = ml_fimpor[dtype]
    dtype_df_lm = lm_fimpor[dtype]

    # Correlate feature weights
    fimpor_corr = pd.Series(
        {
            p: sample_corr(dtype_df[p], dtype_df_lm[p].abs())[0]
            for p in set(dtype_df_lm).intersection(dtype_df)
        }
    )

    # Plot
    nassoc_df_fimpo = pd.concat([nassoc_df, fimpor_corr.rename("feature")], axis=1)

    g = GIPlot.gi_continuous_plot(
        "nassoc",
        dtype,
        "feature",
        nassoc_df_fimpo,
        lowess=True,
        cbar_label="Feature importance correlation",
    )
    g.set_xlabel("Number of associations (LM)")
    g.set_ylabel(f"Score (ML {dtype})")

    plt.savefig(f"{RPATH}/ml_nassoc_{dtype}_fimpo.pdf", bbox_inches="tight")
    plt.close("all")
