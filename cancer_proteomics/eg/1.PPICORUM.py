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
import matplotlib.patches as mpatches
from sklearn import svm
from natsort import natsorted
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegressionCV
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from crispy.GIPlot import GIPlot
from scipy.stats import mannwhitneyu
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from sklearn.mixture import GaussianMixture
from crispy.Utils import Utils
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    CopyNumber,
    Methylation,
    Sample,
    CORUM,
)


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# SWATH proteomics
#

prot = Proteomics().filter(perc_measures=None)
LOG.info(f"Proteomics: {prot.shape}")


# Overlaps
#

ss = Sample().samplesheet

samples = list(set.intersection(set(prot)))
genes = list(set.intersection(set(prot.index)))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")


# Cancer types
#

cancer_types_thres = 15

cancer_types = (
    ss.reindex(samples).reset_index().groupby("cancer_type")["model_id"].agg(set)
)
cancer_types = pd.Series(
    {t: s for t, s in cancer_types.iteritems() if len(s) >= cancer_types_thres}
)
cancer_types["Pancancer"] = set(samples)


# CORUM
#

corum_db = CORUM(protein_subset=set(genes))

corum = dict()
for (p1, p2), i in corum_db.db_melt_symbol.items():
    if (p2, p1) not in corum:
        corum[(p1, p2)] = i


# Correlations
#

corum_corrs = []
corum_corrs_thres = 5

# p1, p2 = "RPL10", "ILF3"
for (p1, p2), i in corum.items():
    LOG.info(f"P1:{p1}; P2:{p2}")

    # ct = "Kidney Carcinoma"
    for ct in cancer_types.index:
        df_ct = prot.reindex(index=[p1, p2], columns=cancer_types[ct]).T.dropna()

        if len(df_ct) <= corum_corrs_thres:
            continue

        r, p = spearmanr(df_ct[p1], df_ct[p2], nan_policy="omit")

        ct_res = dict(
            p1=p1,
            p2=p2,
            ctype=ct,
            spearmanr=r,
            pval=p,
            len=df_ct.shape[0],
            complex_id=i,
            complex=corum_db.db_name[i],
        )

        corum_corrs.append(ct_res)

corum_corrs = pd.DataFrame(corum_corrs).sort_values("spearmanr", ascending=False)
corum_corrs.to_csv(
    f"{RPATH}/1.CORUM_pairs_spearman.csv.gz", compression="gzip", index=False
)


#
#

plot_df = pd.pivot_table(
    corum_corrs.query(f"len >= {cancer_types_thres}"),
    index=["p1", "p2"],
    columns="ctype",
    values="spearmanr",
)

fig = sns.clustermap(
    plot_df.fillna(plot_df.mean()).T,
    mask=plot_df.isna().T,
    cmap="Spectral",
    center=0,
    xticklabels=False,
    figsize=(12, 6),
)

fig.ax_heatmap.set_xlabel("Cancer types")
fig.ax_heatmap.set_ylabel(f"Protein-protein interactions")

plt.savefig(
    f"{RPATH}/1.CORUM_spearman_clustermap.png",
    dpi=600,
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")


#
#

plot_df = pd.pivot_table(
    corum_corrs.query(f"len >= {cancer_types_thres}"),
    index="complex",
    columns="ctype",
    values="spearmanr",
    aggfunc=np.mean
)

fig = sns.clustermap(
    plot_df.fillna(plot_df.mean()).T,
    mask=plot_df.isna().T,
    cmap="Spectral",
    center=0,
    xticklabels=False,
    figsize=(12, 6),
)

fig.ax_heatmap.set_xlabel("Cancer types")
fig.ax_heatmap.set_ylabel(f"Protein-protein interactions")

plt.savefig(
    f"{RPATH}/1.CORUM_spearman_clustermap_complexes.png",
    dpi=600,
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")


#
#

most_var_ppi_thres = 500
most_var_ppi = corum_corrs.query(f"len >= {cancer_types_thres}").groupby(["p1", "p2"])["spearmanr"].std().sort_values(ascending=False)
most_var_ppi = set(most_var_ppi.head(most_var_ppi_thres).index)

plot_df = corum_corrs[[(p1, p2) in most_var_ppi for p1, p2 in corum_corrs[["p1", "p2"]].values]]
plot_df = pd.pivot_table(
    plot_df,
    index=["p1", "p2"],
    columns="ctype",
    values="spearmanr",
)

fig = sns.clustermap(
    plot_df.fillna(plot_df.mean()).T,
    mask=plot_df.isna().T,
    cmap="Spectral",
    center=0,
    xticklabels=False,
    figsize=(12, 6),
)

fig.ax_heatmap.set_xlabel("Cancer types")
fig.ax_heatmap.set_ylabel(f"Protein-protein interactions")

plt.savefig(
    f"{RPATH}/1.CORUM_spearman_clustermap_most_var_{most_var_ppi_thres}.png",
    dpi=600,
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")
