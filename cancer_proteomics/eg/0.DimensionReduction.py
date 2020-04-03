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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR, Sample


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")


def pc_labels(n):
    return [f"PC{i}" for i in np.arange(1, n + 1)]


def dim_reduction_pca(df, pca_ncomps=10):
    df_pca = PCA(n_components=pca_ncomps)

    df_pcs = df_pca.fit_transform(df.T)
    df_pcs = pd.DataFrame(df_pcs, index=df.T.index, columns=pc_labels(pca_ncomps))

    df_vexp = pd.Series(df_pca.explained_variance_ratio_, index=df_pcs.columns)

    return df_pcs, df_vexp


def dim_reduction(
    df,
    pca_ncomps=50,
    tsne_ncomps=2,
    perplexity=30.0,
    early_exaggeration=12.0,
    learning_rate=200.0,
    n_iter=1000,
):
    # PCA
    df_pca = dim_reduction_pca(df, pca_ncomps)[0]

    # tSNE
    df_tsne = TSNE(
        n_components=tsne_ncomps,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
    ).fit_transform(df_pca)
    df_tsne = pd.DataFrame(df_tsne, index=df_pca.index, columns=pc_labels(tsne_ncomps))

    return df_tsne, df_pca


def plot_dim_reduction(data, palette=None, ctype="tSNE"):
    if "model_type" not in data.columns:
        data = data.assign(tissue="All")

    if palette is None:
        palette = dict(All=CrispyPlot.PAL_DBGD[0])

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0), dpi=600)

    for t, df in data.groupby("model_type"):
        ax.scatter(
            df["PC1"],
            df["PC2"],
            c=palette[t],
            marker="o",
            edgecolor="",
            s=5,
            label=t,
            alpha=0.8,
        )
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.axis("off" if ctype == "tSNE" else "on")

    if ctype == "pca":
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        prop={"size": 4},
        frameon=False,
        title="Model type",
    ).get_title().set_fontsize("5")

    return ax


# Data-sets
#

prot, gexp = Proteomics(), GeneExpression()


# Samples
#

ss = prot.ss.copy()
samples = set.intersection(set(prot.get_data()), set(gexp.get_data()))
LOG.info(f"Samples: {len(samples)}")


# Filter data-sets
#

prot = prot.filter(subset=samples)
prot = prot.T.fillna(prot.T.mean()).T
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp.filter(subset=samples)
LOG.info(f"Transcriptomics: {gexp.shape}")


# Dimension reduction
#

prot_tsne, prot_pca = dim_reduction(prot)
gexp_tsne, gexp_pca = dim_reduction(gexp)

dimred = dict(
    tSNE=dict(proteomics=prot_tsne, transcriptomics=gexp_tsne),
    pca=dict(proteomics=prot_pca, transcriptomics=gexp_pca),
)

for ctype in dimred:
    for dtype in dimred[ctype]:
        plot_df = pd.concat(
            [dimred[ctype][dtype], ss["model_type"]], axis=1, sort=False
        ).dropna()

        ax = plot_dim_reduction(plot_df, ctype=ctype, palette=CrispyPlot.PAL_MODEL_TYPE)
        ax.set_title(f"{ctype} - {dtype}")
        plt.savefig(
            f"{RPATH}/0.Dimension_reduction_{dtype}_{ctype}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")
