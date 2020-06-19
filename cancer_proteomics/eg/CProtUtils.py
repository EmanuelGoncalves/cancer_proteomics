#!/usr/bin/env python
# Copyright (C) 2020 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA, FactorAnalysis


def two_vars_correlation(var1, var2, idx_set=None, method="pearson"):
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

    return dict(corr=r, pval=p, len=len(idx_set))


class DimReduction:
    LOG = logging.getLogger("Crispy")

    @staticmethod
    def pc_labels(n):
        return [f"PC{i}" for i in np.arange(1, n + 1)]

    @classmethod
    def dim_reduction_pca(cls, df, pca_ncomps=10, is_factor_analysis=False):
        if is_factor_analysis:
            df_pca = FactorAnalysis(n_components=pca_ncomps).fit(df.T)

        else:
            df_pca = PCA(n_components=pca_ncomps).fit(df.T)

        df_pcs = pd.DataFrame(
            df_pca.transform(df.T), index=df.T.index, columns=cls.pc_labels(pca_ncomps)
        )

        df_loadings = pd.DataFrame(
            df_pca.components_, index=cls.pc_labels(pca_ncomps), columns=df.T.columns
        )

        if is_factor_analysis:
            df_vexp = None

        else:
            df_vexp = pd.Series(df_pca.explained_variance_ratio_, index=df_pcs.columns)

        return dict(pcs=df_pcs, vexp=df_vexp, loadings=df_loadings)

    @classmethod
    def dim_reduction(
        cls,
        df,
        pca_ncomps=50,
        tsne_ncomps=2,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate=200.0,
        n_iter=1000,
    ):
        miss_values = df.isnull().sum().sum()
        if miss_values > 0:
            cls.LOG.warning(
                f"DataFrame has {miss_values} missing values; impute with row mean"
            )
            df = df.T.fillna(df.T.mean()).T

        # PCA
        dimred_dict = cls.dim_reduction_pca(df, pca_ncomps)

        # tSNE
        df_tsne = TSNE(
            n_components=tsne_ncomps,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter=n_iter,
        ).fit_transform(dimred_dict["pcs"])

        dimred_dict["tsne"] = pd.DataFrame(
            df_tsne, index=dimred_dict["pcs"].index, columns=cls.pc_labels(tsne_ncomps)
        )

        return dimred_dict

    @staticmethod
    def plot_dim_reduction(
        data, x="PC1", y="PC2", hue_by=None, palette=None, ctype="tsne"
    ):
        if palette is None:
            palette = dict(All=CrispyPlot.PAL_DBGD[0])

        plot_df = pd.concat(
            [
                data["pcs" if ctype == "pca" else "tsne"][x],
                data["pcs" if ctype == "pca" else "tsne"][y],
            ],
            axis=1,
        )

        if hue_by is not None:
            plot_df = pd.concat([plot_df, hue_by.rename("hue_by")], axis=1).dropna()

        fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0), dpi=600)

        hue_by_df = (
            plot_df.groupby("hue_by") if hue_by is not None else [(None, plot_df)]
        )
        for t, df in hue_by_df:
            ax.scatter(
                df[x],
                df[y],
                c=CrispyPlot.PAL_DTRACE[2] if hue_by is None else palette[t],
                marker="o",
                edgecolor="",
                s=5,
                label=t,
                alpha=0.8,
            )

        ax.set_xlabel("" if ctype == "tsne" else f"{x} ({data['vexp'][x]*100:.1f}%)")
        ax.set_ylabel("" if ctype == "tsne" else f"{y} ({data['vexp'][y]*100:.1f}%)")
        ax.axis("off" if ctype == "tsne" else "on")

        if ctype == "pca":
            ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        if hue_by is not None:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                prop={"size": 4},
                frameon=False,
                title="Model type",
            ).get_title().set_fontsize("5")

        return ax
