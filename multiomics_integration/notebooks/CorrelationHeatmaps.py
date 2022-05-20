#!/usr/bin/env python
# Copyright (C) 2021 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from multiomics_integration.notebooks import DataImport


LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/DIANN/")


if __name__ == "__main__":
    # Read proteomics (Proteins x Cell lines)
    prot = DataImport.read_protein_matrix(map_protein=True)

    #
    prot_corr = prot.T.corr()

    #
    R2_THRES = 0.3
    SKEW_THRES = -1
    FDR_THRES = 0.1

    lm_drug = pd.read_csv(f"{TPATH}/lm_sklearn_degr_drug_annotated_DIANN.csv.gz").query(f"r2 > {R2_THRES}")
    lm_drug_matrix = pd.pivot_table(data=lm_drug, index="x_id", columns="y_id", values="nc_beta")

    lm_crispr = pd.read_csv(f"{TPATH}/lm_sklearn_degr_crispr_annotated_DIANN.csv.gz").query(f"r2 > {R2_THRES}")
    lm_crispr_matrix = pd.pivot_table(data=lm_crispr, index="x_id", columns="y_id", values="nc_beta")

    #
    g = sns.clustermap(
        prot_corr.fillna(0),
        cmap="RdYlGn",
        center=0,
        xticklabels=False,
        yticklabels=False,
        linewidths=0.0,
        cbar_kws={"shrink": 0.5},
        figsize=(4, 4),
    )

    mask = np.tril(np.ones_like(prot_corr))
    values = g.ax_heatmap.collections[0].get_array().reshape(prot_corr.shape)
    new_values = np.ma.array(values, mask=mask)
    g.ax_heatmap.collections[0].set_array(new_values)

    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

    plt.savefig(f"{RPATH}/CorrelationHeatmap_ppi_correlation.png", bbox_inches="tight", dpi=300)
    plt.close("all")

    #
    g_drug = sns.clustermap(
        lm_drug_matrix.reindex(index=prot_corr.index[g.dendrogram_row.reordered_ind]).fillna(0),
        mask=lm_drug_matrix.reindex(index=prot_corr.index[g.dendrogram_row.reordered_ind]).isna(),
        cmap="RdYlGn",
        center=0,
        # vmax=1, vmin=-1,
        robust=True,
        xticklabels=False,
        yticklabels=False,
        row_cluster=False,
        linewidths=0.0,
        cbar_kws={"shrink": 0.5},
        figsize=(4, 4),
    )

    g_drug.ax_heatmap.set_xlabel("")
    g_drug.ax_heatmap.set_ylabel("")

    g_drug.ax_row_dendrogram.set_visible(False)
    g_drug.ax_col_dendrogram.set_visible(False)

    plt.savefig(f"{RPATH}/CorrelationHeatmap_drug.png", bbox_inches="tight", dpi=300)
    plt.close("all")

    #
    g_crispr = sns.clustermap(
        lm_crispr_matrix.reindex(index=prot_corr.index[g.dendrogram_row.reordered_ind]).fillna(0),
        mask=lm_crispr_matrix.reindex(index=prot_corr.index[g.dendrogram_row.reordered_ind]).isna(),
        cmap="RdYlGn",
        center=0,
        # vmax=1, vmin=-1,
        robust=True,
        xticklabels=False,
        yticklabels=False,
        row_cluster=False,
        linewidths=0.0,
        cbar_kws={"shrink": 0.5},
        figsize=(4, 4),
    )

    g_crispr.ax_heatmap.set_xlabel("")
    g_crispr.ax_heatmap.set_ylabel("")

    g_crispr.ax_row_dendrogram.set_visible(False)
    g_crispr.ax_col_dendrogram.set_visible(False)

    plt.savefig(f"{RPATH}/CorrelationHeatmap_crispr.png", bbox_inches="tight", dpi=300)
    plt.close("all")

    #
    df_drug = lm_drug_matrix.reindex(
        index=prot_corr.index[g.dendrogram_row.reordered_ind],
        columns=lm_drug_matrix.columns[g_drug.dendrogram_col.reordered_ind],
    )

    df_crispr = lm_crispr_matrix.reindex(
        index=prot_corr.index[g.dendrogram_row.reordered_ind],
        columns=lm_crispr_matrix.columns[g_crispr.dendrogram_col.reordered_ind],
    )

    df_prot = prot_corr.reindex(
        index=prot_corr.index[g.dendrogram_row.reordered_ind],
        columns=prot_corr.index[g.dendrogram_row.reordered_ind],
    )

    f, axs = plt.subplots(
        2,
        3,
        sharex="none",
        sharey="none",
        figsize=(9, 3),
        gridspec_kw={"height_ratios": (.9, .05), "hspace": .05, "wspace": 0.05}
    )

    for i, df in enumerate([df_drug, df_crispr]):
        sns.heatmap(
            df.fillna(0),
            mask=df.isna(),
            cmap=sns.diverging_palette(240, 10, as_cmap=True, sep=20),
            center=0,
            # vmax=1, vmin=-1,
            robust=True,
            xticklabels=False,
            yticklabels=False,
            linewidths=0.0,
            cbar_ax=axs[1, i],
            cbar_kws={"orientation": "horizontal", "shrink": .5},
            ax=axs[0, i],
        )

        axs[0, i].set_xlabel("")
        axs[0, i].set_ylabel("")

        axs[1, i].set_xlabel("Linear regression beta")

    sns.heatmap(
        df_prot,
        mask=np.tril(np.ones_like(df_prot, dtype=bool)),
        cmap="RdYlGn",
        center=0,
        # vmax=1, vmin=-1,
        robust=True,
        xticklabels=False,
        yticklabels=False,
        linewidths=0.0,
        cbar_ax=axs[1, 2],
        cbar_kws={"orientation": "horizontal", "shrink": .5},
        ax=axs[0, 2],
    )

    axs[0, 2].set_xlabel("")
    axs[0, 2].set_ylabel("")

    axs[1, 2].set_xlabel("Pearson's r")

    #
    axs[0, 0].set_ylabel("Proteins")

    axs[0, 0].set_title("Drugs")
    axs[0, 1].set_title("CRISPR-Cas9")
    axs[0, 2].set_title("Proteins")

    #
    plt.subplots_adjust(hspace=0.05)

    plt.savefig(f"{RPATH}/CorrelationHeatmap.png", bbox_inches="tight", dpi=300)
    plt.close("all")
