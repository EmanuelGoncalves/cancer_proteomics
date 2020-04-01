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
from crispy.GIPlot import GIPlot
from scipy.cluster import hierarchy
from crispy.CrispyPlot import CrispyPlot
from sklearn.preprocessing import quantile_transform, scale
from scipy.stats import pearsonr, spearmanr, gmean, hmean
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR, Sample


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")

# Cancer Driver Genes
#

cgenes = pd.read_csv(f"{DPATH}/cancer_genes_latest.csv.gz")


# Protein abundance attenuation
#

p_attenuated = pd.read_csv(f"{DPATH}/protein_attenuation_table.csv", index_col=0)


# SWATH proteomics
#

prot_obj = Proteomics()

prot = prot_obj.filter()

pep_raw = pd.read_csv(f"{DPATH}/proteomics/E0022_P02-P03_peptide_matrix_raw.csv")
pep_raw_qn = pd.DataFrame(
    quantile_transform(pep_raw.T, output_distribution="normal", copy=True).T,
    index=pep_raw.index,
    columns=pep_raw.columns,
)
# pep_norm = pd.read_csv(
#     f"{DPATH}/proteomics/E0022_P02-P03_peptide_matrix_normalised.txt",
#     sep="\t",
#     index_col=0,
# )


# Gene expression
#

gexp = GeneExpression().filter(
    subset=list(set(prot_obj.manifest.reindex(pep_raw.index)["SIDM"].dropna()))
)
LOG.info(f"Transcriptomics: {gexp.shape}")


# Overlap
#

samples = list(gexp)
LOG.info(f"Samples: {len(samples)}")


#
#

# protein = "MET"
for protein in ["MET", "EGFR"]:
    # Peptide matrices
    m_pep = pep_raw.loc[:, [i.startswith(f"{protein}_") for i in pep_raw.columns]]
    m_pep = m_pep.groupby(prot_obj.manifest["SIDM"]).median()

    m_pep_qn = pep_raw_qn.loc[:, [i.startswith(f"{protein}_") for i in pep_raw_qn.columns]]
    m_pep_qn = m_pep_qn.groupby(prot_obj.manifest["SIDM"]).median()

    m_pep_dict = dict(raw=dict(matrix=m_pep), qn=dict(matrix=m_pep_qn))

    # Clustering
    for k in m_pep_dict:
        data = m_pep_dict[k]["matrix"].dropna(how="all", axis=1)
        corr = spearmanr(data, nan_policy="omit").correlation
        corr_linkage = hierarchy.ward(corr)
        m_pep_dict[k]["linkage"] = corr_linkage

        hierarchy.dendrogram(
            corr_linkage, labels=[i.split("=")[-1] for i in data.columns], leaf_rotation=90
        )
        plt.title(k)
        plt.savefig(
            f"{RPATH}/0.QualityCheck_{protein}_{k}_dendogram.pdf", bbox_inches="tight"
        )
        plt.close("all")

    # Pick clusters
    for k in m_pep_dict:
        cluster_ids = hierarchy.fcluster(
            m_pep_dict[k]["linkage"],
            0.6 * max(m_pep_dict[k]["linkage"][:, 2]),
            criterion="distance",
        )
        m_pep_dict[k]["clusters"] = pd.Series(cluster_ids, index=data.columns)

    # Subset peptide matrix
    for k in m_pep_dict:
        k_clusters = m_pep_dict[k]["clusters"]
        selected_peptides = k_clusters[k_clusters.isin(k_clusters.value_counts().index[:1])]
        m_pep_dict[k]["matrix_top"] = m_pep_dict[k]["matrix"][selected_peptides.index]

    # Clustermap
    #
    for k in m_pep_dict:
        sns.clustermap(m_pep_dict[k]["matrix"].dropna(how="all", axis=1).corr())
        plt.savefig(
            f"{RPATH}/0.QualityCheck_{protein}_{k}_clustermap.pdf", bbox_inches="tight"
        )
        plt.close("all")

    #
    plot_df = pd.DataFrame(
        [
            m_pep_dict[k][m].median(1).rename(f"{k} {m.replace('matrix', '').replace('_', '').replace('top', 'filtered')}")
            for k in m_pep_dict
            for m in ["matrix", "matrix_top"]
        ]
    ).T
    plot_df = pd.concat(
        [
            plot_df,
            prot.loc[protein, samples].rename("protein"),
            gexp.loc[protein, samples].rename("transcript"),
            prot_obj.broad.reindex(columns=samples).loc[protein].rename("broad"),
            prot_obj.coread.reindex(columns=samples).loc[protein].rename("coread"),
            prot_obj.hgsc.reindex(columns=samples).loc[protein].rename("hgsc"),
        ],
        axis=1,
    )
    plot_df = plot_df.loc[:, plot_df.count() > 20]

    grid = sns.PairGrid(plot_df, height=1.1, despine=False, diag_sharey=False)

    for i, j in zip(*np.tril_indices_from(grid.axes, -1)):
        ax = grid.axes[i, j]
        r, p = spearmanr(plot_df.iloc[:, [i]], plot_df.iloc[:, [j]], nan_policy="omit")
        ax.annotate(
            f"Spearman's R={r:.2f}\np={p:.1e}" if p != 0 else f"R={r:.2f}\np<0.0001",
            xy=(0.5, 0.5),
            xycoords=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    grid.map_diag(CrispyPlot.diag_plot, kde=True, hist_kws=dict(linewidth=0), bins=30)
    grid.map_upper(CrispyPlot.triu_scatter_plot, marker="o", edgecolor="", cmap="Spectral_r", s=2)

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.gcf().set_size_inches(1.5 * plot_df.shape[1], 1.5 * plot_df.shape[1])

    plt.savefig(f"{RPATH}/0.QualityCheck_{protein}.pdf", bbox_inches="tight")
    plt.close("all")

    #
    _, axs = plt.subplots(1, m_pep.shape[1], figsize=(1.5 * m_pep.shape[1], 1.5), sharey="all", sharex="all", dpi=600)

    for i, p in enumerate(m_pep):
        ax = axs[i]

        df = pd.concat(
            [
                m_pep[p],
                gexp.loc[protein, samples].rename("transcript"),
                pep_raw[p].groupby(prot_obj.manifest["SIDM"]).count().rename("count"),
            ], axis=1
        ).dropna()

        ax.scatter(
            df[p],
            df["transcript"],
            marker="o",
            edgecolor="",
            s=5,
            alpha=0.8,
            c=GIPlot.PAL_DTRACE[2],
        )

        ax.set_title(f"{p.split('=')[-1]}")
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        if i == 0:
            ax.set_ylabel("transcript")

        ax.set_xlabel("Peptide\nraw intensities")

    plt.suptitle(protein, y=1.1)
    plt.subplots_adjust(hspace=0, wspace=0.05)
    plt.savefig(f"{RPATH}/0.QualityCheck_{protein}_pep_transcript_corrplot.pdf", bbox_inches="tight")
    plt.close("all")

