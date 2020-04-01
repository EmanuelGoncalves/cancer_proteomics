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
from crispy.GIPlot import GIPlot
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import pearsonr, spearmanr
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, scale
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, GeneExpression, CopyNumber, Sample


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")


# SWATH proteomics
#

prot = Proteomics().filter(perc_measures=None)
LOG.info(f"Proteomics: {prot.shape}")


# Gene expression
#

gexp = GeneExpression().filter(subset=list(prot))
LOG.info(f"Transcriptomics: {gexp.shape}")


# Copy number
#

cnv = CopyNumber().filter(subset=list(gexp))
LOG.info(f"Copy number: {cnv.shape}")


# Overlaps
#

samples = list(set.intersection(set(prot), set(gexp), set(cnv)))
genes = list(set.intersection(set(prot.index), set(gexp.index), set(cnv.index)))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")

SS_THRES = 10
ss = Sample().samplesheet
ss_count = (
    ss.loc[samples]
    .reset_index()
    .groupby("tissue")["model_id"]
    .agg(["count", lambda v: set(v)])
)
ss_count.columns = ["count", "set"]
ss_count = ss_count.sort_values("count", ascending=False)


# Number of cell lines per tissue
#

plot_df = ss_count.reset_index()

_, ax = plt.subplots(1, 1, figsize=(1.5, 0.15 * plot_df.shape[0]), dpi=600)

sns.barplot(
    "count",
    "tissue",
    data=plot_df,
    orient="h",
    linewidth=0.0,
    palette=CrispyPlot.PAL_TISSUE_2,
    saturation=1.0,
    ax=ax,
)

ax.set_ylabel("")
ax.set_xlabel("Number of cell lines")
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
ax.axvline(SS_THRES, ls="--", c=CrispyPlot.PAL_DBGD[0], lw=0.3)

plt.savefig(
    f"{RPATH}/1.Attenuation_tissue_nsamples.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


# Correlations with copy number
#


def corr_genes(protein, tissue):
    LOG.info(f"{tissue} - {protein}")

    cell_lines = ss_count.loc[tissue, "set"]

    m = pd.concat(
        [
            prot.reindex(columns=cell_lines).loc[protein].rename("proteomics"),
            gexp.reindex(columns=cell_lines).loc[protein].rename("transcriptomics"),
            cnv.reindex(columns=cell_lines).loc[protein].rename("copynumber"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    prot_r, prot_p = spearmanr(m["copynumber"], m["proteomics"])
    gexp_r, gexp_p = spearmanr(m["copynumber"], m["transcriptomics"])

    return dict(
        protein=protein,
        tissue=tissue,
        protein_corr=prot_r,
        protein_pval=prot_p,
        gexp_corr=gexp_r,
        gexp_pval=gexp_p,
        n_obs=m.shape[0],
    )


patt = pd.DataFrame(
    [
        corr_genes(p, t)
        for t in ss_count.query(f"count >= {SS_THRES}").index
        for p in genes
    ]
)
patt = patt.query(f"n_obs >= {SS_THRES}")
patt = patt.assign(attenuation=patt.eval("gexp_corr - protein_corr"))

# GMM
patt_cluster = []
for t, df in patt.groupby("tissue"):
    df = df.dropna(subset=["attenuation"])
    gmm = GaussianMixture(n_components=2).fit(df[["attenuation"]])
    s_type, clusters = (
        pd.Series(gmm.predict(df[["attenuation"]]), index=df["protein"]),
        pd.Series(gmm.means_[:, 0], index=range(2)),
    )
    patt_cluster.append(
        pd.DataFrame(
            [
                (p, t, "High" if s_type[p] == clusters.argmax() else "Low")
                for p in df["protein"]
            ],
            columns=["protein", "tissue", "cluster"],
        )
    )
patt_cluster = pd.concat(patt_cluster)
patt = pd.concat(
    [
        patt.set_index(["protein", "tissue"]),
        patt_cluster.set_index(["protein", "tissue"]),
    ],
    axis=1,
).reset_index()

# Export
patt.to_csv(f"{RPATH}/1.Attenuation_score.csv", index=False, compression="gzip")

# Attenuation matrix
patt_m = pd.pivot_table(patt, index="tissue", columns="protein", values="attenuation")


#
#

df = ss_count.query(f"count >= {SS_THRES}")

ncols = 5
nrows = np.ceil(df.shape[0] / ncols).astype(int)

ax_min = patt[["gexp_corr", "protein_corr"]].min().min() * 1.1
ax_max = patt[["gexp_corr", "protein_corr"]].max().max() * 1.1

_, axs = plt.subplots(
    nrows, ncols, figsize=(2 * ncols, 2 * nrows), sharex="all", sharey="all", dpi=600
)

for i, t in enumerate(df.index):
    i_row = np.floor(i / ncols).astype(int)
    i_col = np.floor(i % ncols).astype(int)

    ax = axs[i_row, i_col]

    pal = dict(High=CrispyPlot.PAL_TISSUE_2[t], Low=CrispyPlot.PAL_DTRACE[0])

    CrispyPlot.attenuation_scatter(
        "gexp_corr",
        "protein_corr",
        patt.query(f"tissue == '{t}'"),
        pal=pal,
        ax_min=ax_min,
        ax_max=ax_max,
        ax=ax,
    )

    ax.set_title(f"{t}\n(n={ss_count.loc[t, 'count']})", y=0.8)

    if i_col != 0:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Proteomics ~ Copy number\nSpearman's R")

    if i_row != (nrows - 1):
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Transcriptomics ~ Copy number\nSpearman's R")

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{RPATH}/1.Attenuation_scatters.png", bbox_inches="tight")
plt.close("all")


#
#

plot_df = pd.pivot_table(patt.assign(cluster_bin=patt["cluster"] == "High"), index="tissue", columns="protein", values="cluster_bin")
plot_df = plot_df.loc[:, plot_df.sum() > 3]
plot_df = plot_df.applymap(lambda v: v if np.isnan(v) else int(v))

g = sns.clustermap(
    plot_df.fillna(0),
    cmap="viridis",
    linewidth=0,
    mask=plot_df.isnull(),
    figsize=(10, 4),
)
g.ax_heatmap.set_xlabel("Protein")
g.ax_heatmap.set_ylabel("Tissue")
plt.savefig(
    f"{RPATH}/1.Attenuation_clustermap.png",
    bbox_inches="tight",
    transparent=True,
    dpi=600,
)
plt.close("all")
