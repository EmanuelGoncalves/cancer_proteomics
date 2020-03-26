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
from scipy.stats import pearsonr, spearmanr
from crispy.GIPlot import GIPlot
from crispy.CrispyPlot import CrispyPlot
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

prot = Proteomics().filter()
LOG.info(f"Proteomics: {prot.shape}")


# Gene expression
#

gexp = GeneExpression().filter(subset=list(prot))
LOG.info(f"Transcriptomics: {gexp.shape}")


# CRISPR
#

crispr = CRISPR().filter(subset=list(prot))
LOG.info(f"CRISPR: {crispr.shape}")


# Overlaps
#

ss = Sample().samplesheet

samples = list(set.intersection(set(prot), set(gexp)))
genes = list(set.intersection(set(prot.index), set(gexp.index)))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")


# Protein ~ Transcript correlations
#


def corr_genes(g):
    mprot = prot.loc[g, samples].dropna()
    mgexp = gexp.loc[g, mprot.index]
    r, p = pearsonr(mprot, mgexp)
    return dict(gene=g, corr=r, pval=p, len=len(mprot.index))


res = pd.DataFrame([corr_genes(g) for g in genes]).sort_values("pval")
res["fdr"] = multipletests(res["pval"], method="fdr_bh")[1]
res["attenuation"] = (
    p_attenuated.reindex(res["gene"])["attenuation_potential"]
    .replace(np.nan, "NA")
    .values
)
res.to_csv(
    f"{RPATH}/0.Protein_Gexp_correlations.csv.gz", index=False, compression="gzip"
)
# res = pd.read_csv(f"{RPATH}/0.Protein_Gexp_correlations.csv.gz")


# Protein ~ GExp correlation histogram
#

_, ax = plt.subplots(1, 1, figsize=(2.0, 1.5))
sns.distplot(
    res["corr"],
    hist_kws=dict(alpha=0.4, zorder=1, linewidth=0),
    bins=30,
    kde_kws=dict(cut=0, lw=1, zorder=1, alpha=0.8),
    color=CrispyPlot.PAL_DTRACE[2],
    ax=ax,
)
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
ax.set_xlabel("Pearson's R")
ax.set_ylabel("Density")
ax.set_title(f"Protein ~ Transcript (mean R={res['corr'].mean():.2f})")
plt.savefig(
    f"{RPATH}/0.ProteinGexp_corr_histogram.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


# Protein attenuation
#

pal = dict(zip(*(["Low", "High", "NA"], CrispyPlot.PAL_DTRACE.values())))

_, ax = plt.subplots(1, 1, figsize=(1.0, 1.5), dpi=600)
sns.boxplot(
    x="attenuation",
    y="corr",
    notch=True,
    data=res,
    order=["High", "Low", "NA"],
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    medianprops=CrispyPlot.MEDIANPROPS,
    flierprops=CrispyPlot.FLIERPROPS,
    palette=pal,
    showcaps=False,
    saturation=1,
    ax=ax,
)
ax.set_xlabel("Attenuation")
ax.set_ylabel("Transcript ~ Protein")
ax.set_title("Protein expression attenuated\nin tumour patient samples")
ax.axhline(0, ls="-", lw=0.1, c="black", zorder=0)
ax.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)
plt.savefig(
    f"{RPATH}/0.ProteinGexp_corr_attenuation_boxplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Representative examples
#

for gene in [
    "VIM",
    "EGFR",
    "IDH2",
    "NRAS",
    "SMARCB1",
    "ERBB2",
    "STAG1",
    "STAG2",
    "TP53",
    "RAC1",
    "MET",
]:
    plot_df = pd.concat(
        [
            prot.loc[gene, samples].rename("protein"),
            gexp.loc[gene, samples].rename("transcript"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    grid = GIPlot.gi_regression("protein", "transcript", plot_df)
    grid.set_axis_labels(f"Protein", f"Transcript")
    grid.ax_marg_x.set_title(gene)
    plt.savefig(f"{RPATH}/0.Protein_Transcript_scatter_{gene}.pdf", bbox_inches="tight")
    plt.close("all")


# Hexbin correlation with CRISPR
#

plot_df = (
    pd.concat(
        [
            crispr.reindex(index=genes, columns=samples).unstack().rename("CRISPR"),
            gexp.reindex(index=genes, columns=samples).unstack().rename("Transcriptomics"),
            prot.reindex(index=genes, columns=samples).unstack().rename("Proteomics"),
        ],
        axis=1,
        sort=False,
    )
    .dropna()
    .query("Proteomics < 10")
)


for x, y in [("Transcriptomics", "CRISPR"), ("Proteomics", "CRISPR")]:
    _, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

    ax.hexbin(
        plot_df[x],
        plot_df[y],
        cmap="Spectral_r",
        gridsize=100,
        mincnt=1,
        bins="log",
        lw=0,
    )

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

    ax.set_xlabel("Protein (intensities)" if x == "Proteomics" else "Transcript (voom)")
    ax.set_ylabel(f"{y} (scaled log2)")

    plt.savefig(
        f"{RPATH}/0.ProteinGexp_hexbin_{x}_{y}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


#
#

# dtype = "tissue"
dtype_thres = 10
dtype_ss = ss.reindex(samples)[["model_name", "tissue", "cancer_type"]]

for dtype in ["tissue", "cancer_type"]:
    dtype_count = dtype_ss[dtype].value_counts()
    dtype_ss = dtype_ss[dtype_ss[dtype].isin(dtype_count[dtype_count > dtype_thres].index)]

    dtype_ss_corr = pd.DataFrame(
        {
            s: spearmanr(gexp.loc[genes, s], prot.loc[genes, s], nan_policy="omit")
            for s in dtype_ss.index
        },
        index=["corr", "pval"],
    ).T
    dtype_ss_corr[dtype] = dtype_ss.loc[dtype_ss_corr.index][dtype]

    dtype_genes_corr = []
    for ctype, df in dtype_ss.groupby(dtype):
        LOG.info(f"{dtype}={ctype}")
        for g in genes:
            x, y = gexp.loc[g, df.index], prot.loc[g, df.index]

            if y.count() < dtype_thres:
                continue

            c, p = spearmanr(x, y, nan_policy="omit")
            res = dict(ctype=ctype, gene=g, corr=c, pval=p, nsamples=y.count())
            dtype_genes_corr.append(res)
    dtype_genes_corr = pd.DataFrame(dtype_genes_corr)
    dtype_genes_corr_m = pd.pivot_table(
        dtype_genes_corr, index="gene", columns="ctype", values="corr"
    )

    #
    #
    order = dtype_ss_corr.groupby(dtype)["corr"].median().sort_values()

    _, ax = plt.subplots(1, 1, figsize=(1.0, 0.125 * len(order)), dpi=600)
    sns.boxplot(
        x="corr",
        y=dtype,
        data=dtype_ss_corr,
        notch=True,
        order=order.index,
        boxprops=dict(linewidth=0.3),
        whiskerprops=dict(linewidth=0.3),
        medianprops=CrispyPlot.MEDIANPROPS,
        flierprops=CrispyPlot.FLIERPROPS,
        color=GIPlot.PAL_DTRACE[2],
        showcaps=False,
        saturation=1,
        orient="h",
        ax=ax,
    )
    ax.set_xlabel("Transcript ~ Protein")
    ax.set_ylabel("")
    ax.grid(axis="x", lw=0.1, color="#e1e1e1", zorder=0)
    plt.savefig(
        f"{RPATH}/0.ProteinGexp_sample_{dtype}_correlation_boxplot.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")
