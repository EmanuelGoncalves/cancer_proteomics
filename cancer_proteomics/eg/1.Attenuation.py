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
from limix.stats import lrt_pvalues
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from crispy.GIPlot import GIPlot
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import pearsonr, spearmanr, chi2
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, scale
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, GeneExpression, CopyNumber, Sample


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")

ss = Proteomics().ss

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
cnv_norm = np.log2(cnv.divide(ss.loc[cnv.columns, "ploidy"]) + 1)
LOG.info(f"Copy number: {cnv.shape}")


# Overlaps
#

samples = list(set.intersection(set(prot), set(gexp), set(cnv)))
genes = list(set.intersection(set(prot.index), set(gexp.index), set(cnv.index)))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")

SS_THRES = 10

ss["model_type"] = [c if t in ["Lung", "Haematopoietic and Lymphoid"] else t for t, c in ss[["tissue", "cancer_type"]].values]

ss_pal = {**CrispyPlot.PAL_CANCER_TYPE, **CrispyPlot.PAL_TISSUE_2}

ss_count = (
    ss.loc[samples]
    .reset_index()
    .groupby("model_type")["model_id"]
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
    "model_type",
    data=plot_df,
    orient="h",
    linewidth=0.0,
    palette=ss_pal,
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


def log_likelihood(y_true, y_pred):
    n = len(y_true)
    ssr = np.power(y_true - y_pred, 2).sum()
    var = ssr / n

    l = np.longfloat(1 / (np.sqrt(2 * np.pi * var))) ** n * np.exp(
        -(np.power(y_true - y_pred, 2) / (2 * var)).sum()
    )
    ln_l = np.log(l)

    return float(ln_l)


def logratio_test(null_lml, alt_lmls, dof=1):
    lr = 2 * (alt_lmls - null_lml)
    lr_pval = chi2(dof).sf(lr)
    return lr_pval


def lm_logratio_test(y_pred_null, y_pred_alt_lm, y_true, dof=1):
    lm_null_ll = log_likelihood(y_true, y_pred_null)
    lm_alt_ll = log_likelihood(y_true, y_pred_alt_lm)
    return logratio_test(lm_null_ll, lm_alt_ll, dof=dof)


def corr_genes(protein, tissue):
    LOG.info(f"{tissue} - {protein}")

    cell_lines = ss_count.loc[tissue, "set"]

    m = pd.concat(
        [
            prot.reindex(columns=cell_lines).loc[protein].rename("p"),
            gexp.reindex(columns=cell_lines).loc[protein].rename("t"),
            cnv_norm.reindex(columns=cell_lines).loc[protein].rename("c"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    m_res = dict(
        protein=protein,
        tissue=tissue,
        protein_beta=np.nan,
        protein_corr=np.nan,
        protein_pval=np.nan,
        gexp_beta=np.nan,
        gexp_corr=np.nan,
        gexp_pval=np.nan,
        n_obs=m.shape[0],
        attenuation=np.nan,
        lr_pval_t=np.nan,
        lr_pval_p=np.nan,
    )

    if m.shape[0] >= SS_THRES:
        m["p"] = StandardScaler().fit_transform(m[["p"]])[:, 0]
        m["t"] = StandardScaler().fit_transform(m[["t"]])[:, 0]

        lm_p = LinearRegression().fit(m[["p"]], m["c"])
        lm_t = LinearRegression().fit(m[["t"]], m["c"])
        lm_pt = LinearRegression().fit(m[["p", "t"]], m["c"])

        m_res["lr_pval_p"] = lm_logratio_test(
            lm_p.predict(m[["p"]]),
            lm_pt.predict(m[["p", "t"]]),
            m["c"],
        )

        m_res["lr_pval_t"] = lm_logratio_test(
            lm_p.predict(m[["t"]]),
            lm_pt.predict(m[["p", "t"]]),
            m["c"],
        )
        m_res["protein_beta"], m_res["gexp_beta"] = lm_p.coef_[0], lm_t.coef_[0]
        m_res["attenuation"] = m_res["gexp_beta"] - m_res["protein_beta"]

        m_res["protein_corr"], m_res["protein_pval"] = spearmanr(m["c"], m["p"])
        m_res["gexp_corr"], m_res["gexp_pval"] = spearmanr(m["c"], m["t"])

    return m_res


patt = pd.DataFrame(
    [
        corr_genes(p, t)
        for t in ss_count.query(f"count >= {SS_THRES}").index
        for p in genes
    ]
).dropna()

# Adjust p-value
patt["lr_fdr_t"] = multipletests(patt["lr_pval_t"], method="fdr_bh")[1]
patt["lr_fdr_p"] = multipletests(patt["lr_pval_p"], method="fdr_bh")[1]
patt["lr_fdr_t_cluster"] = patt["lr_fdr_t"].apply(lambda v: "High" if v < .01 else "Low")
patt["lr_fdr_p_cluster"] = patt["lr_fdr_p"].apply(lambda v: "High" if v < .01 else "Low")

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

ncols = 6
nrows = np.ceil(df.shape[0] / ncols).astype(int)

ax_min = patt[["gexp_beta", "protein_beta"]].min().min() * 1.1
ax_max = patt[["gexp_beta", "protein_beta"]].max().max() * 1.1

_, axs = plt.subplots(
    nrows, ncols, figsize=(2 * ncols, 2 * nrows), sharex="all", sharey="all", dpi=600
)

for i, t in enumerate(df.index):
    i_row = np.floor(i / ncols).astype(int)
    i_col = np.floor(i % ncols).astype(int)

    ax = axs[i_row, i_col]

    pal = dict(High=ss_pal[t], Low=CrispyPlot.PAL_DTRACE[0])

    CrispyPlot.attenuation_scatter(
        "gexp_beta",
        "protein_beta",
        patt[patt["tissue"] == t],
        "lr_fdr_t_cluster",
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
plot_df = patt_m.copy().dropna(axis=1)

g = sns.clustermap(
    plot_df.fillna(0),
    cmap="Spectral",
    center=0,
    linewidth=0,
    mask=plot_df.isnull(),
    figsize=(10, 6),
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
