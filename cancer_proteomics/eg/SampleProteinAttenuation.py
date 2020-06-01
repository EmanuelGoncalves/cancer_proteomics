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


# Data-sets
#

prot_obj = Proteomics()
gexp_obj = GeneExpression()
cn_obj = CopyNumber()
crispr_obj = CRISPR()
wes_obj = WES()


# Samples
#

ss = prot_obj.ss.dropna(subset=["ploidy"])

samples = set.intersection(
    set(ss.index),
    set(prot_obj.get_data()),
    set(gexp_obj.get_data()),
    set(cn_obj.get_data()),
)
LOG.info(f"Samples: {len(samples)}")


# Filter data-sets
#

prot = prot_obj.filter(subset=set(ss.index))
prot = prot_obj.protein_raw.reindex(prot.index).dropna(how="all")
prot = pd.DataFrame({i: Utils.gkn(prot.loc[i].dropna()).to_dict() for i in prot.index}).T
# prot = prot_obj.broad.copy()
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp_obj.filter(subset=set(ss.index))
LOG.info(f"Transcriptomics: {gexp.shape}")

cn = cn_obj.filter(subset=set(ss.index), dtype="gistic")
# cn_inst = cn_obj.genomic_instability()
LOG.info(f"Copy-Number: {cn.shape}")

wes = wes_obj.filter(subset=samples, min_events=3, recurrence=True)
wes = wes.loc[wes.std(1) > 0]
LOG.info(f"WES: {wes.shape}")

crispr = crispr_obj.filter(dtype="merged", subset=set(ss.index))
LOG.info(f"CRISPR: {crispr.shape}")


# Overlaps
#

samples = list(set.intersection(set(prot), set(gexp), set(cn)))
genes = list(set.intersection(set(prot.index), set(gexp.index), set(cn.index)))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")


# Correlations with copy number
#


def sample_corr(var1, var2, idx_set=None, method="spearman"):
    if idx_set is None:
        idx_set = set(var1.index).intersection(var2.index)

    if method == "spearman":
        r, p = spearmanr(
            var1.reindex(index=idx_set), var2.reindex(index=idx_set), nan_policy="omit"
        )
    else:
        r, p = pearsonr(
            var1.reindex(index=idx_set), var2.reindex(index=idx_set)
        )

    return r, p, len(idx_set)


def corr_genes(s):
    LOG.info(f"Sample={s}")
    prot_r, prot_p, prot_len = sample_corr(cn.loc[genes, s].dropna(), prot.loc[genes, s].dropna())
    gexp_r, gexp_p, gexp_len = sample_corr(cn.loc[genes, s].dropna(), gexp.loc[genes, s].dropna())
    return dict(
        sample=s,
        prot_r=prot_r,
        prot_p=prot_p,
        gexp_r=gexp_r,
        gexp_p=gexp_p,
    )


patt = pd.DataFrame([corr_genes(s) for s in samples]).dropna()
patt = patt.assign(attenuation=patt.eval("gexp_r - prot_r"))
print(patt.sort_values("attenuation"))

# GMM
gmm = GaussianMixture(n_components=2).fit(patt[["attenuation"]])
s_type, clusters = (
    pd.Series(gmm.predict(patt[["attenuation"]]), index=patt.index),
    pd.Series(gmm.means_[:, 0], index=range(2)),
)
patt["cluster"] = [
    "High" if s_type[i] == clusters.argmax() else "Low" for i in patt.index
]

patt.to_csv(f"{RPATH}/Satt_scores.csv.gz", index=False, compression="gzip")


#
#

# Attenuation scatter
#

ax_min = patt[["prot_r", "gexp_r"]].min().min() * 1.1
ax_max = patt[["prot_r", "gexp_r"]].max().max() * 1.1

pal = dict(High=CrispyPlot.PAL_DTRACE[1], Low=CrispyPlot.PAL_DTRACE[0])

g = sns.jointplot(
    "gexp_r",
    "prot_r",
    patt,
    "scatter",
    color=CrispyPlot.PAL_DTRACE[0],
    xlim=[ax_min, ax_max],
    ylim=[ax_min, ax_max],
    space=0,
    s=5,
    edgecolor="w",
    linewidth=0.0,
    marginal_kws={"hist": False, "rug": False},
    stat_func=None,
    alpha=0.1,
)

for n in ["Low", "High"]:
    df = patt.query(f"cluster == '{n}'")
    g.x, g.y = df["gexp_r"], df["prot_r"]
    g.plot_joint(
        sns.regplot,
        color=pal[n],
        fit_reg=False,
        scatter_kws={"s": 3, "alpha": 0.5, "linewidth": 0},
    )
    g.plot_joint(
        sns.kdeplot,
        cmap=sns.light_palette(pal[n], as_cmap=True),
        legend=False,
        shade=False,
        shade_lowest=False,
        n_levels=9,
        alpha=0.8,
        lw=0.1,
    )
    g.plot_marginals(sns.kdeplot, color=pal[n], shade=True, legend=False)

handles = [mpatches.Circle([0, 0], 0.25, facecolor=pal[s], label=s) for s in pal]
g.ax_joint.legend(
    loc="upper left", handles=handles, title="Protein\nattenuation", frameon=False
)

g.ax_joint.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
g.ax_joint.plot([ax_min, ax_max], [ax_min, ax_max], "k--", lw=0.3)

plt.gcf().set_size_inches(2.5, 2.5)
g.set_axis_labels(
    "Transcriptomics ~ Copy number\n(Pearson's R)",
    "Protein ~ Copy number\n(Pearson's R)",
)
plt.savefig(f"{RPATH}/Satt_scatter.pdf", bbox_inches="tight")
plt.close("all")


#
#

covs = pd.concat(
    [
        s_pg_corr["corr"].rename("GexpProtCorr"),
        cn_inst.rename("GenomicInstability"),
        pd.get_dummies(ss["media"]),
        pd.get_dummies(ss["msi_status"]),
        pd.get_dummies(ss["growth_properties"]),
        pd.get_dummies(ss["tissue"])["Haematopoietic and Lymphoid"],
        ss.reindex(index=samples, columns=["ploidy", "mutational_burden", "growth"]),
        prot.count().pipe(np.log2).rename("NProteins"),
        prot_obj.reps.rename("RepsCorrelation"),
        prot_obj.protein_raw.median().rename("GlobalProteomics"),
    ],
    axis=1,
)


#
#

# s = "SIDM00424"
for s in ["SIDM00424", "SIDM00930", "SIDM00462", "SIDM00985"]:
    plot_df = pd.concat(
        [prot[s].rename("prot"), gexp[s].rename("gexp")], axis=1
    ).dropna()
    grid = GIPlot.gi_regression("prot", "gexp", plot_df)
    plt.savefig(f"{RPATH}/satt_scatter_{s}.pdf", bbox_inches="tight", transparent=True)
    plt.close("all")


#
#

genesets = ["c5.all.v7.1.symbols.gmt"]

gs_corr = {}
for s in list(samples):
    LOG.info(f"sample = {s}")
    df_prot = prot[s].reindex(gexp.index).dropna()
    df_gexp = gexp.loc[df_prot.index, s]

    sigs = {
        k: v
        for g in genesets
        for k, v in Enrichment.read_gmt(
            f"{DPATH}/pathways/{g}", subset=set(df_prot.index)
        ).items()
    }

    gs_corr[s] = {s: sample_corr(df_prot, df_gexp, sigs[s])[2] for s in sigs}
gs_corr = pd.DataFrame(gs_corr)
gs_corr.to_csv(f"{RPATH}/satt_gs_corr.csv.gz", compression="gzip")

gs_corr_filter = gs_corr[gs_corr.std(1) > -np.log10(0.01)]
gs_corr_filter = gs_corr_filter[gs_corr_filter.count(1) > 300]

#
#

crispr_covs = LMModels.define_covariates(
    institute=crispr_obj.merged_institute,
    medium=True,
    cancertype=False,
    tissuetype=False,
    mburden=False,
    ploidy=False,
)
crispr_covs = (
    pd.concat(
        [
            crispr_covs,
            covs[["GenomicInstability", "RepsCorrelation", "GlobalProteomics"]],
        ],
        axis=1,
    )
    .reindex(samples)
    .dropna()
)

lm_crispr = LModel(
    Y=crispr[crispr_covs.index].T, X=gs_corr_filter[crispr_covs.index].T, M=crispr_covs
).fit_matrix()
print(lm_crispr.query("fdr < 0.1").head(60))
