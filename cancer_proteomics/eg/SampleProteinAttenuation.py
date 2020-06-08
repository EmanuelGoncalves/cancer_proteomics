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
# prot = prot_obj.protein_raw.reindex(prot.index).dropna(how="all")
# prot = prot[prot.count(1) > 100]
# prot = pd.DataFrame({i: Utils.gkn(prot.loc[i].dropna()).to_dict() for i in prot.index}).T
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


# Data tranformations
#

gexp = pd.DataFrame({i: Utils.gkn(gexp.loc[i].dropna()).to_dict() for i in genes}).T


# Correlations with copy number
#


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


def corr_genes(s):
    LOG.info(f"Sample={s}")
    prot_r, prot_p, prot_len = sample_corr(prot[s], cn[s], genes)
    gexp_r, gexp_p, gexp_len = sample_corr(gexp[s], cn[s], set(prot[s].dropna().index))
    return dict(
        sample=s,
        prot_r=prot_r,
        prot_p=prot_p,
        prot_l=prot_len,
        gexp_r=gexp_r,
        gexp_p=gexp_p,
        gexp_l=gexp_len,
    )


patt = pd.DataFrame([corr_genes(s) for s in samples]).dropna()
patt = patt.assign(attenuation=patt.eval("gexp_r - prot_r"))
patt = patt.set_index("sample")
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
prot_filter = prot[prot.count(1) > 150][patt["attenuation"].index]
prot_corr = prot_filter.T.corrwith(patt["attenuation"])

prot_corr_enr = pd.concat(
    [
        gseapy.ssgsea(
            prot_corr,
            gene_sets=Enrichment.read_gmt(f"{DPATH}/pathways/{g}"),
            no_plot=True,
        )
        .res2d.assign(geneset=g)
        .reset_index()
        for g in [
            "c6.all.v7.1.symbols.gmt",
            "c5.all.v7.1.symbols.gmt",
            "h.all.v7.1.symbols.gmt",
            "c2.all.v7.1.symbols.gmt",
        ]
    ],
    ignore_index=True,
)
prot_corr_enr = prot_corr_enr.rename(columns={"sample1": "nes"}).sort_values("nes")

sig_enr = gseapy.ssgsea(
    prot_corr,
    gene_sets=Enrichment.read_gmt(f"{DPATH}/pathways/c2.all.v7.1.symbols.gmt"),
    no_plot=False,
    permutation_num=1000,
    verbose=True,
)
sig_enr_results = pd.read_csv(f"{RPATH}/satt_scatter.gseapy.ssgsea.gene_sets.report.txt", sep="\t", index_col=0)

ledge_genes = set.intersection(*[{g for g in sig_enr_results.loc[s, "ledge_genes"].split(";")} for s in sig_enr_results.sort_values("es").head(5)["ledge_genes"].index])

#
plot_df = pd.concat([
    patt["prot_r"],
    patt["gexp_r"],
    prot.reindex(ledge_genes).mean().rename("ledge"),
], axis=1).dropna().sort_values("ledge", ascending=False)

g = GIPlot.gi_continuous_plot("gexp_r", "prot_r", "ledge", plot_df, cbar_label="Leading-edge enrichment")

g.set_xlabel("Transcriptomics ~ Copy number\n(Pearson's R)")
g.set_ylabel("Protein ~ Copy number\n(Pearson's R)")

plt.savefig(f"{RPATH}/Satt_scatter_ledge.pdf", bbox_inches="tight")
plt.close("all")
