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
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.Utils import Utils
from crispy.GIPlot import GIPlot
from Enrichment import Enrichment
from crispy.MOFA import MOFA, MOFAPlot
from crispy.CrispyPlot import CrispyPlot
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr, zscore
from sklearn.preprocessing import quantile_transform, MinMaxScaler

LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")
CPATH = "/Users/eg14/Data/cptac/"


# Import TCGA
#

TCGA_GEXP_FILE = f"{DPATH}/GSE62944_merged_expression_voom.tsv"
TCGA_CANCER_TYPE_FILE = f"{DPATH}/GSE62944_06_01_15_TCGA_24_CancerType_Samples.txt"

gexp = pd.read_csv(TCGA_GEXP_FILE, index_col=0, sep="\t")
gexp = gexp.loc[:, [int(c.split("-")[3][:-1]) < 10 for c in gexp.columns]]
gexp.columns = [i[:12] for i in gexp]
gexp = gexp.groupby(gexp.columns, axis=1).mean()
gexp_columns = set(gexp)

ctype = pd.read_csv(TCGA_CANCER_TYPE_FILE, sep="\t", header=None, index_col=0)[1]
ctype.index = [i[:12] for i in ctype.index]
ctype = ctype.reset_index().groupby("index")[1].first()

ctype_pal = sns.color_palette("tab20c").as_hex() + sns.color_palette("tab20b").as_hex()
ctype_pal = dict(zip(natsorted(ctype.value_counts().index), ctype_pal))


#
#

stromal = pd.read_excel(f"{DPATH}/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx", index_col=0)
stromal = stromal.loc[[int(c.split("-")[3][:-1]) < 10 for c in stromal.index]]
stromal.index = [i[:12] for i in stromal.index]

stromal_count = stromal.index.value_counts()
stromal = stromal[~stromal.index.isin(stromal_count[stromal_count != 1].index)]


#
#
def sample_corr(var1, var2, idx_set):
    return spearmanr(
        var1.reindex(index=idx_set), var2.reindex(index=idx_set), nan_policy="omit"
    )


# Proteomics data-sets
dfiles = [
    "Human__TCGA_BRCA__BI__Proteome__QExact__01_28_2016__BI__Gene__CDAP_iTRAQ_UnsharedLogRatio_r2.cct",
    # "Human__TCGA_COADREAD__VU__Proteome__Velos__01_28_2016__VU__Gene__CDAP_UnsharedPrecursorArea_r2.cct",
    "Human__TCGA_OV__JHU__Proteome__Velos__01_28_2016__JHU__Gene__CDAP_iTRAQ_UnsharedLogRatio_r2.cct",
    "Human__TCGA_OV__PNNL__Proteome__Velos___QExact__01_28_2016__PNNL__Gene__CDAP_iTRAQ_UnsharedLogRatio_r2.cct",
]

dmatrix, ms_type = [], []
for dfile in dfiles:
    df = pd.read_csv(f"{CPATH}/linkedomics/{dfile}", sep="\t", index_col=0)

    if "COADREAD" in dfile:
        df = df.replace(0, np.nan)
        # df = df / df.sum()
        df = df.pipe(np.log2)

    # # df = pd.DataFrame(df.pipe(zscore, nan_policy="omit", axis=1), index=df.index, columns=df.columns)
    # df = df[df.count(1) > (df.shape[1] * .5)]
    df = pd.DataFrame({i: Utils.gkn(df.loc[i].dropna()).to_dict() for i in df.index}).T

    # Simplify barcode
    df.columns = [i[:12].replace(".", "-") for i in df]

    # MS type
    ms_type.append(pd.Series("LF" if "COADREAD" in dfile else "TMT", index=df.columns))

    dmatrix.append(df)

ms_type = pd.concat(ms_type).reset_index().groupby("index").first()[0]
dmatrix = pd.concat(dmatrix, axis=1)
LOG.info(f"Assembled data-set: {dmatrix.shape}")

# Dicard poor corelated
remove_samples = {i for i in set(dmatrix) if dmatrix.loc[:, [i]].shape[1] == 2 and dmatrix.loc[:, [i]].corr().iloc[0, 1] < .4}
dmatrix = dmatrix.drop(remove_samples, axis=1)
LOG.info(f"Poor correlating samples removed: {dmatrix.shape}")

# Average replicates
dmatrix = dmatrix.groupby(dmatrix.columns, axis=1).mean()
LOG.info(f"Replicates averaged: {dmatrix.shape}")

# Map to gene-expression
d_idmap = {c: [gc for gc in gexp_columns if c in gc] for c in dmatrix}
d_idmap = {k: v[0] for k, v in d_idmap.items() if len(v) == 1}
dmatrix = dmatrix[d_idmap.keys()].rename(columns=d_idmap)
LOG.info(f"Gexp map (Proteins x Samples): {dmatrix.shape}")

# Finalise and export
dmatrix.to_csv(f"{DPATH}/merged_cptac_tcga_proteomics.csv.gz", compression="gzip")
completeness = dmatrix.count().sum() / np.prod(dmatrix.shape)
LOG.info(f"Completeness: {completeness * 100:.1f}%")


#
#

s_pg_corr = pd.DataFrame(
    {
        s: sample_corr(dmatrix[s], gexp[s], set(dmatrix.index).intersection(gexp.index))
        for s in dmatrix
    },
    index=["corr", "pvalue"],
).T

covariates = pd.concat([
    s_pg_corr["corr"].rename("GExpProtCorr"),
    dmatrix.count().pipe(np.log).rename("NProteins"),
    dmatrix.median().rename("GlobalProteomics"),
    # ms_type.reindex(s_pg_corr.index).str.get_dummies(),
    stromal.reindex(s_pg_corr.index)["CPE"].rename("Purity"),
], axis=1).dropna()


#
#

dmatrix, gexp = dmatrix[covariates.index], gexp[covariates.index]

mofa = MOFA(
    views=dict(proteomics=dmatrix, transcriptomics=gexp),
    covariates=dict(
        proteomics=covariates[["NProteins", "GlobalProteomics", "Purity"]],
        transcriptomics=covariates[["NProteins", "GlobalProteomics", "Purity"]],
    ),
    iterations=2000,
    use_overlap=True,
    convergence_mode="fast",
    factors_n=10,
    from_file=f"{RPATH}/1.MultiOmics_CPTAC.hdf5",
)


# Variance explained across data-sets
#

factors_corr = {}
for f in mofa.factors:
    factors_corr[f] = {}

    for c in covariates:
        fc_samples = list(covariates.reindex(mofa.factors[f].index)[c].dropna().index)
        factors_corr[f][c] = pearsonr(mofa.factors[f][fc_samples], covariates[c][fc_samples])[0]

factors_corr = pd.DataFrame(factors_corr)


MOFAPlot.variance_explained_heatmap(mofa)
plt.savefig(
    f"{RPATH}/1.MultiOmics_CPTAC_factors_rsquared_heatmap.pdf", bbox_inches="tight"
)
plt.close("all")

MOFAPlot.covariates_heatmap(factors_corr, mofa, ctype)
plt.savefig(
    f"{RPATH}/1.MultiOmics_CPTAC_factors_covariates_clustermap.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


#
#

grid = GIPlot.gi_regression("Purity", "GExpProtCorr", covariates)
grid.set_axis_labels("Purity", "GExpProtCorr")
plt.savefig(
    f"{RPATH}/1.MultiOmics_GExpProtCorr_Purity.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Factor 3
#

f = "F2"
factor_df = pd.concat(
    [
        mofa.factors[f],
        covariates,
        ctype,
    ],
    axis=1,
    sort=False,
).dropna(subset=[f])


for y_var in ["GExpProtCorr"]:
    grid = GIPlot.gi_regression(f, y_var, factor_df)
    grid.set_axis_labels(f"CPTAC Factor {f[1:]}", y_var)
    plt.savefig(
        f"{RPATH}/1.MultiOmics_{f}_CPTAC_{y_var}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

#
enr_views = ["proteomics"]
f_enr = mofa.pathway_enrichment(f, views=enr_views)

f_enr_overlap = pd.concat([
    f_enr.set_index("Term|NES")["nes"],
    pd.read_csv(f"{RPATH}/1.MultiOmics_F6_gseapy_proteomics.csv", index_col=0),
], axis=1).dropna()

#

gs_dw = f_enr_overlap[(f_enr_overlap["Sanger&CMRI"] < -0.2) & (f_enr_overlap["nes"] < -0.2)].sort_values("Sanger&CMRI")
gs_highlight = list(gs_dw.index)

gs_palette = pd.Series(
    sns.light_palette("#e6550d", n_colors=len(gs_dw) + 1, reverse=True).as_hex()[:-1],
    index=gs_highlight,
)

_, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

ax.scatter(
    f_enr_overlap["nes"], f_enr_overlap["Sanger&CMRI"], c=GIPlot.PAL_DBGD[2], s=5, linewidths=0
)

for g in gs_highlight:
    ax.scatter(
        f_enr_overlap.loc[g, "nes"],
        f_enr_overlap.loc[g, "Sanger&CMRI"],
        c=gs_palette[g],
        s=10,
        linewidths=0,
        label=g,
    )

cor, pval = spearmanr(f_enr_overlap["nes"], f_enr_overlap["Sanger&CMRI"])
annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
ax.text(0.98, 0.02, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel(f"CPTAC")
ax.set_ylabel("Sanger&CMRI")
# ax.set_title(f"Factor {f[1:]} Proteomics weights enrichment score (NES)")

ax.legend(frameon=False, prop={"size": 4}, loc="center left", bbox_to_anchor=(1, 0.5))

plt.savefig(f"{RPATH}/1.MultiOmics_{f}_gseapy_CPTAC_scatter.pdf", bbox_inches="tight")
plt.close("all")

