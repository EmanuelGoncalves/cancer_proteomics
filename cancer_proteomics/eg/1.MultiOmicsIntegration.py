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
import logging
import matplotlib
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from scipy.stats import spearmanr, ttest_ind
from crispy.MOFA import MOFA, MOFAPlot
from crispy.CrispyPlot import CrispyPlot
from cancer_proteomics.eg.LMModels import LMModels
from crispy.DimensionReduction import dim_reduction, plot_dim_reduction
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    Methylation,
    CopyNumber,
    CRISPR,
    Sample,
    DrugResponse,
    WES,
    Mobem,
)


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")


# Data-sets
#

prot_obj = Proteomics()
methy_obj = Methylation()
gexp_obj = GeneExpression()

wes_obj = WES()
cn_obj = CopyNumber()
mobem_obj = Mobem()

crispr_obj = CRISPR()
drug_obj = DrugResponse()


# Samples
#

ss = prot_obj.ss

samples = set.intersection(
    set(prot_obj.get_data()), set(gexp_obj.get_data()), set(methy_obj.get_data())
)
LOG.info(f"Samples: {len(samples)}")


# Filter data-sets
#

prot = prot_obj.filter(subset=samples)
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp_obj.filter(subset=samples)
gexp = gexp.loc[gexp.std(1) > 1]
LOG.info(f"Transcriptomics: {gexp.shape}")

methy = methy_obj.filter(subset=samples)
methy = methy.loc[methy.std(1) > 0.05]
LOG.info(f"Methylation: {methy.shape}")

crispr = crispr_obj.filter(subset=samples, dtype="merged", abs_thres=0.5, min_events=3)
LOG.info(f"CRISPR: {crispr.shape}")

drespo = drug_obj.filter(subset=samples, filter_max_concentration=True, filter_combinations=True)
drespo = drespo.set_index(pd.Series([";".join(map(str, i)) for i in drespo.index]))
LOG.info(f"Drug response: {drespo.shape}")

cn = cn_obj.filter(subset=samples.intersection(ss.index))
cn = cn.loc[cn.std(1) > 1]
cn = np.log2(cn.divide(ss.loc[cn.columns, "ploidy"]) + 1)
cn = cn.dropna(how="all", axis=1)
cn = cn.fillna(cn.mean())
LOG.info(f"Copy-Number: {cn.shape}")

wes = wes_obj.filter(subset=samples, min_events=3, recurrence=True)
wes = wes.loc[wes.std(1) > 0]
LOG.info(f"WES: {wes.shape}")

mobem = mobem_obj.filter(subset=samples)
mobem = mobem.loc[mobem.std(1) > 0]
LOG.info(f"MOBEM: {mobem.shape}")


# MOFA
#

groupby = ss.loc[samples, "tissue"].apply(
    lambda v: "Haem" if v == "Haematopoietic and Lymphoid" else "Other"
)

mofa = MOFA(
    views=dict(proteomics=prot, transcriptomics=gexp, methylation=methy, drespo=drespo),
    groupby=groupby,
    iterations=2000,
    use_overlap=True,
    convergence_mode="fast",
    factors_n=30,
    from_file=f"{RPATH}/1.MultiOmics.hdf5",
)


# Factor clustermap
#

MOFAPlot.factors_corr_clustermap(mofa)
plt.savefig(f"{RPATH}/1.MultiOmics_factors_corr_clustermap.pdf", bbox_inches="tight")
plt.close("all")


# Variance explained across data-sets
#

MOFAPlot.variance_explained_heatmap(mofa)
plt.savefig(f"{RPATH}/1.MultiOmics_rsquared_heatmap.pdf", bbox_inches="tight")
plt.close("all")


# Sample Protein ~ Transcript correlation
#

s_pg_corr = pd.read_csv(f"{RPATH}/2.SLProteinInteractions_gexp_prot_samples_corr.csv", index_col=0)


# PPI
#

novel_ppis = pd.read_csv(f"{RPATH}/2.SLProteinInteractions.csv.gz").query(f"(prot_fdr < .05) & (prot_corr > 0.5)")


# Factors data-frame integrated with other measurements
#

covariates = pd.concat(
    [
        prot.count().rename("Proteomics n. measurements"),
        prot_obj.protein_raw.median().rename("Global proteomics"),
        methy.mean().rename("Global methylation"),
        s_pg_corr["corr"].rename("gexp_prot_corr"),
        pd.get_dummies(ss["msi_status"]),
        pd.get_dummies(ss["growth_properties"]),
        pd.get_dummies(ss["tissue"])["Haematopoietic and Lymphoid"],
        ss.reindex(index=samples, columns=["ploidy", "mutational_burden", "growth"]),
        wes.loc[["TP53"]].T.add_suffix("_wes"),
        prot.loc[["CDH1", "VIM", "BCL2L1"]].T.add_suffix("_proteomics"),
        gexp_obj.get_data().loc[["CDH1", "VIM", "MCL1", "BCL2L1"]].T.add_suffix("_transcriptomics"),
        methy.loc[["SLC5A1"]].T.add_suffix("_methylation"),
    ],
    axis=1,
).reindex(mofa.factors.index)

covariates_corr = pd.DataFrame(
    {
        f: {
            c: spearmanr(mofa.factors[f], covariates[c], nan_policy="omit")[0]
            for c in covariates
        }
        for f in mofa.factors
    }
)

covariates_df = pd.concat([covariates, mofa.factors], axis=1, sort=False)


# Covairates correlation heatmap
#

MOFAPlot.covariates_heatmap(covariates_corr, mofa)
plt.savefig(
    f"{RPATH}/1.MultiOmics_factors_covariates_clustermap.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# # Factor association analysis
# #
#
# m = LMModels.define_covariates(institute=crispr_obj.merged_institute, cancertype=False, mburden=False, ploidy=False)
# lmm_crispr = LMModels(y=mofa.factors, x=crispr.T, m=m).matrix_lmm()
# lmm_crispr.to_csv(
#     f"{RPATH}/1.MultiOmics_lmm_crispr.csv.gz", index=False, compression="gzip"
# )
# print(lmm_crispr.query("fdr < 0.05").head(60))
#
# m = LMModels.define_covariates(institute=False, cancertype=False, mburden=False, ploidy=False)
# lmm_cnv = LMModels(y=mofa.factors, x=cn.T, m=m).matrix_lmm()
# lmm_cnv.to_csv(
#     f"{RPATH}/1.MultiOmics_lmm_cnv.csv.gz", index=False, compression="gzip"
# )
# print(lmm_cnv.query("fdr < 0.05").head(60))
#
# m = LMModels.define_covariates(institute=False, cancertype=False, mburden=False, ploidy=False)
# lmm_wes = LMModels(y=mofa.factors, x=wes.T, transform_x="none", m=m).matrix_lmm()
# lmm_wes.to_csv(
#     f"{RPATH}/1.MultiOmics_lmm_wes.csv.gz", index=False, compression="gzip"
# )
# print(lmm_wes.query("fdr < 0.05").head(60))
#
# m = LMModels.define_covariates(institute=False, cancertype=False, mburden=False, ploidy=False)
# lmm_mobems = LMModels(y=mofa.factors, x=mobem.T, transform_x="none", m=m).matrix_lmm()
# lmm_mobems.to_csv(
#     f"{RPATH}/1.MultiOmics_lmm_mobems.csv.gz", index=False, compression="gzip"
# )
# print(lmm_mobems.query("fdr < 0.05").head(60))


# Factors 1 and 2 associations
#

f_x, f_y = "F1", "F2"
plot_df = pd.concat(
    [
        mofa.factors[[f_x, f_y]],
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_transcriptomics"),
        prot.loc[["VIM"]].T.add_suffix("_proteomics"),
        mofa.views["methylation"].loc[["PTPRT"]].T.add_suffix("_methylation"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

# Regression plots
for f in [f_x, f_y]:
    for v in ["CDH1_transcriptomics", "VIM_transcriptomics", "VIM_proteomics"]:
        grid = GIPlot.gi_regression(f, v, plot_df)
        grid.set_axis_labels(f"Factor {f[1:]}", v.replace("_", " "))
        plt.savefig(
            f"{RPATH}/1.MultiOmics_{f}_{v}_regression.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

# Tissue plot
ax = GIPlot.gi_tissue_plot(f_x, f_y, plot_df)
ax.set_xlabel(f"Factor {f_x[1:]}")
ax.set_ylabel(f"Factor {f_y[1:]}")
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f_x}_{f_y}_tissue_plot.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

# Continous annotation
for z in [
    "VIM_proteomics",
    "VIM_transcriptomics",
    "CDH1_transcriptomics",
    "PTPRT_methylation",
]:
    ax = GIPlot.gi_continuous_plot(f_x, f_y, z, plot_df, cbar_label=z.replace("_", " "))
    ax.set_xlabel(f"Factor {f_x[1:]}")
    ax.set_ylabel(f"Factor {f_y[1:]}")
    plt.savefig(
        f"{RPATH}/1.MultiOmics_{f_x}_{f_y}_continous{z}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
