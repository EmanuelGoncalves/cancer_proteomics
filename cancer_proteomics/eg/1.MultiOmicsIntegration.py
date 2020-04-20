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
from matplotlib import cm
from Enrichment import Enrichment
from crispy.GIPlot import GIPlot
from scipy.stats import spearmanr, ttest_ind
from crispy.MOFA import MOFA, MOFAPlot
from crispy.CrispyPlot import CrispyPlot
from cancer_proteomics.eg.LMModels import LMModels
from cancer_proteomics.eg.SLinteractionsSklearn import LModel
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

crispr = crispr_obj.filter(subset=samples, dtype="merged")
LOG.info(f"CRISPR: {crispr.shape}")

drespo = drug_obj.filter(
    subset=samples, filter_max_concentration=True, filter_combinations=True
)
drespo = drespo.set_index(pd.Series([";".join(map(str, i)) for i in drespo.index]))
LOG.info(f"Drug response: {drespo.shape}")

cn = cn_obj.filter(subset=samples.intersection(ss.index))
cn = cn.loc[cn.std(1) > 0]
cn = np.log2(cn.divide(ss.loc[cn.columns, "ploidy"]) + 1)
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

s_pg_corr = pd.read_csv(
    f"{RPATH}/2.SLProteinInteractions_gexp_prot_samples_corr.csv", index_col=0
)


# PPI
#

novel_ppis = pd.read_csv(f"{RPATH}/2.SLProteinInteractions.csv.gz").query(
    f"(prot_fdr < .05) & (prot_corr > 0.5)"
)


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
        gexp_obj.get_data()
        .loc[["CDH1", "VIM", "MCL1", "BCL2L1"]]
        .T.add_suffix("_transcriptomics"),
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


# Factor association analysis
#
covs = (
    LMModels.define_covariates(
        institute=crispr_obj.merged_institute,
        medium=True,
        cancertype=False,
        tissuetype=True,
        mburden=False,
        ploidy=True,
    )
    .reindex(crispr.columns)
    .dropna()
)

lmm_crispr = LModel(
    Y=crispr[covs.index].T, X=mofa.factors.loc[covs.index], M=covs
).fit_matrix()
print(lmm_crispr.query("fdr < 0.05").head(60))

lmm_mobems = LModel(
    Y=mofa.factors.loc[covs.index],
    X=mobem[covs.index].T,
    M=covs.drop(columns=["Broad", "Sanger"]),
).fit_matrix()
print(lmm_mobems.query("fdr < 0.05").head(60))


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


# Factor 6
#

f = "F6"
plot_df = pd.concat(
    [
        mofa.factors[[f]],
        covariates[["gexp_prot_corr", "Proteomics n. measurements"]],
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

f_enr = mofa.pathway_enrichment(f, views=["proteomics"], genesets=["c5.bp.v7.0.symbols.gmt", "c2.cp.kegg.v7.0.symbols.gmt"])
f_enr = f_enr.set_index("Term|NES")
f_enr["index"] = np.arange(f_enr.shape[0])

gs_proteasome = Enrichment.read_signature(
    f"{DPATH}/pathways/c2.cp.kegg.v7.0.symbols.gmt", "KEGG_PROTEASOME"
)
gs_ribosome = Enrichment.read_signature(
    f"{DPATH}/pathways/c2.cp.kegg.v7.0.symbols.gmt", "KEGG_RIBOSOME"
)
gs_translational = Enrichment.read_signature(
    f"{DPATH}/pathways/c5.bp.v7.0.symbols.gmt",
    "GO_CYTOPLASMIC_TRANSLATIONAL_INITIATION",
)

#
genes_highlight = [
    "KEGG_PROTEASOME",
    "GO_CYTOPLASMIC_TRANSLATIONAL_INITIATION",
    "KEGG_RIBOSOME",
    "GO_TRANSLATIONAL_INITIATION",

    "GO_CRISTAE_FORMATION",
    "KEGG_ECM_RECEPTOR_INTERACTION",
    "GO_INNER_MITOCHONDRIAL_MEMBRANE_ORGANIZATION",
    "GO_NADH_DEHYDROGENASE_COMPLEX_ASSEMBLY",
]
genes_palette = pd.Series(sns.color_palette("tab20c", n_colors=len(genes_highlight)).as_hex(), index=genes_highlight)

_, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=600)

ax.scatter(f_enr["index"], f_enr["nes"], c=GIPlot.PAL_DBGD[2], s=5, linewidths=0)

for g in genes_highlight:
    ax.scatter(
        f_enr.loc[g, "index"],
        f_enr.loc[g, "nes"],
        c=genes_palette[g],
        s=10,
        linewidths=0,
        label=g,
    )

ax.set_xlabel("Rank of gene-sets")
ax.set_ylabel(f"Enrichment score (NES)")
ax.set_title(f"Factor {f[1:]} weights - Proteomics")
ax.legend(frameon=False, prop={"size": 4})
ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

plt.savefig(f"{RPATH}/1.MultiOmics_{f}_ssgsea_waterfall.pdf", bbox_inches="tight")
plt.close("all")

#
grid = GIPlot.gi_regression(f, "gexp_prot_corr", plot_df)
grid.set_axis_labels(f"Factor {f[1:]}", "GExp ~ Prot correlation")
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_gexp_prot_corr_regression.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")

#
grid = GIPlot.gi_regression(f, "Proteomics n. measurements", plot_df)
grid.set_axis_labels(f"Factor {f[1:]}", "# measurements")
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_prot_nmeas_regression.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")

#
col_colors = CrispyPlot.get_palettes(samples, ss).reindex(plot_df.index)
col_colors["gexp_prot_corr"] = list(
    map(matplotlib.colors.rgb2hex, cm.get_cmap("Blues")(plot_df["gexp_prot_corr"]))
)

MOFAPlot.view_heatmap(
    mofa,
    "proteomics",
    f,
    n_features=200,
    col_colors=col_colors,
    title=f"Proteomics heatmap of Factor{f[1:]} top features",
)
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_heatmap_proteomics.png",
    transparent=True,
    bbox_inches="tight",
    dpi=600,
)
plt.close("all")

#
for n, gs in [
    ("Proteasome", gs_proteasome),
    ("Ribosome", gs_ribosome),
    ("Translational initiation", gs_translational),
]:
    df = pd.concat(
        [
            prot.reindex(gs).mean().rename(f"{n}_prot"),
            gexp.reindex(gs).mean().rename(f"{n}_gexp"),
            covariates[["gexp_prot_corr", "Proteomics n. measurements"]],
            mofa.factors[f],
        ],
        axis=1,
    )

    for x_var in [f"{n}_prot", f"{n}_gexp"]:
        for y_var in ["gexp_prot_corr", "Proteomics n. measurements"]:
            ax = GIPlot.gi_continuous_plot(
                x_var, y_var, f, df, cbar_label=f"Factor {f[1:]}"
            )
            ax.set_xlabel(f"{n}\n(mean {x_var.split('_')[1]})")
            ax.set_ylabel(
                f"Transcript ~ Protein correlation\n(Pearsons'r)"
                if y_var == "gexp_prot_corr"
                else "Number of measurements"
            )
            plt.savefig(
                f"{RPATH}/1.MultiOmics_{f}_{n}_{x_var}_{y_var}_continous.pdf",
                transparent=True,
                bbox_inches="tight",
            )
            plt.close("all")

#
sb_samples = set(prot).intersection(prot_obj.broad)
sb_genes = set(prot.index).intersection(prot_obj.broad.index)
sb_corr = pd.DataFrame(
    {
        s: spearmanr(
            prot.loc[sb_genes, s], prot_obj.broad.loc[sb_genes, s], nan_policy="omit"
        )
        for s in sb_samples
    },
    index=["corr", "pval"],
).T
sb_corr["Factor"] = mofa.factors[f].loc[sb_samples]
sb_corr["gexp_prot_corr"] = plot_df["gexp_prot_corr"].reindex(sb_corr.index)

ax = GIPlot.gi_continuous_plot(
    "corr", "gexp_prot_corr", "Factor", sb_corr, cbar_label=f"Factor {f[1:]}"
)
ax.set_xlabel("Same cell line correlation\nwith CCLE proteomics (Spearman's)")
ax.set_ylabel(f"Transcript ~ Protein correlation\n(Pearsons'r)")
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_factor_corr_broad_regression.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")

# Tissue plot
df = pd.concat([
    mofa.factors[["F1", "F2", "F6"]],
    covariates["gexp_prot_corr"],
], axis=1)

ax = GIPlot.gi_continuous_plot("F1", "F2", f, df, cbar_label=f"Factor {f[1:]}")
ax.set_xlabel(f"Factor F1")
ax.set_ylabel(f"Factor F2")
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_F1_F2_continous.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

ax = GIPlot.gi_continuous_plot("F1", "F2", "gexp_prot_corr", df, cbar_label=f"Transcript ~ Protein correlation")
ax.set_xlabel(f"Factor F1")
ax.set_ylabel(f"Factor F2")
plt.savefig(
    f"{RPATH}/1.MultiOmics_gexp_prot_corr_F1_F2_continous.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")
