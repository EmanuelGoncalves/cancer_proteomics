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
methy_obj = Methylation()
gexp_obj = GeneExpression()

wes_obj = WES()
cn_obj = CopyNumber()
mobem_obj = Mobem()

crispr_obj = CRISPR()
drug_obj = DrugResponse()

metab_obj = Metabolomics()


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

gexp_all = gexp_obj.filter(subset=samples)
gexp = gexp_all.loc[gexp_all.std(1) > 1]
LOG.info(f"Transcriptomics: {gexp.shape}")

methy_all = methy_obj.filter(subset=samples)
methy = methy_all.loc[methy_all.std(1) > 0.05]
LOG.info(f"Methylation: {methy.shape}")

crispr = crispr_obj.filter(subset=samples, dtype="merged")
LOG.info(f"CRISPR: {crispr.shape}")

drespo = drug_obj.filter(subset=samples)
drespo = drespo.set_index(pd.Series([";".join(map(str, i)) for i in drespo.index]))

drespo_maxc = drug_obj.maxconcentration.copy()
drespo_maxc.index = [";".join(map(str, i)) for i in drug_obj.maxconcentration.index]
drespo_maxc = drespo_maxc.reindex(drespo.index)
LOG.info(f"Drug response: {drespo.shape}")

cn = cn_obj.filter(subset=samples.intersection(ss.index))
cn = cn.loc[cn.std(1) > 0]
cn = np.log2(cn.divide(ss.loc[cn.columns, "ploidy"]) + 1)
cn_inst = cn_obj.genomic_instability()
LOG.info(f"Copy-Number: {cn.shape}")

wes = wes_obj.filter(subset=samples, min_events=3, recurrence=True)
wes = wes.loc[wes.std(1) > 0]
LOG.info(f"WES: {wes.shape}")

mobem = mobem_obj.filter(subset=samples)
mobem = mobem.loc[mobem.std(1) > 0]
LOG.info(f"MOBEM: {mobem.shape}")

metab = metab_obj.filter(subset=samples)
metab = metab.loc[metab.std(1) > 0]
LOG.info(f"Metabolomics: {metab.shape}")


# Sample Protein ~ Transcript correlation
#

def sample_corr(var1, var2, idx_set):
    return spearmanr(
        var1.reindex(index=idx_set),
        var2.reindex(index=idx_set),
        nan_policy="omit",
    )


s_pg_corr = pd.DataFrame({
    s: sample_corr(prot[s], gexp_all[s], set(prot.index).intersection(gexp_all.index)) for s in set(prot).intersection(gexp)},
    index=["corr", "pvalue"],
).T

s_pg_corr_broad = pd.DataFrame({
    s: sample_corr(prot_obj.broad[s], gexp[s], set(prot_obj.broad.index).intersection(gexp_all.index)) for s in set(prot_obj.broad).intersection(gexp)},
    index=["corr", "pvalue"],
).T

s_pg_corr_broad_ov = pd.DataFrame({
    s: sample_corr(prot_obj.broad[s], gexp[s], set(prot_obj.broad.index).intersection(gexp_all.index).intersection(prot.index)) for s in set(prot_obj.broad).intersection(gexp)},
    index=["corr", "pvalue"],
).T

bsc_corr = pd.DataFrame({
    s: sample_corr(prot[s], prot_obj.broad[s], set(prot.index).intersection(prot_obj.broad.index)) for s in set(prot).intersection(prot_obj.broad)},
    index=["corr", "pval"],
).T


# Covariates
#

covariates = pd.concat(
    [
        s_pg_corr["corr"].rename("GExpProtCorrSanger&CMRI"),
        s_pg_corr_broad["corr"].rename("GExpProtCorrBroad"),
        bsc_corr["corr"].rename("SamplesOverlapCorr"),
        cn_inst.rename("Genomic instability"),

        mobem.loc[["TP53_mut", "KRAS_mut", "BRAF_mut", "APC_mut", "EWSR1.FLI1_mut", "MYC_mut", "BCR.ABL_mut", "VHL_mut", "STAG2_mut", "gain.cnaPANCAN344..MYCN.", "gain.cnaPANCAN59..CCND1.CTTN.", "loss.cnaPANCAN23..SMAD4.", "DNMT3A_mut"]].T,
        prot.loc[["CDH1", "VIM", "BCL2L1"]].T.add_suffix("_proteomics"),
        gexp_obj.get_data().loc[["CDH1", "VIM", "MCL1", "BCL2L1"]].T.add_suffix("_transcriptomics"),
        methy.loc[["SLC5A1", "MLH1"]].T.add_suffix("_methylation"),
        crispr.loc[["SOX10", "SOX9", "MITF", "MYCN", "BRAF", "KRAS", "TP63", "CTNNB1", "STX4", "FERMT2", "GPX4", "EGFR", "FOXA1", "HNF1A", "GATA3", "TTC7A", "WRN"]].T.add_suffix("_crispr"),
        drespo.loc[["1804;Acetalax;GDSC2", "1372;Trametinib;GDSC2", "1190;Gemcitabine;GDSC2", "1819;Docetaxel;GDSC2", "2106;Uprosertib;GDSC2"]].T,

        pd.get_dummies(ss["media"]),
        pd.get_dummies(ss["msi_status"]),
        pd.get_dummies(ss["growth_properties"]),
        pd.get_dummies(ss["tissue"])["Haematopoietic and Lymphoid"],
        ss.reindex(index=samples, columns=["ploidy", "mutational_burden", "growth"]),

        prot.count().pipe(np.log2).rename("NProteinsSanger&CMRI"),
        prot_obj.broad.count().pipe(np.log2).rename("NProteinsBroad"),
        prot_obj.reps.rename("RepsCorrelationSanger&CMRI"),
        prot_obj.protein_raw.median().rename("Global proteomics Sanger&CMRI"),
        prot_obj.broad.median().rename("Global proteomics Broad"),

        drespo.mean().rename("Mean IC50"),
        methy.mean().rename("Global methylation"),

    ],
    axis=1,
)


# MOFA
#

use_covs = True

groupby = ss.loc[samples, "tissue"].apply(
    lambda v: "Haem" if v == "Haematopoietic and Lymphoid" else "Other"
)

mofa = MOFA(
    views=dict(proteomics=prot, transcriptomics=gexp, methylation=methy, drespo=drespo),
    groupby=groupby,
    covariates=dict(
        proteomics=covariates[["NProteinsSanger&CMRI", "RepsCorrelationSanger&CMRI", "Global proteomics Sanger&CMRI"]],
        methylation=covariates[["Global methylation"]],
        drespo=covariates[["Mean IC50"]],
    ) if use_covs else None,
    iterations=2000,
    use_overlap=True,
    convergence_mode="slow",
    factors_n=10,
    from_file=f"{RPATH}/1.MultiOmics_Sanger&CMRI{'' if use_covs else '_no_covariates'}.hdf5",
)

groupby_broad = ss.loc[set(prot_obj.broad).intersection(gexp), "tissue"].apply(
    lambda v: "Haem" if v == "Haematopoietic and Lymphoid" else "Other"
)

mofa_broad = MOFA(
    views=dict(
        proteomics=prot_obj.broad[groupby_broad.index],
        transcriptomics=gexp[groupby_broad.index],
        methylation=methy[groupby_broad.index],
        drespo=drespo[groupby_broad.index],
    ),
    groupby=groupby_broad,
    covariates=dict(
        proteomics=covariates[["NProteinsBroad", "Global proteomics Broad"]],
        methylation=covariates[["Global methylation"]],
        drespo=covariates[["Mean IC50"]],
    ) if use_covs else None,
    iterations=2000,
    use_overlap=True,
    convergence_mode="slow",
    factors_n=10,
    from_file=f"{RPATH}/1.MultiOmics_Broad{'' if use_covs else '_no_covariates'}.hdf5",
)


# Factors data-frame integrated with other measurements
#

# n, n_factors = "Sanger&CMRI", mofa
for n, n_factors in [("Sanger&CMRI", mofa), ("Broad", mofa_broad)]:
    n_factors_corr = {}

    for f in n_factors.factors:
        n_factors_corr[f] = {}

        for c in covariates:
            fc_samples = list(covariates.reindex(n_factors.factors[f].index)[c].dropna().index)
            n_factors_corr[f][c] = pearsonr(n_factors.factors[f][fc_samples], covariates[c][fc_samples])[0]

    n_factors_corr = pd.DataFrame(n_factors_corr)

    # Factor clustermap
    MOFAPlot.factors_corr_clustermap(n_factors)
    plt.savefig(
        f"{RPATH}/1.MultiOmics_factors_corr_clustermap_{n}{'' if use_covs else '_no_covariates'}.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Variance explained across data-sets
    MOFAPlot.variance_explained_heatmap(n_factors)
    plt.savefig(
        f"{RPATH}/1.MultiOmics_factors_rsquared_heatmap_{n}{'' if use_covs else '_no_covariates'}.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Covairates correlation heatmap
    MOFAPlot.covariates_heatmap(n_factors_corr, n_factors, ss["model_type"])
    plt.savefig(
        f"{RPATH}/1.MultiOmics_factors_covariates_clustermap_{n}{'' if use_covs else '_no_covariates'}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


# Factor association analysis
#

prot_covs = ["NProteinsSanger&CMRI", "RepsCorrelationSanger&CMRI"]

# CRISPR
crispr_covs = LMModels.define_covariates(
    institute=crispr_obj.merged_institute,
    medium=True,
    cancertype=False,
    tissuetype=False,
    mburden=False,
    ploidy=False,
)
crispr_covs = pd.concat([crispr_covs, covariates[prot_covs]], axis=1).reindex(crispr.columns).dropna()

lmm_crispr = LModel(
    Y=crispr[crispr_covs.index].T, X=mofa.factors.loc[crispr_covs.index], M=crispr_covs
).fit_matrix()
print(lmm_crispr.query("fdr < 0.1").head(60))


crispr_covs_broad = LMModels.define_covariates(
    institute=crispr_obj.merged_institute,
    medium=True,
    cancertype=False,
    tissuetype=False,
    mburden=False,
    ploidy=False,
)
crispr_covs_broad = pd.concat([crispr_covs_broad, covariates[["NProteinsBroad"]]], axis=1).reindex(mofa_broad.factors.index).dropna()

lmm_crispr_broad = LModel(
    Y=crispr[crispr_covs_broad.index].T, X=mofa_broad.factors.loc[crispr_covs_broad.index], M=crispr_covs_broad
).fit_matrix()
print(lmm_crispr_broad.query("fdr < 0.1").head(60))

# MOBEMS
mobems_covs = LMModels.define_covariates(
    institute=False,
    medium=True,
    cancertype=False,
    tissuetype=False,
    mburden=False,
    ploidy=False,
).reindex(mobem.columns)
mobems_covs = pd.concat([mobems_covs, covariates[prot_covs]], axis=1).reindex(mofa.factors.index).dropna()

lmm_mobems = LModel(
    Y=mobem[mobems_covs.index].T, X=mofa.factors.loc[mobems_covs.index], M=mobems_covs
).fit_matrix()
print(lmm_mobems.query("fdr < 0.1").head(60))

mobems_covs_broad = LMModels.define_covariates(
    institute=False,
    medium=True,
    cancertype=False,
    tissuetype=False,
    mburden=False,
    ploidy=False,
).reindex(mobem.columns)
mobems_covs_broad = pd.concat([mobems_covs_broad, covariates[["NProteinsBroad"]]], axis=1).reindex(mofa_broad.factors.index).dropna()

lmm_mobems_broad = LModel(
    Y=mobem[mobems_covs_broad.index].T, X=mofa_broad.factors.loc[mobems_covs_broad.index], M=mobems_covs_broad
).fit_matrix()
print(lmm_mobems.query("fdr < 0.1").head(60))


# Factor 1 and 2
#

n, n_factors = "Sanger&CMRI", mofa

f_x, f_y = "F1", "F2"

plot_df = pd.concat(
    [
        n_factors.factors[[f_x, f_y]],
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_transcriptomics"),
        prot.loc[["CDH1", "VIM"]].T.add_suffix("_proteomics"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

# Regression plots
for f in [f_x, f_y]:
    for v in ["CDH1_transcriptomics", "CDH1_proteomics", "VIM_transcriptomics", "VIM_proteomics"]:
        grid = GIPlot.gi_regression(f, v, plot_df)
        grid.set_axis_labels(f"Factor {f[1:]}", v.replace("_", " "))
        plt.savefig(
            f"{RPATH}/1.MultiOmics_{f}_{v}_regression_{n}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

# Tissue plot
ax = GIPlot.gi_tissue_plot(f_x, f_y, plot_df, plot_reg=False)
ax.set_xlabel(f"Factor {f_x[1:]}")
ax.set_ylabel(f"Factor {f_y[1:]}")
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f_x}_{f_y}_tissue_plot_{n}.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

# Continous annotation
for z in [
    "VIM_proteomics",
]:
    ax = GIPlot.gi_continuous_plot(f_x, f_y, z, plot_df, cbar_label=z.replace("_", " "))
    ax.set_xlabel(f"Factor {f_x[1:]}")
    ax.set_ylabel(f"Factor {f_y[1:]}")
    plt.savefig(
        f"{RPATH}/1.MultiOmics_{f_x}_{f_y}_continous_{z}_{n}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# Factor 6
#
f, f_broad = "F6", "F7"
f_name, f_broad_name = f"Factor {f[1:]} Sanger&CMRI", f"Factor {f_broad[1:]} Broad"

factor_df = pd.concat(
    [
        mofa.factors[f].rename(f_name),
        mofa_broad.factors[f_broad].rename(f_broad_name),
        bsc_corr["corr"].rename("Sanger&CMRI_Broad_Corr"),
        covariates[
            [
                "NProteinsSanger&CMRI",
                "NProteinsBroad",
                "GExpProtCorrSanger&CMRI",
                "GExpProtCorrBroad",
            ]
        ],
        s_pg_corr_broad_ov["corr"].rename("GExpProtCorrBroadOverlap"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna(subset=[f_name])

# Protein attenuation correlation between institutes
for x_var, y_var in [
    ("GExpProtCorrSanger&CMRI", "GExpProtCorrBroad"),
    ("GExpProtCorrSanger&CMRI", "GExpProtCorrBroadOverlap"),
    (f_name, f_broad_name),
]:
    plot_df = factor_df[[x_var, y_var]].dropna()

    grid = GIPlot.gi_regression(x_var, y_var, plot_df)
    grid.set_axis_labels(x_var, y_var)
    plt.savefig(
        f"{RPATH}/1.MultiOmics_{f}_{x_var}_{y_var}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

# Factor correlation
for x_var in [f_name, f_broad_name]:
    for y_var in [
        "NProteinsSanger&CMRI",
        "NProteinsBroad",
        "GExpProtCorrSanger&CMRI",
        "GExpProtCorrBroad",
    ]:
        grid = GIPlot.gi_regression(
            x_var, y_var, factor_df.dropna(subset=[x_var, y_var])
        )
        plt.savefig(
            f"{RPATH}/1.MultiOmics_{f}_corr_{x_var}_{y_var}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")


# Ovarlapping samples correlation vs attenuation
plot_df = pd.concat(
    [
        bsc_corr,
        factor_df[[f_name, f_broad_name, "GExpProtCorrSanger&CMRI", "GExpProtCorrBroad"]],
    ],
    axis=1,
)

for y_var, z_var in [
    ("GExpProtCorrSanger&CMRI", f_name),
    ("GExpProtCorrBroad", f_broad_name),
]:
    ax = GIPlot.gi_continuous_plot(
        "corr", y_var, z_var, plot_df, cbar_label=z_var, plot_reg=True
    )
    ax.set_xlabel("Correlation Sanger&CMRI and Broad\n(same cell line)")
    plt.savefig(
        f"{RPATH}/1.MultiOmics_{f}_sample_corr_{y_var}_{z_var}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

# Pathway enrichment heatmap
for n, n_factors, factor in [("Sanger&CMRI", mofa, f), ("Broad", mofa_broad, f_broad)]:
    col_colors = CrispyPlot.get_palettes(samples, ss).reindex(n_factors.factors.index)

    c_values = factor_df.reindex(col_colors.index)[f"GExpProtCorr{n}"]
    col_colors[f"GExpProtCorr{n}"] = list(
        map(matplotlib.colors.rgb2hex, mpl.cm.get_cmap("Blues")(c_values))
    )

    n_features = 500

    for v in ["proteomics", "transcriptomics"]:
        MOFAPlot.factor_weights_scatter(n_factors, v, factor, n_features=3000, label_features=False)
        plt.savefig(
            f"{RPATH}/1.MultiOmics_{f}_top_features_{v}_{n}.pdf",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")

    MOFAPlot.view_heatmap(
        n_factors,
        "proteomics",
        factor,
        center=False,
        standard_scale=0,
        n_features=n_features,
        col_colors=col_colors,
        title=f"Proteomics {n} heatmap of Factor{factor[1:]} top {n_features} features",
    )
    plt.savefig(
        f"{RPATH}/1.MultiOmics_{f}_heatmap_proteomics_{n}.png",
        transparent=True,
        bbox_inches="tight",
        dpi=600,
    )
    plt.close("all")

# Pathway enrichment
enr_views = ["transcriptomics", "proteomics"]
factor_enr = mofa.pathway_enrichment(f, views=enr_views)
factor_enr_broad = mofa_broad.pathway_enrichment(f_broad, views=enr_views)

# Plot
for v in enr_views:
    plot_df = pd.concat([
        factor_enr.query(f"view == '{v}'").set_index("Term|NES")["nes"].rename("Sanger&CMRI"),
        factor_enr_broad.query(f"view == '{v}'").set_index("Term|NES")["nes"].rename("Broad"),
    ], axis=1).dropna()
    plot_df.to_csv(f"{RPATH}/1.MultiOmics_{f}_gseapy_{v}.csv")
    plot_df.index = [" ".join(i.split("_")) for i in plot_df.index]

    gs_dw = plot_df[(plot_df["Sanger&CMRI"] < -0.25) & (plot_df["Broad"] < -0.3)].sort_values("Sanger&CMRI")
    gs_up = plot_df[(plot_df["Sanger&CMRI"] > 0.4) & (plot_df["Broad"] > 0.4)].sort_values("Sanger&CMRI", ascending=False)
    gs_highlight = list(gs_up.index) + list(gs_dw.index)

    gs_palette = pd.Series(
        sns.light_palette("#3182bd", n_colors=len(gs_up) + 1, reverse=True).as_hex()[:-1]
        + sns.light_palette("#e6550d", n_colors=len(gs_dw) + 1, reverse=True).as_hex()[:-1],
        index=gs_highlight,
    )

    _, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

    ax.scatter(
        plot_df["Broad"], plot_df["Sanger&CMRI"], c=GIPlot.PAL_DBGD[2], s=5, linewidths=0
    )

    for g in gs_highlight:
        ax.scatter(
            plot_df.loc[g, "Broad"],
            plot_df.loc[g, "Sanger&CMRI"],
            c=gs_palette[g],
            s=10,
            linewidths=0,
            label=g,
        )

    cor, pval = spearmanr(plot_df["Broad"], plot_df["Sanger&CMRI"])
    annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
    ax.text(0.98, 0.02, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    ax.set_xlabel(f"Broad")
    ax.set_ylabel("Sanger&CMRI")
    ax.set_title(f"Factor {f[1:]} Proteomics weights enrichment score (NES)")

    ax.legend(frameon=False, prop={"size": 4}, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.savefig(f"{RPATH}/1.MultiOmics_{f}_ssgsea_enrichments_{v}.pdf", bbox_inches="tight")
    plt.close("all")

# Plot genesets mean values
for gs in [
    "KEGG_RIBOSOME",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "GO_PROTEASOME_ACCESSORY_COMPLEX",
    "KEGG_PENTOSE_PHOSPHATE_PATHWAY",
    "GO_TRANSLATION_INITIATION_FACTOR_ACTIVITY",
]:
    gs_genes = Enrichment.signature(gs)

    y_vars = ["GExpProtCorrSanger&CMRI", "GExpProtCorrBroad"]

    gs_df = pd.concat(
        [
            prot.reindex(gs_genes).mean().rename(f"Proteomics Sanger&CMRI"),
            prot_obj.broad.reindex(gs_genes).mean().rename(f"Proteomics Broad"),
            gexp.reindex(gs_genes).mean().rename(f"Transcriptomics"),
            factor_df[y_vars + [f_name, f_broad_name]],
        ],
        axis=1,
    )

    for x_var in ["Proteomics Sanger&CMRI", "Proteomics Broad", "Transcriptomics"]:
        for y_var in y_vars:
            z_var = f_broad_name if "Broad" in y_var else f_name
            ax = GIPlot.gi_continuous_plot(
                x_var,
                y_var,
                z_var,
                gs_df.dropna(subset=[x_var, y_var, z_var]),
                cbar_label=z_var,
                plot_reg=True,
            )
            ax.set_xlabel(f"{x_var}")
            ax.set_ylabel(f"{y_var}")
            ax.set_title(gs)
            plt.savefig(
                f"{RPATH}/1.MultiOmics_{f}_{gs}_{x_var}_{y_var}_continous.pdf",
                transparent=True,
                bbox_inches="tight",
            )
            plt.close("all")
