#!/usr/bin/env python
# Copyright (C) 2020 Emanuel Goncalves

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
import matplotlib
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
from crispy.GIPlot import GIPlot
from scripts import two_vars_correlation
from Enrichment import Enrichment
from scipy.stats import spearmanr, pearsonr
from crispy.MOFA import MOFA, MOFAPlot
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    Methylation,
    CopyNumber,
    CRISPR,
    DrugResponse,
    WES,
    Mobem,
)


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")


if __name__ == '__main__':
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
    samples = set.intersection(set(prot_obj.get_data()))
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

    crispr_all = crispr_obj.filter(subset=samples, dtype="merged")
    crispr = crispr_all.loc[crispr_all.std(1) > .1]
    LOG.info(f"CRISPR: {crispr.shape}")

    drespo = drug_obj.filter(subset=samples)
    drespo = drespo.set_index(pd.Series([";".join(map(str, i)) for i in drespo.index]))

    drespo_maxc = drug_obj.maxconcentration.copy()
    drespo_maxc.index = [";".join(map(str, i)) for i in drug_obj.maxconcentration.index]
    drespo_maxc = drespo_maxc.reindex(drespo.index)
    LOG.info(f"Drug response: {drespo.shape}")

    cn_all = cn_obj.filter(subset=samples.intersection(prot_obj.ss.index))
    cn_all = cn_all.loc[cn_all.std(1) > 0]
    cn_all = np.log2(cn_all.divide(prot_obj.ss.loc[cn_all.columns, "ploidy"]) + 1)
    cn = cn_all.loc[cn_all.std(1) > .2]
    cn_inst = cn_obj.genomic_instability()
    LOG.info(f"Copy-Number: {cn.shape}")

    wes = wes_obj.filter(subset=samples, min_events=3, recurrence=True)
    wes = wes.loc[wes.std(1) > 0]
    LOG.info(f"WES: {wes.shape}")

    mobem = mobem_obj.filter(subset=samples)
    mobem = mobem.loc[mobem.std(1) > 0]
    LOG.info(f"MOBEM: {mobem.shape}")

    # Sample Protein ~ Transcript correlation
    #
    s_pg_corr = pd.DataFrame({
        s: two_vars_correlation(prot[s], gexp_all[s]) for s in set(prot).intersection(gexp)},
        index=["corr", "pvalue", "len"],
    ).T

    # Covariates
    #
    covariates = pd.concat(
        [
            s_pg_corr["corr"].rename("Attenuation"),
            cn_inst.rename("Genomic instability"),

            mobem.loc[["TP53_mut", "KRAS_mut", "BRAF_mut"]].T,
            prot.loc[["CDH1", "VIM", "BCL2L1"]].T.add_suffix("_prot"),
            gexp_obj.get_data().loc[["CDH1", "VIM", "MCL1", "BCL2L1"]].T.add_suffix("_gexp"),
            methy.loc[["SLC5A1", "MLH1"]].T.add_suffix("_methy"),
            crispr.loc[["SOX10", "MYCN", "BRAF", "KRAS", "TP63", "EGFR", "FOXA1", "WRN"]].T.add_suffix("_crispr"),
            drespo.loc[["1372;Trametinib;GDSC2", "1190;Gemcitabine;GDSC2", "2106;Uprosertib;GDSC2"]].T,

            pd.get_dummies(crispr_obj.merged_institute),
            pd.get_dummies(prot_obj.ss["media"]),
            pd.get_dummies(prot_obj.ss["msi_status"]),
            pd.get_dummies(prot_obj.ss["growth_properties"]),
            pd.get_dummies(prot_obj.ss["tissue"])[["Haematopoietic and Lymphoid", "Lung"]],
            prot_obj.ss.reindex(index=samples, columns=["ploidy", "mutational_burden", "growth"]),

            prot.count().pipe(np.log2).rename("NProteins"),
            prot_obj.broad.count().pipe(np.log2).rename("NProteinsBroad"),
            prot_obj.reps.rename("RepsCorrelation"),
            prot_obj.protein_raw.median().rename("MedianProteomics"),
            prot_obj.broad.median().rename("MedianProteomicsBroad"),

            drespo.mean().rename("MeanIC50"),
            methy.mean().rename("MeanMethylation"),
        ],
        axis=1,
    )

    # MOFA
    #
    def tissue_class(t):
        if t == "Haematopoietic and Lymphoid":
            return "Haem"
        else:
            return "Other"

    groupby = prot_obj.ss.loc[samples, "tissue"].apply(tissue_class)

    mofa = MOFA(
        views=dict(
            proteomics=prot,
            # proteomics_broad=prot_obj.broad,
            transcriptomics=gexp,
            methylation=methy,
            drespo=drespo,
            # crispr=crispr,
        ),
        groupby=groupby,
        covariates=dict(
            proteomics=covariates[["MedianProteomics", "NProteins", "RepsCorrelation"]],
            # proteomics_broad=covariates[["MedianProteomicsBroad", "NProteinsBroad"]],
            methylation=covariates[["MeanMethylation"]],
            drespo=covariates[["MeanIC50"]],
        ),
        iterations=2000,
        use_overlap=False,
        convergence_mode="fast",
        factors_n=20,
        from_file=f"{RPATH}/MultiOmics.hdf5",
        verbose=2,
    )

    # Factors integrated with other measurements
    #
    n_factors_corr = {}
    for f in mofa.factors:
        n_factors_corr[f] = {}

        for c in covariates:
            fc_samples = list(covariates.reindex(mofa.factors[f].index)[c].dropna().index)
            n_factors_corr[f][c] = pearsonr(mofa.factors[f][fc_samples], covariates[c][fc_samples])[0]
    n_factors_corr = pd.DataFrame(n_factors_corr)

    # Factor clustermap
    MOFAPlot.factors_corr_clustermap(mofa)
    plt.savefig(
        f"{RPATH}/MultiOmics_factors_corr_clustermap.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Variance explained across data-sets
    MOFAPlot.variance_explained_heatmap(mofa)
    plt.savefig(
        f"{RPATH}/MultiOmics_factors_rsquared_heatmap.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Covairates correlation heatmap
    MOFAPlot.covariates_heatmap(n_factors_corr, mofa, prot_obj.ss["model_type"])
    plt.savefig(
        f"{RPATH}/MultiOmics_factors_covariates_clustermap.pdf",
        bbox_inches="tight",
    )
    plt.close("all")

    # Factor 1 and 2
    #
    f_x, f_y = "F1", "F2"

    plot_df = pd.concat(
        [
            mofa.factors[[f_x, f_y]],
            gexp.loc[["CDH1", "VIM"]].T.add_suffix("_transcriptomics"),
            prot.loc[["CDH1", "VIM"]].T.add_suffix("_proteomics"),
            prot_obj.ss["tissue"],
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
                f"{RPATH}/MultiOmics_{f}_{v}_regression.pdf",
                bbox_inches="tight",
                transparent=True,
            )
            plt.close("all")

    # Tissue plot
    ax = GIPlot.gi_tissue_plot(f_x, f_y, plot_df, plot_reg=False)
    ax.set_xlabel(f"Factor {f_x[1:]}")
    ax.set_ylabel(f"Factor {f_y[1:]}")
    plt.savefig(
        f"{RPATH}/MultiOmics_{f_x}_{f_y}_tissue_plot.pdf",
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
            f"{RPATH}/MultiOmics_{f_x}_{f_y}_continous_{z}.pdf",
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
