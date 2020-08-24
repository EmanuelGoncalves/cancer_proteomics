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
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from crispy.Enrichment import Enrichment
from GIPlot import GIPlot
from crispy.MOFA import MOFA
from crispy.DimensionReduction import Dim
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import Proteomics, GeneExpression, CopyNumber, Mobem


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")


if __name__ == "__main__":
    # Data-sets
    #
    cn_obj = CopyNumber()
    gexp_obj = GeneExpression()
    prot_obj = Proteomics()
    mobem_obj = Mobem()

    # Samples
    #
    ss = prot_obj.ss.copy()
    samples = set.intersection(set(prot_obj.get_data()), set(prot_obj.get_data()))
    LOG.info(f"Samples: {len(samples)}")

    # Filter data-sets
    #
    prot = prot_obj.filter(subset=samples)
    LOG.info(f"Proteomics: {prot.shape}")

    prot_broad = prot_obj.broad
    LOG.info(f"Proteomics - Broad: {prot_broad.shape}")

    gexp = gexp_obj.filter(subset=samples)
    LOG.info(f"Transcriptomics: {gexp.shape}")

    cn = cn_obj.filter(subset=samples)
    cn = cn.loc[cn.std(1) > 0]
    LOG.info(f"Copy-Number: {cn.shape}")

    mobem = mobem_obj.filter(subset=samples)
    mobem = mobem.loc[mobem.std(1) > 0]
    LOG.info(f"MOBEM: {mobem.shape}")

    # CPTAC
    #
    cptac = pd.read_csv(f"{RPATH}/merged_cptac_tcga_proteomics.csv.gz", index_col=0)
    cptac_ctype = pd.read_csv(
        f"{RPATH}/merged_cptac_tcga_proteomics_ctype.csv.gz", index_col=0
    ).iloc[:, 0]
    cptac_ctype_pal = cptac_ctype.value_counts()
    cptac_ctype_pal = pd.Series(
        sns.color_palette("tab10").as_hex()[: len(cptac_ctype_pal)],
        index=cptac_ctype_pal.index,
    )

    # Covariates
    #
    s_pg_corr = pd.read_csv(
        f"{RPATH}/SampleProteinTranscript_attenuation.csv.gz", index_col=0
    )

    covariates = pd.concat(
        [
            s_pg_corr["attenuation"].rename("CopyNumberAttenuation"),
            s_pg_corr["gexp_prot_corr"].rename("GeneExpressionAttenuation"),
            s_pg_corr["EMT"],
            s_pg_corr["Proteasome"],
            s_pg_corr["Proteasome_broad"],
            s_pg_corr["TranslationInitiation"],
            s_pg_corr["TranslationInitiation_broad"],
            s_pg_corr["CopyNumberInstability"],
            prot.loc[["CDH1", "VIM"]].T.add_suffix("_prot"),
            gexp.loc[["CDH1", "VIM"]].T.add_suffix("_gexp"),
            pd.get_dummies(prot_obj.ss["media"]),
            pd.get_dummies(prot_obj.ss["growth_properties"]),
            pd.get_dummies(prot_obj.ss["tissue"])[["Haematopoietic and Lymphoid", "Lung"]],
            prot_obj.ss.reindex(
                index=samples, columns=["ploidy", "mutational_burden", "growth", "size"]
            ),
            prot_obj.reps.rename("RepsCorrelation"),
            prot_obj.protein_raw.median().rename("MedianProteomics"),
            prot_obj.broad.median().rename("MedianProteomicsBroad"),
        ],
        axis=1,
    )

    # Regress-out covaraites
    #

    def generate_covariates(
        use_media, use_growth_properties, use_reps_correlation, use_emt_markers
    ):
        covs = []

        if use_media:
            covs.append(pd.get_dummies(prot_obj.ss["media"]))

        if use_growth_properties:
            covs.append(pd.get_dummies(prot_obj.ss["growth_properties"]))

        if use_reps_correlation:
            covs.append(prot_obj.reps.rename("RepsCorrelation"))

        if use_emt_markers:
            covs.append(gexp.loc[["CDH1", "VIM"]].T.add_suffix("_gexp"))

        covs = pd.concat(covs, axis=1, sort=False)
        covs = covs.loc[:, covs.std() > 0]
        covs = covs.dropna().astype(np.float)

        return covs

    def regress_out_covs(
        df,
        use_media,
        use_growth_properties,
        use_reps_correlation,
        use_emt_markers,
        min_observations=100,
    ):
        df_covs = generate_covariates(
            use_media, use_growth_properties, use_reps_correlation, use_emt_markers
        )
        df_covs = df_covs.reindex(df.columns).dropna()

        df_rout = dict()

        for g in df.index:
            if df.loc[g].count() <= min_observations:
                continue

            res = MOFA.lm_residuals(
                df.loc[g, df_covs.index], df_covs, add_intercept=True
            )

            if res is None:
                continue

            df_rout[g] = res

        df_rout = pd.DataFrame(df_rout).T

        return df_rout

    dsets = dict(
        prot=prot.copy(),
        prot_culture=regress_out_covs(prot, True, True, False, False),
        prot_culture_reps=regress_out_covs(prot, True, True, True, False),
        prot_culture_reps_emt=regress_out_covs(prot, True, True, True, True),
        prot_broad=prot_broad.copy(),
        prot_broad_culture=regress_out_covs(prot_broad, True, True, False, False),
        prot_broad_culture_emt=regress_out_covs(prot_broad, True, True, False, True),
        gexp=gexp.copy(),
        cptac=cptac.copy(),
    )

    # Dimension reduction
    #
    dsets_dred = {k: DimReduction.dim_reduction(dsets[k]) for k in dsets}

    for dtype in dsets_dred:
        for ctype in ["pca", "tsne"]:
            ax = DimReduction.plot_dim_reduction(
                dsets_dred[dtype],
                ctype=ctype,
                hue_by=cptac_ctype if dtype is "cptac" else ss["tissue"],
                palette=cptac_ctype_pal
                if dtype is "cptac"
                else CrispyPlot.PAL_TISSUE_2,
            )
            ax.set_title(f"{ctype} - {dtype}")
            plt.savefig(
                f"{RPATH}/DimReduction_{dtype}_{ctype}.pdf",
                bbox_inches="tight",
                transparent=True,
            )
            plt.close("all")

    # PCs correlation
    #
    n_pcs = 30
    pcs_order = DimReduction.pc_labels(n_pcs)

    # Covariates correlation
    dsets_covs = {}
    for dtype in dsets_dred:
        LOG.info(dtype)
        if dtype in ["cptac"]:
            LOG.info("Skip")
            continue

        dsets_covs[dtype] = (
            pd.DataFrame(
                [
                    {
                        **two_vars_correlation(
                            dsets_dred[dtype]["pcs"][pc], covariates[c]
                        ),
                        **dict(dtype=dtype, pc=pc, covariate=c),
                    }
                    for pc in pcs_order
                    for c in covariates
                ]
            )
            .sort_values("pval")
            .dropna()
        )
        # dsets_covs[dtype].to_csv(
        #     f"{RPATH}/DimRed_pcs_covariates_corr_{dtype}.csv.gz",
        #     index=False,
        #     compression="gzip",
        # )

        # Plot
        df_vexp = dsets_dred[dtype]["vexp"][pcs_order]
        df_corr = pd.pivot_table(
            dsets_covs[dtype], index="covariate", columns="pc", values="corr"
        ).loc[covariates.columns, pcs_order]

        f, (axb, axh) = plt.subplots(
            2,
            1,
            sharex="col",
            sharey="row",
            figsize=(n_pcs * 0.225, df_corr.shape[0] * 0.225 + 0.5),
            gridspec_kw=dict(height_ratios=[1, 4]),
        )

        axb.bar(
            np.arange(n_pcs) + 0.5, df_vexp, color=CrispyPlot.PAL_DTRACE[2], linewidth=0
        )
        axb.set_yticks(np.arange(0, df_vexp.max() + 0.05, 0.05))
        axb.set_title(f"Principal component analysis - {dtype}")
        axb.set_ylabel("Total variance")
        # axb.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

        axb_twin = axb.twinx()
        axb_twin.scatter(
            np.arange(n_pcs) + 0.5, df_vexp.cumsum(), c=CrispyPlot.PAL_DTRACE[1], s=6
        )
        axb_twin.plot(
            np.arange(n_pcs) + 0.5,
            df_vexp.cumsum(),
            lw=0.5,
            ls="--",
            c=CrispyPlot.PAL_DTRACE[1],
        )
        axb_twin.set_yticks(np.arange(0, df_vexp.cumsum().max() + 0.1, 0.1))
        axb_twin.set_ylabel("Cumulative variance")

        g = sns.heatmap(
            df_corr,
            cmap="Spectral",
            annot=True,
            cbar=False,
            fmt=".2f",
            linewidths=0.3,
            ax=axh,
            center=0,
            annot_kws={"fontsize": 5},
        )
        axh.set_xlabel("Principal components")
        axh.set_ylabel("")

        plt.subplots_adjust(hspace=0.01)
        plt.savefig(
            f"{RPATH}/DimReduction_pcs_covariates_corr_{dtype}_covariates_heatmap.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

    # PCs correlation
    for x_dtype, y_dtype in [
        ("prot_culture_reps_emt", "prot_broad_culture_emt"),
        ("prot_culture_reps", "prot_broad_culture"),
    ]:
        pcs_corr = pd.DataFrame(
            [
                {
                    **two_vars_correlation(
                        dsets_dred[x_dtype]["pcs"][pc_x],
                        dsets_dred[y_dtype]["pcs"][pc_y],
                    ),
                    **dict(
                        x_dtype=x_dtype,
                        x_dtype_pc=pc_x,
                        x_dtype_vexp=dsets_dred[x_dtype]["vexp"][pc_x],
                        y_dtype=y_dtype,
                        y_dtype_pc=pc_y,
                        y_dtype_vexp=dsets_dred[y_dtype]["vexp"][pc_y],
                    ),
                }
                for pc_x in pcs_order
                for pc_y in pcs_order
            ]
        ).sort_values("pval")
        pcs_corr.to_csv(
            f"{RPATH}/DimReduction_pcs_corr_{x_dtype}_{y_dtype}.csv.gz",
            index=False,
            compression="gzip",
        )

        # PCs clustermap
        plot_df = pd.pivot_table(
            pcs_corr, index="x_dtype_pc", columns="y_dtype_pc", values="corr"
        ).loc[pcs_order, pcs_order]
        plot_df.index.name = x_dtype
        plot_df.columns.name = y_dtype

        fig = sns.clustermap(
            plot_df,
            cmap="Spectral",
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 5},
            linewidth=0.3,
            cbar_pos=None,
            figsize=np.array(plot_df.shape) * 0.225,
        )

        plt.savefig(
            f"{RPATH}/DimReduction_pcs_corr_{x_dtype}_{y_dtype}_clustermap.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

    # PCs regression
    for z_var in ["CopyNumberAttenuation", "GeneExpressionAttenuation"]:
        for x_dtype, x_pc, y_dtype, y_pc in [
            ("prot_culture_reps_emt", "PC1", "prot_broad_culture_emt", "PC1"),
            ("prot_culture_reps_emt", "PC1", "prot_broad_culture_emt", "PC2"),
            ("prot_culture_reps_emt", "PC2", "prot_broad_culture_emt", "PC1"),
            ("prot_culture_reps_emt", "PC2", "prot_broad_culture_emt", "PC2"),
            ("prot_culture_reps", "PC1", "prot_broad_culture", "PC1"),
            ("prot_culture_reps", "PC2", "prot_broad_culture", "PC1"),
        ]:
            plot_df = pd.concat(
                [
                    dsets_dred[x_dtype]["pcs"][x_pc].rename(f"{x_dtype} {x_pc}"),
                    dsets_dred[y_dtype]["pcs"][y_pc].rename(f"{y_dtype} {y_pc}"),
                    covariates[z_var],
                ],
                axis=1,
            ).dropna()

            ax = GIPlot.gi_continuous_plot(
                f"{x_dtype} {x_pc}",
                f"{y_dtype} {y_pc}",
                z_var,
                plot_df,
                plot_reg=True,
                lowess=True,
                mid_point_norm=False,
                cbar_label=z_var,
            )
            ax.set_xlabel(
                f"{x_dtype} {x_pc} ({dsets_dred[x_dtype]['vexp'][x_pc]*100:.1f}%)"
            )
            ax.set_ylabel(
                f"{y_dtype} {y_pc} ({dsets_dred[y_dtype]['vexp'][y_pc]*100:.1f}%)"
            )
            ax.set_title("Proteomics PCs")
            plt.savefig(
                f"{RPATH}/DimReduction_pcs_regression_{x_dtype}_{x_pc}_{y_dtype}_{y_pc}_{z_var}.pdf",
                bbox_inches="tight",
                transparent=True,
            )
            plt.close("all")

    # PCs enrichment
    #
    genesets = [
        "c5.all.v7.1.symbols.gmt",
        "c2.all.v7.1.symbols.gmt",
    ]

    enr_pcs = [
        ("prot", "PC1"),
        ("prot_broad", "PC1"),
        ("prot_culture_reps", "PC1"),
        ("prot_culture_reps", "PC2"),
        ("prot_broad_culture", "PC1"),
        ("prot_culture_reps_emt", "PC1"),
        ("prot_culture_reps_emt", "PC2"),
        ("prot_broad_culture_emt", "PC1"),
        ("prot_broad_culture_emt", "PC2"),
        ("prot_broad_culture_emt", "PC4"),
    ]

    enr_pcs = pd.concat(
        [
            gseapy.ssgsea(
                dsets_dred[dtype]["loadings"].loc[dtype_pc],
                processes=4,
                gene_sets=Enrichment.read_gmt(f"{DPATH}/pathways/{g}"),
                no_plot=True,
            )
            .res2d.assign(geneset=g)
            .assign(dtype=dtype)
            .assign(dtype_pc=dtype_pc)
            .reset_index()
            for dtype, dtype_pc in enr_pcs
            for g in genesets
        ],
        ignore_index=True,
    )
    enr_pcs = enr_pcs.rename(columns={"sample1": "nes"}).sort_values("nes")
    enr_pcs.to_csv(f"{RPATH}/DimReduction_pcs_enr.csv.gz", compression="gzip", index=False)

    # Plot
    enr_pcs_plt = [
        ("prot", "PC1", "prot_broad", "PC1", 0.5),
        ("prot_culture_reps", "PC1", "prot_broad_culture", "PC1", 0.5),
        ("prot_culture_reps", "PC2", "prot_broad_culture", "PC1", 0.5),
        ("prot_culture_reps_emt", "PC1", "prot_broad_culture_emt", "PC2", 0.5),
        ("prot_culture_reps_emt", "PC2", "prot_broad_culture_emt", "PC1", 0.5),
    ]

    for x_dtype, x_pc, y_dtype, y_pc, thres_abs in enr_pcs_plt:
        x_label = f"{x_dtype}_{x_pc}"
        y_label = f"{y_dtype}_{y_pc}"

        plot_df = pd.concat(
            [
                enr_pcs.query(f"(dtype == '{x_dtype}') & (dtype_pc == '{x_pc}')")
                .set_index("Term|NES")["nes"]
                .rename(x_label),
                enr_pcs.query(f"(dtype == '{y_dtype}') & (dtype_pc == '{y_pc}')")
                .set_index("Term|NES")["nes"]
                .rename(y_label),
            ],
            axis=1,
        ).dropna()
        plot_df.index = [i.replace("_", " ") for i in plot_df.index]

        f, ax = plt.subplots(1, 1, figsize=(2, 2))

        ax.scatter(
            plot_df[f"{x_dtype}_{x_pc}"],
            plot_df[f"{y_dtype}_{y_pc}"],
            c=GIPlot.PAL_DBGD[2],
            s=5,
            linewidths=0,
        )

        gs_highlight = plot_df[(plot_df[[x_label, y_label]].abs() > thres_abs).any(1)].sort_values(x_label)
        gs_highlight_dw = gs_highlight.query(f"{x_label} < 0").sort_values(x_label, ascending=False)
        gs_highlight_up = gs_highlight.query(f"{x_label} > 0").sort_values(x_label, ascending=True)
        gs_highlight_pal = pd.Series(
            sns.light_palette(
                "#3182bd", n_colors=len(gs_highlight_dw) + 1, reverse=True
            ).as_hex()[:-1]
            + sns.light_palette(
                "#e6550d", n_colors=len(gs_highlight_up) + 1, reverse=False
            ).as_hex()[1:],
            index=gs_highlight.index,
        )

        for g in gs_highlight.index:
            ax.scatter(
                plot_df.loc[g, x_label],
                plot_df.loc[g, y_label],
                c=gs_highlight_pal[g],
                s=10,
                linewidths=0,
                label=g,
            )

        cor, pval = spearmanr(plot_df[x_label], plot_df[y_label])
        annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
        ax.text(0.98, 0.02, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

        ax.legend(
            frameon=False, prop={"size": 4}, loc="center left", bbox_to_anchor=(1, 0.5)
        )

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

        ax.set_xlabel(
            f"{x_dtype} {x_pc} ({dsets_dred[x_dtype]['vexp'][x_pc] * 100:.1f}%)"
        )
        ax.set_ylabel(
            f"{y_dtype} {y_pc} ({dsets_dred[y_dtype]['vexp'][y_pc] * 100:.1f}%)"
        )
        ax.set_title(f"PCs enrichment scores (NES)")

        plt.savefig(
            f"{RPATH}/DimReduction_pcs_enr_{x_dtype}_{x_pc}_{y_dtype}_{y_pc}.pdf",
            bbox_inches="tight",
        )
        plt.close("all")
