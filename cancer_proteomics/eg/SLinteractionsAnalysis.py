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
import sys
import gseapy
import logging
import argparse
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.Utils import Utils
from crispy.GIPlot import GIPlot
from itertools import zip_longest
from adjustText import adjust_text
from crispy.MOFA import MOFA, MOFAPlot
from sklearn.metrics.ranking import auc
from crispy.Enrichment import Enrichment
from scipy.stats import pearsonr, spearmanr, skew
from eg.CProtUtils import two_vars_correlation
from sklearn.preprocessing import MinMaxScaler
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    CRISPR,
    CORUM,
    Sample,
    DPATH,
    BioGRID,
    HuRI,
    PPI,
    Mobem,
)


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")
TPATH = pkg_resources.resource_filename("tables", "/")


if __name__ == "__main__":
    # Data-sets
    #
    prot_obj = Proteomics()
    prot = prot_obj.filter()
    LOG.info(f"Proteomics: {prot.shape}")

    gexp_obj = GeneExpression()
    gexp = gexp_obj.filter(subset=list(prot))
    LOG.info(f"Transcriptomics: {gexp.shape}")

    crispr_obj = CRISPR()
    crispr = crispr_obj.filter(subset=list(prot))
    LOG.info(f"CRISPR: {crispr.shape}")

    # CORUM + BioGRID
    #
    corum_db = CORUM()
    biogrid_db = BioGRID()
    huri_db = HuRI()

    # Gene sets
    #
    cgenes = pd.read_csv(f"{DPATH}/cancer_genes_latest.csv.gz")
    cgenes = list(set(cgenes["gene_symbol"]))

    patt = pd.read_csv(f"{RPATH}/ProteinTranscript_attenuation.csv.gz", index_col=0)
    patt_low = list(patt.query("cluster == 'Low'").index)
    patt_high = list(patt.query("cluster == 'High'").index)

    # Subtypes
    #
    breast_subtypes = pd.read_csv(f"{DPATH}/breast_subtypes.txt", sep="\t").set_index(
        "model_id"
    )

    # ML scores
    #
    ml_files = dict(
        prot="scores_202005271320_ruv_min.csv",
        mofa="scores_202005291155_mofa.csv",
        rna="scores_202005302251_rna.csv",
    )
    ml_dfs = {f: pd.read_csv(f"{TPATH}/{ml_files[f]}", index_col=0) for f in ml_files}
    ml_scores = pd.DataFrame({f: ml_dfs[f]["val_score"] for f in ml_dfs})

    ml_fimpor = dict(
        prot=pd.read_csv(
            f"{TPATH}/feature_importance_202005271320_ruv_min.csv.gz", index_col=0
        )
    )

    # CRISPR storngly selective
    #
    crispr_selective = pd.concat(
        [
            crispr.apply(skew, axis=1).rename("skew"),
            crispr.median(1).rename("median"),
            (crispr < -0.5).sum(1).rename("nsamples"),
            crispr[crispr < -0.5].median(1).rename("dependency").abs(),
            ml_scores,
        ],
        axis=1,
    )

    crispr_selective_set = set(crispr_selective.query("skew < -3").index)

    # Scatter
    grid = GIPlot.gi_regression(
        "skew",
        "median",
        crispr_selective,
        size="dependency",
        plot_reg=False,
        plot_annot=False,
    )

    grid.ax_joint.axvline(-3, c=GIPlot.PAL_DTRACE[1], lw=0.3, ls="--")
    g_highlight_df = crispr_selective.query("skew < -3").sort_values("skew").head(15)
    labels = [
        grid.ax_joint.text(row["skew"], row["median"], i, color="k", fontsize=4)
        for i, row in g_highlight_df.iterrows()
    ]
    adjust_text(
        labels,
        arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
        ax=grid.ax_joint,
    )
    grid.set_axis_labels("Skewness", "Median")
    grid.ax_marg_x.set_title("CRISPR selective dependencies")
    plt.savefig(
        f"{RPATH}/SLinteractions_CRISPR_selective_dependencies.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Scatter with ML scores
    grid = GIPlot.gi_regression(
        "skew",
        "prot",
        crispr_selective,
        size="dependency",
        size_legend_loc=3,
        plot_reg=False,
        plot_annot=False,
    )

    x_thres, y_thres = -3, 0.4
    grid.ax_joint.axvline(x_thres, c=GIPlot.PAL_DTRACE[1], lw=0.3, ls="--")
    grid.ax_joint.axhline(y_thres, c=GIPlot.PAL_DTRACE[1], lw=0.3, ls="--")

    g_highlight_df = crispr_selective.query(f"(skew < {x_thres}) & (prot > {y_thres})").sort_values("skew")
    labels = [
        grid.ax_joint.text(row["skew"], row["prot"], i, color="k", fontsize=4)
        for i, row in g_highlight_df.iterrows()
    ]
    adjust_text(
        labels,
        arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
        ax=grid.ax_joint,
    )

    grid.set_axis_labels("Skewness", "ML r-squared")
    grid.ax_marg_x.set_title("CRISPR selective dependencies")
    plt.savefig(
        f"{RPATH}/SLinteractions_CRISPR_selective_dependencies_rsquared.pdf",
        bbox_inches="tight",
    )
    plt.close("all")

    # Linear regression scores
    #
    sl_file = "lm_sklearn_degr_crispr"
    sl_lm = pd.read_csv(f"{RPATH}/{sl_file}.csv.gz")

    # CORUM
    sl_lm["corum"] = [
        int((p1, p2) in corum_db.db_melt_symbol) for p1, p2 in sl_lm[["y_id", "x_id"]].values
    ]

    # BioGRID
    sl_lm["biogrid"] = [
        int((p1, p2) in biogrid_db.biogrid) for p1, p2 in sl_lm[["y_id", "x_id"]].values
    ]

    # HuRI
    sl_lm["huri"] = [
        int((p1, p2) in huri_db.huri) for p1, p2 in sl_lm[["y_id", "x_id"]].values
    ]

    # Attenuated protein
    sl_lm["attenuated"] = sl_lm["x_id"].isin(patt_high).astype(int)

    # Strongly selective CRISPR
    sl_lm["crispr_selective"] = sl_lm["y_id"].isin(crispr_selective_set).astype(int)

    # R-squared
    sl_lm["r2"] = ml_scores.loc[sl_lm["y_id"], "prot"].values

    # Export annotated table
    sl_lm.to_csv(f"{RPATH}/{sl_file}_annotated.csv.gz", compression="gzip", index=False)
    # sl_lm = pd.read_csv(f"{RPATH}/{sl_file}_annotated.csv.gz")

    # Feature importance matrix
    sl_lm_fimpor = dict(
        prot=pd.pivot_table(
            sl_lm, index="x_id", columns="y_id", values="beta", fill_value=np.nan
        )
    )

    # ML + LM
    #
    nassoc_df = pd.concat(
        [ml_scores, sl_lm.groupby("y_id")["fdr"].min().rename("nassoc")],
        axis=1,
    ).fillna(0)

    # LM number of associations versus ML R2
    for y_var in ml_scores:
        g = GIPlot.gi_regression("nassoc", y_var, nassoc_df, lowess=True)
        g.set_axis_labels("Number of associations (LM)", f"Score (ML {y_var})")

        plt.savefig(
            f"{RPATH}/SLinteractions_ml_nassoc_{y_var}.pdf", bbox_inches="tight"
        )
        plt.close("all")

    # Feature importance
    for dtype in ml_fimpor:
        # Get ML features weights
        dtype_df = ml_fimpor[dtype]
        dtype_df_lm = sl_lm_fimpor[dtype]

        # Correlate feature weights
        fimpor_corr = pd.Series(
            {
                p: two_vars_correlation(dtype_df[p], dtype_df_lm[p].abs())["corr"]
                for p in set(dtype_df_lm).intersection(dtype_df)
            }
        )

        # Plot
        nassoc_df_fimpo = pd.concat([nassoc_df, fimpor_corr.rename("feature")], axis=1)

        g = GIPlot.gi_continuous_plot(
            "nassoc",
            dtype,
            "feature",
            nassoc_df_fimpo,
            lowess=True,
            cbar_label="Feature importance correlation",
        )
        g.set_xlabel("Number of associations (LM)")
        g.set_ylabel(f"Score (ML {dtype})")

        plt.savefig(
            f"{RPATH}/SLinteractions_ml_nassoc_{dtype}_fimpo.pdf", bbox_inches="tight"
        )
        plt.close("all")

    # Volcano
    #
    plot_df = sl_lm.query("fdr < .1")

    s_transform = MinMaxScaler(feature_range=[1, 10])

    _, ax = plt.subplots(1, 1, figsize=(4.5, 2.5), dpi=600)

    for t, df in plot_df.groupby("ppi"):
        sc = ax.scatter(
            -np.log10(df["pval"]),
            df["beta"],
            s=s_transform.fit_transform(df[["n"]]),
            color=GIPlot.PPI_PAL[t],
            label=t,
            lw=0.0,
            alpha=0.5,
        )

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

    legend1 = ax.legend(
        frameon=False, prop={"size": 4}, title="PPI distance", loc="lower right"
    )
    legend1.get_title().set_fontsize("4")
    ax.add_artist(legend1)

    handles, labels = sc.legend_elements(
        prop="sizes",
        num=8,
        func=lambda x: s_transform.inverse_transform(np.array(x).reshape(-1, 1))[:, 0],
    )
    legend2 = (
        ax.legend(
            handles,
            labels,
            loc="upper right",
            title="# samples",
            frameon=False,
            prop={"size": 4},
        )
        .get_title()
        .set_fontsize("4")
    )

    ax.set_ylabel("Effect size (beta)")
    ax.set_xlabel("Association p-value (-log10)")
    ax.set_title("CRISPR ~ Protein associations")

    plt.savefig(
        f"{RPATH}/SLinteractions_volcano.png", transparent=True, bbox_inches="tight"
    )
    plt.close("all")

    # CRISPR Hits
    #
    hits = sl_lm.query("(fdr < .1) & (r2 > 0.4)")

    # Top associations
    #
    for gene in ["FOXA1", "PAX8", "PAX5", "BCL2", "TTC7A", "MYCN"]:
        plot_df = hits.query(f"y_id == '{gene}'").head(10).reset_index(drop=True).reset_index()
        plot_df = plot_df.assign(logpval=-np.log10(plot_df["pval"]).values)

        fig, ax = plt.subplots(1, 1, figsize=(plot_df.shape[0] * .2, 1.5))

        for t, df in plot_df.groupby("ppi"):
            ax.bar(
                df["index"].values,
                df["logpval"].values,
                color=GIPlot.PPI_PAL[t],
                align="center",
                zorder=5,
                linewidth=0,
            )

        for g, p in plot_df[["x_id", "index"]].values:
            ax.text(
                p,
                0.1,
                g,
                ha="center",
                va="bottom",
                fontsize=6,
                zorder=10,
                rotation="vertical",
                color="white",
            )

        for x, y, t, b in plot_df[["index", "logpval", "ppi", "beta"]].values:
            c = GIPlot.PAL_DTRACE[0] if t == "T" else GIPlot.PAL_DTRACE[2]

            ax.text(x, y + 0.25, t, color=c, ha="center", fontsize=6, zorder=10)
            ax.text(
                x,
                -0.5,
                f"{b:.1f}",
                color=c,
                ha="center",
                va="top",
                fontsize=6,
                rotation="vertical",
                zorder=10,
            )

        ax.set_title(f"{gene} (R-squared={plot_df['r2'].max():.2f})")
        plt.ylabel(f"Linear regressions\n(p-value log10)")

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
        ax.axes.get_xaxis().set_ticks([])

        plt.savefig(
            f"{RPATH}/SLinteractions_top_associations_{gene}.pdf", bbox_inches="tight"
        )
        plt.close("all")

    # Associations
    #
    gi_pairs = [
        ("HNRNPH1", "HNRNPH1"),
        ("PRKAR1A", "PRKAR1A"),
        ("BSG", "FOXA1"),
        ("LMNA", "PAX5"),
    ]

    for p, c in gi_pairs:
        # p, c = "CD44", "MYCN"
        plot_df = pd.concat(
            [
                crispr.loc[[c]].T.add_suffix("_crispr"),
                prot.loc[[p]].T.add_suffix("_prot"),
                gexp.loc[[p]].T.add_suffix("_gexp"),
                prot_obj.ss["tissue"],
            ],
            axis=1,
            sort=False,
        ).dropna(subset=[f"{c}_crispr", f"{p}_prot"])

        ax = GIPlot.gi_tissue_plot(f"{p}_prot", f"{c}_crispr", plot_df)
        ax.set_xlabel(f"{p}\nProtein intensities")
        ax.set_ylabel(f"{c}\nCRISPR log FC")
        plt.savefig(
            f"{RPATH}/SLinteractions_{p}_{c}_regression_tissue_plot.pdf",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")

        ax = GIPlot.gi_tissue_plot(
            f"{p}_gexp",
            f"{c}_crispr",
            plot_df.dropna(subset=[f"{c}_crispr", f"{p}_gexp"]),
        )
        ax.set_xlabel(f"{p}\nGene expression (RNA-Seq voom)")
        ax.set_ylabel(f"{c}\nCRISPR log FC")
        plt.savefig(
            f"{RPATH}/SLinteractions_{p}_{c}_regression_tissue_plot_gexp.pdf",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")

    # FOXA1 ~ BSG
    p, c = ("BSG", "FOXA1")

    plot_df = pd.concat(
        [
            crispr.loc[[c]].T.add_suffix("_crispr"),
            prot.loc[[p]].T.add_suffix("_prot"),
            gexp.loc[[p]].T.add_suffix("_gexp"),
            prot_obj.broad.loc[[p, "SLC16A1"]].T.add_suffix("_broad"),
            prot_obj.ss["tissue"],
        ],
        axis=1,
        sort=False,
    ).dropna(subset=[f"{c}_crispr", f"{p}_prot"])

    # Association with BROAD
    for p_idx in [p, "SLC16A1"]:
        ax = GIPlot.gi_tissue_plot(
            f"{p_idx}_broad", f"{c}_crispr", plot_df.dropna(subset=[f"{p_idx}_broad"])
        )
        ax.set_xlabel(f"{p_idx}\nProtein intensities (CCLE)")
        ax.set_ylabel(f"{c}\nCRISPR log FC")
        plt.savefig(
            f"{RPATH}/SLinteractions_{p_idx}_CCLE_{c}_regression_tissue_plot.pdf",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")

    # Tissue-specific regression
    ax = GIPlot.gi_tissue_plot(
        f"{p}_prot",
        f"{c}_crispr",
        plot_df[plot_df["tissue"].isin(["Breast", "Prostate"])],
    )
    ax.set_xlabel(f"{p}\nProtein intensities")
    ax.set_ylabel(f"{c}\nCRISPR log FC")
    plt.savefig(
        f"{RPATH}/SLinteractions_{p}_{c}_regression_tissue_plot_selected.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    # Split by subtypes
    ctypes = ["pam50", "ER", "PR", "HER2", "p53", "BRCA1", "BRCA2", "PIK3CA", "PTEN"]

    _, axs = plt.subplots(
        len(ctypes), 1, figsize=(2.5, 1 * len(ctypes)), dpi=600, sharex=True
    )

    for i, ptype in enumerate(ctypes):
        df = pd.concat(
            [
                plot_df[f"{c}_crispr"],
                breast_subtypes[ptype].replace({1.0: "Mut", 0.0: "WT"}),
            ],
            axis=1,
        ).dropna()

        order = set(df[ptype])
        palette = pd.Series(
            sns.color_palette("Set1", n_colors=len(order)).as_hex(), index=order
        )

        GIPlot.gi_classification(
            f"{c}_crispr",
            ptype,
            df,
            palette=palette.to_dict(),
            orient="h",
            notch=False,
            order=order,
            ax=axs[i],
        )

        axs[i].set_xlabel(
            f"{c} CRISPR log2 fold-change" if i == (len(ctypes) - 1) else None
        )

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(
        f"{RPATH}/SLinteractions_PAM50_{c}_boxplot.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
