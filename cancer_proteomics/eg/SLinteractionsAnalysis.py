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
    sl_file = "lm_sklearn_degr_crispr_per_tissue"
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
    sl_lm["skew"] = crispr_selective.loc[sl_lm["y_id"], "skew"].values

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
