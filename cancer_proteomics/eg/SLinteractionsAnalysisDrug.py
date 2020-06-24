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
from crispy.GIPlot import GIPlot
from adjustText import adjust_text
from itertools import zip_longest
from crispy.MOFA import MOFA, MOFAPlot
from sklearn.metrics.ranking import auc
from crispy.Enrichment import Enrichment
from scipy.stats import pearsonr, spearmanr, skew
from pathlib import Path, PurePath
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
    DrugResponse,
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

    drespo_obj = DrugResponse()

    drespo = drespo_obj.filter()
    drespo.index = [";".join(map(str, i)) for i in drespo.index]

    dmax = drespo_obj.drugresponse.groupby(["drug_id", "drug_name", "dataset"])[
        "max_screening_conc"
    ].first()
    dmax = (dmax * 0.5).pipe(np.log)
    dmax.index = [";".join(map(str, i)) for i in dmax.index]

    dtargets = drespo_obj.drugresponse.groupby(["drug_id", "drug_name", "dataset"])[
        "putative_gene_target"
    ].first()
    dtargets.index = [";".join(map(str, i)) for i in dtargets.index]
    LOG.info(f"Drug: {drespo.shape}")

    # Gene sets
    #
    cgenes = pd.read_csv(f"{DPATH}/cancer_genes_latest.csv.gz")
    cgenes = list(set(cgenes["gene_symbol"]))

    patt = pd.read_csv(f"{RPATH}/ProteinTranscript_attenuation.csv.gz", index_col=0)
    patt_low = list(patt.query("cluster == 'Low'").index)
    patt_high = list(patt.query("cluster == 'High'").index)

    # ML scores
    #
    ml_files = dict(prot="score_dl_min300_ic50_eg_id.csv")
    ml_dfs = {f: pd.read_csv(f"{TPATH}/{ml_files[f]}", index_col=0) for f in ml_files}
    ml_scores = pd.DataFrame({f: ml_dfs[f]["test"] for f in ml_dfs})

    # Strongly selective drugs
    #
    drug_selective = pd.concat(
        [
            drespo.apply(skew, axis=1, nan_policy="omit").rename("skew"),
            drespo.median(1).rename("median"),
            (drespo.T < dmax[drespo.index]).sum().rename("nsamples"),
            drespo.T[drespo.T < dmax[drespo.index]].median().rename("dependency"),
            ml_scores,
        ],
        axis=1,
    )
    drug_selective["name"] = [i.split(";")[1] for i in drug_selective.index]

    drug_selective_set = set(drug_selective.query("skew < -2").index)

    # Scatter
    grid = GIPlot.gi_regression(
        "skew",
        "median",
        drug_selective,
        size="dependency",
        size_inverse=True,
        size_legend_title="Median IC50",
        plot_reg=False,
        plot_annot=False,
    )

    grid.ax_joint.axvline(-1, c=GIPlot.PAL_DTRACE[1], lw=0.3, ls="--")
    g_highlight_df = drug_selective.query("skew < -1").sort_values("skew").head(5)
    labels = [
        grid.ax_joint.text(
            row["skew"], row["median"], row["name"], color="k", fontsize=4
        )
        for _, row in g_highlight_df.iterrows()
    ]
    adjust_text(
        labels,
        arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
        ax=grid.ax_joint,
    )
    grid.set_axis_labels("Skewness", "Median")
    grid.ax_marg_x.set_title("Drug selective dependencies")
    plt.savefig(
        f"{RPATH}/SLinteractions_drug_selective_dependencies.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Scatter with ML scores
    grid = GIPlot.gi_regression(
        "skew",
        "prot",
        drug_selective,
        size="dependency",
        size_inverse=True,
        size_legend_loc=3,
        size_legend_title="Median IC50",
        plot_reg=False,
        plot_annot=False,
    )

    x_thres, y_thres = -1, 0.4
    grid.ax_joint.axvline(x_thres, c=GIPlot.PAL_DTRACE[1], lw=0.3, ls="--")
    grid.ax_joint.axhline(y_thres, c=GIPlot.PAL_DTRACE[1], lw=0.3, ls="--")

    g_highlight_df = (
        drug_selective.query(f"(skew < {x_thres}) & (prot > {y_thres})")
        .sort_values("skew")
        .head(5)
    )
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
    grid.ax_marg_x.set_title("Drug selective dependencies")
    plt.savefig(
        f"{RPATH}/SLinteractions_drug_selective_dependencies_rsquared.pdf",
        bbox_inches="tight",
    )
    plt.close("all")

    # Linear regression scores
    #
    sl_file = "lm_sklearn_degr_drug"
    sl_lm = pd.read_csv(f"{RPATH}/{sl_file}.csv.gz")

    # # String distance
    # sl_lm["target"] = dtargets.loc[sl_lm["y"]].values
    #
    # ppi = PPI().build_string_ppi(score_thres=900)
    # sl_lm = PPI.ppi_annotation(
    #     sl_lm, ppi, x_var="x", y_var="target", ppi_var="string_dist"
    # )

    # Attenuated protein
    sl_lm["attenuated"] = sl_lm["x_id"].isin(patt_high).astype(int)

    # R-squared
    sl_lm["r2"] = ml_scores.reindex(sl_lm["y_id"])["prot"].values

    # Strongly selective
    sl_lm["selectivity"] = drug_selective.loc[sl_lm["y_id"], "skew"].values

    # Export annotated table
    sl_lm.to_csv(f"{RPATH}/{sl_file}_annotated.csv.gz", compression="gzip", index=False)

    # Feature importance matrix
    sl_lm_fimpor = dict(
        prot=pd.pivot_table(
            sl_lm, index="x_id", columns="y_id", values="beta", fill_value=np.nan
        )
    )

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
        f"{RPATH}/SLinteractions_drug_volcano.png",
        bbox_inches="tight",
    )
    plt.close("all")

    # CRISPR Hits
    #
    hits = sl_lm.query("(fdr < .1) & (r2 > 0.4)")

    # Top associations
    for gene in [
        "277;Linifanib;GDSC1",
        "1403;AZD6094;GDSC1",
        "1013;Nilotinib;GDSC1",
        "254;Quizartinib;GDSC1",
        "412;Venotoclax;GDSC1",
        "1862;MG-132;GDSC2",
        "1191;Bortezomib;GDSC2",
    ]:
        plot_df = (
            hits.query(f"y_id == '{gene}'").head(10).reset_index(drop=True).reset_index()
        )
        plot_df = plot_df.assign(logpval=-np.log10(plot_df["pval"]).values)
        plot_df = plot_df.fillna("X")

        fig, ax = plt.subplots(1, 1, figsize=(plot_df.shape[0] * 0.2, 1.5))

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
            if str(t) != "nan":
                c = GIPlot.PAL_DTRACE[1] if t == "T" else GIPlot.PAL_DTRACE[2]

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
            f"{RPATH}/SLinteractions_drug_top_associations_{gene}.pdf", bbox_inches="tight"
        )
        plt.close("all")

    # Associations
    gi_pairs = [
        ("TES", "1403;AZD6094;GDSC1"),
        ("MET", "1403;AZD6094;GDSC1"),
        ("PSMA2", "1862;MG-132;GDSC2"),
        ("PSMB6", "1191;Bortezomib;GDSC2"),
    ]

    for p, c in gi_pairs:
        # p, c = "REXO4", "1013;Nilotinib;GDSC1"
        plot_df = pd.concat(
            [
                drespo.loc[[c]].T.add_suffix("_drug"),
                prot.loc[[p]].T.add_suffix("_prot"),
                gexp.loc[[p]].T.add_suffix("_gexp"),
                prot_obj.ss["tissue"],
            ],
            axis=1,
            sort=False,
        ).dropna(subset=[f"{c}_drug", f"{p}_prot"])

        ax = GIPlot.gi_tissue_plot(f"{p}_prot", f"{c}_drug", plot_df)
        ax.set_xlabel(f"{p}\nProtein intensities")
        ax.set_ylabel(f"{c}\nDrug response IC50")
        plt.savefig(
            f"{RPATH}/SLinteractions_drug_{p}_{c}_regression_tissue_plot.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

        ax = GIPlot.gi_tissue_plot(
            f"{p}_gexp",
            f"{c}_drug",
            plot_df.dropna(subset=[f"{c}_drug", f"{p}_gexp"]),
        )
        ax.set_xlabel(f"{p}\nGene expression (RNA-Seq voom)")
        ax.set_ylabel(f"{c}\nDrug response IC50")
        plt.savefig(
            f"{RPATH}/SLinteractions_drug_{p}_{c}_regression_tissue_plot_gexp.pdf",
            bbox_inches="tight",
        )
        plt.close("all")
