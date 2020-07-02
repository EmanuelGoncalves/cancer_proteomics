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
from adjustText import adjust_text
from matplotlib_venn import venn2, venn2_circles
from eg.CProtUtils import two_vars_correlation
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
    Sample,
    Metabolomics,
)


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")


if __name__ == "__main__":
    # Data-sets
    #

    wes_obj = WES()

    mobem_obj = Mobem()
    cn_obj = CopyNumber()

    prot_obj = Proteomics()
    gexp_obj = GeneExpression()

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

    gexp = gexp_obj.filter(subset=samples)
    LOG.info(f"Transcriptomics: {gexp.shape}")

    crispr = crispr_obj.filter(subset=samples, dtype="merged")
    LOG.info(f"CRISPR: {crispr.shape}")

    drespo = drug_obj.filter(subset=samples)
    drespo = drespo.set_index(pd.Series([";".join(map(str, i)) for i in drespo.index]))

    drespo_maxc = drug_obj.maxconcentration.copy()
    drespo_maxc.index = [";".join(map(str, i)) for i in drug_obj.maxconcentration.index]
    drespo_maxc = drespo_maxc.reindex(drespo.index)
    LOG.info(f"Drug response: {drespo.shape}")

    cn = cn_obj.filter(subset=samples.intersection(prot_obj.ss.index))
    cn = np.log2(cn.divide(prot_obj.ss.loc[cn.columns, "ploidy"]) + 1)
    LOG.info(f"Copy-Number: {cn.shape}")

    wes = wes_obj.filter(subset=samples, min_events=3, recurrence=True)
    wes = wes.loc[wes.std(1) > 0]
    LOG.info(f"WES: {wes.shape}")

    mobem = mobem_obj.filter(subset=samples)
    mobem = mobem.loc[mobem.std(1) > 0]
    LOG.info(f"MOBEM: {mobem.shape}")

    # LM associations
    #
    lm_drug = pd.read_csv(f"{RPATH}/lm_sklearn_degr_drug_annotated.csv.gz")
    lm_crispr = pd.read_csv(f"{RPATH}/lm_sklearn_degr_crispr_annotated.csv.gz")

    # Selective and predictive dependencies
    #
    R2_THRES = 0.2
    SKEW_THRES = -2
    FDR_THRES = 0.01

    dep_df = pd.concat(
        [
            lm_drug.groupby("y_id")[["r2", "skew"]]
            .first()
            .reset_index()
            .assign(dtype="drug"),
            lm_crispr.groupby("y_id")[["r2", "skew"]]
            .first()
            .reset_index()
            .assign(dtype="crispr"),
        ]
    ).dropna()

    # Selectivity plot
    #
    _, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

    plot_info = [("crispr", "o", "#009EAC"), ("drug", "X", "#FEC041")]
    for i, (n, m, c) in enumerate(plot_info):
        n_df = dep_df.query(f"dtype == '{n}'")

        n_ax = ax if n == "crispr" else ax.twiny()

        n_ax.scatter(
            n_df["skew"],
            n_df["r2"],
            marker=m,
            s=3,
            c=c,
            zorder=(i + 1),
            alpha=0.8,
            lw=0,
        )

        n_ax.set_xlabel(f"{n} skewness", color=c)
        n_ax.set_ylabel("R2")

        labels = n_df.query(f"r2 > {R2_THRES}").sort_values("skew").head(20)
        labels = [
            n_ax.text(
                row["skew"],
                row["r2"],
                row["y_id"] if n == "crispr" else row["y_id"].split(";")[1],
                color="k",
                fontsize=4,
                zorder=3,
            )
            for _, row in labels.iterrows()
        ]
        adjust_text(
            labels,
            arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
            ax=n_ax,
        )

    ax.grid(axis="y", lw=0.1, color="#e1e1e1", zorder=0)
    ax.axhline(R2_THRES, c="#E3213D", lw=0.3, ls="--")

    plt.savefig(
        f"{RPATH}/TopHits_selectivity_predictive_scatter.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Predictive features of selective and predictive dependencies
    #
    tophits = dep_df.query(f"(r2 > {R2_THRES}) & (skew < {SKEW_THRES})")

    tophits_feat_drug = set(lm_drug.query(f"fdr < {FDR_THRES}")["x_id"])
    tophits_feat_crispr = set(lm_crispr.query(f"fdr < {FDR_THRES}")["x_id"])
    tophits_feat_union = set.union(tophits_feat_drug, tophits_feat_crispr)

    venn_groups = [tophits_feat_drug, tophits_feat_crispr]
    venn2(venn_groups, set_labels=["Drug", "CRISPR"], set_colors=["#FEC041", "#009EAC"])
    venn2_circles(venn_groups, linewidth=0.5)
    plt.title(f"Top protein features (FDR < {FDR_THRES*100:.0f}%)")
    plt.savefig(f"{RPATH}/TopHits_features_venn.pdf", bbox_inches="tight")
    plt.close("all")

    tophits_feat = pd.concat(
        [
            lm_drug[lm_drug["x_id"].isin(tophits_feat_union)].assign(dtype="drug"),
            lm_crispr[lm_crispr["x_id"].isin(tophits_feat_union)].assign(
                dtype="crispr"
            ),
        ]
    )
    tophits_feat = tophits_feat[tophits_feat["y_id"].isin(tophits["y_id"])]
    tophits_feat = tophits_feat.query(f"fdr < {FDR_THRES}")

    def calculate_score(pval, beta):
        s = np.log10(pval)
        if beta > 0:
            s *= -1
        return s

    tophits_feat["score"] = [
        calculate_score(p, b) for p, b in tophits_feat[["pval", "beta"]].values
    ]

    # Top hits features clustermap
    #
    plot_df = pd.pivot_table(
        tophits_feat, index="y_id", columns="x_id", values="score", fill_value=0
    ).T

    fig = sns.clustermap(
        plot_df,
        cmap="RdBu_r",
        center=0,
        mask=plot_df.replace(0, np.nan).isnull(),
        figsize=(20, 7),
    )

    plt.savefig(f"{RPATH}/TopHits_features_clustermap.pdf", bbox_inches="tight")
    plt.close("all")

    # Top hits scatter
    #
    plot_df = tophits_feat[
        ~tophits_feat["y_id"].isin(tophits_feat["y_id"].value_counts().head(5).index)
    ]

    _, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

    plot_info = [("crispr", "o", "#009EAC"), ("drug", "X", "#FEC041")]
    for i, (n, m, c) in enumerate(plot_info):
        n_plot_df = plot_df.query(f"dtype == '{n}'")

        for ttype in ["ppi != 'T'", "ppi == 'T'"]:
            ax.scatter(
                n_plot_df.query(ttype)["beta"],
                n_plot_df.query(ttype)["r2"],
                marker=m,
                s=3,
                c=c,
                zorder=(i + 1),
                alpha=0.8,
                lw=.3 if ttype == "ppi == 'T'" else 0,
                edgecolors=CrispyPlot.PAL_DTRACE[1] if ttype == "ppi == 'T'" else None,
                label=n,
            )

    ax.grid(axis="y", lw=0.1, color="#e1e1e1", zorder=0)

    ax.set_xlabel(f"LM association score (+/- log10 pval)")
    ax.set_ylabel("R2")

    plt.legend(
        frameon=False, prop={"size": 4}
    )

    plt.savefig(f"{RPATH}/TopHits_features_scatter.pdf", bbox_inches="tight")
    plt.close("all")

    # Top dependencies
    #

    topdep = ["WRN", "STAG1", "1403;AZD6094;GDSC1", "BRAF"]

    # Top associations
    for y_id in topdep:
        # y_id = "MECOM"
        plot_df = (
            tophits_feat.query(f"y_id == '{y_id}'")
            .head(10)
            .reset_index(drop=True)
            .reset_index()
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

        ax.set_title(f"{y_id} (R-squared={plot_df['r2'].max():.2f})")
        plt.ylabel(f"Linear regressions\n(p-value log10)")

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
        ax.axes.get_xaxis().set_ticks([])

        plt.savefig(f"{RPATH}/TopHits_top_associations_{y_id}.pdf", bbox_inches="tight")
        plt.close("all")

    #
    #
    gi_pairs = [
        (
            "RPL22",
            "WRN",
            "crispr",
            ["Large Intestine", "Endometrium", "Stomach", "Ovary"],
        ),
        ("RAD21", "STAG1", "crispr", ["Bone", "Central Nervous System", "Breast"]),
        ("MET", "1403;AZD6094;GDSC1", "drug", ["Stomach", "Esophagus", "Lung"]),
        ("ACIN1", "BRAF", "crispr", ["Skin", "Breast", "Large Intestine", "Ovary"]),
    ]

    for p, c, dtype in gi_pairs:
        p, c, dtype, ctissues = ("VTN", "1373;Dabrafenib;GDSC1", "drug", ["Skin", "Breast", "Ovary"])

        plot_df = pd.concat(
            [
                drespo.loc[[c]].T.add_suffix("_y")
                if dtype == "drug"
                else crispr.loc[[c]].T.add_suffix("_y"),
                prot.loc[[p]].T.add_suffix("_prot"),
                gexp.loc[[p]].T.add_suffix("_gexp"),
                prot_obj.ss["tissue"],
            ],
            axis=1,
            sort=False,
        ).dropna(subset=[f"{c}_y", f"{p}_prot"])

        # Protein
        ax = GIPlot.gi_tissue_plot(f"{p}_prot", f"{c}_y", plot_df)
        ax.set_xlabel(f"{p}\nProtein intensities")
        ax.set_ylabel(
            f"{c}\n{'Drug response IC50' if dtype == 'drug' else 'CRISPR-Cas9 (log2 FC)'}"
        )
        plt.savefig(
            f"{RPATH}/TopHits_{p}_{c}_{dtype}_regression_tissue_plot.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

        # Protein
        if ctissues is not None:
            ax = GIPlot.gi_tissue_plot(
                f"{p}_prot", f"{c}_y", plot_df[plot_df["tissue"].isin(ctissues)]
            )
            ax.set_xlabel(f"{p}\nProtein intensities")
            ax.set_ylabel(
                f"{c}\n{'Drug response IC50' if dtype == 'drug' else 'CRISPR-Cas9 (log2 FC)'}"
            )
            plt.savefig(
                f"{RPATH}/TopHits_{p}_{c}_{dtype}_regression_tissue_plot_selected.pdf",
                bbox_inches="tight",
            )
            plt.close("all")

        # Gene expression
        ax = GIPlot.gi_tissue_plot(
            f"{p}_gexp", f"{c}_y", plot_df.dropna(subset=[f"{c}_y", f"{p}_gexp"])
        )
        ax.set_xlabel(f"{p}\nGene expression (RNA-Seq voom)")
        ax.set_ylabel(
            f"{c}\n{'Drug response IC50' if dtype == 'drug' else 'CRISPR-Cas9 (log2 FC)'}"
        )
        plt.savefig(
            f"{RPATH}/TopHits_{p}_{c}_{dtype}_regression_tissue_plot_gexp.pdf",
            bbox_inches="tight",
        )
        plt.close("all")
