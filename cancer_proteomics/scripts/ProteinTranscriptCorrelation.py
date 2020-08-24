#!/usr/bin/env python
# Copyright (C) 2020 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from adjustText import adjust_text
from scipy.stats import mannwhitneyu
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from sklearn.mixture import GaussianMixture
from scripts import two_vars_correlation
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, GeneExpression, CopyNumber


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

    cnv_obj = CopyNumber()
    cnv = cnv_obj.filter(subset=list(prot))
    cnv_norm = np.log2(cnv.divide(prot_obj.ss.loc[cnv.columns, "ploidy"]) + 1)
    LOG.info(f"Copy number: {cnv.shape}")

    # Overlaps
    #
    samples = list(set.intersection(set(prot), set(gexp), set(cnv)))
    genes = list(set.intersection(set(prot.index), set(gexp.index), set(cnv.index)))
    LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")

    # cyclops
    #
    cyclops = pd.read_excel(f"{TPATH}/mmc2.xlsx", skiprows=3)

    # Protein/Gene correlation
    #
    pg_corr = pd.DataFrame(
        [
            two_vars_correlation(prot.loc[g], gexp.loc[g], method="spearman")
            for g in genes
        ]
    ).sort_values("pval")
    pg_corr["fdr"] = multipletests(pg_corr["pval"], method="fdr_bh")[1]
    pg_corr.to_csv(
        f"{RPATH}/ProteinTranscript_correlations.csv.gz",
        index=False,
        compression="gzip",
    )

    # Histogram
    _, ax = plt.subplots(1, 1, figsize=(2.0, 1.5))
    sns.distplot(
        pg_corr["corr"],
        hist_kws=dict(alpha=0.4, zorder=1, linewidth=0),
        bins=30,
        kde_kws=dict(cut=0, lw=1, zorder=1, alpha=0.8),
        color=CrispyPlot.PAL_DTRACE[2],
        ax=ax,
    )
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
    ax.set_xlabel("Pearson's R")
    ax.set_ylabel("Density")
    ax.set_title(f"Protein ~ Transcript (mean R={pg_corr['corr'].mean():.2f})")
    plt.savefig(
        f"{RPATH}/ProteinTranscript_histogram.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    # Protein/Gene correlation per tissue
    #
    pg_corr_tissue = pd.DataFrame(
        [
            {
                **two_vars_correlation(
                    prot.loc[g], gexp.loc[g], idx_set=t_samples, method="spearman"
                ),
                **dict(tissue=t),
            }
            for t, t_samples in prot_obj.ss.reindex(samples)
            .reset_index()
            .groupby("tissue")["model_id"]
            if len(t_samples) > 15
            for g in genes
        ]
    )
    pg_corr_tissue = pg_corr_tissue.query("len > 15")
    pg_corr_tissue["fdr"] = multipletests(pg_corr_tissue["pval"], method="fdr_bh")[1]
    pg_corr_tissue.to_csv(
        f"{RPATH}/ProteinTranscript_correlations_tissue.csv.gz",
        index=False,
        compression="gzip",
    )

    # Boxplot
    order = pg_corr_tissue.groupby("tissue")["corr"].median().sort_values()

    _, ax = plt.subplots(1, 1, figsize=(1.0, 0.125 * len(order)), dpi=600)
    sns.boxplot(
        x="corr",
        y="tissue",
        data=pg_corr_tissue,
        notch=True,
        order=order.index,
        boxprops=dict(linewidth=0.3),
        whiskerprops=dict(linewidth=0.3),
        medianprops=CrispyPlot.MEDIANPROPS,
        flierprops=CrispyPlot.FLIERPROPS,
        color=GIPlot.PAL_DTRACE[2],
        showcaps=False,
        saturation=1,
        orient="h",
        ax=ax,
    )
    ax.set_xlabel("Transcript ~ Protein")
    ax.set_ylabel("")
    ax.grid(axis="x", lw=0.1, color="#e1e1e1", zorder=0)
    plt.savefig(
        f"{RPATH}/ProteinTranscript_tissue_boxplot.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    # Protein/Gene correlation with CopyNumber - Attenuation
    #
    patt_corr = pd.DataFrame(
        {
            g: pd.concat(
                [
                    pd.Series(two_vars_correlation(cnv.loc[g], prot.loc[g])).add_prefix(
                        "prot_"
                    ),
                    pd.Series(two_vars_correlation(cnv.loc[g], gexp.loc[g])).add_prefix(
                        "gexp_"
                    ),
                ]
            )
            for g in genes
        }
    ).T.sort_values("gexp_pval")
    patt_corr = patt_corr.query("(prot_len > 15) & (gexp_len > 15)")
    patt_corr["attenuation"] = patt_corr["gexp_corr"] - patt_corr["prot_corr"]

    gmm = GaussianMixture(n_components=2).fit(patt_corr[["attenuation"]])
    s_type, clusters = (
        pd.Series(gmm.predict(patt_corr[["attenuation"]]), index=patt_corr.index),
        pd.Series(gmm.means_[:, 0], index=range(2)),
    )
    patt_corr["cluster"] = [
        "High" if s_type[p] == clusters.argmax() else "Low" for p in patt_corr.index
    ]

    patt_corr.to_csv(
        f"{RPATH}/ProteinTranscript_attenuation.csv.gz", compression="gzip"
    )
    # patt_corr = pd.read_csv(f"{RPATH}/ProteinTranscript_attenuation.csv.gz", index_col=0)

    # Scatter
    g = CrispyPlot.attenuation_scatter("gexp_corr", "prot_corr", patt_corr)
    g.set_axis_labels(
        "Transcriptomics ~ Copy number\n(Pearson's R)",
        "Protein ~ Copy number\n(Pearson's R)",
    )
    plt.savefig(
        f"{RPATH}/ProteinTranscript_attenuation_scatter.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Pathway enrichement analysis of attenuated proteins
    background = set(patt_corr.index)
    sublist = set(patt_corr.query("cluster == 'High'").index)

    enr_obj = Enrichment(
        gmts=["c5.all.v7.1.symbols.gmt"], sig_min_len=15, padj_method="fdr_bh"
    )

    enr = enr_obj.hypergeom_enrichments(sublist, background, "c5.all.v7.1.symbols.gmt")
    enr = enr[enr["adj.p_value"] < 0.01].head(30).reset_index()
    enr["name"] = [i[3:].lower().replace("_", " ") for i in enr["gset"]]

    _, ax = plt.subplots(1, 1, figsize=(2.0, 5.0), dpi=600)

    sns.barplot(
        -np.log10(enr["adj.p_value"]),
        enr["name"],
        orient="h",
        color=CrispyPlot.PAL_DTRACE[2],
        ax=ax,
    )

    for i, (_, row) in enumerate(enr.iterrows()):
        plt.text(
            -np.log10(row["adj.p_value"]),
            i,
            f"{row['len_intersection']}/{row['len_sig']}",
            va="center",
            ha="left",
            fontsize=5,
            zorder=10,
            color=CrispyPlot.PAL_DTRACE[2],
        )

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
    ax.set_xlabel("Hypergeometric test (-log10 FDR)")
    ax.set_ylabel("")
    ax.set_title(f"GO celluar component enrichment - attenuated proteins")

    plt.savefig(
        f"{RPATH}/ProteinTranscript_attenuation_enrichment.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    # Attenuation scatter gene highlights
    signatures = [
        "GO_PROTEIN_MODIFICATION_BY_SMALL_PROTEIN_CONJUGATION",
        "GO_TRANSLATIONAL_INITIATION",
        "GO_RIBOSOMAL_SUBUNIT",
    ]
    signatures = {
        s: set(enr_obj.get_signature("c5.all.v7.1.symbols.gmt", s)).intersection(
            patt_corr.index
        )
        for s in signatures
    }

    plot_df = patt_corr.copy()
    plot_df = plot_df.assign(
        signature=[[s for s in signatures if g in signatures[s]] for g in plot_df.index]
    )
    plot_df = plot_df.assign(
        signature=[i[0] if len(i) > 0 else "All" for i in plot_df["signature"]]
    )

    ax_min = plot_df[["gexp_corr", "prot_corr"]].min().min() * 1.1
    ax_max = plot_df[["gexp_corr", "prot_corr"]].max().max() * 1.1

    discrete_pal = pd.Series(
        sns.color_palette("tab10").as_hex()[: len(signatures)], index=signatures
    )
    discrete_pal["All"] = CrispyPlot.PAL_DTRACE[0]

    grid = GIPlot.gi_regression_marginal(
        "gexp_corr",
        "prot_corr",
        "signature",
        plot_df,
        plot_reg=False,
        plot_annot=False,
        scatter_kws=dict(edgecolor="w", lw=0.1, s=8),
        discrete_pal=discrete_pal,
    )

    grid.ax_joint.plot([ax_min, ax_max], [ax_min, ax_max], "k--", lw=0.3)
    grid.ax_joint.set_xlim(ax_min, ax_max)
    grid.ax_joint.set_ylim(ax_min, ax_max)

    labels = [
        grid.ax_joint.text(row["gexp_corr"], row["prot_corr"], i, color="k", fontsize=4)
        for i, row in plot_df.query("signature != 'All'")
        .sort_values("attenuation", ascending=False)
        .head(15)
        .iterrows()
    ]
    adjust_text(
        labels,
        arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
        ax=grid.ax_joint,
    )

    plt.gcf().set_size_inches(2.5, 2.5)

    plt.savefig(
        f"{RPATH}/ProteinTranscript_attenuation_scatter_signatures.pdf",
        bbox_inches="tight",
    )
    plt.close("all")

    # Cyclops
    #
    cyclops_fdr = cyclops.groupby("Gene")["Left sided FDR"].min()
    patt_corr["cyclops"] = cyclops_fdr.reindex(patt_corr.index)

    # Boxplot
    fdr_thres = 0.25
    plot_df = patt_corr.dropna(subset=["cyclops"])
    plot_df["cyclops_signif"] = (plot_df["cyclops"] < fdr_thres).astype(int)

    fdr_thres_count = plot_df["cyclops_signif"].value_counts()
    plot_df["name"] = [
        f"FDR {'<' if i else '>='} {fdr_thres*100:.0f}% (N={fdr_thres_count[i]})"
        for i in plot_df["cyclops_signif"]
    ]

    _, ax = plt.subplots(1, 1, figsize=(2.0, 1.0), dpi=600)
    sns.boxplot(
        x="attenuation",
        y="name",
        data=plot_df,
        notch=True,
        boxprops=dict(linewidth=0.3),
        whiskerprops=dict(linewidth=0.3),
        medianprops=CrispyPlot.MEDIANPROPS,
        flierprops=CrispyPlot.FLIERPROPS,
        color=GIPlot.PAL_DTRACE[2],
        showcaps=False,
        saturation=1,
        orient="h",
        zorder=1,
        ax=ax,
    )

    def rand_jitter(arr, stdev=0.2):
        return arr + np.random.randn(len(arr)) * stdev

    g_highlight = [
        "PSMC2",
        "PHF5A",
        "RPS15",
        "PUF60",
        "RPS11",
        "SF3A2",
        "SNRNP70",
        "RBM17",
        "PSMA4",
    ]
    g_highlight_df = plot_df.reindex(g_highlight)
    g_highlight_df["y_jitter"] = rand_jitter(g_highlight_df["cyclops_signif"])

    ax.scatter(
        g_highlight_df["attenuation"],
        g_highlight_df["y_jitter"],
        color=CrispyPlot.PAL_DTRACE[1],
        lw=0.1,
        edgecolor="white",
        s=5,
        zorder=2,
    )

    labels = [
        ax.text(row["attenuation"], row["y_jitter"], i, color="k", fontsize=4)
        for i, row in g_highlight_df.iterrows()
    ]
    adjust_text(
        labels, arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3), ax=ax
    )

    ax.set_xlabel("Attenuation score")
    ax.set_ylabel("Cyclops")

    t, p = mannwhitneyu(
        plot_df.query(f"cyclops < {fdr_thres}")["attenuation"],
        plot_df.query(f"cyclops >= {fdr_thres}")["attenuation"],
    )
    ax.set_title(f"Mann-Whitney rank test p-value={p:.2e}")

    ax.grid(axis="x", lw=0.1, color="#e1e1e1", zorder=0)
    plt.savefig(
        f"{RPATH}/ProteinTranscript_cyclops_boxplot.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    #
    g = "PSMC2"

    plot_df = pd.concat([
        gexp.loc[g].rename("gexp"),
        prot.loc[g].rename("prot"),
        cnv.loc[g].rename("cnv"),
    ], axis=1).sort_values("cnv", ascending=False)

    _, ax = plt.subplots(1, 1, figsize=(2.5, 2.0), dpi=600)

    GIPlot.gi_continuous_plot("gexp", "prot", "cnv", plot_df, ax=ax, mid_point_norm=False)

    plt.savefig(
        f"{RPATH}/ProteinTranscript_cyclops_omics_{g}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")
