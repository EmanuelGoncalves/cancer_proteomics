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
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from GIPlot import GIPlot
from crispy.Utils import Utils
from crispy.CrispyPlot import CrispyPlot
from sklearn.mixture import GaussianMixture
from eg.CProtUtils import two_vars_correlation
from crispy.Enrichment import Enrichment, SSGSEA
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

    prot_broad = prot_obj.broad
    LOG.info(f"Proteomics Broad: {prot_broad.shape}")

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
    genes = list(
        set.intersection(
            set(prot.index), set(gexp.index), set(cnv.index), set(prot_broad.index)
        )
    )
    LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")

    # Data tranformations
    #
    gexp_t = pd.DataFrame(
        {i: Utils.gkn(gexp.loc[i].dropna()).to_dict() for i in genes}
    ).T

    # Sample-wise Protein/Gene correlation with CopyNumber - Attenuation
    #
    satt_corr = pd.DataFrame(
        {
            s: pd.concat(
                [
                    pd.Series(two_vars_correlation(cnv[s], prot[s])).add_prefix("prot_"),
                    pd.Series(two_vars_correlation(cnv[s], prot_broad[s])).add_prefix("prot_broad_") if s in prot_broad else pd.Series(),
                    pd.Series(two_vars_correlation(cnv[s], gexp_t[s])).add_prefix("gexp_"),
                    pd.Series(two_vars_correlation(gexp_t[s], prot[s])).add_prefix("gexp_prot_"),
                    pd.Series(two_vars_correlation(gexp_t[s], prot_broad[s])).add_prefix("gexp_prot_broad_") if s in prot_broad else pd.Series(),
                ]
            )
            for s in samples
        }
    ).T.sort_values("gexp_pval")
    satt_corr = satt_corr.dropna(subset=["gexp_corr", "prot_corr"])
    satt_corr["attenuation"] = satt_corr.eval("gexp_corr - prot_corr")
    satt_corr["attenuation_broad"] = satt_corr.eval("gexp_corr - prot_broad_corr")

    # Discretise attenuated samples
    gmm = GaussianMixture(n_components=2, means_init=[[0], [0.4]]).fit(
        satt_corr[["attenuation"]]
    )
    s_type, clusters = (
        pd.Series(gmm.predict(satt_corr[["attenuation"]]), index=satt_corr.index),
        pd.Series(gmm.means_[:, 0], index=range(2)),
    )
    satt_corr["cluster"] = [
        "High" if s_type[p] == clusters.argmax() else "Low" for p in satt_corr.index
    ]

    # Pathway enrichment
    emt_sig = Enrichment.read_gmt(f"{DPATH}/pathways/emt.symbols.gmt", min_size=0)
    emt_enr = pd.Series({s: SSGSEA.gsea(gexp[s], emt_sig["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"])[0] for s in gexp})

    proteasome_sig = Enrichment.read_gmt(f"{DPATH}/pathways/proteasome.symbols.gmt", min_size=0)
    proteasome_enr = pd.Series({s: SSGSEA.gsea(prot[s].dropna(), proteasome_sig["BIOCARTA_PROTEASOME_PATHWAY"])[0] for s in prot})
    proteasome_enr_broad = pd.Series({s: SSGSEA.gsea(prot_broad[s].dropna(), proteasome_sig["BIOCARTA_PROTEASOME_PATHWAY"])[0] for s in prot_broad})

    translation_sig = Enrichment.read_gmt(f"{DPATH}/pathways/translation_initiation.symbols.gmt", min_size=0)
    translation_enr = pd.Series({s: SSGSEA.gsea(prot[s].dropna(), translation_sig["GO_TRANSLATIONAL_INITIATION"])[0] for s in prot})
    translation_enr_broad = pd.Series({s: SSGSEA.gsea(prot_broad[s].dropna(), translation_sig["GO_TRANSLATIONAL_INITIATION"])[0] for s in prot_broad})

    # Annotate samples with cell line information
    cfeatures = pd.concat(
        [
            emt_enr.rename("EMT"),
            proteasome_enr.rename("Proteasome"),
            proteasome_enr_broad.rename("Proteasome_broad"),
            translation_enr.rename("TranslationInitiation"),
            translation_enr_broad.rename("TranslationInitiation_broad"),
            CopyNumber().genomic_instability().rename("CopyNumberInstability"),
            pd.get_dummies(prot_obj.ss["msi_status"])["MSI"],
            pd.get_dummies(prot_obj.ss["media"]),
            pd.get_dummies(prot_obj.ss["growth_properties"]),
            pd.get_dummies(prot_obj.ss["tissue"])[["Haematopoietic and Lymphoid", "Lung"]],
            prot_obj.ss.reindex(columns=["ploidy", "mutational_burden", "size", "growth"]),
            prot_obj.reps.rename("RepsCorrelation"),
            prot_obj.protein_raw.median().rename("MedianProteomics"),
        ],
        axis=1,
    ).reindex(satt_corr.index)

    satt_corr = pd.concat([satt_corr, cfeatures], axis=1)

    # Export
    satt_corr.to_csv(
        f"{RPATH}/SampleProteinTranscript_attenuation.csv.gz", compression="gzip"
    )
    # satt_corr = pd.read_csv(f"{RPATH}/SampleProteinTranscript_attenuation.csv.gz", index_col=0)

    # Scatter
    for y_var in ["prot_corr", "prot_broad_corr"]:
        g = CrispyPlot.attenuation_scatter(
            "gexp_corr", y_var, satt_corr.dropna(subset=["gexp_corr", y_var])
        )
        g.set_axis_labels(
            "Transcriptomics ~ Copy number\n(Pearson's R)",
            "Protein ~ Copy number\n(Pearson's R)",
        )
        plt.savefig(
            f"{RPATH}/ProteinTranscriptSample_attenuation_scatter_{y_var}.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

    # Clustermap
    plot_df = satt_corr[["attenuation", "attenuation_broad", "gexp_prot_corr", "gexp_prot_broad_corr"] + list(cfeatures.columns)]
    plot_df = plot_df.corr()

    fig = sns.clustermap(
        plot_df,
        cmap="Spectral",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 5},
        linewidth=0.3,
        cbar_pos=None,
        figsize=np.array(plot_df.shape) * 0.275,
    )

    plt.savefig(
        f"{RPATH}/ProteinTranscriptSample_cfeatures_clustermap.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    #
    for z_var in ["CopyNumberInstability", "ploidy"]:
        ax = GIPlot.gi_continuous_plot(
            "prot_corr",
            "gexp_prot_corr",
            z_var,
            satt_corr, mid_point_norm=False
        )
        ax.set_xlabel("Sanger&CMRI\nProtein ~ Copy number (Pearson's R)")
        ax.set_ylabel("Sanger&CMRI\nProtein ~ Transcript (Pearson's R)")
        plt.savefig(
            f"{RPATH}/ProteinTranscriptSample_prot_gexp_regression_{z_var}.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

    #
    x_var, y_var, z_var = "ploidy", "CopyNumberInstability", "size"
    ax = GIPlot.gi_continuous_plot(
        x_var,
        y_var,
        z_var,
        satt_corr, mid_point_norm=False
    )
    plt.savefig(
        f"{RPATH}/ProteinTranscriptSample_regression_{x_var}_{y_var}_{z_var}.pdf",
        bbox_inches="tight",
    )
    plt.close("all")

