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
import matplotlib.pyplot as plt
from GIPlot import GIPlot
from crispy.Utils import Utils
from crispy.CrispyPlot import CrispyPlot
from sklearn.mixture import GaussianMixture
from eg.CProtUtils import two_vars_correlation
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
                    pd.Series(two_vars_correlation(cnv[s], prot[s])).add_prefix(
                        "prot_"
                    ),
                    pd.Series(two_vars_correlation(cnv[s], prot_broad[s])).add_prefix(
                        "prot_broad_"
                    )
                    if s in prot_broad
                    else pd.Series(),
                    pd.Series(two_vars_correlation(cnv[s], gexp_t[s])).add_prefix(
                        "gexp_"
                    ),
                ]
            )
            for s in samples
        }
    ).T.sort_values("gexp_pval")
    satt_corr = satt_corr.dropna(subset=["gexp_corr", "prot_corr"])
    satt_corr["attenuation"] = satt_corr.eval("gexp_corr - prot_corr")
    satt_corr["attenuation_broad"] = satt_corr.eval("gexp_corr - prot_broad_corr")

    #
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

    satt_corr.to_csv(
        f"{RPATH}/SampleProteinTranscript_attenuation.csv.gz", compression="gzip"
    )

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

    # Attenuation regression
    g = GIPlot.gi_regression("prot_corr", "prot_broad_corr", satt_corr)
    g.set_axis_labels(
        "Sanger&CMRI\nProtein ~ Copy number (Pearson's R)",
        "Broad\nProtein ~ Copy number (Pearson's R)"
    )
    plt.savefig(
        f"{RPATH}/ProteinTranscriptSample_attenuation_regression.pdf",
        bbox_inches="tight",
    )
    plt.close("all")
