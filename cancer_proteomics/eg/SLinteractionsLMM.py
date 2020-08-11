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
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from GIPlot import GIPlot
from adjustText import adjust_text
from sklearn.decomposition import PCA
from crispy.CrispyPlot import CrispyPlot
from crispy.LMModels import LMModels, LModel
from crispy.Enrichment import Enrichment, SSGSEA
from eg.CProtUtils import two_vars_correlation
from crispy.Utils import Utils
from crispy.DataImporter import (
    Proteomics,
    CRISPR,
    GeneExpression,
    DrugResponse,
    CopyNumber,
    PPI,
)

LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")
TPATH = pkg_resources.resource_filename("tables", "/")


if __name__ == "__main__":
    # PPI
    #
    ppi = PPI().build_string_ppi(score_thres=900)

    # Gene information
    #
    ginfo = pd.read_csv(f"{TPATH}/mart_export.txt", sep="\t")
    ginfo["mean_pos"] = ginfo[["Gene end (bp)", "Gene start (bp)"]].mean(1)
    ginfo = ginfo[ginfo["Chromosome/scaffold name"].isin(Utils.CHR_ORDER)]

    ginfo_pos = pd.concat([
        ginfo.groupby("Gene name")["Chromosome/scaffold name"].first().rename("chr"),
        ginfo.groupby("Gene name")["mean_pos"].mean().rename("chr_pos"),
    ], axis=1)

    # Y matrices
    #
    gexp_obj = GeneExpression()
    gexp = gexp_obj.filter()
    LOG.info(f"Gexp: {gexp.shape}")

    prot_obj = Proteomics()
    prot = prot_obj.filter()
    prot = prot[prot.count(1) > 300]
    LOG.info(f"Prot: {prot.shape}")

    # X matrices
    #
    crispr_obj = CRISPR()
    crispr = crispr_obj.filter(dtype="merged")
    LOG.info(f"CRISPR: {crispr.shape}")

    drespo_obj = DrugResponse()
    drespo = drespo_obj.filter()
    drespo = drespo[drespo.count(1) > 300]
    drespo = drespo[["+" not in i for i in drespo.index]]
    drespo.index = [";".join(map(str, i)) for i in drespo.index]

    dtargets = drespo_obj.drugresponse.groupby(["drug_id", "drug_name", "dataset"])[
        "putative_gene_target"
    ].first()
    dtargets.index = [";".join(map(str, i)) for i in dtargets.index]
    LOG.info(f"Drug: {drespo.shape}")

    # Covariates
    #
    covariates = pd.concat(
        [
            prot_obj.ss["growth"],
            pd.get_dummies(prot_obj.ss["media"]),
            pd.get_dummies(prot_obj.ss["growth_properties"]),
            pd.get_dummies(crispr_obj.merged_institute)["Sanger"],
            prot_obj.reps.rename("RepsCorrelation"),
            drespo.mean().rename("MeanIC50"),
        ],
        axis=1,
    )

    # Dimension reduction
    #
    gexp_pca = pd.DataFrame(
        PCA(n_components=10).fit_transform(gexp.T), index=gexp.columns
    ).add_prefix("PC")

    # LMs: Proteomics
    #
    X = LMModels.transform_matrix(prot, t_type="None", fillna_func=None).T
    M2 = LMModels.transform_matrix(gexp, t_type="None").T
    genes = set.intersection(set(X), set(M2))

    for t, t_df in prot_obj.ss.groupby("tissue"):
        LOG.info(f"Tissue = {t}")

        # Drug
        Y = LMModels.transform_matrix(drespo, t_type="None").T
        M = pd.concat([covariates.drop("Sanger", axis=1), gexp_pca], axis=1).dropna()
        samples = set.intersection(set(Y.index), set(X.index), set(M.index), set(M2.index), set(t_df.index))
        LOG.info(f"Proteomics ~ Drug: samples={len(samples)}; genes/proteins={len(genes)}; covariates={covariates.shape[1]}")

        if len(samples) < 25:
            LOG.info(f"SKIPPED {t} Drug: samples = {t_df.shape[0]}")
            continue

        lm_prot_drug = LModel(
            Y=Y.loc[samples],
            X=X.loc[samples, genes],
            M=M.loc[samples],
            M2=M2.loc[samples, genes],
            verbose=0,
        ).fit_matrix()

        lm_prot_drug["target"] = dtargets.loc[lm_prot_drug["y_id"]].values
        lm_prot_drug = PPI.ppi_annotation(
            lm_prot_drug, ppi, x_var="target", y_var="x_id", ppi_var="ppi"
        )
        lm_prot_drug = LMModels.multipletests(lm_prot_drug).sort_values("fdr")

        lm_prot_drug["chr"] = ginfo_pos.reindex(lm_prot_drug["x_id"])["chr"].values
        lm_prot_drug["chr_pos"] = ginfo_pos.reindex(lm_prot_drug["x_id"])["chr_pos"].values

        lm_prot_drug.to_csv(
            f"{RPATH}/lm_sklearn_degr_drug_{t}.csv.gz", index=False, compression="gzip"
        )
        # lm_prot_drug = pd.read_csv(f"{RPATH}/lm_sklearn_degr_drug.csv.gz")

        # CRISPR
        Y = LMModels.transform_matrix(crispr, t_type="None").T
        M = pd.concat([covariates.drop("MeanIC50", axis=1), gexp_pca], axis=1).dropna()
        samples = set.intersection(set(Y.index), set(X.index), set(M.index), set(M2.index), set(t_df.index))
        LOG.info(
            f"Proteomics ~ CRISPR: samples={len(samples)}; genes/proteins={len(genes)}"
        )

        if len(samples) < 25:
            LOG.info(f"SKIPPED {t} CRISPR: samples = {t_df.shape[0]}")
            continue

        lm_prot_crispr = LModel(
            Y=Y.loc[samples],
            X=X.loc[samples, genes],
            M=M.loc[samples],
            M2=M2.loc[samples, genes],
            verbose=0,
        ).fit_matrix()

        lm_prot_crispr = PPI.ppi_annotation(
            lm_prot_crispr, ppi, x_var="x_id", y_var="y_id", ppi_var="ppi"
        )
        lm_prot_crispr = LMModels.multipletests(lm_prot_crispr).sort_values("fdr")

        lm_prot_crispr["chr"] = ginfo_pos.reindex(lm_prot_crispr["x_id"])["chr"].values
        lm_prot_crispr["chr_pos"] = ginfo_pos.reindex(lm_prot_crispr["x_id"])["chr_pos"].values

        lm_prot_crispr.to_csv(
            f"{RPATH}/lm_sklearn_degr_crispr_{t}.csv.gz", index=False, compression="gzip"
        )
        # lm_prot_crispr = pd.read_csv(f"{RPATH}/lm_sklearn_degr_crispr.csv.gz")

    # Plots
    #
    GIPlot.gi_manhattan(lm_prot_drug)
    plt.savefig(
        f"{RPATH}/LM_drug_manhattan_plot.png",
        bbox_inches="tight",
    )
    plt.close("all")

    GIPlot.gi_manhattan(lm_prot_crispr)
    plt.savefig(
        f"{RPATH}/LM_crispr_manhattan_plot.png",
        bbox_inches="tight",
    )
    plt.close("all")