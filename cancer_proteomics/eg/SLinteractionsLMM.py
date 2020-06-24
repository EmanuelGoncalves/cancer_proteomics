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
import pandas as pd
import pkg_resources
from sklearn.decomposition import PCA
from crispy.LMModels import LMModels, LModel
from crispy.DataImporter import (
    Proteomics,
    CRISPR,
    GeneExpression,
    DrugResponse,
    CopyNumber,
    PPI,
)


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


if __name__ == "__main__":
    # PPI
    #
    ppi = PPI().build_string_ppi(score_thres=900)

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

    # Custom features
    #
    patt = pd.read_csv(
        f"{RPATH}/SampleProteinTranscript_attenuation.csv.gz", index_col=0
    )
    cins = CopyNumber().genomic_instability()

    cfeatures = pd.concat(
        [
            patt["attenuation"].rename("ProteinAttenuation"),
            cins.rename("CopyNumberInstability"),
            pd.get_dummies(prot_obj.ss["msi_status"])["MSI"],
            prot_obj.ss.reindex(columns=["ploidy", "mutational_burden"]),
        ],
        axis=1,
    ).dropna()

    # Covariates
    #
    covariates = pd.concat(
        [
            prot_obj.ss["growth"],
            pd.get_dummies(prot_obj.ss["media"]),
            pd.get_dummies(prot_obj.ss["growth_properties"]),
            pd.get_dummies(prot_obj.ss["tissue"])["Haematopoietic and Lymphoid"].rename("Haem"),
            pd.get_dummies(crispr_obj.merged_institute)["Sanger"],
            prot_obj.reps.rename("RepsCorrelation"),
            drespo.mean().rename("MeanIC50"),
        ],
        axis=1,
    )

    # Dimension reduction
    #
    gexp_pca = pd.DataFrame(
        PCA(n_components=30).fit_transform(gexp.T), index=gexp.columns
    ).add_prefix("PC")

    # LMMs: Proteomics
    #
    X = LMModels.transform_matrix(prot, t_type="None", fillna_func=None).T
    M2 = LMModels.transform_matrix(gexp, t_type="None").T
    genes = set.intersection(set(X), set(M2))

    # Drug
    Y = LMModels.transform_matrix(drespo, t_type="None").T
    M = pd.concat([covariates.drop("Sanger", axis=1), gexp_pca], axis=1).dropna()
    samples = set.intersection(set(Y.index), set(X.index), set(M.index), set(M2.index))
    LOG.info(f"Proteomics ~ Drug: samples={len(samples)}; genes/proteins={len(genes)}")

    lm_prot_drug = LModel(
        Y=Y.loc[samples],
        X=X.loc[samples, genes],
        M=M.loc[samples],
        M2=M2.loc[samples, genes],
    ).fit_matrix()

    lm_prot_drug["target"] = dtargets.loc[lm_prot_drug["y_id"]].values
    lm_prot_drug = PPI.ppi_annotation(
        lm_prot_drug, ppi, x_var="target", y_var="x_id", ppi_var="ppi"
    )

    lm_prot_drug.to_csv(
        f"{RPATH}/lm_sklearn_degr_drug.csv.gz", index=False, compression="gzip"
    )

    # CRISPR
    Y = LMModels.transform_matrix(crispr, t_type="None").T
    M = pd.concat([covariates.drop("MeanIC50", axis=1), gexp_pca], axis=1).dropna()
    samples = set.intersection(set(Y.index), set(X.index), set(M.index), set(M2.index))
    LOG.info(
        f"Proteomics ~ CRISPR: samples={len(samples)}; genes/proteins={len(genes)}"
    )

    lm_prot_crispr = LModel(
        Y=Y.loc[samples],
        X=X.loc[samples, genes],
        M=M.loc[samples],
        M2=M2.loc[samples, genes],
    ).fit_matrix()

    lm_prot_crispr = PPI.ppi_annotation(
        lm_prot_crispr, ppi, x_var="x_id", y_var="y_id", ppi_var="ppi"
    )

    lm_prot_crispr.to_csv(
        f"{RPATH}/lm_sklearn_degr_crispr.csv.gz", index=False, compression="gzip"
    )
