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
from scipy.stats import skew
from crispy.DataImporter import PPI
from sklearn.decomposition import PCA
from crispy.LMModels import LMModels, LModel
from cancer_proteomics.notebooks import DataImport


LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/")


# ### Imports

# PPI
ppi = PPI(ddir=PPIPATH).build_string_ppi(score_thres=900)

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
prot = DataImport.read_protein_matrix(map_protein=True)

# Read Transcriptomics
gexp = DataImport.read_gene_matrix()

# Read CRISPR
crispr = DataImport.read_crispr_matrix()
crispr_institute = DataImport.read_crispr_institute()[crispr.columns]
crispr_skew = crispr.apply(skew, axis=1, nan_policy="omit").astype(float)

# Read Drug-response
drespo = DataImport.read_drug_response()
dtargets = DataImport.read_drug_target()
drespo_skew = drespo.apply(skew, axis=1, nan_policy="omit").astype(float)


# ### Gene expression dimension reduction

gexp_pca = pd.DataFrame(
    PCA(n_components=10).fit_transform(gexp.T), index=gexp.columns
).add_prefix("PC")


# ### Linear regressions

X = LMModels.transform_matrix(prot, t_type="None", fillna_func=None).T
M2 = LMModels.transform_matrix(gexp, t_type="None").T
features = set.intersection(set(X), set(M2))

# ## Drug response
Y = LMModels.transform_matrix(drespo, t_type="None").T

# Covariates with and without gene expression (transcript and PCA) as covariates
covs_drug = (
    pd.concat(
        [
            ss["media"].str.get_dummies(),
            drespo.mean().rename("MeanIC50"),
            ss["growth_properties"].str.get_dummies(),
            ss[["replicates_correlation", "growth", "ploidy"]],
            ss["Tissue_type"].str.get_dummies()["Haematopoietic and Lymphoid"],
        ],
        axis=1,
    )
    .reindex(index=prot.columns)
    .dropna()
)

M_without_gexp = pd.concat([covs_drug], axis=1).dropna()
M_with_gexp = pd.concat([covs_drug, gexp_pca], axis=1).dropna()

# Overlapping samples
samples = set.intersection(
    set(Y.index), set(X.index), set(M_with_gexp.index), set(M2.index)
)

# Associations without gene expression as covaraites
lm_prot_drug = LModel(
    Y=Y.loc[samples], X=X.loc[samples, features], M=M_without_gexp.loc[samples]
).fit_matrix()
lm_prot_drug = lm_prot_drug.query("n > 60")
lm_prot_drug = LMModels.multipletests(lm_prot_drug).sort_values("fdr")

# Associations with gene expression as covariate
lm_prot_drug_gexp = LModel(
    Y=Y.loc[samples],
    X=X.loc[samples, features],
    M=M_with_gexp.loc[samples],
    M2=M2.loc[samples, features],
).fit_matrix()
lm_prot_drug_gexp = lm_prot_drug_gexp.query("n > 60")
lm_prot_drug_gexp = LMModels.multipletests(lm_prot_drug_gexp).sort_values("fdr")

# Merge associations
lm_drug = pd.concat(
    [
        lm_prot_drug_gexp.set_index(["y_id", "x_id"]),
        lm_prot_drug.set_index(["y_id", "x_id"])[["beta", "lr", "pval", "fdr"]].add_prefix("nc_"),
    ],
    axis=1,
).reset_index()

# Annotate merged table
lm_drug = DataImport.lm_ppi_annotate_table(
    lm_drug, ppi, drespo_skew, drug_targets=dtargets
)
lm_drug = lm_drug.sort_values("fdr")
lm_drug.to_csv(
    f"{TPATH}/lm_sklearn_degr_drug_annotated_DIANN.csv.gz", compression="gzip", index=False
)

# ## CRISPR
Y = LMModels.transform_matrix(crispr, t_type="None").T

# Covariates with and without gene expression (transcript and PCA) as covariates
covs_crispr = (
    pd.concat(
        [
            ss["media"].str.get_dummies(),
            crispr_institute.str.get_dummies(),
            ss["growth_properties"].str.get_dummies(),
            ss[["replicates_correlation", "growth", "ploidy"]],
            ss["Tissue_type"].str.get_dummies()["Haematopoietic and Lymphoid"],
        ],
        axis=1,
    )
    .reindex(index=prot.columns)
    .dropna()
)

M_without_gexp = pd.concat([covs_crispr], axis=1).dropna()
M_with_gexp = pd.concat([covs_crispr, gexp_pca], axis=1).dropna()

# Overlapping samples
samples = set.intersection(
    set(Y.index), set(X.index), set(M_with_gexp.index), set(M2.index)
)

# Associations without gene expression
lm_prot_crispr = LModel(
    Y=Y.loc[samples], X=X.loc[samples, features], M=M_without_gexp.loc[samples]
).fit_matrix()
lm_prot_crispr = lm_prot_crispr.query("n > 60")
lm_prot_crispr = LMModels.multipletests(lm_prot_crispr).sort_values("fdr")

# Associations with gene expression as covariate
lm_prot_crispr_gexp = LModel(
    Y=Y.loc[samples],
    X=X.loc[samples, features],
    M=M_with_gexp.loc[samples],
    M2=M2.loc[samples, features],
).fit_matrix()
lm_prot_crispr_gexp = lm_prot_crispr_gexp.query("n > 60")
lm_prot_crispr_gexp = LMModels.multipletests(lm_prot_crispr_gexp).sort_values("fdr")

# Merge associations
lm_crispr = pd.concat(
    [
        lm_prot_crispr_gexp.set_index(["y_id", "x_id"]),
        lm_prot_crispr.set_index(["y_id", "x_id"])[
            ["beta", "lr", "pval", "fdr"]
        ].add_prefix("nc_"),
    ],
    axis=1,
).reset_index()

# Annotate merged table
lm_crispr = DataImport.lm_ppi_annotate_table(lm_crispr, ppi, crispr_skew)
lm_crispr = lm_crispr.sort_values("fdr")
lm_crispr.to_csv(
    f"{TPATH}/lm_sklearn_degr_crispr_annotated_DIANN.csv.gz", compression="gzip", index=False
)
