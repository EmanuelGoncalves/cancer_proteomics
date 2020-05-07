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
from natsort import natsorted
from crispy.Utils import Utils
from crispy.GIPlot import GIPlot
from Enrichment import Enrichment
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr, zscore
from sklearn.preprocessing import quantile_transform


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")
CPATH = "/Users/eg14/Data/cptac/"


# Import TCGA
#

TCGA_GEXP_FILE = f"{DPATH}/GSE62944_merged_expression_voom.tsv"
TCGA_CANCER_TYPE_FILE = f"{DPATH}/GSE62944_06_01_15_TCGA_24_CancerType_Samples.txt"

gexp = pd.read_csv(TCGA_GEXP_FILE, index_col=0, sep="\t")
gexp = gexp.loc[:, [int(c.split("-")[3][:-1]) < 10 for c in gexp.columns]]
gexp.columns = [i[:12] for i in gexp]
gexp = gexp.groupby(gexp.columns, axis=1).mean()
gexp_columns = set(gexp)

ctype = pd.read_csv(TCGA_CANCER_TYPE_FILE, sep="\t", header=None, index_col=0)[1]
ctype.index = [i[:12] for i in ctype.index]
ctype = ctype.reset_index().groupby("index")[1].first()

ctype_pal = sns.color_palette("tab20c").as_hex() + sns.color_palette("tab20b").as_hex()
ctype_pal = dict(zip(natsorted(ctype.value_counts().index), ctype_pal))


def download_files(ffile=f"{CPATH}/PDC_file_manifest_04302020_220307.csv"):
    import wget

    flist = pd.read_csv(ffile)

    for f_name, f_url in flist[["File Name", "File Download Link"]].values:
        LOG.info(f"{f_name}: {f_url}")
        wget.download(f_url, out=f"{CPATH}/{f_name}")


def sample_corr(var1, var2, idx_set):
    return spearmanr(
        var1.reindex(index=idx_set), var2.reindex(index=idx_set), nan_policy="omit"
    )

#
# def cptac_pipeline():
#     # Import proteomics matrix
#     dfile = "CPTAC2_Breast_Prospective_Collection_BI_Proteome.tmt10.tsv"
#     dmatrix = pd.read_csv(f"{CPATH}/{dfile}", sep="\t")
#
#     # Set gene ids as index
#     if dmatrix["Gene"].duplicated().any():
#         LOG.warning("Duplicated Gene IDs")
#
#     dmatrix = dmatrix.groupby("Gene").mean().drop(["Mean", "Median", "StdDev"], errors="ignore")
#     LOG.info(f"Proteins x Samples: {dmatrix.shape}")
#
#     # Select unshared peptides measurements
#     dtype = " Unshared Log Ratio" if len(
#         [c for c in dmatrix if c.endswith(" Unshared Log Ratio")]) else " Unshared Area"
#     dmatrix = dmatrix[[c for c in dmatrix if c.endswith(dtype)]]
#     dmatrix.columns = [c.split(" ")[0] for c in dmatrix]
#
#     # Check missing values
#     completeness = dmatrix.count().sum() / np.prod(dmatrix.shape)
#     if (completeness == 1) and (dtype == " Unshared Area"):
#         LOG.info("No missing values: replace 0s with NaNs")
#         dmatrix = dmatrix.replace(0, np.nan)
#         completeness = dmatrix.count().sum() / np.prod(dmatrix.shape)
#     LOG.info(f"Completeness: {completeness * 100:.1f}%")
#
#     # Log transform if peak area used
#     if dtype == " Unshared Area":
#         LOG.info("Peaks areas present: log2 scale")
#         dmatrix = dmatrix.pipe(np.log2)
#         dmatrix.columns = [c[:-3] for c in dmatrix]
#
#     # Map IDs to TCGA gene expression
#     d_idmap = {c: [gc for gc in gexp_columns if c in gc] for c in dmatrix}
#     d_idmap = {k: v[0] for k, v in d_idmap.items() if len(v) == 1}
#     dmatrix = dmatrix[d_idmap.keys()].rename(columns=d_idmap)
#     LOG.info(f"Gexp map (Proteins x Samples): {dmatrix.shape}")
#
#     # Drop duplicates samples
#     dmatrix = dmatrix.loc[:, ~dmatrix.columns.duplicated(keep=False)]
#     LOG.info(f"Drop duplicated columns: {dmatrix.shape}")
#
#     # Quantile transform per sample
#     dmatrix = pd.DataFrame(
#         quantile_transform(dmatrix.T, output_distribution="normal").T,
#         index=dmatrix.index,
#         columns=dmatrix.columns,
#     )


# Proteomics data-sets
dfiles = [
    "Human__TCGA_BRCA__BI__Proteome__QExact__01_28_2016__BI__Gene__CDAP_iTRAQ_UnsharedLogRatio_r2.cct",
    "Human__TCGA_COADREAD__VU__Proteome__Velos__01_28_2016__VU__Gene__CDAP_UnsharedPrecursorArea_r2.cct",
    "Human__TCGA_OV__JHU__Proteome__Velos__01_28_2016__JHU__Gene__CDAP_iTRAQ_UnsharedLogRatio_r2.cct",
    "Human__TCGA_OV__PNNL__Proteome__Velos___QExact__01_28_2016__PNNL__Gene__CDAP_iTRAQ_UnsharedLogRatio_r2.cct",
]

dmatrix = []
for dfile in dfiles:
    df = pd.read_csv(f"{CPATH}/linkedomics/{dfile}", sep="\t", index_col=0)

    if "COADREAD" in dfile:
        df = df.replace(0, np.nan)
        df = df / df.sum()
        df = df.pipe(np.log2)
        df = pd.DataFrame(df.pipe(zscore, nan_policy="omit"), index=df.index, columns=df.columns)

    # Simplify barcode
    dmatrix.columns = [i[:12] for i in dmatrix]

    dmatrix.append(df)

dmatrix = pd.concat(dmatrix, axis=1)
dmatrix.columns = [c.replace(".", "-") for c in dmatrix]
LOG.info(f"Assembled data-set: {dmatrix.shape}")

# Dicard poor corelated
remove_samples = {i for i in set(dmatrix) if dmatrix.loc[:, [i]].shape[1] == 2 and dmatrix.loc[:, [i]].corr().iloc[0, 1] < .4}
dmatrix = dmatrix.drop(remove_samples, axis=1)
LOG.info(f"Poor correlating samples removed: {dmatrix.shape}")

# Average replicates
dmatrix = dmatrix.groupby(dmatrix.columns, axis=1).mean()
LOG.info(f"Replicates averaged: {dmatrix.shape}")

# Map to gene-expression
d_idmap = {c: [gc for gc in gexp_columns if c in gc] for c in dmatrix}
d_idmap = {k: v[0] for k, v in d_idmap.items() if len(v) == 1}
dmatrix = dmatrix[d_idmap.keys()].rename(columns=d_idmap)
LOG.info(f"Gexp map (Proteins x Samples): {dmatrix.shape}")

# Regress-out covariates
clinical = pd.read_csv(f'{CPATH}/tcga_clinical.csv').dropna(subset=['patient.gender', 'patient.days_to_birth'])
clinical['patient.days_to_birth'] *= -1

samples = set(clinical['patient.bcr_patient_barcode']).intersection(dmatrix)

clinical_gender = clinical.groupby('patient.bcr_patient_barcode')['patient.gender'].first()
clinical_age = clinical.groupby('patient.bcr_patient_barcode')['patient.days_to_birth'].first()
clinical_disease = clinical.groupby('patient.bcr_patient_barcode')["admin.disease_code"].first()

design = pd.concat([
    clinical_age,
    clinical_gender.str.get_dummies(),
    clinical_disease.str.get_dummies(),
], axis=1).reindex(samples)
design = design.loc[:, design.std() > 0]


def rm_batch(x, y):
    ys = y.dropna()
    xs = x.loc[ys.index]

    lm = LinearRegression().fit(xs, ys)

    return ys - xs.dot(lm.coef_) - lm.intercept_


dmatrix = pd.DataFrame({p: rm_batch(design, dmatrix.loc[p, design.index]) for p in dmatrix.index}).T

# Normalise
dmatrix = dmatrix[dmatrix.count(1) > (dmatrix.shape[1] * .5)]
dmatrix = pd.DataFrame({i: Utils.gkn(dmatrix.loc[i].dropna()).to_dict() for i in dmatrix.index}).T

# Finalise and export
dmatrix.to_csv(f"{DPATH}/merged_cptac_tcga_proteomics.csv.gz", compression="gzip")
completeness = dmatrix.count().sum() / np.prod(dmatrix.shape)
LOG.info(f"Completeness: {completeness * 100:.1f}%")


#
#

s_pg_corr = pd.DataFrame(
    {
        s: sample_corr(dmatrix[s], gexp[s], set(dmatrix.index).intersection(gexp.index))
        for s in dmatrix
    },
    index=["corr", "pvalue"],
).T

gss = [
    "GO_TRANSLATION_INITIATION_FACTOR_ACTIVITY",
    "KEGG_PENTOSE_PHOSPHATE_PATHWAY",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
]

gss_genes = (
    set.union(*[Enrichment.signature(g) for g in gss])
    .intersection(gexp.index)
    .intersection(dmatrix.index)
)

s_pg_gss_corr = pd.DataFrame(
    {s: sample_corr(dmatrix[s], gexp[s], gss_genes) for s in dmatrix},
    index=["corr", "pvalue"],
).T

#
plot_df = pd.concat(
    [
        s_pg_gss_corr["corr"].rename("GeneSets"),
        s_pg_corr["corr"].rename("Overall"),
        dmatrix.median().rename("protein"),
        gexp.loc[gss_genes].median().rename("transcript"),
        ctype.rename("ctype"),
    ],
    axis=1,
).dropna()

grid = GIPlot.gi_regression("GeneSets", "Overall", plot_df, hue="ctype", palette=ctype_pal)
grid.set_axis_labels("Translation initiation\nTranscript ~ Protein", "Overall Transcript ~ Protein")
plt.savefig(
    f"{RPATH}/1.MultiOmics_CPTAC_GS_corr.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")

#
for ctypes in [["OV", "BRCA"], ["COAD", "READ"]]:
    for x_var in ["protein", "transcript"]:
        grid = GIPlot.gi_regression(
            x_var,
            "Overall",
            plot_df[plot_df["ctype"].isin(ctypes)],
            hue="ctype",
            palette=ctype_pal
        )
        grid.set_axis_labels(f"Translation initiation\n{x_var} median", "Overall Transcript ~ Protein")
        plt.savefig(
            f"{RPATH}/1.MultiOmics_CPTAC_GS_corr_continuous_{x_var}_{'_'.join(ctypes)}.pdf",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")
