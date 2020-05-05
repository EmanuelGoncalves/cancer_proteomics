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
from natsort import natsorted
from crispy.Utils import Utils
from crispy.GIPlot import GIPlot
from Enrichment import Enrichment
from scipy.stats import spearmanr, pearsonr
from crispy.MOFA import MOFA, MOFAPlot
from crispy.CrispyPlot import CrispyPlot
from cancer_proteomics.eg.LMModels import LMModels
from sklearn.preprocessing import quantile_transform
from cancer_proteomics.eg.SLinteractionsSklearn import LModel
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
CPATH = "/Users/eg14/Data/cptac/"


# Import TCGA
#

TCGA_GEXP_FILE = f"{DPATH}/GSE62944_merged_expression_voom.tsv"
TCGA_CANCER_TYPE_FILE = f"{DPATH}/GSE62944_06_01_15_TCGA_24_CancerType_Samples.txt"

gexp = pd.read_csv(TCGA_GEXP_FILE, index_col=0, sep="\t")
gexp = gexp.loc[:, [int(c.split("-")[3][:-1]) < 10 for c in gexp.columns]]
gexp_columns = set(gexp)

ctype = pd.read_csv(TCGA_CANCER_TYPE_FILE, sep="\t", header=None, index_col=0)[1]
ctype = ctype.append(pd.Series({x: "Normal" for x in gexp.columns if x not in ctype}))

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


def cptac_pipeline():
    # Import proteomics matrix
    dfile = "CPTAC2_Breast_Prospective_Collection_BI_Proteome.tmt10.tsv"
    dmatrix = pd.read_csv(f"{CPATH}/{dfile}", sep="\t")

    # Set gene ids as index
    if dmatrix["Gene"].duplicated().any():
        LOG.warning("Duplicated Gene IDs")

    dmatrix = dmatrix.groupby("Gene").mean().drop(["Mean", "Median", "StdDev"], errors="ignore")
    LOG.info(f"Proteins x Samples: {dmatrix.shape}")

    # Select unshared peptides measurements
    dtype = " Unshared Log Ratio" if len(
        [c for c in dmatrix if c.endswith(" Unshared Log Ratio")]) else " Unshared Area"
    dmatrix = dmatrix[[c for c in dmatrix if c.endswith(dtype)]]
    dmatrix.columns = [c.split(" ")[0] for c in dmatrix]

    # Check missing values
    completeness = dmatrix.count().sum() / np.prod(dmatrix.shape)
    if (completeness == 1) and (dtype == " Unshared Area"):
        LOG.info("No missing values: replace 0s with NaNs")
        dmatrix = dmatrix.replace(0, np.nan)
        completeness = dmatrix.count().sum() / np.prod(dmatrix.shape)
    LOG.info(f"Completeness: {completeness * 100:.1f}%")

    # Log transform if peak area used
    if dtype == " Unshared Area":
        LOG.info("Peaks areas present: log2 scale")
        dmatrix = dmatrix.pipe(np.log2)
        dmatrix.columns = [c[:-3] for c in dmatrix]

    # Map IDs to TCGA gene expression
    d_idmap = {c: [gc for gc in gexp_columns if c in gc] for c in dmatrix}
    d_idmap = {k: v[0] for k, v in d_idmap.items() if len(v) == 1}
    dmatrix = dmatrix[d_idmap.keys()].rename(columns=d_idmap)
    LOG.info(f"Gexp map (Proteins x Samples): {dmatrix.shape}")

    # Drop duplicates samples
    dmatrix = dmatrix.loc[:, ~dmatrix.columns.duplicated(keep=False)]
    LOG.info(f"Drop duplicated columns: {dmatrix.shape}")

    # Quantile transform per sample
    dmatrix = pd.DataFrame(
        quantile_transform(dmatrix.T, output_distribution="normal").T,
        index=dmatrix.index,
        columns=dmatrix.columns,
    )


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
        df = df.replace(0, np.nan).dropna(how="all")

    df = pd.DataFrame(
        quantile_transform(df.T, output_distribution="normal").T,
        index=df.index,
        columns=df.columns,
    )

    dmatrix.append(df)

dmatrix = pd.concat(dmatrix, axis=1)
dmatrix.columns = [c.replace(".", "-") for c in dmatrix]

d_idmap = {c: [gc for gc in gexp_columns if c in gc] for c in dmatrix}
d_idmap = {k: v[0] for k, v in d_idmap.items() if len(v) == 1}
dmatrix = dmatrix[d_idmap.keys()].rename(columns=d_idmap)
LOG.info(f"Gexp map (Proteins x Samples): {dmatrix.shape}")

dmatrix = dmatrix.groupby(dmatrix.columns, axis=1).agg(np.nanmean)
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
        dmatrix.loc[gss_genes].median().rename("protein"),
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
