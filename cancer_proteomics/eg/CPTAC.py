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
from crispy.Utils import Utils

LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")


CPTAC_DPATH = "/Users/eg14/Data/cptac/"
CPTAC_DATASETS = [
    ("BRCA", "Human__TCGA_BRCA__BI__Proteome__QExact__01_28_2016__BI__Gene__CDAP_iTRAQ_UnsharedLogRatio_r2.cct"),
    #("COREAD", "Human__TCGA_COADREAD__VU__Proteome__Velos__01_28_2016__VU__Gene__CDAP_UnsharedPrecursorArea_r2.cct"),
    ("OV", "Human__TCGA_OV__JHU__Proteome__Velos__01_28_2016__JHU__Gene__CDAP_iTRAQ_UnsharedLogRatio_r2.cct"),
    ("OV", "Human__TCGA_OV__PNNL__Proteome__Velos___QExact__01_28_2016__PNNL__Gene__CDAP_iTRAQ_UnsharedLogRatio_r2.cct"),
]

TCGA_GEXP_FILE = f"{DPATH}/GSE62944_merged_expression_voom.tsv"
TCGA_CANCER_TYPE_FILE = f"{DPATH}/GSE62944_06_01_15_TCGA_24_CancerType_Samples.txt"


if __name__ == "__main__":
    # Import gene-expression
    #
    gexp = pd.read_csv(TCGA_GEXP_FILE, index_col=0, sep="\t")
    gexp = gexp.loc[:, [int(c.split("-")[3][:-1]) < 10 for c in gexp.columns]]
    gexp.columns = [i[:12] for i in gexp]
    gexp = gexp.groupby(gexp.columns, axis=1).mean()
    gexp_columns = set(gexp)

    # Stromal
    #
    stromal = pd.read_excel(
        f"{DPATH}/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx", index_col=0
    )
    stromal = stromal.loc[[int(c.split("-")[3][:-1]) < 10 for c in stromal.index]]
    stromal.index = [i[:12] for i in stromal.index]

    stromal_count = stromal.index.value_counts()
    stromal = stromal[~stromal.index.isin(stromal_count[stromal_count != 1].index)]

    # Import proteomics data-sets
    #
    dmatrix, ms_type, ctypes = [], [], []
    for ctype, dfile in CPTAC_DATASETS:
        df = pd.read_csv(f"{CPTAC_DPATH}/linkedomics/{dfile}", sep="\t", index_col=0)

        if "COADREAD" in dfile:
            df = df.replace(0, np.nan)
            df = df.pipe(np.log2)

        df = pd.DataFrame(
            {i: Utils.gkn(df.loc[i].dropna()).to_dict() for i in df.index}
        ).T

        # Simplify barcode
        df.columns = [i[:12].replace(".", "-") for i in df]

        # Cancer type
        ctypes.append(pd.Series(ctype, index=df.columns))

        # MS type
        ms_type.append(
            pd.Series("LF" if "COADREAD" in dfile else "TMT", index=df.columns)
        )

        dmatrix.append(df)

    ms_type = pd.concat(ms_type).reset_index().groupby("index").first()[0]
    dmatrix = pd.concat(dmatrix, axis=1)
    ctypes = pd.concat(ctypes).reset_index().groupby("index").first()[0]
    LOG.info(f"Assembled data-set: {dmatrix.shape}")

    # Dicard poor corelated
    remove_samples = {
        i
        for i in set(dmatrix)
        if dmatrix.loc[:, [i]].shape[1] == 2
        and dmatrix.loc[:, [i]].corr().iloc[0, 1] < 0.4
    }
    dmatrix = dmatrix.drop(remove_samples, axis=1)
    LOG.info(f"Poor correlating samples removed: {dmatrix.shape}")

    # Average replicates
    dmatrix = dmatrix.groupby(dmatrix.columns, axis=1).mean()
    LOG.info(f"Replicates averaged: {dmatrix.shape}")

    # # Map to gene-expression
    # d_idmap = {c: [gc for gc in gexp_columns if c in gc] for c in dmatrix}
    # d_idmap = {k: v[0] for k, v in d_idmap.items() if len(v) == 1}
    # dmatrix = dmatrix[d_idmap.keys()].rename(columns=d_idmap)
    # LOG.info(f"Gexp map (Proteins x Samples): {dmatrix.shape}")

    # Finalise and export
    dmatrix.to_csv(f"{RPATH}/merged_cptac_tcga_proteomics.csv.gz", compression="gzip")
    ctypes[dmatrix.columns].to_csv(f"{RPATH}/merged_cptac_tcga_proteomics_ctype.csv.gz", compression="gzip")

    # dmatrix = pd.read_csv(f"{RPATH}/merged_cptac_tcga_proteomics.csv.gz", index_col=0)
    completeness = dmatrix.count().sum() / np.prod(dmatrix.shape)
    LOG.info(f"Completeness: {completeness * 100:.1f}%")
