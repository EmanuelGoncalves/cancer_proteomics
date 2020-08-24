#!/usr/bin/env python
# Copyright (C) 2020 Emanuel Goncalves

import os
import logging
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt
from GIPlot import GIPlot
from crispy.DataImporter import Proteomics


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")
TPATH = pkg_resources.resource_filename("tables", "/")


if __name__ == "__main__":

    # Proteomics
    #
    prot_obj = Proteomics()
    prot = prot_obj.filter()
    LOG.info(f"Prot: {prot.shape}")

    # Assemble tissue association files
    #
    FDR = 0.01

    lm_prot_drug, lm_prot_crispr = [], []

    for t, t_df in prot_obj.ss.groupby("tissue"):
        LOG.info(f"Tissue = {t}")

        lm_drug_file = f"{RPATH}/lm_sklearn_degr_drug_{t}.csv.gz"
        if os.path.exists(lm_drug_file):
            lm_drug = pd.read_csv(lm_drug_file)
            lm_prot_drug.append(lm_drug.query(f"fdr < {FDR}").assign(tissue=t))

        lm_crispr_file = f"{RPATH}/lm_sklearn_degr_crispr_{t}.csv.gz"
        if os.path.exists(lm_crispr_file):
            lm_cirspr = pd.read_csv(lm_crispr_file)
            lm_prot_crispr.append(lm_cirspr.query(f"fdr < {FDR}").assign(tissue=t))

    lm_prot_drug = pd.concat(lm_prot_drug)
    lm_prot_drug.sort_values("fdr").to_csv(
        f"{RPATH}/lm_sklearn_degr_drug_per_tissue.csv.gz", index=False, compression="gzip"
    )

    lm_prot_crispr = pd.concat(lm_prot_crispr)
    lm_prot_crispr.sort_values("fdr").to_csv(
        f"{RPATH}/lm_sklearn_degr_crispr_per_tissue.csv.gz", index=False, compression="gzip"
    )
