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

import pandas as pd
import pkg_resources
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import Proteomics, Sample


DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("reports", "eg/")


# Samplesheet
#

ss = Sample().samplesheet


# Proteomics
#

prot = Proteomics().get_data()

#
#

info_cols = [
    "model_name",
    "tissue",
    "cancer_type",
    "model_type",
    "BROAD_ID",
    "growth",
    "msi_status",
    "ploidy",
    "growth_properties",
    "mutational_burden",
]

ss_prot = ss.reindex(prot.columns)[info_cols]
ss_prot.index.name = "model_id"
ss_prot["palette"] = pd.Series(CrispyPlot.PAL_MODEL_TYPE)[ss_prot["model_type"]].values
ss_prot.to_csv(f"{DPATH}/proteomics/E0022_P06_samplehseet.csv")
