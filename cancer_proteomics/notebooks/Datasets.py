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

import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from crispy import CrispyPlot
from cancer_proteomics.notebooks import DataImport


RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/DIANN/")

# ### Imports

# CMP samplesheet
cmp = DataImport.read_cmp_samplesheet().reset_index()
cmp["CCLE_ID_SHORT"] = cmp["CCLE_ID"].apply(lambda v: v.split("_")[0] if str(v).lower() != 'nan' else np.nan)

# Proteomics
prot = DataImport.read_protein_matrix(map_protein=True)

# Read proteomics BROAD (Proteins x Cell lines)
prot_broad = DataImport.read_protein_matrix_broad()
prot_broad = prot_broad.rename(columns=cmp.set_index("CCLE_ID")["model_id"])

# Read Transcriptomics
gexp = DataImport.read_gene_matrix()

# Read CRISPR
crispr = DataImport.read_crispr_matrix()
crispr = crispr.rename(columns=cmp.set_index("BROAD_ID")["model_id"])

# Read Methylation
methy = DataImport.read_methylation_matrix()

# Read Drug-response
drespo = DataImport.read_drug_response()

# Read CTD2
ctd2 = DataImport.read_ctrp()
ctd2 = ctd2.rename(columns=cmp.groupby("CCLE_ID_SHORT")["model_id"].first())

# Read copy number
cnv = DataImport.read_copy_number()

# Read PRISM
prism = DataImport.read_prism_depmap_19q4()
prism = prism.rename(columns=cmp.set_index("BROAD_ID")["model_id"])

# Read WES
wes = DataImport.read_wes()

# #
samples = set(cmp["model_id"])

datasets = [
    ("Proteomics (CMRI&Sanger)", set(prot)),
    ("Methylation", set(methy)),
    ("Mutation", set(wes["model_id"])),
    ("Copy number", set(cnv)),
    ("Transcriptomics", set(gexp)),
    ("Proteomics (CCLE)", set(prot_broad)),
    ("Drug response (Sanger)", set(drespo)),
    ("Drug response (CTD2)", set(ctd2)),
    ("Drug response (PRISM)", set(prism)),
    ("Gene essentiality (Broad&Sanger)", set(crispr)),
]

plot_df = pd.DataFrame({n: {s: s in dsamples for s in samples} for n, dsamples in datasets}).astype(int).T
plot_df = plot_df[plot_df.sum().sort_values(ascending=False).index]
plot_df = plot_df.loc[:, plot_df.sum() > 0]
plot_df = plot_df.loc[plot_df.sum(1).sort_values(ascending=False).index]

plot_df = plot_df.loc[[
    'Proteomics (CMRI&Sanger)',
    'Transcriptomics',
    'Drug response (Sanger)',
    'Mutation',
    'Copy number',
    'Methylation',
    'Drug response (CTD2)',
    'Gene essentiality (Broad&Sanger)',
    'Drug response (PRISM)',
    'Proteomics (CCLE)',
]]

plot_df.T.to_csv(f"{RPATH}/Datasets_overlap.csv")

nsamples = plot_df.sum(1)

plot_df.loc["Proteomics (CMRI&Sanger)"] *= 2
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', ["white", CrispyPlot.PAL_DTRACE[2], CrispyPlot.PAL_DTRACE[1]], 3)

_, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=600)

sns.heatmap(plot_df, xticklabels=False, cmap=cmap, cbar=False, ax=ax)

for i, c in enumerate(plot_df.index):
    ax.text(20, i + 0.5, f"N={nsamples[c]}", ha="left", va="center", fontsize=6)

ax.set_title(f"Cell Model Passports Database\n(N={plot_df.shape[1]})")

plt.savefig(f"{RPATH}/Datasets_overlap.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/Datasets_overlap.png", bbox_inches="tight")
plt.close("all")
