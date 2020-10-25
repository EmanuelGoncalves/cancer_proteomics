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
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from crispy.MOFA import MOFA, MOFAPlot
from cancer_proteomics.notebooks import DataImport, two_vars_correlation, PALETTE_TTYPE


LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/")


# ### Imports

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
prot = DataImport.read_protein_matrix(map_protein=True, min_measurements=300)

# Read proteomics BROAD (Proteins x Cell lines)
prot_broad = DataImport.read_protein_matrix_broad()

# Read Transcriptomics
gexp = DataImport.read_gene_matrix()

# Read CRISPR
crispr = DataImport.read_crispr_matrix()

# Read Methylation
methy = DataImport.read_methylation_matrix()

# Read Drug-response
drespo = DataImport.read_drug_response(min_measurements=300)


# ### Covariates

covariates = pd.concat(
    [
        ss["CopyNumberAttenuation"],
        ss["GeneExpressionAttenuation"],
        ss["EMT"],
        ss["Proteasome"],
        ss["TranslationInitiation"],
        ss["CopyNumberInstability"],
        prot.loc[["CDH1", "VIM"]].T.add_suffix("_prot"),
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_gexp"),
        pd.get_dummies(ss["media"]),
        pd.get_dummies(ss["growth_properties"]),
        pd.get_dummies(ss["tissue"])[["Haematopoietic and Lymphoid", "Lung"]],
        ss[["ploidy", "mutational_burden", "growth", "size"]],
        ss["replicates_correlation"].rename("RepsCorrelation"),
        prot.mean().rename("MeanProteomics"),
        prot_broad.mean().rename("MeanProteomicsBroad"),
        methy.mean().rename("MeanMethylation"),
        drespo.mean().rename("MeanDrugResponse"),
    ],
    axis=1,
)

# ### MOFA

# Group Haematopoietic and Lymphoid cell lines separetly from the rest
groupby = ss.loc[prot.columns, "tissue"].apply(
    lambda v: "Haem" if v == "Haematopoietic and Lymphoid" else "Other"
)

mofa = MOFA(
    views=dict(proteomics=prot, proteomics_broad=prot_broad, transcriptomics=gexp, methylation=methy, drespo=drespo),
    covariates=dict(
        proteomics=covariates[["MeanProteomics"]].dropna(),
    ),
    groupby=groupby,
    iterations=2000,
    use_overlap=False,
    convergence_mode="fast",
    factors_n=15,
    from_file=f"{TPATH}/MultiOmics.hdf5",
    verbose=2,
)


# ### Factors integrated with other measurements
#
n_factors_corr = {}
for f in mofa.factors:
    n_factors_corr[f] = {}

    for c in covariates:
        fc_samples = list(covariates.reindex(mofa.factors[f].index)[c].dropna().index)
        n_factors_corr[f][c] = two_vars_correlation(
            mofa.factors[f][fc_samples], covariates[c][fc_samples]
        )["corr"]
n_factors_corr = pd.DataFrame(n_factors_corr)

# Factor clustermap
MOFAPlot.factors_corr_clustermap(mofa)
plt.savefig(f"{RPATH}/MultiOmics_factors_corr_clustermap.pdf", bbox_inches="tight")
plt.savefig(
    f"{RPATH}/MultiOmics_factors_corr_clustermap.png", bbox_inches="tight", dpi=600
)
plt.close("all")

# Variance explained across data-sets
MOFAPlot.variance_explained_heatmap(mofa)
plt.savefig(f"{RPATH}/MultiOmics_factors_rsquared_heatmap.pdf", bbox_inches="tight")
plt.savefig(
    f"{RPATH}/MultiOmics_factors_rsquared_heatmap.png", bbox_inches="tight", dpi=600
)
plt.close("all")

# Covairates correlation heatmap
MOFAPlot.covariates_heatmap(n_factors_corr, mofa, ss["tissue"])
plt.savefig(
    f"{RPATH}/MultiOmics_factors_covariates_clustermap.pdf", bbox_inches="tight"
)
plt.savefig(
    f"{RPATH}/MultiOmics_factors_covariates_clustermap.png",
    bbox_inches="tight",
    dpi=600,
)
plt.close("all")


# ### MOFA Factor 1 and 2
f_x, f_y = "F1", "F2"

plot_df = pd.concat(
    [
        mofa.factors[[f_x, f_y]],
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_transcriptomics"),
        prot.loc[["CDH1", "VIM"]].T.add_suffix("_proteomics"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
)

# Tissue plot
ax = GIPlot.gi_tissue_plot(f_x, f_y, plot_df, plot_reg=False, pal=PALETTE_TTYPE)
ax.set_xlabel(f"Factor {f_x[1:]}")
ax.set_ylabel(f"Factor {f_y[1:]}")
plt.savefig(f"{RPATH}/MultiOmics_{f_x}_{f_y}_tissue_plot.pdf", bbox_inches="tight")
plt.savefig(
    f"{RPATH}/MultiOmics_{f_x}_{f_y}_tissue_plot.png", bbox_inches="tight", dpi=600
)
plt.close("all")

# Continous annotation
for z in ["VIM_proteomics", "CDH1_proteomics"]:
    ax = GIPlot.gi_continuous_plot(f_x, f_y, z, plot_df, cbar_label=z.replace("_", " "))
    ax.set_xlabel(f"Factor {f_x[1:]}")
    ax.set_ylabel(f"Factor {f_y[1:]}")
    plt.savefig(
        f"{RPATH}/MultiOmics_{f_x}_{f_y}_continous_{z}.pdf", bbox_inches="tight"
    )
    plt.savefig(
        f"{RPATH}/MultiOmics_{f_x}_{f_y}_continous_{z}.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close("all")

# Export matrix
plot_df = pd.concat([mofa.rsquare[k].T.add_prefix(f"{k}_").T for k in mofa.rsquare] + [n_factors_corr.T.add_prefix(f"Corr_").T])
