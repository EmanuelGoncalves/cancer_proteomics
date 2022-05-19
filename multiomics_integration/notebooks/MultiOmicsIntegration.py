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
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from crispy.LMModels import LModel
from crispy.MOFA import MOFA, MOFAPlot
from crispy.Enrichment import Enrichment
from sklearn.preprocessing import scale
from cancer_proteomics.notebooks import DataImport, two_vars_correlation, PALETTE_TTYPE


LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/DIANN/")

# ### Imports

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
prot = DataImport.read_protein_matrix(map_protein=True, min_measurements=60)
prot = pd.DataFrame(scale(prot.T), index=prot.columns, columns=prot.index).T

# Read proteomics BROAD (Proteins x Cell lines)
prot_broad = DataImport.read_protein_matrix_broad()

# Read Transcriptomics
gexp = DataImport.read_gene_matrix()

# Read CRISPR
crispr = DataImport.read_crispr_matrix()
crispr_institute = DataImport.read_crispr_institute()[crispr.columns]

# Read Methylation
methy = DataImport.read_methylation_matrix()

# Read Drug-response
drespo = DataImport.read_drug_response(min_measurements=3)
dtargets = DataImport.read_drug_target()

# Mobems
mobems = DataImport.read_mobem()


# ### Covariates
covariates = pd.concat(
    [
        ss["CopyNumberAttenuation"],
        ss["GeneExpressionCorrelation"],
        ss["Proteasome"],
        ss["TranslationInitiation"],
        ss["CopyNumberInstability"],
        ss["EMT"],
        prot.loc[["CDH1", "VIM"]].T.add_suffix("_prot"),
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_gexp"),
        pd.get_dummies(ss["media"]),
        pd.get_dummies(ss["growth_properties"]),
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
groupby = ss.loc[prot.columns, "Tissue_type"].apply(
    lambda v: "Haem" if v == "Haematopoietic and Lymphoid" else "Other"
)

mofa = MOFA(
    views=dict(
        proteomics=prot,
        proteomics_broad=prot_broad,
        transcriptomics=gexp,
        methylation=methy,
        drespo=drespo,
    ),
    covariates=dict(
        proteomics=covariates[["MeanProteomics"]].dropna(),
    ),
    groupby=groupby,
    iterations=2000,
    use_overlap=False,
    convergence_mode="slow",
    factors_n=15,
    from_file=f"{TPATH}/MultiOmics_DIANN.hdf5",
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

n_factors_pval = {}
for f in mofa.factors:
    n_factors_pval[f] = {}

    for c in covariates:
        fc_samples = list(covariates.reindex(mofa.factors[f].index)[c].dropna().index)
        n_factors_pval[f][c] = two_vars_correlation(
            mofa.factors[f][fc_samples], covariates[c][fc_samples]
        )
n_factors_pval = pd.DataFrame(n_factors_pval)

# Factor clustermap
MOFAPlot.factors_corr_clustermap(mofa)
plt.savefig(f"{RPATH}/MultiOmics_factors_corr_clustermap.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/MultiOmics_factors_corr_clustermap.png", bbox_inches="tight", dpi=600)
plt.close("all")

# Variance explained across data-sets
MOFAPlot.variance_explained_heatmap(mofa)
plt.savefig(f"{RPATH}/MultiOmics_factors_rsquared_heatmap.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/MultiOmics_factors_rsquared_heatmap.png", bbox_inches="tight", dpi=600)
plt.close("all")

# ### Overview heatmap
# Tissue-specific proteins
tissue_proteins = pd.read_csv(f"{DPATH}/tissue_specific_protein_list.txt", sep="\t")
tissue_proteins["id"] = tissue_proteins["Protein"].apply(lambda v: v.split(";")[1]).values
tissue_proteins["GeneSymbol"] = DataImport.map_gene_name().reindex(tissue_proteins["id"])["GeneSymbol"].values

# Build gmt dict
gmts = tissue_proteins.groupby("Tissue")["GeneSymbol"].agg(set).to_dict()
gmts = {k: v for k, v in gmts.items() if len(v) >= 10}

# Factor enrichment
enr_obj = Enrichment(dict(tissue=gmts), sig_min_len=10)
enr = pd.DataFrame(
    {
        f: enr_obj.gsea_enrichments(mofa.weights["proteomics"][f])["e_score"].to_dict()
        for f in mofa.weights["proteomics"]
    }
)

# Export matrix
covs_main = ["EMT", "VIM_prot", "VIM_gexp", "CDH1_prot", "CDH1_gexp"]
covs_supp = [f for f in n_factors_corr.index if f not in covs_main]

plot_df_main = pd.concat(
    [mofa.rsquare[k].T.add_prefix(f"{k}_").T for k in mofa.rsquare]
    + [n_factors_corr.T[covs_main].add_prefix(f"Main_").T]
    + [enr.T.add_prefix(f"Tissue_").T]
)

plot_df_supp = pd.concat(
    [mofa.rsquare[k].T.add_prefix(f"{k}_").T for k in mofa.rsquare]
    + [n_factors_corr.T[covs_supp].add_prefix(f"Supp_").T]
    + [enr.T.add_prefix(f"Tissue_").T]
)

for plot_df, ratio, name in [
    (plot_df_main.copy(), 3, "Main"),
    (plot_df_supp.copy(), 6, "Supp"),
]:
    f, axs = plt.subplots(
        4,
        1,
        sharex="col",
        sharey="none",
        gridspec_kw={"height_ratios": [3] * 2 + [ratio] + [3]},
        figsize=(plot_df.shape[1] * 0.22, plot_df.shape[0] * 0.22),
    )

    for i, n in enumerate(["Haem", "Other"]):
        df = plot_df[[i.startswith(n) for i in plot_df.index]]
        df.index = [i.split("_")[1] for i in df.index]
        g = sns.heatmap(
            df,
            cmap="Blues",
            annot=True,
            cbar=False,
            fmt=".1f",
            linewidths=0.5,
            ax=axs[i],
            vmin=0,
            annot_kws={"fontsize": 5},
        )
        axs[i].set_ylabel(f"{n} cell lines")

    df = plot_df[[i.split("_")[0] in ["Main", "Supp"] for i in plot_df.index]]
    df.index = [i.replace("_prot", " (Proteomics)").replace("_gexp", " (Transcriptomics)").split("_")[1] for i in df.index]
    sns.heatmap(
        df,
        cmap="RdYlGn",
        center=0,
        annot=True,
        cbar=False,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"fontsize": 5},
        ax=axs[2],
    )

    df = plot_df[[i.split("_")[0] in ["Tissue"] for i in plot_df.index]]
    df.index = [i.split("_")[1] for i in df.index]
    sns.heatmap(
        df,
        cmap=sns.diverging_palette(240, 10, as_cmap=True, sep=100),
        center=0,
        annot=True,
        cbar=False,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"fontsize": 5},
        ax=axs[3],
    )

    plt.subplots_adjust(hspace=0.1)

    plt.savefig(f"{RPATH}/MultiOmics_clustermap_merged_{name}.pdf", bbox_inches="tight")
    plt.savefig(f"{RPATH}/MultiOmics_clustermap_merged_{name}.png", bbox_inches="tight", dpi=600)
    plt.close("all")


# ### MOFA Factor 1 and 2
f_x, f_y = "F1", "F2"

plot_df = pd.concat(
    [
        mofa.factors[[f_x, f_y]],
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_transcriptomics"),
        prot.loc[["CDH1", "VIM"]].T.add_suffix("_proteomics"),
        ss["Tissue_type"],
    ],
    axis=1,
    sort=False,
)

# Tissue plot
ax = GIPlot.gi_tissue_plot(f_x, f_y, plot_df, hue="Tissue_type", plot_reg=False, pal=PALETTE_TTYPE)
ax.set_xlabel(f"Factor {f_x[1:]}")
ax.set_ylabel(f"Factor {f_y[1:]}")
plt.savefig(f"{RPATH}/MultiOmics_{f_x}_{f_y}_tissue_plot.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/MultiOmics_{f_x}_{f_y}_tissue_plot.png", bbox_inches="tight", dpi=600)
plt.close("all")

# Continous annotation
for z in ["VIM_proteomics", "CDH1_proteomics"]:
    ax = GIPlot.gi_continuous_plot(f_x, f_y, z, plot_df, cbar_label=z.replace("_", " "))
    ax.set_xlabel(f"Factor {f_x[1:]}")
    ax.set_ylabel(f"Factor {f_y[1:]}")
    plt.savefig(f"{RPATH}/MultiOmics_{f_x}_{f_y}_continous_{z}.pdf", bbox_inches="tight")
    plt.savefig(f"{RPATH}/MultiOmics_{f_x}_{f_y}_continous_{z}.png", bbox_inches="tight", dpi=600)
    plt.close("all")

# Ridge plot VIM proteomics per tissue
df = plot_df.dropna(subset=["VIM_proteomics", "Tissue_type"])

order = df.groupby("Tissue_type")["VIM_proteomics"].mean().sort_values()

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

g = sns.FacetGrid(
    df,
    row="Tissue_type",
    hue="Tissue_type",
    row_order=list(order.index),

    aspect=15,
    height=.5,
    palette=PALETTE_TTYPE,
)

g.map(
    sns.kdeplot,
    "VIM_proteomics",
    bw_adjust=.5,
    clip_on=False,
    fill=True,
    alpha=1,
    linewidth=.5
)
# g.map(
#     sns.kdeplot,
#     "VIM_proteomics",
#     clip_on=False,
#     color="w",
#     lw=.5,
#     bw_adjust=.5
# )

g.refline(y=0, linewidth=0.25, linestyle="-", color=None, clip_on=False)


def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "VIM_proteomics")

g.figure.subplots_adjust(hspace=-.5)

g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

g.set_axis_labels(f"VIM proteomics", "")

plt.savefig(f"{RPATH}/MultiOmics_{f_x}_{f_y}_continous_VIM_ridge_plot.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/MultiOmics_{f_x}_{f_y}_continous_VIM_ridge_plot.png", bbox_inches="tight", dpi=600)
plt.close("all")


# ### MOFA Factor 11
f = "F12"

# GSEA enrichment plot
enr_obj = Enrichment(dict(tissue=gmts), sig_min_len=10, permutations=int(1e4))

enr_obj.plot(mofa.weights["proteomics"][f], "tissue", "Skin", vertical_lines=True)

plt.gcf().set_size_inches(3, 2)

plt.savefig(f"{RPATH}/MultiOmics_Skin_{f}_GSEA_plot.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/MultiOmics_Skin_{f}_GSEA_plot.png", bbox_inches="tight", dpi=600)
plt.close("all")

#
sel_drugs = ["2508;Trametinib;GDSC2", "1062;Selumetinib;GDSC2", "1373;Dabrafenib;GDSC2"]
sel_crispr = ["BRAF", "SOX10", "MITF", "MAPK1"]
sel_mobems = ["BRAF_mut"]

plot_df = pd.concat([
    mofa.factors[f],
    drespo.loc[sel_drugs].T,
    crispr.loc[sel_crispr].T,
    mobems.loc[sel_mobems].T,
    ss["Tissue_type"],
], axis=1)

plot_df["Skin"] = [i if i == "Skin" else "Other" for i in plot_df["Tissue_type"]]

pal = {
    "Skin": PALETTE_TTYPE["Skin"],
    "Other": "#e1e1e1",
    0: "#E1E1E1",
}

pal_order = ["Other", "Skin"]

for y_var in ["1373;Dabrafenib;GDSC2", "BRAF"]:
    df = plot_df.dropna(subset=[f, y_var, "Skin"])

    g = GIPlot.gi_regression_marginal(
        x=f,
        y=y_var,
        z="Skin",
        plot_df=df,
        discrete_pal=pal,
        hue_order=pal_order,
        legend_title="Skin - BRAF mut",
        scatter_kws=dict(edgecolor="w", lw=0.1, s=16)
    )

    sns.scatterplot(
        x=f,
        y=y_var,
        hue="Skin",
        hue_order=pal_order,
        style="BRAF_mut",
        style_order=[0, 1],
        data=df.sort_values("Skin"),
        palette=pal,
        legend=False,
        ax=g.ax_joint,
    )

    g.ax_joint.set_xlabel(f"Factor {f[1:]}")
    g.ax_joint.set_ylabel(y_var)

    plt.gcf().set_size_inches(2, 2)

    plt.savefig(f"{RPATH}/MultiOmics_Skin_{f}_regression_{y_var}.pdf", bbox_inches="tight")
    plt.savefig(f"{RPATH}/MultiOmics_Skin_{f}_regression_{y_var}.png", bbox_inches="tight", dpi=600)
    plt.close("all")

##
f = "F6"
f_drugs = pd.concat([
    mofa.weights["drespo"][f].sort_values(ascending=False),
    dtargets,
], axis=1).dropna(subset=[f])
f_drugs.to_csv(f"{TPATH}/Factor_{f}_drug_loadings.csv")

mofa.weights["proteomics"][f].sort_values().to_csv(f"{TPATH}/Factor_{f}_protein_loadings.csv")
