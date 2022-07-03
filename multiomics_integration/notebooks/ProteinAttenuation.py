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
from natsort import natsorted
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from adjustText import adjust_text
import matplotlib.patches as mpatches
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests
from multiomics_integration.notebooks import DataImport, two_vars_correlation
from crispy.DataImporter import CORUM


LOG = logging.getLogger("multiomics_integration")
DPATH = pkg_resources.resource_filename("data", "/")
TPATH = pkg_resources.resource_filename("tables", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
RPATH = pkg_resources.resource_filename("multiomics_integration", "plots/DIANN/")


# ### Imports

# Read samplesheet
ss = DataImport.read_samplesheet()

# Read proteomics (Proteins x Cell lines)
prot = DataImport.read_protein_matrix(map_protein=True)

# Read Transcriptomics
gexp = DataImport.read_gene_matrix()

# Read copy number
cnv = DataImport.read_copy_number()


# ### Overlaps
samples = list(set.intersection(set(prot), set(gexp), set(cnv)))
genes = list(set.intersection(set(prot.index), set(gexp.index), set(cnv.index)))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")


# ### Protein/Gene correlation
# Spearman's rho
pg_corr = pd.DataFrame(
    [two_vars_correlation(prot.loc[g], gexp.loc[g], method="spearman") for g in genes]
).assign(protein=genes)
pg_corr = pg_corr.sort_values("pval").dropna()
pg_corr["fdr"] = multipletests(pg_corr["pval"], method="fdr_bh")[1]
pg_corr.to_csv(f"{TPATH}/ProteinAttenuation_correlations_DIANN.csv.gz", index=False, compression="gzip")

# Histogram
_, ax = plt.subplots(1, 1, figsize=(2.0, 1.5), dpi=600)
sns.distplot(
    pg_corr["corr"].values,
    hist_kws=dict(alpha=0.4, zorder=1, linewidth=0),
    bins=30,
    kde_kws=dict(cut=0, lw=1, zorder=1, alpha=0.8),
    color=CrispyPlot.PAL_DTRACE[2],
    ax=ax,
)
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
ax.set_xlabel("Pearson's R")
ax.set_ylabel("Density")
ax.set_title(f"Protein ~ Transcript (mean R={pg_corr['corr'].mean():.2f})")
plt.savefig(f"{RPATH}/ProteinAttenuation_histogram.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/ProteinAttenuation_histogram.png", bbox_inches="tight")
plt.close("all")


# ### Protein/Gene correlation per tissue
# spearman's rho
pg_corr_tissue = pd.DataFrame(
    [
        {
            **two_vars_correlation(
                prot.loc[g], gexp.loc[g], idx_set=t_samples, method="spearman"
            ),
            **dict(tissue=t),
        }
        for t, t_samples in ss.reindex(samples)
        .reset_index()
        .groupby("Tissue_type")["model_id"]
        if len(t_samples) > 15
        for g in genes
    ]
)
pg_corr_tissue = pg_corr_tissue.query("len > 15")
pg_corr_tissue["fdr"] = multipletests(pg_corr_tissue["pval"], method="fdr_bh")[1]
pg_corr_tissue.to_csv(
    f"{TPATH}/ProteinAttenuation_correlations_tissue_DIANN.csv.gz",
    index=False,
    compression="gzip",
)

# Boxplot
order = pg_corr_tissue.groupby("tissue")["corr"].median().sort_values()

_, ax = plt.subplots(1, 1, figsize=(1.0, 0.125 * len(order)), dpi=600)
sns.boxplot(
    x="corr",
    y="tissue",
    data=pg_corr_tissue,
    notch=True,
    order=order.index,
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    medianprops=CrispyPlot.MEDIANPROPS,
    flierprops=CrispyPlot.FLIERPROPS,
    color=CrispyPlot.PAL_DTRACE[2],
    showcaps=False,
    saturation=1,
    orient="h",
    ax=ax,
)
ax.set_xlabel("Transcript ~ Protein")
ax.set_ylabel("")
ax.grid(axis="x", lw=0.1, color="#e1e1e1", zorder=0)
plt.savefig(f"{RPATH}/ProteinAttenuation_tissue_boxplot.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/ProteinAttenuation_tissue_boxplot.png", bbox_inches="tight")
plt.close("all")


# ### Protein/Gene correlation with CopyNumber - Attenuation
#
patt_corr = pd.DataFrame(
    {
        g: pd.concat(
            [
                pd.Series(two_vars_correlation(cnv.loc[g], prot.loc[g])).add_prefix("prot_"),
                pd.Series(two_vars_correlation(cnv.loc[g], gexp.loc[g])).add_prefix("gexp_"),
            ]
        )
        for g in genes
    }
).T.sort_values("gexp_pval").dropna()
patt_corr = patt_corr.query("(prot_len > 15) & (gexp_len > 15)")
patt_corr["attenuation"] = patt_corr["gexp_corr"] - patt_corr["prot_corr"]

gmm = GaussianMixture(n_components=2).fit(patt_corr[["attenuation"]])
s_type, clusters = (
    pd.Series(gmm.predict(patt_corr[["attenuation"]]), index=patt_corr.index),
    pd.Series(gmm.means_[:, 0], index=range(2)),
)
patt_corr["cluster"] = [
    "High" if s_type[p] == clusters.argmax() else "Low" for p in patt_corr.index
]

# Annotate with CORUM information
corum_db = CORUM(ddir=PPIPATH)

corum_db = corum_db.db.copy()
corum_db["subunits"] = corum_db["subunits(Gene name)"].apply(lambda v: set(v.split(";"))).values
corum_db["complex_id"] = [f"{v1}: {v2}" for v1, v2 in corum_db[["ComplexID", "ComplexName"]].values]

corum = corum_db.set_index("complex_id")["subunits"].to_dict()

patt_corr["CORUM"] = [";".join([c for c in corum if p in corum[c]]) for p in patt_corr.index]

# Sort and export
patt_corr = patt_corr.sort_values("attenuation", ascending=False)
patt_corr.to_csv(f"{TPATH}/ProteinAttenuation_attenuation_DIANN.csv.gz", compression="gzip")
# patt_corr = pd.read_csv(f"{TPATH}/ProteinAttenuation_attenuation.csv.gz", index_col=0)

# ### Pathway enrichement analysis of attenuated proteins
background = set(patt_corr.index)
sublist = set(patt_corr.query("cluster == 'High'").index)

enr_obj = Enrichment(gmts=["c5.all.v7.1.symbols.gmt"], sig_min_len=15, padj_method="fdr_bh")

enr = enr_obj.hypergeom_enrichments(sublist, background, "c5.all.v7.1.symbols.gmt")
enr = enr[enr["adj.p_value"] < 0.01].head(30).reset_index()
enr["name"] = [i[3:].lower().replace("_", " ") for i in enr["gset"]]
enr = enr.replace({"adj.p_value": 0}, value=enr[enr["adj.p_value"] != 0]["adj.p_value"].min())

_, ax = plt.subplots(1, 1, figsize=(2.0, 5.0), dpi=600)

sns.barplot(
    -np.log10(enr["adj.p_value"]),
    enr["name"],
    orient="h",
    color=CrispyPlot.PAL_DTRACE[2],
    ax=ax,
)

for i, (_, row) in enumerate(enr.iterrows()):
    if row["p_value"] == 0:
        plt.text(
            -np.log10(row["adj.p_value"]),
            i,
            f">",
            va="center",
            ha="left",
            fontsize=5,
            zorder=10,
            color=CrispyPlot.PAL_DTRACE[2],
        )

    plt.text(
        -np.log10(row["adj.p_value"]) * 0.99,
        i,
        f"{row['len_intersection']}/{row['len_sig']}",
        va="center",
        ha="right",
        fontsize=5,
        zorder=10,
        color="white",
    )

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
ax.set_xlabel("Hypergeometric test (-log10 FDR)")
ax.set_ylabel("")
ax.set_title(f"GO celluar component enrichment - attenuated proteins")

plt.savefig(f"{RPATH}/ProteinAttenuation_enrichment.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/ProteinAttenuation_enrichment.png", bbox_inches="tight")
plt.close("all")

"""
Attenuation plot
"""
# Attenuation scatter gene highlights
complex_attenuation = pd.DataFrame([dict(
    complex=c, attenuation=a, prot_corr=p, gexp_corr=e, prot_len=n, protein=i
) for i, l, a, p, e, n in patt_corr.reset_index()[["index", "CORUM", "attenuation", "prot_corr", "gexp_corr", "prot_len"]].values if l != "" for c in l.split(";")])

complex_attenuation = pd.concat([
    complex_attenuation.groupby("complex")["prot_corr"].median(),
    complex_attenuation.groupby("complex")["gexp_corr"].median(),
    complex_attenuation.groupby("complex")["prot_corr"].count().rename("count"),
    complex_attenuation.groupby("complex")["protein"].agg(set).rename("proteins"),
], axis=1)

# Data
signatures_order = [
    "229: NAT complex",
    "191: 20S proteasome",
    "126: CCT complex (chaperonin containing TCP1 complex)",
]
signatures = {
    s: set(corum_db.query(f"complex_id == '{s}'")["subunits(Gene name)"].iloc[0].split(";"))
    for s in signatures_order
}

plot_df = patt_corr.copy().assign(
    signature=[[s for s in signatures if g in signatures[s]] for g in patt_corr.index]
)
plot_df = plot_df.assign(
    signature=[i[0] if len(i) > 0 else "Other" for i in plot_df["signature"]]
)

discrete_pal = pd.Series(
    sns.color_palette("tab10").as_hex()[: len(signatures)], index=signatures
)
discrete_pal["Other"] = CrispyPlot.PAL_DTRACE[0]

# Plot
g = GIPlot.gi_regression_marginal(
    "gexp_corr", "prot_corr", "signature",
    plot_df,
    discrete_pal=discrete_pal.to_dict(),
    plot_reg=False,
    plot_annot=False,
    plot_legend=False,
    hue_order=["Other"] + signatures_order,
    scatter_kws=dict(edgecolor="w", lw=0.1, s=16),
    base=0.25,
)

# Axis labels
g.ax_joint.set_xlabel("Copy number and Transcriptomics\nPearson's r")
g.ax_joint.set_ylabel("Copy number and Proteomics\nPearson's r")

# Set axis limits
ax_min = plot_df[["gexp_corr", "prot_corr"]].min().min() * 1.1
ax_max = plot_df[["gexp_corr", "prot_corr"]].max().max() * 1.1

g.ax_joint.set_xlim(ax_min, ax_max)
g.ax_joint.set_ylim(ax_min, ax_max)

g.ax_marg_x.set_xlim(ax_min, ax_max)
g.ax_marg_y.set_ylim(ax_min, ax_max)

# Plot diagonal
g.ax_joint.plot([ax_min, ax_max], [ax_min, ax_max], "k--", lw=0.3)

# Plot labels
labels = [
    g.ax_joint.text(row["gexp_corr"], row["prot_corr"], i, color="k", fontsize=6)
    for i, row in plot_df.query("signature != 'Other'")
    .sort_values("attenuation", ascending=False)
    .head(15)
    .iterrows()
]
adjust_text(
    labels,
    arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
    ax=g.ax_joint,
)

# Plot legend
handles = [
    mpatches.Circle(
        (0.0, 0.0),
        0.25,
        facecolor=discrete_pal[t],
        label=f"{t} (N={(plot_df['signature'] == t).sum()})",
    )
    for t in natsorted(set(plot_df["signature"]))
]

g.ax_joint.legend(
    handles=handles,
    loc="upper left",
    frameon=False,
)

# Set size
plt.gcf().set_size_inches(2.5, 2.5)

plt.savefig(f"{RPATH}/ProteinTranscript_attenuation_scatter_signatures.pdf", bbox_inches="tight")
plt.savefig(f"{RPATH}/ProteinTranscript_attenuation_scatter_signatures.png", bbox_inches="tight")
plt.close("all")
