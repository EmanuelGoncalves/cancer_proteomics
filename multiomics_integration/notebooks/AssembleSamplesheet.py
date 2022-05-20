#!/usr/bin/env python
# Copyright (C) 2021 Emanuel Goncalves

import h5py
import logging
import numpy as np
import pandas as pd
import pkg_resources
from crispy.MOFA import MOFA
from crispy.Utils import Utils
from crispy.Enrichment import Enrichment
from multiomics_integration.notebooks import DataImport, two_vars_correlation

LOG = logging.getLogger("cancer_proteomics")
DPATH = pkg_resources.resource_filename("data", "/")
PPIPATH = pkg_resources.resource_filename("data", "ppi/")
TPATH = pkg_resources.resource_filename("tables", "/")
RPATH = pkg_resources.resource_filename("cancer_proteomics", "plots/")


# Cell line information
sinfo = pd.read_csv(
    f"{DPATH}/e0022_diann_051021_sample_mapping_averaged.txt", sep="\t"
)

# Add Haem lineage
haem_lineage = pd.read_excel(f"{DPATH}/Hematopoietic_080321.xlsx", index_col=1)
sinfo["Haem_lineage"] = haem_lineage.reindex(sinfo["SIDM"])["Lineage"].values

# Cell Model Passports mapping
cmp = DataImport.read_cmp_samplesheet()

sinfo["BROAD_ID"] = cmp.loc[sinfo["SIDM"].values, "BROAD_ID"].values
sinfo["CCLE_ID"] = cmp.loc[sinfo["SIDM"].values, "CCLE_ID"].values
sinfo["ploidy"] = cmp.loc[sinfo["SIDM"].values, "ploidy"].values
sinfo["mutational_burden"] = cmp.loc[sinfo["SIDM"].values, "mutational_burden"].values
sinfo["msi_status"] = cmp.loc[sinfo["SIDM"].values, "msi_status"].values
sinfo["growth_properties"] = cmp.loc[sinfo["SIDM"].values, "growth_properties"].values

# Deprecated samplesheet
ss_deprecated = pd.read_excel(
    f"{DPATH}/SupplementaryTable1_Deprecated.xlsx", index_col=0
)
sinfo["growth"] = ss_deprecated.loc[sinfo["SIDM"].values, "growth"].values
sinfo["size"] = ss_deprecated.loc[sinfo["SIDM"].values, "size"].values
sinfo["media"] = ss_deprecated.loc[sinfo["SIDM"].values, "media"].values

# Replicates correlation
prot_reps_map = pd.read_csv(f"{DPATH}/e0022_diann_051021_sample_mapping_replicates.txt", sep="\t")
prot_reps_map = prot_reps_map.loc[[not i.startswith("Control_HEK293T") for i in prot_reps_map["Project_Identifier"]]]
prot_reps_map = prot_reps_map[["Automatic_MS_filename", "Project_Identifier"]]

prot_reps = pd.read_csv(
    f"{DPATH}/e0022_diann_051021_frozen_matrix.txt",
    sep="\t",
    index_col=0,
).T


def mean_rep_correlation(reps):
    reps_corr = prot_reps[reps].corr()

    keep = np.triu(np.ones(reps_corr.shape), 1).astype("bool").reshape(reps_corr.size)
    reps_corr_mean = reps_corr.unstack()[keep].rename("pearson").dropna().mean()

    return reps_corr_mean


cell_line_corrs = pd.Series(
    {
        k: mean_rep_correlation(v)
        for k, v in prot_reps_map.groupby("Project_Identifier")["Automatic_MS_filename"]
        .agg(set)
        .to_dict()
        .items()
    }
)

sinfo["replicates_correlation"] = cell_line_corrs.loc[sinfo["Project_Identifier"].values].values

# Numebr of proteins per cell line
prot = pd.read_csv(f"{DPATH}/e0022_diann_051021_frozen_matrix_averaged.txt", sep="\t", index_col=0).T
prot.columns = [c.split(";")[0] for c in prot]

sinfo["number_of_proteins"] = prot.count().loc[sinfo["SIDM"].values].values

# GO term enrichments
prot = DataImport.read_protein_matrix(map_protein=True, min_measurements=3)

emt_sig = Enrichment.read_gmt(f"{DPATH}/pathways/emt.symbols.gmt")
emt_sig = emt_sig["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"]

proteasome_sig = Enrichment.read_gmt(f"{DPATH}/pathways/proteasome.symbols.gmt")
proteasome_sig = proteasome_sig["BIOCARTA_PROTEASOME_PATHWAY"]

translation_sig = Enrichment.read_gmt(f"{DPATH}/pathways/translation_initiation.symbols.gmt")
translation_sig = translation_sig["GO_TRANSLATIONAL_INITIATION"]

gene_sets = dict(
    EMT=emt_sig,
    Proteasome=proteasome_sig,
    TranslationInitiation=translation_sig,
)

enr_obj = Enrichment(
    gmts=dict(pathways=gene_sets), sig_min_len=5, padj_method="fdr_bh"
)

enr = pd.DataFrame(
    {
        c: enr_obj.gsea_enrichments(prot[c].dropna())["e_score"].to_dict()
        for c in prot
    }
)

sinfo["EMT"] = enr.T.loc[sinfo["SIDM"].values, "EMT"].values
sinfo["Proteasome"] = enr.T.loc[sinfo["SIDM"].values, "Proteasome"].values
sinfo["TranslationInitiation"] = enr.T.loc[sinfo["SIDM"].values, "TranslationInitiation"].values

# Copy Number instability
sinfo["CopyNumberInstability"] = ss_deprecated.loc[sinfo["SIDM"].values, "CopyNumberInstability"].values

# Gene expression & Copy number correlation
gexp = DataImport.read_gene_matrix()
cnv = DataImport.read_copy_number()

samples = list(set.intersection(set(prot), set(gexp), set(cnv)))
genes = list(
    set.intersection(
        set(prot.index), set(gexp.index), set(cnv.index), set(prot.index)
    )
)
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")

# gexp_t = pd.DataFrame(
#     {i: Utils.gkn(gexp.loc[i].dropna()).to_dict() for i in genes}
# ).T

satt_corr = pd.DataFrame(
    {
        s: dict(
            CNV_Prot=two_vars_correlation(cnv[s], prot[s])["corr"],
            CNV_GExp=two_vars_correlation(cnv[s], gexp[s])["corr"],
            GExp_Prot=two_vars_correlation(gexp[s], prot[s])["corr"],
        )
        for s in samples
    }
).T

satt_corr = satt_corr.dropna(subset=["CNV_GExp", "CNV_Prot"])
satt_corr["attenuation"] = satt_corr.eval("CNV_GExp - CNV_Prot")

sinfo["GeneExpressionCorrelation"] = satt_corr.reindex(sinfo["SIDM"].values)["GExp_Prot"].values
sinfo["CopyNumberAttenuation"] = satt_corr.reindex(sinfo["SIDM"].values)["attenuation"].values

# MOFA factors
mofa_file = h5py.File(f"{TPATH}/MultiOmics_DIANN.hdf5", "r")
factors = MOFA.get_factors(mofa_file)

sinfo = pd.concat([sinfo.set_index("SIDM"), factors], axis=1)
sinfo = sinfo.reset_index().rename(columns=dict(index="model_id"))


# Export
ms_info = pd.read_csv(
    f"{DPATH}/e0022_diann_mapping_file_100921_replicate.txt.gz", sep="\t"
).drop(columns=["Code", "Daisy_chain", "Replicate"], errors="ignore")

legend = pd.DataFrame([
  dict(Field="model_id", Description="Cell line identifier (CellModelPassport)"),
  dict(Field="Project_Identifier", Description="Cell line project identifier"),
  dict(Field="Cell_line", Description="Cell line name"),
  dict(Field="Tissue_type", Description="Cell line tissue of origin"),
  dict(Field="Haem_lineage", Description="Lineage of haematopoietic and Lymphoid cell lines"),
  dict(Field="Cancer_type", Description="Cell line cancer type"),
  dict(Field="Cancer_subtype", Description="Cell line cancer subtype"),
  dict(Field="BROAD_ID", Description="Cell line BROAD ID"),
  dict(Field="CCLE_ID", Description="Cell line CCLE ID"),
  dict(Field="ploidy", Description="Cell line ploidy, estimated from SNP6 copy number data"),
  dict(Field="mutational_burden", Description="Cell line mutational burden, estimated from WES data"),
  dict(Field="growth_properties", Description="Cell line culture conditions"),
  dict(Field="growth", Description="Cell line growth rate"),
  dict(Field="size", Description="Cell line size"),
  dict(Field="media", Description="Cell line culture media"),
  dict(Field="replicates_correlation", Description="Cell line proteomics replicate correlation, mean Pearson's R of cell lines' replicates"),
  dict(Field="number_of_proteins", Description="Number of proteins quantified per cell line"),
  dict(Field="EMT", Description="Proteomics GSEA enrichment scores of MSigDB - HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION gene signature"),
  dict(Field="Proteasome", Description="Proteomics GSEA enrichment scores of MSigDB - BIOCARTA_PROTEASOME_PATHWAY gene signature"),
  dict(Field="TranslationInitiation", Description="Proteomics GSEA enrichment scores of MSigDB - GO_TRANSLATIONAL_INITIATION gene signature"),
  dict(Field="CopyNumberInstability", Description="Copy number instability estimated using SNP6 copy number data"),
  dict(Field="GeneExpressionCorrelation", Description="Correlation between gene expression (transcriptomics) and proteomics (Pearson's R)"),
  dict(Field="CopyNumberAttenuation", Description="Differential between the correlation coefficients of Copy Number ~ Gene Expression and Copy Number ~ Proteomics"),
  dict(Field="FXX", Description="MOFA factors (XX representing factors from 1 to 15)"),
])

with pd.ExcelWriter(f"{DPATH}/SupplementaryTable1.xlsx") as writer:
    legend.to_excel(writer, sheet_name="Legend", index=False)
    sinfo.to_excel(writer, sheet_name="Table S1a", index=False)
    ms_info.to_excel(writer, sheet_name="Table S1b", index=False)
