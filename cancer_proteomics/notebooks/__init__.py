#!/usr/bin/env python
# Copyright (C) 2020 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import FactorAnalysis, PCA


PALETTE_CTYPE = {
    "B-Cell Non-Hodgkin's Lymphoma": "#007fff",
    "Plasma Cell Myeloma": "#665d1e",
    "Acute Myeloid Leukemia": "#ffbf00",
    "Other Solid Carcinomas": "#fbceb1",
    "Gastric Carcinoma": "#ff033e",
    "T-Lymphoblastic Leukemia": "#ab274f",
    "Breast Carcinoma": "#f2f3f4",
    "Ewing's Sarcoma": "#7cb9e8",
    "Ovarian Carcinoma": "#efdecd",
    "Melanoma": "#8db600",
    "Head and Neck Carcinoma": "#e9d66b",
    "Endometrial Carcinoma": "#b284be",
    "Glioblastoma": "#b2beb5",
    "Small Cell Lung Carcinoma": "#f0f8ff",
    "Pancreatic Carcinoma": "#6e7f80",
    "Burkitt's Lymphoma": "#ff7e00",
    "Mesothelioma": "#87a96b",
    "B-Lymphoblastic Leukemia": "#c9ffe5",
    "Prostate Carcinoma": "#9f2b68",
    "Neuroblastoma": "#00ffff",
    "Non-Small Cell Lung Carcinoma": "#008000",
    "Chronic Myelogenous Leukemia": "#cd9575",
    "Colorectal Carcinoma": "#72a0c1",
    "Cervical Carcinoma": "#a32638",
    "Kidney Carcinoma": "#9966cc",
    "Oral Cavity Carcinoma": "#f19cbb",
    "Hepatocellular Carcinoma": "#e32636",
    "Non-Cancerous": "#3b7a57",
    "Biliary Tract Carcinoma": "#faebd7",
    "Glioma": "#fdee00",
    "Squamous Cell Lung Carcinoma": "#00308f",
    "Bladder Carcinoma": "#7fffd4",
    "Esophageal Squamous Cell Carcinoma": "#c46210",
    "Hodgkin's Lymphoma": "#a8bb19",
    "Rhabdomyosarcoma": "#ff9966",
    "Thyroid Gland Carcinoma": "#a52a2a",
    "T-Cell Non-Hodgkin's Lymphoma": "#568203",
    "Esophageal Carcinoma": "#4b5320",
    "Other Blood Carcinomas": "#5d8aa8",
    "Osteosarcoma": "#8f9779",
    "Chondrosarcoma": "#915c83",
}

PALETTE_TTYPE = {
    "Haematopoietic and Lymphoid": "#007fff",
    "Lung": "#665d1e",
    "Stomach": "#ffbf00",
    "Breast": "#fbceb1",
    "Bone": "#ff033e",
    "Ovary": "#ab274f",
    "Skin": "#f2f3f4",
    "Head and Neck": "#7cb9e8",
    "Endometrium": "#efdecd",
    "Adrenal Gland": "#8db600",
    "Central Nervous System": "#e9d66b",
    "Pancreas": "#b284be",
    "Prostate": "#b2beb5",
    "Peripheral Nervous System": "#f0f8ff",
    "Large Intestine": "#6e7f80",
    "Cervix": "#ff7e00",
    "Kidney": "#87a96b",
    "Liver": "#c9ffe5",
    "Soft Tissue": "#9f2b68",
    "Biliary Tract": "#00ffff",
    "Bladder": "#008000",
    "Esophagus": "#cd9575",
    "Testis": "#72a0c1",
    "Thyroid": "#a32638",
    "Vulva": "#9966cc",
    "Uterus": "#f19cbb",
    "Small Intestine": "#e32636",
    "Placenta": "#3b7a57",
}

PALETTE_INSTRUMENT = {
    "M01": "#66c2a5",
    "M02": "#fc8d62",
    "M03": "#8da0cb",
    "M04": "#e78ac3",
    "M05": "#a6d854",
    "M06": "#ffd92f",
}

PALETTE_BATCH = {
    "P01": "#7fc97f",
    "P02": "#beaed4",
    "P03": "#fdc086",
    "P04": "#386cb0",
    "P05": "#f0027f",
    "P06": "#bf5b17",
}

PALETTE_PERTURB = {
    "BT-549 10% FBS": "#1f77b4",
    "BT-549 1% FBS": "#aec7e8",
    "T-47D 10% FBS": "#ff7f0e",
    "T-47D 1% FBS": "#ffbb78",
    "HCC1395 10% FBS": "#2ca02c",
    "HCC1395 1% FBS": "#98df8a",
    "HCC1143 10% FBS": "#d62728",
    "HCC1143 1% FBS": "#ff9896",
}


def two_vars_correlation(var1, var2, idx_set=None, method="pearson", verbose=0):
    if verbose > 0:
        print(f"Var1={var1.name}; Var2={var2.name}")

    if idx_set is None:
        idx_set = set(var1.dropna().index).intersection(var2.dropna().index)

    else:
        idx_set = set(var1.reindex(idx_set).dropna().index).intersection(
            var2.reindex(idx_set).dropna().index
        )

    if method == "spearman":
        r, p = spearmanr(
            var1.reindex(index=idx_set), var2.reindex(index=idx_set), nan_policy="omit"
        )
    else:
        r, p = pearsonr(var1.reindex(index=idx_set), var2.reindex(index=idx_set))

    return dict(corr=r, pval=p, len=len(idx_set))


class DataImport:
    DPATH = pkg_resources.resource_filename("data", "/")

    @classmethod
    def read_samplesheet(cls):
        """
        Read cancer cell lines samplesheet

        :return:
        """
        return pd.read_csv(f"{cls.DPATH}/E0022_P06_samplehseet.csv", index_col=0)

    @classmethod
    def read_manifest(cls):
        """
        Read SWATH-MS proteomics samples manifest

        :return:
        """
        # Import samplesheet
        ss = cls.read_samplesheet()

        # Import manifest
        manifest = pd.read_csv(
            f"{cls.DPATH}/E0022_P06_final_sample_map.txt", index_col=0, sep="\t"
        )

        # Remove excluded samples
        exclude_man = manifest[~manifest["SIDM"].isin(ss.index)]
        manifest = manifest[~manifest.index.isin(exclude_man.index)]

        return manifest

    @classmethod
    def read_protein_matrix(cls, map_protein=False, subset=None):
        """
        Read protein level matrix.

        :return:
        """
        # Read manifest
        manifest = cls.read_manifest()

        # Read protein level normalised intensities
        protein = pd.read_csv(
            f"{cls.DPATH}/E0022_P06_Protein_Matrix_ProNorM.tsv.gz",
            sep="\t",
            index_col=0,
        ).T

        # Discard control samples
        protein = protein.rename(columns=manifest.groupby("Cell_line")["SIDM"].first())

        exclude_controls = [
            "Control_HEK293T_lys",
            "Control_HEK293T_std_H002",
            "Control_HEK293T_std_H003",
        ]
        protein = protein.drop(columns=exclude_controls)

        # Map protein to gene symbols
        if map_protein:
            pmap = cls.map_gene_name().reindex(protein.index)["GeneSymbol"].dropna()

            protein = protein[protein.index.isin(pmap.index)]
            protein = protein.groupby(pmap.reindex(protein.index)).mean()

        if subset is not None:
            protein = protein.loc[:, protein.columns.isin(subset)]

        return protein

    @classmethod
    def read_protein_perturbation_manifest(cls):
        manifest = pd.read_csv(
            f"{cls.DPATH}/E0019_BreastCan_ProteinMatrix_LoessNorm_byDiffacto.csv",
            index_col=0,
        ).T.iloc[:12].T

        manifest = manifest.replace({"Cell Line": {
            "T-47D 10%FBS": "T-47D 10% FBS",
            "T-47D 1%FBS": "T-47D 1% FBS",
        }})

        return manifest

    @classmethod
    def read_protein_perturbation(cls, map_protein=False, subset=None):
        """
        Read protein level matrix.

        :return:
        """
        # Read protein level normalised intensities
        protein = pd.read_csv(
            f"{cls.DPATH}/E0019_BreastCan_ProteinMatrix_LoessNorm_byDiffacto.csv",
            index_col=0,
        ).T.iloc[12:].astype(float)

        # Map protein to gene symbols
        if map_protein:
            pmap = cls.map_gene_name().reindex(protein.index)["GeneSymbol"].dropna()

            protein = protein[protein.index.isin(pmap.index)]
            protein = protein.groupby(pmap.reindex(protein.index)).mean()

        if subset is not None:
            protein = protein.loc[:, protein.columns.isin(subset)]

        return protein

    @classmethod
    def read_protein_matrix_broad(cls, subset=None):
        protein = pd.read_csv(f"{cls.DPATH}/broad_tmt.csv.gz", compression="gzip")
        protein = (
            protein.dropna(subset=["Gene_Symbol"])
            .groupby("Gene_Symbol")
            .agg(np.nanmean)
        )

        if subset is not None:
            protein = protein.loc[:, protein.columns.isin(subset)]

        return protein

    @classmethod
    def read_peptide_raw_mean(cls):
        peptide_raw_mean = pd.read_csv(
            f"{cls.DPATH}/E0022_P06_Protein_Matrix_Raw_Mean_Intensities.tsv.gz",
            sep="\t",
            index_col=0,
        ).iloc[:, 0]
        return peptide_raw_mean

    @classmethod
    def read_gene_matrix(cls, subset=None):
        data = pd.read_csv(f"{cls.DPATH}/rnaseq_voom.csv.gz", index_col=0)

        if subset is not None:
            data = data.loc[:, data.columns.isin(subset)]

        return data

    @classmethod
    def read_copy_number(cls):
        return pd.read_csv(f"{cls.DPATH}/copynumber_total_new_map.csv.gz", index_col=0)

    @classmethod
    def map_gene_name(cls, index_col="Entry name"):
        idmap = pd.read_csv(f"{cls.DPATH}/uniprot_human_idmap.tab.gz", sep="\t")

        if index_col is not None:
            idmap = idmap.dropna(subset=[index_col]).set_index(index_col)

        idmap["GeneSymbol"] = idmap["Gene names  (primary )"].apply(
            lambda v: v.split("; ")[0] if str(v).lower() != "nan" else v
        )

        return idmap

    @classmethod
    def read_crispr_sids(cls):
        sid = cls.read_samplesheet()
        sid = sid.reset_index().dropna(subset=["BROAD_ID"])
        sid = sid.groupby("BROAD_ID")["model_id"].first()
        return sid

    @classmethod
    def read_crispr_matrix(cls, subset=None):
        merged = pd.read_csv(
            f"{cls.DPATH}/CRISPRcleanR_FC.txt.gz", index_col=0, sep="\t"
        )

        sid = cls.read_crispr_sids()
        merged = merged.rename(columns=sid)

        if subset is not None:
            merged = merged.loc[:, merged.columns.isin(subset)]

        return merged

    @classmethod
    def read_crispr_institute(cls):
        merged = cls.read_crispr_matrix()

        merged_institute = pd.Series(
            {c: "Broad" if c.startswith("ACH-") else "Sanger" for c in merged}
        )

        sid = cls.read_crispr_sids()
        merged_institute = merged_institute.rename(index=sid)

        return merged_institute

    @classmethod
    def read_drug_response(
        cls,
        as_matrix=True,
        drug_columns=["drug_id", "drug_name", "dataset"],
        sample_columns=["model_id"],
        dtype="ln_IC50",
        subset=None,
    ):
        drugresponse = pd.read_csv(
            f"{cls.DPATH}/DrugResponse_PANCANCER_GDSC1_GDSC2_20200602.csv.gz"
        )

        drugresponse = drugresponse[~drugresponse["cell_line_name"].isin(["LS-1034"])]

        if as_matrix:
            drugresponse = pd.pivot_table(
                drugresponse,
                index=drug_columns,
                columns=sample_columns,
                values=dtype,
                fill_value=np.nan,
            )

        if subset is not None:
            drugresponse = drugresponse.loc[:, drugresponse.columns.isin(subset)]

        return drugresponse

    @classmethod
    def read_drug_max_concentration(
        cls, drug_columns=["drug_id", "drug_name", "dataset"]
    ):
        drugresponse = cls.read_drug_response(as_matrix=False)

        maxconcentration = drugresponse.groupby(drug_columns)[
            "max_screening_conc"
        ].first()

        return maxconcentration

    @classmethod
    def read_methylation_matrix(cls, subset=None):
        methy_promoter = pd.read_csv(
            f"{cls.DPATH}/methy_beta_gene_promoter.csv.gz", index_col=0
        )

        if subset is not None:
            methy_promoter = methy_promoter.loc[:, methy_promoter.columns.isin(subset)]

        return methy_promoter


class DimReduction:
    LOG = logging.getLogger("cancer_proteomics")

    @staticmethod
    def pc_labels(n):
        return [f"PC{i}" for i in np.arange(1, n + 1)]

    @classmethod
    def dim_reduction_pca(cls, df, pca_ncomps=10, is_factor_analysis=False):
        if is_factor_analysis:
            df_pca = FactorAnalysis(n_components=pca_ncomps).fit(df.T)

        else:
            df_pca = PCA(n_components=pca_ncomps).fit(df.T)

        df_pcs = pd.DataFrame(
            df_pca.transform(df.T), index=df.T.index, columns=cls.pc_labels(pca_ncomps)
        )

        df_loadings = pd.DataFrame(
            df_pca.components_, index=cls.pc_labels(pca_ncomps), columns=df.T.columns
        )

        if is_factor_analysis:
            df_vexp = None

        else:
            df_vexp = pd.Series(df_pca.explained_variance_ratio_, index=df_pcs.columns)

        return dict(pcs=df_pcs, vexp=df_vexp, loadings=df_loadings)

    @classmethod
    def dim_reduction(
        cls,
        df,
        pca_ncomps=50,
        tsne_ncomps=2,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate=200.0,
        n_iter=1000,
    ):
        miss_values = df.isnull().sum().sum()
        if miss_values > 0:
            cls.LOG.warning(
                f"DataFrame has {miss_values} missing values; impute with row mean"
            )
            df = df.T.fillna(df.T.mean()).T

        # PCA
        dimred_dict = cls.dim_reduction_pca(df, pca_ncomps)

        # tSNE
        df_tsne = TSNE(
            n_components=tsne_ncomps,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter=n_iter,
        ).fit_transform(dimred_dict["pcs"])

        dimred_dict["tsne"] = pd.DataFrame(
            df_tsne, index=dimred_dict["pcs"].index, columns=cls.pc_labels(tsne_ncomps)
        )

        return dimred_dict

    @staticmethod
    def plot_dim_reduction(
        data, x="PC1", y="PC2", hue_by=None, palette=None, ctype="tsne", ax=None
    ):
        if palette is None:
            palette = dict(All=CrispyPlot.PAL_DBGD[0])

        plot_df = pd.concat(
            [
                data["pcs" if ctype == "pca" else "tsne"][x],
                data["pcs" if ctype == "pca" else "tsne"][y],
            ],
            axis=1,
        )

        if hue_by is not None:
            plot_df = pd.concat([plot_df, hue_by.rename("hue_by")], axis=1).dropna()

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0), dpi=600)

        hue_by_df = (
            plot_df.groupby("hue_by") if hue_by is not None else [(None, plot_df)]
        )
        for t, df in hue_by_df:
            ax.scatter(
                df[x].values,
                df[y].values,
                c=CrispyPlot.PAL_DTRACE[2] if hue_by is None else palette[t],
                marker="o",
                linewidths=0,
                s=5,
                label=t,
                alpha=0.8,
            )

        ax.set_xlabel("" if ctype == "tsne" else f"{x} ({data['vexp'][x]*100:.1f}%)")
        ax.set_ylabel("" if ctype == "tsne" else f"{y} ({data['vexp'][y]*100:.1f}%)")
        ax.axis("off" if ctype == "tsne" else "on")

        if ctype == "pca":
            ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        if hue_by is not None:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                prop={"size": 4},
                frameon=False,
                title="Tissue type",
            ).get_title().set_fontsize("5")

        return ax
