#!/usr/bin/env python
# Copyright (C) 2020 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt
from crispy.Utils import Utils
from sklearn.manifold import TSNE
from crispy.DataImporter import PPI
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import spearmanr, pearsonr
from crispy.LMModels import LMModels, LModel
from sklearn.decomposition import FactorAnalysis, PCA


PALETTE_CTYPE = {
    "Non-Small Cell Lung Carcinoma": "#007fff",
    "Prostate Carcinoma": "#665d1e",
    "Gastric Carcinoma": "#ffbf00",
    "Glioblastoma": "#fbceb1",
    "Melanoma": "#ff033e",
    "Bladder Carcinoma": "#ab274f",
    "B-Lymphoblastic Leukemia": "#d5e6f7",
    "Kidney Carcinoma": "#7cb9e8",
    "Thyroid Gland Carcinoma": "#efdecd",
    "Rhabdomyosarcoma": "#8db600",
    "Head and Neck Carcinoma": "#e9d66b",
    "Ovarian Carcinoma": "#b284be",
    "B-Cell Non-Hodgkin's Lymphoma": "#b2beb5",
    "Other Solid Carcinomas": "#10b36f",
    "Ewing's Sarcoma": "#6e7f80",
    "T-Lymphoblastic Leukemia": "#ff7e00",
    "Plasma Cell Myeloma": "#87a96b",
    "Endometrial Carcinoma": "#c9ffe5",
    "Non-Cancerous": "#9f2b68",
    "Breast Carcinoma": "#00ffff",
    "Pancreatic Carcinoma": "#008000",
    "Neuroblastoma": "#cd9575",
    "Burkitt's Lymphoma": "#72a0c1",
    "Hairy Cell Leukemia": "#a32638",
    "Chronic Myelogenous Leukemia": "#9966cc",
    "Glioma": "#f19cbb",
    "Cervical Carcinoma": "#e32636",
    "Colorectal Carcinoma": "#3b7a57",
    "Hepatocellular Carcinoma": "#faebd7",
    "Vulvar carcinoma": "#fdee00",
    "Osteosarcoma": "#00308f",
    "Chondrosarcoma": "#7fffd4",
    "Small Cell Lung Carcinoma": "#c46210",
    "Esophageal Carcinoma": "#a8bb19",
    "Uncertain": "#ff9966",
    "T-Cell Non-Hodgkin's Lymphoma": "#a52a2a",
    "Non-small Cell Lung Carcinoma": "#568203",
    "Other Sarcomas": "#4b5320",
    "Biliary Tract Carcinoma": "#5d8aa8",
    "Acute Myeloid Leukemia": "#8f9779",
    "Hodgkin's Lymphoma": "#915c83",
    "Mesothelioma": "#841b2d",
    "B-Lymphoblastic leukemia": "#a4c639",
    "Other Blood Cancers": "#3b444b",
    "Carcinoid Tumour": "#006600",
    "Leiomyosarcoma": "#0000ff",
    "T-cell Non-Hodgkin's Lymphoma": "#666699",
}

PALETTE_TTYPE = {
    "Lung": "#007fff",
    "Prostate": "#665d1e",
    "Stomach": "#ffbf00",
    "Central Nervous System": "#fbceb1",
    "Skin": "#ff033e",
    "Bladder": "#ab274f",
    "Haematopoietic and Lymphoid": "#d5e6f7",
    "Kidney": "#7cb9e8",
    "Thyroid": "#efdecd",
    "Soft Tissue": "#8db600",
    "Head and Neck": "#e9d66b",
    "Ovary": "#b284be",
    "Bone": "#b2beb5",
    "Endometrium": "#10b36f",
    "Breast": "#6e7f80",
    "Pancreas": "#ff7e00",
    "Peripheral Nervous System": "#87a96b",
    "Cervix": "#c9ffe5",
    "Large Intestine": "#9f2b68",
    "Liver": "#00ffff",
    "Vulva": "#008000",
    "Esophagus": "#cd9575",
    "Biliary Tract": "#72a0c1",
    "Other tissue": "#a32638",
    "Small Intestine": "#9966cc",
    "Placenta": "#f19cbb",
    "Testis": "#e32636",
    "Adrenal Gland": "#3b7a57",
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
    "MRC-5 (EXPO)": "#756bb1",
    "MRC-5 Arrested (5 days @100% confluent)": "#9e9ac8",
    "MRC-5 Arrested (8 days @100% confluent)": "#bcbddc",
}

PPI_PAL = {
    "T": "#fc8d62",
    "1": "#656565",
    "2": "#7c7c7c",
    "3": "#949494",
    "4": "#ababab",
    "5+": "#c3c3c3",
    "-": "#2b8cbe",
    "X": "#2ca02c",
}

PPI_ORDER = ["T", "1", "2", "3", "4", "5+", "-"]


def two_vars_correlation(var1, var2, idx_set=None, method="pearson", min_n=15, verbose=0):
    if verbose > 0:
        print(f"Var1={var1.name}; Var2={var2.name}")

    if idx_set is None:
        idx_set = set(var1.dropna().index).intersection(var2.dropna().index)

    else:
        idx_set = set(var1.reindex(idx_set).dropna().index).intersection(
            var2.reindex(idx_set).dropna().index
        )

    if len(idx_set) <= min_n:
        return dict(corr=np.nan, pval=np.nan, len=len(idx_set))

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
    def lm_ppi_annotate_table(cls, table, ppi, y_skew, drug_targets=None):
        if drug_targets is not None:
            # Machine learning scores
            ml_scores = pd.read_csv(
                f"{cls.DPATH}/score_dl_min300_ic50_eg_id_20211005.csv", index_col=0
            )["corr"]
            table["r2"] = ml_scores.reindex(table["y_id"]).values

            # Drug target annotation
            table["target"] = drug_targets.loc[table["y_id"]].values

            # PPI annotation
            table = PPI.ppi_annotation(
                table, ppi, x_var="target", y_var="x_id", ppi_var="ppi"
            )

            # Skewness annotation
            table["skew"] = y_skew.loc[table["y_id"]].values

        else:
            # Machine learning scores
            ml_scores = pd.read_csv(
                f"{cls.DPATH}/score_dl_crispr_protein_20211005.csv", index_col=0
            )["corr"]
            table["r2"] = ml_scores.reindex(table["y_id"]).values

            # PPI annotation
            table = PPI.ppi_annotation(
                table, ppi, x_var="y_id", y_var="x_id", ppi_var="ppi"
            )

            # Skewness annotation
            table["skew"] = y_skew.loc[table["y_id"]].values

        return table

    @classmethod
    def read_samplesheet(cls):
        """
        Read cancer cell lines samplesheet

        :return:
        """
        return pd.read_excel(
            f"{cls.DPATH}/SupplementaryTable1.xlsx", index_col=0, sheet_name="Cell lines"
        )

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
            f"{cls.DPATH}/e0022_diann_051021_sample_mapping_averaged.txt", index_col=0, sep="\t"
        )

        # Remove excluded samples
        exclude_man = manifest[~manifest["SIDM"].isin(ss.index)]
        manifest = manifest[~manifest.index.isin(exclude_man.index)]

        return manifest

    @classmethod
    def read_protein_matrix_unfiltered(cls, prot_file, map_protein=False):
        protein = pd.read_csv(f"{cls.DPATH}/{prot_file}", sep="\t", index_col=0).T

        protein = protein.drop(index=["RT-Kit-WR", "ProCal", "RMISv2"], errors="ignore")

        protein.columns = [c.split(";")[0] for c in protein]
        protein.index = [c.split(";")[1] for c in protein.index]

        if map_protein:
            pmap = cls.map_gene_name().reindex(protein.index)["GeneSymbol"].dropna()

            protein = protein[protein.index.isin(pmap.index)]
            protein = protein.groupby(pmap.reindex(protein.index)).mean()

        return protein

    @classmethod
    def read_protein_matrix(cls, map_protein=False, subset=None, min_measurements=None):
        """
        Read protein level matrix.

        :return:
        """
        # Read protein level normalised intensities
        protein = pd.read_csv(
            f"{cls.DPATH}/e0022_diann_051021_working_matrix_averaged.txt",
            sep="\t",
            index_col=0,
        ).T

        protein.columns = [c.split(";")[0] for c in protein]
        protein.index = [c.split(";")[1] for c in protein.index]

        exclude_controls = [
            "Control_HEK293T_lys",
            "Control_HEK293T_std_H002",
            "Control_HEK293T_std_H003",
        ]
        protein = protein.drop(columns=exclude_controls, errors="ignore")

        # Map protein to gene symbols
        if map_protein:
            pmap = cls.map_gene_name().reindex(protein.index)["GeneSymbol"].dropna()

            protein = protein[protein.index.isin(pmap.index)]
            protein = protein.groupby(pmap.reindex(protein.index)).mean()

        if subset is not None:
            protein = protein.loc[:, protein.columns.isin(subset)]

        if min_measurements is not None:
            protein = protein[protein.count(1) >= min_measurements]

        return protein

    @classmethod
    def read_protein_perturbation_manifest(cls):
        manifest = (
            pd.read_csv(
                f"{cls.DPATH}/E0019_BreastCan_ProteinMatrix_LoessNorm_byDiffacto.csv",
                index_col=0,
            )
            .T.iloc[:12]
            .T
        )

        manifest = manifest.replace(
            {
                "Cell Line": {
                    "T-47D 10%FBS": "T-47D 10% FBS",
                    "T-47D 1%FBS": "T-47D 1% FBS",
                }
            }
        )

        return manifest

    @classmethod
    def read_protein_perturbation(cls, map_protein=False, subset=None):
        """
        Read protein level matrix.

        :return:
        """
        # Read protein level normalised intensities
        protein = (
            pd.read_csv(
                f"{cls.DPATH}/E0019_BreastCan_ProteinMatrix_LoessNorm_byDiffacto.csv",
                index_col=0,
            )
            .T.iloc[12:]
            .astype(float)
        )

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
        protein_raw_mean = pd.read_csv(
            f"{cls.DPATH}/e0022_diann_protein_matrix_070921_raw_averaged.txt.gz",
            sep="\t",
            index_col=0,
        ).T

        protein_raw_mean.columns = [i.split(";")[0] for i in protein_raw_mean]
        protein_raw_mean.index = [i.split(";")[1] for i in protein_raw_mean.index]

        # Map protein
        pmap = cls.map_gene_name().reindex(protein_raw_mean.index)["GeneSymbol"].dropna()
        protein_raw_mean = protein_raw_mean[protein_raw_mean.index.isin(pmap.index)]
        protein_raw_mean = protein_raw_mean.groupby(pmap.reindex(protein_raw_mean.index)).mean()

        # Calculate mean
        protein_raw_mean = protein_raw_mean.mean(1)

        return protein_raw_mean

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
    def map_gene_name(cls, index_col="Entry name", value_col="Gene names  (primary )"):
        idmap = pd.read_csv(f"{cls.DPATH}/uniprot_human_idmap.tab.gz", sep="\t")

        if index_col is not None:
            idmap = idmap.dropna(subset=[index_col]).set_index(index_col)

        idmap["GeneSymbol"] = idmap[value_col].apply(
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
        merged = (
            pd.read_csv(
                f"{cls.DPATH}/crispr_proteomics_model_id_20210223.csv.gz",
                sep=",",
            )
            .drop(columns=["Cell_line"])
            .set_index("model_id")
            .T
        )

        sid = cls.read_crispr_sids()
        merged = merged.rename(columns=sid)

        if subset is not None:
            merged = merged.loc[:, merged.columns.isin(subset)]

        return merged

    @classmethod
    def read_crispr_institute(cls):
        cannot = pd.read_csv(
            f"{cls.DPATH}/crispr_proteomics_model_id_annotation_20210223.tsv", sep="\t"
        )
        cannot = cannot.query("Duplicate == False").dropna(subset=["CMP_ID"])
        return cannot.set_index("CMP_ID")["source"]

    @staticmethod
    def read_cmp_samplesheet():
        cmp_ss = pd.read_csv("https://cog.sanger.ac.uk/cmp/download/model_list_20210719.csv", index_col=0)
        return cmp_ss

    @classmethod
    def read_mobem(cls, drop_factors=True, add_msi=True):
        idmap = cls.read_cmp_samplesheet()
        idmap = idmap.reset_index().dropna(subset=["COSMIC_ID", "model_id"]).set_index("COSMIC_ID")["model_id"]

        mobem = pd.read_csv(f"{cls.DPATH}/PANCAN_mobem.csv.gz", index_col=0)
        mobem = mobem[mobem.index.astype(str).isin(idmap.index)]
        mobem = mobem.set_index(idmap[mobem.index.astype(str)].values)

        if drop_factors is not None:
            mobem = mobem.drop(columns={"TISSUE_FACTOR", "MSI_FACTOR", "MEDIA_FACTOR"})

        if add_msi:
            msi_status = cls.read_cmp_samplesheet()["msi_status"]
            mobem["msi_status"] = (msi_status.loc[mobem.index] == "MSI").astype(int).values

        mobem = mobem.astype(int).T

        return mobem

    @classmethod
    def read_drug_response(
        cls,
        as_matrix=True,
        drug_columns=["drug_id", "drug_name", "dataset"],
        sample_columns=["model_id"],
        dtype="ln_IC50",
        subset=None,
        min_measurements=None,
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

            drugresponse = drugresponse.set_index(
                pd.Series([";".join(map(str, i)) for i in drugresponse.index])
            )

        if subset is not None:
            drugresponse = drugresponse.loc[:, drugresponse.columns.isin(subset)]

        if min_measurements is not None:
            drugresponse = drugresponse[drugresponse.count(1) > 300]

        return drugresponse

    @classmethod
    def read_drug_max_concentration(
        cls, drug_columns=["drug_id", "drug_name", "dataset"]
    ):
        drugresponse = cls.read_drug_response(as_matrix=False)

        maxconcentration = drugresponse.groupby(drug_columns)[
            "max_screening_conc"
        ].first()

        maxconcentration.index = [";".join(map(str, i)) for i in maxconcentration.index]

        return maxconcentration

    @classmethod
    def read_drug_target(cls):
        drugresponse = cls.read_drug_response(as_matrix=False)

        dtargets = drugresponse.groupby(["drug_id", "drug_name", "dataset"])[
            "putative_gene_target"
        ].first()

        dtargets.index = [";".join(map(str, i)) for i in dtargets.index]

        dtargets = dtargets.fillna(np.nan)

        return dtargets

    @classmethod
    def read_methylation_matrix(cls, subset=None):
        methy_promoter = pd.read_csv(
            f"{cls.DPATH}/methy_beta_gene_promoter.csv.gz", index_col=0
        )

        if subset is not None:
            methy_promoter = methy_promoter.loc[:, methy_promoter.columns.isin(subset)]

        return methy_promoter

    @classmethod
    def read_ctrp(cls, subset=None):
        drespo = pd.read_csv(f"{cls.DPATH}/CTRPv2.0_AUC_parsed.csv.gz", index_col=0)

        return drespo

    @classmethod
    def read_wes(cls):
        data = pd.read_csv(f"{cls.DPATH}/WES_variants.csv.gz")
        return data

    @classmethod
    def read_prism(cls):
        ss = cls.read_samplesheet().reset_index().set_index("model_name")
        data = pd.read_csv(f"{cls.DPATH}/drug_all_ccle_secondary_processed_auc.csv")
        data["model_id"] = ss.loc[data["cell_line_name"], "model_id"].values
        data = pd.pivot_table(data, index="drug_id", columns="model_id", values="auc")
        return data

    @classmethod
    def read_prism_depmap_19q4(cls):
        data = pd.read_csv(
            f"{cls.DPATH}/secondary-screen-dose-response-curve-parameters.csv"
        )
        data = pd.pivot_table(data, index="broad_id", columns="depmap_id", values="auc")
        return data


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
