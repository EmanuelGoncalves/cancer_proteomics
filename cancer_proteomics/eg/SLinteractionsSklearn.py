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
from crispy import QCplot
from scipy.stats import chi2
from natsort import natsorted
from crispy.GIPlot import GIPlot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from cancer_proteomics.eg.LMModels import LMModels
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, CRISPR, GeneExpression, DrugResponse, Sample


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


class LModel:
    def __init__(
        self,
        Y,
        X,
        M,
        M2=None,
        normalize=False,
        fit_intercept=True,
        copy_X=True,
        n_jobs=4,
        verbose=1,
    ):
        self.samples = set.intersection(
            set(Y.index),
            set(X.index),
            set(M.index),
            set(Y.index) if M2 is None else set(M2.index),
        )

        self.X = X.loc[self.samples]
        self.X = self.X.loc[:, self.X.count() > (M.shape[1] + (1 if M2 is None else 2))]
        self.X_ma = np.ma.masked_invalid(self.X.values)

        self.Y = Y.loc[self.samples]
        self.Y = self.Y.loc[:, self.Y.std() > 0]

        self.M = M.loc[self.samples]

        self.M2 = M2.loc[self.samples, self.X.columns] if M2 is not None else M2

        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.log = logging.getLogger("Crispy")

    def model_regressor(self):
        regressor = LinearRegression(
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
        )
        return regressor

    @staticmethod
    def loglike(y_true, y_pred):
        nobs = len(y_true)
        nobs2 = nobs / 2.0

        ssr = np.power(y_true - y_pred, 2).sum()

        llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2

        return llf

    def fit_matrix(self):
        lms = []

        # x_idx, x_var, y_var = 3167, "VIM", "BAX"
        for x_idx, x_var in enumerate(self.X):
            if self.verbose > 0:
                self.log.info(f"LM={x_var} ({x_idx})")

            # Mask NaNs
            x_ma = np.ma.mask_rowcols(self.X_ma[:, [x_idx]], axis=0)

            # Build matrices
            x = self.X.iloc[~x_ma.mask.any(axis=1), x_idx]
            x = pd.DataFrame(
                StandardScaler().fit_transform(x.to_frame()),
                index=x.index,
                columns=[x_var],
            )

            y = self.Y.iloc[~x_ma.mask.any(axis=1), :]
            y = pd.DataFrame(
                StandardScaler().fit_transform(y), index=y.index, columns=y.columns
            )

            # Covariate matrix (remove invariable features and add noise)
            m = self.M.iloc[~x_ma.mask.any(axis=1), :]
            if self.M2 is not None:
                m2 = self.M2.iloc[~x_ma.mask.any(axis=1), x_idx]
                m2 = pd.DataFrame(
                    StandardScaler().fit_transform(m2.to_frame()),
                    index=m2.index,
                    columns=[x_var],
                )
                m = pd.concat([m2, m], axis=1)
            m = m.loc[:, m.std() > 0]
            m += np.random.normal(0, 1e-4, m.shape)

            # Fit covariate model
            lm_small = self.model_regressor().fit(m, y)
            lm_small_ll = self.loglike(y, lm_small.predict(m))

            # Fit full model: covariates + feature
            lm_full_x = np.concatenate([m, x], axis=1)
            lm_full = self.model_regressor().fit(lm_full_x, y)
            lm_full_ll = self.loglike(y, lm_full.predict(lm_full_x))

            # Log-ratio test
            lr = 2 * (lm_full_ll - lm_small_ll)
            lr_pval = chi2(1).sf(lr)

            # Assemble + append results
            res = pd.DataFrame(
                dict(
                    y=y.columns,
                    x=x_var,
                    n=len(x),
                    b=lm_full.coef_[:, -1],
                    lr=lr.values,
                    covs=m.shape[1],
                    pval=lr_pval,
                    fdr=multipletests(lr_pval, method="fdr_bh")[1],
                )
            )
            lms.append(res)

        lms = pd.concat(lms, ignore_index=True).sort_values("pval")

        return lms


if __name__ == "__main__":
    # Data-sets
    #
    gexp_obj = GeneExpression()
    prot_obj = Proteomics()
    crispr_obj = CRISPR()
    drespo_obj = DrugResponse()

    # Samples
    #
    samples_crispr = set.intersection(
        set(prot_obj.get_data()), set(crispr_obj.get_data(dtype="merged"))
    )
    LOG.info(f"CRISPR samples: {len(samples_crispr)}")

    samples_drug = set.intersection(
        set(prot_obj.get_data()), set(drespo_obj.get_data())
    )
    LOG.info(f"Drug samples: {len(samples_drug)}")

    # Filter data-sets
    gexp = gexp_obj.filter()
    LOG.info(f"Gexp: {gexp.shape}")

    prot = prot_obj.filter()
    LOG.info(f"Prot: {prot.shape}")

    crispr = crispr_obj.filter(dtype="merged")
    LOG.info(f"CRISPR: {crispr.shape}")

    drespo = drespo_obj.filter()
    drespo.index = [";".join(map(str, i)) for i in drespo.index]
    drespo = drespo.drop(["2638;AZD5991;GDSC2", "2459;Rabusertib;GDSC2", "2502;CCT241533;GDSC2", "2265;Galunisertib;GDSC2"])
    LOG.info(f"Drug: {drespo.shape}")

    # Drug PCA
    #
    drespo_fillna = drespo[(drespo.count(1) / drespo.shape[1]) > 0.5]
    drespo_fillna = drespo_fillna.T.fillna(drespo_fillna.T.mean()).T

    n_components = 10
    pc_labels = [f"PC{i+1}" for i in range(n_components)]

    drug_pca = PCA(n_components=n_components).fit(drespo_fillna.T)
    drug_vexp = pd.Series(drug_pca.explained_variance_ratio_, index=pc_labels)
    drug_pcs = pd.DataFrame(
        drug_pca.transform(drespo_fillna.T), index=drespo_fillna.columns, columns=pc_labels
    )

    drug_pca_df = pd.concat([
        Sample().samplesheet.reindex(index=drug_pcs.index, columns=["growth"]),
        drug_pcs,
    ], axis=1)

    #
    g = sns.clustermap(
        drug_pca_df.corr(),
        cmap="Spectral",
        annot=True,
        center=0,
        fmt=".2f",
        annot_kws=dict(size=4),
        lw=0.05,
        figsize=(3, 3),
    )

    plt.savefig(
        f"{RPATH}/drug_pca_clustermap.pdf",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close("all")

    #
    y_var = "PC1"
    g = GIPlot.gi_regression("growth", y_var, drug_pca_df, lowess=True)
    g.set_axis_labels("Growth rate", f"{y_var} ({drug_vexp[y_var]*100:.1f}%)")
    plt.savefig(
        f"{RPATH}/drug_pca_regression_growth.pdf",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close("all")

    # Covariates
    #

    # CRISPR
    covs_crispr = LMModels.define_covariates(
        institute=crispr_obj.merged_institute,
        medium=True,
        cancertype=False,
        tissuetype=True,
        mburden=False,
        ploidy=True,
    )
    covs_crispr = covs_crispr.reindex(samples_crispr).dropna()
    samples_crispr = set(covs_crispr.index)
    LOG.info(
        f"CRISPR: Samples={len(samples_crispr)}; Covs={covs_crispr.shape[1]}"
    )

    # Drug
    covs_drug = LMModels.define_covariates(
        institute=False,
        medium=True,
        cancertype=False,
        tissuetype=True,
        mburden=False,
        ploidy=True,
    )
    covs_drug = pd.concat([covs_drug, drug_pcs["PC1"]], axis=1)
    covs_drug = covs_drug.reindex(samples_drug).dropna()
    samples_drug = set(covs_drug.index)
    LOG.info(f"Drug: Samples={len(samples_drug)}; Covs={covs_drug.shape[1]}")

    # Reduce to independent variables with more observations than covariates
    #
    proteins_crispr = set(
        prot[prot[samples_crispr].count(1) > (covs_crispr.shape[1] + 1)].index
    )
    proteins_drug = set(
        prot[prot[samples_drug].count(1) > (covs_drug.shape[1] + 1)].index
    )
    LOG.info(
        f"Prot after filter: CRISPR={len(proteins_crispr)}; Drug={len(proteins_drug)}"
    )

    genes_crispr = set(gexp.index).intersection(proteins_crispr)
    genes_drug = set(gexp.index).intersection(proteins_drug)
    LOG.info(
        f"Gexp after filter: CRISPR={len(genes_crispr)}; Drug={len(genes_drug)}"
    )

    # Gene expression samples overlap
    #
    samples_crispr_gexp = samples_crispr.intersection(gexp)
    samples_drug_gexp = samples_drug.intersection(gexp)
    LOG.info(
        f"Gexp samples: CRISPR={len(samples_crispr_gexp)}; Drug={len(samples_drug_gexp)}"
    )

    # Protein and gene reduce to independent variables with more observations than covariates
    #
    proteins_gexp_crispr = set(
        prot[prot[samples_crispr_gexp].count(1) > (covs_crispr.shape[1] + 2)].index
    ).intersection(genes_crispr)
    proteins_gexp_drug = set(
        prot[prot[samples_drug_gexp].count(1) > (covs_drug.shape[1] + 2)].index
    ).intersection(genes_drug)
    LOG.info(
        f"Prot and Gexp after filter: CRISPR={len(proteins_gexp_crispr)}; Drug={len(proteins_gexp_drug)}"
    )

    # LMs: CRISPR
    #

    # Protein
    prot_lm = LModel(
        Y=crispr[samples_crispr].T,
        X=prot.loc[proteins_crispr, samples_crispr].T,
        M=covs_crispr.loc[samples_crispr],
    ).fit_matrix()
    prot_lm.to_csv(
        f"{RPATH}/lm_sklearn_protein_crispr.csv.gz", index=False, compression="gzip"
    )

    # Gene-expression
    gexp_lm = LModel(
        Y=crispr[samples_crispr_gexp].T,
        X=gexp.loc[genes_crispr, samples_crispr_gexp].T,
        M=covs_crispr.loc[samples_crispr_gexp],
    ).fit_matrix()
    gexp_lm.to_csv(
        f"{RPATH}/lm_sklearn_transcript_crispr.csv.gz", index=False, compression="gzip"
    )

    # Protein - Gene-expression
    degr_lm = LModel(
        Y=crispr[samples_crispr_gexp].T,
        X=prot.loc[proteins_gexp_crispr, samples_crispr_gexp].T,
        M=covs_crispr.loc[samples_crispr_gexp],
        M2=gexp.loc[proteins_gexp_crispr, samples_crispr_gexp].T,
    ).fit_matrix()
    degr_lm.to_csv(
        f"{RPATH}/lm_sklearn_degr_crispr.csv.gz", index=False, compression="gzip"
    )

    # LMs: Drug
    #
    drespo_meas = pd.Series({d: ";".join(drespo.loc[d].dropna().index) for d in drespo.index})

    drespo_group = drespo.count(1).rename("count")
    drespo_group = drespo_group.to_frame().assign(drug_id=drespo_group.index)
    drespo_group = drespo_group.groupby(drespo_meas).agg(set)
    drespo_group["count"] = drespo_group["count"].apply(lambda v: list(v)[0])
    drespo_group["count_drugs"] = drespo_group["drug_id"].apply(lambda v: len(v))
    drespo_group = drespo_group.sort_values("count_drugs", ascending=False)

    # Protein
    drug_prot_lm = []
    for idx, row in drespo_group.iterrows():
        d_samples = set(idx.split(";")).intersection(samples_drug)
        d_ids = list(row["drug_id"])
        LOG.info(f"{d_ids}")

        if len(d_samples) <= covs_drug.shape[1] + 1:
            continue

        d_lm = LModel(
            Y=drespo.loc[d_ids, d_samples].T,
            X=prot.loc[proteins_drug, d_samples].T,
            M=covs_drug.loc[d_samples],
            verbose=0,
        ).fit_matrix()

        drug_prot_lm.append(d_lm)

    drug_prot_lm = pd.concat(drug_prot_lm).sort_values("pval")
    drug_prot_lm.to_csv(
        f"{RPATH}/lm_sklearn_protein_drug.csv.gz", index=False, compression="gzip"
    )

    # Gene-expression
    drug_gexp_lm = []
    for idx, row in drespo_group.iterrows():
        d_samples = set(idx.split(";")).intersection(samples_drug_gexp)
        d_ids = list(row["drug_id"])
        LOG.info(f"{d_ids}")

        if len(d_samples) <= covs_drug.shape[1] + 1:
            continue

        d_lm = LModel(
            Y=drespo.loc[d_ids, d_samples].T,
            X=gexp.loc[genes_drug, d_samples].T,
            M=covs_drug.loc[d_samples],
            verbose=0,
        ).fit_matrix()

        drug_gexp_lm.append(d_lm)

    drug_gexp_lm = pd.concat(drug_gexp_lm).sort_values("pval")
    drug_gexp_lm.to_csv(
        f"{RPATH}/lm_sklearn_transcript_drug.csv.gz", index=False, compression="gzip"
    )

    # Protein - Gene-expression
    drug_degr_lm = []
    for idx, row in drespo_group.iterrows():
        d_samples = set(idx.split(";")).intersection(samples_drug_gexp)
        d_ids = list(row["drug_id"])
        LOG.info(f"{d_ids}")

        if len(d_samples) <= covs_drug.shape[1] + 2:
            continue

        d_lm = LModel(
            Y=drespo.loc[d_ids, d_samples].T,
            X=prot.loc[proteins_gexp_drug, d_samples].T,
            M=covs_drug.loc[d_samples],
            M2=gexp.loc[proteins_gexp_drug, d_samples].T,
            verbose=0,
        ).fit_matrix()

        drug_degr_lm.append(d_lm)

    drug_degr_lm = pd.concat(drug_degr_lm).sort_values("pval")
    drug_degr_lm.to_csv(
        f"{RPATH}/lm_sklearn_degr_drug.csv.gz", index=False, compression="gzip"
    )
