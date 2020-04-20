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
from scipy.stats import chi2
from natsort import natsorted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from cancer_proteomics.eg.LMModels import LMModels
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, CRISPR, GeneExpression


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


class LModel:
    def __init__(
        self, Y, X, M, M2=None, normalize=False, fit_intercept=True, copy_X=True, n_jobs=4
    ):
        self.samples = set.intersection(set(Y.index), set(X.index), set(M.index), set(Y.index) if M2 is None else set(M2.index))

        self.X = X.loc[self.samples]
        self.X = self.X.loc[:, self.X.count() > (M.shape[1] + (1 if M2 is None else 2))]
        self.X_ma = np.ma.masked_invalid(self.X.values)

        self.Y = Y.loc[self.samples]

        self.M = M.loc[self.samples]

        self.M2 = M2.loc[self.samples, self.X.columns] if M2 is not None else M2

        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs

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
    gexp_obj, prot_obj, crispr_obj = GeneExpression(), Proteomics(), CRISPR()

    # Samples
    #
    samples = set.intersection(
        set(prot_obj.get_data()), set(crispr_obj.get_data(dtype="merged"))
    )
    LOG.info(f"Samples: {len(samples)}")

    # Filter data-sets
    gexp = gexp_obj.filter(subset=samples)
    LOG.info(f"Transcriptomics: {gexp.shape}")

    prot = prot_obj.filter(subset=samples)
    LOG.info(f"Proteomics: {prot.shape}")

    crispr = crispr_obj.filter(dtype="merged", subset=samples)
    LOG.info(f"CRISPR: {crispr.shape}")

    # Covariates
    #
    covariates = LMModels.define_covariates(
        institute=crispr_obj.merged_institute,
        medium=True,
        cancertype=False,
        tissuetype=True,
        mburden=False,
        ploidy=True,
    )
    covariates = covariates.reindex(samples).dropna()

    samples = set(covariates.index)
    LOG.info(f"Samples={len(samples)}; Covariates={covariates.shape[1]}")

    # Reduce to independent variables with more observations than covariates
    #
    prot = prot[prot[samples].count(1) > (covariates.shape[1] + 1)]
    LOG.info(f"Proteomics after filter: {prot.shape}")

    # LMs: CRISPR ~ Protein
    #
    prot_lm = LModel(
        Y=crispr[samples].T, X=prot[samples].T, M=covariates.loc[samples]
    ).fit_matrix()

    prot_lm.to_csv(
        f"{RPATH}/lm_sklearn_protein_crispr.csv.gz", index=False, compression="gzip"
    )

    # LM: CRISPR ~ Gene-expression
    #
    gexp = gexp.reindex(
        index=natsorted(set(gexp.index).intersection(prot.index)),
        columns=samples.intersection(set(gexp.columns)),
    )
    LOG.info(f"Transcriptomics after filter: {gexp.shape}")

    gexp_lm = LModel(
        Y=crispr[gexp.columns].T, X=gexp.T, M=covariates.loc[gexp.columns]
    ).fit_matrix()

    gexp_lm.to_csv(
        f"{RPATH}/lm_sklearn_transcript_crispr.csv.gz", index=False, compression="gzip"
    )

    # LM: CRISPR ~ Protein - Gene-expression
    #
    prot = prot.reindex(index=gexp.index, columns=gexp.columns)
    prot = prot[prot.count(1) > (covariates.shape[1] + 2)]
    gexp = gexp.reindex(index=prot.index, columns=prot.columns)
    LOG.info(f"Proteomics after filter with GExp: {prot.shape}")

    degr_lm = LModel(
        Y=crispr[gexp.columns].T,
        X=prot.T,
        M=covariates.loc[gexp.columns],
        M2=gexp.T,
    ).fit_matrix()

    degr_lm.to_csv(
        f"{RPATH}/lm_sklearn_degr_crispr.csv.gz", index=False, compression="gzip"
    )

