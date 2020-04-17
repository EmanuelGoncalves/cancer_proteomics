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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from cancer_proteomics.eg.LMModels import LMModels
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, CRISPR


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


class LModel:
    def __init__(self, Y, X, M):
        self.samples = set.intersection(set(Y.index), set(X.index), set(M.index))

        self.Y = Y.loc[samples]
        self.Y_ma = np.ma.masked_invalid(self.Y.values)

        self.X = X.loc[samples]
        self.M = M.loc[samples]

        self.normalize = True
        self.fit_intercept = True
        self.copy_X = True

        self.log = logging.getLogger("Crispy")

    def model_regressor(self):
        regressor = LinearRegression(
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy_X=self.copy_X,
        )
        return regressor

    @staticmethod
    def loglike(y_true, y_pred):
        nobs = len(y_true)
        nobs2 = len(y_true) / 2.0

        ssr = np.power(y_true - y_pred, 2).sum()

        llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2

        return llf

    def fit_matrix(self):
        # y_idx, y_var = 1473, "KRT7"
        lms = []
        for y_idx, y_var in enumerate(self.Y):
            self.log.info(f"LM={y_var} ({y_idx})")

            # Mask NaNs
            y_ma = np.ma.mask_rowcols(self.Y_ma[:, [y_idx]], axis=0)

            # Build matrices
            y = self.Y.iloc[~y_ma.mask.any(axis=1), y_idx]
            y = pd.Series(StandardScaler().fit_transform(y.to_frame())[:, 0], index=y.index)

            x = self.X.iloc[~y_ma.mask.any(axis=1), :]
            x = pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)

            m = self.M.iloc[~y_ma.mask.any(axis=1), :]
            m = m.loc[:, m.std() > 0]

            # Fit covariate model
            lm_small = self.model_regressor().fit(m, y)
            lm_small_ll = self.loglike(y, lm_small.predict(m))

            # Iterate over all possible features
            y_lms = []
            # x_idx, x_var = 3309, "CD8B"
            for x_idx, x_var in enumerate(self.X):
                x_ = np.concatenate([m, x.iloc[:, [x_idx]]], axis=1)

                # Fit full model
                lm_full = self.model_regressor().fit(x_, y)
                lm_full_ll = self.loglike(y, lm_full.predict(x_))

                # Log-ratio test
                lr = 2 * (lm_full_ll - lm_small_ll)
                lr_pval = chi2(1).sf(lr)

                # Results
                y_lms.append(
                    dict(
                        y=y_var,
                        x=x_var,
                        n=y.shape[0],
                        b=lm_full.coef_[-1],
                        lr=lr,
                        pval=lr_pval,
                        covs=m.shape[1],
                    )
                )

            y_lms = pd.DataFrame(y_lms)
            y_lms["fdr"] = multipletests(y_lms["pval"], method="fdr_bh")[1]

            lms.append(y_lms)
        lms = pd.concat(lms, ignore_index=True).sort_values("pval")
        return lms


# Data-sets
#
prot_obj, crispr_obj = Proteomics(), CRISPR()

# Samples
#
samples = set.intersection(
    set(prot_obj.get_data()),
    set(crispr_obj.get_data(dtype="merged")),
)
LOG.info(f"Samples: {len(samples)}")

# Filter data-sets
#
prot = prot_obj.filter(subset=samples, perc_measures=0.03)
LOG.info(f"Proteomics: {prot.shape}")

crispr = crispr_obj.filter(dtype="merged", subset=samples, abs_thres=0.5, min_events=5)
LOG.info(f"CRISPR: {crispr.shape}")

# Covariates
#
covariates = LMModels.define_covariates(
    institute=crispr_obj.merged_institute, cancertype=False, tissuetype=True, mburden=False, ploidy=False
)

# Protein ~ CRISPR LMMs
#
prot_lm = LModel(
    Y=prot[samples].T, X=crispr[samples].T, M=covariates.loc[samples]
).fit_matrix()
prot_lm.to_csv(f"{RPATH}/lm_sklearn_protein_crispr.csv.gz", index=False, compression="gzip")
