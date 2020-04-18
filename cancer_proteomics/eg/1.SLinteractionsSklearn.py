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
import matplotlib.pyplot as plt
from scipy.stats import chi2, spearmanr
from sklearn.linear_model import LinearRegression
from cancer_proteomics.eg.LMModels import LMModels
from crispy.DataImporter import Proteomics, CRISPR
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler, quantile_transform
from crispy.DimensionReduction import dim_reduction_pca


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


class LModel:
    def __init__(
        self, Y, X, M, normalize=False, fit_intercept=True, copy_X=True, n_jobs=4
    ):
        self.samples = set.intersection(set(Y.index), set(X.index), set(M.index))

        self.X = X.loc[samples]
        self.X_ma = np.ma.masked_invalid(self.X.values)

        self.Y = Y.loc[samples]

        self.M = M.loc[samples]

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

        # x_idx, x_var, y_var = 510, "CLEC2B", "PSMF1"
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

            m = self.M.iloc[~x_ma.mask.any(axis=1), :]
            m = m.loc[:, m.std() > 0]
            m += np.random.normal(0, 1e-4, m.shape)
            # m = m.loc[:, (m.dtypes != np.uint8) | (m.sum() > 3)]

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


# Data-sets
#

prot_obj, crispr_obj = Proteomics(), CRISPR()


# Samples
#

samples = set.intersection(
    set(prot_obj.get_data()), set(crispr_obj.get_data(dtype="merged"))
)
LOG.info(f"Samples: {len(samples)}")


# Filter data-sets

prot = prot_obj.filter(subset=samples, perc_measures=0.03)
LOG.info(f"Proteomics: {prot.shape}")

crispr = crispr_obj.filter(dtype="merged", subset=samples, abs_thres=0.5, min_events=5)
LOG.info(f"CRISPR: {crispr.shape}")


#
#

crispr_all = crispr_obj.filter(dtype="merged", subset=samples)
crispr_pca = dim_reduction_pca(crispr_all)

prot_all = prot_obj.filter(subset=samples, perc_measures=0.5)
prot_all = prot_all.T.fillna(prot_all.T.mean()).T
prot_pca = dim_reduction_pca(prot_all)

s_pg_corr = pd.read_csv(
    f"{RPATH}/2.SLProteinInteractions_gexp_prot_samples_corr.csv", index_col=0
)

putative_covariates = pd.concat(
    [
        prot.count().rename("Proteomics n. measurements"),
        prot_obj.protein_raw.median().rename("Global proteomics"),
        s_pg_corr["corr"].rename("gexp_prot_corr"),
        pd.get_dummies(prot_obj.ss["msi_status"]),
        pd.get_dummies(prot_obj.ss["growth_properties"]),
        pd.get_dummies(prot_obj.ss["tissue"])["Haematopoietic and Lymphoid"],
        prot_obj.ss.reindex(
            index=samples, columns=["ploidy", "mutational_burden", "growth"]
        ),
    ],
    axis=1,
).loc[crispr_pca[0].index]

putative_covariates_corr = {
    d: pd.DataFrame(
        {
            f: {
                c: spearmanr(df[f], putative_covariates[c], nan_policy="omit")[0]
                for c in putative_covariates
            }
            for f in df
        }
    )
    for d, df in [("crispr", crispr_pca[0]), ("prot", prot_pca[0])]
}

for d, df in putative_covariates_corr.items():
    fig = sns.clustermap(
        df,
        cmap="Spectral",
        center=0,
        cbar=False,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"fontsize": 5},
        cbar_pos=None,
        figsize=[df.shape[0] * 0.4] * 2,
    )
    fig.ax_heatmap.set_xlabel("")
    fig.ax_heatmap.set_ylabel("")

    plt.savefig(
        f"{RPATH}/1.SL_covariates_pca_corrmap_{d}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


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
# covariates = pd.concat([
#     covariates,
#     pd.get_dummies(prot_obj.ss["tissue"])["Haematopoietic and Lymphoid"],
# ], axis=1)
covariates = covariates.reindex(samples).dropna()

samples = set(covariates.index)
LOG.info(f"Samples={len(samples)}; Covariates={covariates.shape[1]}")


# Reduce to independent variables with more observations than covariates
#

prot = prot[prot[samples].count(1) > (covariates.shape[1] + 1)]


# Protein ~ CRISPR LMMs
#

prot_lm = LModel(
    Y=crispr[samples].T, X=prot[samples].T, M=covariates.loc[samples]
).fit_matrix()


# Export
#

prot_lm.to_csv(
    f"{RPATH}/lm_sklearn_protein_crispr.csv.gz", index=False, compression="gzip"
)
