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

import os
import logging
import argparse
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from natsort import natsorted
from scipy.stats import spearmanr, chi2
from sklearn.linear_model import LinearRegression
from cancer_proteomics.eg.LMModels import LMModels
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR


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

    def fit(self, y_idx, x_idx):
        y_ma = np.ma.mask_rowcols(self.Y_ma[:, [y_idx]], axis=0)

        y = self.Y.iloc[~y_ma.mask.any(axis=1), y_idx]
        x = self.X.iloc[~y_ma.mask.any(axis=1), [x_idx]]
        m = self.M.iloc[~y_ma.mask.any(axis=1), :]

        lm_full = self.model_regressor()
        lm_full = lm_full.fit(np.concatenate([m, x], axis=1), y)
        lm_full_ll = LMModels.log_likelihood(
            y, lm_full.predict(np.concatenate([m, x], axis=1))
        )

        lm_small = self.model_regressor()
        lm_small = lm_small.fit(m, y)
        lm_small_ll = LMModels.log_likelihood(y, lm_small.predict(m))

        lr = 2 * (lm_full_ll - lm_small_ll)
        lr_pval = chi2(1).sf(lr)

        res = dict(n=y.shape[0], b=lm_full.coef_[-1], lr=lr, pval=lr_pval)

        return res

    def fit_matrix(self):
        # y_idx, y_var = 3277, "VPS4A"
        lms = []
        for y_idx, y_var in enumerate(self.Y):
            self.log.info(f"LM={y_var} ({y_idx})")

            # Mask NaNs
            y_ma = np.ma.mask_rowcols(self.Y_ma[:, [y_idx]], axis=0)

            # Build matrices
            y = self.Y.iloc[~y_ma.mask.any(axis=1), y_idx]
            x = self.X.iloc[~y_ma.mask.any(axis=1), :]
            m = self.M.iloc[~y_ma.mask.any(axis=1), :]

            # Fit covariate model
            lm_small = self.model_regressor().fit(m, y)
            lm_small_ll = LMModels.log_likelihood(y, lm_small.predict(m))

            # Iterate over all possible features
            y_lms = []
            for x_idx, x_var in enumerate(self.X):
                x_ = np.concatenate([m, x.iloc[:, [x_idx]]], axis=1)

                # Fit full model
                lm_full = self.model_regressor().fit(x_, y)
                lm_full_ll = LMModels.log_likelihood(y, lm_full.predict(x_))

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
                    )
                )

            y_lms = pd.DataFrame(y_lms)
            y_lms["fdr"] = multipletests(y_lms["pval"], method="fdr_bh")[1]

            lms.append(y_lms)
        lms = pd.concat(lms, ignore_index=True).sort_values("pval")
        return lms


# Data-sets
#
prot_obj, gexp_obj, crispr_obj = Proteomics(), GeneExpression(), CRISPR()

# Samples
#
samples = set.intersection(
    set(prot_obj.get_data()),
    set(gexp_obj.get_data()),
    set(crispr_obj.get_data(dtype="merged")),
)
LOG.info(f"Samples: {len(samples)}")

# Filter data-sets
#
prot = prot_obj.filter(subset=samples, perc_measures=0.03)
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp_obj.filter(subset=samples)
LOG.info(f"Transcriptomics: {gexp.shape}")

crispr = crispr_obj.filter(dtype="merged", subset=samples, abs_thres=0.5, min_events=5)
LOG.info(f"CRISPR: {crispr.shape}")

# Genes
#

genes = natsorted(list(set.intersection(set(prot.index), set(gexp.index))))
LOG.info(f"Genes: {len(genes)}")

# Covariates
#

covariates = LMModels.define_covariates(
    institute=crispr_obj.merged_institute, cancertype=False, mburden=False, ploidy=False
)

# Protein ~ CRISPR LMMs
#
prot_lm = LModel(
    Y=prot[samples].T, X=crispr[samples].T, M=covariates.loc[samples]
).fit_matrix()

# Gene-expression ~ CRISPR LMMs
#
gexp_lm = LModel(
    Y=gexp[samples].T, X=crispr[samples].T, M=covariates.loc[samples]
).fit_matrix()


#
#

c, p = "ITGA3", "VPS4A"

plot_df = pd.concat(
    [
        crispr.loc[[c]].T.add_suffix("_crispr"),
        prot.loc[[p]].T.add_suffix("_prot"),
        prot_obj.ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

grid = GIPlot.gi_regression(f"{p}_prot", f"{c}_crispr", plot_df)
grid.set_axis_labels(f"{p}\nProtein intensities", f"{c}\nCRISPR log FC")
plt.savefig(
    f"{RPATH}/1.LM_{p}_{c}_regression.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")

ax = GIPlot.gi_tissue_plot(f"{p}_prot", f"{c}_crispr", plot_df)
ax.set_xlabel(f"{p}\nProtein intensities")
ax.set_ylabel(f"{c}\nCRISPR log FC")
plt.savefig(
    f"{RPATH}/1.LM_{p}_{c}_regression_tissue_plot.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")
