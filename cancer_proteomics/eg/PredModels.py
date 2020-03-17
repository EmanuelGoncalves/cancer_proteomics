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

import sys

sys.path.extend(
    [
        "/Users/eg14/Projects/cancer_proteomics",
        "/Users/eg14/Projects/cancer_proteomics/cancer_proteomics",
        "/Users/eg14/Projects/cancer_proteomics/drug_response",
    ]
)

import os
import logging
import argparse
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict
from scipy.cluster import hierarchy
from natsort import natsorted
from crispy.GIPlot import GIPlot
from natsort import natsorted
from itertools import zip_longest
from sklearn.mixture import GaussianMixture
from cancer_proteomics.eg.LMModels import LMModels
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR, Sample, Mobem


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


class PredModels:
    PREDICTORS = ["SVM", "RF", "EN"]

    def __init__(
        self, y, x, transform_y="scale", transform_x="scale", x_min_events=None
    ):
        # Matrices
        self.y = y.copy()
        LOG.info(f"Y matrix: {self.y.shape}")

        self.x = x.copy()
        LOG.info(f"X matrix: {self.x.shape}")

        # Preprocessing steps
        self.transform_y = transform_y
        self.transform_x = transform_x
        self.x_min_events = x_min_events

        # Samples overlap
        self.samples = list(set.intersection(set(self.y.index), set(self.x.index)))
        LOG.info(f"Overlapping samples: {len(self.samples)}")

    @staticmethod
    def transform_matrix(matrix, t_type="scale"):
        if t_type == "scale":
            from sklearn.preprocessing import StandardScaler

            matrix = pd.DataFrame(
                StandardScaler().fit_transform(matrix),
                index=matrix.index,
                columns=matrix.columns,
            )

        elif t_type == "rank":
            matrix = matrix.rank(axis=1).values

        else:
            LOG.warning(
                f"{t_type} transformation not supported. Original matrix returned."
            )

        return matrix

    def build_y(self, y):
        y_ = self.transform_matrix(y.loc[self.samples], t_type=self.transform_y)
        return y_

    def build_x(self, x):
        x_ = x.loc[self.samples, x.std() > 0]

        if self.x_min_events is not None:
            x_ = x_.loc[:, x_.sum() >= self.x_min_events]

        else:
            x_ = self.transform_matrix(x_, t_type=self.transform_x)

        return x_

    def prepare_inputs(self, y_var):
        # Remove NaNs from y
        y = self.build_y(self.y[[y_var]]).dropna().iloc[:, 0]

        # Subset X
        X = self.build_x(self.x).loc[y.index]
        X = X.loc[:, np.std(X, axis=0) > 0]

        return X, y

    @staticmethod
    def feature_selection_top_spearman(X, y, q):
        feature_rank = pd.Series({f: spearmanr(y, c)[0] for f, c in X.iteritems()})
        q_thres = np.quantile(feature_rank.abs(), q)
        feature_selected = feature_rank[feature_rank.abs() >= q_thres]
        return feature_selected

    @staticmethod
    def get_predictor(predictor_type):
        if predictor_type == "SVM":
            predictor = SVR(kernel="rbf", C=1, epsilon=0.1)

        elif predictor_type == "RF":
            predictor = RandomForestRegressor(
                n_estimators=100, min_samples_split=5, max_depth=5, max_features=None
            )

        elif predictor_type == "EN":
            predictor = ElasticNet(fit_intercept=True, alpha=0.1)

        else:
            assert False, f"Predictor type {predictor_type} not supported"

        return predictor

    def train(self, y_var, features_percentile, n_repeats, cv):
        # Assemble matrices
        X, y = self.prepare_inputs(y_var)

        # Feature selection
        f_selected = self.feature_selection_top_spearman(X, y, q=features_percentile)
        X = X[f_selected.index]

        # Regressor cross-validation
        predictors_stats = {
            ptype: dict(rsquared=[], fvalue=[]) for ptype in self.PREDICTORS
        }

        for train, test in cv.split(X, y):
            for ptype in self.PREDICTORS:
                predictor = self.get_predictor(ptype)

                predictor = predictor.fit(X.iloc[train], y.iloc[train])

                cv_rsquared = predictor.score(X.iloc[test], y.iloc[test])
                predictors_stats[ptype]["rsquared"].append(cv_rsquared)

                cv_fvalue = permutation_importance(predictor, X, y, n_repeats=n_repeats)
                cv_fvalue = np.median(cv_fvalue["importances"], 1)
                predictors_stats[ptype]["fvalue"].append(cv_fvalue)

        predictors_stats = {
            ptype: dict(
                rsquared=np.median(val["rsquared"]), fvalue=np.median(val["fvalue"], 0)
            )
            for ptype, val in predictors_stats.items()
        }

        LOG.info(
            f"{y_var}: "
            + "; ".join(
                [f"{p}={v['rsquared']:.2f}" for p, v in predictors_stats.items()]
            )
        )

        # Result
        res = pd.DataFrame(
            dict(
                y_id=y_var,
                x_id=X.columns,
                x_spearman=f_selected[X.columns],
                x_nfeatures=X.shape[1],
                nsamples=y.shape[0],
            )
        ).reset_index(drop=True)

        for ptype, val in predictors_stats.items():
            res[f"x_rsquared_{ptype}"] = val["fvalue"]
            res[f"rsquared_{ptype}"] = val["rsquared"]

        return res

    def train_matrix(self, features_percentile=0.99, n_repeats=5, cv=None):
        cv = ShuffleSplit(test_size=0.3, n_splits=5) if cv is None else cv

        res = [
            self.train(
                y_var,
                features_percentile=features_percentile,
                n_repeats=n_repeats,
                cv=cv,
            )
            for y_var in self.y.columns
        ]
        res = pd.concat(res, ignore_index=True)
        return res


if __name__ == "__main__":
    # Data-sets
    #
    prot, crispr = Proteomics(), CRISPR()

    # Samples
    #
    ss = Sample().samplesheet
    samples = set.intersection(set(prot.get_data()), set(crispr.get_data()))
    LOG.info(f"Samples: {len(samples)}")

    # Filter data-sets
    #
    prot = prot.filter(subset=samples, perc_measures=0.05)
    LOG.info(f"Proteomics: {prot.shape}")

    crispr = crispr.filter(subset=samples, abs_thres=0.5, min_events=5)
    LOG.info(f"CRISPR: {crispr.shape}")

    #
    #
    pmodels = PredModels(prot.T, crispr.T)

    gi = pmodels.train_matrix(features_percentile=0.99, n_repeats=3, cv=ShuffleSplit(test_size=0.3, n_splits=3))

    gi.to_csv(f"{RPATH}/pred_models.csv.gz", index=False, compression="gzip")
