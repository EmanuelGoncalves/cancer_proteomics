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
        "/Users/eg14/Projects/crispy",
        "/Users/eg14/Projects/crispy/crispy",
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
from scipy.stats import spearmanr, chi2
from collections import defaultdict
from scipy.cluster import hierarchy
from natsort import natsorted
from crispy.GIPlot import GIPlot
from natsort import natsorted
from itertools import zip_longest
from sklearn.mixture import GaussianMixture
from cancer_proteomics.eg.LMModels import LMModels
from sklearn.svm import SVR
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR, Sample, Mobem, PPI


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

        # Import PPI
        self.ppi = PPI().build_string_ppi(score_thres=900)

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
    def log_likelihood(y_true, y_pred):
        n = len(y_true)
        ssr = np.power(y_true - y_pred, 2).sum()
        var = ssr / n

        l = np.longfloat(1 / (np.sqrt(2 * np.pi * var))) ** n * np.exp(
            -(np.power(y_true - y_pred, 2) / (2 * var)).sum()
        )
        ln_l = np.log(l)

        return float(ln_l)

    @staticmethod
    def define_covariates(
        std_filter=True,
        medium=True,
        cancertype=True,
        mburden=True,
        ploidy=True,
        institute=True,
    ):
        # Imports
        samplesheet = Sample().samplesheet

        # Covariates
        covariates = []

        # CRISPR institute of origin
        if institute:
            covariates.append(pd.get_dummies(samplesheet["institute"]).astype(int))

        # Cell lines culture conditions
        if medium:
            culture = pd.get_dummies(samplesheet["growth_properties"]).drop(
                columns=["Unknown"]
            )
            covariates.append(culture)

        # Cancer type
        if cancertype:
            ctype = pd.get_dummies(samplesheet["cancer_type"])
            covariates.append(ctype)

        # Mutation burden
        if mburden:
            m_burdern = samplesheet["mutational_burden"]
            covariates.append(m_burdern)

        # Ploidy
        if ploidy:
            ploidy = samplesheet["ploidy"]
            covariates.append(ploidy)

        # Merge covariates
        covariates = pd.concat(covariates, axis=1, sort=False)

        # Remove covariates with zero standard deviation
        if std_filter:
            covariates = covariates.loc[:, covariates.std() > 0]

        return covariates.dropna()

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

    def train(self, y_id, features_percentile, cv, ):
        # Assemble matrices
        X, y = self.prepare_inputs(y_id)

        # Feature selection
        f_selected = self.feature_selection_top_spearman(X, y, q=features_percentile)
        X = X[f_selected.index]

        # Regressor cross-validation
        p_rsquared = {ptype: [] for ptype in self.PREDICTORS}

        for train, test in cv.split(X, y):
            cv_predictors = dict()

            for ptype in self.PREDICTORS:
                predictor = self.get_predictor(ptype).fit(X.iloc[train], y.iloc[train])

                cv_rsquared = predictor.score(X.iloc[test], y.iloc[test])

                p_rsquared[ptype].append(cv_rsquared)
                cv_predictors[ptype] = predictor

        p_rsquared = pd.DataFrame(p_rsquared).median()
        LOG.info(
            f"{y_id}:"
            + p_rsquared.to_string(float_format=lambda v: f"{v:.2}").replace("\n", "\t")
        )

        # # No feature importance analysis for lowly predictable protein
        # if sum(p_rsquared > 0.3) < 2:
        #     LOG.info(f"{y_id} skipped")
        #     return None
        #
        # # Feature importance
        # p_fvalue = {ptype: [] for ptype in self.PREDICTORS}
        #
        # for train, test in cv.split(X, y):
        #     for ptype in self.PREDICTORS:
        #         f_score = []
        #         for f in X:
        #             f_predictor = self.get_predictor(ptype).fit(
        #                 X[[f]].iloc[train], y.iloc[train]
        #             )
        #             f_score.append(f_predictor.score(X[[f]].iloc[test], y.iloc[test]))
        #
        #         p_fvalue[ptype].append(f_score)
        #
        # p_fvalue = pd.DataFrame(
        #     {ptype: pd.DataFrame(val).median() for ptype, val in p_fvalue.items()}
        # )
        # p_varexp = p_fvalue.divide(p_rsquared)

        # Result
        res = pd.DataFrame(
            dict(
                y_id=y_id,
                x_id=X.columns,
                x_spearman=f_selected[X.columns],
                x_nfeatures=X.shape[1],
                nsamples=y.shape[0],
            )
        ).reset_index(drop=True)

        for ptype in self.PREDICTORS:
            # res[f"x_rsquared_{ptype}"] = p_fvalue[ptype]
            # res[f"x_varexp_{ptype}"] = p_varexp[ptype]
            res[f"rsquared_{ptype}"] = p_rsquared[ptype]

        return res

    def feature_importance(self, y_id, x_ids=None):
        # Assemble matrices
        X, y = self.prepare_inputs(y_id)

        if x_ids is None:
            x_ids = list(X)

        # Covariates
        m = self.define_covariates(cancertype=False).reindex(y.index).dropna()

        # Align matrices
        X, y = X.loc[m.index], y[m.index]

        # Smaller model - only covariates
        lm_s = LinearRegression().fit(m, y)
        lm_s_ll = self.log_likelihood(y, lm_s.predict(m))

        # Logratio test with full model - covariates + feature
        res = []
        for x_id in x_ids:
            x_var_x = np.concatenate((m, X[[x_id]]), axis=1)
            lm_f = LinearRegression().fit(x_var_x, y)
            lm_f_ll = self.log_likelihood(y, lm_f.predict(x_var_x))

            lr = 2 * (lm_f_ll - lm_s_ll)
            lr_pval = chi2(1).sf(lr)

            x_var_res = dict(
                y_id=y_id,
                x_coef=lm_f.coef_[-1],
                x_id=x_id,
                x_lr=lr,
                x_pval=lr_pval,
            )
            res.append(x_var_res)

        res = pd.DataFrame(res)
        res = res.assign(x_fdr=multipletests(res["x_pval"], method="fdr_bh")[1])
        res = PPI.ppi_annotation(res, self.ppi).sort_values("x_fdr").head(60)
        return res

    def train_matrix(self, features_percentile=0.99, cv=None):
        cv = ShuffleSplit(test_size=0.3, n_splits=5) if cv is None else cv

        res = []

        for y_var in self.y.columns:
            y_var_res = self.train(
                y_var, features_percentile=features_percentile, cv=cv
            )

            if y_var_res is None:
                continue

            res.append(y_var_res)

        res = pd.concat(res, ignore_index=True)

        # Annotate individual features
        res_annot = pd.concat([self.feature_importance(g, x_ids=set(df["x_id"])) for g, df in res.groupby("y_id")])
        res = pd.concat([res.set_index(["y_id", "x_id"]), res_annot.set_index(["y_id", "x_id"])], axis=1).reset_index()

        return res


if __name__ == "__main__":
    METHOD_PAL = pd.Series(
        ["#1f77b4", "#2ca02c", "#d62728"], index=PredModels.PREDICTORS
    )

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

    # Genetic interactions
    #
    pmodels = PredModels(prot.T, crispr.T)

    gi = pmodels.train_matrix(
        features_percentile=0.99, cv=KFold(n_splits=10)
    )
    gi.to_csv(f"{RPATH}/pred_models.csv.gz", index=False, compression="gzip")
    # gi = pd.read_csv(f"{RPATH}/pred_models.csv.gz")

    #
    #
    plot_df = pd.DataFrame({p: gi[f"rsquared_{p}"] for p in pmodels.PREDICTORS})

    fig, ax = plt.subplots(1, 1, figsize=(1.5, 2), dpi=600)

    sns.boxenplot(
        data=plot_df, linewidth=0.1, palette=METHOD_PAL.to_dict(), saturation=1, ax=ax
    )

    ax.set_ylabel("Pearson's R")
    ax.grid(True, axis="y", ls="-", lw=0.1, alpha=1.0, zorder=0)
    plt.savefig(
        f"{RPATH}/1.Pred_ptype_rsquared_violin.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    #
    #
    g = sns.clustermap(
        plot_df.corr(),
        cmap="viridis",
        figsize=(2, 2),
        annot=True,
        fmt=".2f",
        linewidth=0.3,
    )
    plt.savefig(
        f"{RPATH}/1.Pred_ptype_rsquared_clustermap.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    #
    #
    plot_df = gi.groupby("y_id")[[f"rsquared_{p}" for p in pmodels.PREDICTORS]].median()
    plot_df = plot_df.assign(rsquared_mean=plot_df.mean(1))

    _, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=600)
    sns.distplot(
        plot_df["rsquared_mean"],
        kde=False,
        hist_kws=dict(linewidth=0, alpha=1, color=GIPlot.PAL_DTRACE[2]),
        ax=ax,
    )
    ax.set_xlabel(f"Mean Pearson's R")
    ax.set_ylabel(f"Number of proteins")
    ax.grid(True, axis="y", ls="-", lw=0.1, alpha=1.0, zorder=0)
    plt.savefig(
        f"{RPATH}/1.Pred_ptype_rsquared_mean_displot.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    #
    #
    plot_df = gi[[f"rsquared_{p}" for p in pmodels.PREDICTORS]]
    plot_df = (
        pd.concat(
            [
                gi,
                plot_df.mean(1).rename("mean_rsquared"),
                plot_df.std(1).rename("std_rsquared"),
            ],
            axis=1,
        )
        .groupby("y_id")
        .median()
    )
    plot_df = plot_df.loc[:, [not c.startswith("x_") for c in plot_df]]

    s_transform = MinMaxScaler(feature_range=[1, 10])

    _, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)
    g = ax.scatter(
        plot_df["mean_rsquared"],
        plot_df["std_rsquared"],
        s=s_transform.fit_transform(plot_df[["nsamples"]]),
        edgecolor="white",
        lw=0.1,
        color=GIPlot.PAL_DTRACE[2],
    )
    sns.regplot(
        "mean_rsquared",
        "std_rsquared",
        data=plot_df,
        lowess=True,
        truncate=True,
        scatter=False,
        line_kws=dict(lw=1.0, color=GIPlot.PAL_DTRACE[1]),
        ax=ax,
    )
    plt.legend(
        *g.legend_elements(
            "sizes",
            num=6,
            func=lambda x: s_transform.inverse_transform(np.array(x).reshape(-1, 1))[
                :, 0
            ],
        ),
        frameon=False,
        prop={"size": 5},
        title="# Samples",
    ).get_title().set_fontsize("5")
    ax.set_xlabel("Mean R-squared")
    ax.set_ylabel("Std R-squared")
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
    plt.savefig(
        f"{RPATH}/1.Pred_ptype_scatter.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")

    #
    #
    plot_df = gi[[f"rsquared_{p}" for p in pmodels.PREDICTORS]]
    gi_top = pd.concat([gi, plot_df.mean(1).rename("mean_rsquared"), plot_df.std(1).rename("std_rsquared")], axis=1)
    gi_top = gi_top.query("(std_rsquared < 0.05) & (nsamples > 160)")
    gi_top = gi_top.sort_values("x_fdr")

    #
    #
    y_id, x_id = "VIM", "FERMT2"

    plot_df = pd.concat(
        [
            prot.loc[[y_id]].T.add_suffix("_prot"),
            crispr.loc[[x_id]].T.add_suffix("_crispr"),
            ss["tissue"],
        ],
        axis=1,
        sort=False,
    ).dropna()

    grid = GIPlot.gi_regression(f"{x_id}_crispr", f"{y_id}_prot", plot_df, lowess=True)
    plt.savefig(
        f"{RPATH}/1.Pred_{y_id}_{x_id}_regression.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    ax = GIPlot.gi_tissue_plot(f"{x_id}_crispr", f"{y_id}_prot", plot_df)
    plt.savefig(
        f"{RPATH}/1.Pred_{y_id}_{x_id}_regression_tissue_plot.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
