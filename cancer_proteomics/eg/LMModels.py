#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from crispy.Utils import Utils
from scipy.stats import spearmanr, chi2
from statsmodels.stats.multitest import multipletests


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("reports", "eg/")


class LMModels:
    """"
    Class to perform the linear regression models
    """ ""

    RES_ORDER = [
        "y_id",
        "x_id",
        "beta",
        "beta_se",
        "pval",
        "fdr",
        "nsamples",
        "ncovariates",
    ]

    def __init__(
        self,
        y,
        x,
        k=None,
        m=None,
        m2=None,
        x_feature_type="all",
        m2_feature_type="same_y",
        add_intercept=True,
        lik="normal",
        transform_y="scale",
        transform_x="scale",
        transform_m2="scale",
        x_min_events=None,
        institute=True,
        verbose=1,
    ):
        # Misc
        self.verbose = verbose
        self.x_feature_type = x_feature_type
        self.m2_feature_type = m2_feature_type
        self.add_intercept = add_intercept

        # LIMIX parameters
        self.lik = lik

        # Preprocessing steps
        self.transform_y = transform_y
        self.transform_x = transform_x
        self.transform_m2 = transform_m2
        self.x_min_events = x_min_events

        # Build random effects and covariates matrices
        self.k = self.kinship(x) if k is None else k.copy()
        self.m = self.define_covariates(institute=institute) if m is None else m.copy()

        # Samples overlap
        self.samples = list(
            set.intersection(
                set(y.index),
                set(x.index),
                set(self.m.index),
                set(self.k.index),
                set(y.index) if m2 is None else set(m2.index),
            )
        )
        LOG.info(f"Samples: {len(self.samples)}")

        # Y matrix
        self.y, self.y_columns = self.__build_y(y.copy())

        # X matrix
        self.x, self.x_columns = self.__build_x(x.copy())

        # Covariates
        self.m = self.m.loc[self.samples, self.m.std() > 0]
        self.m, self.m_columns = self.m.values, np.array(list(self.m.columns))

        # Random effects matrix
        self.k = self.k.loc[self.samples, self.samples].values

        LOG.info(
            f"Y: {self.y.shape[1]}; X: {self.x.shape[1]}; M: {self.m.shape[1]}; K: {self.k.shape[1]}"
        )

        # Second covariates matrix
        if m2 is not None:
            self.m2, self.m2_columns = self.__build_m2(m2.copy())
            LOG.info(f"M2: {self.m2.shape[1]}")

        else:
            self.m2, self.m2_columns = None, None

    def __build_y(self, y):
        """
        Method to build the y matrix.

        :param y:
        :return:
        """
        y_ = self.transform_matrix(y.loc[self.samples], t_type=self.transform_y)
        return y_.values, np.array(list(y_.columns))

    def __build_m2(self, m2):
        """
        Method to build the m2 matrix.
        :param m2:
        :return:
        """
        m2_ = self.transform_matrix(m2.loc[self.samples], t_type=self.transform_m2)
        return m2_.values, np.array(list(m2_.columns))

    def __build_x(self, x):
        """
        Method to build the x matrix.
        :param x:
        :return:
        """
        x_ = x.loc[self.samples, x.std() > 0]

        if self.x_min_events is not None:
            x_ = x_.loc[:, x_.sum() >= self.x_min_events]
        else:
            x_ = self.transform_matrix(x_, t_type=self.transform_x)

        return x_.values, np.array(list(x_.columns))

    def __prepare_inputs__(self, y_var):
        # Define samples with NaNs
        y_idx = list(self.y_columns).index(y_var)
        y_nans_idx = np.isnan(self.y[:, y_idx])

        if self.verbose > 0:
            LOG.info(f"y_id: {y_var} ({y_idx}); N samples: {sum(1 - y_nans_idx)}")

        # Remove NaNs from y
        y_ = self.y[y_nans_idx == 0][:, [y_idx]]

        # Subset X
        x_ = self.x[y_nans_idx == 0]

        if self.x_feature_type == "drop_y":
            if y_var not in self.x_columns:
                LOG.warning(f"[x_feature_type=drop_y] Y feature {y_idx} not in X")

            x_ = x_[:, self.x_columns != y_var]
            x_vars = self.x_columns[self.x_columns != y_var]

        elif self.x_feature_type == "same_y":
            if y_var not in self.x_columns:
                LOG.error(f"[x_feature_type=same_y] Y feature {y_idx} not in X")

            x_ = x_[:, self.x_columns == y_var]
            x_vars = self.x_columns[self.x_columns == y_var]

        else:
            x_vars = self.x_columns[np.std(x_, axis=0) > 0]
            x_ = x_[:, np.std(x_, axis=0) > 0]

        # Subset m
        m_ = self.m[y_nans_idx == 0]
        m_ = m_[:, np.std(m_, axis=0) > 0]

        if (self.m2 is not None) and (self.m2_feature_type == "same_y"):
            m_ = np.append(
                m_, self.m2[y_nans_idx == 0][:, self.m2_columns == y_var], axis=1
            )

        if self.add_intercept:
            m_ = np.insert(m_, m_.shape[1], values=1, axis=1)

        # Subset random effects matrix
        k_ = self.k[:, y_nans_idx == 0][y_nans_idx == 0, :]

        return y_, y_nans_idx, x_, x_vars, m_, k_

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

    def rfr(self, y_var, top_n_features=10):
        """
        Non-linear regression with RandomForestRegression, using measurements of the y matrix for the variable
        specified by y_var.

        :param y_var: String y variable name
        :param n_estimators:
        :param max_features:
        :return: pandas.DataFrame of the associations
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import ShuffleSplit, GridSearchCV

        # Assemble matrices
        y_, y_nans_idx, x_, x_vars, m_, k_ = self.__prepare_inputs__(y_var)

        # Rank features based on Spearman R
        feature_rank = [spearmanr(y_, i) for i in x_.T]
        feature_rank_idx = sorted(
            range(len(feature_rank)),
            key=lambda k: abs(feature_rank[k][0]),
            reverse=True,
        )

        # Pick top features
        x_new_ = x_[:, feature_rank_idx[:top_n_features]]

        # Full model: top features + covariates + [random effects]
        x_new_full_ = np.concatenate((m_, x_new_), axis=1)

        features_pr = []
        cv = ShuffleSplit(test_size=0.3, n_splits=10)
        for idx_train, idx_test in cv.split(x_new_full_, y_[:, 0]):
            idx_rf = RandomForestRegressor(n_estimators=100, min_samples_split=5, max_depth=5, max_features=None)
            idx_rf = idx_rf.fit(x_new_full_[idx_train], y_[idx_train, 0])

            train_r2 = idx_rf.score(x_new_full_[idx_train], y_[idx_train, 0])
            test_r2 = idx_rf.score(x_new_full_[idx_test], y_[idx_test, 0])
            LOG.info(f"R2: train={train_r2:.2f}; test={test_r2:.2f}")

            idx_rf = permutation_importance(idx_rf, x_new_full_[idx_test], y_[idx_test, 0])
            idx_pr_mean = idx_rf.importances_mean[-top_n_features:]
            features_pr.append(idx_pr_mean)

        res = pd.DataFrame(
            dict(
                y_id=y_var,
                x_id=x_vars[feature_rank_idx[:top_n_features]],
                permutation_importance=np.median(features_pr, axis=0),
                nsamples=sum(1 - y_nans_idx),
                ncovariates=m_.shape[1],
            )
        ).sort_values("permutation_importance", ascending=False)
        print(res)

        return res

    def svm(self, y_var):
        from sklearn.svm import SVR
        from sklearn.model_selection import ShuffleSplit, GridSearchCV

        # Assemble matrices
        y_, y_nans_idx, x_, x_vars, m_, k_ = self.__prepare_inputs__(y_var)

        r2_train, r2_test = [], []
        for f_idx in range(x_.shape[1]):
            f_r2_train, f_r2_test = [], []

            cv = ShuffleSplit(test_size=0.3, n_splits=10)
            for f_train, f_test in cv.split(x_, y_[:, 0]):
                f_svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
                f_svr = f_svr.fit(x_[f_train][:, [f_idx]], y_[f_train, 0])

                cv_r2_train = f_svr.score(x_[f_train][:, [f_idx]], y_[f_train, 0])
                cv_r2_test = f_svr.score(x_[f_test][:, [f_idx]], y_[f_test, 0])

                f_r2_train.append(cv_r2_train)
                f_r2_test.append(cv_r2_test)

            f_r2_train, f_r2_test = np.median(f_r2_train), np.median(f_r2_test)
            LOG.info(f"R2: train={f_r2_train:.2f}; test={f_r2_test:.2f}")

            r2_train.append(f_r2_train)
            r2_test.append(f_r2_test)

        res = pd.DataFrame(
            dict(
                y_id=y_var,
                x_id=x_vars,
                r2_test=r2_test,
                r2_train=r2_train,
                nsamples=sum(1 - y_nans_idx),
                ncovariates=m_.shape[1],
            )
        ).sort_values("r2_test", ascending=False)
        print(res.head(60))

        x_idx = 343
        plot_df = pd.DataFrame(dict(
            y=y_[:, 0],
            x=x_[:, x_idx],
        ))

        param_grid = dict(
            kernel=["rbf", "poly"],
            degree=[1, 2, 3],
            C=[5, 10, 100],
            epsilon=[.1, .5, 1],
        )
        rgr = GridSearchCV(SVR(), param_grid, cv=ShuffleSplit(test_size=0.3, n_splits=10))
        rgr = rgr.fit(plot_df[["x"]], plot_df["y"])
        LOG.info(rgr.best_params_)
        LOG.info(rgr.best_score_)

        grid = GIPlot.gi_regression("x", "y", plot_df, plot_reg=False)
        plot_df = plot_df.sort_values("x")
        grid.ax_joint.plot(plot_df["x"], rgr.best_estimator_.predict(plot_df[["x"]]), color=GIPlot.PAL_DTRACE[1], lw=2)
        plt.savefig(
            f"{RPATH}/1.SVM_SL_test_regression.pdf", bbox_inches="tight", transparent=True
        )
        plt.close("all")

    def lmm(self, y_var):
        """
        Linear regression method, using measurements of the y matrix for the variable specified by y_var.

        :param y_var: String y variable name
        :return: pandas.DataFrame of the associations
        """
        import limix

        y_, y_nans_idx, x_, x_vars, m_, k_ = self.__prepare_inputs__(y_var)

        # Linear Mixed Model
        lmm = limix.qtl.scan(G=x_, Y=y_, K=k_, M=m_, lik=self.lik, verbose=False)

        # Build results
        lmm_betas = lmm.effsizes["h2"].query("effect_type == 'candidate'")
        lmm = pd.DataFrame(
            dict(
                y_id=y_var,
                x_id=x_vars,
                beta=list(lmm_betas["effsize"].round(5)),
                beta_se=list(lmm_betas["effsize_se"].round(5)),
                pval=list(lmm.stats.loc[lmm_betas["test"], "pv20"]),
                nsamples=sum(1 - y_nans_idx),
                ncovariates=m_.shape[1],
            )
        )

        return lmm

    def matrix_lmm(self, pval_adj="fdr_bh", pval_adj_overall=False):
        # Iterate through Y variables
        res = []

        for y_var in self.y_columns:
            res.append(self.lmm(y_var=y_var))

        res = pd.concat(res, ignore_index=True)

        # Multiple p-value correction
        if pval_adj_overall:
            res = res.assign(fdr=multipletests(res["pval"], method=pval_adj)[1])

        else:
            res = self.multipletests(res, field="pval", pval_method=pval_adj)

        return res.sort_values("fdr")[self.RES_ORDER]

    def write_lmm(self, output_folder):
        for i in self.y_columns:
            self.lmm(y_var=i).to_csv(
                f"{output_folder}/{i}.csv.gz", index=False, compression="gzip"
            )

    @staticmethod
    def multipletests(
        parsed_results, pval_method="fdr_bh", field="pval", idx_cols=None
    ):
        idx_cols = ["y_id"] if idx_cols is None else idx_cols

        parsed_results_adj = []

        for idx, df in parsed_results.groupby(idx_cols):
            df = df.assign(fdr=multipletests(df[field], method=pval_method)[1])
            parsed_results_adj.append(df)

        parsed_results_adj = pd.concat(parsed_results_adj, ignore_index=True)

        return parsed_results_adj

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

    @staticmethod
    def define_covariates(
        std_filter=True,
        medium=True,
        cancertype=True,
        mburden=True,
        ploidy=True,
        institute=True,
    ):
        from crispy.DataImporter import Sample

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
    def kinship(k, decimal_places=5):
        K = k.dot(k.T)
        K /= K.values.diagonal().mean()
        return K.round(decimal_places)
