"""Model zoo and honest validation for small materials datasets.

Key lessons encoded here:
 1. Split by ALLOY (GroupKFold), not by row. Several tests of the same alloy are highly
    correlated; a random split lets the model "recognise" the alloy and inflates scores.
 2. Test EXTRAPOLATION explicitly (e.g. train <= 950 C, predict > 950 C) - this is how the
    model will be used (predicting long lives / new conditions).
 3. Report UNCERTAINTY. Gaussian processes give a predictive std; for any model,
    split-conformal prediction gives distribution-free intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import LarsonMillerBaseline


class HybridLMP(BaseEstimator, RegressorMixin):
    """Physics baseline (Larson-Miller master curve) + ML model on the residual.

    X must be a DataFrame containing T_C and stress_MPa plus the ML feature columns.
    """

    def __init__(self, ml_model=None, columns=None):
        self.ml_model = ml_model
        self.columns = columns

    def fit(self, X, y):
        self.baseline_ = LarsonMillerBaseline().fit(X, y)
        resid = np.asarray(y) - self.baseline_.predict(X)
        self.ml_ = clone(self.ml_model).fit(X[self.columns], resid)
        return self

    def predict(self, X):
        return self.baseline_.predict(X) + self.ml_.predict(X[self.columns])


class LMPTarget(BaseEstimator, RegressorMixin):
    """Learn the Larson-Miller parameter instead of log life (physics-informed target).

        LMP = T[K] * (C + log10 t_r) / 1000   ->   log10 t_r = 1000 LMP / T - C

    The temperature dependence of life is then built in exactly, and the ML model only
    has to learn how LMP varies with stress and composition - a much smoother, mostly
    additive function that tree ensembles handle well.
    """

    def __init__(self, ml_model=None, columns=None, C: float = 20.0):
        self.ml_model = ml_model
        self.columns = columns
        self.C = C

    def fit(self, X, y):
        T = X["T_C"].to_numpy() + 273.15
        lmp = T * (self.C + np.asarray(y)) / 1000.0
        self.ml_ = clone(self.ml_model).fit(X[self.columns], lmp)
        return self

    def predict(self, X):
        T = X["T_C"].to_numpy() + 273.15
        return 1000.0 * self.ml_.predict(X[self.columns]) / T - self.C


class BaselineOnly(BaseEstimator, RegressorMixin):
    """Composition-blind Larson-Miller master curve (the classical engineering approach)."""

    def fit(self, X, y):
        self.baseline_ = LarsonMillerBaseline().fit(X, y)
        return self

    def predict(self, X):
        return self.baseline_.predict(X)


def _lbfgs_capped(obj_func, initial_theta, bounds, maxiter: int = 60):
    """Hyper-parameter optimiser with an iteration cap (ARD kernels on ~1000 points are slow)."""
    from scipy.optimize import minimize
    res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": maxiter})
    return res.x, res.fun


class SubsampledGP(GaussianProcessRegressor):
    """GaussianProcessRegressor that optimises its kernel on at most `max_n` random rows.

    Exact GPs scale as O(n^3) in time and O(n^2 d) in memory for ARD gradients; for
    ~1000 points x 25 features that dominates the run time. Hyper-parameters are learned
    on a subsample, then the GP is conditioned on ALL training data with that kernel fixed.
    """

    def __init__(self, kernel=None, *, alpha=1e-10, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, n_targets=None, random_state=None, max_n=500):
        super().__init__(kernel=kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
                         normalize_y=normalize_y, copy_X_train=copy_X_train, n_targets=n_targets,
                         random_state=random_state)
        self.max_n = max_n

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) > self.max_n:
            idx = np.random.default_rng(0).choice(len(X), self.max_n, replace=False)
            super().fit(X[idx], y[idx])
            fixed = GaussianProcessRegressor(kernel=self.kernel_, optimizer=None, normalize_y=self.normalize_y,
                                             alpha=self.alpha).fit(X, y)
            for attr in ("X_train_", "y_train_", "L_", "alpha_", "_y_train_mean", "_y_train_std", "kernel_"):
                setattr(self, attr, getattr(fixed, attr))
            return self
        return super().fit(X, y)


def gp_model(n_features: int):
    """GP with an anisotropic (ARD) Matern-5/2 kernel: one length scale per feature, so the
    fitted length scales also act as a feature-relevance measure."""
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_features), nu=2.5) + WhiteKernel(0.02)
    return make_pipeline(StandardScaler(), SubsampledGP(kernel=kernel, normalize_y=True, optimizer=_lbfgs_capped,
                                                        random_state=0, max_n=500))


def model_zoo(columns: list[str]):
    """Dictionary name -> estimator. Estimators take a DataFrame (column selection inside)."""
    sel = ColumnSelector(columns)
    return {
        "Larson-Miller (no composition)": BaselineOnly(),
        "Ridge (linear)": make_pipeline(sel, StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "Random forest": make_pipeline(sel, RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0)),
        "Gradient boosting": make_pipeline(sel, HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                                             max_leaf_nodes=15, l2_regularization=1.0,
                                                                             random_state=0)),
        "Neural network (MLP)": make_pipeline(sel, StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                                                  max_iter=3000, early_stopping=True,
                                                                                  random_state=0)),
        "Gaussian process": make_pipeline(sel, gp_model(len(columns))),
        "Hybrid: LMP + gradient boosting": HybridLMP(
            HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=0), columns),
        "Gradient boosting on LMP target": LMPTarget(
            HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=0), columns),
    }


class ColumnSelector(BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].to_numpy(float)


def metrics(y, yhat) -> dict:
    return {"R2": r2_score(y, yhat), "RMSE_log10": float(np.sqrt(mean_squared_error(y, yhat))),
            "MAE_log10": mean_absolute_error(y, yhat),
            "within_factor_2": float(np.mean(np.abs(np.asarray(y) - yhat) <= np.log10(2)))}


def cross_validate(model, df: pd.DataFrame, y, groups=None, n_splits: int = 5, seed: int = 0):
    """Out-of-fold predictions with GroupKFold (groups given) or shuffled KFold."""
    y = np.asarray(y)
    oof = np.zeros_like(y, dtype=float)
    splitter = GroupKFold(n_splits=n_splits) if groups is not None else KFold(n_splits, shuffle=True, random_state=seed)
    for tr, te in splitter.split(df, y, groups):
        m = clone(model).fit(df.iloc[tr], y[tr])
        oof[te] = m.predict(df.iloc[te])
    return oof


def extrapolation_split(df: pd.DataFrame, T_max_train: float = 950.0):
    tr = df["T_C"] <= T_max_train
    return np.where(tr)[0], np.where(~tr)[0]


def conformal_interval(model, df_train, y_train, df_test, alpha: float = 0.1, groups=None, seed: int = 0):
    """Split-conformal prediction interval (coverage ~ 1 - alpha).

    Calibrate the absolute residuals on held-out ALLOYS, then use their (1-alpha) quantile
    as a symmetric interval half-width around the model prediction.
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(groups) if groups is not None else np.arange(len(df_train))
    cal_ids = rng.choice(ids, size=max(1, len(ids) // 4), replace=False)
    cal = np.isin(groups if groups is not None else np.arange(len(df_train)), cal_ids)
    m = clone(model).fit(df_train[~cal], np.asarray(y_train)[~cal])
    res = np.abs(np.asarray(y_train)[cal] - m.predict(df_train[cal]))
    n = len(res)
    q = np.quantile(res, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    pred = m.predict(df_test)
    return pred, pred - q, pred + q, q
