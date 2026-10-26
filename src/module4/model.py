"""Pipelines, tuning, calibration and the CreditRiskModel bundle."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def category_levels(train: pd.DataFrame) -> list:
    return [sorted(train[c].fillna("missing").astype(str).unique()) for c in config.CATEGORICAL]


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[config.FEATURES].copy()
    for c in config.CATEGORICAL:
        X[c] = X[c].fillna("missing").astype(str)
    return X


def make_preprocessor(levels, scale: bool) -> ColumnTransformer:
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    num = Pipeline([("imp", imp), ("sc", StandardScaler())]) if scale else imp
    cat = OneHotEncoder(categories=levels, handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, config.NUMERIC), ("cat", cat, config.CATEGORICAL)])


def feature_names(levels) -> list:
    """Output column -> original feature, in transformer order (numeric first, then one-hot blocks)."""
    names = list(config.NUMERIC)
    for col, lv in zip(config.CATEGORICAL, levels):
        names += [f"{col}={v}" for v in lv]
    return names


def origin_index(levels) -> np.ndarray:
    idx = list(range(len(config.NUMERIC)))
    for j, lv in enumerate(levels):
        idx += [len(config.NUMERIC) + j] * len(lv)
    return np.array(idx)


def candidates(levels, seed=config.SEED):
    mono = [1 if n in config.MONOTONE_UP else 0 for n in feature_names(levels)]
    return {
        "logistic_regression": (
            Pipeline([("pre", make_preprocessor(levels, True)),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0]}),
        "random_forest": (
            Pipeline([("pre", make_preprocessor(levels, False)),
                      ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                     n_jobs=-1, random_state=seed))]),
            {"clf__max_depth": [5, 8, 12], "clf__min_samples_leaf": [20, 50, 100]}),
        "lightgbm": (
            Pipeline([("pre", make_preprocessor(levels, False)),
                      ("clf", LGBMClassifier(class_weight="balanced", monotone_constraints=mono, verbose=-1,
                                             random_state=seed, n_jobs=-1))]),
            {"clf__n_estimators": [100, 150, 250, 400], "clf__learning_rate": [0.03, 0.05, 0.1],
             "clf__num_leaves": [15, 31, 63], "clf__min_child_samples": [20, 50, 100]}),
    }


def tune(name, pipe, grid, X, y, n_iter=config.N_ITER, seed=config.SEED, sample=config.CV_SAMPLE):
    """Random search with stratified k-fold CV on a sample (computing limits); refit on ALL training data."""
    size = int(np.prod([len(v) for v in grid.values()]))
    if len(X) > sample:
        idx = np.random.default_rng(seed).choice(len(X), sample, replace=False)
        Xs, ys = X.iloc[idx], y.iloc[idx]
    else:
        Xs, ys = X, y
    search = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, size), scoring="roc_auc",
                                cv=StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed),
                                random_state=seed, refit=False, n_jobs=1)
    search.fit(Xs, ys)
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    pipe.set_params(**search.best_params_)
    pipe.fit(X, y)
    return pipe, best, float(search.best_score_), pd.DataFrame(search.cv_results_)


@dataclass
class CreditRiskModel:
    """Fitted pipeline + isotonic calibrator + tier cutoffs (set on 2014 validation data)."""
    pipeline: Pipeline
    calibrator: IsotonicRegression
    cut_medium: float
    cut_high: float
    levels: list
    name: str = "model"
    background: object = None      # small sample of features, used only if a linear model needs SHAP

    def raw(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        return self.calibrator.predict(self.raw(X))

    def tier(self, raw) -> np.ndarray:
        raw = np.asarray(raw)
        return np.where(raw >= self.cut_high, "high", np.where(raw >= self.cut_medium, "medium", "low"))

    def flag(self, raw) -> np.ndarray:
        return (np.asarray(raw) >= self.cut_high).astype(int)


def calibrate(pipe, levels, Xval, yval, name="model") -> CreditRiskModel:
    raw = pipe.predict_proba(Xval)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, yval)
    cut_high = float(np.quantile(raw, 1 - config.HIGH_SHARE))
    cut_med = float(np.quantile(raw, 1 - config.HIGH_SHARE - config.MEDIUM_SHARE))
    bg = Xval.sample(min(300, len(Xval)), random_state=0)
    return CreditRiskModel(pipe, iso, cut_med, cut_high, levels, name, bg)


class ScoreEstimator(ClassifierMixin, BaseEstimator):
    """Thin prefit estimator so Fairlearn's ThresholdOptimizer can wrap the calibrated model."""

    def __init__(self, model=None):
        self.model = model
        self.classes_ = np.array([0, 1])

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        p = self.model.raw(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.model.flag(self.model.raw(X))
