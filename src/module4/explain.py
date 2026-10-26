"""SHAP explanations aggregated to the original business features, plus plain-language reasons."""
import numpy as np
import pandas as pd
import shap

from . import config
from .model import feature_frame, origin_index


class Explainer:
    def __init__(self, model):
        self.model = model
        self.pre = model.pipeline.named_steps["pre"]
        self.clf = model.pipeline.named_steps["clf"]
        self.origin = origin_index(model.levels)
        from sklearn.linear_model import LogisticRegression
        if isinstance(self.clf, LogisticRegression):
            self.tree = shap.LinearExplainer(self.clf, self.pre.transform(model.background))
        else:
            self.tree = shap.TreeExplainer(self.clf)

    def values(self, X: pd.DataFrame):
        """Return (aggregated SHAP matrix n x 22 in log-odds, base value)."""
        Xt = self.pre.transform(X)
        sv = self.tree.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        agg = np.zeros((sv.shape[0], len(config.FEATURES)))
        for j in range(sv.shape[1]):
            agg[:, self.origin[j]] += sv[:, j]
        base = self.tree.expected_value
        base = float(np.ravel(base)[-1])
        return agg, base

    def reasons(self, X_row: pd.DataFrame, k=3):
        agg, _ = self.values(X_row)
        row = X_row.iloc[0]
        order = np.argsort(-np.abs(agg[0]))[:k]
        return [{"feature": config.FEATURES[i], "value": _py(row[config.FEATURES[i]]),
                 "effect": "raises risk" if agg[0][i] > 0 else "lowers risk",
                 "contribution": round(float(agg[0][i]), 4)} for i in order]


def _py(v):
    return v.item() if hasattr(v, "item") else v


def global_importance(agg) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": config.FEATURES, "mean_abs_shap": np.abs(agg).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def encoded(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric view of features for beeswarm colouring (categories -> codes)."""
    E = X[config.FEATURES].copy()
    for c in config.CATEGORICAL:
        E[c] = E[c].astype("category").cat.codes
    return E.astype(float)
