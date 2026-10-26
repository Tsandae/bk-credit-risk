"""Shared fixtures for the Module 4 tests (small synthetic data; never used for reported results)."""
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("shap")
pytest.importorskip("fastapi")
pytest.importorskip("fairlearn")

from src.module4 import config, data  # noqa: E402
from src.module4.demo_data import make_demo_curated  # noqa: E402
from src.module4.model import calibrate, candidates, category_levels, feature_frame, tune  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    df = make_demo_curated(9000, seed=3)
    tr, va, te = data.split(df)
    edges = data.income_edges(tr)
    return [data.add_groups(d, edges) for d in (tr, va, te)]


@pytest.fixture(scope="module")
def fitted(frames):
    tr, va, _ = frames
    levels = category_levels(tr)
    pipe, grid = candidates(levels)["lightgbm"]
    pipe, best, cv, _ = tune("lightgbm", pipe, grid, feature_frame(tr), tr[config.TARGET], n_iter=2, sample=3000)
    return calibrate(pipe, levels, feature_frame(frames[1]), frames[1][config.TARGET], "lightgbm")


