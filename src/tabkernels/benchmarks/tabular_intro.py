"""Chapter 1 benchmark slice: XGBoost vs MLP on a few small datasets.

Used by Figure 1.2 of the introductory chapter.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def run_grinsztajn_slice(
    n_train: int = 500,
    n_test: int = 200,
    n_features: int = 8,
    n_seeds: int = 3,
    models: Sequence[str] = ("xgboost", "mlp", "linear"),
) -> pd.DataFrame:
    """Run a small synthetic benchmark in the spirit of Grinsztajn 2022.

    The "datasets" are synthetic: each is a regression task with a different
    underlying function class. This is enough to illustrate the Chapter 1
    figure without external data dependencies.

    Returns a long-format DataFrame with columns: dataset, model, seed,
    test_mse, runtime_seconds.
    """
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None
    import time

    # Synthetic dataset generators.
    def linear_data(rng):
        X = rng.randn(n_train + n_test, n_features)
        beta = rng.randn(n_features)
        y = X @ beta + 0.1 * rng.randn(n_train + n_test)
        return X, y

    def sharp_axis_data(rng):
        # Tabular-like: sharp axis-aligned thresholds, mimics tree-friendly structure.
        X = rng.randn(n_train + n_test, n_features)
        y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 0).astype(float) + 0.1 * rng.randn(n_train + n_test)
        return X, y

    def smooth_data(rng):
        X = rng.randn(n_train + n_test, n_features)
        y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * rng.randn(n_train + n_test)
        return X, y

    datasets = {
        "synthetic_linear": linear_data,
        "synthetic_sharp_axis": sharp_axis_data,
        "synthetic_smooth": smooth_data,
    }

    rows = []
    for ds_name, gen in datasets.items():
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            X, y = gen(rng)
            X_tr, X_te = X[:n_train], X[n_train:]
            y_tr, y_te = y[:n_train], y[n_train:]
            for model_name in models:
                t0 = time.time()
                if model_name == "linear":
                    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                elif model_name == "xgboost" and XGBRegressor is not None:
                    m = XGBRegressor(n_estimators=100, max_depth=4, verbosity=0,
                                     random_state=seed)
                elif model_name == "mlp":
                    m = make_pipeline(
                        StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(64, 64),
                                     max_iter=500, random_state=seed),
                    )
                else:
                    continue
                m.fit(X_tr, y_tr)
                pred = m.predict(X_te)
                mse = float(((pred - y_te) ** 2).mean())
                rows.append({
                    "dataset": ds_name,
                    "model": model_name,
                    "seed": seed,
                    "test_mse": mse,
                    "runtime_seconds": time.time() - t0,
                })
    return pd.DataFrame(rows)
