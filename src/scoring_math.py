"""v3.8.0 statistical helper functions; no knee or truth code.

Only change from v3.8.0 (v2.1.0): loco_adjust can leave rows with |y| > train_abs_max out of the ridge
*training* set. Every row is still adjusted, and the centring median still uses all finite rows.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def geo_mean(values: list[float]) -> float:
    values = np.asarray(values, float)
    if np.any(~np.isfinite(values)):
        return math.nan
    values = np.clip(values, 0.0, 1.0)
    if np.any(values <= 0):
        return 0.0
    return float(np.exp(np.log(values).mean()))


def harmonic(values: list[float]) -> float:
    values = np.asarray(values, float)
    if np.any(~np.isfinite(values)):
        return math.nan
    if np.any(values <= 0):
        return 0.0
    return float(len(values) / np.sum(1.0 / values))


def feature_matrix(df: pd.DataFrame, q_cols: list[str], la_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    q_by_suffix = {x.removeprefix("Q_"): x for x in q_cols}
    la_by_suffix = {x.removeprefix("LA_"): x for x in la_cols}
    common = [x for x in q_by_suffix if x in la_by_suffix]
    if len(common) < 2:
        raise ValueError(f"Need at least two matched global/local ancestry components; Q={q_cols}, LA={la_cols}")
    # Omit the final component to remove compositional collinearity.
    used = common[:-1]
    arrays, names = [], []
    for suffix in used:
        q = pd.to_numeric(df[q_by_suffix[suffix]], errors="coerce").to_numpy(float)
        la = pd.to_numeric(df[la_by_suffix[suffix]], errors="coerce").to_numpy(float)
        arrays.extend([q, la - q]); names.extend([f"Q_{suffix}", f"EXCESS_{suffix}"])
    q_all = np.column_stack([pd.to_numeric(df[q_by_suffix[x]], errors="coerce") for x in common])
    la_all = np.column_stack([pd.to_numeric(df[la_by_suffix[x]], errors="coerce") for x in common])
    divergence = 0.5 * np.abs(la_all - q_all).sum(axis=1)
    safe = np.clip(la_all, 1e-12, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1) / math.log(len(common))
    arrays.extend([divergence, entropy]); names.extend(["LA_GLOBAL_TV", "LA_ENTROPY"])
    return np.column_stack(arrays), names


def loco_adjust(df: pd.DataFrame, y_col: str, X: np.ndarray, ridge_lambda: float,
                train_abs_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    chrom = pd.to_numeric(df["CHR"], errors="coerce").to_numpy(int)
    adjusted = y.copy()
    fallback = np.ones(len(df), dtype=int)
    center = float(np.nanmedian(y)) if np.any(np.isfinite(y)) else math.nan
    # Z_H divides by max(1.4826*wMAD, 1e-6), so a near-constant background gives |Z_H| up to ~1e5. A handful of
    # such rows dominates the least-squares intercept and shifts every adjusted score; keep them out of training.
    trainable = np.isfinite(y) if train_abs_max is None else np.isfinite(y) & (np.abs(y) <= train_abs_max)
    for c in np.unique(chrom):
        test = chrom == c
        train = ~test & trainable & np.all(np.isfinite(X), axis=1)
        predmask = test & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if train.sum() < max(20, X.shape[1] + 5) or predmask.sum() == 0:
            continue
        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0)
        sd[sd < 1e-12] = 1.0
        model = Ridge(alpha=ridge_lambda, fit_intercept=True)
        model.fit((X[train] - mu) / sd, y[train])
        pred = model.predict((X[predmask] - mu) / sd)
        adjusted[predmask] = center + y[predmask] - pred
        fallback[predmask] = 0
    return adjusted, fallback


