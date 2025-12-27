from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def standardized_mean_diff(x_a: np.ndarray, x_b: np.ndarray) -> float:
    x_a = x_a.astype(float)
    x_b = x_b.astype(float)
    m1, m2 = np.nanmean(x_a), np.nanmean(x_b)
    v1, v2 = np.nanvar(x_a, ddof=1), np.nanvar(x_b, ddof=1)
    pooled = np.sqrt((v1 + v2) / 2.0) if (v1 + v2) > 0 else 0.0
    if pooled == 0:
        return 0.0
    return float((m2 - m1) / pooled)  # B - A


def chi_square_parity(df: pd.DataFrame, col: str) -> dict:
    tab = pd.crosstab(df['variant_assigned'], df[col])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"check": f"parity_{col}", "p_value": None, "effect": 0.0, "status": "WARN", "details": "insufficient categories"}
    chi2, p, _, _ = chi2_contingency(tab)
    n = tab.values.sum()
    k = min(tab.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n > 0 and k > 0 else 0.0
    return {"check": f"parity_{col}", "p_value": float(p), "effect": float(cramers_v), "status": "OK", "details": tab.to_dict()}


def baseline_equivalence(df: pd.DataFrame, baseline_col: str = "baseline_converted") -> dict:
    a = df.loc[df["variant_assigned"] == "A", baseline_col].to_numpy()
    b = df.loc[df["variant_assigned"] == "B", baseline_col].to_numpy()
    smd = standardized_mean_diff(a, b)
    return {"check": "baseline_equivalence", "smd": float(smd), "status": "OK", "details": {"mean_A": float(np.mean(a)), "mean_B": float(np.mean(b))}}


def run_pre_checks(df: pd.DataFrame) -> list[dict]:
    out = []
    out.append(baseline_equivalence(df))
    for col in ["device_type", "geo", "traffic_source"]:
        if col in df.columns:
            out.append(chi_square_parity(df, col))
    return out

