from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chisquare


def srm_check(df: pd.DataFrame, expected_a: float = 0.5, expected_b: float = 0.5) -> dict:
    counts = df["variant_assigned"].value_counts().reindex(["A", "B"]).fillna(0).to_numpy()
    n = counts.sum()
    exp = np.array([expected_a, expected_b]) * n
    stat, p = chisquare(counts, f_exp=exp)
    deviation = float(abs((counts[0] / n) - expected_a)) if n > 0 else 0.0
    return {
        "check": "srm",
        "p_value": float(p),
        "stat": float(stat),
        "deviation": deviation,
        "counts": {"A": int(counts[0]), "B": int(counts[1])},
        "status": "OK",
    }


def leakage_check(df: pd.DataFrame) -> dict:
    if "variant_exposed" not in df.columns:
        return {"check": "leakage", "rate": None, "status": "WARN", "details": "variant_exposed missing"}
    exposed = df[df["variant_exposed"].notna()].copy()
    if exposed.empty:
        return {"check": "leakage", "rate": 0.0, "status": "WARN", "details": "no exposed users"}
    leak = (exposed["variant_exposed"] != exposed["variant_assigned"]).mean()
    not_exposed_rate = df["variant_exposed"].isna().mean()
    return {
        "check": "leakage",
        "leakage_rate": float(leak),
        "not_exposed_rate": float(not_exposed_rate),
        "status": "OK",
    }


def contamination_check(df: pd.DataFrame) -> dict:
    if "in_other_experiment" not in df.columns:
        return {"check": "contamination", "rate": None, "status": "WARN", "details": "in_other_experiment missing"}
    rates = df.groupby("variant_assigned")["in_other_experiment"].mean().to_dict()
    diff = float(rates.get("B", 0.0) - rates.get("A", 0.0))
    return {"check": "contamination", "rates": {k: float(v) for k, v in rates.items()}, "diff_B_minus_A": diff, "status": "OK"}


def volatility_check(exp_daily: pd.DataFrame) -> dict:
    # Simple MVP: compare day-to-day conversion_rate variance across variants
    if exp_daily is None or exp_daily.empty:
        return {"check": "volatility", "status": "WARN", "details": "exp_daily missing/empty"}
    d = exp_daily.copy()
    if "conversion_rate" not in d.columns:
        return {"check": "volatility", "status": "WARN", "details": "conversion_rate missing"}
    var_by_v = d.groupby("variant_assigned")["conversion_rate"].var(ddof=1).to_dict()
    a, b = float(var_by_v.get("A", 0.0) or 0.0), float(var_by_v.get("B", 0.0) or 0.0)
    ratio = (b / a) if a > 0 else (np.inf if b > 0 else 1.0)
    return {"check": "volatility", "var": {"A": a, "B": b}, "var_ratio_B_over_A": float(ratio), "status": "OK"}


def run_inexp_checks(df: pd.DataFrame, exp_daily: pd.DataFrame | None, expected_split: dict) -> list[dict]:
    out = []
    out.append(srm_check(df, expected_split.get("A", 0.5), expected_split.get("B", 0.5)))
    out.append(leakage_check(df))
    out.append(contamination_check(df))
    out.append(volatility_check(exp_daily))
    return out
