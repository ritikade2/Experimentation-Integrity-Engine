from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict


@dataclass
class SimParams:
    experiment_id: str = "EXP_2025_001"
    n_users: int = 20000
    start_date: str = "2025-07-01"
    n_days: int = 21

    base_cr: float = 0.040
    true_lift: float = 0.002  # absolute lift, e.g. 0.002 = +0.2pp
    # inject issues
    srm_shift: float = 0.0  # +0.05 means 55/45 instead of 50/50 for A
    device_imbalance: float = 0.0  # pushes more mobile into one variant
    geo_imbalance: float = 0.0
    source_imbalance: float = 0.0
    leakage_rate: float = 0.0  # share of exposed users who see opposite variant
    not_exposed_rate: float = 0.0
    contamination_rate: float = 0.0  # overall rate
    contamination_bias: float = 0.0  # correlation with B (+ means more contamination in B)
    drift_strength: float = 0.0  # conversion drift over time (variant-differential)

    seed: int = 7


def _choice(rng, values, probs, n):
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(values, size=n, p=probs)


def generate_exp_users(params: SimParams) -> pd.DataFrame:
    rng = np.random.default_rng(params.seed)

    start = pd.to_datetime(params.start_date)
    # spread assignments over days
    assigned_day = rng.integers(0, params.n_days, size=params.n_users)
    assignment_ts = pd.Series([start] * params.n_users)

    assignment_ts = (
        assignment_ts
        + pd.to_timedelta(assigned_day, unit="D")
        + pd.to_timedelta(rng.integers(0, 24 * 3600, size=params.n_users), unit="s")
    )

    assignment_ts = pd.to_datetime(assignment_ts).dt.tz_localize(None)



    # SRM: expected 50/50; shift modifies P(A)
    pA = 0.5 + params.srm_shift
    pA = float(np.clip(pA, 0.01, 0.99))
    variant_assigned = _choice(rng, ["A", "B"], [pA, 1 - pA], params.n_users)

    # covariates base distributions
    device_vals = ["desktop", "mobile", "tablet"]
    geo_vals = ["US", "CA", "IN", "GB", "AU"]
    src_vals = ["organic", "paid", "email", "referral", "direct"]

    device_probs = np.array([0.45, 0.48, 0.07])
    geo_probs = np.array([0.55, 0.10, 0.15, 0.10, 0.10])
    src_probs = np.array([0.42, 0.18, 0.10, 0.15, 0.15])

    # introduce imbalance by skewing one category in B
    def skew(probs, idx, strength, in_variant):
        probs = probs.copy()
        if strength == 0:
            return probs
        # apply skew only to B (for simplicity)
        if in_variant == "B":
            probs[idx] = probs[idx] * (1 + strength)
        else:
            probs[idx] = probs[idx] * (1 - strength * 0.5)
        probs = np.clip(probs, 1e-6, None)
        return probs / probs.sum()

    device = []
    geo = []
    traffic_source = []
    for v in variant_assigned:
        device.append(_choice(rng, device_vals, skew(device_probs, 1, params.device_imbalance, v), 1)[0])  # mobile idx=1
        geo.append(_choice(rng, geo_vals, skew(geo_probs, 2, params.geo_imbalance, v), 1)[0])  # IN idx=2
        traffic_source.append(_choice(rng, src_vals, skew(src_probs, 1, params.source_imbalance, v), 1)[0])  # paid idx=1

    device = np.array(device)
    geo = np.array(geo)
    traffic_source = np.array(traffic_source)

    # baseline_converted independent of treatment (pre-period)
    baseline_converted = rng.binomial(1, 0.06, size=params.n_users)

    # exposure
    is_exposed = rng.random(params.n_users) >= params.not_exposed_rate
    first_exposure_ts = assignment_ts + pd.to_timedelta(rng.integers(1, 600, size=params.n_users), unit="s")
    first_exposure_ts = first_exposure_ts.where(is_exposed, pd.NaT)

    variant_exposed = np.where(is_exposed, variant_assigned, None).astype(object)

    # leakage among exposed
    leak_mask = (rng.random(params.n_users) < params.leakage_rate) & is_exposed
    variant_exposed = np.where(leak_mask, np.where(np.array(variant_assigned) == "A", "B", "A"), variant_exposed)

    # contamination proxy correlated with B if contamination_bias>0
    base_contam = params.contamination_rate
    in_other = np.zeros(params.n_users, dtype=int)
    for i, v in enumerate(variant_assigned):
        p = base_contam
        if params.contamination_bias != 0:
            if v == "B":
                p = np.clip(p * (1 + params.contamination_bias), 0, 1)
            else:
                p = np.clip(p * (1 - 0.5 * params.contamination_bias), 0, 1)
        in_other[i] = rng.binomial(1, p)

    # conversion probability
    assigned_day_norm = assigned_day / max(params.n_days - 1, 1)
    # drift affects B more if drift_strength > 0
    drift = params.drift_strength * assigned_day_norm * (np.array(variant_assigned) == "B").astype(float)

    p_conv = params.base_cr + drift
    p_conv += (np.array(variant_assigned) == "B").astype(float) * params.true_lift
    # contamination can inflate conversion slightly (fake lift driver)
    p_conv += in_other * 0.003

    p_conv = np.clip(p_conv, 0.0001, 0.95)
    converted = rng.binomial(1, p_conv)

    # conversion timestamp within 7 days if converted
    conv_delay_days = rng.integers(0, 7, size=params.n_users)
    conversion_ts = assignment_ts + pd.to_timedelta(conv_delay_days, unit="D") + pd.to_timedelta(
        rng.integers(0, 24 * 3600, size=params.n_users), unit="s")
    conversion_ts = conversion_ts.where(converted == 1, pd.NaT)

    df = pd.DataFrame(
        {
            "experiment_id": params.experiment_id,
            "user_id": [f"u{i:06d}" for i in range(params.n_users)],
            "variant_assigned": variant_assigned,
            "assignment_ts": pd.to_datetime(assignment_ts).dt.tz_localize(None),
            "variant_exposed": variant_exposed,
            "first_exposure_ts": pd.to_datetime(first_exposure_ts).dt.tz_localize(None),
            "converted": converted.astype(int),
            "conversion_ts": pd.to_datetime(conversion_ts).dt.tz_localize(None),
            "device_type": device,
            "geo": geo,
            "traffic_source": traffic_source,
            "in_other_experiment": in_other.astype(int),
            "baseline_converted": baseline_converted.astype(int),
        }
    )
    return df


def build_exp_daily(exp_users: pd.DataFrame) -> pd.DataFrame:
    df = exp_users.copy()
    df["date"] = pd.to_datetime(df["assignment_ts"]).dt.date

    daily = (
        df.groupby(["experiment_id", "date", "variant_assigned"], as_index=False)
        .agg(
            assigned_users=("user_id", "nunique"),
            exposed_users=("variant_exposed", lambda x: x.notna().sum()),
            conversions=("converted", "sum"),
        )
    )
    daily["conversion_rate"] = daily["conversions"] / daily["assigned_users"].replace({0: np.nan})
    return daily
