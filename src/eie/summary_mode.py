from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency
from eie.inputs import ManualIntegrityInputs


@dataclass
class Coverage:
    '''
    What was the model able to evaluate in summary/manual mode.
    '''
    evaluated: List[str]
    not_evaluated: List[str]


def _expected_counts_from_split(total: int, expected_split: Dict[str, float]) -> Tuple[float, float]:
    a = float(expected_split.get("A", 0.0))
    b = float(expected_split.get("B", 0.0))
    s = a + b
    a, b = a / s, b / s
    return total * a, total * b


def srm_check_manual(inp: ManualIntegrityInputs) -> dict:
    """
    SRM from intended split and observed assignment counts.
    """
    obsA, obsB = inp.observed_assigned.A, inp.observed_assigned.B
    total = obsA + obsB
    exp = inp.expected_split.normalize()
    expA, expB = _expected_counts_from_split(total, {"A": exp.A, "B": exp.B})

    stat, p = chisquare([obsA, obsB], f_exp=[expA, expB])
    # deviation from expected proportion (absolute)
    obs_propA = obsA / max(total, 1)
    dev = abs(obs_propA - exp.A)

    return {
        "check": "srm",
        "status": "OK" if p >= 0.001 and dev < 0.02 else ("WARN" if p >= 1e-6 and dev < 0.05 else "FAIL"),
        "p_value": float(p),
        "deviation": float(dev),
        "observed": {"A": int(obsA), "B": int(obsB)},
        "expected_prop": {"A": float(exp.A), "B": float(exp.B)},
    }


def _normalize_expected_dist(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in weights.values())
    return {k: float(v) / s for k, v in weights.items()}


def _flatten_observed_dist(observed: Dict[str, Dict[str, int]]) -> Tuple[List[str], np.ndarray]:
    '''
    observed: category -> {"A": count, "B": count}
    Reurns categories list and 2D table shape (2,k)
    '''
    cats = list(observed.keys())
    a = [int(observed[c]["A"]) for c in cats]
    b = [int(observed[c]["B"]) for c in cats]
    table = np.array([a, b], dtype = float)
    return cats, table

def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    '''
    Total variation distance
    '''
    return float(0.5 * np.abs(p - q).sum())

def covariate_parity_manual(
        name: str, 
        expected_weights: Optional[Dict[str, float]],
        observed_counts: Optional[Dict[str, Dict[str, int]]],
    )-> Optional[dict]:
    '''
    Parity check from expected distribution and observed distributions by variant.
    Returns a dict similar to full mode parity checks.
    '''
    if expected_weights is None or observed_counts is None:
        return None
    
    exp = _normalize_expected_dist(expected_weights)
    cats, table = _flatten_observed_dist(observed_counts)

    # Build expected counts for each variant based on its totals
    totalA = float(table[0].sum())
    totalB = float(table[1].sum())
    exp_vec = np.array([exp.get(c, 0.0) for c in cats], dtype = float)
    if exp_vec.sum() <= 0:
        return{
            "check": f"parity_{name}",
            "status": "WARN",
            "p_value": None,
            "effect_size": None,
            "details": "Expected distribution had no overlap with observed categories."
        }
    expA = exp_vec * totalA
    expB = exp_vec * totalB

    # 1. Chi-square vs expected for each arm and take the worse p (conservative)
    _, pA = chisquare(table[0], f_exp = expA + 1e-9)
    _, pB = chisquare(table[0], f_exp = expB + 1e-9)
    p = float(min(pA, pB))

    # 2. Effect size: compre the observed distributions between A and B (TVD)
    p_obsA = table[0] / max(totalA, 1.0)
    p_obsB = table[1] / max(totalB, 1.0)
    eff = _tvd(p_obsA, p_obsB)

    status = "OK"
    if p < 1e-3 or eff > 0.05:
        status = "WARN"
    if p < 1e-6 or eff > 0.10:
        status = "FAIL"
    return {
        "check": f"parity_{name}",
        "status": status,
        "p_value": p,
        "categories": cats, 
        "tvd_AB": float(eff)
    }
def ci_stability_manual(inp: ManualIntegrityInputs, n_boots: int = 400, seed: int = 7) -> dict:
    '''
    Bootstrap CI for lift using only (n, conversions) per variant, 
    If daily series exists, compare early vs late.
    '''
    rng = np.random.default_rng(seed)

    nA, nB = inp.observed_assigned.A, inp.observed_assigned.A
    cA, cB = inp.observed_conversions.A, inp.observed_conversions.B

    if nA < 50 or nB < 50:
        return {
            "check": "ci_stability",
            "status": "WARN",
            "details": "Insufficient sample for CI stabiltiy."
            }
    pA = cA / max(nA, 1)
    pB = cB / max(nB, 1)
    lifts = np.empty(n_boots, dtype = float)
    for i in range(n_boots):
        # binomial resample for aggregated rates
        a = rng.binomial(nA, pA)/max(nA, 1)
        b = rng.binomial(nB, pB)/max(nB, 1)
    
    lo, hi = np.percentile(lifts, [2.5, 97.5])
    base = {"ci_low": float(lo), "ci_high": float(hi), "ci_width": float(hi - lo), "lift_mean": float(lifts.mean())}

    # If no daiyl series is provided, cant do the early/late overlap. So mark as OK with base CI only.
    if not inp.daily:
        return {
            "check": "ci_stability",
            "status": "OK",
            "severity": 0.0,
            "base": base,
            "note": "No daily series provided for early/late stability." 
        }
    # Building early/late from daily rows
    daily  = pd.DataFrame([{
        "date": r.date, 
        "A_n":r.assigned.A, "B_n": r.assigned.B,
        "A_c": r.conversions.A, "B_c": r.conversions.B
     } for r in inp.daily])
    daily["date"] = pd.to_datetime(daily["date"], errors = "coerce")
    daily = daily.dropna(subset = ["date"]).sort_values("date")
    if len(daily) < 4:
        return{
            "check": "ci_stability",
            "status": "OK",
            "severity": 0.0,
            "base": base,
            "note": "Too few daily points for early/late split."
        }
    mid = daily["date"].iloc[len(daily)//2]
    early = daily[daily["date"] <= mid]
    late = daily[daily["date"] > mid]

    def boot_from_daily(df: pd.DataFrame, seed2: int) -> dict:
        nA2, nB2 = int(df["A_n"].sum()), int(df["B_n"].sum()) 
        cA2, cB2 = int(df["A_c"].sum()), int(df["B_c"].sum())
        if nA2 < 50 or nB2 < 50: 
            return {"OK": False}
        pA2, pB2 = cA2 / max(nA2, 1), cB2 / max(nB2, 1)
        rng2 = np.random.default_rng(seed2)
        lifts2 = np.empty(max(200, n_boots//2), dtype = float)
        for i in range(len(lifts2)):
            a = rng2.binomial(nA2, pA2) / max(nA2, 1)
            b = rng2.binomial(nB2, pB2) / max(nB2, 1)
            lifts2[i] = b - a
        lo2, hi2 = np.percentile(lifts2, [2.5, 97.5])
        return{"OK": True, "ci_low": float(lo2), "ci_high": float(hi2), "ci_width": float(hi2 - lo2), "lift_mean": float(lifts2.mean())}
    
    e = boot_from_daily(early, seed2 = 11)
    l = boot_from_daily(late, seed2 = 13)
    if not e.get("ok") or not l.get("ok"):
        return{"check": "ci_stability", "status": "OK", "severity": 0.0, "base": base, "note": "Early/late stability not computed (insufficient data)."}
    
    # Overlap ratio
    inter = max(0.0, min(e["ci_high"], l["ci_high"]) - max(e["ci_low"], l["ci_low"]))
    union = max(e["ci_high"], l["ci_high"]) - min(e["ci_low"], l["ci_low"])
    overlap = float(inter / union) if union > 0 else 0.0

    center_shift = abs(float(e["lift_mean"]) - float(l["lift_mean"]))
    width = max(base["ci_width"], 1e-9)

    severity = max(min(1.0, center_shift / width), min(1.0, (0.5 - overlap) / 0.5))
    status = "OK"
    if overlap < 0.5 or center_shift / width > 0.6:
        status = "WARN"
    if overlap < 0.25 or center_shift / width > 0.9:
        status = "FAIL"
    return{
        "check": "ci_stability",
        "status": status, 
        "severity": float(severity),
        "base": base,
        "early": e,
        "late": l,
        "overlap": overlap,
        "center_shift": float(center_shift)
    }

def volatility_manual(inp: ManualIntegrityInputs) -> Optional[dict]:
    '''
    Volatility (or drift) from daily conversion rates if daily series exists.
    '''
    if not inp.daily:
        return None
    daily = pd.DataFrame([{
        "date": r.date,
        "A_n": r.assigned.A, "B_n": r.assigned.B,
        "A_c": r.conversions.A, "B_c": r.conversions.B,
    } for r in inp.daily])
    daily["date"] = pd.to_datetime(daily["date"], errors = "coerce")
    daily = daily.dropna(subset = ["date"]).sort_values("date")
    if len(daily) < 4:
        return {"check": "volatility", "status": "OK", "severity": 0.0, "note": "Too few daily points."}
    
    daily["pA"] = daily["A_c"] / daily["A_n"].clip(lower=1)
    daily["pB"] = daily["B_c"] / daily["B_n"].clip(lower=1)
    vol = float((daily["pB"] - daily["pA"]).std(ddof=1))

    status = "OK"
    severity = min(1.0, vol / 0.01)  # 1pp std as "high"
    if vol > 0.005:
        status = "WARN"
    if vol > 0.01:
        status = "FAIL"

    return {"check": "volatility", "status": status, "severity": float(severity), "std_lift_daily": float(vol)}


def run_summary_checks(inp: ManualIntegrityInputs) -> Tuple[List[dict], Coverage]:
    checks: List[dict] = []

    # SRM
    checks.append(srm_check_manual(inp))

    # Covariate parity (if provided)
    parity = []
    p_dev = covariate_parity_manual(
        "device_type",
        inp.expected_device,
        inp.observed_device.categories if inp.observed_device else None,
    )
    if p_dev:
        parity.append(p_dev)

    p_geo = covariate_parity_manual(
        "geo",
        inp.expected_geo,
        inp.observed_geo.categories if inp.observed_geo else None,
    )
    if p_geo:
        parity.append(p_geo)

    p_src = covariate_parity_manual(
        "traffic_source",
        inp.expected_source,
        inp.observed_source.categories if inp.observed_source else None,
    )
    if p_src:
        parity.append(p_src)

    checks.extend(parity)

    # Post: CI stability
    checks.append(ci_stability_manual(inp, n_boots=400))

    # In-experiment: volatility (only if daily provided)
    vol = volatility_manual(inp)
    if vol:
        checks.append(vol)

    evaluated = ["srm", "ci_stability"]
    if p_dev: evaluated.append("parity_device_type")
    if p_geo: evaluated.append("parity_geo")
    if p_src: evaluated.append("parity_traffic_source")
    if vol: evaluated.append("volatility")

    not_evaluated = [
        "leakage",
        "not_exposed",
        "contamination_overlap",
        "baseline_equivalence",
        "exposure_timing",
        "user_level_sensitivity",
    ]

    return checks, Coverage(evaluated=evaluated, not_evaluated=not_evaluated)
