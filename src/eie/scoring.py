from __future__ import annotations
import math
from typing import Dict, List


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def p_to_severity(p: float | None) -> float:
    # smaller p => higher severity
    if p is None:
        return 0.25
    if p <= 1e-6:
        return 1.0
    return float(clamp(-math.log10(p) / 6.0, 0.0, 1.0))


def score_report(pre: List[dict], inexp: List[dict], post: List[dict], cfg, mode: str = "full") -> dict:
    w = cfg.weights
    penalties = []
    reasons = []

    # Sample Ratio Mismatch (works for both full mode and summary mode)
    all_checks = (pre or []) + (inexp or []) + (post or [])
    srm = next((x for x in all_checks if x.get("check") == "srm"), None)
    if srm:
        dev = float(srm.get("deviation") or 0.0)

        # Practical severity based on deviation size
        dev_sev = clamp(
            (dev - cfg.srm_dev_warn) /
            (cfg.srm_dev_fail - cfg.srm_dev_warn),
            0.0,
            1.0,
        )

        # Statistical severity based on p-value
        p_sev = p_to_severity(srm.get("p_value"))

        # Combine (requires both to matter)
        sev = dev_sev * p_sev

        pen = w.srm * sev
        penalties.append(("srm", pen))
        if sev > 0.3:
            pval = srm.get("p_value")
            reasons.append(f"SRM risk (p={None if pval is None else float(pval):.3g}, deviation={dev:.3%})")


    # Covariate parity via chi-square p-values + effect sizes
    parity = [x for x in pre if str(x.get("check", "")).startswith("parity_")]
    if parity:
        worst = 0.0
        worst_name = None
        worst_p = None
        for item in parity:
            p = item.get("p_value")
            eff = float(item.get("effect_size", item.get("effect", 0.0)) or 0.0)
            sev = max(clamp(eff / 0.15, 0.0, 1.0), 0.4 * p_to_severity(p))
            if sev > worst:
                worst, worst_name, worst_p = sev, item["check"], p
        pen = w.covariate_imbalance * worst
        penalties.append(("covariate_imbalance", pen))
        if worst > 0.3 and worst_name:
            reasons.append(f"Traffic/covariate imbalance ({worst_name}, p={None if worst_p is None else f'{worst_p:.3g}'})")

    # Baseline equivalence via SMD
    base = next((x for x in pre if x["check"] == "baseline_equivalence"), None)
    if base:
        smd = abs(float(base.get("smd", 0.0) or 0.0))
        sev = clamp(smd / cfg.smd_fail, 0.0, 1.0)
        pen = w.baseline_noneq * sev
        penalties.append(("baseline_noneq", pen))
        if sev > 0.3:
            reasons.append(f"Baseline non-equivalence (SMD={base.get('smd'):.3f})")

    # Leakage + not exposed
    leak = next((x for x in inexp if x["check"] == "leakage"), None)
    if leak:
        leak_rate = float(leak.get("leakage_rate") or 0.0)
        not_exp = float(leak.get("not_exposed_rate") or 0.0)
        sev = max(
            clamp(leak_rate / cfg.leakage_fail, 0.0, 1.0),
            clamp(not_exp / cfg.not_exposed_fail, 0.0, 1.0),
        )
        pen = w.leakage * sev
        penalties.append(("leakage", pen))
        if sev > 0.3:
            reasons.append(f"Exposure integrity issues (leakage={leak_rate:.2%}, not_exposed={not_exp:.2%})")

    # Contamination proxy
    cont = next((x for x in inexp if x["check"] == "contamination"), None)
    if cont:
        rates = cont.get("rates", {})
        avg = (float(rates.get("A", 0.0)) + float(rates.get("B", 0.0))) / 2.0
        diff = abs(float(cont.get("diff_B_minus_A") or 0.0))
        sev = max(clamp(avg / cfg.contamination_fail, 0.0, 1.0), clamp(diff / cfg.contamination_fail, 0.0, 1.0))
        pen = w.contamination * sev
        penalties.append(("contamination", pen))
        if sev > 0.3:
            reasons.append(f"Concurrent experiment contamination risk (A={rates.get('A',0):.2%}, B={rates.get('B',0):.2%})")

    # Volatility
    vol = next((x for x in all_checks if x.get("check") == "volatility"), None)
    if vol and vol.get("status") != "WARN":
        ratio = float(vol.get("var_ratio_B_over_A") or 1.0)
        sev = clamp(abs(math.log(ratio)) / math.log(3), 0.0, 1.0) if ratio > 0 else 0.0
        pen = w.volatility * sev
        penalties.append(("volatility", pen))
        if sev > 0.3:
            reasons.append(f"Metric volatility drift (var ratio B/A={ratio:.2f})")
    
    # CI stability (post)
    ci = next((x for x in all_checks if x.get("check") == "ci_stability"), None)
    if ci and ci.get("status") != "WARN":
        sev = float(ci.get("severity") or 0.0)
        pen = w.ci_instability * clamp(sev, 0.0, 1.0)
        penalties.append(("ci_instability", pen))
        if sev > 0.3:
            reasons.append(f"CI stability concerns (overlap={ci.get('overlap', 0):.2f}, center_shift={ci.get('center_shift', 0):.4f})")


    total_pen = sum(p for _, p in penalties)
    raw_score = 100.0 - total_pen

    # Cap score when running in summary/manual mode (missing user-level dimensions)
    max_score = 100.0
    if mode == "summary":
        max_score = 80.0
    score = clamp(raw_score, 0.0, max_score)

    # Labeling: scale thresholds if score is capped (summary mode)
    scale = max_score/100.0

    if score >= 85 * scale:
        label = "HIGH"
    elif score >= 65 * scale:
        label = "MODERATE"
    elif score >= 40 * scale:
        label = "LOW"
    else:
        label = "UNSAFE"

    return {
        "score": round(score, 1),
        "label": label,
        "penalties": [{"component": k, "penalty": round(v, 2)} for k, v in penalties],
        "top_reasons": reasons[:3],
    }
