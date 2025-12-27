from __future__ import annotations
import pandas as pd
from typing import Optional, Dict, Any

from eie.config import EngineConfig
from eie.checks.pre import run_pre_checks
from eie.checks.inexp import run_inexp_checks
from eie.checks.post import run_post_checks
from eie.scoring import score_report

from eie.inputs import ManualIntegrityInputs
from eie.summary_mode import run_summary_checks

# Detailed Full Report
def integrity_report(exp_users: pd.DataFrame, exp_daily: Optional[pd.DataFrame] = None, cfg: EngineConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or EngineConfig()

    pre = run_pre_checks(exp_users)
    inexp = run_inexp_checks(
        exp_users,
        exp_daily,
        expected_split={"A": cfg.expected_split.A, "B": cfg.expected_split.B},
    )
    post = run_post_checks(exp_users, exp_daily)

    score = score_report(pre, inexp, post, cfg)

    # simple narrative
    narrative = []
    narrative.append(f"Reliability: {score['label']} (score {score['score']}/100).")
    if score["top_reasons"]:
        narrative.append("Top concerns: " + "; ".join(score["top_reasons"]) + ".")
    else:
        narrative.append("No major integrity risks detected from available signals.")
    narrative_text = " ".join(narrative)

    return {
        "summary": score,
        "narrative": narrative_text,
        "checks": {"pre": pre, "in_experiment": inexp, "post": post},
    }


# Summary Report (when user enters limited info)
def integrity_report_manual(inp: ManualIntegrityInputs, cfg: EngineConfig | None = None):
    '''
    Manual / summary mode integrity report.
    Produces a scoped audit: some checks are not evaluated due to missing user-level data.
    '''
    cfg = cfg or EngineConfig()

    checks, coverage = run_summary_checks(inp)

    score = score_report(pre=[], inexp=[], post=checks, cfg=cfg, mode="summary")

    narrative = [f"Reliability: {score['label']} (score {score['score']}/100)."]
    if score.get("top_reasons"):
        narrative.append("Top concerns: " + "; ".join(score["top_reasons"]) + ".")
    if coverage.not_evaluated:
        narrative.append("Note: Summary-based audit; some user-level integrity checks were not evaluated.")
    narrative_text = " ".join(narrative)

    return {
        "mode": "summary",
        "coverage": {"evaluated": coverage.evaluated, "not_evaluated": coverage.not_evaluated},
        "checks": {"summary": checks},
        "summary": score,
        "narrative": narrative_text,
    }
