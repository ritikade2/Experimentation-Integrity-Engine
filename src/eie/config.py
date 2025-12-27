from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict

# Test/Control Group Slit
class ExpectedSplit(BaseModel):
    A: float = 0.5
    B: float = 0.5

# Weights for Integrity Failures
class Weights(BaseModel):
    srm: float = 40.0 # Sample Ratio Mismatch (i.e. traffic plit not as intended). Can't fix, have to invalidate experiment.
    covariate_imbalance: float = 20.0 # Device/geo/traffic source imbalance. Mitigated by stratification, reweighting, regression adjustment.
    baseline_noneq: float = 20.0 # Baseline KPI non-equivalence & violates assumption of exchangeability. Indicates broken randomization. 
    leakage: float = 25.0 # Users seeing wrong variant or switching. This destroys causal contrast effecting lift magnitudes. Worse than imbalance.
    contamination: float = 20.0 # Concurrent experiemntation interference. Can inflate or mask lift.  
    volatility: float = 15.0 # Metric instability during the run. Often caused by rollout issues, logging problems, or traffic shocks. 
    ci_instability: float = 25.0 # Confidence interval instability (post experiment). Attacks decision confidence. Makes conclusion unreliable.

class EngineConfig(BaseModel):
    expected_split: ExpectedSplit = ExpectedSplit()
    alpha: float = 0.05
    # thresholds
    smd_warn: float = 0.10
    smd_fail: float = 0.20
    leakage_warn: float = 0.01
    leakage_fail: float = 0.03
    not_exposed_warn: float = 0.05
    not_exposed_fail: float = 0.10
    contamination_warn: float = 0.05
    contamination_fail: float = 0.10
    srm_dev_warn: float = 0.01
    srm_dev_fail: float = 0.02

    
    # scoring weights
    weights: Weights = Weights()

    