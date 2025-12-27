from __future__ import annotations
import numpy as np
import pandas as pd

def _lift_from_df(df: pd.DataFrame) -> dict:
    '''
    Conversion rate by assigned variant
    '''
    g = df.groupby("variant_assigned")["converted"].agg(["sum", "count"])
    if "A" not in g.index or "B" not in g.index:
        return {"ok": False, "reason": "missing A or B"}
    pA = g.loc["A", "sum"]/max(g.loc["A", "count"], 1)
    pB = g.loc["B", "sum"]/max(g.loc["B", "count"], 1)
    return {"ok": True, "pA": float(pA), "pB": float(pB), "lift": float(pB - pA),
            "nA": int(g.loc["A", "count"]), "nB": int(g.loc["B", "count"])}

def _bootstrap_lift(df: pd.DataFrame, n_boot: int = 400, seed: int = 7) -> dict:
    '''
    Stratified bootstrap: resample users within each assigned variant separately. 
    Returns CI and distribution for lift pB - pA
    '''
    rng = np.random.default_rng(seed)

    dfA = df[df["variant_assigned"] == "A"]
    dfB = df[df["variant_assigned"] == "B"]

    nA, nB = len(dfA), len(dfB)

    if nA < 50 or nB < 50:
        return {"ok": False, "reason": "insufficient sample for bootstrap"}
    
    a_conv = dfA["converted"].to_numpy()
    b_conv = dfB["converted"].to_numpy()

    lifts = np.empty(n_boot, dtype = float)
    for i in range(n_boot):
        a_idx = rng.integers(0, nA, size = nA)
        b_idx = rng.integers(0, nB, size = nB) 
        pA = a_conv[a_idx].mean()
        pB = b_conv[b_idx].mean()
        lifts[i] = pB - pA

    lo, hi = np.percentile(lifts, [2.5, 97.5])
    return {
        "ok": True,
        "n_boot": int(n_boot),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_width": float(hi - lo),
        "lift_mean": float(lifts.mean()),
        "lift_std": float(lifts.std(ddof=1)),
    }    


def _window_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    split by assignment date median 
    (early vs late)
    '''
    d = df.copy()
    d["assignment_date"] = pd.to_datetime(d["assignment_ts"]).dt.date
    med = pd.to_datetime(d["assignment_date"]).median().date()
    early = d[d["assignment_date"] <= med]
    late = d[d["assignment_date"] > med ]
    return early, late
    

def _overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    inter = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    union = max(a_hi, b_hi) - min(a_lo, b_lo)
    return float(inter / union) if union > 0 else 0.0



def _ci_stability_check(exp_users: pd.DataFrame, n_boot: int = 400) -> dict:
    base = _bootstrap_lift(exp_users, n_boot = n_boot, seed = 7)
    if not base["ok"]:
        return {"check": "ci_stability", "status": "WARN", "details": base}
    
    early, late = _window_split(exp_users)
    e = _bootstrap_lift(early, n_boot = max(200, n_boot // 2), seed = 11)
    l = _bootstrap_lift(late, n_boot = max(200, n_boot // 2), seed = 13)

    if not e["ok"] or not l["ok"]:
        return {"check": "ci_stability", "status": "WARN", "details": {"base": base, "early": e, "late": l}}

    overlap = _overlap(e["ci_low"], e["ci_high"], l["ci_low"], l["ci_high"])
    center_shift = abs(e["lift_mean"] - l["lift_mean"])
    width = base["ci_width"]

    
    # Simple severity: big center shift relative to overall CI width OR low overlap
    severity = 0.0 
    if width > 0:
        severity = max(severity, min(1.0, center_shift / width))
    severity = max(severity, min(1.0, (0.5 - overlap) / 0.5 )) # if overlap < 0.5 -> severity rises

    status = "OK"
    if overlap < 0.5 or (width > 0 and center_shift / width > 0.6):
        status = "WARN"
    if overlap < 0.25 or (width > 0 and center_shift / width > 0.9):
        status = "FAIL"
    
    return {
        "check": "ci_stability",
        "status": status,
        "severity": severity,
        "base": base, 
        "early": e,
        "late": l,
        "overlap": float(overlap),
        "center_shift": float(center_shift)
    }


def sensitivity_analysis(exp_users: pd.DataFrame) -> dict:
    base = _lift_from_df(exp_users)
    if not base["ok"]:
        return {"check": "sensitivity", "status": "WARN", "details": base}

    scenarios = {}

    def add(name: str, df: pd.DataFrame):
        r = _lift_from_df(df)
        scenarios[name] = r
    
    # 1) Exclude contaminated users (proxy)
    if "in_other_experiment" in exp_users.columns:
        add("exclude_contaminated", exp_users[exp_users["in_other_experiment"] == 0])

    # 2) Exclude paid traffic
    if "traffic_source" in exp_users.columns:
        add("exclude_paid", exp_users[exp_users["traffic_source"] != "paid"])

    # 3) Exposed only (drop not-exposed)
    if "variant_exposed" in exp_users.columns:
        add("exposed_only", exp_users[exp_users["variant_exposed"].notna()])

    # 4) Strict compliance: exposed users where exposed == assigned
    if "variant_exposed" in exp_users.columns:
        compliant = exp_users[(exp_users["variant_exposed"].notna()) & (exp_users["variant_exposed"] == exp_users["variant_assigned"])]
        add("compliant_only", compliant)
    
    # compute fragility: max abs delta in lift vs base
    deltas = []
    for k, r in scenarios.items():
        if r.get("ok"):
            deltas.append(abs(r["lift"] - base["lift"]))

    max_delta = float(max(deltas)) if deltas else 0.0

    # status thresholds (absolute lift shift)
    status = "OK"
    if max_delta > 0.002:  # 0.2pp
        status = "WARN"
    if max_delta > 0.005:  # 0.5pp
        status = "FAIL"

    return {
        "check": "sensitivity",
        "status": status,
        "base": base,
        "scenarios": scenarios,
        "max_abs_lift_shift": float(max_delta)
    }

def run_post_checks(exp_users: pd.DataFrame, exp_daily: pd.DataFrame | None) -> list[dict]:
    out =[]
    out.append(_ci_stability_check(exp_users, n_boot = 400))
    out.append(sensitivity_analysis(exp_users))
    return out
        




# # MVP: keep structure; Before adding bootstrap CI stability
# def run_post_checks(exp_users: pd.DataFrame, exp_daily: pd.DataFrame | None) -> list[dict]:
#     return [{"check": "post_placeholder", "status": "OK", "details": "Post checks (CI stability / sensitivity) coming next."}]