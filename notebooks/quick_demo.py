import pandas as pd
from eie.simulate import SimParams, generate_exp_users, build_exp_daily
from eie.report import integrity_report
from eie.config import EngineConfig

# params = SimParams(
#     n_users=30000,
#     true_lift=0.002,
#     srm_shift=0.05,
#     device_imbalance=0.10,
#     leakage_rate=0.02,
#     not_exposed_rate=0.06,
#     contamination_rate=0.08,
#     contamination_bias=0.30,
#     drift_strength=0.004,
# )
# exp_users = generate_exp_users(params)
# exp_daily = build_exp_daily(exp_users)

# exp_users.to_csv("data/exp_users.csv", index=False)
# exp_daily.to_csv("data/exp_daily.csv", index=False)

# report = integrity_report(exp_users, exp_daily, cfg=EngineConfig())
# print(report["narrative"])
# print(report["summary"])


def run_scenario(name: str, params: SimParams):
    exp_users = generate_exp_users(params)
    exp_daily = build_exp_daily(exp_users)
    report = integrity_report(exp_users, exp_daily, cfg=EngineConfig())

    print(f"\n=== {name} ===")
    print(report["narrative"])
    print(report["summary"])

if __name__ == "__main__":
    # Scenario A: CLEAN (no injected issues)
    params_clean = SimParams(
        n_users=30000,
        true_lift=0.002,
        # all issue toggles default to 0.0
    )
    # Scenario B: BROKEN (inject issues)
    params_broken = SimParams(
        n_users=30000,
        true_lift=0.002,
        srm_shift=0.05,
        device_imbalance=0.10,
        leakage_rate=0.02,
        not_exposed_rate=0.06,
        contamination_rate=0.08,
        contamination_bias=0.30,
        drift_strength=0.004,
    )

    run_scenario("Scenario A: CLEAN", params_clean)
    run_scenario("Scenario B: BROKEN", params_broken)