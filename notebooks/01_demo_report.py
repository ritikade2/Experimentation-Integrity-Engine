from eie.simulate import SimParams, generate_exp_users, build_exp_daily
from eie.report import integrity_report
from eie.config import EngineConfig

def run(name, params):
    exp_users = generate_exp_users(params)
    exp_daily = build_exp_daily(exp_users)

    report = integrity_report(exp_users, exp_daily, cfg=EngineConfig())

    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(report["narrative"])
    print("Score breakdown:", report["summary"])
    return exp_users, exp_daily, report

if __name__ == "__main__":
    clean = SimParams(n_users=30000, true_lift=0.002)

    broken = SimParams(
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

    u1, d1, r1 = run("Scenario A: CLEAN", clean)
    u2, d2, r2 = run("Scenario B: BROKEN", broken)

    # save files for screenshots / README
    u2.to_csv("data/exp_users.csv", index=False)
    d2.to_csv("data/exp_daily.csv", index=False)
