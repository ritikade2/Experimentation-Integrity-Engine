import pandas as pd
import streamlit as st

from eie.simulate import SimParams, generate_exp_users, build_exp_daily
from eie.report import integrity_report, integrity_report_manual
from eie.config import EngineConfig

from eie.inputs import ManualIntegrityInputs, VariantCounts, ExpectedSplit, DistCounts, ManualDailyRow
from eie.io import (
    validate_exp_users,
    validate_exp_daily,
    exp_users_template,
    exp_daily_template,
)

st.set_page_config(page_title="Experimentation Integrity Engine", layout="wide")

st.title("Experimentation Integrity Engine")
st.caption("Audits experimentation integrity — not KPI computation.")

cfg = EngineConfig()


def render(report, exp_daily=None):
    s = report["summary"]
    st.metric("Reliability", f"{s['label']} ({s['score']}/100)")
    st.write(report["narrative"])

    if "coverage" in report:
        with st.expander("Coverage (what was evaluated)", expanded=False):
            st.json(report["coverage"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Penalties")
        st.dataframe(pd.DataFrame(s["penalties"]))
    with c2:
        st.subheader("Top reasons")
        st.write(s["top_reasons"] or ["No major integrity risks detected."])

    st.subheader("Checks")
    st.json(report["checks"], expanded=False)

    if exp_daily is not None and not exp_daily.empty:
        st.subheader("Daily conversion rate")
        d = exp_daily.copy()
        d["date"] = pd.to_datetime(d["date"])
        pivot = d.pivot_table(index="date", columns="variant_assigned", values="conversion_rate")
        st.line_chart(pivot)


st.subheader("Check Experiment Integrity")
with st.expander("How to use this tool", expanded=True):
    st.markdown("""
**Experimentation Integrity Engine** audits *whether an experiment’s results can be trusted* — not whether Variant B won.

Choose how you want to provide information:
- **Upload CSV (recommended):** Full audit using user-level data & daily experiment + user level data (optional).
- **Enter details manually:** Summary-based audit when raw data isn’t available.
- **Simulate scenarios:** Explore how integrity failures affect reliability.

The reliability score is explicitly scoped to the information provided.
""")

mode = st.radio(
    "How would you like to provide your experiment information?",
    ["Upload CSV (recommended)", "Enter details manually (summary audit)", "Simulate scenarios (advanced/learn)"],
    index=0,
)

st.divider()

# Mode 1: Upload CSV
if mode.startswith("Upload"):
    st.caption("Required: exp_users.csv. Optional: exp_daily.csv. If exp_daily is missing, it will be computed.")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download exp_users template",
            data=exp_users_template().to_csv(index=False).encode("utf-8"),
            file_name="exp_users_template.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download exp_daily template (optional)",
            data=exp_daily_template().to_csv(index=False).encode("utf-8"),
            file_name="exp_daily_template.csv",
            mime="text/csv",
        )

    st.divider()

    u_file = st.file_uploader("Upload exp_users.csv (required)", type=["csv"])
    d_file = st.file_uploader("Upload exp_daily.csv (optional)", type=["csv"])

    if u_file:
        exp_users_raw = pd.read_csv(u_file)
        exp_users, vr_u = validate_exp_users(exp_users_raw)

        if not vr_u.ok:
            st.error("exp_users.csv failed validation. Fix the issues below and re-upload.")
            for e in vr_u.errors:
                st.write(f"• [{e.code}] {e.message}")
            if vr_u.warnings:
                st.warning("Warnings")
                for w in vr_u.warnings:
                    st.write(f"• [{w.code}] {w.message}")
            st.stop()

        if vr_u.warnings:
            st.warning("exp_users.csv warnings (audit will proceed):")
            for w in vr_u.warnings:
                st.write(f"• [{w.code}] {w.message}")

        if d_file:
            exp_daily_raw = pd.read_csv(d_file)
            exp_daily, vr_d = validate_exp_daily(exp_daily_raw)
            if not vr_d.ok:
                st.error("exp_daily.csv failed validation. Either fix it or omit it (we can compute it).")
                for e in vr_d.errors:
                    st.write(f"• [{e.code}] {e.message}")
                if vr_d.warnings:
                    st.warning("Warnings")
                    for w in vr_d.warnings:
                        st.write(f"• [{w.code}] {w.message}")
                st.stop()
            if vr_d.warnings:
                st.warning("exp_daily.csv warnings (audit will proceed):")
                for w in vr_d.warnings:
                    st.write(f"• [{w.code}] {w.message}")
        else:
            exp_daily = build_exp_daily(exp_users)

        report = integrity_report(exp_users, exp_daily, cfg=cfg)
        render(report, exp_daily)


# Mode 2: Manual entry (Summary audit)
elif mode.startswith("Enter details"):
    st.caption("Summary-based audit. Some user-level integrity checks cannot be evaluated without raw data.")
    with st.expander("What this mode can and cannot evaluate", expanded=True):
        st.markdown(
"""
**Evaluated from summary inputs**
- SRM (expected vs observed traffic split)
- Covariate parity (device/geo/source) if you provide expected + observed distributions
- CI stability (bootstrap using counts)
- Volatility/drift if you provide a daily series

**Not evaluated (requires user-level data)**
- Exposure leakage (assigned vs exposed)
- Not-exposed rate
- True concurrent experiment overlap
- User-level sensitivity filtering
""" 
)

    experiment_id = st.text_input("experiment_id", value="manual_exp")

    col1, col2, col3 = st.columns(3)
    with col1:
        exp_A = st.number_input("Expected split A (weight)", value=0.5, step=0.05, format="%.2f")
        exp_B = st.number_input("Expected split B (weight)", value=0.5, step=0.05, format="%.2f")
    with col2:
        obs_A = st.number_input("Observed assigned users A", value=15000, step=500)
        obs_B = st.number_input("Observed assigned users B", value=15000, step=500)
    with col3:
        conv_A = st.number_input("Observed conversions A", value=600, step=50)
        conv_B = st.number_input("Observed conversions B", value=630, step=50)

    st.divider()
    st.subheader("Optional: expected vs observed distributions (counts)")

    add_device = st.checkbox("Add device distribution", value=False)
    add_geo = st.checkbox("Add geo distribution", value=False)
    add_source = st.checkbox("Add traffic_source distribution", value=False)

    def dist_editor(label: str):
        st.caption(f"{label}: Provide expected weights and observed counts by variant.")
        expected_text = st.text_area(
            f"{label} expected weights (JSON dict)",
            value='{"mobile": 0.6, "desktop": 0.4}',
            height=80,
            key=f"{label}_exp",
        )
        observed_text = st.text_area(
            f"{label} observed counts (JSON dict of category -> {{A,B}})",
            value='{"mobile": {"A": 9000, "B": 8800}, "desktop": {"A": 6000, "B": 6200}}',
            height=120,
            key=f"{label}_obs",
        )
        return expected_text, observed_text

    expected_device = observed_device = None
    expected_geo = observed_geo = None
    expected_source = observed_source = None

    import json
    if add_device:
        e_txt, o_txt = dist_editor("device")
        expected_device = json.loads(e_txt)
        observed_device = DistCounts(categories={k: VariantCounts(**v) for k, v in json.loads(o_txt).items()})

    if add_geo:
        e_txt, o_txt = dist_editor("geo")
        expected_geo = json.loads(e_txt)
        observed_geo = DistCounts(categories={k: VariantCounts(**v) for k, v in json.loads(o_txt).items()})

    if add_source:
        e_txt, o_txt = dist_editor("source")
        expected_source = json.loads(e_txt)
        observed_source = DistCounts(categories={k: VariantCounts(**v) for k, v in json.loads(o_txt).items()})

    st.divider()
    st.subheader("Optional: daily series (for volatility/drift)")
    add_daily = st.checkbox("Add daily series", value=False)

    daily_rows = None
    if add_daily:
        st.caption("Paste JSON list of daily rows with assigned and conversions by variant.")
        daily_txt = st.text_area(
            "Daily series JSON",
            value='[{"date":"2025-01-01","assigned":{"A":7500,"B":7500},"conversions":{"A":300,"B":315}},{"date":"2025-01-02","assigned":{"A":7500,"B":7500},"conversions":{"A":300,"B":315}}]',
            height=140,
        )
        daily_list = json.loads(daily_txt)
        daily_rows = [ManualDailyRow(**r) for r in daily_list]

    if st.button("Run integrity check (summary audit)"):
        inp = ManualIntegrityInputs(
            experiment_id=experiment_id,
            expected_split=ExpectedSplit(A=float(exp_A), B=float(exp_B)),
            observed_assigned=VariantCounts(A=int(obs_A), B=int(obs_B)),
            observed_conversions=VariantCounts(A=int(conv_A), B=int(conv_B)),
            expected_device=expected_device,
            observed_device=observed_device,
            expected_geo=expected_geo,
            observed_geo=observed_geo,
            expected_source=expected_source,
            observed_source=observed_source,
            daily=daily_rows,
        )
        report = integrity_report_manual(inp, cfg=cfg)
        render(report, exp_daily=None)


# Mode 3: Simulation
else:
    st.caption("Simulation and validation mode. Use this to learn how integrity failures affect reliability.")

    colA, colB, colC = st.columns(3)
    with colA:
        n_users = st.number_input("n_users", min_value=2000, max_value=200000, value=30000, step=1000)
        true_lift = st.number_input("true_lift (absolute)", value=0.002, step=0.001, format="%.3f")
        srm_shift = st.number_input("srm_shift (+ means more A)", value=0.00, step=0.01, format="%.2f")
    with colB:
        device_imbalance = st.number_input("device_imbalance", value=0.00, step=0.05, format="%.2f")
        geo_imbalance = st.number_input("geo_imbalance", value=0.00, step=0.05, format="%.2f")
        source_imbalance = st.number_input("source_imbalance", value=0.00, step=0.05, format="%.2f")
    with colC:
        leakage_rate = st.number_input("leakage_rate", value=0.00, step=0.01, format="%.2f")
        not_exposed_rate = st.number_input("not_exposed_rate", value=0.00, step=0.01, format="%.2f")
        contamination_rate = st.number_input("contamination_rate", value=0.00, step=0.02, format="%.2f")
        contamination_bias = st.number_input("contamination_bias", value=0.00, step=0.10, format="%.2f", disabled=float(contamination_rate) <= 0.0)
        if float(contamination_rate) <= 0.0:
            contamination_bias = 0.0
        drift_strength = st.number_input("drift_strength", value=0.00, step=0.002, format="%.3f")

    if st.button("Run integrity report (simulation)"):
        params = SimParams(
            n_users=int(n_users),
            true_lift=float(true_lift),
            srm_shift=float(srm_shift),
            device_imbalance=float(device_imbalance),
            geo_imbalance=float(geo_imbalance),
            source_imbalance=float(source_imbalance),
            leakage_rate=float(leakage_rate),
            not_exposed_rate=float(not_exposed_rate),
            contamination_rate=float(contamination_rate),
            contamination_bias=float(contamination_bias),
            drift_strength=float(drift_strength),
        )
        exp_users = generate_exp_users(params)
        exp_daily = build_exp_daily(exp_users)
        report = integrity_report(exp_users, exp_daily, cfg=cfg)
        render(report, exp_daily)
