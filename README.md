# Experimentation Integrity Engine (EIE)

**Audits experiment systems - not KPI lift.**

The Experimentation Integrity Engine (EIE) evaluates whether experiment's results are *methodologically thrustworthy* before anyone interprets statistical outcomes or declares winners. 

Instead of answering:
> "Did Variant B win?"

EIE answers:
> **"Should we even believe this experiment?"**

---

## Why this exists

Modern experimentation platforms make it easy to compute lift, confidence intervals, and p-values — but they rarely audit the *validity of the experiment itself*.

In practice, many experiments are affected by:
- Sample Ratio Mismatch (SRM)
- Traffic and device imbalance
- Exposure leakage
- Concurrent experiment contamination
- Metric instability over time

These issues can invalidate conclusions even when results appear statistically significant.

EIE operates at a **meta-analytics layer**, auditing the *experimentation system* rather than the metric outcome.

---

## What the Engine Evaluates

### Pre-experiment integrity
- Randomization balance
- Baseline equivalence
- Traffic / device / geo parity

### In-experiment integrity
- Sample Ratio Mismatch (SRM)
- Exposure leakage (assigned vs exposed)
- Not-exposed users
- Concurrent experiment contamination
- Metric volatility and drift

### Post-experiment rigor
- Bootstrap confidence-interval stability
- Early vs late sensitivity
- Reliability scoring with explicit penalties

---


## What the Engine Does *Not* Do

- Does **not** compute lift or declare winners
- Does **not** optimize KPIs
- Does **not** replace experimentation platforms

EIE is intentionally scoped to **trust assessment**, not performance interpretation.

---

## Reliability Output

Each Audit Produces:

- **Reliability score (0–100)**
- **Reliability label**: HIGH / MODERATE / LOW / UNSAFE
- **Explicit penalty breakdown**
- **Top integrity risks**
- **Coverage disclosure** (what was evaluated vs not evaluated)

Example output:

> *“This experiment shows a 2% lift, but reliability is LOW due to SRM and traffic imbalance. Interpretation is unsafe.”*

---

## Audit Modes

EIE supports **three audit modes**, depending on data availability.

### 1) Upload Experiment Data (Recommended)
**Full integrity audit using user-level data**

**Inputs**
- `exp_users.csv` (required)
- `exp_daily.csv` (optional; computed if missing)

**Evaluates**
- All integrity dimensions, including exposure and contamination

**Use when**
- You have access to raw experiment logs

---

### 2) Enter Experiment Details Manually (Summary Audit)
**Scoped audit when raw data is unavailable**

**Inputs**
- Intended vs observed traffic split
- Users and conversions by variant
- Optional device / geo / source distributions
- Optional daily aggregates

**Evaluates**
- SRM
- Covariate parity (if provided)
- CI stability
- Volatility (if daily data exists)

**Notes**
- User-level checks (leakage, contamination) are marked *Not Evaluated*
- Reliability score is explicitly capped to reflect reduced coverage

---

### 3) Simulate Integrity Scenarios (Advanced / Learn Mode)
**Explore how integrity failures affect trust**

**Inputs**
- Synthetic controls for SRM, imbalance, leakage, contamination, drift

**Use when**
- Learning
- Demonstration
- Stress-testing integrity assumptions

---

## Architecture Overview

- **Core Engine**: Modular integrity checks + scoring
- **Input Adapters**: CSV validation, summary inputs, simulation
- **Scoring Layer**: Weighted penalties + mode-aware caps
- **UI**: Streamlit app with a single integrity-check flow

All audit modes converge into the same scoring and reporting framework.

---

## Technology Stack

- Python
- Pandas, NumPy, SciPy
- Pydantic (input validation)
- Streamlit (UI)
- Modular scoring and diagnostics architecture

---

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate    # On macOS / Linux
# .venv\Scripts\activate     # On Windows

pip install -r requirements.txt
PYTHONPATH=src streamlit run app.py
```
