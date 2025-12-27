from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Pydantic: schema definitions
class ExpUsersSchema(BaseModel):
    '''
    Column contract for exp_users. 
    Required columns are the minimal  set needed for core integrity checks.
    Optional columns unlock more checks (exposure, comtamination, parity, baseline)
    '''
    required: List[str] = Field(default_factory = lambda: [
        "experiment_id", 
        "user_id",
        "variant_assigned",
        "assignment_ts",
        "converted",
    ])
    optional: List[str] = Field(default_factory = lambda: [
        "variant_exposed",
        "first_exposure_ts",
        "device_type",
        "geo",
        "traffic_source",
        "in_other_experiment",
        "baseline_converted",
    ])
    allowed_variants: List[str] = Field(default_factory = lambda: ["A", "B"])

    @field_validator("required", "optional")
    @classmethod
    def _no_duplicates(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("Duplicate column names in schema definition.")
        return v
    
class ExpDailySchema(BaseModel):
    '''
    Column contract for exp_daily (optional csv file)
    If absent, program can compute the integrity using exp_users.
    '''
    required: List[str] = Field(default_factory = lambda: [
        "experiment_id",
        "date",
        "variant_assigned",
        "assigned_users",
        "conversions",
    ])
    optional: List[str] = Field(default_factory= lambda: [
        "exposed_users",
        "conversion_rate",
    ])
    allowed_variants: List[str] = Field(default_factory= lambda: ["A", "B"])

# Dataclasses: Validation Result
@dataclass
class ValidationErrorItem:
    code: str
    message: str
    column: Optional[str] = None

@dataclass
class ValidationResult:
    ok: bool
    errors: List[ValidationErrorItem]
    warnings: list[ValidationErrorItem]

# Helpers
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalize column names: strip, lower, spaces to underscores
    This makes uploads tolerant to minor formatting differences
    '''
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out

def _coerce_bool01(series: pd.Series) -> pd.Series:
    '''
    Coerce to 0/1 ints when possible.
    Accepts: 0/1, True/False, '0'/'1', 'true'/'false', 'yes','no'
    '''
    s = series.copy()
    # If numeric, keep 0/1
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int)
    # Strings
    s = s.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1, "0": 0,
        "true": 1, "false": 0, 
        "yes": 1, "no":0, 
        "y": 1, "n":0,
        "t": 1, "f":0,
    }
    return s.map(mapping)

def _validate_variants(series: pd.Series, allowed: List[str]) -> Tuple[bool, Optional[str]]:
    bad = series.dropna().astype(str).str.upper().loc[~series.dropna().astype(str).str.upper().isin(allowed)]
    if len(bad) > 0:
        sample = ", ".join(bad.unique()[:5])
        return False, f"Invalid variant values found (sample: {sample}. Allowed: {allowed}."
    return True, None

def _safe_to_datetime(series: pd.Series) -> Tuple[pd.Series, Optional[str]]:
    dt = pd.to_datetime(series, errors = 'coerce', utc = False)
    if dt.isna().mean() >0.25: # too many failed parses
        return dt, "More than 25% of the timestamps could not be parsed. Ensure ISO-like formats."
    return dt, None

# Public API
def validate_exp_users(df_raw: pd.DataFrame, schema: Optional[ExpUsersSchema] = None) -> Tuple[pd.DataFrame, ValidationResult]:
    schema = schema or ExpUsersSchema()
    errors: List[ValidationErrorItem] = []
    warnings: List[ValidationErrorItem] = []

    df = _normalize_columns(df_raw)

    missing = [c for c in schema.required if c not in df.columns]
    if missing:
        errors.append(ValidationErrorItem(
            code = "MISSING_REQUIRED_COLUMNS",
            message = f"Missing required columns: {missing}",
        ))
        return df, ValidationResult(ok = False, errors = errors, warnings = warnings)
    
    # Variant assigned
    df["variant_assigned"] = df["variant_assigned"].astype(str).str.upper().str.strip()
    okv, msg = _validate_variants(df["variant_assigned"], schema.allowed_variants)
    if not okv:
        errors.append(ValidationErrorItem(
            code = "INVALID_VARIANT_ASSIGNED", 
            message = msg, 
            column = "variant_assigned"
        ))
    
    # Converted
    converted = _coerce_bool01(df["converted"])
    if converted.isna().any():
        errors.append(ValidationErrorItem(
            code = "INVALID_CONVERTED",
            message = "converted must be 0/1 or boolean-like values (true/false).",
            column = "converted",
        ))
    else:
        # enforce int 0/1
        conv_int = converted.fillna(0).astype(int)
        if not conv_int.isin([0, 1]).all():
            errors.append(ValidationErrorItem(
                code = "INVALID_CONVETED_RANGE",
                message = "converted contains values outside {0, 1}.",
                column = "converted",
            ))
        df["converted"] = conv_int
    
    # Optional: exposure columns (validate only if present)
    if "variant_exposed" in df.columns:
        df["variant_exposed"] = df["variant_exposed"].replace({None: pd.NA})
        vexp = df["variant_exposed"].dropna().astype(str).str.upper().str.strip()
        okx, msgx = _validate_variants(vexp, schema.allowed_variants)
        if not okx:
            errors.append(ValidationErrorItem(
                code = "INVALID_VARIANT_EXPOSED",
                message = msgx,
                column = "variant_exposed"
            ))
        df["variant_exposed"] = df["variant_exposed"].astype("string")
    if "first_exposure_ts" in df.columns:
        dt2, warn2 = _safe_to_datetime(df["first_exposure_ts"])
        df["first_exposure_ts"] = dt2
        if warn2:
            warnings.append(ValidationErrorItem(
                code="BAD_FIRST_EXPOSURE_TS", 
                message=warn2, 
                column="first_exposure_ts"
            ))
    
    # Optional: in_other_experiment
    if "in_other_experiment" in df.columns:
        tmp = _coerce_bool01(df["in_other_experiment"])
        if tmp.isna().any():
            warnings.append(ValidationErrorItem(
                code = "BAD_IN_OTHER_EXPERIMENT",
                message = "in_other_experiment could not be fully parsed into 0/1. Treating unparseable as 0.",
                column = "in_other_experiment"
            ))
        tmp = tmp.fillna(0)
        df["in_other_experiment"] = tmp.astype(int).clip(0, 1)
    
    # Optional: baseline_converted
    if "baseline_converted" in df.columns:
        tmp = _coerce_bool01(df["baseline_converted"])
        if tmp.isna().any():
            warnings.append(ValidationErrorItem(
                code = "BAD_BASELINE_CONVERTED",
                message = "baseline_converted coult not be fully parsed into 0/1. Treateing unparseable as 0.",
                column = "baseline_converted"
            ))
        tmp = tmp.fillna(0)
        df["baseline_converted"] = tmp.astype(int).clip(0, 1)
    
    # Duplicate rows (same experiment_id & user_id) is usually a data bug
    dup_key  = df.duplicated(subset=["experiment_id", "user_id"], keep = False)
    if dup_key.any():
        warnings.append(ValidationErrorItem(
            code = "DUPLICATE_EXPERIMENT_USER",
            message = f"FOUND {int(dup_key.sum())} rows with duplicate (experiment_id, user_id). Consider de-duplicating.",
        ))
    ok = len(errors) == 0
    return df, ValidationResult( ok = ok, errors = errors, warnings = warnings)

def validate_exp_daily(df_raw: pd.DataFrame, schema: Optional[ExpDailySchema] = None) -> Tuple[pd.DataFrame, ValidationResult]:
    schema = schema or ExpDailySchema()
    errors: List[ValidationErrorItem] = []
    warnings: List[ValidationErrorItem] = []

    df = _normalize_columns(df_raw)

    missing = [c for c in schema.required if c not in df.columns]
    if missing:
        errors.append(ValidationErrorItem(
            code = "MISSING_REQUIRED_COLUMNS",
            message = f"Missing required columns: {missing}",
        ))
        return df, ValidationResult(ok = False, errors = errors, warnings = warnings)

    df["variant_assigned"] = df["variant_assigned"].astype(str).str.upper().str.strip()
    okv, msg = _validate_variants(df["variant_assigned"], schema.allowed_variants)
    if not okv:
        errors.append(ValidationErrorItem(code="INVALID_VARIANT_ASSIGNED", message=msg, column="variant_assigned"))
    
    # date
    df["date"] = pd.to_datetime(df["date"], errors = "coerce")
    if df["date"].isna().mean() > 0.25:
        errors.append(ValidationErrorItem(
            code = "BAD_DATE", 
            message = "Mote than 25% of the dates could not be parsed.",
            column = "date"
        ))
    else:
        df["date"] = df["date"].dt.date
    
    # numeric counts 
    for col in ["assigned_users", "conversions"]:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')
        if df[col].isna().any():
            errors.append(ValidationErrorItem(
                code = "BAD_NUMERIC",
                message = f"{col} must be numeric.",
                column = col
            ))
        else:
            df[col] = df[col].astype(int)
    
    if "exposed_users" in df.columns:
        df["exposed_users"] = pd.to_numeric(df["exposed_users"], errors = "coerce")
        if df["exposed_users"].isna().any():
            warnings.append(ValidationErrorItem(
                code = "BAD_EXPOSED_USERS", 
                message = "exposed_users had non-numeric values; ignoring.", 
                column = "exposed_users"
            ))
        else:
            df["exposed_users"] = df["exposed_users"].astype(int)

    
    if "conversion_rate" not in df.columns:
        df["conversion_rate"] = df["conversions"] / df["assigned_users"].clip(lower=1)
    else:
        df["conversion_rate"] = pd.to_numeric(df["conversion_rate"], errors = "coerce")
        if df["conversion_rate"].isna().any():
            warnings.append(ValidationErrorItem(
                code = "BAD_CONVERSION_RATE",
                message = "conversion_rate had invalid values; recomputing.",
                column = "conversion_rate",
            )) 
    ok = len(errors) == 0
    return df, ValidationResult(ok = ok, errors = errors, warnings = warnings)


def exp_users_template(n_rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "experiment_id": ["exp_001"] * n_rows,
        "user_id": [f"u_{i:04d}" for i in range(n_rows)],
        "variant_assigned": ["A", "B", "A", "B", "A"][:n_rows],
        "assignment_ts": ["2025-01-01 10:00:00"] * n_rows,
        "converted": [0, 1, 0, 0, 1][:n_rows],
        # Optional columns below (keep if you want exposure checks)
        "variant_exposed": ["A", "B", None, "B", "A"][:n_rows],
        "first_exposure_ts": ["2025-01-01 10:00:05"] * n_rows,
        "device_type": ["mobile", "desktop", "mobile", "desktop", "mobile"][:n_rows],
        "geo": ["US", "US", "CA", "US", "IN"][:n_rows],
        "traffic_source": ["organic", "paid", "organic", "organic", "email"][:n_rows],
        "in_other_experiment": [0, 0, 1, 0, 0][:n_rows],
        "baseline_converted": [0, 0, 0, 1, 0][:n_rows],
        })

def exp_daily_template(n_rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "experiment_id": ["exp_001"]*n_rows,
        "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"][:n_rows],
        "variant_assigned": ["A", "B", "A", "B"][:n_rows],
        "assigned_users":[1000, 1000, 1000, 1000][:n_rows],
        "exposed_users":[950, 960, 940, 955][:n_rows],
        "conversions":[45,48,44,50][:n_rows],
        "conversion_rate":[0.045, 0.048, 0.044, 0.050][:n_rows],
    })

        