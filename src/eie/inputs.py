from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

Variant = Literal["A", "B"]

class VariantCounts(BaseModel):
    '''
    Counts by variant. Used for assignemnts and conversions.
    '''
    A: int = Field(..., ge = 0)
    B: int = Field(..., ge = 0)

    @field_validator("A", "B")
    @classmethod
    def _int(cls, v: int) -> int:
        return int(v)
    
    def total(self) -> int:
        return int(self.A + self.B)
    
class ExpectedSplit(BaseModel):
    '''
    Intended traffic split (weights).
    Example: A = 0.8, B = 0.2 or A = 0.5, B = 0.5
    '''
    A: float = Field(..., ge = 0)
    B: float = Field(..., ge = 0) 
    @field_validator("B")
    @classmethod
    def _nonzero_sum(cls, v: float, info):
        a = float(info.data.get("A", 0.0))
        if a + float(v) <= 0:
            raise ValueError("Expected split A+B must be > 0.")
        return float(v)
    def normalize(self) -> "ExpectedSplit":
        s = float(self.A + self.B)
        return ExpectedSplit (A = float(self.A / s), B = float(self.B / s))

class DistCounts(BaseModel):
    '''
    Distribution counts for a categorical covariate, for each variant. 
    Example: {mobile: {"A": 1200, "B": 1100}, "desktop":{"A": 800, "B": 900}}
    '''
    categories: Dict[str, VariantCounts]
    def categories_list(self) -> List[str]:
        return list(self.categories.keys())

 
class ManualDailyRow(BaseModel):
    '''
    Optional daily aggregates for volatility/drift checks. 
    Provide assigned_users adn conversions by variant for each date.
    '''
    date: str # YYYY-MM-DD 
    assigned: VariantCounts
    conversions: VariantCounts


class ManualIntegrityInputs(BaseModel):
    '''
    Manual/Summary mode inputs.
    This mode is designed for users who cannot upload user-level rows. 
    It produces a *scoped* integrity audit (some checks not evaluated).
    '''
    experiment_id: str = Field(default = "manual_exp")
    expected_split: ExpectedSplit = Field(default_factory = lambda: ExpectedSplit(A = 0.5, B = 0.5))
    
    observed_assigned: VariantCounts
    observed_conversions: VariantCounts

    # Optional: Expected vs Observed covariate distributions (counts)
    expected_device: Optional[Dict[str, float]] = None # weights (sum doesn't have to be 1)
    observed_device: Optional[DistCounts] = None

    expected_geo: Optional[Dict[str, float]] = None
    observed_geo: Optional[DistCounts] = None

    expected_source: Optional[Dict[str, float]] = None
    observed_source: Optional[DistCounts] = None

    # Optional daily series for valitility/drift checks
    daily: Optional[List[ManualDailyRow]] = None

    @field_validator("observed_conversions")
    @classmethod
    def _conversions_not_exceed_assigned(cls, v: VariantCounts, info):
        assigned: VariantCounts = info.data.get("onberved_assigned")
        if assigned is None:
            return v
        if v.A > assigned.A or v.B > assigned.B:
            raise ValueError("Conversions cannot exceed assigned users for A or B.")
    
    @field_validator("expected_device", "expected_geo", "expected_source")
    @classmethod
    def _expected_weights_nonnegative(cls, v):
        if v is None:
            return v
        for k, w in v.items():
            if float(w) < 0:
                raise ValueError(f"Expected weight for '{k}' must be >= 0.")
            if sum(float(x) for x in v.values()) <=0:
                raise ValueError("Expected distribution weights must sum to > 0.")
            return v
