"""Pydantic request/response schemas — every endpoint validates input."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src import config


class CustomerRecord(BaseModel):
    duration: float = Field(..., ge=0, le=120)
    credit_amount: float = Field(..., ge=0, le=200000)
    installment_commitment: float = Field(..., ge=0, le=100)
    residence_since: float = Field(..., ge=0, le=50)
    age: float = Field(..., ge=18, le=100)
    existing_credits: float = Field(..., ge=0, le=20)
    num_dependents: float = Field(..., ge=0, le=20)
    checking_status: str
    credit_history: str
    purpose: str
    savings_status: str
    employment: str
    personal_status: str
    other_parties: str
    property_magnitude: str
    other_payment_plans: str
    housing: str
    job: str
    own_telephone: str
    foreign_worker: str


class RiskResponse(BaseModel):
    label: Literal["good", "bad"]
    risk_level: str
    p_good: float
    p_bad: float
    explanations: list[dict]


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    verdict: str


class SegmentResponse(BaseModel):
    cluster: int
    profile: dict


class LoanRequest(BaseModel):
    p_good: float = Field(..., ge=0, le=1)
    monthly_income: float = Field(..., ge=0)
    monthly_expenses: float = Field(..., ge=0)
    requested_amount: float = Field(..., ge=0)
    duration_months: int = Field(24, ge=1, le=120)
    cluster_label: int | None = None


class HealthResponse(BaseModel):
    status: str
    model_version: str = config.MODEL_VERSION
