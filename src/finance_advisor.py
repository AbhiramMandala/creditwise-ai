"""Rule-based personal finance insight generator (kept from legacy finance_advisor.py).

Deterministic 50/30/20 + savings-rate + debt-to-income rules. No external data,
no fake macro forecasts — the legacy 'macroeconomic insights' import (fn module)
did not exist in the repo and is dropped.
"""
from __future__ import annotations


def finance_health(
    income: float, expenses: float, savings: float, debt: float, emergency_goal: float = 10000
) -> dict:
    monthly_savings = income - expenses
    savings_rate = (monthly_savings / income) if income else 0.0
    dti = (debt / income) if income else 0.0
    score = 100
    notes = []
    if savings_rate < 0.20:
        score -= 20
        notes.append("Savings rate below 20% — trim wants or automate transfers.")
    if dti > 0.40:
        score -= 30
        notes.append("Debt-to-income above 40% — prioritise high-interest debt.")
    if savings < emergency_goal:
        score -= 20
        notes.append("Emergency fund below goal — build 3–6 months of expenses.")
    if not notes:
        notes.append("Solid budgeting: keep emergency fund topped up.")
    needs, wants, save_rec = income * 0.5, income * 0.3, income * 0.2
    return {
        "monthly_savings": round(monthly_savings, 2),
        "savings_rate": round(savings_rate, 3),
        "debt_to_income": round(dti, 3),
        "health_score": max(score, 0),
        "budget_50_30_20": {
            "needs": round(needs, 2),
            "wants": round(wants, 2),
            "savings": round(save_rec, 2),
        },
        "notes": notes,
        "disclaimer": "Educational insights only — not professional financial advice.",
    }
