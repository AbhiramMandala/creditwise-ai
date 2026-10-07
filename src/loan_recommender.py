"""Transparent loan recommendation engine (decision support, NOT approval).

Inputs: predicted P(good), monthly income/expenses, requested amount, cluster.
Rules are explicit and every recommendation carries its reason string.
Affordability: monthly instalment ≈ amount/duration; burden = instalment/income.
"""
from __future__ import annotations


def recommend_loan(
    p_good: float,
    monthly_income: float,
    monthly_expenses: float,
    requested_amount: float,
    duration_months: int = 24,
    cluster_label: int | None = None,
) -> dict:
    disclaim = (
        "Decision-support suggestion only — not a loan approval or financial advice."
    )
    if monthly_income <= 0:
        return {
            "decision": "insufficient_data",
            "products": [],
            "max_affordable_amount": 0.0,
            "reasons": ["Monthly income must be positive to assess affordability."],
            "disclaimer": disclaim,
        }
    duration_months = max(int(duration_months), 1)
    disposable = monthly_income - monthly_expenses
    burden = (requested_amount / duration_months) / monthly_income
    max_affordable = max(disposable * duration_months * 0.4, 0.0)

    reasons = [
        f"Predicted repayment likelihood {p_good:.0%}; burden {burden:.0%} of income.",
        f"Disposable income ${disposable:,.0f}/mo supports ≈ ${max_affordable:,.0f} over {duration_months}mo.",
    ]
    if cluster_label is not None:
        reasons.append(f"Customer segment #{cluster_label} considered.")

    if p_good >= 0.7 and burden <= 0.30 and disposable > 0:
        decision = "recommend"
        products = ["Standard term loan", "Low-rate personal loan"]
        reasons.append("Low predicted risk + affordable burden → recommend.")
    elif p_good >= 0.5 and burden <= 0.40:
        decision = "conditional"
        products = ["Secured loan", "Shorter-term loan with review"]
        reasons.append("Moderate risk → smaller amount or collateral advised.")
    else:
        decision = "not_recommended"
        products = ["Financial counselling", "Savings plan first"]
        reasons.append("High predicted risk or unaffordable burden → do not recommend now.")
    return {
        "decision": decision,
        "products": products,
        "max_affordable_amount": round(float(max_affordable), 2),
        "burden_ratio": round(float(burden), 3),
        "reasons": reasons,
        "disclaimer": disclaim,
    }
