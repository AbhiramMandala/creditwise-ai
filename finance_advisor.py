import streamlit as st
import matplotlib.pyplot as plt

def personal_finance_advisor():
    st.header("🌟 Personal Finance Advisor")
    st.write("Get personalized financial advice based on your income, expenses, and goals.")

    # Input fields for user's financial details
    st.subheader("💰 Your Financial Details")
    income = st.number_input("Monthly Income", min_value=0, value=5000)
    expenses = st.number_input("Monthly Expenses", min_value=0, value=3000)
    savings = st.number_input("Current Savings", min_value=0, value=10000)
    debt = st.number_input("Total Debt", min_value=0, value=5000)
    age = st.number_input("Your Age", min_value=18, max_value=100, value=30)
    retirement_age = st.number_input("Planned Retirement Age", min_value=age, max_value=100, value=65)

    # Expense breakdown (optional but useful)
    st.subheader("📊 Expense Breakdown")
    rent = st.number_input("Rent/Mortgage", min_value=0, value=1200)
    food = st.number_input("Food & Groceries", min_value=0, value=600)
    entertainment = st.number_input("Entertainment & Shopping", min_value=0, value=300)
    insurance = st.number_input("Insurance", min_value=0, value=400)
    others = expenses - (rent + food + entertainment + insurance)

    # Financial goals
    st.subheader("🎯 Your Financial Goals")
    emergency_fund = st.number_input("Emergency Fund Goal", min_value=0, value=10000)
    vacation_goal = st.number_input("Vacation Savings Goal", min_value=0, value=5000)
    home_down_payment = st.number_input("Home Down Payment Goal", min_value=0, value=20000)

    if st.button("💡 Get Financial Advice"):
        # Monthly savings
        monthly_savings = income - expenses
        st.write(f"### 💵 Monthly Savings: **${monthly_savings:,.2f}**")

        # Budget Planning
        st.write("### 📑 Budget Planning")
        if monthly_savings > 0:
            st.success("✅ You are saving money each month. Great job!")
        else:
            st.error("❌ You are spending more than you earn. Consider reducing your expenses.")

        # Savings Goals Progress
        st.write("### 🚀 Savings Goal Progress")
        st.write(f"**Emergency Fund:**")
        st.progress(min(savings / emergency_fund, 1.0))

        st.write(f"**Vacation Fund:**")
        st.progress(0)  # Starts from 0, assuming no vacation savings saved yet.

        st.write(f"**Home Down Payment:**")
        st.progress(0)

        # Expense Breakdown Chart
        st.write("### 📊 Expense Breakdown")
        labels = ['Rent', 'Food', 'Entertainment', 'Insurance', 'Others']
        values = [rent, food, entertainment, insurance, others]
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

        # Recommended Budget Comparison
        st.write("### 📏 Recommended Budget Allocation (50/30/20 Rule)")
        needs = income * 0.50
        wants = income * 0.30
        savings_recommended = income * 0.20

        actual_needs = rent + food + insurance
        actual_wants = entertainment + others
        actual_savings = monthly_savings

        st.write(f"✅ **Needs (50%) Recommended: ${needs:,.2f} - Actual: ${actual_needs:,.2f}**")
        st.write(f"✅ **Wants (30%) Recommended: ${wants:,.2f} - Actual: ${actual_wants:,.2f}**")
        st.write(f"✅ **Savings (20%) Recommended: ${savings_recommended:,.2f} - Actual: ${actual_savings:,.2f}**")

        # Debt Management
        st.write("### 💳 Debt Management")
        if debt > 0:
            st.warning(f"You have **${debt:,.2f}** in debt. Consider strategies like:")
            st.write("- ❄️ Snowball Method: Pay off small debts first.")
            st.write("- ⛷️ Avalanche Method: Pay off high-interest debts first.")
        else:
            st.success("✅ You are debt-free. Great job!")

        # Inflation-Adjusted Retirement Planning
        st.write("### 🏖️ Retirement Planning (with 3% inflation)")
        years_to_retirement = retirement_age - age
        future_income = income * (1.03 ** years_to_retirement)
        retirement_savings_needed = future_income * 12 * (100 - retirement_age)
        st.write(f"You have **{years_to_retirement} years** until retirement.")
        st.write(f"💡 You need approximately **${retirement_savings_needed:,.2f}** to retire comfortably (adjusted for inflation).")

        # Financial Health Score
        st.write("### ❤️ Financial Health Score")
        savings_rate = (savings / income) * 100
        debt_to_income_ratio = (debt / income) * 100

        score = 100
        if savings_rate < 20:
            score -= 20
        if debt_to_income_ratio > 40:
            score -= 30
        if savings < emergency_fund:
            score -= 20

        st.metric("🏅 Financial Health Score", score)

        # Future Wealth Simulator
        st.write("### 🔮 Future Wealth Simulation (Assuming 6% annual return)")
        future_years = st.slider("Years to simulate", 1, years_to_retirement, 10)
        future_savings = savings
        future_values = []
        for _ in range(future_years):
            future_savings = future_savings * 1.06 + (monthly_savings * 12)
            future_values.append(future_savings)

        st.line_chart(future_values)

        # Gamification: Achievements
        st.write("### 🎖️ Achievements Unlocked")
        if savings >= 10000:
            st.write("✅ **Saved First $10,000**")
        if debt == 0:
            st.write("✅ **Debt-Free Champion**")
        if savings_rate >= 30:
            st.write("✅ **Savings Master (Savings Rate > 30%)**")

        # Helpful Resources
        st.write("### 📚 Helpful Financial Resources")
        st.markdown("[Explore Best Savings Accounts](https://www.bankrate.com/banking/savings/best-savings-accounts/)")
        st.markdown("[Debt Payoff Calculators](https://www.bankrate.com/calculators/managing-debt/debt-payoff-calculator.aspx)")
        st.markdown("[Investment Basics for Beginners](https://www.investopedia.com/investing-4427775)")

