import streamlit as st

def loan_recommendations():
    """
    Provide personalized loan recommendations based on customer profiles.
    """
    st.header("Loan Product Recommendations")
    st.write(
        "This section suggests loan products tailored to your financial needs and profile. "
        "Please provide some details to receive personalized recommendations."
    )

    # Input fields for customer details
    st.subheader("Your Financial Profile")
    income = st.number_input("Monthly Income (in USD)", min_value=0, value=3000, 
                             help="Enter your monthly income after taxes.")
    expenses = st.number_input("Monthly Expenses (in USD)", min_value=0, value=2000, 
                               help="Enter your total monthly expenses.")
    savings = st.number_input("Current Savings (in USD)", min_value=0, value=5000, 
                              help="Enter the amount you currently have in savings.")
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700, 
                             help="Select your credit score. A higher score improves your loan eligibility.")
    loan_purpose = st.selectbox("Purpose of Loan", 
                                ["Personal Use", "Home Purchase/Renovation", "Car Purchase", "Education", "Business"], 
                                help="Select the purpose of the loan.")

    # Calculate disposable income
    disposable_income = income - expenses

    # Loan eligibility criteria
    st.subheader("Loan Eligibility")
    st.write(
        "Based on your financial profile, here's an assessment of your loan eligibility:"
    )

    if disposable_income > 1000 and credit_score >= 650:
        st.success("✅ You are eligible for most loan products.")
    elif disposable_income > 500 and credit_score >= 600:
        st.warning("⚠️ You are eligible for some loan products, but with higher interest rates.")
    else:
        st.error("❌ You may not be eligible for most loan products at this time.")

    # Personalized loan recommendations
    st.subheader("Personalized Loan Recommendations")
    st.write(
        "Based on your financial profile and loan purpose, here are some recommended loan products:"
    )

    if loan_purpose == "Personal Use":
        st.write("- **Personal Loan**: Ideal for short-term financial needs like vacations, weddings, or medical expenses.")
        st.write("- **Line of Credit**: Flexible borrowing option for ongoing expenses.")
        st.write(
            f"**Eligibility**: You can borrow up to **${disposable_income * 12:.2f}** (12 months of disposable income)."
        )

    elif loan_purpose == "Home Purchase/Renovation":
        st.write("- **Home Loan**: Best for purchasing or renovating a home.")
        st.write("- **Home Equity Loan**: Use the equity in your home to borrow at lower interest rates.")
        st.write(
            f"**Eligibility**: You can borrow up to **${savings * 5:.2f}** (5 times your savings)."
        )

    elif loan_purpose == "Car Purchase":
        st.write("- **Car Loan**: Perfect for buying a new or used car.")
        st.write("- **Lease Financing**: Flexible option for driving a new car without owning it.")
        st.write(
            f"**Eligibility**: You can borrow up to **${disposable_income * 24:.2f}** (24 months of disposable income)."
        )

    elif loan_purpose == "Education":
        st.write("- **Education Loan**: Designed to cover tuition fees and other educational expenses.")
        st.write("- **Scholarship/Grant**: Explore free funding options for education.")
        st.write(
            f"**Eligibility**: You can borrow up to **${savings * 3:.2f}** (3 times your savings)."
        )

    elif loan_purpose == "Business":
        st.write("- **Business Loan**: Ideal for starting or expanding a business.")
        st.write("- **Small Business Grant**: Explore free funding options for small businesses.")
        st.write(
            f"**Eligibility**: You can borrow up to **${disposable_income * 36:.2f}** (36 months of disposable income)."
        )

    # Tips for improving eligibility
    st.subheader("Tips to Improve Loan Eligibility")
    st.write(
        "If you're not eligible for your desired loan, here are some tips to improve your chances:"
    )
    st.write("- **Increase Savings**: Save more to improve your financial stability.")
    st.write("- **Reduce Expenses**: Lower your monthly expenses to increase disposable income.")
    st.write("- **Improve Credit Score**: Pay bills on time and reduce outstanding debt to boost your credit score.")

    # Conclusion
    st.write("### Conclusion")
    st.write(
        "Personalized loan recommendations help you find the best loan products for your needs. "
        "Use the insights and tips above to improve your financial profile and loan eligibility."
    )