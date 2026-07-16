import sys
from pathlib import Path

import streamlit as st
from data_loader import load_data
from credit_risk_model import credit_risk_prediction
from clustering_model import customer_segmentation
from anomaly_detection import anomaly_detection
from fairness_analysis import fairness_analysis
from loan_recommendations import loan_recommendations
from dashboard import dashboard
from finance_advisor import personal_finance_advisor
from fn import Personal_Finance_Advisor_with_Macroeconomic_Insights

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Streamlit App
def main():
    st.title("Credit Risk Management System")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = ["Home", "Credit Risk Prediction", "Customer Segmentation", "Anomaly Detection", 
               "Fairness Analysis", "Loan Recommendations", "Dashboard", "Personal Finance Advisor","Personal_Finance_Advisor_with_Macroeconomic_Insights"]
    choice = st.sidebar.selectbox("Choose a section", options)

    # Load data
    data = load_data()

    if choice == "Home":
        st.header("Welcome to the Credit Risk Management System")
        st.write("This system helps you predict credit risk, segment customers, detect anomalies, and more.")
        st.write("Use the sidebar to navigate through the features.")
        st.write("### Features:")
        st.write("- **Credit Risk Prediction**: Predict whether a customer is a good or bad credit risk.")
        st.write("- **Customer Segmentation**: Group customers into clusters based on credit behavior.")
        st.write("- **Anomaly Detection**: Identify unusual or fraudulent loan applications.")
        st.write("- **Fairness Analysis**: Analyze potential biases in the model.")
        st.write("- **Loan Recommendations**: Suggest loan products based on customer profiles.")
        st.write("- **Dashboard**: Visualize key insights and distributions.")
        st.write("- **Personal Finance Advisor**: Get personalized financial advice.")
        st.write("- **Personal Finance Advisor_1**: Get economic insights.")

    elif choice == "Credit Risk Prediction":
        credit_risk_prediction(data)

    elif choice == "Customer Segmentation":
        customer_segmentation(data)

    elif choice == "Anomaly Detection":
        anomaly_detection(data)

    elif choice == "Fairness Analysis":
        fairness_analysis(data)

    elif choice == "Loan Recommendations":
        loan_recommendations()

    elif choice == "Dashboard":
        dashboard(data)

    elif choice == "Personal Finance Advisor":
        personal_finance_advisor()
        
    # elif choice == "Personal Finance Advisor_1":
    #     Personal_Finance_Advisor_with_Macroeconomic_Insights()    

if __name__ == "__main__":
    main()