import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def dashboard(data):
    """
    Display a dashboard with key insights and visualizations.
    """
    st.header("Dashboard")
    st.write(
        "This dashboard provides key insights and visualizations to help you understand the credit risk dataset. "
        "Explore the distributions, trends, and relationships in the data."
    )

    # Credit Risk Distribution
    st.write("### Credit Risk Distribution")
    st.write(
        "The bar chart below shows the distribution of customers by credit risk class. "
        "This helps you understand the proportion of 'good' and 'bad' credit risks in the dataset."
    )
    risk_distribution = data["class"].value_counts()
    st.bar_chart(risk_distribution)

    # Add insights
    st.write("#### Insights")
    st.write(
        f"- **Good Credit Risks**: {risk_distribution.get('good', 0)} customers are classified as 'good' credit risks. "
        "These customers are likely to repay their loans on time."
    )
    st.write(
        f"- **Bad Credit Risks**: {risk_distribution.get('bad', 0)} customers are classified as 'bad' credit risks. "
        "These customers may have difficulty repaying their loans."
    )

    # Credit Amount Distribution
    st.write("### Credit Amount Distribution")
    st.write(
        "The histogram below shows the distribution of credit amounts requested by customers. "
        "This helps you understand the range and frequency of loan amounts."
    )
    fig, ax = plt.subplots()
    sns.histplot(data["credit_amount"], bins=20, kde=True, ax=ax, color="skyblue")
    ax.set_xlabel("Credit Amount")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Credit Amounts")
    st.pyplot(fig)

    # Add insights
    st.write("#### Insights")
    st.write(
        "- Most customers request credit amounts in the **lower range** (e.g., below $5,000). "
        "This may indicate a preference for smaller, short-term loans."
    )
    st.write(
        "- A smaller number of customers request **larger credit amounts** (e.g., above $10,000). "
        "These loans may require stricter eligibility criteria."
    )

    # Age Distribution
    st.write("### Age Distribution")
    st.write(
        "The histogram below shows the distribution of customer ages. "
        "This helps you understand the demographic profile of customers."
    )
    fig, ax = plt.subplots()
    sns.histplot(data["age"], bins=20, kde=True, ax=ax, color="salmon")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Customer Ages")
    st.pyplot(fig)

    # Add insights
    st.write("#### Insights")
    st.write(
        "- Most customers are in the **age range of 20-40 years**. "
        "This may indicate a younger customer base seeking credit for personal or business needs."
    )
    st.write(
        "- Older customers (e.g., above 50 years) are less represented. "
        "This may reflect lower credit demand or stricter eligibility criteria for older individuals."
    )

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    st.write(
        "The heatmap below shows the correlation between numeric features in the dataset. "
        "This helps you understand relationships between variables, such as credit amount and age."
    )
    numeric_data = data.select_dtypes(include=[float, int])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    # Add insights
    st.write("#### Insights")
    st.write(
        "- **Strong Positive Correlation**: Variables with a correlation close to 1 indicate a strong positive relationship. "
        "For example, credit amount and loan duration may be positively correlated."
    )
    st.write(
        "- **Strong Negative Correlation**: Variables with a correlation close to -1 indicate a strong negative relationship. "
        "For example, age and credit risk may be negatively correlated."
    )

    # Key Statistics
    st.write("### Key Statistics")
    st.write(
        "The table below summarizes key statistics for numeric features in the dataset. "
        "This provides a quick overview of the data distribution."
    )
    st.write(numeric_data.describe())

    # Conclusion
    st.write("### Conclusion")
    st.write(
        "This dashboard provides a comprehensive overview of the credit risk dataset. "
        "Use the visualizations and insights above to understand customer behavior, identify trends, and make data-driven decisions."
    )