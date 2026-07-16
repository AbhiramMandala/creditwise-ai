import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data

def fairness_analysis(data):
    """
    Perform fairness analysis to identify potential biases in the model.
    """
    st.header("Fairness Analysis")
    st.write(
        "This section analyzes potential biases in the model based on sensitive attributes such as gender, age, or employment status. "
        "Understanding these biases is crucial for ensuring fair and ethical decision-making."
    )

    # Convert 'class' to numeric values for fairness analysis
    data["class_numeric"] = data["class"].apply(lambda x: 1 if x == "good" else 0)

    # Bias Analysis by Gender
    if "personal_status" in data.columns:
        st.write("### Bias Analysis by Gender")
        st.write(
            "The bar chart below shows the average credit risk score (probability of being a 'good' credit risk) "
            "for different gender groups. This helps identify any disparities in how the model treats different genders."
        )

        # Group by gender and calculate the mean credit risk score
        bias_analysis = data.groupby("personal_status")["class_numeric"].mean()

        # Display the bar chart
        st.bar_chart(bias_analysis)

        # Add insights
        st.write("#### Insights")
        st.write(
            "- If the average credit risk score varies significantly across genders, it may indicate bias in the model. "
            "- For example, if one gender group consistently receives lower scores, the model may be unfairly disadvantaging that group."
        )

    # Bias Analysis by Age
    if "age" in data.columns:
        st.write("### Bias Analysis by Age")
        st.write(
            "The scatter plot below shows the relationship between age and credit risk score. "
            "This helps identify any age-related biases in the model."
        )

        # Create a scatter plot
        fig, ax = plt.subplots()
        sns.scatterplot(x="age", y="class_numeric", data=data, ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Credit Risk Score")
        ax.set_title("Credit Risk Score by Age")
        st.pyplot(fig)

        # Add insights
        st.write("#### Insights")
        st.write(
            "- If younger or older individuals consistently receive lower scores, the model may be biased against certain age groups. "
            "- Ensure that age is not disproportionately influencing the model's predictions."
        )

    # Bias Analysis by Employment Status
    if "employment" in data.columns:
        st.write("### Bias Analysis by Employment Status")
        st.write(
            "The bar chart below shows the average credit risk score for different employment statuses. "
            "This helps identify any biases related to employment."
        )

        # Group by employment status and calculate the mean credit risk score
        employment_bias = data.groupby("employment")["class_numeric"].mean()

        # Display the bar chart
        st.bar_chart(employment_bias)

        # Add insights
        st.write("#### Insights")
        st.write(
            "- If unemployed individuals consistently receive lower scores, the model may be unfairly penalizing them. "
            "- Consider whether employment status is being used appropriately in the model."
        )

    # General Recommendations
    st.write("### Recommendations for Reducing Bias")
    st.write(
        "If biases are identified, here are some steps you can take to address them:"
    )
    st.write(
        "- **Re-evaluate Features**: Ensure that sensitive attributes (e.g., gender, age) are not disproportionately influencing the model. "
        "- **Use Fairness Metrics**: Incorporate fairness metrics (e.g., demographic parity, equalized odds) into the model evaluation process. "
        "- **Data Augmentation**: Collect more diverse and representative data to reduce biases. "
        "- **Post-Processing**: Apply post-processing techniques to adjust model predictions and ensure fairness."
    )

    # Conclusion
    st.write("### Conclusion")
    st.write(
        "Fairness analysis is essential for building ethical and trustworthy models. "
        "By identifying and addressing biases, you can ensure that your model treats all customers fairly and equitably."
    )