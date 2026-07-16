import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_loader import preprocess_data

def customer_segmentation(data):
    """
    Perform customer segmentation using KMeans clustering and display insights.
    """
    st.header("Customer Segmentation")
    st.write(
        "Group customers into clusters based on their credit behavior. "
        "This helps in understanding different customer groups and tailoring strategies for each group."
    )

    # Preprocess the data
    data = preprocess_data(data)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop("class", axis=1))

    # Train the KMeans model
    st.write("### Training the KMeans Clustering Model")
    st.write(
        "We use the **KMeans clustering algorithm** to group customers into 3 clusters based on their credit behavior. "
        "The data is standardized to ensure all features contribute equally to the clustering process."
    )

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(scaled_data)

    # Assign clusters to the data
    data["Cluster"] = kmeans.predict(scaled_data)

    # Display cluster distribution
    st.write("### Cluster Distribution")
    st.write(
        "The bar chart below shows the number of customers in each cluster. "
        "This helps you understand the size of each customer group."
    )
    cluster_counts = data["Cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    # Display cluster characteristics
    st.write("### Cluster Characteristics")
    st.write(
        "The table below shows the average values of key features for each cluster. "
        "This helps you understand the behavior and characteristics of customers in each group."
    )

    # Exclude non-numeric columns when calculating the mean
    numeric_data = data.select_dtypes(include=[np.number])
    cluster_summary = numeric_data.groupby("Cluster").mean()

    # Display the cluster summary table
    st.write(cluster_summary)

    # Add explanations for each cluster
    st.write("### Insights into Each Cluster")
    st.write(
        "Here are some insights into what each cluster represents based on the average values:"
    )

    # Example insights (customize based on your dataset)
    st.write("- **Cluster 0**: Customers with moderate credit amounts and stable employment. Likely to be low-risk borrowers.")
    st.write("- **Cluster 1**: Customers with high credit amounts and longer loan durations. May represent high-risk borrowers.")
    st.write("- **Cluster 2**: Customers with low credit amounts and short loan durations. Likely to be cautious borrowers.")

    # Visualize clusters using pair plots
    st.write("### Pair Plot of Clusters")
    st.write(
        "The pair plot below shows the relationship between key features for each cluster. "
        "This helps you visualize how the clusters differ across multiple dimensions."
    )

    # Select a subset of features for visualization
    features_for_plot = ["age", "credit_amount", "duration", "Cluster"]
    plot_data = data[features_for_plot]

    # Create a pair plot
    fig = sns.pairplot(plot_data, hue="Cluster", palette="viridis")
    st.pyplot(fig)

    # Provide actionable recommendations
    st.write("### Recommendations for Each Cluster")
    st.write(
        "Based on the cluster characteristics, here are some recommendations for targeting each group:"
    )
    st.write("- **Cluster 0**: Offer personalized loan products with competitive interest rates to retain these low-risk customers.")
    st.write("- **Cluster 1**: Implement stricter credit checks and offer financial education programs to reduce risk.")
    st.write("- **Cluster 2**: Encourage these customers to take larger loans by offering incentives and flexible repayment options.")

    # Add a conclusion
    st.write("### Conclusion")
    st.write(
        "Customer segmentation helps you understand your customers better and tailor your strategies to meet their needs. "
        "Use the insights and recommendations above to improve your credit risk management and customer engagement."
    )